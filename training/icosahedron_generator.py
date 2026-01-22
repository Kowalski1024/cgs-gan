import torch
from torch import nn
import numpy as np
import math
from torch import Tensor
from torch_geometric.utils import spmm
from dnnlib import EasyDict
from torch_utils import persistence
from training.networks_stylegan2 import FullyConnectedLayer
from training.topology import TopologyFactory


def bounded_log_sigmoid(
    raw: torch.Tensor, log_min: float, log_max: float
) -> torch.Tensor:
    span = log_max - log_min
    return log_min + (span * torch.sigmoid(raw))


def fmm_modulate_linear(
    x: torch.Tensor,
    weight: torch.Tensor,
    styles: torch.Tensor,
    activation: str = "demod",
) -> torch.Tensor:
    # x: [B, N, C], styles: [B, size]
    c_in = x.shape[-1]
    c_out, c_in_weight = weight.shape
    
    B = styles.shape[0]
    rank = styles.shape[1] // (c_in + c_out)
    assert styles.shape[1] % (c_in + c_out) == 0
    
    # Construct batched modulation: [B, c_out, c_in]
    left_matrix = styles[:, : c_out * rank].view(B, c_out, rank)  # [B, c_out, rank]
    right_matrix = styles[:, c_out * rank :].view(B, rank, c_in)  # [B, rank, c_in]
    modulation = torch.bmm(left_matrix, right_matrix) / np.sqrt(rank)  # [B, c_out, c_in]
    
    if activation == "tanh":
        modulation = modulation.tanh()
    elif activation == "sigmoid":
        modulation = modulation.sigmoid() - 0.5
    
    # Batched weight modulation: [B, c_out, c_in]
    W = weight.unsqueeze(0) * (modulation + 1.0)  # [B, c_out, c_in]
    if activation == "demod":
        W = W / (W.norm(dim=2, keepdim=True) + 1e-8)  # [B, c_out, c_in]
    W = W.to(dtype=x.dtype)
    
    # Batched matmul: [B, N, c_in] @ [B, c_in, c_out] -> [B, N, c_out]
    out = torch.bmm(x, W.transpose(1, 2))

    return out


@persistence.persistent_class
class ScaledLeakyReLU(nn.Module):
    def __init__(self, negative_slope=0.2, inplace=False):
        super().__init__()
        self.negative_slope = negative_slope
        self.inplace = inplace
        self.gain = math.sqrt(2) 

    def forward(self, x):
        out = torch.nn.functional.leaky_relu(x, self.negative_slope, self.inplace)
        return out * self.gain


@persistence.persistent_class
class SynthesisLayer(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        w_dim,
        activation=True,
        rank=10,
    ):
        super().__init__()
        self.out_channels = out_channels
        self.w_dim = w_dim
        self.affine = FullyConnectedLayer(self.w_dim, (in_channels + out_channels) * rank, bias_init=1.0)

        self.weight = torch.nn.Parameter(torch.randn([out_channels, in_channels]))
        self.bias = torch.nn.Parameter(torch.zeros([1, 1, out_channels]))
        self.activation = ScaledLeakyReLU() if activation else nn.Identity()
        
    def forward(self, x, w):
        styles = self.affine(w).squeeze(1)

        x = fmm_modulate_linear(
            x=x, weight=self.weight, styles=styles
        )

        x = x + self.bias
        x = self.activation(x)
        return x


@persistence.persistent_class
class ToXYZLayer(nn.Module):
    def __init__(
        self,
        in_channels,
        w_dim,
        rank=10,
        level_idx=None,
    ):
        super().__init__()
        self.layer = SynthesisLayer(in_channels, 3, w_dim, activation=False, rank=rank)
        self.layer.weight.data.fill_(0.0)

        if level_idx is not None:
            base_limit = 0.4
            decay_factor = 1.0
            
            self.pos_limit = base_limit / (decay_factor ** (level_idx - 1))
        else:
            self.pos_limit = None

    def forward(self, x, w):
        x = self.layer(x, w)
        if self.pos_limit is not None:
            x = torch.tanh(x) * self.pos_limit
        return x


@persistence.persistent_class
class ToAttributesLayer(torch.nn.Module):
    def __init__(
        self,
        in_channels,
        w_dim,
        out_channels=11,
        rank=10,
        is_base=False, 
        scale_log_init=None, 
        scale_log_min=None, 
        scale_log_max=None,
    ):
        super().__init__()
        assert not (
            is_base and any(param is None for param in [scale_log_init, scale_log_min, scale_log_max])
            ), "scale params be provided for base level"
        assert out_channels in [11, 14], "output channels must be 11 or 14"

        self.layer = SynthesisLayer(in_channels, out_channels, w_dim, activation=False, rank=rank)
        self.layer.weight.data.fill_(0.0)

        if is_base:
            span = scale_log_max - scale_log_min
            target_prob = np.clip((scale_log_init - scale_log_min) / span, 1e-4, 1-1e-4)
            scale_init_bias = float(np.log(target_prob / (1 - target_prob)))
            opacity_init_bias = float(np.log(0.9 / (1 - 0.9)))

            if out_channels == 11:
                self.layer.bias.data[:, :, 7].fill_(opacity_init_bias)    # Opacity
                self.layer.bias.data[:, :, 3:7].fill_(scale_init_bias)    # Scale
                
            # If outputting 14 channels (XYZ, Rot, Scale, Opac, SH):
            elif out_channels == 14:
                self.layer.bias.data[:, :, 10].fill_(opacity_init_bias)   # Opacity
                self.layer.bias.data[:, :, 6:10].fill_(scale_init_bias)   # Scale
        else:
            shrink_val = float(np.log(0.65)) 
            if out_channels == 11:
                self.layer.bias.data[:, :, 3:7].fill_(shrink_val)  # Scale
            elif out_channels == 14:
                self.layer.bias.data[:, :, 6:10].fill_(shrink_val)  # Scale

    def forward(self, x, w):
        x = self.layer(x, w)
        return x


@persistence.persistent_class
class MeshUpsample(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, subdiv_map):
        """
        x: [B, N_prev, C]
        subdiv_map: [2, N_new] (Indices of parents for new points)
        Returns: [B, N_prev + N_new, C]
        """
        # 1. Get features of parents
        # subdiv_map[0] -> Parent A, subdiv_map[1] -> Parent B
        idx_a = subdiv_map[0]
        idx_b = subdiv_map[1]

        feat_a = x[:, idx_a, :]
        feat_b = x[:, idx_b, :]

        # 2. Linear Interpolation (Average)
        x_new = (feat_a + feat_b) * 0.5

        # 3. Concatenate (Growth)
        # Order must match the topology generation: [Old, New]
        out = torch.cat([x, x_new], dim=1)

        return out


@persistence.persistent_class
class GaussianEncoding(nn.Module):
    """Fourier features like in f.py (cos/sin of random projections)."""

    def __init__(self, sigma: float, input_size: int, encoded_size: int):
        super().__init__()
        b = torch.randn((encoded_size, input_size)) * sigma
        self.register_buffer("b", b)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        vp = 2 * np.pi * x @ self.b.t()
        return torch.cat((torch.cos(vp), torch.sin(vp)), dim=-1)


@persistence.persistent_class
class PixelNorm(nn.Module):
    def __init__(self, epsilon=1e-8):
        super().__init__()
        self.epsilon = epsilon

    def forward(self, x):
        # x: [B, N, C]
        # Normalize over the Channel dimension (dim=2)
        return x * torch.rsqrt(torch.mean(x ** 2, dim=2, keepdim=True) + self.epsilon)


@persistence.persistent_class
class MeshConv(nn.Module):
    def __init__(self, in_channels, out_channels, activation=True):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.activation = ScaledLeakyReLU() if activation else nn.Identity()

        # Weight: [Out, In, 2]   (self, neighbor)
        self.weight = nn.Parameter(torch.randn(out_channels, in_channels, 2))
        self.bias = nn.Parameter(torch.zeros(1, 1, out_channels))

        self.w_scale = 1 / np.sqrt(in_channels)

    def forward(self, x, dense_edge_index):
        B, N, C = x.shape
        w = self.weight * self.w_scale 

        # -------------------------------------------------
        # 1. Neighbor aggregation
        # -------------------------------------------------
        x_neigh = x[:, dense_edge_index]  # [B, N, K, C]
        x_neigh, _ = x_neigh.max(dim=2)  # [B, N, C]

        # -------------------------------------------------
        # 2. Convert to [B, C, N] for convolution
        # -------------------------------------------------
        feat_self = x.permute(0, 2, 1)  # [B, C, N]
        feat_neigh = x_neigh.permute(0, 2, 1)  # [B, C, N]

        # -------------------------------------------------
        # 3. Convolution (shared weights)
        # -------------------------------------------------
        w_self = w[:, :, 0].unsqueeze(0).expand(B, -1, -1)
        out = torch.bmm(w_self, feat_self)

        w_neigh = w[:, :, 1].unsqueeze(0).expand(B, -1, -1)
        out = out + torch.bmm(w_neigh, feat_neigh)

        # -------------------------------------------------
        # 4. Convert back to [B, N, C] and apply bias + activation
        # -------------------------------------------------
        out = out.permute(0, 2, 1)  # [B, N, Out]
        out = out + self.bias
        out = self.activation(out)
        return out


@persistence.persistent_class
class AttributeBlock(nn.Module):
    def __init__(self, in_channels, out_channels, w_dim):
        super().__init__()
        self.mesh_conv = MeshConv(in_channels, out_channels, activation=True)
        self.synthesis_1 = SynthesisLayer(out_channels, out_channels, w_dim, activation=True)
        self.synthesis_2 = SynthesisLayer(out_channels, out_channels, w_dim, activation=False)
        self.skip = FullyConnectedLayer(in_channels, out_channels, activation='linear')
        self.act = ScaledLeakyReLU()

    def forward(self, x, dense_edge_index, w):
        skip = self.skip(x)
        x = self.mesh_conv(x, dense_edge_index)
        x = self.synthesis_1(x, w)
        x = self.synthesis_2(x, w)
        x = self.act(x + skip)
        return x


@persistence.persistent_class
class GeometryBlock(nn.Module):
    def __init__(self, in_channels, out_channels, w_dim):
        super().__init__()
        self.synthesis_delta = SynthesisLayer(in_channels, 3, w_dim, activation=False)
        self.synthesis_1 = SynthesisLayer(in_channels * 2 + 12, out_channels, w_dim, activation=True)
        self.synthesis_2 = SynthesisLayer(out_channels, out_channels, w_dim, activation=False)
        self.skip = FullyConnectedLayer(in_channels, out_channels, activation='linear')
        self.act = ScaledLeakyReLU()
        self.pixel_norm = PixelNorm()

    def forward(self, x, pos, dense_edge_index, w):
        skip = self.skip(x)
        pos_delta = self.synthesis_delta(x, w)
        pos_delta = torch.tanh(pos_delta)

        x_neighbors = x[:, dense_edge_index]  # [B, N, 6, C]
        x_aggr, _ = x_neighbors.max(dim=2)  # [B, N, C]

        rel_pos = (pos[dense_edge_index] - pos.unsqueeze(1)).unsqueeze(0)  # [1, N, 6, 3]
        rel_pos = rel_pos + pos_delta.unsqueeze(2)  # [B, N, 6, 3]
        pos_mean = rel_pos.mean(dim=2)  # [B, N, 3]
        pos_std = rel_pos.std(dim=2)    # [B, N, 3]
        pos_min, _ = rel_pos.min(dim=2)  # [B, N, 3]
        pos_max, _ = rel_pos.max(dim=2)  # [B, N, 3]

        out = torch.cat([pos_min, pos_max, pos_mean, pos_std, x, x_aggr], dim=-1)
        out = self.pixel_norm(out)

        out = self.synthesis_1(out, w)
        out = self.synthesis_2(out, w)
        out = self.act(out + skip)

        return out


@persistence.persistent_class
class GaussBlock(nn.Module):
    def __init__(self, geo_in_channels, geo_out_channels, attr_in_channels, attr_out_channels, w_dim, level_idx=None):
        super().__init__()
        self.upsample = MeshUpsample()
        self.geometry_block = GeometryBlock(geo_in_channels, geo_out_channels, w_dim)
        self.attribute_block = AttributeBlock(attr_in_channels, attr_out_channels, w_dim)
        self.proj_layer = FullyConnectedLayer(
            geo_out_channels, 
            attr_out_channels, 
            activation='lrelu',
            weight_init=0.0,
        )
        self.to_attr = ToAttributesLayer(attr_out_channels, w_dim)
        self.to_xyz = ToXYZLayer(geo_out_channels, w_dim, level_idx=level_idx)

    def forward(self, geo_feat, attr_feat, topology, w):
        # Upsample
        geo_feat = self.upsample(geo_feat, topology.subdiv_map)
        attr_feat = self.upsample(attr_feat, topology.subdiv_map)

        # GNN Blocks
        geo_feat = self.geometry_block(geo_feat, topology.verts, topology.dense_edge_index, w)
        attr_feat = self.attribute_block(attr_feat, topology.dense_edge_index, w)
        # Cross Attention
        proj_feat = self.proj_layer(geo_feat)
        attr_feat = attr_feat + proj_feat

        # To XYZ and Attributes
        geo_update = self.to_xyz(geo_feat, w)
        attr_update = self.to_attr(attr_feat, w)

        return geo_update, attr_update, geo_feat, attr_feat


@persistence.persistent_class
class PointGenerator(nn.Module):
    def __init__(
        self,
        w_dim,
        options={},
    ):
        super().__init__()
        self.max_level = 5
        self.scale_max = 0.02
        self.scale_min = 1e-6
        self.scale_init = 0.01
        self.curr_level = 5
        self.num_ws = 0

        self._scale_log_min = float(np.log(self.scale_min).round(2))
        self._scale_log_max = float(np.log(self.scale_max).round(2))
        self._scale_log_init = float(np.log(self.scale_init).round(2))

        self.topology_stack = TopologyFactory.precompute_icosahedron_stack(
            max_level=self.max_level,
            device='cpu',
        )

        self.attr_channels = {1: 512, 2: 512, 3: 256, 4: 256, 5: 128, 6: 64, 7: 32}
        self.geo_channels = {k: max(v // 2, 8) for k, v in self.attr_channels.items()}

        self.pos_enc_dim = self.geo_channels[1]
        self.pos_encoder = GaussianEncoding(
            sigma=10.0, input_size=3, encoded_size=self.pos_enc_dim // 2
        )
        self.l1_attr = FullyConnectedLayer(self.pos_enc_dim, self.attr_channels[1], activation='lrelu')

        self.l1_to_xyz = ToXYZLayer(
            in_channels=self.pos_enc_dim,
            w_dim=w_dim,
            level_idx=None,
        )

        self.l1_to_attr = ToAttributesLayer(
            in_channels=self.attr_channels[1],
            w_dim=w_dim,
            out_channels=11,
            is_base=True,
            scale_log_init=self._scale_log_init,
            scale_log_min=self._scale_log_min,
            scale_log_max=self._scale_log_max,
        )

        self.upsample = MeshUpsample()
        self.synthesis_blocks = nn.ModuleList()

        for level in range(2, self.max_level + 1):
            geo_in_ch = self.geo_channels[level - 1]
            geo_out_ch = self.geo_channels[level]
            attr_in_ch = self.attr_channels[level - 1]
            attr_out_ch = self.attr_channels[level]

            block = GaussBlock(
                geo_in_channels=geo_in_ch,
                geo_out_channels=geo_out_ch,
                attr_in_channels=attr_in_ch,
                attr_out_channels=attr_out_ch,
                w_dim=w_dim,
                level_idx=None,
            )
            self.synthesis_blocks.append(block)

    def _apply(self, fn):
        """
        Override _apply to handle custom data structures.
        PyTorch calls this for .to(), .cuda(), .cpu(), .type(), etc.
        """
        super()._apply(fn)
        self.topology_stack.map_tensors(fn)
        
        return self

    def forward(self, ws, level=None):
        if level is None:
            level = self.curr_level

        topo_l1 = self.topology_stack[1]

        anchors = topo_l1.verts.unsqueeze(0).repeat(len(ws), 1, 1)  # [B, 42, 3]

        x = self.pos_encoder(anchors)   # [B, 42, pos_enc_dim]
        geo_curr = x                    # [B, 42, geo_ch_l1]
        attr_curr = self.l1_attr(x)     # [B, 42, attr_ch_l1]

        geo_skip = anchors * 0.5
        attr_skip = self.l1_to_attr(attr_curr, ws)

        for i, block in enumerate(self.synthesis_blocks):
            step = i + 1
            target_level = i + 2
            topo_next = self.topology_stack[target_level]

            geo_update, attr_update, geo_curr, attr_curr = block(
                geo_curr, attr_curr, topo_next, ws
            )

            geo_skip = self.upsample(geo_skip, topo_next.subdiv_map) + geo_update
            attr_skip = self.upsample(attr_skip, topo_next.subdiv_map) + attr_update

            if target_level == level:
                break
        
        final_splats = torch.cat(
            [geo_skip, attr_skip], dim=-1
        )  # [B, N, 3 + 11]
        return self.get_splats(final_splats)

    def get_splats(self, raw_params):
        B, N, C = raw_params.shape
        assert C == 14, "Expected 14 channels for splat parameters"

        xyz, rot, scale, opac, color = torch.split(
            raw_params,
            [3, 4, 3, 1, 3],
            dim=-1,
        )

        scale = bounded_log_sigmoid(scale, log_min=self._scale_log_min, log_max=self._scale_log_max).exp()
        rot = torch.nn.functional.normalize(rot, dim=-1)
        xyz = torch.tanh(xyz)  # Limit positions to [-1, 1]
        opac = torch.sigmoid(opac)  # Limit opacity to [0, 1]

        return xyz, scale, rot, color, opac
            

    


    