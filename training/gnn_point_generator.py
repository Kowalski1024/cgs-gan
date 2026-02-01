import torch
from torch import nn
import numpy as np
import math
from torch_utils.ops.neighbor_max import neighbor_max
from torch_utils.ops.geo_stats import geo_stats
from training.gaussian import trunc_exp
from training.topology import TopologyFactory
from training.networks_stylegan2 import FullyConnectedLayer


SCALE_MAX = 0.02
SCALE_MIN = 1e-6
SCALE_INIT = 0.01

SQRT_2 = np.sqrt(2.0)

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


class ScaledLeakyReLU(nn.Module):
    def __init__(self, negative_slope=0.2, inplace=False):
        super().__init__()
        self.negative_slope = negative_slope
        self.inplace = inplace
        self.gain = math.sqrt(2.0 / (1 + negative_slope ** 2))

    def forward(self, x):
        out = torch.nn.functional.leaky_relu(x, self.negative_slope, self.inplace)
        return out * self.gain


class SynthesisLayer(torch.nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        w_dim,
        activation=ScaledLeakyReLU(inplace=True),
        rank=10,
    ):
        super().__init__()
        self.out_channels = out_channels
        self.w_dim = w_dim
        self.affine = FullyConnectedLayer(self.w_dim, (in_channels + out_channels) * rank, bias_init=1)

        self.weight = torch.nn.Parameter(torch.randn([out_channels, in_channels]))
        self.bias = torch.nn.Parameter(torch.zeros([out_channels]))
        self.activation = activation
        
    def forward(self, x, w):
        styles = self.affine(w)

        x = fmm_modulate_linear(
            x=x, weight=self.weight, styles=styles, activation="demod"
        )

        if self.bias is not None:
            x = x + self.bias.view(1, -1)  # Reshape bias for broadcasting

        x = self.activation(x)

        return x


class PixelNorm(nn.Module):
    def __init__(self, epsilon=1e-8):
        super().__init__()
        self.epsilon = epsilon

    def forward(self, x):
        # x: [B, N, C]
        # Normalize over the Channel dimension (dim=2)
        return x * torch.rsqrt(torch.mean(x ** 2, dim=2, keepdim=True) + self.epsilon)



class GNNConv(nn.Module):
    def __init__(self, in_channels, out_channels, w_dim, geometry_aware=False):
        super().__init__()
        self.channel_in = in_channels
        self.channel_out = out_channels
        self.w_dim = w_dim
        self.geometry_aware = geometry_aware
        self.norm = PixelNorm()

        self.layer_1 = SynthesisLayer(in_channels, out_channels, w_dim)
        self.layer_3 = SynthesisLayer(out_channels, out_channels, w_dim, activation=nn.Identity())
        if geometry_aware:
            self.layer_2 = SynthesisLayer(out_channels * 2 + 12, out_channels, w_dim)
            self.layer_geo = SynthesisLayer(in_channels, 3, w_dim, activation=nn.Tanh())
        else:
            self.layer_2 = SynthesisLayer(out_channels * 2, out_channels, w_dim)
            self.layer_geo = None

        if in_channels != out_channels:
            self.residual = FullyConnectedLayer(in_channels, out_channels)
        else:
            self.residual = nn.Identity()

        self.act = ScaledLeakyReLU(inplace=True)


    def forward(self, x, edge_index, w, pos=None):
        skip = self.residual(x)
        out = self.layer_1(x, w)

        out_aggr, _ = neighbor_max(out, edge_index)

        out = torch.cat([out, out_aggr], dim=-1)
        out = self.norm(out)

        # Normalization, final layer, skip connection, and activation
        out = self.layer_2(out, w)
        out = self.layer_3(out, w)
        out = (skip + out) / SQRT_2
        out = self.act(out)
        return out
        


class MeshUpsample(nn.Module):
    def __init__(self, division_factor=2, noise=False):
        super().__init__()
        self.division_factor = division_factor
        self.noise = noise

        if self.noise:
            self.noise_strength = nn.Parameter(torch.zeros(1))

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
        x_new = (feat_a + feat_b) / self.division_factor

        if self.noise:
            B, N_new, C = x_new.shape
            noise = torch.randn(B, N_new, 1, device=x.device) * self.noise_strength
            x_new = x_new + noise

        # 3. Concatenate (Growth)
        # Order must match the topology generation: [Old, New]
        out = torch.cat([x, x_new], dim=1)

        return out


class GaussianEncoding(torch.nn.Module):
    """Fourier features like in f.py (cos/sin of random projections)."""

    def __init__(self, sigma: float, input_size: int, encoded_size: int):
        super().__init__()
        b = torch.randn((encoded_size, input_size)) * sigma
        self.register_buffer("b", b)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        vp = 2 * np.pi * x @ self.b.t()
        return torch.cat((torch.cos(vp), torch.sin(vp)), dim=-1) * SQRT_2
    

def _stats(x):
    return x.mean().item(), x.std().item()


def _stats_spalt(x):
    xyz, rot, scale, color, opac = torch.split(
            x,
            [3, 4, 3, 3, 1], # xyz, rotation, scaling, shs, opacity
            dim=-1,
        )
    print(f"xyz: {_stats(xyz)}, rot: {_stats(rot)}, scale: {_stats(scale)}, color: {_stats(color)}, opac: {_stats(opac)}, ")

class SynthesisBlock(nn.Module):
    def __init__(self, in_channels, out_channels, w_dim):
        super().__init__()
        self.geo_conv = GNNConv(in_channels, out_channels, w_dim, geometry_aware=False)
        self.attr_conv = GNNConv(in_channels * 2, out_channels * 2, w_dim)
        self.proj_head = nn.Sequential(
            FullyConnectedLayer(out_channels, out_channels * 2),
        )

    def forward(self, x, y, w, topology):
        x = self.geo_conv(x, topology.dense_edge_index, w, pos=topology.verts)
        y = self.attr_conv(y, topology.dense_edge_index, w)
        y = (self.proj_head(x) + y) / np.sqrt(2.0)
        # print(f"SynthesisBlock - x: {_stats(x)}, _y: {_stats(_y)}, y: {_stats(y)}")
        return x, y


class Decoder(nn.Module):
    def __init__(self, channel_in, features, w_dim, is_base=False, lr_mul=1.0, rank=10):
        super().__init__()
        channels_out = sum(features.values())
        self.features = features
        self.channels_out = channels_out
        self.is_base = is_base
        self.w_dim = w_dim
        self.lr_mul = lr_mul # Store for init logic

        # 1. Affine: Maps latent W to the Low-Rank modulation weights
        # We use bias_init=1 so that initially modulation is identity (1.0)
        self.affine = FullyConnectedLayer(
            self.w_dim, 
            (channel_in + channels_out) * rank, 
            bias_init=1,
            lr_multiplier=1.0 # Mapping itself usually stays at 1.0
        )

        # 2. Parameters: Initialized N(0, 1)
        self.weight = torch.nn.Parameter(torch.randn([channels_out, channel_in]))
        self.bias = torch.nn.Parameter(torch.zeros([channels_out]))

        # 3. Runtime Gains (Equalized Learning Rate logic)
        self.weight_gain = self.lr_mul / np.sqrt(channel_in)
        self.bias_gain = self.lr_mul

        # 4. Cold Start Initialization
        self.init_parameters()

    @torch.no_grad()
    def init_parameters(self):
        # Zero weights ensure that at step 0, the style modulation has no effect
        # and the output is purely determined by the bias (the Sphere).
        nn.init.zeros_(self.weight)
        self.bias.zero_()
        
        if self.is_base:
            cur = 0
            for key, val in self.features.items():
                if key == "xyz":
                    target = 0.0
                    self.bias[cur : cur + val].fill_(target / self.lr_mul)
                elif key == "rotation":
                    # Identity Quaternion [1, 0, 0, 0]
                    self.bias[cur : cur + val].fill_(0.0)
                    self.bias[cur] = 1.0 / self.lr_mul 
                elif key == "scale":
                    # Target ~0.01 physical size
                    target = math.log(SCALE_MAX * 0.5 + 1e-8)
                    self.bias[cur : cur + val].fill_(target / self.lr_mul)
                elif key == "opacity":
                    # Target 0.8 opacity
                    target = 1.4
                    self.bias[cur : cur + val].fill_(target / self.lr_mul)
                elif key == "shs":
                    self.bias[cur : cur + val].fill_(0.0)
                cur += val

    def forward(self, x, w):
        # Equalize weights and bias at runtime
        w_scaled = self.weight * self.weight_gain
        b_scaled = self.bias * self.bias_gain

        # Get style from mapping network
        styles = self.affine(w)

        # fmm_modulate_linear logic:
        # Note: Ensure this function internally DOES NOT perform 'demod' 
        # when activation is None or a specific flag is passed.
        x = fmm_modulate_linear(
            x=x, 
            weight=w_scaled, 
            styles=styles, 
            activation=None # CRITICAL: No demodulation here!
        )

        # Apply equalized bias
        x = x + b_scaled.view(1, 1, -1) 
        
        return x


class CloudGenerator(nn.Module):
    def __init__(self, channels=256, num_pts=1024, z_dim=128, blocks=2):
        super().__init__()
        self.z_dim = z_dim
        self.blocks = blocks
        self.max_level = 7
        self.topology_stack = TopologyFactory.precompute_icosahedron_stack(
            max_level=self.max_level,
            device='cpu',
        )

        self._scale_log_min = float(np.log(SCALE_MIN).round(2))
        self._scale_log_max = float(np.log(SCALE_MAX).round(2))

        self.channels = {1: 256, 2: 256, 3: 128, 4: 64, 5: 32, 6: 32, 7: 16}
        self.blocks = nn.ModuleList()
        self.xyz_decoders = nn.ModuleList()
        self.attr_decoders = nn.ModuleList()
        self.xyz_upsamplers = nn.ModuleList()
        self.attr_upsamplers = nn.ModuleList()

        self.features_geo = {"xyz": 3}
        self.features_attr = {
            "rotation": 4,
            "scale": 3,
            "shs": 3,
            "opacity": 1,
        }

        for i in range(2, self.max_level + 1):
            in_ch = self.channels[i - 1]
            out_ch = self.channels[i]
            self.blocks.append(
                SynthesisBlock(in_ch, out_ch, z_dim)
            )
            self.xyz_decoders.append(
                Decoder(out_ch, self.features_geo, z_dim, is_base=(i==2), lr_mul=0.1)
            )
            self.attr_decoders.append(
                Decoder(out_ch * 2, self.features_attr, z_dim, is_base=(i==2), lr_mul=0.1)
            )
            self.xyz_upsamplers.append(
                MeshUpsample(division_factor=SQRT_2, noise=False)
            )
            self.attr_upsamplers.append(
                MeshUpsample(division_factor=SQRT_2, noise=False)
            )

        self.encoder = GaussianEncoding(
            sigma=10.0,
            input_size=3,
            encoded_size=self.channels[1] // 2,
        )


        self.proj_head = nn.Sequential(
            FullyConnectedLayer(self.channels[1], self.channels[1] * 2)
        )

        self.upsample_skip = MeshUpsample(division_factor=2)
        # self.upsample_main = MeshUpsample(division_factor=SQRT_2)
        self.sphere_pos_scale = 0.25

    def forward(self, w):
        topology = self.topology_stack[1]
        x = self.encoder(topology.verts)  # [N, C]
        x = x.unsqueeze(0).expand(w.shape[0], -1, -1)  # [B, N, C]
        y = self.proj_head(x)

        # print(f"Start x: {_stats(x)}, y: {_stats(y)}")

        x_skip = None
        y_skip = None
        for i, block in enumerate(self.blocks):
            # print(f"Block {i}")
            current_topology = self.topology_stack[1 + i]
            x, y = block(x, y, w[:, 0], current_topology)
            new_xyz = self.xyz_decoders[i](x, w[:, 0])  # [B, N, 3]
            new_attr = self.attr_decoders[i](y, w[:, 0])  # [B, N, C]
            if x_skip is None and y_skip is None:
                sphere_pos = current_topology.verts.unsqueeze(0).to(x.device)
                x_skip = new_xyz + sphere_pos * self.sphere_pos_scale
                y_skip = new_attr
            else:
                subdiv_map = current_topology.subdiv_map
                x_skip = self.upsample_skip(x_skip, subdiv_map)
                y_skip = self.upsample_skip(y_skip, subdiv_map)
                x_skip = x_skip + new_xyz
                y_skip = y_skip + new_attr
            if i + 2 <= self.max_level:
                subdiv_map = self.topology_stack[1 + i + 1].subdiv_map
                x = self.xyz_upsamplers[i](x, subdiv_map)
                y = self.attr_upsamplers[i](y, subdiv_map)

        raw_splats_vec = torch.cat([x_skip, y_skip], dim=-1)
        # _stats_spalt(raw_splats_vec)
        return self.to_splats(raw_splats_vec)

    def to_splats(self, raw_params):
        B, N, C = raw_params.shape
        assert C == 14, "Expected 14 channels for splat parameters"

        xyz, rot, scale, color, opac = torch.split(
            raw_params,
            [3, 4, 3, 3, 1], # xyz, rotation, scaling, shs, opacity
            dim=-1,
        )

        scale = trunc_exp(scale).clamp(min=SCALE_MIN, max=SCALE_MAX)
        rot = torch.nn.functional.normalize(rot, p=2, dim=-1, eps=1e-8)
        xyz = torch.tanh(xyz)  # Limit positions to [-1, 1]
        opac = torch.sigmoid(opac)  # Limit opacity to [0, 1]

        color = torch.tanh(color) * 1.1
        color = (color + 1.0) * 0.5

        min_s, _ = scale.min(dim=-1, keepdim=True)
        target_max = min_s * 5.0
        scale = torch.minimum(scale, target_max)

        return xyz, scale, rot, color, opac

    def _apply(self, fn):
        """
        Override _apply to handle custom data structures.
        PyTorch calls this for .to(), .cuda(), .cpu(), .type(), etc.
        """
        super()._apply(fn)
        self.topology_stack.map_tensors(fn)
        
        return self


class PointGenerator(nn.Module):
    def __init__(
        self,
        w_dim,
        options={},
    ):
        super().__init__()
        self.num_pts = options["num_pts"]
        self.point_encoder = CloudGenerator(num_pts=self.num_pts, z_dim=w_dim)
        self.point_encoder.sphere_pos_scale = options.get("init_pos_scale", 0.25)
        self.num_ws = 18
        self.z_dim = w_dim

    def forward(self, ws):
        return self.point_encoder(ws)