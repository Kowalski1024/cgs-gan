import torch
from torch import nn
from torch_geometric.nn import PointGNNConv, global_max_pool
import numpy as np
import math
from torch import Tensor
from torch.nn import BatchNorm1d
from torch_geometric.nn.models import MLP
from torch_geometric.typing import Adj, OptTensor
from torch_geometric import nn as gnn
from torch_geometric.nn.inits import reset
from itertools import pairwise
from torch_geometric.nn.models.linkx import SparseLinear
from torch_geometric.utils import spmm
from torch_utils.ops.neighbor_max import neighbor_max
from torch_utils.ops.geo_stats import geo_stats
from dnnlib import EasyDict
from training.gaussian import GaussianDecoder, trunc_exp
from training.topology import TopologyFactory


SCALE_MAX = 0.02
SCALE_MIN = 1e-6
SCALE_INIT = 0.01

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


class SynthesisLayer(torch.nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        w_dim,
        activation=nn.LeakyReLU(inplace=True),
        rank=10,
    ):
        super().__init__()
        self.out_channels = out_channels
        self.w_dim = w_dim
        self.affine = nn.Linear(self.w_dim, (in_channels + out_channels) * rank)

        self.weight = torch.nn.Parameter(torch.randn([out_channels, in_channels]))
        self.bias = torch.nn.Parameter(torch.zeros([out_channels]))
        self.activation = activation

        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)

        nn.init.ones_(self.affine.bias)
        
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

        self.layer_1 = SynthesisLayer(in_channels, out_channels, w_dim)
        self.layer_3 = SynthesisLayer(out_channels, out_channels, w_dim, activation=nn.Identity())
        if geometry_aware:
            self.layer_2 = SynthesisLayer(out_channels * 2 + 12, out_channels, w_dim)
            self.layer_geo = SynthesisLayer(in_channels, 3, w_dim, activation=nn.Tanh())
        else:
            self.layer_2 = SynthesisLayer(out_channels * 2, out_channels, w_dim)
            self.layer_geo = None

        if in_channels != out_channels:
            self.residual = nn.Linear(in_channels, out_channels)
        else:
            self.residual = nn.Identity()

        self.norm = PixelNorm()
        self.act = nn.LeakyReLU(inplace=True)


    def forward(self, x, edge_index, w, pos=None):
        skip = self.residual(x)
        out = self.layer_1(x, w)

        # Feature aggregation
        out_aggr, _ = neighbor_max(out, edge_index)

        if self.geometry_aware:
            assert pos is not None, "Position tensor 'pos' must be provided for geometry-aware GNNConv."
            delta = self.layer_geo(x, w)

            # Geometric aggregation
            pos_var, pos_mean, pos_min, pos_max = geo_stats(pos, delta, edge_index)
            pos_std = torch.sqrt(pos_var + 1e-8)

            # Fusion
            out = torch.cat([pos_min, pos_max, pos_mean, pos_std, out, out_aggr], dim=-1)
        else:
            # Fusion
            out = torch.cat([out, out_aggr], dim=-1)

        # Normalization, final layer, skip connection, and activation
        out = self.norm(out)
        out = self.layer_2(out, w)
        out = self.layer_3(out, w)
        out = skip + out
        out = self.act(out)
        return out
        


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


class GaussianEncoding(torch.nn.Module):
    """Fourier features like in f.py (cos/sin of random projections)."""

    def __init__(self, sigma: float, input_size: int, encoded_size: int):
        super().__init__()
        b = torch.randn((encoded_size, input_size)) * sigma
        self.register_buffer("b", b)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        vp = 2 * np.pi * x @ self.b.t()
        return torch.cat((torch.cos(vp), torch.sin(vp)), dim=-1)


class SynthesisBlock(nn.Module):
    def __init__(self, in_channels, out_channels, w_dim):
        super().__init__()
        self.geo_conv = GNNConv(in_channels, out_channels, w_dim, geometry_aware=False)
        self.attr_conv = GNNConv(in_channels * 2, out_channels * 2, w_dim)
        self.proj_head = nn.Sequential(
            nn.Linear(out_channels, out_channels * 2),
        )

    def forward(self, x, y, w, topology):
        x = self.geo_conv(x, topology.dense_edge_index, w, pos=topology.verts)
        y = self.attr_conv(y, topology.dense_edge_index, w)
        y = (self.proj_head(x) + y) / np.sqrt(2.0)
        return x, y


class Decoder(nn.Module):
    def __init__(self, channel_in, features, w_dim, is_base=False):
        super().__init__()

        self.channel_in = channel_in
        self.features = features
        self.is_base = is_base

        self.channels_out = sum(features.values())

        # self.layer = SynthesisLayer(channel_in, self.channels_out, w_dim, activation=nn.Identity())
        self.layer = nn.Linear(channel_in, self.channels_out)

        bias = self.layer.bias
        weight = self.layer.weight
        
        if is_base:
            val_sum = 0
            scale_log_init = float(np.log(SCALE_INIT).round(2))
            scale_log_min = float(np.log(SCALE_MIN).round(2))
            scale_log_max = float(np.log(SCALE_MAX).round(2))

            span = scale_log_max - scale_log_min
            target_prob = np.clip((scale_log_init - scale_log_min) / span, 1e-4, 1-1e-4)
            scale_init_bias = float(np.log(target_prob / (1 - target_prob)))
            opacity_init_bias = float(np.log(0.9 / (1 - 0.9)))
            scale_init_bias = -5.0
            for key, val in features.items():
                if key == "opacity":
                    bias.data[val_sum : val_sum + val].fill_(opacity_init_bias)
                elif key == "scale":
                    bias.data[val_sum : val_sum + val].fill_(scale_init_bias)
                elif key == "rotation":
                    bias.data[val_sum : val_sum + val].fill_(0.0)
                    bias.data[val_sum] = 1.0

                val_sum += val
        else:
            bias.data.fill_(0.0)
            weight.data.fill_(0.0)

    def forward(self, x, w):
        return self.layer(x)


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
                Decoder(out_ch, self.features_geo, z_dim, is_base=(i==2))
            )
            self.attr_decoders.append(
                Decoder(out_ch * 2, self.features_attr, z_dim, is_base=(i==2))
            )

        self.encoder = GaussianEncoding(
            sigma=10.0,
            input_size=3,
            encoded_size=self.channels[1] // 2,
        )


        self.proj_head = nn.Sequential(
            nn.Linear(self.channels[1], self.channels[1] * 2)
        )

        self.upsample = MeshUpsample()

    def forward(self, w):
        topology = self.topology_stack[1]
        x = self.encoder(topology.verts)  # [N, C]
        x = x.unsqueeze(0).expand(w.shape[0], -1, -1)  # [B, N, C]
        y = self.proj_head(x)

        x_skip = None
        y_skip = None
        for i, block in enumerate(self.blocks):
            x, y = block(x, y, w[:, 0], self.topology_stack[1 + i])
            new_xyz = self.xyz_decoders[i](x, w[:, 0])  # [B, N, 3]
            new_attr = self.attr_decoders[i](y, w[:, 0])  # [B, N, C]
            if x_skip is None and y_skip is None:
                x_skip = new_xyz
                y_skip = new_attr
            else:
                subdiv_map = self.topology_stack[1 + i].subdiv_map
                x_skip = self.upsample(x_skip, subdiv_map)
                y_skip = self.upsample(y_skip, subdiv_map)
                x_skip = x_skip + new_xyz
                y_skip = y_skip + new_attr
            if i + 2 <= self.max_level:
                subdiv_map = self.topology_stack[1 + i + 1].subdiv_map
                x = self.upsample(x, subdiv_map)
                y = self.upsample(y, subdiv_map)

        raw_splats_vec = torch.cat([x_skip, y_skip], dim=-1)
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
        rot = torch.nn.functional.normalize(rot, dim=-1)
        xyz = torch.tanh(xyz)  # Limit positions to [-1, 1]
        opac = torch.sigmoid(opac)  # Limit opacity to [0, 1]

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
        self.num_ws = 18
        self.z_dim = w_dim

    def forward(self, ws):
        return self.point_encoder(ws)