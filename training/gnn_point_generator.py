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
from dnnlib import EasyDict
from training.gaussian import GaussianDecoder
from training.topology import TopologyFactory


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
        channels_last=False,
        activation=nn.LeakyReLU(inplace=True),
        noise=False,
        rank=10,
    ):
        super().__init__()
        self.out_channels = out_channels
        self.w_dim = w_dim
        self.affine = nn.Linear(self.w_dim, (in_channels + out_channels) * rank)

        memory_format = (
            torch.channels_last if channels_last else torch.contiguous_format
        )
        self.weight = torch.nn.Parameter(
            torch.randn([out_channels, in_channels]).to(memory_format=memory_format)
        )
        self.bias = torch.nn.Parameter(torch.zeros([out_channels]))
        self.noise_strength = torch.nn.Parameter(torch.zeros([]))
        self.activation = activation

        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x, w):
        styles = self.affine(w)

        x = fmm_modulate_linear(
            x=x, weight=self.weight, styles=styles, activation="demod"
        )

        if self.bias is not None:
            x = x + self.bias.view(1, -1)  # Reshape bias for broadcasting

        x = self.activation(x)

        return x


class LINKX(nn.Module):
    """
    Optimized LINKX operator for fixed topology (Icosahedron).
    Replaces sparse adjacency matmul with dense embedding lookups.
    """
    def __init__(
        self,
        num_nodes: int,
        in_channels: int,
        hidden_channels: int,
        out_channels: int,
        num_layers: int,
        w_dim: int,
        num_edge_layers: int = 1,
        num_node_layers: int = 1,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.num_nodes = num_nodes
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_edge_layers = num_edge_layers
        self.num_layers = num_layers

        self.edge_emb = nn.Embedding(num_nodes, hidden_channels)

        channels = [in_channels] + [hidden_channels] * num_node_layers
        self.node_mlp = MLP(channels, dropout=0.0, act_first=True, act="leakyrelu")

        self.cat_lin1 = torch.nn.Linear(hidden_channels, hidden_channels)
        self.cat_lin2 = torch.nn.Linear(hidden_channels, hidden_channels)

        channels = [hidden_channels] * num_layers + [out_channels]
        self.final_mlp = nn.ModuleList()
        for channel_in, channel_out in pairwise(channels):
            self.final_mlp.append(SynthesisLayer(channel_in, channel_out, w_dim))

        self.leakyrelu = nn.LeakyReLU(inplace=True)

        self.reset_parameters()

    def reset_parameters(self):
        r"""Resets all learnable parameters of the module."""
        self.node_mlp.reset_parameters()
        self.cat_lin1.reset_parameters()
        self.cat_lin2.reset_parameters()

    def forward(
        self,
        x,   # [N, in_C]
        edge_index, # [N, 6]
        w,             # [L, B, W_dim] or list
    ):
        """
        dense_edge_index: Must handle padding for degree-5 nodes (e.g. repeat last neighbor).
        """
        B = w.shape[0]
        # --- Branch A: Structure (Edge) ---
        # 1. Retrieve neighbor weights: [N, 6] -> [N, 6, H]
        nb_weights = self.edge_emb(edge_index)
        
        # 2. Aggregation (Sum): [N, 6, H] -> [N, H]
        # This computes A * W effectively
        out = nb_weights.sum(dim=1)
            
        # 4. First Mixing
        out = out + self.cat_lin1(out)  # [N, H]
        
        # Expand to batch dimension
        out = out.unsqueeze(0).expand(B, -1, -1)  # [B, N, H]

        # --- Branch B: Node Features ---
        if x is not None:
            x = self.node_mlp(x)
            out = out + x
            out = out + self.cat_lin2(x)

        out = self.leakyrelu(out)
        
        for i, layer in enumerate(self.final_mlp):
            # w[i] shape: [B, W_dim]
            out = layer(out, w)
            
        return out

class MeshConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        # Weight: [Out, In, 2]   (self, neighbor)
        self.weight = nn.Parameter(torch.randn(out_channels, in_channels, 2))
        self.bias = nn.Parameter(torch.zeros(1, 1, out_channels))

        self.act = nn.LeakyReLU(inplace=True)

        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x, edge_index):
        x = x.permute(0, 2, 1)  # [B, C, N]
        B, C, N = x.shape

        # -------------------------------------------------
        # 1. Neighbor aggregation
        # -------------------------------------------------
        feat_self = x

        x_perm = x.permute(0, 2, 1)  # [B, N, C]
        x_neigh = x_perm[:, edge_index]  # [B, N, K, C]
        x_neigh, _ = x_neigh.max(dim=2)          # [B, N, C]
        feat_neigh = x_neigh.permute(0, 2, 1)  # [B, C, N]

        # -------------------------------------------------
        # 2. Convolution (shared weights)
        # -------------------------------------------------
        w_self = self.weight[:, :, 0].unsqueeze(0).expand(B, -1, -1)
        out = torch.bmm(w_self, feat_self)

        w_neigh = self.weight[:, :, 1].unsqueeze(0).expand(B, -1, -1)
        out = out + torch.bmm(w_neigh, feat_neigh)

        # -------------------------------------------------
        # 3. Bias
        # -------------------------------------------------
        out = out.permute(0, 2, 1)  # [B, N, C]
        out = out + self.bias
        

        out = self.act(out)
        return out
        
class BlockTest(nn.Module):
    def __init__(self, in_channels, out_channels, w_dim):
        super().__init__()
        self.conv1 = MeshConv(in_channels, out_channels)
        if in_channels != out_channels:
            self.residual = nn.Linear(in_channels, out_channels)
        else:
            self.residual = nn.Identity()
        self.conv2 = SynthesisLayer(out_channels, out_channels, w_dim)
        self.conv3 = SynthesisLayer(out_channels, out_channels, w_dim, activation=nn.Identity())
        self.activation = nn.LeakyReLU(inplace=True)

    def forward(self, x, edge_index, w):
        skip = self.residual(x)
        x = self.conv1(x, edge_index)
        x = self.conv2(x, w)
        x = self.conv3(x, w)
        x = x + skip
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


class PointGNNConv(nn.Module):
    r"""The PointGNN operator from the `"Point-GNN: Graph Neural Network for
    3D Object Detection in a Point Cloud" <https://arxiv.org/abs/2003.01251>`_
    paper.
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        z_dim,
    ):
        super().__init__()

        self.mlp_h = nn.ModuleList(
            [
                # SynthesisLayer(channels, channels // 2, z_dim),
                SynthesisLayer(in_channels, 3, z_dim, activation=nn.Tanh()),
            ]
        )

        self.mlp_g = nn.ModuleList(
            [
                SynthesisLayer(in_channels * 2 + 12, out_channels, z_dim),
                SynthesisLayer(out_channels, out_channels, z_dim, activation=nn.Identity()),
            ]
        )
        if in_channels != out_channels:
            self.residual = nn.Linear(in_channels, out_channels)
        else:
            self.residual = nn.Identity()
        self.act = nn.LeakyReLU(inplace=True)
        self.norm = PixelNorm()

        self.reset_parameters()

    def reset_parameters(self):
        reset(self.mlp_h)
        reset(self.mlp_g)

    def forward(self, x: Tensor, pos: Tensor, edge_index: Adj, w: Tensor) -> Tensor:
        skip = self.residual(x)
        delta = x
        for i, layer in enumerate(self.mlp_h):
            delta = layer(delta, w)
            
        # --- 2. Feature Aggregation (Memory Optimized) ---
        # Gather features: [B, N, 6, C]
        # We immediately Max-Pool. We do NOT concatenate geometry yet.
        # This saves significant VRAM bandwidth.
        x_neighbors = x[:, edge_index]  # [B, N, 6, C]
        x_aggr, _ = x_neighbors.max(dim=2)  # [B, N, C]
        
        # --- 3. Geometric Aggregation ---
        # Formula: pos_j - (pos_i - delta_i)  ==> (pos_j - pos_i) + delta_i
        rel_pos = (pos[edge_index] - pos.unsqueeze(1)).unsqueeze(0)  # [1, N, 6, 3]
        rel_pos = rel_pos + delta.unsqueeze(2)  # [B, N, 6, 3]
        pos_mean = rel_pos.mean(dim=2)  # [B, N, 3]
        pos_std = rel_pos.std(dim=2)    # [B, N, 3]
        pos_min, _ = rel_pos.min(dim=2)  # [B, N, 3]
        pos_max, _ = rel_pos.max(dim=2)  # [B, N, 3]
        
        # --- 4. Fusion ---
        # Now we concat small tensors [N, 3] and [N, C] -> [N, C+3]
        out = torch.cat([pos_min, pos_max, pos_mean, pos_std, x, x_aggr], dim=-1)

        out = self.norm(out)
        
        for i, layer in enumerate(self.mlp_g):
            out = layer(out, w)
            
        out = skip + out
        out = self.act(out)
        return out

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}(\n"
            f"  mlp_h={self.mlp_h},\n"
            f"  mlp_g={self.mlp_g},\n"
            f")"
        )


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
        self.geo_conv = PointGNNConv(in_channels, out_channels, w_dim)
        self.attr_conv = TestBlock(in_channels, out_channels, w_dim)

    def forward(self, x, pos, edge_index, w):
        pass


class CloudGenerator(nn.Module):
    def __init__(self, channels=256, num_pts=1024, z_dim=128, blocks=2):
        super().__init__()
        self.z_dim = z_dim
        self.blocks = blocks
        self.topology_stack = TopologyFactory.precompute_icosahedron_stack(
            max_level=5,
            device='cpu',
        )
        self.encoder = GaussianEncoding(
            sigma=10.0,
            input_size=3,
            encoded_size=128,
        )

        self.proj_head = nn.Sequential(
            nn.Linear(channels, channels * 2),
        )

        self.geo_blocks = nn.ModuleList(
            [
                PointGNNConv(channels, channels, z_dim)
                for _ in range(3)
            ]
        )

        self.attr_blocks = nn.ModuleList(
            [
                BlockTest(channels * 2, channels * 2, z_dim)
                for _ in range(3)
            ]
        )

        self.layer_2 = SynthesisLayer(channels, channels // 2, z_dim, noise=False)
        self.tail = nn.Sequential(
            nn.Linear(channels // 2, 3),
            nn.Tanh(),
        )
        self.upsample = MeshUpsample()

    def forward(self, w):
        level = 2
        level_2 = self.topology_stack[level]
        pos = level_2.verts 
        edge_index = level_2.dense_edge_index
        x = self.encoder(pos)  # [N, C]
        x = x.unsqueeze(0).expand(w.shape[0], -1, -1)  # [B, N, C]

        y = self.proj_head(x)
        for i in range(3):
            x = self.geo_blocks[i](x, pos, edge_index, w[:, 0])
            y = self.attr_blocks[i](y, edge_index, w[:, 0])
            y = (self.proj_head(x) + y) / np.sqrt(2.0)
            next_level = level + 1
            topology_next = self.topology_stack[next_level]
            x = self.upsample(x, topology_next.subdiv_map)
            y = self.upsample(y, topology_next.subdiv_map)
            pos = topology_next.verts
            edge_index = topology_next.dense_edge_index
            level = next_level

        new_pos = self.layer_2(x, w[:, 0])
        new_pos = self.tail(new_pos)

        return new_pos, y

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
        self.decoder_scale = GaussianDecoder(
            {}, 512, hidden_channles=128
        )
        # self.decoder_rotation = GaussianDecoder({}, 512, hidden_channles=128)
        self.decoder_color = GaussianDecoder(
            {
                "opacity": 1,
                "shs": 3,
                "scaling": 3, 
                "rotation": 4,
            },
            512,
            hidden_channles=256,
        )
        self.num_ws = 18
        self.z_dim = w_dim

    def forward(self, ws):
        B = ws.shape[0]

        xyz = torch.empty((B, self.num_pts, 3), device=ws.device)
        scale = torch.empty((B, self.num_pts, 3), device=ws.device)
        rotation = torch.empty((B, self.num_pts, 4), device=ws.device)
        opacity = torch.empty((B, self.num_pts, 1), device=ws.device)
        color = torch.empty((B, self.num_pts, 3), device=ws.device)

        point_cloud, y = self.point_encoder(ws)
        # scale_features = self.decoder_scale(
        #     torch.cat([gaussians_features, pre_feat], dim=-1)
        # )
        color_features = self.decoder_color(y)
        gaussian_model = EasyDict(
            xyz=point_cloud,
            # **scale_features,
            **color_features,
        )

        xyz = gaussian_model.xyz
        scale = gaussian_model.scaling
        rotation = gaussian_model.rotation
        opacity = gaussian_model.opacity
        color = gaussian_model.shs

        return (
            xyz,
            scale,
            rotation,
            color,
            opacity,
        )
