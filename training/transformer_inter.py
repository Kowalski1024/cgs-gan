import torch
import torch.nn as nn
from torch_utils import persistence
import math
import numpy as np
from training.networks_stylegan2 import FullyConnectedLayer
from torch_geometric import nn as gnn
from torch_geometric.data import Data
from torch_geometric.nn.models.linkx import SparseLinear


@persistence.persistent_class
class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.scale = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        norm = x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return norm * self.scale


@persistence.persistent_class
class InstanceNorm1d(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        dtype = x.dtype
        x = x.to(torch.float32)
        x = (x - x.mean(1, keepdim=True)) / (x.std(1, keepdim=True, unbiased=False) + 1e-8)
        return x.to(dtype)
        
    
@persistence.persistent_class
class AdaptiveNorm(nn.Module):
    def __init__(
        self,
        dim,
        w_dim,
        weight_init=1.0,
    ):
        super().__init__()

        self.gamma = nn.Linear(w_dim, dim)
        self.beta = nn.Linear(w_dim, dim)
        nn.init.constant_(self.gamma.bias, 1.0)
        nn.init.constant_(self.beta.bias, 0.0)
        self.norm = InstanceNorm1d()

    def forward(self, x, w):
        if len(w.shape) == 2:
            return self.norm(x) * self.gamma(w).unsqueeze(1) + self.beta(w).unsqueeze(1)
        else:
            return self.norm(x) * self.gamma(w) + self.beta(w)

@persistence.persistent_class
class MultiheadAttention(nn.Module):
    def __init__(
        self,
        *,
        width: int,
        heads: int,
    ):
        super().__init__()
        self.width = width
        self.heads = heads
        
        self.attention = QKVMultiheadAttention(heads=heads)
        self.c_qkv = nn.Linear(width, width * 3)
        self.c_proj = nn.Linear(width, width)

    def forward(self, x):
        x = self.c_qkv(x)
        x = self.attention(x)
        x = self.c_proj(x)
        return x


@persistence.persistent_class
class MLP(nn.Module):
    def __init__(self, *, width: int):
        super().__init__()
        self.out_channels = width
        self.gelu = nn.GELU()
        self.c_fc = nn.Linear(width, width * 4)
        self.c_proj = nn.Linear(width * 4, width)

    def forward(self, x, w=None):
        return self.c_proj(self.gelu(self.c_fc(x)))


@persistence.persistent_class
class AdaINMLP(nn.Module):
    def __init__(self, width: int, w_dim: int):
        super().__init__()
        self.out_channels = width
        self.lrelu = nn.LeakyReLU()
        self.c_fc = nn.Linear(width, width * 4, bias=False)
        self.c_proj = nn.Linear(width * 4, width, bias=False)

        self.bias_1 = nn.Parameter(torch.zeros(width * 4))
        self.bias_2 = nn.Parameter(torch.zeros(width))

        self.gamma_1 = nn.Linear(w_dim, width)
        self.gamma_2 = nn.Linear(w_dim, width * 4)

        self.gamma_1.bias.data.fill_(1.0)
        self.gamma_2.bias.data.fill_(1.0)

    def forward(self, x, w):
        x = x * self.gamma_1(w)
        x = self.c_fc(x)
        x = x / (x.std(dim=-1, keepdim=True) + 1e-8) + self.bias_1
        x = self.lrelu(x)
        x = x * self.gamma_2(w)
        x = self.c_proj(x)
        x = x / (x.std(dim=-1, keepdim=True) + 1e-8) + self.bias_2
        return x

@persistence.persistent_class
class QKVMultiheadAttention(nn.Module):
    def __init__(self, *, heads: int):
        super().__init__()
        self.heads = heads

    def forward(self, qkv):
        bs, n_ctx, width = qkv.shape
        attn_ch = width // self.heads // 3
        scale = 1 / math.sqrt(math.sqrt(attn_ch))
        qkv = qkv.view(bs, n_ctx, self.heads, -1)
        q, k, v = torch.split(qkv, attn_ch, dim=-1)
        
        # Dot product attention
        weight = torch.einsum(
            "bthc,bshc->bhts", q * scale, k * scale
        )  # More stable with f16 than dividing afterwards
        wdtype = weight.dtype
        weight = torch.softmax(weight.float(), dim=-1).type(wdtype)
        return torch.einsum("bhts,bshc->bthc", weight, v).reshape(bs, n_ctx, -1)
    

class PointGNNConv(gnn.MessagePassing):
    """Two_stage-style PointGNN conv.

    Message: [pos_j - pos_i + delta_i, x_j]
    """

    def __init__(self, *, feat_dim: int):
        super().__init__(aggr="max")

        self.mlp_h = nn.Sequential(
            nn.Linear(feat_dim, 3),
            nn.Tanh(),
        )

        self.mlp_g = nn.Sequential(
            nn.Linear(feat_dim + 3, feat_dim),
        )

    def forward(self, x: torch.Tensor, pos: torch.Tensor, edge_index) -> torch.Tensor:
        delta = self.mlp_h(x)
        out = self.propagate(edge_index, x=x, pos=pos, delta=delta)
        out = self.mlp_g(out)
        return x + out

    def message(
        self,
        pos_j: torch.Tensor,
        pos_i: torch.Tensor,
        x_j: torch.Tensor,
        delta_i: torch.Tensor,
    ) -> torch.Tensor:
        return torch.cat([pos_j - pos_i + delta_i, x_j], dim=-1)
    

@persistence.persistent_class
class ResidualGNNBlock(nn.Module):
    def __init__(
        self,
        *,
        width: int,
        w_dim,
    ):
        super().__init__()
        self.gnn = PointGNNConv(feat_dim=width)
        self.ln_1 = AdaINMLP(width, w_dim=w_dim)
        self.ls_1 = nn.Linear(w_dim, width)
        nn.init.zeros_(self.ls_1.weight)

    def forward(self, x: torch.Tensor, pos: torch.Tensor, edge_index, w: torch.Tensor):
        B, num_points, C = x.shape
        pos = pos.view(B * num_points, 3)
        y = self.ln_1(x, w).view(B * num_points, C)
        y = self.gnn(y, pos, edge_index).view(B, num_points, C)
        x = x + y * self.ls_1(w)
        return x


class LINKXConv(nn.Module):
    def __init__(
        self,
        *,
        num_nodes: int,
        width: int,

    ):
        super().__init__()

        self.edge_lin = SparseLinear(num_nodes, width)

        self.cat_lin1 = torch.nn.Linear(width, width)
        self.cat_lin2 = torch.nn.Linear(width, width)

        self.leakyrelu = nn.LeakyReLU(inplace=True)


    def forward(self, x: torch.Tensor, edge_index) -> torch.Tensor:
        N = x.size(1)
        src, dst = edge_index  # [2, E]

        mask0 = (src < N) & (dst < N)
        edge_index0 = edge_index[:, mask0] 
        out = self.edge_lin(edge_index0, None)

        out = out + self.cat_lin1(out)

        out = out.unsqueeze(0)
        out = out + x
        out = out + self.cat_lin2(out)

        out = self.leakyrelu(out)

        return x


class ResidualLINKXBlock(nn.Module):
    def __init__(
        self,
        *,
        width: int,
        w_dim,
        num_nodes: int,
    ):
        super().__init__()
        self.gnn = LINKXConv(num_nodes=num_nodes, width=width)
        self.mlp_1 = AdaINMLP(width=width, w_dim=w_dim)
        self.mlp_2 = AdaINMLP(width=width, w_dim=w_dim)
        self.ls_1 = nn.Linear(w_dim, width)
        self.ls_2 = nn.Linear(w_dim, width)
        nn.init.zeros_(self.ls_1.weight)
        nn.init.zeros_(self.ls_2.weight)

    def forward(self, x: torch.Tensor, edge_index, w: torch.Tensor):
        B, num_points, C = x.shape
        x = x + self.mlp_1(self.gnn(x, edge_index), w) * self.ls_1(w)
        x = x + self.mlp_2(x, w) * self.ls_2(w)
        
        return x


@persistence.persistent_class
class ResidualAttentionBlock(nn.Module):
    def __init__(
        self,
        *,
        width: int,
        heads: int,
        w_dim,
    ):
        super().__init__()
        self.attn = MultiheadAttention(
            width=width,
            heads=heads,
        )
        self.mlp = MLP(width=width)
        self.ln_1 = AdaptiveNorm(width, w_dim=w_dim)
        self.ln_2 = AdaptiveNorm(width, w_dim=w_dim)
        self.ls_1 = nn.Linear(w_dim, width)
        self.ls_2 = nn.Linear(w_dim, width)
        nn.init.zeros_(self.ls_1.weight)
        nn.init.zeros_(self.ls_2.weight)

    def forward(self, x: torch.Tensor, w: torch.Tensor):
        x = x + self.attn(self.ln_1(x, w)) * self.ls_1(w)
        x = x + self.mlp(self.ln_2(x, w)) * self.ls_2(w)
        return x
    

@persistence.persistent_class
class ResidualMLPBlock(nn.Module):
    def __init__(
        self,
        *,
        width: int,
        w_dim,
    ):
        super().__init__()
        self.mlp = MLP(width=width)
        self.ln_1 = AdaptiveNorm(width, w_dim=w_dim)
        self.ls_1 = nn.Linear(w_dim, width)
        nn.init.zeros_(self.ls_1.weight)

    def forward(self, x: torch.Tensor, w: torch.Tensor):
        x = x + self.mlp(self.ln_1(x, w)) * self.ls_1(w)
        return x


@persistence.persistent_class
class Transformer(nn.Module):
    def __init__(
            self,
            *,
            w_dim,
            width: int,
            layers: int,
            heads: int = 8,
            num_first_blocks: int = 6,
    ):
        super().__init__()
        self.width = width
        self.layers = layers
        self.num_first_blocks = num_first_blocks
        self.linkx_resblocks_first = nn.ModuleList([
            ResidualLINKXBlock(
                width=width,
                w_dim=w_dim,
                num_nodes=512,
            )
            for _ in range(num_first_blocks)
        ])
        self.gnn_resblocks_first = nn.ModuleList([
            ResidualGNNBlock(
                width=width,
                w_dim=w_dim
            )
            for _ in range(num_first_blocks)
        ])


        self.linkx_resblocks = nn.ModuleList([
            ResidualLINKXBlock(
                width=width,
                w_dim=w_dim,
                num_nodes=512,
            )
            for _ in range(layers)
        ])
        self.gnn_resblocks = nn.ModuleList([
            ResidualGNNBlock(
                width=width,
                w_dim=w_dim
            )
            for _ in range(layers)
        ])

        self.mlp = AdaINMLP(width=width, w_dim=w_dim)
        self.norm = InstanceNorm1d()
        self.lrelu = nn.LeakyReLU(inplace=True)

        self.global_conv = nn.Sequential(
            nn.Linear(width, width),
            nn.LeakyReLU(inplace=True),
        )
        self.fuse_global = nn.Sequential(
            nn.Linear(width * 2, width),
            nn.LeakyReLU(inplace=True),
        )

    def global_pooling(self, x: torch.Tensor) -> torch.Tensor:
        g = x.max(dim=1).values
        g = self.global_conv(g)
        g = g.unsqueeze(1).expand(-1, x.shape[1], -1)
        x = self.fuse_global(torch.cat([x, g], dim=-1))
        return x

    def forward(self, x: torch.Tensor, pos: torch.Tensor, edge_index, ws: torch.Tensor):
        results = []

        for i in range(self.num_first_blocks):
            x = self.linkx_resblocks_first[i](x, edge_index, ws)

        t = x

        for i in range(self.layers):
            x = self.gnn_resblocks[i](x, pos, edge_index, ws)
            
            y = self.global_pooling(x)
            t = (t + y) / np.sqrt(2)

            t = self.linkx_resblocks[i](t, edge_index, ws)

            y = self.lrelu(self.mlp(y, ws))
            y = self.norm(y)
            results.append((y ,t))

        return results
