import torch
import torch.nn as nn
from torch_utils import persistence
import math
import numpy as np
from training.networks_stylegan2 import FullyConnectedLayer
from torch_geometric import nn as gnn
from torch_geometric.data import Data


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
            nn.Linear(feat_dim, feat_dim // 2),
            nn.LeakyReLU(inplace=True),
            nn.Linear(feat_dim // 2, 3),
            nn.Tanh(),
        )

        self.mlp_g = nn.Sequential(
            nn.Linear(feat_dim + 3, feat_dim),
            nn.LeakyReLU(inplace=True),
            nn.Linear(feat_dim, feat_dim),
        )

        self.layer_scale = nn.Parameter(torch.ones(feat_dim) * 0.01)

    def forward(self, x: torch.Tensor, pos: torch.Tensor, edge_index) -> torch.Tensor:
        delta = self.mlp_h(x)
        out = self.propagate(edge_index, x=x * self.layer_scale, pos=pos, delta=delta)
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
        self.ln_1 = AdaptiveNorm(width, w_dim=w_dim)
        self.ls_1 = nn.Linear(w_dim, width)
        nn.init.zeros_(self.ls_1.weight)

    def forward(self, x: torch.Tensor, pos: torch.Tensor, edge_index, w: torch.Tensor):
        B, num_points, C = x.shape
        pos = pos.view(B * num_points, 3)
        y = self.ln_1(x, w).view(B * num_points, C)
        y = self.gnn(y, pos, edge_index).view(B, num_points, C)
        x = x + y * self.ls_1(w)
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
    ):
        super().__init__()
        self.width = width
        self.layers = layers
        self.attn_resblocks = nn.ModuleList([
            ResidualAttentionBlock(
                width=width,
                heads=heads,
                w_dim=w_dim
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

        self.mlp = MLP(width=width)

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

        t = x
        for gnn_resblock, attn_resblock in zip(self.gnn_resblocks, self.attn_resblocks):
            x = gnn_resblock(x, pos, edge_index, ws)
            
            y = self.global_pooling(x)
            t = (t + y) / np.sqrt(2)

            t = attn_resblock(t, ws)

            y = self.mlp(y, ws)
            results.append((y ,t))

        return results
