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


def fmm_modulate_linear(
    x: torch.Tensor,
    weight: torch.Tensor,
    styles: torch.Tensor,
    activation: str = "demod",
) -> torch.Tensor:
    points_num, c_in = x.shape
    c_out, c_in = weight.shape
    rank = styles.shape[0] // (c_in + c_out)

    assert styles.shape[0] % (c_in + c_out) == 0
    assert len(styles.shape) == 1

    # Now, we need to construct a [c_out, c_in] matrix
    left_matrix = styles[: c_out * rank]  # [left_matrix_size]
    right_matrix = styles[c_out * rank :]  # [right_matrix_size]

    left_matrix = left_matrix.view(c_out, rank)  # [c_out, rank]
    right_matrix = right_matrix.view(rank, c_in)  # [c_out, rank]

    # Imagine, that the output of `self.affine` (in SynthesisLayer) is N(0, 1)
    # Then, std of weights is sqrt(rank). Converting it back to N(0, 1)
    modulation = left_matrix @ right_matrix / np.sqrt(rank)  # [c_out, c_in]

    if activation == "tanh":
        modulation = modulation.tanh()
    elif activation == "sigmoid":
        modulation = modulation.sigmoid() - 0.5

    W = weight * (modulation + 1.0)  # [c_out, c_in]
    if activation == "demod":
        W = W / (W.norm(dim=1, keepdim=True) + 1e-8)  # [c_out, c_in]
    W = W.to(dtype=x.dtype)

    out = x @ W.T

    return out


class SynthesisLayer(torch.nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        w_dim,
        channels_last=False,
        activation=nn.LeakyReLU(inplace=True),
        noise=True,
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
        self.noise = noise

        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x, w):
        styles = self.affine(w).squeeze(0)

        x = fmm_modulate_linear(
            x=x, weight=self.weight, styles=styles, activation="demod"
        )

        if self.bias is not None:
            x = x + self.bias.view(1, -1)  # Reshape bias for broadcasting

        x = self.activation(x)

        if self.noise:
            noise = (
                torch.randn(x.shape[0], self.out_channels, device=x.device)
                * self.noise_strength
            )
            x = x + noise
        return x


class LINKX(torch.nn.Module):
    r"""The LINKX model from the `"Large Scale Learning on Non-Homophilous
    Graphs: New Benchmarks and Strong Simple Methods"
    <https://arxiv.org/abs/2110.14446>`_ paper.
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

        self.edge_lin = SparseLinear(num_nodes, hidden_channels)

        if self.num_edge_layers > 1:
            self.edge_norm = BatchNorm1d(hidden_channels)
            channels = [hidden_channels] * num_edge_layers
            self.edge_mlp = MLP(channels, dropout=0.0, act_first=True, act="leakyrelu")
        else:
            self.edge_norm = None
            self.edge_mlp = None

        self.linear_edge = SynthesisLayer(hidden_channels, hidden_channels, w_dim)

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
        self.edge_lin.reset_parameters()
        if self.edge_norm is not None:
            self.edge_norm.reset_parameters()
        if self.edge_mlp is not None:
            self.edge_mlp.reset_parameters()
        self.node_mlp.reset_parameters()
        self.cat_lin1.reset_parameters()
        self.cat_lin2.reset_parameters()

    def forward(
        self,
        x: OptTensor,
        edge_index: Adj,
        w=None,
    ) -> Tensor:
        """"""  # noqa: D419
        out = self.edge_lin(edge_index)
        out = self.linear_edge(out, w)

        out = out + self.cat_lin1(out)

        if x is not None:
            x = self.node_mlp(x)
            out = out + x
            out = out + self.cat_lin2(x)

        out = self.leakyrelu(out)
        for i, layer in enumerate(self.final_mlp):
            out = layer(out, w)
        return out

    def extra_repr(self):
        return (
            f"num_nodes={self.num_nodes}, "
            f"layers={self.num_layers}, "
            f"in_channels={self.in_channels}, "
            f"out_channels={self.out_channels}"
        )


class SparseLinear(gnn.MessagePassing):
    def __init__(self, in_channels: int, out_channels: int, bias: bool = True):
        super().__init__(aggr="add")
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.weight = nn.Parameter(torch.empty(in_channels, out_channels))
        if bias:
            self.bias = nn.Parameter(torch.empty(out_channels))
        else:
            self.register_parameter("bias", None)

        self.reset_parameters()

    def reset_parameters(self):
        gnn.inits.kaiming_uniform(self.weight, fan=self.in_channels, a=math.sqrt(5))
        gnn.inits.uniform(self.in_channels, self.bias)

    def forward(
        self,
        edge_index: Adj,
        edge_weight: OptTensor = None,
    ) -> Tensor:
        # propagate_type: (weight: Tensor, edge_weight: OptTensor)
        out = self.propagate(edge_index, weight=self.weight, edge_weight=edge_weight)

        if self.bias is not None:
            out = out + self.bias

        return out

    def message(self, weight_j: Tensor, edge_weight: OptTensor) -> Tensor:
        if edge_weight is None:
            return weight_j
        else:
            return edge_weight.view(-1, 1) * weight_j

    def message_and_aggregate(self, adj_t: Adj, weight: Tensor) -> Tensor:
        return spmm(adj_t, weight, reduce=self.aggr)


class BiasBlock(torch.nn.Module):
    def __init__(
        self,
        num_nodes: int,
        in_channels: int,
        out_channels: int,
        w_dim: int,
        normalize: bool = True,
    ):
        super().__init__()
        self.num_nodes = num_nodes
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.normalize = normalize

        self.edge_lin = SparseLinear(num_nodes, out_channels)
        self.edge_lin2 = SynthesisLayer(
            out_channels, out_channels, w_dim=w_dim, activation=nn.Identity()
        )
        self.linear = SynthesisLayer(in_channels, out_channels, w_dim=w_dim)
        self.linear2 = SynthesisLayer(
            in_channels, out_channels, w_dim=w_dim, activation=nn.Identity()
        )
        self.act = nn.LeakyReLU(inplace=True)

        self.reset_parameters()

    def reset_parameters(self):
        reset(self.edge_lin)
        reset(self.edge_lin2)
        reset(self.linear)
        reset(self.linear2)

    def forward(self, x: OptTensor, edge_index: Adj, w) -> Tensor:
        x = self.linear(x, w)
        x = self.linear2(x, w)

        out = self.edge_lin(edge_index)
        out = self.edge_lin2(out, w)

        return self.act(x + out)


class GNNConv(gnn.MessagePassing):
    def __init__(
        self,
        channels_in: int,
        channels_out: int,
        w_dim: int,
        edge_scale_init: float = 0.01,
        **kwargs,
    ):
        kwargs.setdefault("aggr", "add")
        super().__init__(**kwargs)
        self.channels_in = channels_in
        self.channels_out = channels_out
        self.edge_scale_init = edge_scale_init

        self.lin_in = SynthesisLayer(
            channels_in, channels_out, w_dim=w_dim, activation=nn.Identity()
        )
        self.lin_hidden = SynthesisLayer(
            channels_in, channels_out, w_dim=w_dim, activation=nn.Identity()
        )
        self.lin_edge = SynthesisLayer(
            channels_out, channels_out, w_dim=w_dim, activation=nn.Identity()
        )
        self.edge_scale = nn.Parameter(torch.empty(channels_out))
        self.act = nn.LeakyReLU(inplace=True)
        self.act2 = nn.LeakyReLU()

        self.reset_parameters()

    def reset_parameters(self):
        super().reset_parameters()
        torch.nn.init.constant_(self.edge_scale, self.edge_scale_init)
        reset(self.lin_in)
        reset(self.lin_hidden)
        reset(self.lin_edge)

    def forward(self, x: Tensor, edge_index: Adj, w) -> Tensor:
        x = self.lin_in(x, w)

        edges = x * self.edge_scale
        out = self.propagate(edge_index, x=edges)
        out = self.lin_edge(out, w)
        x = self.act2(x)
        x = self.lin_hidden(x, w)

        return self.act(x + out)

    def message(self, x_j: Tensor) -> Tensor:
        return x_j

    def message_and_aggregate(self, adj_t: Adj, x) -> Tensor:
        return spmm(adj_t, x, reduce=self.aggr)


class PointGNNConv(gnn.MessagePassing):
    r"""The PointGNN operator from the `"Point-GNN: Graph Neural Network for
    3D Object Detection in a Point Cloud" <https://arxiv.org/abs/2003.01251>`_
    paper.
    """

    def __init__(
        self,
        channels,
        out_channels,
        z_dim,
        **kwargs,
    ):
        kwargs.setdefault("aggr", "add")
        super().__init__(**kwargs)

        self.mlp_h = nn.ModuleList(
            [
                SynthesisLayer(channels, channels // 2, z_dim),
                SynthesisLayer(channels // 2, 3, z_dim, activation=nn.Tanh()),
            ]
        )

        self.mlp_g = nn.ModuleList(
            [
                SynthesisLayer(channels + 3, channels, z_dim),
                SynthesisLayer(channels, channels, z_dim),
            ]
        )
        self.edge_scale = nn.Parameter(torch.ones(channels) * 0.01)

        self.reset_parameters()

    def reset_parameters(self):
        super().reset_parameters()
        reset(self.mlp_h)
        reset(self.mlp_g)

    def forward(self, x: Tensor, pos: Tensor, edge_index: Adj, w: Tensor) -> Tensor:
        delta = x
        for i, layer in enumerate(self.mlp_h):
            delta = layer(delta, w)
        out = self.propagate(edge_index, x=x * self.edge_scale, pos=pos, delta=delta)
        for i, layer in enumerate(self.mlp_g):
            out = layer(out, w)
        return x + out

    def message(
        self, pos_j: Tensor, pos_i: Tensor, x_i: Tensor, x_j: Tensor, delta_i: Tensor
    ) -> Tensor:
        # Use the passed delta_i directly, no need to calculate it here
        return torch.cat([pos_j - pos_i + delta_i, x_j], dim=-1)

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}(\n"
            f"  mlp_h={self.mlp_h},\n"
            f"  mlp_g={self.mlp_g},\n"
            f")"
        )


class CloudGenerator(nn.Module):
    def __init__(self, channels=128, num_pts=1024, z_dim=128, blocks=2):
        super().__init__()
        self.z_dim = z_dim
        self.blocks = blocks

        self.pos_offset = nn.Parameter(torch.zeros(1, 3))
        self.pos_scale = nn.Parameter(torch.ones(1, 3))

        self.global_conv = nn.Sequential(
            nn.Linear(channels, channels),
            nn.LeakyReLU(inplace=True),
            nn.Linear(channels, channels),
            nn.LeakyReLU(inplace=True),
        )

        self.tail = nn.Sequential(
            nn.Linear(channels // 2, 3),
            nn.Tanh(),
        )

        self.synthetic_block1 = PointGNNConv(128, 128, z_dim)
        self.synthetic_block2 = PointGNNConv(128, 128, z_dim)
        # self.synthetic_block3 = PointGNNConv(128, 128, z_dim)
        # self.synthetic_block8 = PointGNNConv(128, 128, z_dim)
        self.synthetic_block4 = LINKX(num_pts, 256, 256, 256, 2, z_dim)
        self.synthetic_block5 = LINKX(num_pts, 256, 256, 256, 2, z_dim)
        # self.synthetic_block6 = LINKX(POINTS, 256, 256, 256, 2, z_dim)
        # self.synthetic_block7 = LINKX(POINTS, 256, 256, 256, 2, z_dim)

        self.layer_1 = SynthesisLayer(channels * 2, channels, z_dim, noise=False)
        self.layer_2 = SynthesisLayer(channels, channels // 2, z_dim, noise=False)

    def forward(self, pos, x, edge_index, batch, w):
        x = self.synthetic_block1(x, pos, edge_index, w[0])
        x = self.synthetic_block2(x, pos, edge_index, w[0])
        # x = self.synthetic_block3(x, pos, edge_index, w[0])
        # x = self.synthetic_block8(x, edge_index, w[0])

        h = global_max_pool(x, batch)
        h = self.global_conv(h)
        h = h.repeat(x.size(0), 1)

        x = torch.cat([x, h], dim=-1)
        new_pos = self.layer_1(x, w[0])
        new_pos = self.layer_2(new_pos, w[0])
        new_pos = self.tail(new_pos) * self.pos_scale + self.pos_offset
        x = x.detach()
        pre_feat = x
        x = self.synthetic_block4(x, edge_index, w[0])
        x = self.synthetic_block5(x, edge_index, w[0])
        # x = self.synthetic_block6(x, edge_index, w[0])
        # x = self.synthetic_block7(x, edge_index, w[0])

        return new_pos, pre_feat, x


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
            {"scaling": 3, "rotation": 4}, 512, hidden_channles=128
        )
        # self.decoder_rotation = GaussianDecoder({}, 512, hidden_channles=128)
        self.decoder_color = GaussianDecoder(
            {
                "opacity": 1,
                "shs": 3,
            },
            512,
            hidden_channles=128,
        )
        self.num_ws = 18
        self.z_dim = w_dim

    def forward(self, pos, x, edge_index, ws):
        B = ws.shape[0]

        xyz = torch.empty((B, self.num_pts, 3), device=ws.device)
        scale = torch.empty((B, self.num_pts, 3), device=ws.device)
        rotation = torch.empty((B, self.num_pts, 4), device=ws.device)
        opacity = torch.empty((B, self.num_pts, 1), device=ws.device)
        color = torch.empty((B, self.num_pts, 3), device=ws.device)

        for i, w_i in enumerate(ws):
            point_cloud, pre_feat, gaussians_features = self.point_encoder(
                pos, x, edge_index, None, w_i
            )
            scale_features = self.decoder_scale(
                torch.cat([gaussians_features, pre_feat], dim=-1)
            )
            color_features = self.decoder_color(
                torch.cat([gaussians_features, pre_feat], dim=-1)
            )
            gaussian_model = EasyDict(
                xyz=point_cloud,
                **scale_features,
                **color_features,
            )

            xyz[i] = gaussian_model.xyz
            scale[i] = gaussian_model.scaling
            rotation[i] = gaussian_model.rotation
            opacity[i] = gaussian_model.opacity
            color[i] = gaussian_model.shs

        return (
            xyz,
            scale,
            rotation,
            color,
            opacity,
        )
