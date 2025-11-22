import numpy as np
import rff
import torch
from torch import Tensor, nn
from torch_geometric import nn as gnn
from torch_geometric.data import Data
from torch_geometric.nn import knn_graph, EdgeConv
from torch_geometric.nn.inits import reset
from torch_geometric.typing import Adj, OptTensor
from utils.model_utils import trunc_exp


def inverse_sigmoid(x):
    return np.log(x / (1 - x))


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.scale = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        norm = x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return norm * self.scale


class PointGNNConv(gnn.MessagePassing):
    r"""The PointGNN operator from the `"Point-GNN: Graph Neural Network for
    3D Object Detection in a Point Cloud" <https://arxiv.org/abs/2003.01251>`_
    paper.
    """

    def __init__(
        self,
        mlp_h: torch.nn.Module,
        mlp_g: torch.nn.Module,
        **kwargs,
    ):
        kwargs.setdefault("aggr", "mean")
        super().__init__(**kwargs)

        self.mlp_h = mlp_h
        self.mlp_g = mlp_g
        self.layer_scale = nn.Parameter(torch.ones(128) * 0.01)

        # self.reset_parameters()

    def reset_parameters(self):
        super().reset_parameters()
        reset(self.mlp_h)
        reset(self.mlp_g)

    def forward(self, x: Tensor, pos: Tensor, edge_index: Adj) -> Tensor:
        # Calculate delta in the forward function
        delta = self.mlp_h(x)

        # Pass delta to the message function
        out = self.propagate(edge_index, x=x * self.layer_scale, pos=pos, delta=delta)
        out = self.mlp_g(out)
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


class GNNConv(nn.Module):
    def __init__(self, din_in, dim_out):
        super().__init__()

        self.mlp_h = nn.Sequential(
            nn.Linear(din_in, din_in // 2),
            nn.LeakyReLU(inplace=True),
            nn.Linear(din_in // 2, 3),
            nn.Tanh(),
        )

        self.mlp_g = nn.Sequential(
            nn.Linear(dim_out + 3, dim_out),
            nn.LeakyReLU(inplace=True),
            nn.Linear(dim_out, din_in),
        )

        self.network = PointGNNConv(self.mlp_h, self.mlp_g)

    def forward(self, x, pos, edge_index):
        return self.network(x, pos, edge_index)


class LINKX(torch.nn.Module):
    def __init__(
        self,
        num_nodes: int,
        in_channels: int,
        hidden_channels: int,
        out_channels: int,
        num_layers: int,
        num_edge_layers: int = 1,
        num_node_layers: int = 1,
        dropout: float = 0.0,
    ):
        super().__init__()

        self.num_nodes = num_nodes
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_edge_layers = num_edge_layers

        # Replaced SparseLinear with SAGEConv to avoid large parameter matrix and dependency on num_nodes
        self.edge_lin = gnn.SAGEConv(in_channels, hidden_channels)

        channels = [in_channels] + [hidden_channels] * num_node_layers
        self.node_mlp = gnn.MLP(channels, dropout=0.0, act_first=True)

        self.cat_lin1 = torch.nn.Linear(hidden_channels, hidden_channels)
        self.cat_lin2 = torch.nn.Linear(hidden_channels, hidden_channels)

        channels = [hidden_channels] * num_layers + [out_channels]
        self.final_mlp = gnn.MLP(channels, dropout=dropout, act_first=True)
        # self.edge_weights = nn.Parameter(torch.ones(num_nodes * 6))

        self.reset_parameters()

    def reset_parameters(self):
        r"""Resets all learnable parameters of the module."""
        self.edge_lin.reset_parameters()
        self.node_mlp.reset_parameters()
        self.cat_lin1.reset_parameters()
        self.cat_lin2.reset_parameters()
        self.final_mlp.reset_parameters()

    def forward(
        self,
        x: OptTensor,
        edge_index: Adj,
        edge_weight: OptTensor = None,
    ) -> Tensor:
        """"""  # noqa: D419
        # out = self.edge_lin(edge_index, self.edge_weights)
        out = self.edge_lin(x, edge_index)
        out = nn.functional.leaky_relu(out, inplace=True)
        out = self.cat_lin1(out)

        if x is not None:
            x = self.node_mlp(x)
            out = out + x
            out = out + self.cat_lin2(x)

        return self.final_mlp(nn.functional.leaky_relu(out, inplace=True))

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}(num_nodes={self.num_nodes}, "
            f"in_channels={self.in_channels}, "
            f"out_channels={self.out_channels})"
        )


class GaussDecoder(nn.Module):
    def __init__(
        self, in_dim: int, mid_dim: int, shs_degree: int = 0, use_rgb: bool = False
    ):
        super().__init__()
        assert shs_degree == 0, "SH not implemented yet"

        self.feature_channels = {
            "rotation": 4,
            "opacity": 1,
            "scaling": 3,
        }

        if use_rgb:
            self.feature_channels["color"] = 3
        else:
            self.feature_channels["shs"] = 3 * (shs_degree + 1) ** 2

        self.decoder = nn.Sequential(
            nn.Linear(in_dim, in_dim),
            RMSNorm(in_dim),
            nn.LeakyReLU(inplace=True),
            nn.Linear(in_dim, mid_dim),
            RMSNorm(mid_dim),
        )

        self.decoders = torch.nn.ModuleList()
        self.scaling_modulator = nn.Sequential(
            nn.Linear(mid_dim, 3),
            nn.Sigmoid(),
        )

        for key, channels in self.feature_channels.items():
            layer = nn.Linear(mid_dim, channels)

            if key == "scaling":
                torch.nn.init.constant_(layer.bias, -2.0)
            elif key == "shs":
                torch.nn.init.constant_(layer.bias, 0.0)
                torch.nn.init.constant_(layer.weight, 0.0)
            elif key == "rotation":
                torch.nn.init.constant_(layer.bias, 0)
                torch.nn.init.constant_(layer.bias[0], 1.0)
            elif key == "opacity":
                torch.nn.init.constant_(layer.bias, inverse_sigmoid(0.1))

            self.decoders.append(layer)

        self.reset_parameters()

    def reset_parameters(self):
        for i, (key, _) in enumerate(self.feature_channels.items()):
            layer = self.decoders[i]
            if key == "scaling":
                torch.nn.init.constant_(layer.bias, -2.0)
            elif key == "shs":
                torch.nn.init.constant_(layer.bias, 0.0)
                torch.nn.init.constant_(layer.weight, 0.0)
            elif key == "rotation":
                torch.nn.init.constant_(layer.bias, 0)
                torch.nn.init.constant_(layer.bias[0], 1.0)
            elif key == "opacity":
                torch.nn.init.constant_(layer.bias, inverse_sigmoid(0.1))

    def forward(self, features: torch.Tensor) -> dict[str, torch.Tensor]:
        features = self.decoder(features)

        outputs = {}
        for i, (key, _) in enumerate(self.feature_channels.items()):
            feature = self.decoders[i](features)
            if key == "scaling":
                scaling = trunc_exp(feature)
                scaling = torch.clamp(scaling, 0.0, 0.03)
                modulator = self.scaling_modulator(features)
                out = scaling * modulator
            elif key == "opacity":
                out = torch.sigmoid(feature)
            elif key == "rotation":
                out = nn.functional.normalize(feature, dim=-1)
            elif key == "shs":
                _features_dc = feature.unsqueeze(1).contiguous()
                _features_rest = feature.unsqueeze(1)[:, 0:0].contiguous()
                out = torch.cat((_features_dc, _features_rest), dim=-1)
            elif key == "color":
                color = torch.tanh(feature) * 1.1
                out = (color + 1) / 2

            outputs[key] = out

        return outputs


class TwoStageModel(nn.Module):
    def __init__(
        self,
        num_points: int,
        shs_degree: int = 0,
        use_rgb: bool = False,
        dynamic_graph: bool = False,
        k: int = 20,
    ):
        super().__init__()
        self.dynamic_graph = dynamic_graph
        self.k = k

        self.encoder = rff.layers.GaussianEncoding(
            sigma=10.0, input_size=3, encoded_size=64
        )

        self.global_conv = nn.Sequential(
            nn.Linear(128, 128),
            RMSNorm(128),
            nn.LeakyReLU(inplace=True),
        )

        if self.dynamic_graph:
            self.dynamic_conv = EdgeConv(
                nn=nn.Sequential(
                    nn.Linear(2 * 256, 256),
                    nn.LeakyReLU(inplace=True),
                    nn.Linear(256, 256),
                    nn.LeakyReLU(inplace=True),
                ),
                aggr="max",
            )

        self.position_decoder = nn.Sequential(
            nn.Linear(256, 128),
            RMSNorm(128),
            nn.LeakyReLU(inplace=True),
            nn.Linear(128, 3),
        )

        # Initialize the last layer with small weights to keep positions close to 0 initially
        nn.init.normal_(self.position_decoder[-1].weight, mean=0.0, std=0.001)
        nn.init.constant_(self.position_decoder[-1].bias, 0.0)
        self.gaussian_decoder = GaussDecoder(
            256, 128, shs_degree=shs_degree, use_rgb=use_rgb
        )
        self.point_convs = nn.ModuleList(
            [
                GNNConv(128, 128),
                GNNConv(128, 128),
            ]
        )
        self.gaussian_conv = nn.ModuleList(
            [
                LINKX(num_points, 256, 256, 256, 2),
                LINKX(num_points, 256, 256, 256, 2),
            ]
        )

        self.initialize_weights()

    def initialize_weights(self):
        # Generic initialization for all Linear layers
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(
                    m.weight, a=0.01, mode="fan_in", nonlinearity="leaky_relu"
                )
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

        # Re-apply special initializations

        # Position Decoder Last Layer
        nn.init.normal_(self.position_decoder[-1].weight, mean=0.0, std=0.001)
        nn.init.constant_(self.position_decoder[-1].bias, 0.0)

        # Gaussian Decoder Heads
        self.gaussian_decoder.reset_parameters()

    def forward(self, points: Data) -> dict[str, torch.Tensor]:
        pos = points.pos
        edge_index = points.edge_index

        x = self.encoder(pos)

        for conv in self.point_convs:
            x = conv(x, pos, edge_index)

        h = gnn.global_max_pool(x, None)
        h = self.global_conv(h)
        h = h.repeat(x.size(0), 1)

        x = torch.cat([x, h], dim=-1)
        point_features = x

        if self.dynamic_graph:
            edge_index_dyn = knn_graph(
                x, k=self.k, batch=points.batch if hasattr(points, "batch") else None
            )
            x = self.dynamic_conv(x, edge_index_dyn)

        for conv in self.gaussian_conv:
            x = conv(x, edge_index)

        new_pos = self.position_decoder(point_features)
        gaussian_params = self.gaussian_decoder(x)
        gaussian_params["xyz"] = new_pos

        return gaussian_params
