import torch
from torch import Tensor, nn
from dnnlib import EasyDict
from torch.nn import functional as F
from torch_geometric import nn as gnn
from torch_geometric.typing import Adj
import numpy as np
from training.networks_stylegan2 import FullyConnectedLayer
from torch_utils.ops import bias_act


class ModLayer(nn.Module):
    def __init__(self, channels, z_dim, activation="lrelu", demod=True, use_noise=False):
        super().__init__()
        self.affine = FullyConnectedLayer(z_dim, channels, bias_init=1)
        self.act_gain = bias_act.activation_funcs[activation].def_gain
        self.bias = nn.Parameter(torch.zeros(channels))
        self.demod = demod
        self.use_noise = use_noise
        self.activation = activation

        if use_noise:
            self.noise_strength = nn.Parameter(torch.zeros([]))

    def forward(self, x, w, gain=1):
        style_scales  = self.affine(w)
        x = x * style_scales

        if self.demod:
            x = F.normalize(x, dim=-1, p=2)

        if self.use_noise:
            x = x + self.noise_strength * torch.randn_like(x, dtype=x.dtype)

        act_gain = self.act_gain * gain
        return bias_act.bias_act(x, self.bias.to(x.dtype), act=self.activation, gain=act_gain)


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
                ModLinear(channels, channels // 2, z_dim, use_noise=True),
                ModLinear(channels // 2, 3, z_dim, activation="tanh", demod=False),
            ]
        )

        self.mlp_g = nn.ModuleList(
            [
                ModLinear(channels + 3, channels, z_dim, use_noise=True),
                ModLinear(channels, channels, z_dim, use_noise=True),
            ]
        )
        self.edge_scale = nn.Parameter(torch.ones(channels) * 0.01)

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


# class PositionLayer(nn.Module):
#     def __init__(self, in_channels, out_channels, z_dim, activation="leaky_relu", demod=True):
#         super().__init__()
#         self.gnn = PointGNNConv(in_channels, out_channels, z_dim)

#     def forward(self, x, pos, edge_index, w):
#         x = self.gnn(x, pos, edge_index, w)
#         return x
    

class ModLinear(nn.Module):
    def __init__(self, in_channels, out_channels, z_dim, activation="lrelu", demod=True, use_noise=False):
        super().__init__()
        self.linear = FullyConnectedLayer(in_channels, out_channels)
        self.mod_layer = ModLayer(out_channels, z_dim, activation=activation, demod=demod, use_noise=use_noise)

    def forward(self, x, w):
        x = self.linear(x)
        x = self.mod_layer(x, w)
        return x
    

class GaussianDecoder(nn.Module):
    def __init__(self, feature_channels: dict, in_channels, hidden_channels, z_dim):
        super().__init__()
        self.feature_channels = feature_channels

        self.mlp = nn.ModuleList(
            [
                ModLinear(in_channels, hidden_channels, z_dim, use_noise=True),
                ModLinear(hidden_channels, hidden_channels, z_dim, use_noise=True),
            ]
        )

        self.decoders = torch.nn.ModuleList()
        for key, channels in feature_channels.items():
            layer = ModLinear(hidden_channels, channels, z_dim, activation="linear", demod=False)

            if key == "rotation":
                torch.nn.init.constant_(layer.linear.bias, 0)
                torch.nn.init.constant_(layer.linear.bias[0], 1.0)
            elif key == "opacity":
                torch.nn.init.constant_(layer.linear.bias, np.log(0.1 / (1 - 0.1)))

            self.decoders.append(layer)

    def forward(self, x, w):
        for layer in self.mlp:
            x = layer(x, w)

        ret = {}
        for k, layer in zip(self.feature_channels.keys(), self.decoders):
            ret[k] = layer(x, w)

        return ret

class CloudGenerator(nn.Module):
    def __init__(self, channels=128, num_pts=1024, z_dim=128, blocks=2):
        super().__init__()
        self.z_dim = z_dim
        self.blocks = blocks

        self.pos_offset = nn.Parameter(torch.zeros(1, 3))
        self.pos_scale = nn.Parameter(torch.ones(1, 3))

        self.global_conv = nn.Sequential(
            FullyConnectedLayer(channels, channels, activation="lrelu"),
            FullyConnectedLayer(channels, channels, activation="lrelu"),
        )

        self.tail = nn.ModuleList(
            [
                ModLinear(channels * 2, channels, z_dim, use_noise=True),
                ModLinear(channels, channels // 2, z_dim, use_noise=True),
                ModLinear(channels // 2, 3, z_dim, activation="tanh", demod=False)
            ]
        )

        self.pos_blocks = nn.ModuleList(
            [
                PointGNNConv(128, 128, z_dim),
                PointGNNConv(128, 128, z_dim),
                # PositionLayer(128, 128, z_dim),
                # PositionLayer(128, 128, z_dim),
            ]
        )

        self.feature_blocks = nn.ModuleList(
            [
                PointGNNConv(256, 256, z_dim),
                PointGNNConv(256, 256, z_dim),
                # PositionLayer(256, 256, z_dim),
                # PositionLayer(256, 256, z_dim),
            ]
        )

    def forward(self, pos, x, edge_index, batch, w):
        for block in self.pos_blocks:
            x = block(x, pos, edge_index, w)

        h = gnn.global_max_pool(x, batch)
        h = self.global_conv(h)
        h = h.repeat(x.size(0), 1)

        x = torch.cat([x, h], dim=-1)

        new_pos = x
        for block in self.tail:
            new_pos = block(new_pos, w)
        new_pos = new_pos * self.pos_scale + self.pos_offset
        
        x = x.detach()
        pre_feat = x

        for block in self.feature_blocks:
            x = block(x, pos, edge_index, w)

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
            {"scaling": 3, "rotation": 4}, 512, hidden_channels=128, z_dim=w_dim
        )

        self.decoder_color = GaussianDecoder(
            {
                "opacity": 1,
                "shs": 3,
            },
            512,
            hidden_channels=128,
            z_dim=w_dim,
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
                torch.cat([gaussians_features, pre_feat], dim=-1), w_i
            )
            color_features = self.decoder_color(
                torch.cat([gaussians_features, pre_feat], dim=-1), w_i
            )
            gaussian_model = EasyDict(
                xyz=point_cloud,
                **scale_features,
                **color_features,
            )

            xyz[i] = gaussian_model.xyz
            scale[i] = torch.tanh(gaussian_model.scaling * 0.05) * 20
            rotation[i] = torch.tanh(gaussian_model.rotation * 0.05) * 20
            opacity[i] = torch.tanh(gaussian_model.opacity * 0.05) * 20
            color[i] = torch.tanh(gaussian_model.shs * 0.05) * 20

        return (
            xyz,
            scale,
            rotation,
            color,
            opacity,
        )
