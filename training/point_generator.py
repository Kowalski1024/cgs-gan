import torch
import torch.nn as nn
import numpy as np

from dnnlib import EasyDict
from training.networks_stylegan2 import FullyConnectedLayer
from torch_utils import persistence
from training.gnn_inter import GNNConv, BiasBlock


@persistence.persistent_class
class Identity(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, **kwargs):
        return x


@persistence.persistent_class
class LinLinear(nn.Module):
    def __init__(self, ch_in, ch_out, is_first=False, bias=True):
        super().__init__()

        self.linear = nn.Linear(ch_in, ch_out, bias=bias)
        if is_first:
            nn.init.uniform_(
                self.linear.weight, -np.sqrt(9 / ch_in), np.sqrt(9 / ch_in)
            )
        else:
            nn.init.uniform_(
                self.linear.weight, -np.sqrt(3 / ch_in), np.sqrt(3 / ch_in)
            )

    def forward(self, x):
        return self.linear(x)


@persistence.persistent_class
class SinActivation(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return torch.sin(x)


@persistence.persistent_class
class LFF(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.ffm = LinLinear(3, hidden_size, is_first=True)
        self.activation = SinActivation()

    def forward(self, x):
        x = self.ffm(x)
        x = self.activation(x)
        return x


@persistence.persistent_class
class constant_PE(nn.Module):
    def __init__(self, pe_dim, pe_res):
        super().__init__()

        self.pe_dim = pe_dim
        self.const_pe = torch.nn.Parameter(torch.randn([3, pe_dim, 1, pe_res]))

    def forward(self, pos):
        B, L, _ = pos.shape

        x_coord = pos  # (x + 1) / 2 # range of x: (-1, 1)
        x_coord = x_coord.unsqueeze(-1)  # [B, L, 3, 1]
        x_coord = x_coord.permute(0, 2, 1, 3).reshape(
            3 * B, 1, L, 1
        )  # [B * 3, 1, L, 1]
        x_coord = torch.cat(
            [torch.zeros_like(x_coord), x_coord], dim=-1
        )  # [B * 3, 1, L, 2]

        const_pe = self.const_pe.repeat([B, 1, 1, 1])  # [B * 3, C, 1, L]
        const_emb = torch.nn.functional.grid_sample(
            const_pe, x_coord, mode="bilinear"
        )  # [B * 3, C, 1, L]
        const_emb = (
            const_emb.reshape(B, 3, self.pe_dim, 1, L)
            .sum(1)
            .reshape(B, self.pe_dim, L)
            .permute(0, 2, 1)
        )  # [B, L, C]
        return const_emb / np.sqrt(3)


@persistence.persistent_class
class PointUpsample_subpixel(torch.nn.Module):
    def __init__(
        self,
        in_features,  # Number of input features.
        out_features,  # Number of output features.
        upsample_ratio,
    ):
        super().__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.upsample_ratio = upsample_ratio

        self.subpixel = FullyConnectedLayer(
            in_features, out_features * (upsample_ratio), activation="lrelu"
        )

        if in_features != out_features:
            self.res_fc = FullyConnectedLayer(
                in_features, out_features, activation="linear"
            )
        else:
            self.res_fc = Identity()

    def forward(self, x):
        # x: [B, L, C],
        B, L, C = x.shape
        x_upsampled = self.subpixel(x).reshape(
            B, L * self.upsample_ratio, self.out_features
        )
        x_upsampled = (
            self.res_fc(x).repeat_interleave(self.upsample_ratio, dim=1) + x_upsampled
        ) / np.sqrt(2)
        return x_upsampled


@persistence.persistent_class
class CoordInjection_const(torch.nn.Module):
    def __init__(
        self,
        pe_dim,  # Number of input features.
        pe_res,  # resolution for constant pe
    ):
        super().__init__()
        self.learnable_pe = LFF(pe_dim)
        self.const_pe = constant_PE(pe_dim, pe_res)

    def forward(self, pos, x=None, type="cat"):
        if x is not None:
            x = torch.cat(
                [x, self.learnable_pe(pos).to(x.dtype), self.const_pe(pos).to(x.dtype)],
                dim=-1,
            )
        else:
            x = torch.cat([self.learnable_pe(pos), self.const_pe(pos)], dim=-1)
        return x


def get_scaled_directional_vector_from_quaternion(r, s, eps=0.001):
    # r, s: [B, npoints, c]
    N, npoints, _ = r.shape
    r, s = r.reshape([N * npoints, -1]), s.reshape([N * npoints, -1])

    # Rotation activation (normalize)
    norm = torch.sqrt(
        r[:, 0] * r[:, 0] + r[:, 1] * r[:, 1] + r[:, 2] * r[:, 2] + r[:, 3] * r[:, 3]
    )
    q = r / (norm[:, None] + eps)

    # R = torch.zeros((q.size(0), 3, 3), device='cuda')
    R = torch.zeros((q.size(0), 3, 3), device=r.device)

    r = q[:, 0]
    x = q[:, 1]
    y = q[:, 2]
    z = q[:, 3]

    R[:, 0, 0] = 1 - 2 * (y * y + z * z)
    R[:, 0, 1] = 2 * (x * y - r * z)
    R[:, 0, 2] = 2 * (x * z + r * y)
    R[:, 1, 0] = 2 * (x * y + r * z)
    R[:, 1, 1] = 1 - 2 * (x * x + z * z)
    R[:, 1, 2] = 2 * (y * z - r * x)
    R[:, 2, 0] = 2 * (x * z - r * y)
    R[:, 2, 1] = 2 * (y * z + r * x)
    R[:, 2, 2] = 1 - 2 * (x * x + y * y)

    # Scaling activation (exp)
    s = torch.exp(s)
    L = torch.zeros((s.shape[0], 3, 3), dtype=torch.float, device=r.device)

    L[:, 0, 0] = s[:, 0]
    L[:, 1, 1] = s[:, 1]
    L[:, 2, 2] = s[:, 2]

    L = R @ L

    L = L.reshape([N, npoints, 3, 3])
    return L


def inverse_sigmoid(x):
    return torch.log(x / (1 - x))


class GaussianScene:
    def __init__(self, device, batch_size):
        self.xyz = torch.empty((batch_size, 0, 3), device=device)
        self.scale = torch.empty((batch_size, 0, 3), device=device)
        self.rotation = torch.empty((batch_size, 0, 4), device=device)
        self.opacity = torch.empty((batch_size, 0, 1), device=device)
        self.color = torch.empty((batch_size, 0, 3), device=device)

    def put(self, idx, xyz, scale, rotation, color, opacity):
        self.xyz[idx] = xyz
        self.scale[idx] = scale
        self.rotation[idx] = rotation
        self.color[idx] = color
        self.opacity[idx] = opacity

    def concat(self, new_scene):
        self.xyz = torch.cat([self.xyz, new_scene.xyz], dim=1)
        self.scale = torch.cat([self.scale, new_scene.scale], dim=1)
        self.rotation = torch.cat([self.rotation, new_scene.rotation], dim=1)
        self.color = torch.cat([self.color, new_scene.color], dim=1)
        self.opacity = torch.cat([self.opacity, new_scene.opacity], dim=1)


@persistence.persistent_class
class PointGenerator(nn.Module):
    def __init__(
        self,
        w_dim,  # Intermediate latent (W) dimensionality.
        options={},
    ):
        super().__init__()

        self.conv_in = CoordInjection_const(
            pe_dim=256, pe_res=512
        )  # this will be used by default
        self.num_pts = options["num_pts"]
        self.gnn_num_blocks = options["gnn_num_blocks"]
        self.bias_num_blocks = options["bias_num_blocks"]

        _position_keys = EasyDict(
            xyz=EasyDict(out_dim=3, weight_init=1.0, lr_mult=1.0, bias_init=0.0),
            scale=EasyDict(out_dim=3, weight_init=0.1, lr_mult=1.0, bias_init=-1.0),
            rotation=EasyDict(out_dim=4, weight_init=0.1, lr_mult=1.0, bias_init=0.0),
        )

        _feature_keys = EasyDict(
            color=EasyDict(out_dim=3, weight_init=1.0, lr_mult=1.0, bias_init=0.0),
            opacity=EasyDict(out_dim=1, weight_init=1.0, lr_mult=1.0, bias_init=0.0),
        )

        self.num_ws = 0
        self.position_gnn = nn.ModuleList(
            [GNNConv(512, 512, w_dim=w_dim) for _ in range(self.gnn_num_blocks)]
        )
        self.feature_gnn = nn.ModuleList(
            [
                BiasBlock(self.num_pts, 256, 256, w_dim=w_dim)
                for _ in range(self.bias_num_blocks)
            ]
        )

        self.position_decoder = nn.ModuleDict()
        self.feature_decoder = nn.ModuleDict()
        self.linear = FullyConnectedLayer(
            in_features=512,  # input features from the transformer
            out_features=256,  # output features for the feature transformer
            lr_multiplier=1.0,
            activation="linear",
            weight_init=1.0,
            bias_init=0.0,
        )

        self._position_keys = _position_keys
        self._feature_keys = _feature_keys

        for k in _position_keys.keys():
            self.position_decoder[k] = FullyConnectedLayer(
                in_features=512,  # input features from the transformer
                out_features=_position_keys[k].out_dim,
                lr_multiplier=_position_keys[k].lr_mult,
                activation="linear",
                weight_init=_position_keys[k].weight_init,
                bias_init=_position_keys[k].bias_init,
            )

        for k in _feature_keys.keys():
            self.feature_decoder[k] = FullyConnectedLayer(
                in_features=256,  # input features from the transformer
                out_features=_feature_keys[k].out_dim,
                lr_multiplier=_feature_keys[k].lr_mult,
                activation="linear",
                weight_init=_feature_keys[k].weight_init,
                bias_init=_feature_keys[k].bias_init,
            )

        self.register_buffer("scale_init", torch.ones([3]) * options["scale_init"])
        self.register_buffer("rotation_init", torch.tensor([1, 0, 0, 0]))
        self.register_buffer("color_init", torch.tensor(torch.zeros([3])))
        self.register_buffer("opacity_init", inverse_sigmoid(0.1 * torch.ones([1])))

        self.xyz_output_scale = options["xyz_output_scale"]  # default: 0.1

    def forward(self, x, ws):
        B = ws.shape[0]

        prev_anchors = EasyDict(
            xyz=0,
            scale=torch.tensor(-4.0, device=x.device),
            rotation=self.rotation_init,
            color=self.color_init,
            opacity=self.opacity_init,
        )

        output_gaussians = GaussianScene(device=x.device, batch_size=B)

        for i, w_i in enumerate(ws):
            for i in range(self.gnn_num_blocks):
                x = self.position_gnn[i](x, None, w_i)
            x_pos = x

            for i in range(self.bias_num_blocks):
                x = self.feature_gnn[i](x, None, w_i)
            x_feat = x

            out = EasyDict(
                **{
                    k: self.position_decoder[k](x_pos)
                    for k in self._position_keys.keys()
                },
                **{
                    k: self.feature_decoder[k](x_feat)
                    for k in self._feature_keys.keys()
                },
            )

            new_gaussian = self.postprocessing_block(out, prev_anchors)
            output_gaussians.put(i, **new_gaussian)

        output_gaussians.xyz = torch.clamp(output_gaussians.xyz, -0.5, 0.5)

        return (
            output_gaussians.xyz,
            output_gaussians.scale,
            output_gaussians.rotation,
            output_gaussians.color,
            output_gaussians.opacity,
        )

    def postprocessing_block(self, gaussian: EasyDict, prev_anchor: EasyDict):
        rotation_new = gaussian.rotation + prev_anchor.rotation
        color_new = gaussian.color + prev_anchor.color
        opacity_new = gaussian.opacity + prev_anchor.opacity

        xyz_new = torch.tanh(gaussian.xyz) * self.xyz_output_scale
        scale_new = prev_anchor.scale + gaussian.scale

        new_gaussian = EasyDict(
            xyz=xyz_new,
            scale=torch.tanh(scale_new * 0.05) * 20,
            rotation=torch.tanh(rotation_new * 0.05) * 20,
            color=torch.tanh(color_new * 0.05) * 20,
            opacity=torch.tanh(opacity_new * 0.05) * 20,
        )
        return new_gaussian
