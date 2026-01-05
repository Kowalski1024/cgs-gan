import torch
import torch.nn as nn
import numpy as np
from typing import Literal

from dnnlib import EasyDict
from training.networks_stylegan2 import FullyConnectedLayer
from training.transformer_inter import Transformer, ResidualMLPBlock
from torch_utils import persistence
import math


LOG_MIN = float(np.log(1e-6).round(2))
LOG_MAX = float(np.log(0.02).round(2))


def soft_log_anchor_filter(x, anchor: Literal["max", "min"] = "min", max_ratio=5.0):
    if anchor == "max":
        ref, _ = x.max(dim=-1, keepdim=True)
        distance = ref - x
        sign = -1.0
        
    elif anchor == "min":
        ref, _ = x.min(dim=-1, keepdim=True)
        distance = x - ref
        sign = 1.0
        
    limit = math.log(max_ratio)
    compressed_dist = limit * torch.tanh(distance / (limit + 1e-8))
    
    final_l = ref + (sign * compressed_dist)
    
    return final_l


@persistence.persistent_class
class PointUpsample_subpixel(torch.nn.Module):
    def __init__(
        self,
        in_features: int,  # Number of input features.
        out_features: int,  # Number of output features.
        upsample_ratio: int,
    ):
        super().__init__()

        self.upsample_ratio = upsample_ratio
        self.out_features = out_features

        self.subpixel = nn.Sequential(
            nn.Linear(in_features, out_features * self.upsample_ratio),
            nn.LeakyReLU(inplace=True),
        )
        self.res_fc = (
            nn.Identity()
            if in_features == out_features
            else nn.Linear(in_features, out_features)
        )

        nn.init.normal_(self.subpixel[0].weight, mean=0.0, std=0.01)
        nn.init.constant_(self.subpixel[0].bias, 0.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.upsample_ratio == 1:
            return x
        b, seq_len, _c = x.shape
        x_up = self.subpixel(x).reshape(
            b, seq_len * self.upsample_ratio, self.out_features
        )
        res = self.res_fc(x).repeat_interleave(self.upsample_ratio, dim=1)
        return (x_up + res) / math.sqrt(2)
    

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
class GaussAttrDecoder(nn.Module):
    """TransformerModel-style attribute decoder (no xyz).

    Note: here `shs` is treated as RGB color (per user request).
    """

    def __init__(self, in_dim: int):
        super().__init__()
        self.feature_channels = {"rotation": 4, "opacity": 1, "scale": 3, "color": 3}

        self.decoders = nn.ModuleDict(
            {k: nn.Linear(in_dim, c) for k, c in self.feature_channels.items()}
        )

        for key, layer in self.decoders.items():
            if key == "scaling":
                nn.init.xavier_uniform_(layer.weight, gain=0.3)
            elif key == "rotation":
                torch.nn.init.constant_(layer.bias, 0)
                torch.nn.init.constant_(layer.bias[0], 1.0)
            elif key == "opacity":
                torch.nn.init.constant_(layer.bias, inverse_sigmoid(0.5))
            elif key == "color":
                nn.init.constant_(layer.bias, 0.0)

    def forward(self, feats: torch.Tensor) -> dict[str, torch.Tensor]:
        out: dict[str, torch.Tensor] = {}
        for key, layer in self.decoders.items():
            raw = layer(feats)
            if key == "scale":
                log_s = bounded_log_sigmoid(raw, LOG_MIN, LOG_MAX)
                # log_s = soft_log_anchor_filter(log_s)
                out[key] = torch.exp(log_s)
            elif key == "opacity":
                out[key] = torch.sigmoid(raw)
            elif key == "rotation":
                out[key] = nn.functional.normalize(raw, dim=-1)
            elif key == "color":
                color = torch.tanh(raw) * 1.1
                out[key] = (color + 1) / 2
                # out[key] = raw

        return out


def inverse_sigmoid(x: float) -> float:
    return float(np.log(x / (1 - x)))


def bounded_log_sigmoid(
    raw: torch.Tensor, log_min: float, log_max: float
) -> torch.Tensor:
    span = log_max - log_min
    return log_min + (span * torch.sigmoid(raw))


class GaussianScene:
    def __init__(self, device, batch_size):
        self.xyz = torch.empty((batch_size, 0, 3), device=device)
        self.scale = torch.empty((batch_size, 0, 3), device=device)
        self.rotation = torch.empty((batch_size, 0, 4), device=device)
        self.opacity = torch.empty((batch_size, 0, 1), device=device)
        self.color = torch.empty((batch_size, 0, 3), device=device)

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

        self.conv_in = GaussianEncoding(sigma=10.0, input_size=3, encoded_size=512 // 2)  # this will be used by default
        self.n_transformer = options["n_transformer"]
        self.upsample_ratio =       [1, 4, 4,  4]
        self.upsample_ratio_accum = [1, 4, 16, 64]

        self.num_ws = 0
        self.transformer = Transformer(width=512, layers=self.n_transformer, w_dim=w_dim)
        self.upsample_layers = nn.ModuleList([
            PointUpsample_subpixel(
                in_features=512,
                out_features=512,
                upsample_ratio=self.upsample_ratio_accum[i]
            ) for i in range(self.n_transformer)
        ])

        self.attr_decoder = GaussAttrDecoder(512)

        self.xyz_head = nn.Linear(512, 3)

        nn.init.normal_(
            self.xyz_head[-1].weight, mean=0.0, std=float(0.01)
        )
        nn.init.constant_(self.xyz_head[-1].bias, 0.0)

    def forward(self, pos, edge_index, ws):
        B, num_points, C = pos.shape

        output_gaussians = GaussianScene(device=pos.device, batch_size=B)

        pos0 = pos * 0.1

        x = self.conv_in(pos0) # positional encoding

        transformer_out = self.transformer(x, pos0, edge_index, ws)

        for i in range(self.n_transformer):
            # create features (512 points, 512 channels)
            current_features_x, current_features_t = transformer_out[i]
            upsampled_features_x = self.upsample_layers[i](current_features_x)
            upsampled_features_t = self.upsample_layers[i](current_features_t)

            acc = self.upsample_ratio_accum[i]
            pos_up = pos0.repeat_interleave(acc, dim=1)
            pos_delta = torch.tanh(self.xyz_head(upsampled_features_x))
            pos_level = pos_up + pos_delta

            out_level = self.attr_decoder(upsampled_features_t)

            # generate gaussians
            new_gaussian = EasyDict(xyz=pos_level, **out_level)

            output_gaussians.concat(new_gaussian)

        # Output phase
        B, num_points, _ = output_gaussians.xyz.shape
        output_gaussians.xyz = output_gaussians.xyz.view(B, num_points, -1)
        output_gaussians.scale = output_gaussians.scale.view(B, num_points, -1)
        output_gaussians.rotation = output_gaussians.rotation.view(B, num_points, -1)
        output_gaussians.color = output_gaussians.color.view(B, num_points, -1)
        output_gaussians.opacity = output_gaussians.opacity.view(B, num_points, -1)

        return (
            output_gaussians.xyz,
            output_gaussians.scale,
            output_gaussians.rotation,
            output_gaussians.color,
            output_gaussians.opacity,
        )
