import math
import numpy as np
import torch
from torch import nn
import rff
from torch_geometric import nn as gnn
from torch_geometric.nn import SAGEConv
from torch_geometric.typing import Adj, OptTensor


class GradScaler(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, scale):
        ctx.scale = scale
        return x

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output * ctx.scale, None


def scale_grad(x, scale):
    return GradScaler.apply(x, scale)


# --- StyleGAN2 Components (Adapted from gnn_point_generator.py) ---
class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.scale = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor, _=None) -> torch.Tensor:
        norm = x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return norm * self.scale


def inverse_sigmoid(x):
    return np.log(x / (1 - x))


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

    # Construct [c_out, c_in] matrix from styles
    left_matrix = styles[: c_out * rank]
    right_matrix = styles[c_out * rank :]

    left_matrix = left_matrix.view(c_out, rank)
    right_matrix = right_matrix.view(rank, c_in)

    # Modulation
    modulation = left_matrix @ right_matrix / np.sqrt(rank)

    if activation == "tanh":
        modulation = modulation.tanh()
    elif activation == "sigmoid":
        modulation = modulation.sigmoid() - 0.5

    W = weight * (modulation + 1.0)
    if activation == "demod":
        W = W / (W.norm(dim=1, keepdim=True) + 1e-8)
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
        # The affine layer projects latent w to style parameters
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
        # w: [w_dim] (Single latent vector for the batch/layer)
        # Note: This implementation assumes w is [w_dim], not [B, w_dim] if processing one by one.
        # But TwoStageGANGenerator iterates over batch, so w is [w_dim].

        styles = self.affine(w).squeeze(0)  # [style_dim]

        x = fmm_modulate_linear(
            x=x, weight=self.weight, styles=styles, activation="demod"
        )

        if self.bias is not None:
            x = x + self.bias.view(1, -1)

        if self.activation is not None:
            x = self.activation(x)

        if self.noise:
            noise = (
                torch.randn(x.shape[0], self.out_channels, device=x.device)
                * self.noise_strength
            )
            x = x + noise
        return x


# --- Styled Components ---


class StyledPointGNNConv(gnn.MessagePassing):
    """
    PointGNNConv where the internal MLPs are modulated by style w.
    """

    def __init__(
        self,
        channels_in,
        channels_out,
        w_dim,
        **kwargs,
    ):
        kwargs.setdefault("aggr", "mean")
        super().__init__(**kwargs)

        # mlp_h: Calculates delta (offset)
        self.mlp_h = nn.ModuleList(
            [
                SynthesisLayer(channels_in, channels_in // 2, w_dim),
                SynthesisLayer(
                    channels_in // 2, 3, w_dim, activation=nn.Tanh(), noise=False
                ),
            ]
        )

        # mlp_g: Updates features
        self.mlp_g = nn.ModuleList(
            [
                SynthesisLayer(channels_out + 3, channels_out, w_dim),
                SynthesisLayer(
                    channels_out,
                    channels_out,
                    w_dim,
                    activation=nn.Identity(),
                    noise=False,
                ),
            ]
        )

        self.layer_scale = nn.Parameter(torch.ones(channels_in) * 0.01)

    def forward(
        self, x: torch.Tensor, pos: torch.Tensor, edge_index: Adj, w: torch.Tensor
    ) -> torch.Tensor:
        # Calculate delta
        delta = x
        for layer in self.mlp_h:
            delta = layer(delta, w)

        # Propagate
        out = self.propagate(edge_index, x=x * self.layer_scale, pos=pos, delta=delta)

        # Update features
        for layer in self.mlp_g:
            out = layer(out, w)

        return x + out

    def message(
        self,
        pos_j: torch.Tensor,
        pos_i: torch.Tensor,
        x_i: torch.Tensor,
        x_j: torch.Tensor,
        delta_i: torch.Tensor,
    ) -> torch.Tensor:
        return torch.cat([pos_j - pos_i + delta_i, x_j], dim=-1)


class StyledLINKX(nn.Module):
    """
    LINKX where the MLPs are modulated by style w.
    """

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        out_channels: int,
        w_dim: int,
        num_layers: int = 2,
    ):
        super().__init__()

        # SAGEConv for geometric aggregation (hard to modulate directly, keeping standard)
        self.edge_lin = SAGEConv(in_channels, hidden_channels)

        # Node MLP (Residual path)
        self.node_mlp = SynthesisLayer(in_channels, hidden_channels, w_dim, noise=False)

        # Mixing layers
        self.cat_lin1 = SynthesisLayer(
            hidden_channels,
            hidden_channels,
            w_dim,
            activation=nn.Identity(),
            noise=False,
        )
        self.cat_lin2 = SynthesisLayer(
            hidden_channels,
            hidden_channels,
            w_dim,
            activation=nn.Identity(),
            noise=False,
        )

        # Final MLP
        self.final_mlp = nn.ModuleList()
        channels = [hidden_channels] * num_layers + [out_channels]
        for i in range(len(channels) - 1):
            self.final_mlp.append(SynthesisLayer(channels[i], channels[i + 1], w_dim))

    def forward(
        self,
        x: torch.Tensor,
        edge_index: Adj,
        w: torch.Tensor,
    ) -> torch.Tensor:
        # 1. Graph Aggregation
        out = self.edge_lin(x, edge_index)
        out = nn.functional.leaky_relu(out, inplace=True)

        # 2. Modulated Mixing
        out = self.cat_lin1(out, w)

        if x is not None:
            x_res = self.node_mlp(x, w)
            out = out + x_res
            out = out + self.cat_lin2(x_res, w)

        out = nn.functional.leaky_relu(out, inplace=True)

        # 3. Final Modulated MLP
        for layer in self.final_mlp:
            out = layer(out, w)

        return out


class _TruncExp(torch.autograd.Function):  # pylint: disable=abstract-method
    # Implementation from torch-ngp:
    # https://github.com/ashawkey/torch-ngp/blob/93b08a0d4ec1cc6e69d85df7f0acdfb99603b628/activation.py
    @staticmethod
    @torch.cuda.amp.custom_fwd(cast_inputs=torch.float32)
    def forward(ctx, x):  # pylint: disable=arguments-differ
        ctx.save_for_backward(x)
        return torch.exp(x)

    @staticmethod
    @torch.cuda.amp.custom_bwd
    def backward(ctx, g):  # pylint: disable=arguments-differ
        x = ctx.saved_tensors[0]
        return g * torch.exp(torch.clamp(x, max=15))


trunc_exp = _TruncExp.apply


class StyledGaussDecoder(nn.Module):
    """
    Gaussian Decoder with modulated MLPs.
    """

    def __init__(
        self,
        in_dim: int,
        mid_dim: int,
        w_dim: int,
        shs_degree: int = 0,
        use_rgb: bool = False,
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

        # 1. Geometry Branch (Scale, Rotation, Opacity)
        # Compresses to mid_dim to force compact geometric representation
        self.geo_mlp = nn.Sequential(
            nn.Linear(in_dim, in_dim),
            RMSNorm(in_dim),
            nn.LeakyReLU(inplace=True),
            nn.Linear(in_dim, mid_dim),
            RMSNorm(mid_dim),
        )

        # 2. Appearance Branch (Color/SH)
        # Reverted to RMSNorm + LeakyReLU for stability.
        # Kept the expansion (in_dim * 2) for capacity.
        self.color_mlp = nn.Sequential(
            nn.Linear(in_dim, in_dim * 2),
            RMSNorm(in_dim * 2),
            nn.LeakyReLU(inplace=True),
            nn.Linear(in_dim * 2, in_dim),
            RMSNorm(in_dim),
            nn.LeakyReLU(inplace=True),
        )

        self.decoders = torch.nn.ModuleList()

        for key, channels in self.feature_channels.items():
            if key in ["color", "shs"]:
                layer = nn.Linear(in_dim, channels)
            else:
                layer = nn.Linear(mid_dim, channels)

            if key == "scaling":
                torch.nn.init.constant_(layer.bias, -5.0)
            elif key == "shs":
                torch.nn.init.constant_(layer.bias, 0.0)
                torch.nn.init.constant_(layer.weight, 0.0)
            elif key == "rotation":
                torch.nn.init.constant_(layer.bias, 0)
                torch.nn.init.constant_(layer.bias[0], 1.0)
            elif key == "opacity":
                torch.nn.init.constant_(layer.bias, inverse_sigmoid(0.05))
            elif key == "color":
                # Initialize for Sigmoid
                # Sigmoid(0) = 0.5 (Grey)
                nn.init.xavier_uniform_(layer.weight, gain=0.1)
                nn.init.constant_(layer.bias, 0.0)

            self.decoders.append(layer)

        self.reset_parameters()

    def reset_parameters(self):
        for i, (key, _) in enumerate(self.feature_channels.items()):
            layer = self.decoders[i]
            if key == "scaling":
                torch.nn.init.constant_(layer.bias, -5.0)
            elif key == "shs":
                torch.nn.init.constant_(layer.bias, 0.0)
                torch.nn.init.constant_(layer.weight, 0.0)
            elif key == "rotation":
                torch.nn.init.constant_(layer.bias, 0)
                torch.nn.init.constant_(layer.bias[0], 1.0)
            elif key == "opacity":
                torch.nn.init.constant_(layer.bias, inverse_sigmoid(0.05))
            elif key == "color":
                nn.init.xavier_uniform_(layer.weight, gain=0.1)
                nn.init.constant_(layer.bias, 0.0)

    def forward(self, features: torch.Tensor) -> dict[str, torch.Tensor]:
        geo_features = self.geo_mlp(features)
        color_features = self.color_mlp(features)

        # Throttle color gradients
        color_features = scale_grad(color_features, 0.5)

        outputs = {}
        for i, (key, _) in enumerate(self.feature_channels.items()):
            if key in ["color", "shs"]:
                feature = self.decoders[i](color_features)
            else:
                feature = self.decoders[i](geo_features)

            if key == "scaling":
                # Softplus is safer than Exp for stability.
                # Adding a bias ensures we don't start at 0 size.
                scales = torch.nn.functional.softplus(feature) + 0.001

                # HARD CONSTRAINT: Limit max anisotropy (max_scale / min_scale)
                # This is a differentiable approximation to keep scales somewhat uniform
                # Ideally, handle this in the loss, but architectural clamping works too.
                out = torch.clamp(scales, max=1.0)
            elif key == "opacity":
                out = torch.sigmoid(feature)
            elif key == "rotation":
                out = nn.functional.normalize(feature, dim=-1)
            elif key == "shs":
                _features_dc = feature.unsqueeze(1).contiguous()
                _features_rest = feature.unsqueeze(1)[:, 0:0].contiguous()
                out = torch.cat((_features_dc, _features_rest), dim=-1)
            elif key == "color":
                # Sigmoid * 1.2 - 0.1
                # Range: [-0.1, 1.1]
                # Allows reaching 0 and 1 easily, but uses Sigmoid curve which matches [0,1] data better
                color = torch.sigmoid(feature) * 1.2 - 0.1
                out = color

            outputs[key] = out

        return outputs


class PointGenerator(nn.Module):
    """
    StyleGAN2-inspired Generator using Modulated Layers.
    """

    def __init__(
        self,
        w_dim: int,
        options: dict = {},
    ):
        super().__init__()
        self.num_points = options.get("num_pts", 8192)
        self.k = options.get("k", 20)
        shs_degree = options.get("shs_degree", 0)
        use_rgb = options.get("use_rgb", True)
        self.num_ws = options.get("num_ws", 18)
        self.w_dim = w_dim

        # 1. Positional Encoding
        self.encoder = rff.layers.GaussianEncoding(
            sigma=10.0, input_size=3, encoded_size=64
        )

        # 2. Stage 1 Backbone (Geometry) - Modulated
        self.point_convs = nn.ModuleList(
            [
                StyledPointGNNConv(128, 128, w_dim),
                StyledPointGNNConv(128, 128, w_dim),
            ]
        )

        self.global_conv = nn.Sequential(
            nn.Linear(128, 128),
            RMSNorm(128),
            nn.LeakyReLU(inplace=True),
        )

        # 3. Position Decoder - Modulated
        self.position_decoder = nn.ModuleList(
            [
                SynthesisLayer(256, 128, w_dim),
                RMSNorm(128),
                SynthesisLayer(128, 128, w_dim),
                RMSNorm(128),
                nn.Linear(128, 3),  # Final projection standard
            ]
        )
        nn.init.normal_(self.position_decoder[-1].weight, mean=0.0, std=0.001)
        nn.init.constant_(self.position_decoder[-1].bias, 0.0)

        # 4. Stage 2 Backbone (Appearance) - Modulated
        self.pos_encoder2 = rff.layers.GaussianEncoding(
            sigma=10.0, input_size=3, encoded_size=64
        )

        self.gaussian_conv = nn.ModuleList(
            [
                StyledLINKX(256 + 128, 256, 256, w_dim, num_layers=2),
                StyledLINKX(256, 256, 256, w_dim, num_layers=2),
            ]
        )

        # 5. Gaussian Decoder - Modulated
        self.gaussian_decoder = StyledGaussDecoder(
            512, 128, w_dim, shs_degree=shs_degree, use_rgb=use_rgb
        )

    def forward_single(self, pos, x, edge_index, w):
        # --- STAGE 1: GEOMETRY ---
        if x is None or x.shape[-1] != 128:
            x = self.encoder(pos)

        # Modulated GNN Convs
        for conv in self.point_convs:
            x = conv(x, pos, edge_index, w)

        # Global Feature Injection (Concatenation still useful for global context)
        # But we can also rely on modulation.
        # Let's keep concatenation to match TwoStageModel structure,
        # but generate 'h' via a modulated layer or just use w?
        # TwoStageModel used global_max_pool -> MLP.
        # Here we can just repeat w? Or project w.
        # Let's project w to 128 to match dimensions.
        # Actually, let's just use a learnable constant or the pooled features modulated by w.
        # Simpler: Just use the pooled features from x, modulated.

        # Global Pooling
        h = gnn.global_max_pool(x, None)  # [1, 128]
        h = self.global_conv(h)
        h = h.repeat(x.size(0), 1)  # [N, 128]

        x = torch.cat([x, h], dim=-1)  # [N, 256]
        point_features = x

        # Predict Position
        pos_feat = point_features
        for layer in self.position_decoder[:-1]:
            pos_feat = layer(pos_feat, w)

        # Throttle position gradients
        pos_feat = scale_grad(pos_feat, 0.1)

        new_pos = self.position_decoder[-1](pos_feat)

        # --- STAGE 2: APPEARANCE ---
        pos_features = self.pos_encoder2(new_pos)
        x_stage2 = torch.cat([x, pos_features], dim=-1)  # [N, 256 + 128]

        for conv in self.gaussian_conv:
            x_stage2 = conv(x_stage2, edge_index, w)

        # Decode
        # Concatenate Stage 1 features (x) with Stage 2 features (x_stage2)
        decoder_input = torch.cat([x_stage2, x], dim=-1)  # [N, 256 + 256]
        gaussian_params = self.gaussian_decoder(decoder_input)

        xyz = new_pos
        scale = gaussian_params["scaling"]
        rotation = gaussian_params["rotation"]
        opacity = gaussian_params["opacity"]

        if "color" in gaussian_params:
            color = gaussian_params["color"]
        else:
            color = gaussian_params["shs"]

        return xyz, scale, rotation, color, opacity

    def forward(self, pos, x, edge_index, ws):
        B = ws.shape[0]
        xyz_list, scale_list, rot_list, color_list, opac_list = [], [], [], [], []

        for i in range(B):
            if ws.dim() == 3:
                w = ws[i, 0]
            else:
                w = ws[i]

            xyz, scale, rot, color, opacity = self.forward_single(pos, x, edge_index, w)

            xyz_list.append(xyz)
            scale_list.append(scale)
            rot_list.append(rot)
            color_list.append(color)
            opac_list.append(opacity)

        return (
            torch.stack(xyz_list),
            torch.stack(scale_list),
            torch.stack(rot_list),
            torch.stack(color_list),
            torch.stack(opac_list),
        )
