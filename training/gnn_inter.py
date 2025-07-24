import torch
from torch import nn
import numpy as np
from torch import Tensor
from torch_geometric.typing import Adj, OptTensor
from torch_geometric import nn as gnn
from torch_utils import persistence
from torch_geometric.utils import spmm
from training.networks_stylegan2 import FullyConnectedLayer
from torch_utils.ops import bias_act
from torch_utils import misc


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

    x = x.view(points_num, c_in, 1)
    out = torch.matmul(W, x)  # [num_rays, c_out, 1]
    out = out.view(points_num, c_out)  # [num_rays, c_out]

    return out


@persistence.persistent_class
class SynthesisLayer(torch.nn.Module):
    def __init__(
        self,
        in_channels,  # Number of input channels.
        out_channels,  # Number of output channels.
        w_dim,  # Intermediate latent (W) dimensionality.
        rank=10,  # Rank for FMM modulation (NEW parameter for fmm_modulate_linear).
        activation="lrelu",  # Activation function: 'relu', 'lrelu', etc.
        linear_clamp=None,  # Clamp the output of linear layers to +-X, None = disable clamping (RENAMED from conv_clamp).
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.w_dim = w_dim
        self.rank = rank  # Store rank
        self.activation = activation
        self.linear_clamp = linear_clamp  # Renamed clamp

        self.act_gain = bias_act.activation_funcs[activation].def_gain
        self.affine = FullyConnectedLayer(
            w_dim, (in_channels + out_channels) * rank, bias_init=1
        )
        self.weight = torch.nn.Parameter(torch.randn([out_channels, in_channels]))
        self.bias = torch.nn.Parameter(torch.zeros([out_channels]))

        # Removed parameters from original convolutional layer that are no longer applicable:
        # resolution, up, use_noise, kernel_size, resample_filter, padding, conv_clamp, channels_last
        # Also removed noise_const, noise_strength as noise is not part of fmm_modulate_linear.

    def forward(self, x, w, gain=1):
        x_in = x
        misc.assert_shape(
            x_in, [None, self.in_channels]
        )  # None for N_points (can be any number)

        # Ensure w is 1D: [w_dim].
        if w.ndim == 2 and w.shape[0] == 1:
            w_in = w.squeeze(0)  # [w_dim]
        elif w.ndim == 1:
            w_in = w  # [w_dim]
        else:
            raise ValueError(
                f"Input w must be 1D ([w_dim]) or 2D ([1, w_dim]), got {w.ndim}D shape {w.shape}"
            )

        misc.assert_shape(w_in, [self.w_dim])

        # Generate styles from the latent vector w. styles will be 1D.
        # self.affine expects [batch_size, w_dim] as input, so we add a batch_size of 1.
        # The output `styles` will be [1, (in_channels + out_channels) * rank], so we squeeze it to 1D.
        styles = self.affine(w_in.unsqueeze(0)).squeeze(0)

        # Apply Factorized Matrix Modulation linear layer.
        # The 'activation' argument in fmm_modulate_linear is for the modulation matrix itself (e.g., "demod").
        # The main layer activation ('lrelu', 'relu') is applied later via bias_act.
        x_modulated = fmm_modulate_linear(
            x=x_in,  # Input features [N_points, in_channels]
            weight=self.weight,  # Base weights [out_channels, in_channels]
            styles=styles,  # Modulation vector [ (in_channels + out_channels) * rank ]
            activation="demod",  # Type of activation/normalization for the modulation matrix
        )

        # Calculate gain and clamp for the final activation.
        act_gain = self.act_gain * gain
        act_clamp = self.linear_clamp * gain if self.linear_clamp is not None else None

        # Apply bias and activation function.
        x_out = bias_act.bias_act(
            x_modulated,
            self.bias.to(x_modulated.dtype),
            act=self.activation,
            gain=act_gain,
            clamp=act_clamp,
        )

        return x_out

    def extra_repr(self):
        return " ".join(
            [
                f"in_channels={self.in_channels:d}, out_channels={self.out_channels:d}, w_dim={self.w_dim:d},",
                f"rank={self.rank:d}, activation={self.activation:s}, linear_clamp={self.linear_clamp}",
            ]
        )


@persistence.persistent_class
class BiasBlock(torch.nn.Module):
    def __init__(
        self,
        num_nodes: int,
        in_channels: int,
        out_channels: int,
        w_dim: int,
        edge_scale_init: float = 0.01,
    ):
        super().__init__()
        self.num_nodes = num_nodes
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.edge_lin = nn.Parameter(torch.randn(num_nodes, out_channels))
        self.edge_lin2 = SynthesisLayer(out_channels, out_channels, w_dim=w_dim)
        self.linear = SynthesisLayer(in_channels, out_channels, w_dim=w_dim)
        self.edge_scale = nn.Parameter(torch.full((out_channels,), edge_scale_init))

    def forward(self, x: OptTensor, edge_index: Adj, w) -> Tensor:
        x = self.linear(x, w)

        out = self.edge_lin * self.edge_scale
        out = self.edge_lin2(out, w)

        return out + x


class GNNConv(gnn.MessagePassing):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        w_dim: int,
        edge_scale_init: float = 0.01,
        **kwargs,
    ):
        kwargs.setdefault("aggr", "add")
        super().__init__(**kwargs)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.edge_scale_init = edge_scale_init

        self.lin_in = SynthesisLayer(in_channels, out_channels, w_dim=w_dim)
        self.lin_edge = SynthesisLayer(out_channels, out_channels, w_dim=w_dim)
        self.edge_scale = nn.Parameter(torch.full((out_channels,), edge_scale_init))

    def forward(self, x: Tensor, edge_index: Adj, w) -> Tensor:
        x = self.lin_in(x, w)

        edges = x * self.edge_scale
        out = self.propagate(edge_index, x=edges)
        out = self.lin_edge(out, w)

        return out + x

    def message(self, x_j: Tensor) -> Tensor:
        return x_j

    def message_and_aggregate(self, adj_t: Adj, x) -> Tensor:
        return spmm(adj_t, x, reduce=self.aggr)
