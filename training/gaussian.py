import torch
import torch.nn as nn
import numpy as np
from dnnlib import EasyDict


class GaussianDecoder(nn.Module):
    def __init__(
        self,
        feature_channels,
        in_channels,
        hidden_channles=128,
        use_rgb=True,
        use_pc=True,
    ):
        super().__init__()
        self.use_rgb = use_rgb
        self.use_pc = use_pc
        self.feature_channels = feature_channels

        self.mlp = nn.Sequential(
            nn.Linear(in_channels, 256),
            nn.LeakyReLU(inplace=True),
            nn.Linear(256, 256),
            nn.LeakyReLU(inplace=True),
        )

        self.decoders = torch.nn.ModuleList()

        for key, channels in self.feature_channels.items():
            layer = nn.Linear(256, channels)

            if key == "scaling":
                torch.nn.init.constant_(layer.bias, -5.0)
            elif key == "rotation":
                torch.nn.init.constant_(layer.bias, 0)
                torch.nn.init.constant_(layer.bias[0], 1.0)
            elif key == "opacity":
                torch.nn.init.constant_(layer.bias, np.log(0.1 / (1 - 0.1)))

            self.decoders.append(layer)

    def forward(self, x, pc=None):
        x = self.mlp(x)

        ret = {}
        for k, layer in zip(self.feature_channels.keys(), self.decoders):
            v = layer(x)
            if k == "rotation":
                v = torch.nn.functional.normalize(v)
            elif k == "scaling":
                v = trunc_exp(v)
                v = torch.clamp(v, min=0, max=0.02)
            elif k == "opacity":
                v = torch.sigmoid(v)
            elif k == "shs":
                v = torch.tanh(v) * 1.1
                v = (v + 1) / 2
                v = torch.reshape(v, (v.shape[0], 3))
            elif k == "xyz":
                max_step = 1.2 / 32
                v = (torch.sigmoid(v) - 0.5) * max_step
                v = v + pc
            ret[k] = v

        return EasyDict(**ret)


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
