import torch
import torch.nn as nn
from typing import NamedTuple
import numpy as np


class GaussianModel(NamedTuple):
    xyz: torch.Tensor = None
    opacity: torch.Tensor = None
    rotation: torch.Tensor = None
    scaling: torch.Tensor = None
    shs: torch.Tensor = None

    def load_state_dict(self, state_dict):
        for k, v in state_dict.items():
            setattr(self, k, v)

    def __repr__(self) -> str:
        return f"GaussianModel(xyz={self.xyz.shape}, opacity={self.opacity.shape}, rotation={self.rotation.shape}, scaling={self.scaling.shape}, shs={self.shs.shape})"


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
            nn.Linear(in_channels, hidden_channles),
            nn.LeakyReLU(inplace=True),
            nn.Linear(hidden_channles, hidden_channles),
            nn.LeakyReLU(inplace=True),
        )

        self.decoders = torch.nn.ModuleList()
        self.scaling_modulator = None

        for key, channels in self.feature_channels.items():
            layer = nn.Linear(hidden_channles, channels)

            if key == "scaling":
                # torch.nn.init.constant_(layer.bias, -5.0)
                self.scaling_modulator = nn.Linear(hidden_channles, 3)
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
                v = torch.nn.functional.normalize(v, dim=-1)
            elif k == "scaling":
                scale_base = trunc_exp(v - 4.0)
                scale_base = torch.clamp(scale_base, min=1e-4, max=0.03)

                # modulator = self.scaling_modulator(x)
                v = scale_base # * torch.sigmoid(modulator)

                if (v < 0.0).any():
                    print("WARNING: Non-positive scales detected!")

                if torch.isinf(v).any():
                    print("CRITICAL ERROR: Infs in final scales before returning!")

                if torch.isnan(v).any():
                    print("CRITICAL ERROR: NaNs in final scales before returning!")
            elif k == "opacity":
                v = torch.sigmoid(v)
            elif k == "shs":
                pass
            elif k == "xyz":
                v = pc
            ret[k] = v

        return ret


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
