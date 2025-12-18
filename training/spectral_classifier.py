"""Spectral-domain classifier (PyTorch port of SSD-GAN StyleGAN `C_basic`).

This module computes an FFT radial profile from grayscale images and returns
logits via a spectrally-normalized linear layer.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Union

import torch

from torch_utils import persistence


def _to_image_tensor(img: Union[torch.Tensor, Dict[str, Any]]) -> torch.Tensor:
    if isinstance(img, dict):
        return img["image"]
    return img


@persistence.persistent_class
class SpectralDomainClassifier(torch.nn.Module):
    """Spectral domain classifier (SSD-GAN `C_basic` port).

    Input: RGB images in NCHW.
    Output: logits of shape [N, max(label_size, 1)].

    The network:
    - Convert to grayscale
    - Compute 2D FFT magnitude spectrum (log scale)
    - Compute radial average profile
    - Min-max normalize profile per sample
    - Apply spectrally-normalized linear layer
    """

    def __init__(
        self,
        img_resolution: int,
        img_channels: int = 3,
        label_size: int = 0,
        use_spectral_norm: bool = True,
        eps: float = 1e-10,
    ):
        super().__init__()
        if img_resolution < 4 or (img_resolution & (img_resolution - 1)) != 0:
            raise ValueError(f"img_resolution must be power-of-two >= 4, got {img_resolution}")
        if img_channels not in (1, 3):
            raise ValueError(f"img_channels must be 1 or 3, got {img_channels}")

        self.img_resolution = int(img_resolution)
        self.img_channels = int(img_channels)
        self.label_size = int(label_size)
        self.out_dim = max(self.label_size, 1)
        self.eps = float(eps)

        # Precompute radius bins for the given resolution.
        y = torch.arange(self.img_resolution, dtype=torch.float32)
        x = torch.arange(self.img_resolution, dtype=torch.float32)
        yy, xx = torch.meshgrid(y, x, indexing="ij")
        center = self.img_resolution / 2.0
        radius = torch.sqrt((xx - center) ** 2 + (yy - center) ** 2).to(torch.long)
        radius_flat = radius.reshape(-1)
        profile_dim = int(radius_flat.max().item()) + 1

        # Counts per radius bin.
        nr = torch.bincount(radius_flat, minlength=profile_dim).to(torch.float32)

        self.profile_dim = profile_dim
        self.register_buffer("radius_flat", radius_flat, persistent=False)
        self.register_buffer("nr", nr, persistent=False)

        fc = torch.nn.Linear(self.profile_dim, self.out_dim, bias=True)
        self.fc = torch.nn.utils.spectral_norm(fc) if use_spectral_norm else fc

    def forward(self, img: Union[torch.Tensor, Dict[str, Any]], c: Optional[torch.Tensor] = None) -> torch.Tensor:
        _ = c  # label conditioning is handled by the loss in SSD; kept for API compatibility.
        img_t = _to_image_tensor(img)
        if img_t.ndim != 4:
            raise ValueError(f"Expected NCHW image tensor, got shape {tuple(img_t.shape)}")
        n, ch, h, w = img_t.shape
        if ch != self.img_channels:
            raise ValueError(f"Expected {self.img_channels} channels, got {ch}")
        if h != self.img_resolution or w != self.img_resolution:
            raise ValueError(
                f"Expected resolution {self.img_resolution}x{self.img_resolution}, got {h}x{w}"
            )

        # Convert to grayscale.
        img_t = img_t.to(torch.float32)
        if self.img_channels == 3:
            r = img_t[:, 0, :, :]
            g = img_t[:, 1, :, :]
            b = img_t[:, 2, :, :]
            gray = 0.2989 * r + 0.5870 * g + 0.1140 * b
        else:
            gray = img_t[:, 0, :, :]

        # FFT magnitude spectrum (log scale).
        fft = torch.fft.fft2(gray)
        mag = 20.0 * torch.log(torch.abs(fft) + self.eps)

        # Radial profile: average magnitude per radius bin.
        # Implemented via scatter-add for broad CUDA compatibility.
        mag_flat = mag.reshape(n, -1)
        radial = []
        for i in range(n):
            sums = torch.zeros(self.profile_dim, device=mag.device, dtype=mag.dtype)
            sums.scatter_add_(0, self.radius_flat, mag_flat[i])
            prof = sums / (self.nr.to(mag.dtype) + self.eps)
            radial.append(prof)
        radial_prof = torch.stack(radial, dim=0)

        # Min-max normalize per sample.
        rp_max = radial_prof.max(dim=1, keepdim=True).values
        rp_min = radial_prof.min(dim=1, keepdim=True).values
        radial_prof = (radial_prof - rp_min) / (rp_max - rp_min + self.eps)

        scores = self.fc(radial_prof)
        return scores
