# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

"""Custom PyTorch op for geometric neighbor statistics."""

import os
import torch

from .. import custom_ops
from .. import misc

#----------------------------------------------------------------------------

_plugin = None


def _init():
    global _plugin
    if _plugin is None:
        _plugin = custom_ops.get_plugin(
            module_name="geo_stats_plugin",
            sources=["geo_stats.cpp", "geo_stats.cu"],
            headers=["geo_stats.h"],
            source_dir=os.path.dirname(__file__),
            extra_cuda_cflags=["--use_fast_math"],
        )
    return True


#----------------------------------------------------------------------------


def geo_stats(pos: torch.Tensor, delta: torch.Tensor, edge_index: torch.Tensor, impl: str = "cuda"):
    """Compute geometric statistics over dense edge lists.

    Args:
        pos: Tensor [N, 3]
        delta: Tensor [B, N, 3]
        edge_index: Tensor [N, K] (int64) dense neighbor table
        impl: "cuda" or "ref"

    Returns:
        var: Tensor [B, N, 3]
        mean: Tensor [B, N, 3]
        min: Tensor [B, N, 3]
        max: Tensor [B, N, 3]
    """
    assert isinstance(pos, torch.Tensor)
    assert isinstance(delta, torch.Tensor)
    assert isinstance(edge_index, torch.Tensor)
    assert impl in ["ref", "cuda"]

    if impl == "cuda" and pos.is_cuda and delta.is_cuda and edge_index.is_cuda and _init():
        return _GeoStatsCuda.apply(pos, delta, edge_index)

    return _geo_stats_ref(pos, delta, edge_index)


@misc.profiled_function
def _geo_stats_ref(pos: torch.Tensor, delta: torch.Tensor, edge_index: torch.Tensor):
    assert pos.ndim == 2
    assert delta.ndim == 3
    assert edge_index.ndim == 2
    rel_pos = (pos[edge_index] - pos.unsqueeze(1)).unsqueeze(0)  # [1, N, K, 3]
    rel_pos = rel_pos + delta.unsqueeze(2)  # [B, N, K, 3]
    pos_var, pos_mean = torch.var_mean(rel_pos, dim=2, unbiased=False)
    pos_min = torch.amin(rel_pos, dim=2)
    pos_max = torch.amax(rel_pos, dim=2)
    return pos_var, pos_mean, pos_min, pos_max


class _GeoStatsCuda(torch.autograd.Function):
    @staticmethod
    def forward(ctx, pos, delta, edge_index):
        pos = pos.contiguous()
        delta = delta.contiguous()
        edge_index = edge_index.contiguous()
        pos_var, pos_mean, pos_min, pos_max = _plugin.geo_stats_forward(pos, delta, edge_index)
        ctx.save_for_backward(delta, edge_index)
        return pos_var, pos_mean, pos_min, pos_max

    @staticmethod
    def backward(ctx, grad_var, grad_mean, grad_min, grad_max):
        delta, edge_index = ctx.saved_tensors
        if grad_var is None:
            grad_var = torch.zeros_like(delta)
        if grad_mean is None:
            grad_mean = torch.zeros_like(delta)
        if grad_min is None:
            grad_min = torch.zeros_like(delta)
        if grad_max is None:
            grad_max = torch.zeros_like(delta)
        grad_var = grad_var.contiguous()
        grad_mean = grad_mean.contiguous()
        grad_min = grad_min.contiguous()
        grad_max = grad_max.contiguous()
        grad_delta = _plugin.geo_stats_backward(grad_var, grad_mean, grad_min, grad_max, delta, edge_index)
        return None, grad_delta, None
