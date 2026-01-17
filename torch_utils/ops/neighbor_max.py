# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

"""Custom PyTorch op for neighbor max aggregation."""

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
            module_name="neighbor_max_plugin",
            sources=["neighbor_max.cpp", "neighbor_max.cu"],
            headers=["neighbor_max.h"],
            source_dir=os.path.dirname(__file__),
            extra_cuda_cflags=["--use_fast_math"],
        )
    return True


#----------------------------------------------------------------------------


def neighbor_max(x: torch.Tensor, edge_index: torch.Tensor, impl: str = "cuda"):
    """Compute neighbor max aggregation for dense edge lists.

    Args:
        x: Tensor [B, N, C]
        edge_index: Tensor [N, K] (int64) dense neighbor table
        impl: "cuda" or "ref"

    Returns:
        values: Tensor [B, N, C]
        indices: Tensor [B, N, C] (neighbor slot 0..K-1)
    """
    assert isinstance(x, torch.Tensor)
    assert isinstance(edge_index, torch.Tensor)
    assert impl in ["ref", "cuda"]

    if impl == "cuda" and x.is_cuda and edge_index.is_cuda and _init():
        return _NeighborMaxCuda.apply(x, edge_index)

    return _neighbor_max_ref(x, edge_index)


@misc.profiled_function
def _neighbor_max_ref(x: torch.Tensor, edge_index: torch.Tensor):
    assert x.ndim == 3
    assert edge_index.ndim == 2
    x_neighbors = x[:, edge_index]
    return x_neighbors.max(dim=2)


class _NeighborMaxCuda(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, edge_index):
        x = x.contiguous()
        edge_index = edge_index.contiguous()
        values, indices = _plugin.neighbor_max_forward(x, edge_index)
        ctx.save_for_backward(edge_index, indices)
        return values, indices

    @staticmethod
    def backward(ctx, grad_values, grad_indices):
        edge_index, indices = ctx.saved_tensors
        grad_values = grad_values.contiguous()
        grad_x = _plugin.neighbor_max_backward(grad_values, edge_index, indices)
        return grad_x, None
