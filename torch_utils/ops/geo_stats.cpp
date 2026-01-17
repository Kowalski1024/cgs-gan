/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: LicenseRef-NvidiaProprietary
 *
 * NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
 * property and proprietary rights in and to this material, related
 * documentation and any modifications thereto. Any use, reproduction,
 * disclosure or distribution of this material and related documentation
 * without an express license agreement from NVIDIA CORPORATION or
 * its affiliates is strictly prohibited.
 */

#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include "geo_stats.h"

//------------------------------------------------------------------------

static void validate_inputs(torch::Tensor pos, torch::Tensor delta, torch::Tensor edge_index)
{
    TORCH_CHECK(pos.is_cuda(), "pos must reside on CUDA device");
    TORCH_CHECK(delta.is_cuda(), "delta must reside on CUDA device");
    TORCH_CHECK(edge_index.is_cuda(), "edge_index must reside on CUDA device");
    TORCH_CHECK(pos.dim() == 2, "pos must have shape [N, 3]");
    TORCH_CHECK(pos.size(1) == 3, "pos must have shape [N, 3]");
    TORCH_CHECK(delta.dim() == 3, "delta must have shape [B, N, 3]");
    TORCH_CHECK(delta.size(2) == 3, "delta must have shape [B, N, 3]");
    TORCH_CHECK(edge_index.dim() == 2, "edge_index must have shape [N, K]");
    TORCH_CHECK(edge_index.scalar_type() == torch::kInt64, "edge_index must be int64");
    TORCH_CHECK(pos.scalar_type() == delta.scalar_type(), "pos and delta must have same dtype");
    TORCH_CHECK(pos.is_contiguous(), "pos must be contiguous");
    TORCH_CHECK(delta.is_contiguous(), "delta must be contiguous");
    TORCH_CHECK(edge_index.is_contiguous(), "edge_index must be contiguous");
    TORCH_CHECK(pos.size(0) == edge_index.size(0), "edge_index first dim must match N");
    TORCH_CHECK(delta.size(1) == pos.size(0), "delta second dim must match N");
}

//------------------------------------------------------------------------

std::vector<torch::Tensor> geo_stats_forward(torch::Tensor pos, torch::Tensor delta, torch::Tensor edge_index)
{
    validate_inputs(pos, delta, edge_index);
    const at::cuda::OptionalCUDAGuard device_guard(device_of(delta));
    return geo_stats_forward_cuda(pos, delta, edge_index);
}

//------------------------------------------------------------------------

torch::Tensor geo_stats_backward(
    torch::Tensor grad_var,
    torch::Tensor grad_mean,
    torch::Tensor grad_min,
    torch::Tensor grad_max,
    torch::Tensor delta,
    torch::Tensor edge_index)
{
    TORCH_CHECK(grad_var.is_cuda(), "grad_var must reside on CUDA device");
    TORCH_CHECK(grad_mean.is_cuda(), "grad_mean must reside on CUDA device");
    TORCH_CHECK(grad_min.is_cuda(), "grad_min must reside on CUDA device");
    TORCH_CHECK(grad_max.is_cuda(), "grad_max must reside on CUDA device");
    TORCH_CHECK(delta.is_cuda(), "delta must reside on CUDA device");
    TORCH_CHECK(edge_index.is_cuda(), "edge_index must reside on CUDA device");
    TORCH_CHECK(grad_var.is_contiguous(), "grad_var must be contiguous");
    TORCH_CHECK(grad_mean.is_contiguous(), "grad_mean must be contiguous");
    TORCH_CHECK(grad_min.is_contiguous(), "grad_min must be contiguous");
    TORCH_CHECK(grad_max.is_contiguous(), "grad_max must be contiguous");
    TORCH_CHECK(delta.is_contiguous(), "delta must be contiguous");
    TORCH_CHECK(edge_index.is_contiguous(), "edge_index must be contiguous");
    TORCH_CHECK(grad_var.dim() == 3, "grad_var must have shape [B, N, 3]");
    TORCH_CHECK(grad_mean.dim() == 3, "grad_mean must have shape [B, N, 3]");
    TORCH_CHECK(grad_min.dim() == 3, "grad_min must have shape [B, N, 3]");
    TORCH_CHECK(grad_max.dim() == 3, "grad_max must have shape [B, N, 3]");
    TORCH_CHECK(delta.dim() == 3, "delta must have shape [B, N, 3]");
    TORCH_CHECK(edge_index.dim() == 2, "edge_index must have shape [N, K]");
    TORCH_CHECK(grad_var.size(1) == edge_index.size(0), "edge_index first dim must match N");
    TORCH_CHECK(grad_var.sizes() == grad_mean.sizes(), "grad_mean must match grad_var shape");
    TORCH_CHECK(grad_var.sizes() == grad_min.sizes(), "grad_min must match grad_var shape");
    TORCH_CHECK(grad_var.sizes() == grad_max.sizes(), "grad_max must match grad_var shape");
    TORCH_CHECK(grad_var.sizes() == delta.sizes(), "delta must match grad_var shape");
    const at::cuda::OptionalCUDAGuard device_guard(device_of(delta));
    return geo_stats_backward_cuda(grad_var, grad_mean, grad_min, grad_max, delta);
}

//------------------------------------------------------------------------

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("geo_stats_forward", &geo_stats_forward);
    m.def("geo_stats_backward", &geo_stats_backward);
}

//------------------------------------------------------------------------
