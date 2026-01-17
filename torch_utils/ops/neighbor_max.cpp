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
#include "neighbor_max.h"

//------------------------------------------------------------------------

static void validate_inputs(torch::Tensor x, torch::Tensor edge_index)
{
    TORCH_CHECK(x.is_cuda(), "x must reside on CUDA device");
    TORCH_CHECK(edge_index.is_cuda(), "edge_index must reside on CUDA device");
    TORCH_CHECK(x.dim() == 3, "x must have shape [B, N, C]");
    TORCH_CHECK(edge_index.dim() == 2, "edge_index must have shape [N, K]");
    TORCH_CHECK(edge_index.scalar_type() == torch::kInt64, "edge_index must be int64");
    TORCH_CHECK(x.is_contiguous(), "x must be contiguous");
    TORCH_CHECK(edge_index.is_contiguous(), "edge_index must be contiguous");
    TORCH_CHECK(x.size(1) == edge_index.size(0), "edge_index first dim must match N");
}

//------------------------------------------------------------------------

std::vector<torch::Tensor> neighbor_max_forward(torch::Tensor x, torch::Tensor edge_index)
{
    validate_inputs(x, edge_index);
    const at::cuda::OptionalCUDAGuard device_guard(device_of(x));
    return neighbor_max_forward_cuda(x, edge_index);
}

//------------------------------------------------------------------------

torch::Tensor neighbor_max_backward(torch::Tensor grad_y, torch::Tensor edge_index, torch::Tensor indices)
{
    TORCH_CHECK(grad_y.is_cuda(), "grad_y must reside on CUDA device");
    TORCH_CHECK(indices.is_cuda(), "indices must reside on CUDA device");
    TORCH_CHECK(indices.scalar_type() == torch::kInt64, "indices must be int64");
    TORCH_CHECK(grad_y.is_contiguous(), "grad_y must be contiguous");
    TORCH_CHECK(indices.is_contiguous(), "indices must be contiguous");
    TORCH_CHECK(grad_y.dim() == 3, "grad_y must have shape [B, N, C]");
    TORCH_CHECK(edge_index.dim() == 2, "edge_index must have shape [N, K]");
    TORCH_CHECK(grad_y.size(1) == edge_index.size(0), "edge_index first dim must match N");
    const at::cuda::OptionalCUDAGuard device_guard(device_of(grad_y));
    return neighbor_max_backward_cuda(grad_y, edge_index, indices);
}

//------------------------------------------------------------------------

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("neighbor_max_forward", &neighbor_max_forward);
    m.def("neighbor_max_backward", &neighbor_max_backward);
}

//------------------------------------------------------------------------
