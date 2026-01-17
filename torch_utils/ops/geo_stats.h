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

#pragma once

#include <torch/extension.h>
#include <vector>

std::vector<torch::Tensor> geo_stats_forward_cuda(torch::Tensor pos, torch::Tensor delta, torch::Tensor edge_index);
torch::Tensor geo_stats_backward_cuda(
    torch::Tensor grad_var,
    torch::Tensor grad_mean,
    torch::Tensor grad_min,
    torch::Tensor grad_max,
    torch::Tensor delta);
