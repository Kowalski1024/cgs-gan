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

std::vector<torch::Tensor> neighbor_max_forward_cuda(torch::Tensor x, torch::Tensor edge_index);
torch::Tensor neighbor_max_backward_cuda(torch::Tensor grad_y, torch::Tensor edge_index, torch::Tensor indices);
