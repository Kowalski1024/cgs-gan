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
#include <cuda.h>
#include <cuda_runtime.h>
#include "geo_stats.h"

//------------------------------------------------------------------------

template <typename scalar_t>
__global__ void geo_stats_forward_kernel(
    const scalar_t* __restrict__ pos,
    const scalar_t* __restrict__ delta,
    const int64_t* __restrict__ edge,
    scalar_t* __restrict__ out_var,
    scalar_t* __restrict__ out_mean,
    scalar_t* __restrict__ out_min,
    scalar_t* __restrict__ out_max,
    int B, int N, int K)
{
    int64_t index = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    int64_t total = (int64_t)B * N * 3;
    if (index >= total)
        return;

    int64_t tmp = index;
    int64_t b = tmp / (N * 3);
    tmp -= b * (N * 3);
    int64_t n = tmp / 3;
    int64_t c = tmp - n * 3;

    const int64_t pos_base = n * 3 + c;
    const scalar_t pos_i = pos[pos_base];
    const int64_t delta_index = (b * N + n) * 3 + c;
    const scalar_t delta_v = delta[delta_index];

    float sum = 0.0f;
    float sumsq = 0.0f;
    float min_v = 0.0f;
    float max_v = 0.0f;

    for (int64_t k = 0; k < K; ++k)
    {
        int64_t nb = edge[n * K + k];
        float v = (float)(pos[nb * 3 + c] - pos_i + delta_v);
        sum += v;
        sumsq += v * v;
        if (k == 0)
        {
            min_v = v;
            max_v = v;
        }
        else
        {
            if (v < min_v) min_v = v;
            if (v > max_v) max_v = v;
        }
    }

    float mean = sum / (float)K;
    float var = sumsq / (float)K - mean * mean;
    if (var < 0.0f) var = 0.0f;

    out_var[index] = (scalar_t)var;
    out_mean[index] = (scalar_t)mean;
    out_min[index] = (scalar_t)min_v;
    out_max[index] = (scalar_t)max_v;
}

//------------------------------------------------------------------------

template <typename scalar_t>
__global__ void geo_stats_backward_kernel(
    const scalar_t* __restrict__ grad_var,
    const scalar_t* __restrict__ grad_mean,
    const scalar_t* __restrict__ grad_min,
    const scalar_t* __restrict__ grad_max,
    scalar_t* __restrict__ grad_delta,
    int total)
{
    int64_t index = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= total)
        return;

    grad_delta[index] = grad_mean[index] + grad_min[index] + grad_max[index];
    (void)grad_var;
}

//------------------------------------------------------------------------

std::vector<torch::Tensor> geo_stats_forward_cuda(torch::Tensor pos, torch::Tensor delta, torch::Tensor edge_index)
{
    const int B = (int)delta.size(0);
    const int N = (int)delta.size(1);
    const int K = (int)edge_index.size(1);

    auto out_var = torch::empty({B, N, 3}, delta.options());
    auto out_mean = torch::empty({B, N, 3}, delta.options());
    auto out_min = torch::empty({B, N, 3}, delta.options());
    auto out_max = torch::empty({B, N, 3}, delta.options());

    const int threads = 256;
    const int64_t total = (int64_t)B * N * 3;
    const int blocks = (int)((total + threads - 1) / threads);

    const at::cuda::OptionalCUDAGuard device_guard(device_of(delta));
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(delta.scalar_type(), "geo_stats_forward_cuda", [&]
    {
        geo_stats_forward_kernel<scalar_t><<<blocks, threads, 0, stream>>>(
            pos.data_ptr<scalar_t>(),
            delta.data_ptr<scalar_t>(),
            edge_index.data_ptr<int64_t>(),
            out_var.data_ptr<scalar_t>(),
            out_mean.data_ptr<scalar_t>(),
            out_min.data_ptr<scalar_t>(),
            out_max.data_ptr<scalar_t>(),
            B, N, K);
    });
    AT_CUDA_CHECK(cudaGetLastError());

    return {out_var, out_mean, out_min, out_max};
}

//------------------------------------------------------------------------

torch::Tensor geo_stats_backward_cuda(
    torch::Tensor grad_var,
    torch::Tensor grad_mean,
    torch::Tensor grad_min,
    torch::Tensor grad_max,
    torch::Tensor delta)
{
    const int B = (int)delta.size(0);
    const int N = (int)delta.size(1);

    auto grad_delta = torch::empty({B, N, 3}, delta.options());

    const int threads = 256;
    const int64_t total = (int64_t)B * N * 3;
    const int blocks = (int)((total + threads - 1) / threads);

    const at::cuda::OptionalCUDAGuard device_guard(device_of(delta));
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(delta.scalar_type(), "geo_stats_backward_cuda", [&]
    {
        geo_stats_backward_kernel<scalar_t><<<blocks, threads, 0, stream>>>(
            grad_var.data_ptr<scalar_t>(),
            grad_mean.data_ptr<scalar_t>(),
            grad_min.data_ptr<scalar_t>(),
            grad_max.data_ptr<scalar_t>(),
            grad_delta.data_ptr<scalar_t>(),
            (int)total);
    });
    AT_CUDA_CHECK(cudaGetLastError());

    return grad_delta;
}

//------------------------------------------------------------------------
