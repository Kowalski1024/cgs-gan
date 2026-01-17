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
#include <cuda_fp16.h>
#include "neighbor_max.h"

//------------------------------------------------------------------------

template <typename scalar_t>
__device__ __forceinline__ void atomic_add(scalar_t* addr, scalar_t val)
{
    atomicAdd(addr, val);
}

__device__ __forceinline__ void atomic_add_half(at::Half* address, at::Half val)
{
    atomicAdd(reinterpret_cast<__half*>(address), static_cast<__half>(val));
}

template <>
__device__ __forceinline__ void atomic_add<at::Half>(at::Half* addr, at::Half val)
{
    atomic_add_half(addr, val);
}

//------------------------------------------------------------------------

template <typename scalar_t>
__global__ void neighbor_max_forward_kernel(
    const scalar_t* __restrict__ x,
    const int64_t* __restrict__ edge,
    scalar_t* __restrict__ y,
    int64_t* __restrict__ idx,
    int B, int N, int C, int K)
{
    int64_t index = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    int64_t total = (int64_t)B * N * C;
    if (index >= total)
        return;

    int64_t tmp = index;
    int64_t b = tmp / (N * C);
    tmp -= b * (N * C);
    int64_t n = tmp / C;
    int64_t c = tmp - n * C;

    int64_t edge_base = n * K;
    int64_t best_k = 0;
    int64_t neighbor = edge[edge_base];
    scalar_t best = x[(b * N + neighbor) * C + c];

    for (int64_t k = 1; k < K; ++k)
    {
        int64_t nb = edge[edge_base + k];
        scalar_t v = x[(b * N + nb) * C + c];
        if (v > best)
        {
            best = v;
            best_k = k;
        }
    }

    y[index] = best;
    idx[index] = best_k;
}

//------------------------------------------------------------------------

template <typename scalar_t>
__global__ void neighbor_max_backward_kernel(
    const scalar_t* __restrict__ grad_y,
    const int64_t* __restrict__ edge,
    const int64_t* __restrict__ idx,
    scalar_t* __restrict__ grad_x,
    int B, int N, int C, int K)
{
    int64_t index = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    int64_t total = (int64_t)B * N * C;
    if (index >= total)
        return;

    int64_t tmp = index;
    int64_t b = tmp / (N * C);
    tmp -= b * (N * C);
    int64_t n = tmp / C;
    int64_t c = tmp - n * C;

    int64_t k = idx[index];
    int64_t nb = edge[n * K + k];
    int64_t gx_index = (b * N + nb) * C + c;
    atomic_add<scalar_t>(&grad_x[gx_index], grad_y[index]);
}

//------------------------------------------------------------------------

std::vector<torch::Tensor> neighbor_max_forward_cuda(torch::Tensor x, torch::Tensor edge_index)
{
    const int B = (int)x.size(0);
    const int N = (int)x.size(1);
    const int C = (int)x.size(2);
    const int K = (int)edge_index.size(1);

    auto y = torch::empty({B, N, C}, x.options());
    auto idx = torch::empty({B, N, C}, edge_index.options().dtype(torch::kInt64));

    const int threads = 256;
    const int64_t total = (int64_t)B * N * C;
    const int blocks = (int)((total + threads - 1) / threads);

    const at::cuda::OptionalCUDAGuard device_guard(device_of(x));
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(x.scalar_type(), "neighbor_max_forward_cuda", [&]
    {
        neighbor_max_forward_kernel<scalar_t><<<blocks, threads, 0, stream>>>(
            x.data_ptr<scalar_t>(),
            edge_index.data_ptr<int64_t>(),
            y.data_ptr<scalar_t>(),
            idx.data_ptr<int64_t>(),
            B, N, C, K);
    });
    AT_CUDA_CHECK(cudaGetLastError());

    return {y, idx};
}

//------------------------------------------------------------------------

torch::Tensor neighbor_max_backward_cuda(torch::Tensor grad_y, torch::Tensor edge_index, torch::Tensor indices)
{
    const int B = (int)grad_y.size(0);
    const int N = (int)grad_y.size(1);
    const int C = (int)grad_y.size(2);
    const int K = (int)edge_index.size(1);

    auto grad_x = torch::zeros({B, N, C}, grad_y.options());

    const int threads = 256;
    const int64_t total = (int64_t)B * N * C;
    const int blocks = (int)((total + threads - 1) / threads);

    const at::cuda::OptionalCUDAGuard device_guard(device_of(grad_y));
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(grad_y.scalar_type(), "neighbor_max_backward_cuda", [&]
    {
        neighbor_max_backward_kernel<scalar_t><<<blocks, threads, 0, stream>>>(
            grad_y.data_ptr<scalar_t>(),
            edge_index.data_ptr<int64_t>(),
            indices.data_ptr<int64_t>(),
            grad_x.data_ptr<scalar_t>(),
            B, N, C, K);
    });
    AT_CUDA_CHECK(cudaGetLastError());

    return grad_x;
}

//------------------------------------------------------------------------
