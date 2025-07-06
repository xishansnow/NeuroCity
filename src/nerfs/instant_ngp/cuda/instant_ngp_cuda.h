#pragma once

#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// CUDA kernel function declarations
__global__ void hash_encode_forward_kernel(
    const float* __restrict__ positions,
    const float* __restrict__ embeddings,
    float* __restrict__ encoded,
    const int* __restrict__ resolutions,
    const uint32_t* __restrict__ offsets,
    const int N,
    const int num_levels,
    const int feature_dim,
    const uint32_t hashmap_size,
    const float scale,
    const float3 aabb_min,
    const float3 aabb_max
);

__global__ void hash_encode_backward_kernel(
    const float* __restrict__ positions,
    const float* __restrict__ grad_encoded,
    float* __restrict__ grad_embeddings,
    const int* __restrict__ resolutions,
    const uint32_t* __restrict__ offsets,
    const int N,
    const int num_levels,
    const int feature_dim,
    const uint32_t hashmap_size,
    const float scale,
    const float3 aabb_min,
    const float3 aabb_max
);

__global__ void sh_encode_kernel(
    const float* __restrict__ directions,
    float* __restrict__ encoded,
    const int N,
    const int degree
);

// C++ wrapper functions
void hash_encode_forward_wrapper(
    const float* positions,
    const float* embeddings,
    float* encoded,
    const int* resolutions,
    const uint32_t* offsets,
    const int N,
    const int num_levels,
    const int feature_dim,
    const uint32_t hashmap_size,
    const float scale,
    const float3 aabb_min,
    const float3 aabb_max
);

void hash_encode_backward_wrapper(
    const float* positions,
    const float* grad_encoded,
    float* grad_embeddings,
    const int* resolutions,
    const uint32_t* offsets,
    const int N,
    const int num_levels,
    const int feature_dim,
    const uint32_t hashmap_size,
    const float scale,
    const float3 aabb_min,
    const float3 aabb_max
);

void sh_encode_wrapper(
    const float* directions,
    float* encoded,
    const int N,
    const int degree
);
