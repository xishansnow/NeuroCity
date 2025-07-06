#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// CUDA kernel for trilinear interpolation
__global__ void trilinear_interpolate_kernel(
    const float* __restrict__ features,      // [V, F]
    const float* __restrict__ points,        // [N, 3]
    const int* __restrict__ voxel_indices,   // [N, 8]
    const float* __restrict__ weights,       // [N, 8]
    float* __restrict__ output,              // [N, F]
    const int N,
    const int F
) {
    const int point_idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int feature_idx = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (point_idx >= N || feature_idx >= F) return;
    
    float acc = 0.0f;
    
    // Compute weighted sum of neighboring features
    #pragma unroll
    for (int i = 0; i < 8; i++) {
        const int voxel_idx = voxel_indices[point_idx * 8 + i];
        const float weight = weights[point_idx * 8 + i];
        acc += weight * features[voxel_idx * F + feature_idx];
    }
    
    // Store interpolated feature
    output[point_idx * F + feature_idx] = acc;
}

// CUDA kernel for computing interpolation weights
__global__ void compute_weights_kernel(
    const float* __restrict__ points,        // [N, 3]
    const float* __restrict__ voxel_coords,  // [N, 8, 3]
    float* __restrict__ weights,             // [N, 8]
    const int N
) {
    const int point_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (point_idx >= N) return;
    
    const float3 point = make_float3(
        points[point_idx * 3],
        points[point_idx * 3 + 1],
        points[point_idx * 3 + 2]
    );
    
    // Compute weights for all 8 corners
    #pragma unroll
    for (int i = 0; i < 8; i++) {
        const float3 corner = make_float3(
            voxel_coords[point_idx * 24 + i * 3],
            voxel_coords[point_idx * 24 + i * 3 + 1],
            voxel_coords[point_idx * 24 + i * 3 + 2]
        );
        
        const float dx = 1.0f - fabsf(point.x - corner.x);
        const float dy = 1.0f - fabsf(point.y - corner.y);
        const float dz = 1.0f - fabsf(point.z - corner.z);
        
        weights[point_idx * 8 + i] = dx * dy * dz;
    }
}

// C++ interface
torch::Tensor trilinear_interpolate_cuda(
    torch::Tensor features,
    torch::Tensor points,
    torch::Tensor voxel_indices,
    torch::Tensor weights
) {
    // Ensure input tensors are on CUDA and contiguous
    features = features.to(torch::kCUDA).contiguous();
    points = points.to(torch::kCUDA).contiguous();
    voxel_indices = voxel_indices.to(torch::kCUDA).contiguous();
    weights = weights.to(torch::kCUDA).contiguous();
    
    const int N = points.size(0);      // Number of query points
    const int F = features.size(1);    // Feature dimension
    
    auto output = torch::zeros({N, F}, features.options());
    
    const dim3 threads(16, 16);
    const dim3 blocks(
        (N + threads.x - 1) / threads.x,
        (F + threads.y - 1) / threads.y
    );
    
    trilinear_interpolate_kernel<<<blocks, threads>>>(
        features.data_ptr<float>(),
        points.data_ptr<float>(),
        voxel_indices.data_ptr<int>(),
        weights.data_ptr<float>(),
        output.data_ptr<float>(),
        N,
        F
    );
    
    // Check for CUDA errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA kernel error: %s\n", cudaGetErrorString(err));
    }
    
    return output;
}

torch::Tensor compute_interpolation_weights_cuda(
    torch::Tensor points,
    torch::Tensor voxel_coords
) {
    // Ensure input tensors are on CUDA and contiguous
    points = points.to(torch::kCUDA).contiguous();
    voxel_coords = voxel_coords.to(torch::kCUDA).contiguous();
    
    const int N = points.size(0);
    auto weights = torch::zeros({N, 8}, points.options());
    
    const int threads = 256;
    const int blocks = (N + threads - 1) / threads;
    
    compute_weights_kernel<<<blocks, threads>>>(
        points.data_ptr<float>(),
        voxel_coords.data_ptr<float>(),
        weights.data_ptr<float>(),
        N
    );
    
    // Check for CUDA errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA kernel error: %s\n", cudaGetErrorString(err));
    }
    
    return weights;
} 