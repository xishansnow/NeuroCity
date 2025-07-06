#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// CUDA kernel for ray-voxel intersection
__global__ void ray_voxel_intersect_kernel(
    const float* __restrict__ ray_origins,     // [N, 3]
    const float* __restrict__ ray_directions,  // [N, 3]
    const float* __restrict__ voxel_centers,   // [V, 3]
    const float voxel_size,
    bool* __restrict__ intersections,          // [N, V]
    const int N,
    const int V
) {
    const int ray_idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int voxel_idx = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (ray_idx >= N || voxel_idx >= V) return;
    
    // Load ray origin and direction
    const float3 origin = make_float3(
        ray_origins[ray_idx * 3],
        ray_origins[ray_idx * 3 + 1],
        ray_origins[ray_idx * 3 + 2]
    );
    
    const float3 direction = make_float3(
        ray_directions[ray_idx * 3],
        ray_directions[ray_idx * 3 + 1],
        ray_directions[ray_idx * 3 + 2]
    );
    
    // Load voxel center
    const float3 center = make_float3(
        voxel_centers[voxel_idx * 3],
        voxel_centers[voxel_idx * 3 + 1],
        voxel_centers[voxel_idx * 3 + 2]
    );
    
    // Compute AABB bounds
    const float half_size = voxel_size * 0.5f;
    const float3 aabb_min = make_float3(
        center.x - half_size,
        center.y - half_size,
        center.z - half_size
    );
    const float3 aabb_max = make_float3(
        center.x + half_size,
        center.y + half_size,
        center.z + half_size
    );
    
    // Ray-AABB intersection test
    float3 inv_dir = make_float3(
        1.0f / (direction.x + 1e-8f),
        1.0f / (direction.y + 1e-8f),
        1.0f / (direction.z + 1e-8f)
    );
    
    float3 t_min = make_float3(
        (aabb_min.x - origin.x) * inv_dir.x,
        (aabb_min.y - origin.y) * inv_dir.y,
        (aabb_min.z - origin.z) * inv_dir.z
    );
    
    float3 t_max = make_float3(
        (aabb_max.x - origin.x) * inv_dir.x,
        (aabb_max.y - origin.y) * inv_dir.y,
        (aabb_max.z - origin.z) * inv_dir.z
    );
    
    float t_enter = fmaxf(fmaxf(
        fminf(t_min.x, t_max.x),
        fminf(t_min.y, t_max.y)
    ), fminf(t_min.z, t_max.z));
    
    float t_exit = fminf(fminf(
        fmaxf(t_min.x, t_max.x),
        fmaxf(t_min.y, t_max.y)
    ), fmaxf(t_min.z, t_max.z));
    
    // Write intersection result
    intersections[ray_idx * V + voxel_idx] = (t_enter <= t_exit) && (t_exit > 0);
}

// C++ interface
torch::Tensor ray_voxel_intersect_cuda(
    torch::Tensor ray_origins,
    torch::Tensor ray_directions,
    torch::Tensor voxel_centers,
    float voxel_size
) {
    // Ensure input tensors are on CUDA and contiguous
    ray_origins = ray_origins.to(torch::kCUDA).contiguous();
    ray_directions = ray_directions.to(torch::kCUDA).contiguous();
    voxel_centers = voxel_centers.to(torch::kCUDA).contiguous();
    
    const int N = ray_origins.size(0);
    const int V = voxel_centers.size(0);
    
    auto intersections = torch::zeros({N, V}, torch::TensorOptions().dtype(torch::kBool).device(torch::kCUDA));
    
    const dim3 threads(16, 16);
    const dim3 blocks(
        (N + threads.x - 1) / threads.x,
        (V + threads.y - 1) / threads.y
    );
    
    ray_voxel_intersect_kernel<<<blocks, threads>>>(
        ray_origins.data_ptr<float>(),
        ray_directions.data_ptr<float>(),
        voxel_centers.data_ptr<float>(),
        voxel_size,
        intersections.data_ptr<bool>(),
        N,
        V
    );
    
    // Check for CUDA errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA kernel error: %s\n", cudaGetErrorString(err));
    }
    
    return intersections;
} 