#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <cub/cub.cuh>

// Hash encoding utilities
__host__ __device__ inline uint32_t hash_combine(uint32_t seed, uint32_t value) {
    return seed ^ (value + 0x9e3779b9 + (seed << 6) + (seed >> 2));
}

__host__ __device__ inline uint32_t spatial_hash(int x, int y, int z) {
    uint32_t hash = 0;
    hash = hash_combine(hash, (uint32_t)x);
    hash = hash_combine(hash, (uint32_t)y);
    hash = hash_combine(hash, (uint32_t)z);
    return hash;
}

__host__ __device__ inline uint32_t grid_hash(int level, int x, int y, int z, uint32_t hashmap_size) {
    return spatial_hash(x, y, z) % hashmap_size;
}

// Device utilities
__device__ inline float lerp(float a, float b, float t) {
    return a + t * (b - a);
}

__device__ inline float3 make_float3_safe(float x, float y, float z) {
    return make_float3(x, y, z);
}

// Hash encoding forward kernel
__global__ void hash_encode_forward_kernel(
    const float* __restrict__ positions,     // [N, 3]
    const float* __restrict__ embeddings,    // [num_levels, hashmap_size, feature_dim]
    float* __restrict__ encoded,             // [N, num_levels * feature_dim]
    const int* __restrict__ resolutions,     // [num_levels]
    const uint32_t* __restrict__ offsets,    // [num_levels]
    const int N,
    const int num_levels,
    const int feature_dim,
    const uint32_t hashmap_size,
    const float scale,
    const float3 aabb_min,
    const float3 aabb_max
) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;
    
    // Load position
    const float3 pos = make_float3_safe(
        positions[idx * 3],
        positions[idx * 3 + 1],
        positions[idx * 3 + 2]
    );
    
    // Normalize position to [0, 1]
    const float3 pos_normalized = make_float3_safe(
        (pos.x - aabb_min.x) / (aabb_max.x - aabb_min.x),
        (pos.y - aabb_min.y) / (aabb_max.y - aabb_min.y),
        (pos.z - aabb_min.z) / (aabb_max.z - aabb_min.z)
    );
    
    // Clamp to valid range
    const float3 pos_clamped = make_float3_safe(
        fmaxf(0.0f, fminf(1.0f, pos_normalized.x)),
        fmaxf(0.0f, fminf(1.0f, pos_normalized.y)),
        fmaxf(0.0f, fminf(1.0f, pos_normalized.z))
    );
    
    // Process each level
    for (int level = 0; level < num_levels; level++) {
        const int resolution = resolutions[level];
        const uint32_t offset = offsets[level];
        
        // Scale position to grid resolution
        const float3 pos_scaled = make_float3_safe(
            pos_clamped.x * (resolution - 1),
            pos_clamped.y * (resolution - 1),
            pos_clamped.z * (resolution - 1)
        );
        
        // Get grid coordinates
        const int x0 = (int)floorf(pos_scaled.x);
        const int y0 = (int)floorf(pos_scaled.y);
        const int z0 = (int)floorf(pos_scaled.z);
        const int x1 = min(x0 + 1, resolution - 1);
        const int y1 = min(y0 + 1, resolution - 1);
        const int z1 = min(z0 + 1, resolution - 1);
        
        // Interpolation weights
        const float wx = pos_scaled.x - x0;
        const float wy = pos_scaled.y - y0;
        const float wz = pos_scaled.z - z0;
        
        // Trilinear interpolation over 8 corners
        float interpolated[32]; // Max feature_dim
        for (int f = 0; f < feature_dim; f++) {
            interpolated[f] = 0.0f;
        }
        
        // Sample 8 corners and interpolate
        for (int dz = 0; dz <= 1; dz++) {
            for (int dy = 0; dy <= 1; dy++) {
                for (int dx = 0; dx <= 1; dx++) {
                    const int gx = (dx == 0) ? x0 : x1;
                    const int gy = (dy == 0) ? y0 : y1;
                    const int gz = (dz == 0) ? z0 : z1;
                    
                    const uint32_t hash_idx = grid_hash(level, gx, gy, gz, hashmap_size);
                    const uint32_t embedding_offset = offset + hash_idx * feature_dim;
                    
                    const float weight = 
                        (dx == 0 ? (1.0f - wx) : wx) *
                        (dy == 0 ? (1.0f - wy) : wy) *
                        (dz == 0 ? (1.0f - wz) : wz);
                    
                    for (int f = 0; f < feature_dim; f++) {
                        interpolated[f] += weight * embeddings[embedding_offset + f];
                    }
                }
            }
        }
        
        // Store interpolated features
        const int output_offset = idx * (num_levels * feature_dim) + level * feature_dim;
        for (int f = 0; f < feature_dim; f++) {
            encoded[output_offset + f] = interpolated[f];
        }
    }
}

// Hash encoding backward kernel
__global__ void hash_encode_backward_kernel(
    const float* __restrict__ positions,     // [N, 3]
    const float* __restrict__ grad_encoded,  // [N, num_levels * feature_dim]
    float* __restrict__ grad_embeddings,     // [num_levels, hashmap_size, feature_dim]
    const int* __restrict__ resolutions,     // [num_levels]
    const uint32_t* __restrict__ offsets,    // [num_levels]
    const int N,
    const int num_levels,
    const int feature_dim,
    const uint32_t hashmap_size,
    const float scale,
    const float3 aabb_min,
    const float3 aabb_max
) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;
    
    // Load position
    const float3 pos = make_float3_safe(
        positions[idx * 3],
        positions[idx * 3 + 1],
        positions[idx * 3 + 2]
    );
    
    // Normalize position to [0, 1]
    const float3 pos_normalized = make_float3_safe(
        (pos.x - aabb_min.x) / (aabb_max.x - aabb_min.x),
        (pos.y - aabb_min.y) / (aabb_max.y - aabb_min.y),
        (pos.z - aabb_min.z) / (aabb_max.z - aabb_min.z)
    );
    
    // Clamp to valid range
    const float3 pos_clamped = make_float3_safe(
        fmaxf(0.0f, fminf(1.0f, pos_normalized.x)),
        fmaxf(0.0f, fminf(1.0f, pos_normalized.y)),
        fmaxf(0.0f, fminf(1.0f, pos_normalized.z))
    );
    
    // Process each level
    for (int level = 0; level < num_levels; level++) {
        const int resolution = resolutions[level];
        const uint32_t offset = offsets[level];
        
        // Scale position to grid resolution
        const float3 pos_scaled = make_float3_safe(
            pos_clamped.x * (resolution - 1),
            pos_clamped.y * (resolution - 1),
            pos_clamped.z * (resolution - 1)
        );
        
        // Get grid coordinates
        const int x0 = (int)floorf(pos_scaled.x);
        const int y0 = (int)floorf(pos_scaled.y);
        const int z0 = (int)floorf(pos_scaled.z);
        const int x1 = min(x0 + 1, resolution - 1);
        const int y1 = min(y0 + 1, resolution - 1);
        const int z1 = min(z0 + 1, resolution - 1);
        
        // Interpolation weights
        const float wx = pos_scaled.x - x0;
        const float wy = pos_scaled.y - y0;
        const float wz = pos_scaled.z - z0;
        
        // Load gradients for this level
        const int grad_offset = idx * (num_levels * feature_dim) + level * feature_dim;
        
        // Distribute gradients to 8 corners
        for (int dz = 0; dz <= 1; dz++) {
            for (int dy = 0; dy <= 1; dy++) {
                for (int dx = 0; dx <= 1; dx++) {
                    const int gx = (dx == 0) ? x0 : x1;
                    const int gy = (dy == 0) ? y0 : y1;
                    const int gz = (dz == 0) ? z0 : z1;
                    
                    const uint32_t hash_idx = grid_hash(level, gx, gy, gz, hashmap_size);
                    const uint32_t embedding_offset = offset + hash_idx * feature_dim;
                    
                    const float weight = 
                        (dx == 0 ? (1.0f - wx) : wx) *
                        (dy == 0 ? (1.0f - wy) : wy) *
                        (dz == 0 ? (1.0f - wz) : wz);
                    
                    for (int f = 0; f < feature_dim; f++) {
                        atomicAdd(&grad_embeddings[embedding_offset + f], 
                                 weight * grad_encoded[grad_offset + f]);
                    }
                }
            }
        }
    }
}

// Spherical harmonics encoding kernel
__global__ void sh_encode_kernel(
    const float* __restrict__ directions,    // [N, 3]
    float* __restrict__ encoded,             // [N, sh_dim]
    const int N,
    const int degree
) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;
    
    const float x = directions[idx * 3];
    const float y = directions[idx * 3 + 1];
    const float z = directions[idx * 3 + 2];
    
    int output_idx = 0;
    
    // Degree 0
    if (degree >= 0) {
        encoded[idx * ((degree + 1) * (degree + 1)) + output_idx++] = 0.28209479177387814f; // 1/(2*sqrt(pi))
    }
    
    // Degree 1
    if (degree >= 1) {
        encoded[idx * ((degree + 1) * (degree + 1)) + output_idx++] = -0.48860251190291987f * y; // -sqrt(3)*y/(2*sqrt(pi))
        encoded[idx * ((degree + 1) * (degree + 1)) + output_idx++] = 0.48860251190291987f * z;  // sqrt(3)*z/(2*sqrt(pi))
        encoded[idx * ((degree + 1) * (degree + 1)) + output_idx++] = -0.48860251190291987f * x; // -sqrt(3)*x/(2*sqrt(pi))
    }
    
    // Degree 2
    if (degree >= 2) {
        encoded[idx * ((degree + 1) * (degree + 1)) + output_idx++] = 1.0925484305920792f * x * y;  // sqrt(15)*x*y/(2*sqrt(pi))
        encoded[idx * ((degree + 1) * (degree + 1)) + output_idx++] = -1.0925484305920792f * y * z; // -sqrt(15)*y*z/(2*sqrt(pi))
        encoded[idx * ((degree + 1) * (degree + 1)) + output_idx++] = 0.31539156525252005f * (2.0f * z * z - x * x - y * y); // sqrt(5)*(3*z^2-1)/(4*sqrt(pi))
        encoded[idx * ((degree + 1) * (degree + 1)) + output_idx++] = -1.0925484305920792f * x * z; // -sqrt(15)*x*z/(2*sqrt(pi))
        encoded[idx * ((degree + 1) * (degree + 1)) + output_idx++] = 0.5462742152960396f * (x * x - y * y); // sqrt(15)*(x^2-y^2)/(4*sqrt(pi))
    }
    
    // Higher degrees can be added as needed
}

// CUDA function declarations for C++ interface
torch::Tensor hash_encode_forward_cuda(
    torch::Tensor positions,
    torch::Tensor embeddings,
    torch::Tensor resolutions,
    torch::Tensor offsets,
    int num_levels,
    int feature_dim,
    uint32_t hashmap_size,
    float scale,
    torch::Tensor aabb_min,
    torch::Tensor aabb_max
);

torch::Tensor hash_encode_backward_cuda(
    torch::Tensor positions,
    torch::Tensor grad_encoded,
    torch::Tensor embeddings_shape,
    torch::Tensor resolutions,
    torch::Tensor offsets,
    int num_levels,
    int feature_dim,
    uint32_t hashmap_size,
    float scale,
    torch::Tensor aabb_min,
    torch::Tensor aabb_max
);

torch::Tensor sh_encode_cuda(
    torch::Tensor directions,
    int degree
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
) {
    const int threads = 256;
    const int blocks = (N + threads - 1) / threads;
    
    hash_encode_forward_kernel<<<blocks, threads>>>(
        positions, embeddings, encoded, resolutions, offsets,
        N, num_levels, feature_dim, hashmap_size, scale, aabb_min, aabb_max
    );
}

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
) {
    const int threads = 256;
    const int blocks = (N + threads - 1) / threads;
    
    hash_encode_backward_kernel<<<blocks, threads>>>(
        positions, grad_encoded, grad_embeddings, resolutions, offsets,
        N, num_levels, feature_dim, hashmap_size, scale, aabb_min, aabb_max
    );
}

void sh_encode_wrapper(
    const float* directions,
    float* encoded,
    const int N,
    const int degree
) {
    const int threads = 256;
    const int blocks = (N + threads - 1) / threads;
    
    sh_encode_kernel<<<blocks, threads>>>(directions, encoded, N, degree);
}
