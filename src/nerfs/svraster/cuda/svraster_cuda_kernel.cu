#include "svraster_cuda_kernel.h"
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <math.h>
#include <cuda_fp16.h>
#include <cub/cub.cuh>

__device__ __forceinline__ float3 make_float3_from_scalar(float s) {
    return make_float3(s, s, s);
}

__device__ __forceinline__ float3 operator+(const float3& a, const float3& b) {
    return make_float3(a.x + b.x, a.y + b.y, a.z + b.z);
}

__device__ __forceinline__ float3 operator-(const float3& a, const float3& b) {
    return make_float3(a.x - b.x, a.y - b.y, a.z - b.z);
}

__device__ __forceinline__ float3 operator*(const float3& a, float s) {
    return make_float3(a.x * s, a.y * s, a.z * s);
}

__device__ __forceinline__ float3 operator*(float s, const float3& a) {
    return make_float3(s * a.x, s * a.y, s * a.z);
}

__device__ __forceinline__ float3 operator/(const float3& a, const float3& b) {
    return make_float3(a.x / b.x, a.y / b.y, a.z / b.z);
}

__device__ __forceinline__ float3 fminf(const float3& a, const float3& b) {
    return make_float3(fminf(a.x, b.x), fminf(a.y, b.y), fminf(a.z, b.z));
}

__device__ __forceinline__ float3 fmaxf(const float3& a, const float3& b) {
    return make_float3(fmaxf(a.x, b.x), fmaxf(a.y, b.y), fmaxf(a.z, b.z));
}

__device__ __forceinline__ float dot(const float3& a, const float3& b) {
    return a.x * b.x + a.y * b.y + a.z * b.z;
}

__device__ __forceinline__ float length(const float3& v) {
    return sqrtf(v.x * v.x + v.y * v.y + v.z * v.z);
}

__device__ __forceinline__ void operator+=(float3& a, const float3& b) {
    a.x += b.x; a.y += b.y; a.z += b.z;
}

// Device helper functions
__device__ bool ray_aabb_intersection(
    const Ray& ray,
    const Voxel& voxel,
    float& t_near,
    float& t_far
) {
    float3 box_min = voxel.position - make_float3(voxel.size * 0.5f, voxel.size * 0.5f, voxel.size * 0.5f);
    float3 box_max = voxel.position + make_float3(voxel.size * 0.5f, voxel.size * 0.5f, voxel.size * 0.5f);
    
    float3 t_min = (box_min - ray.origin) / ray.direction;
    float3 t_max = (box_max - ray.origin) / ray.direction;
    
    float3 t1 = fminf(t_min, t_max);
    float3 t2 = fmaxf(t_min, t_max);
    
    t_near = fmaxf(fmaxf(t1.x, t1.y), t1.z);
    t_far = fminf(fminf(t2.x, t2.y), t2.z);
    
    return t_far > t_near && t_far > 0.0f;
}

__device__ int morton_encode_3d(int x, int y, int z) {
    // Simple Morton encoding for 3D coordinates
    // Using 10 bits per coordinate (max 1024³ resolution)
    
    // Ensure input in valid range
    x = max(0, min(x, 1023));
    y = max(0, min(y, 1023));
    z = max(0, min(z, 1023));
    
    // Simple bit interleaving
    int morton = 0;
    for (int i = 0; i < 10; i++) {
        morton |= ((x & (1 << i)) << (2 * i)) |
                  ((y & (1 << i)) << (2 * i + 1)) |
                  ((z & (1 << i)) << (2 * i + 2));
    }
    
    return morton;
}

__device__ int compute_morton_code(const Voxel& voxel, const float3& scene_min, const float3& scene_size) {
    // Normalize voxel position to [0, 1]
    float3 normalized;
    normalized.x = (voxel.position.x - scene_min.x) / scene_size.x;
    normalized.y = (voxel.position.y - scene_min.y) / scene_size.y;
    normalized.z = (voxel.position.z - scene_min.z) / scene_size.z;
    
    // Convert to integer grid coordinates (10-bit precision)
    int x = (int)(normalized.x * 1023.0f);
    int y = (int)(normalized.y * 1023.0f);
    int z = (int)(normalized.z * 1023.0f);
    
    return morton_encode_3d(x, y, z);
}

__device__ float3 alpha_compositing(
    const Voxel* voxels,
    const int* intersection_indices,
    const float* intersection_t_near,
    const float* intersection_t_far,
    int intersection_count,
    const Ray& ray,
    float3 background_color
) {
    float3 rgb_acc = make_float3(0.0f, 0.0f, 0.0f);
    float transmittance = 1.0f;
    
    for (int i = 0; i < intersection_count && transmittance > 0.01f; i++) {
        int voxel_idx = intersection_indices[i];
        Voxel voxel = voxels[voxel_idx];
        
        float t_near = intersection_t_near[i];
        float t_far = intersection_t_far[i];
        
        // Compute opacity for this voxel
        float delta_t = t_far - t_near;
        float opacity = 1.0f - expf(-expf(voxel.density) * delta_t);
        
        // Alpha compositing
        float weight = opacity * transmittance;
        rgb_acc += weight * voxel.color;
        transmittance *= (1.0f - opacity);
    }
    
    // Add background contribution
    rgb_acc += transmittance * background_color;
    
    return rgb_acc;
}

// CUDA Kernels
__global__ void ray_voxel_intersection_kernel(
    const Ray* rays,
    const Voxel* voxels,
    int* intersection_counts,
    int* intersection_indices,
    float* intersection_t_near,
    float* intersection_t_far,
    int num_rays,
    int num_voxels,
    int max_intersections_per_ray
) {
    int ray_idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (ray_idx >= num_rays) return;
    
    Ray ray = rays[ray_idx];
    int intersection_count = 0;
    
    // Each thread processes one ray against all voxels
    for (int voxel_idx = 0; voxel_idx < num_voxels && intersection_count < max_intersections_per_ray; voxel_idx++) {
        Voxel voxel = voxels[voxel_idx];
        float t_near, t_far;
        
        if (ray_aabb_intersection(ray, voxel, t_near, t_far)) {
            int global_idx = ray_idx * max_intersections_per_ray + intersection_count;
            intersection_indices[global_idx] = voxel_idx;
            intersection_t_near[global_idx] = t_near;
            intersection_t_far[global_idx] = t_far;
            intersection_count++;
        }
    }
    
    intersection_counts[ray_idx] = intersection_count;
}

__global__ void compute_morton_codes_kernel(
    Voxel* voxels,
    const float3 scene_min,
    const float3 scene_size,
    int num_voxels
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx >= num_voxels) return;
    
    voxels[idx].morton_code = compute_morton_code(voxels[idx], scene_min, scene_size);
}

__global__ void sort_voxels_by_ray_direction_kernel(
    Voxel* voxels,
    const float3 ray_direction,
    int num_voxels
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx >= num_voxels) return;
    
    // Project voxel center onto ray direction
    float proj_dist = dot(voxels[idx].position, ray_direction);
    
    // Store projection distance for sorting
    voxels[idx].morton_code = __float_as_int(proj_dist);  // Reuse morton_code field
}

__global__ void voxel_rasterization_kernel(
    const Ray* rays,
    const Voxel* voxels,
    const int* intersection_counts,
    const int* intersection_indices,
    const float* intersection_t_near,
    const float* intersection_t_far,
    float3* output_colors,
    float* output_depths,
    int num_rays,
    int max_intersections_per_ray,
    float3 background_color
) {
    int ray_idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (ray_idx >= num_rays) return;
    
    Ray ray = rays[ray_idx];
    int intersection_count = intersection_counts[ray_idx];
    
    // Get intersection data for this ray
    const int* ray_intersections = intersection_indices + ray_idx * max_intersections_per_ray;
    const float* ray_t_near = intersection_t_near + ray_idx * max_intersections_per_ray;
    const float* ray_t_far = intersection_t_far + ray_idx * max_intersections_per_ray;
    
    // Perform alpha compositing
    float3 color = alpha_compositing(
        voxels, ray_intersections, ray_t_near, ray_t_far,
        intersection_count, ray, background_color
    );
    
    // Compute depth (simplified)
    float depth = (intersection_count > 0) ? ray_t_near[0] : 1000.0f;
    
    output_colors[ray_idx] = color;
    output_depths[ray_idx] = depth;
}

__global__ void compute_voxel_gradients_kernel(
    const Voxel* voxels,
    const float3* ray_origins,
    const float3* ray_directions,
    const float3* target_colors,
    float* voxel_gradients,
    int num_voxels,
    int num_rays,
    int* ray_voxel_mapping,
    int max_rays_per_voxel
) {
    int voxel_idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (voxel_idx >= num_voxels) return;
    
    float total_gradient = 0.0f;
    int ray_count = 0;
    
    // Compute gradient for this voxel based on all rays that intersect it
    for (int ray_idx = 0; ray_idx < num_rays && ray_count < max_rays_per_voxel; ray_idx++) {
        if (ray_voxel_mapping[ray_idx * num_voxels + voxel_idx]) {
            // This ray intersects this voxel
            float3 ray_o = ray_origins[ray_idx];
            float3 ray_d = ray_directions[ray_idx];
            float3 target = target_colors[ray_idx];
            
            // Compute contribution of this voxel to this ray
            Voxel voxel = voxels[voxel_idx];
            float3 voxel_center = voxel.position;
            
            // Simplified gradient computation
            float distance = length(ray_o - voxel_center);
            float contribution = expf(-distance * voxel.density);
            
            // Gradient based on color difference
            float3 color_diff = target - voxel.color;
            float gradient = length(color_diff) * contribution;
            
            total_gradient += gradient;
            ray_count++;
        }
    }
    
    voxel_gradients[voxel_idx] = (ray_count > 0) ? total_gradient / ray_count : 0.0f;
}

__global__ void adaptive_subdivision_kernel(
    Voxel* voxels,
    const float* voxel_gradients,
    int* subdivision_flags,
    int* new_voxel_count,
    float subdivision_threshold,
    int max_level,
    int num_voxels
) {
    int voxel_idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (voxel_idx >= num_voxels) return;
    
    Voxel voxel = voxels[voxel_idx];
    float gradient = voxel_gradients[voxel_idx];
    
    // Determine if voxel should be subdivided
    bool should_subdivide = (gradient > subdivision_threshold && 
                           voxel.level < max_level &&
                           expf(voxel.density) > 0.01f);
    
    subdivision_flags[voxel_idx] = should_subdivide ? 1 : 0;
    
    if (should_subdivide) {
        // Count new voxels (8 children per subdivided voxel)
        atomicAdd(new_voxel_count, 8);
    }
}

__global__ void subdivide_voxels_kernel(
    const Voxel* old_voxels,
    Voxel* new_voxels,
    const int* subdivision_flags,
    int* voxel_mapping,
    int num_old_voxels,
    int num_new_voxels
) {
    int voxel_idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (voxel_idx >= num_old_voxels) return;
    
    if (subdivision_flags[voxel_idx]) {
        Voxel parent_voxel = old_voxels[voxel_idx];
        float new_size = parent_voxel.size * 0.5f;
        float3 parent_pos = parent_voxel.position;
        
        // Create 8 child voxels
        for (int child = 0; child < 8; child++) {
            int new_idx = voxel_mapping[voxel_idx] + child;
            
            // Compute child position
            float3 offset = make_float3(
                (child & 1) ? new_size * 0.5f : -new_size * 0.5f,
                (child & 2) ? new_size * 0.5f : -new_size * 0.5f,
                (child & 4) ? new_size * 0.5f : -new_size * 0.5f
            );
            
            Voxel child_voxel;
            child_voxel.position = parent_pos + offset;
            child_voxel.size = new_size;
            child_voxel.density = parent_voxel.density * 0.5f;  // Inherit density
            child_voxel.color = parent_voxel.color;  // Inherit color
            child_voxel.level = parent_voxel.level + 1;
            child_voxel.morton_code = 0;  // Will be computed later
            
            new_voxels[new_idx] = child_voxel;
        }
    }
}

__global__ void voxel_pruning_kernel(
    const Voxel* voxels,
    int* keep_flags,
    float pruning_threshold,
    int num_voxels
) {
    int voxel_idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (voxel_idx >= num_voxels) return;
    
    Voxel voxel = voxels[voxel_idx];
    float density = expf(voxel.density);
    
    // Keep voxel if density is above threshold
    keep_flags[voxel_idx] = (density > pruning_threshold) ? 1 : 0;
}

__global__ void initialize_voxel_grid_kernel(
    Voxel* voxels,
    int base_resolution,
    int num_voxels
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx >= num_voxels) return;
    
    // Convert linear index to 3D grid coordinates
    int x = idx % base_resolution;
    int y = (idx / base_resolution) % base_resolution;
    int z = idx / (base_resolution * base_resolution);
    
    // Convert to world coordinates
    float voxel_size = 2.0f / base_resolution;  // Assuming scene bounds [-1, 1]
    float3 position = make_float3(
        (x + 0.5f) * voxel_size - 1.0f,
        (y + 0.5f) * voxel_size - 1.0f,
        (z + 0.5f) * voxel_size - 1.0f
    );
    
    // Initialize voxel
    voxels[idx].position = position;
    voxels[idx].size = voxel_size;
    voxels[idx].density = 0.0f;  // Start with zero density
    voxels[idx].color = make_float3(0.5f, 0.5f, 0.5f);  // Gray color
    voxels[idx].level = 0;
    voxels[idx].morton_code = 0;
}

__global__ void update_voxel_parameters_kernel(
    Voxel* voxels,
    const float* gradients,
    float learning_rate,
    int num_voxels
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx >= num_voxels) return;
    
    // Update density based on gradient
    voxels[idx].density += gradients[idx] * learning_rate;
    
    // Clamp density to reasonable range
    voxels[idx].density = fmaxf(-10.0f, fminf(10.0f, voxels[idx].density));
}

__global__ void radix_sort_kernel(
    Voxel* voxels,
    Voxel* temp_voxels,
    int* histograms,
    int num_voxels,
    int bit_offset
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx >= num_voxels) return;
    
    // Compute histogram for current bit
    int bit = (voxels[idx].morton_code >> bit_offset) & 1;
    atomicAdd(&histograms[bit], 1);
    
    __syncthreads();
    
    // Compute prefix sum
    int position;
    if (bit == 0) {
        position = idx - histograms[1];
    } else {
        position = num_voxels - histograms[0] + (idx - histograms[1]);
    }
    
    // Copy to temporary array
    temp_voxels[position] = voxels[idx];
}

__global__ void copy_back_kernel(
    Voxel* voxels,
    const Voxel* temp_voxels,
    int num_voxels
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx >= num_voxels) return;
    
    voxels[idx] = temp_voxels[idx];
}

// Kernel launch wrapper functions
void launch_ray_voxel_intersection_kernel(
    int grid_size, int block_size,
    const Ray* rays,
    const Voxel* voxels,
    int* intersection_counts,
    int* intersection_indices,
    float* intersection_t_near,
    float* intersection_t_far,
    int num_rays,
    int num_voxels,
    int max_intersections_per_ray
) {
    ray_voxel_intersection_kernel<<<grid_size, block_size>>>(
        rays, voxels, intersection_counts, intersection_indices,
        intersection_t_near, intersection_t_far, num_rays, num_voxels,
        max_intersections_per_ray
    );
}

void launch_voxel_rasterization_kernel(
    int grid_size, int block_size,
    const Ray* rays,
    const Voxel* voxels,
    const int* intersection_counts,
    const int* intersection_indices,
    const float* intersection_t_near,
    const float* intersection_t_far,
    float3* output_colors,
    float* output_depths,
    int num_rays,
    int max_intersections_per_ray,
    float3 background_color
) {
    voxel_rasterization_kernel<<<grid_size, block_size>>>(
        rays, voxels, intersection_counts, intersection_indices,
        intersection_t_near, intersection_t_far, output_colors, output_depths,
        num_rays, max_intersections_per_ray, background_color
    );
}

void launch_compute_morton_codes_kernel(
    int grid_size, int block_size,
    Voxel* voxels,
    const float3 scene_min,
    const float3 scene_size,
    int num_voxels
) {
    compute_morton_codes_kernel<<<grid_size, block_size>>>(
        voxels, scene_min, scene_size, num_voxels
    );
}

void launch_adaptive_subdivision_kernel(
    int grid_size, int block_size,
    Voxel* voxels,
    const float* voxel_gradients,
    int* subdivision_flags,
    int* new_voxel_count,
    float subdivision_threshold,
    int max_level,
    int num_voxels
) {
    adaptive_subdivision_kernel<<<grid_size, block_size>>>(
        voxels, voxel_gradients, subdivision_flags, new_voxel_count,
        subdivision_threshold, max_level, num_voxels
    );
}
