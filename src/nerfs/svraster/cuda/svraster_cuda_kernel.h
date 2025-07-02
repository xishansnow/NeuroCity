#ifndef SVRASTER_CUDA_KERNEL_H
#define SVRASTER_CUDA_KERNEL_H

#include <cuda_runtime.h>
#include <torch/extension.h>

// Core data structures
struct Voxel {
    float3 position;     // Voxel center position
    float size;          // Voxel size
    float density;       // Density value
    float3 color;        // RGB color
    int level;           // Octree level
    int morton_code;     // Morton code for sorting
};

struct Ray {
    float3 origin;       // Ray origin
    float3 direction;    // Ray direction (normalized)
};

// CUDA kernel declarations
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
);

__global__ void compute_morton_codes_kernel(
    Voxel* voxels,
    const float3 scene_min,
    const float3 scene_size,
    int num_voxels
);

__global__ void sort_voxels_by_ray_direction_kernel(
    Voxel* voxels,
    const float3 ray_direction,
    int num_voxels
);

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
);

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
);

__global__ void adaptive_subdivision_kernel(
    Voxel* voxels,
    const float* voxel_gradients,
    int* subdivision_flags,
    int* new_voxel_count,
    float subdivision_threshold,
    int max_level,
    int num_voxels
);

__global__ void subdivide_voxels_kernel(
    const Voxel* old_voxels,
    Voxel* new_voxels,
    const int* subdivision_flags,
    int* voxel_mapping,
    int num_old_voxels,
    int num_new_voxels
);

__global__ void voxel_pruning_kernel(
    const Voxel* voxels,
    int* keep_flags,
    float pruning_threshold,
    int num_voxels
);

__global__ void initialize_voxel_grid_kernel(
    Voxel* voxels,
    int base_resolution,
    int num_voxels
);

__global__ void update_voxel_parameters_kernel(
    Voxel* voxels,
    const float* gradients,
    float learning_rate,
    int num_voxels
);

__global__ void radix_sort_kernel(
    Voxel* voxels,
    Voxel* temp_voxels,
    int* histograms,
    int num_voxels,
    int bit_offset
);

__global__ void copy_back_kernel(
    Voxel* voxels,
    const Voxel* temp_voxels,
    int num_voxels
);

// Device helper functions
__device__ bool ray_aabb_intersection(
    const Ray& ray,
    const Voxel& voxel,
    float& t_near,
    float& t_far
);

__device__ int morton_encode_3d(int x, int y, int z);
__device__ int compute_morton_code(const Voxel& voxel, const float3& scene_min, const float3& scene_size);
__device__ float3 alpha_compositing(
    const Voxel* voxels,
    const int* intersection_indices,
    const float* intersection_t_near,
    const float* intersection_t_far,
    int intersection_count,
    const Ray& ray,
    float3 background_color
);

// Utility functions
void radix_sort_voxels(Voxel* voxels, int num_voxels);

#endif // SVRASTER_CUDA_KERNEL_H 