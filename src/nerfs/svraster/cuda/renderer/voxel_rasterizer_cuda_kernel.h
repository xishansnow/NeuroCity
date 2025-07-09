#ifndef VOXEL_RASTERIZER_CUDA_KERNEL_H
#define VOXEL_RASTERIZER_CUDA_KERNEL_H

#include <cuda_runtime.h>
#include <torch/extension.h>

// Matrix type definitions
struct float3x3
{
    float m[3][3];
};

struct float4x4
{
    float m[4][4];
};

// Voxel data structure for rasterization
struct RasterVoxel
{
    float3 position; // Voxel center position in world space
    float3 size;     // Voxel size (can be anisotropic)
    float density;   // Density value
    float3 color;    // RGB color
    int voxel_idx;   // Original voxel index
};

// Screen space voxel data
struct ScreenVoxel
{
    float2 screen_pos; // Screen space position
    float depth;       // Depth in camera space
    float screen_size; // Projected size on screen
    float density;     // Density value
    float3 color;      // RGB color
    float3 world_pos;  // Original world position
    float3 world_size; // Original world size
    int voxel_idx;     // Original voxel index
};

// Camera parameters
struct CameraParams
{
    float4x4 camera_matrix;  // World to camera transformation
    float3x3 intrinsics;     // Camera intrinsics matrix
    float2 viewport_size;    // Viewport width and height
    float near_plane;        // Near clipping plane
    float far_plane;         // Far clipping plane
    float3 background_color; // Background color
};

// Framebuffer structure
struct Framebuffer
{
    float3 *color_buffer; // Color buffer [H, W, 3]
    float *depth_buffer;  // Depth buffer [H, W]
    float *alpha_buffer;  // Alpha buffer [H, W]
    int width;
    int height;
};

// CUDA kernel declarations for voxel rasterization pipeline

// 1. Projection kernel - project voxels to screen space
__global__ void project_voxels_to_screen_kernel(
    const RasterVoxel *voxels,
    ScreenVoxel *screen_voxels,
    const CameraParams camera_params,
    int num_voxels);

// 2. Frustum culling kernel - remove voxels outside view frustum
__global__ void frustum_culling_kernel(
    const ScreenVoxel *screen_voxels,
    ScreenVoxel *visible_voxels,
    int *visible_count,
    const CameraParams camera_params,
    int num_voxels);

// 3. Depth sorting kernel - sort voxels by depth (back-to-front)
__global__ void depth_sort_kernel(
    ScreenVoxel *voxels,
    int num_voxels);

// 4. Main rasterization kernel - rasterize voxels to pixels
__global__ void rasterize_voxels_kernel(
    const ScreenVoxel *sorted_voxels,
    Framebuffer framebuffer,
    const CameraParams camera_params,
    int num_voxels);

// 5. Single voxel rasterization kernel
__global__ void rasterize_single_voxel_kernel(
    const ScreenVoxel *voxel,
    Framebuffer framebuffer,
    const CameraParams camera_params);

// 6. Alpha blending kernel for transparent voxels
__global__ void alpha_blending_kernel(
    const float3 *new_color,
    const float *new_alpha,
    const float *new_depth,
    Framebuffer framebuffer,
    int width,
    int height);

// 7. Spherical harmonics evaluation kernel
__global__ void eval_spherical_harmonics_kernel(
    const float *sh_coeffs,
    const float3 *view_directions,
    float3 *colors,
    int sh_degree,
    int num_voxels);

// Device helper functions
__device__ float3 project_point_to_screen(
    const float3 &world_pos,
    const CameraParams &camera_params);

__device__ float compute_screen_size(
    const float3 &world_size,
    float depth,
    const CameraParams &camera_params);

__device__ bool is_voxel_visible(
    const ScreenVoxel &voxel,
    const CameraParams &camera_params);

__device__ float3 compute_voxel_color_sh(
    const float *sh_coeffs,
    const float3 &view_dir,
    int sh_degree);

__device__ float compute_voxel_alpha(
    float density,
    const float3 &world_size,
    const char *density_activation);

__device__ float3 apply_color_activation(
    const float3 &color,
    const char *color_activation);

// Kernel launch wrapper functions
void launch_project_voxels_kernel(
    int grid_size, int block_size,
    const RasterVoxel *voxels,
    ScreenVoxel *screen_voxels,
    const CameraParams &camera_params,
    int num_voxels);

void launch_frustum_culling_kernel(
    int grid_size, int block_size,
    const ScreenVoxel *screen_voxels,
    ScreenVoxel *visible_voxels,
    int *visible_count,
    const CameraParams &camera_params,
    int num_voxels);

void launch_depth_sort_kernel(
    int grid_size, int block_size,
    ScreenVoxel *voxels,
    int num_voxels);

void launch_rasterize_voxels_kernel(
    int grid_size, int block_size,
    const ScreenVoxel *sorted_voxels,
    Framebuffer &framebuffer,
    const CameraParams &camera_params,
    int num_voxels);

void launch_alpha_blending_kernel(
    int grid_size, int block_size,
    const float3 *new_color,
    const float *new_alpha,
    const float *new_depth,
    Framebuffer &framebuffer,
    int width,
    int height);

// Utility functions for memory management
void allocate_framebuffer(Framebuffer &framebuffer, int width, int height);
void free_framebuffer(Framebuffer &framebuffer);

// Radix sort for depth sorting (if needed)
void radix_sort_screen_voxels(ScreenVoxel *voxels, int num_voxels);

#endif // VOXEL_RASTERIZER_CUDA_KERNEL_H