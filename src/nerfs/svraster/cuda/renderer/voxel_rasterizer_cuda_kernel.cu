#include "voxel_rasterizer_cuda_kernel.h"
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <math.h>
#include <cuda_fp16.h>
#include <cub/cub.cuh>

// CUDA math utilities
__device__ __forceinline__ float3 make_float3_from_scalar(float s)
{
  return make_float3(s, s, s);
}

__device__ __forceinline__ float3 operator+(const float3 &a, const float3 &b)
{
  return make_float3(a.x + b.x, a.y + b.y, a.z + b.z);
}

__device__ __forceinline__ float3 operator-(const float3 &a, const float3 &b)
{
  return make_float3(a.x - b.x, a.y - b.y, a.z - b.z);
}

__device__ __forceinline__ float3 operator*(const float3 &a, float s)
{
  return make_float3(a.x * s, a.y * s, a.z * s);
}

__device__ __forceinline__ float3 operator*(float s, const float3 &a)
{
  return make_float3(s * a.x, s * a.y, s * a.z);
}

__device__ __forceinline__ float dot(const float3 &a, const float3 &b)
{
  return a.x * b.x + a.y * b.y + a.z * b.z;
}

__device__ __forceinline__ float length(const float3 &v)
{
  return sqrtf(v.x * v.x + v.y * v.y + v.z * v.z);
}

__device__ __forceinline__ float3 normalize(const float3 &v)
{
  float len = length(v);
  if (len > 1e-6f)
  {
    return make_float3(v.x / len, v.y / len, v.z / len);
  }
  else
  {
    return make_float3(0.0f, 0.0f, 1.0f);
  }
}

// Matrix-vector operations for camera transformations
__device__ __forceinline__ float3 transform_point(const float4x4 &matrix, const float3 &point)
{
  float4 homogeneous = make_float4(point.x, point.y, point.z, 1.0f);
  float4 result;

  result.x = matrix.m[0][0] * homogeneous.x + matrix.m[0][1] * homogeneous.y +
             matrix.m[0][2] * homogeneous.z + matrix.m[0][3] * homogeneous.w;
  result.y = matrix.m[1][0] * homogeneous.x + matrix.m[1][1] * homogeneous.y +
             matrix.m[1][2] * homogeneous.z + matrix.m[1][3] * homogeneous.w;
  result.z = matrix.m[2][0] * homogeneous.x + matrix.m[2][1] * homogeneous.y +
             matrix.m[2][2] * homogeneous.z + matrix.m[2][3] * homogeneous.w;
  result.w = matrix.m[3][0] * homogeneous.x + matrix.m[3][1] * homogeneous.y +
             matrix.m[3][2] * homogeneous.z + matrix.m[3][3] * homogeneous.w;

  return make_float3(result.x / result.w, result.y / result.w, result.z / result.w);
}

__device__ __forceinline__ float3 transform_vector(const float3x3 &matrix, const float3 &vector)
{
  return make_float3(
      matrix.m[0][0] * vector.x + matrix.m[0][1] * vector.y + matrix.m[0][2] * vector.z,
      matrix.m[1][0] * vector.x + matrix.m[1][1] * vector.y + matrix.m[1][2] * vector.z,
      matrix.m[2][0] * vector.x + matrix.m[2][1] * vector.y + matrix.m[2][2] * vector.z);
}

// Device helper functions implementation
__device__ float3 project_point_to_screen(
    const float3 &world_pos,
    const CameraParams &camera_params)
{
  // Transform to camera space
  float3 camera_pos = transform_point(camera_params.camera_matrix, world_pos);

  // Project to screen space using intrinsics
  float3 screen_pos = transform_vector(camera_params.intrinsics, camera_pos);
  screen_pos.x /= screen_pos.z;
  screen_pos.y /= screen_pos.z;

  return make_float3(screen_pos.x, screen_pos.y, camera_pos.z);
}

__device__ float compute_screen_size(
    const float3 &world_size,
    float depth,
    const CameraParams &camera_params)
{
  // Simplified perspective projection for screen size
  float focal_length = camera_params.intrinsics.m[0][0];
  float screen_size = world_size.x * focal_length / fmaxf(depth, 0.1f);
  return screen_size;
}

__device__ bool is_voxel_visible(
    const ScreenVoxel &voxel,
    const CameraParams &camera_params)
{
  // Depth test
  if (voxel.depth <= camera_params.near_plane || voxel.depth >= camera_params.far_plane)
  {
    return false;
  }

  // Screen bounds test (considering voxel size)
  float half_size = voxel.screen_size * 0.5f;
  if (voxel.screen_pos.x + half_size < 0 || voxel.screen_pos.x - half_size >= camera_params.viewport_size.x ||
      voxel.screen_pos.y + half_size < 0 || voxel.screen_pos.y - half_size >= camera_params.viewport_size.y)
  {
    return false;
  }

  return true;
}

__device__ float3 compute_voxel_color_sh(
    const float *sh_coeffs,
    const float3 &view_dir,
    int sh_degree)
{
  // Simplified spherical harmonics evaluation
  // For now, we'll use a basic implementation
  // In practice, you'd want to use a more efficient SH evaluation

  float3 color = make_float3(0.0f, 0.0f, 0.0f);

  if (sh_degree == 0)
  {
    // Constant term
    color.x = sh_coeffs[0];
    color.y = sh_coeffs[1];
    color.z = sh_coeffs[2];
  }
  else if (sh_degree == 1)
  {
    // Linear terms
    color.x = sh_coeffs[0] + sh_coeffs[3] * view_dir.x + sh_coeffs[4] * view_dir.y + sh_coeffs[5] * view_dir.z;
    color.y = sh_coeffs[1] + sh_coeffs[6] * view_dir.x + sh_coeffs[7] * view_dir.y + sh_coeffs[8] * view_dir.z;
    color.z = sh_coeffs[2] + sh_coeffs[9] * view_dir.x + sh_coeffs[10] * view_dir.y + sh_coeffs[11] * view_dir.z;
  }
  else
  {
    // Higher degree - simplified
    color.x = sh_coeffs[0];
    color.y = sh_coeffs[1];
    color.z = sh_coeffs[2];
  }

  return color;
}

__device__ float compute_voxel_alpha(
    float density,
    const float3 &world_size,
    const char *density_activation)
{
  float sigma;

  // Simple character comparison instead of strcmp
  if (density_activation[0] == 'e' && density_activation[1] == 'x' && density_activation[2] == 'p' && density_activation[3] == '\0')
  {
    sigma = expf(density);
  }
  else
  {
    sigma = fmaxf(density, 0.0f); // ReLU
  }

  // Simplified alpha computation based on voxel size
  float voxel_size = fmaxf(fmaxf(world_size.x, world_size.y), world_size.z);
  float alpha = 1.0f - expf(-sigma * voxel_size);
  return fminf(fmaxf(alpha, 0.0f), 1.0f);
}

__device__ float3 apply_color_activation(
    const float3 &color,
    const char *color_activation)
{
  // Simple character comparison instead of strcmp
  if (color_activation[0] == 's' && color_activation[1] == 'i' && color_activation[2] == 'g' &&
      color_activation[3] == 'm' && color_activation[4] == 'o' && color_activation[5] == 'i' &&
      color_activation[6] == 'd' && color_activation[7] == '\0')
  {
    return make_float3(
        1.0f / (1.0f + expf(-color.x)),
        1.0f / (1.0f + expf(-color.y)),
        1.0f / (1.0f + expf(-color.z)));
  }
  else if (color_activation[0] == 't' && color_activation[1] == 'a' && color_activation[2] == 'n' &&
           color_activation[3] == 'h' && color_activation[4] == '\0')
  {
    return make_float3(
        (tanhf(color.x) + 1.0f) * 0.5f,
        (tanhf(color.y) + 1.0f) * 0.5f,
        (tanhf(color.z) + 1.0f) * 0.5f);
  }
  else
  {
    // Clamp
    return make_float3(
        fminf(fmaxf(color.x, 0.0f), 1.0f),
        fminf(fmaxf(color.y, 0.0f), 1.0f),
        fminf(fmaxf(color.z, 0.0f), 1.0f));
  }
}

// CUDA Kernels implementation

__global__ void project_voxels_to_screen_kernel(
    const RasterVoxel *voxels,
    ScreenVoxel *screen_voxels,
    const CameraParams camera_params,
    int num_voxels)
{
  int idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (idx >= num_voxels)
    return;

  RasterVoxel voxel = voxels[idx];

  // Project to screen space
  float3 screen_pos_3d = project_point_to_screen(voxel.position, camera_params);

  // Create screen voxel
  ScreenVoxel screen_voxel;
  screen_voxel.screen_pos = make_float2(screen_pos_3d.x, screen_pos_3d.y);
  screen_voxel.depth = screen_pos_3d.z;
  screen_voxel.screen_size = compute_screen_size(voxel.size, screen_pos_3d.z, camera_params);
  screen_voxel.density = voxel.density;
  screen_voxel.color = voxel.color;
  screen_voxel.world_pos = voxel.position;
  screen_voxel.world_size = voxel.size;
  screen_voxel.voxel_idx = voxel.voxel_idx;

  screen_voxels[idx] = screen_voxel;
}

__global__ void frustum_culling_kernel(
    const ScreenVoxel *screen_voxels,
    ScreenVoxel *visible_voxels,
    int *visible_count,
    const CameraParams camera_params,
    int num_voxels)
{
  int idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (idx >= num_voxels)
    return;

  ScreenVoxel voxel = screen_voxels[idx];

  if (is_voxel_visible(voxel, camera_params))
  {
    int visible_idx = atomicAdd(visible_count, 1);
    visible_voxels[visible_idx] = voxel;
  }
}

__global__ void depth_sort_kernel(
    ScreenVoxel *voxels,
    int num_voxels)
{
  // This is a simplified sorting kernel
  // In practice, you'd want to use a more efficient sorting algorithm like radix sort

  int idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (idx >= num_voxels)
    return;

  // Simple bubble sort within a block (not efficient for large arrays)
  // For production use, implement radix sort or use thrust/cub

  __shared__ ScreenVoxel shared_voxels[256];
  __shared__ bool swapped;

  // Load voxels into shared memory
  if (idx < num_voxels)
  {
    shared_voxels[threadIdx.x] = voxels[idx];
  }
  __syncthreads();

  // Simple bubble sort (back-to-front)
  for (int i = 0; i < blockDim.x - 1; i++)
  {
    swapped = false;
    __syncthreads();

    if (threadIdx.x < blockDim.x - 1 && threadIdx.x + blockIdx.x * blockDim.x < num_voxels - 1)
    {
      if (shared_voxels[threadIdx.x].depth < shared_voxels[threadIdx.x + 1].depth)
      {
        ScreenVoxel temp = shared_voxels[threadIdx.x];
        shared_voxels[threadIdx.x] = shared_voxels[threadIdx.x + 1];
        shared_voxels[threadIdx.x + 1] = temp;
        swapped = true;
      }
    }
    __syncthreads();

    if (!swapped)
      break;
  }

  // Write back
  if (idx < num_voxels)
  {
    voxels[idx] = shared_voxels[threadIdx.x];
  }
}

__global__ void rasterize_voxels_kernel(
    const ScreenVoxel *sorted_voxels,
    Framebuffer framebuffer,
    const CameraParams camera_params,
    int num_voxels)
{
  int voxel_idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (voxel_idx >= num_voxels)
    return;

  ScreenVoxel voxel = sorted_voxels[voxel_idx];

  // Compute voxel color (with spherical harmonics if needed)
  float3 voxel_color = voxel.color; // Simplified for now

  // Compute voxel alpha
  float voxel_alpha = compute_voxel_alpha(voxel.density, voxel.world_size, "exp");

  // Compute pixel range for this voxel
  float half_size = voxel.screen_size * 0.5f;
  int min_x = max(0, (int)(voxel.screen_pos.x - half_size));
  int max_x = min(framebuffer.width, (int)(voxel.screen_pos.x + half_size) + 1);
  int min_y = max(0, (int)(voxel.screen_pos.y - half_size));
  int max_y = min(framebuffer.height, (int)(voxel.screen_pos.y + half_size) + 1);

  // Rasterize to pixels
  for (int y = min_y; y < max_y; y++)
  {
    for (int x = min_x; x < max_x; x++)
    {
      // Distance from pixel center to voxel center
      float dx = x - voxel.screen_pos.x;
      float dy = y - voxel.screen_pos.y;
      float distance = sqrtf(dx * dx + dy * dy);

      if (distance <= half_size)
      {
        // Compute pixel alpha (distance attenuation)
        float pixel_alpha = voxel_alpha * (1.0f - distance / half_size);
        pixel_alpha = fminf(fmaxf(pixel_alpha, 0.0f), 1.0f);

        // Atomic alpha blending
        int pixel_idx = y * framebuffer.width + x;
        float current_alpha = framebuffer.alpha_buffer[pixel_idx];
        float blend_factor = pixel_alpha * (1.0f - current_alpha);

        if (blend_factor > 0.0f)
        {
          // Atomic color blending
          atomicAdd(&framebuffer.color_buffer[pixel_idx].x, voxel_color.x * blend_factor);
          atomicAdd(&framebuffer.color_buffer[pixel_idx].y, voxel_color.y * blend_factor);
          atomicAdd(&framebuffer.color_buffer[pixel_idx].z, voxel_color.z * blend_factor);

          // Atomic alpha update
          atomicAdd(&framebuffer.alpha_buffer[pixel_idx], blend_factor);

          // Atomic depth update (weighted average)
          atomicAdd(&framebuffer.depth_buffer[pixel_idx], voxel.depth * blend_factor);
        }
      }
    }
  }
}

__global__ void alpha_blending_kernel(
    const float3 *new_color,
    const float *new_alpha,
    const float *new_depth,
    Framebuffer framebuffer,
    int width,
    int height)
{
  int pixel_idx = blockIdx.x * blockDim.x + threadIdx.x;
  int x = pixel_idx % width;
  int y = pixel_idx / width;

  if (x >= width || y >= height)
    return;

  int idx = y * width + x;

  float current_alpha = framebuffer.alpha_buffer[idx];
  float new_alpha_val = new_alpha[idx];
  float blend_factor = new_alpha_val * (1.0f - current_alpha);

  if (blend_factor > 0.0f)
  {
    // Blend colors
    framebuffer.color_buffer[idx] = framebuffer.color_buffer[idx] * (1.0f - blend_factor) +
                                    new_color[idx] * blend_factor;

    // Update alpha
    framebuffer.alpha_buffer[idx] = current_alpha + blend_factor;

    // Update depth (weighted average)
    framebuffer.depth_buffer[idx] = framebuffer.depth_buffer[idx] * (1.0f - blend_factor) +
                                    new_depth[idx] * blend_factor;
  }
}

// Kernel launch wrapper functions
void launch_project_voxels_kernel(
    int grid_size, int block_size,
    const RasterVoxel *voxels,
    ScreenVoxel *screen_voxels,
    const CameraParams &camera_params,
    int num_voxels)
{
  project_voxels_to_screen_kernel<<<grid_size, block_size>>>(
      voxels, screen_voxels, camera_params, num_voxels);
}

void launch_frustum_culling_kernel(
    int grid_size, int block_size,
    const ScreenVoxel *screen_voxels,
    ScreenVoxel *visible_voxels,
    int *visible_count,
    const CameraParams &camera_params,
    int num_voxels)
{
  // Reset visible count
  cudaMemset(visible_count, 0, sizeof(int));

  frustum_culling_kernel<<<grid_size, block_size>>>(
      screen_voxels, visible_voxels, visible_count, camera_params, num_voxels);
}

void launch_depth_sort_kernel(
    int grid_size, int block_size,
    ScreenVoxel *voxels,
    int num_voxels)
{
  depth_sort_kernel<<<grid_size, block_size>>>(voxels, num_voxels);
}

void launch_rasterize_voxels_kernel(
    int grid_size, int block_size,
    const ScreenVoxel *sorted_voxels,
    Framebuffer &framebuffer,
    const CameraParams &camera_params,
    int num_voxels)
{
  rasterize_voxels_kernel<<<grid_size, block_size>>>(
      sorted_voxels, framebuffer, camera_params, num_voxels);
}

void launch_alpha_blending_kernel(
    int grid_size, int block_size,
    const float3 *new_color,
    const float *new_alpha,
    const float *new_depth,
    Framebuffer &framebuffer,
    int width,
    int height)
{
  alpha_blending_kernel<<<grid_size, block_size>>>(
      new_color, new_alpha, new_depth, framebuffer, width, height);
}

// Utility functions
void allocate_framebuffer(Framebuffer &framebuffer, int width, int height)
{
  framebuffer.width = width;
  framebuffer.height = height;

  cudaMalloc(&framebuffer.color_buffer, width * height * sizeof(float3));
  cudaMalloc(&framebuffer.depth_buffer, width * height * sizeof(float));
  cudaMalloc(&framebuffer.alpha_buffer, width * height * sizeof(float));

  // Initialize with background
  cudaMemset(framebuffer.color_buffer, 0, width * height * sizeof(float3));
  cudaMemset(framebuffer.depth_buffer, 0, width * height * sizeof(float));
  cudaMemset(framebuffer.alpha_buffer, 0, width * height * sizeof(float));
}

void free_framebuffer(Framebuffer &framebuffer)
{
  if (framebuffer.color_buffer)
    cudaFree(framebuffer.color_buffer);
  if (framebuffer.depth_buffer)
    cudaFree(framebuffer.depth_buffer);
  if (framebuffer.alpha_buffer)
    cudaFree(framebuffer.alpha_buffer);

  framebuffer.color_buffer = nullptr;
  framebuffer.depth_buffer = nullptr;
  framebuffer.alpha_buffer = nullptr;
}

void radix_sort_screen_voxels(ScreenVoxel *voxels, int num_voxels)
{
  // Implementation using CUB library for efficient sorting
  // This is a placeholder - implement with CUB::DeviceRadixSort
  // For now, we'll use the simple kernel-based sorting
}