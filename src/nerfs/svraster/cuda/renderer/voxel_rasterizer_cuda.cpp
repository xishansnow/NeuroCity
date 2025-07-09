#include <torch/extension.h>
#include <cuda_runtime.h>
#include "voxel_rasterizer_cuda_kernel.h"

// Error checking macro
#define CUDA_CHECK(call)                                       \
  do                                                           \
  {                                                            \
    cudaError_t err = call;                                    \
    if (err != cudaSuccess)                                    \
    {                                                          \
      printf("CUDA error at %s:%d - %s\n", __FILE__, __LINE__, \
             cudaGetErrorString(err));                         \
      throw std::runtime_error("CUDA error");                  \
    }                                                          \
  } while (0)

// Main voxel rasterization function
std::tuple<torch::Tensor, torch::Tensor> voxel_rasterization_cuda(
    torch::Tensor voxel_positions, // [N, 3]
    torch::Tensor voxel_sizes,     // [N] or [N, 3]
    torch::Tensor voxel_densities, // [N]
    torch::Tensor voxel_colors,    // [N, C] where C is color dimension
    torch::Tensor camera_matrix,   // [4, 4]
    torch::Tensor intrinsics,      // [3, 3]
    torch::Tensor viewport_size,   // [2]
    float near_plane,
    float far_plane,
    torch::Tensor background_color, // [3]
    const std::string &density_activation,
    const std::string &color_activation,
    int sh_degree)
{
  // Get dimensions
  int num_voxels = voxel_positions.size(0);
  int width = viewport_size[0].item<int>();
  int height = viewport_size[1].item<int>();

  // Ensure tensors are on GPU and contiguous
  voxel_positions = voxel_positions.contiguous().cuda();
  voxel_sizes = voxel_sizes.contiguous().cuda();
  voxel_densities = voxel_densities.contiguous().cuda();
  voxel_colors = voxel_colors.contiguous().cuda();
  camera_matrix = camera_matrix.contiguous().cuda();
  intrinsics = intrinsics.contiguous().cuda();
  background_color = background_color.contiguous().cuda();

  // Allocate output tensors
  auto output_rgb = torch::zeros({height, width, 3},
                                 torch::dtype(torch::kFloat32).device(torch::kCUDA));
  auto output_depth = torch::full({height, width}, far_plane,
                                  torch::dtype(torch::kFloat32).device(torch::kCUDA));

  // Convert voxel data to CUDA format
  RasterVoxel *d_voxels;
  CUDA_CHECK(cudaMalloc(&d_voxels, num_voxels * sizeof(RasterVoxel)));

  // Copy voxel data
  for (int i = 0; i < num_voxels; i++)
  {
    RasterVoxel voxel;
    voxel.position = make_float3(
        voxel_positions[i][0].item<float>(),
        voxel_positions[i][1].item<float>(),
        voxel_positions[i][2].item<float>());

    // Handle different size formats
    if (voxel_sizes.dim() == 2)
    {
      voxel.size = make_float3(
          voxel_sizes[i][0].item<float>(),
          voxel_sizes[i][1].item<float>(),
          voxel_sizes[i][2].item<float>());
    }
    else
    {
      float size_val = voxel_sizes[i].item<float>();
      voxel.size = make_float3(size_val, size_val, size_val);
    }

    voxel.density = voxel_densities[i].item<float>();

    // Handle color data (simplified - take first 3 channels)
    int color_dim = voxel_colors.size(1);
    if (color_dim >= 3)
    {
      voxel.color = make_float3(
          voxel_colors[i][0].item<float>(),
          voxel_colors[i][1].item<float>(),
          voxel_colors[i][2].item<float>());
    }
    else
    {
      // Pad with zeros if needed
      voxel.color = make_float3(
          color_dim > 0 ? voxel_colors[i][0].item<float>() : 0.0f,
          color_dim > 1 ? voxel_colors[i][1].item<float>() : 0.0f,
          color_dim > 2 ? voxel_colors[i][2].item<float>() : 0.0f);
    }

    voxel.voxel_idx = i;

    CUDA_CHECK(cudaMemcpy(d_voxels + i, &voxel, sizeof(RasterVoxel), cudaMemcpyHostToDevice));
  }

  // Convert camera parameters to CUDA format
  CameraParams camera_params;

  // Copy camera matrix
  for (int i = 0; i < 4; i++)
  {
    for (int j = 0; j < 4; j++)
    {
      camera_params.camera_matrix.m[i][j] = camera_matrix[i][j].item<float>();
    }
  }

  // Copy intrinsics matrix
  for (int i = 0; i < 3; i++)
  {
    for (int j = 0; j < 3; j++)
    {
      camera_params.intrinsics.m[i][j] = intrinsics[i][j].item<float>();
    }
  }

  camera_params.viewport_size = make_float2(width, height);
  camera_params.near_plane = near_plane;
  camera_params.far_plane = far_plane;
  camera_params.background_color = make_float3(
      background_color[0].item<float>(),
      background_color[1].item<float>(),
      background_color[2].item<float>());

  // Allocate intermediate buffers
  ScreenVoxel *d_screen_voxels;
  ScreenVoxel *d_visible_voxels;
  int *d_visible_count;

  CUDA_CHECK(cudaMalloc(&d_screen_voxels, num_voxels * sizeof(ScreenVoxel)));
  CUDA_CHECK(cudaMalloc(&d_visible_voxels, num_voxels * sizeof(ScreenVoxel)));
  CUDA_CHECK(cudaMalloc(&d_visible_count, sizeof(int)));

  // Allocate framebuffer
  Framebuffer framebuffer;
  allocate_framebuffer(framebuffer, width, height);

  // Launch kernels
  int block_size = 256;
  int grid_size = (num_voxels + block_size - 1) / block_size;

  // 1. Project voxels to screen space
  launch_project_voxels_kernel(
      grid_size, block_size,
      d_voxels, d_screen_voxels, camera_params, num_voxels);

  // 2. Frustum culling
  launch_frustum_culling_kernel(
      grid_size, block_size,
      d_screen_voxels, d_visible_voxels, d_visible_count, camera_params, num_voxels);

  // Get visible count
  int visible_count;
  CUDA_CHECK(cudaMemcpy(&visible_count, d_visible_count, sizeof(int), cudaMemcpyDeviceToHost));

  if (visible_count > 0)
  {
    // 3. Depth sorting (back-to-front)
    int sort_grid_size = (visible_count + block_size - 1) / block_size;
    launch_depth_sort_kernel(sort_grid_size, block_size, d_visible_voxels, visible_count);

    // 4. Rasterization
    int raster_grid_size = (visible_count + block_size - 1) / block_size;
    launch_rasterize_voxels_kernel(
        raster_grid_size, block_size,
        d_visible_voxels, framebuffer, camera_params, visible_count);
  }

  // Copy results back to CPU
  CUDA_CHECK(cudaMemcpy(output_rgb.data_ptr(), framebuffer.color_buffer,
                        width * height * sizeof(float3), cudaMemcpyDeviceToDevice));
  CUDA_CHECK(cudaMemcpy(output_depth.data_ptr(), framebuffer.depth_buffer,
                        width * height * sizeof(float), cudaMemcpyDeviceToDevice));

  // Cleanup
  cudaFree(d_voxels);
  cudaFree(d_screen_voxels);
  cudaFree(d_visible_voxels);
  cudaFree(d_visible_count);
  free_framebuffer(framebuffer);

  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());

  return std::make_tuple(output_rgb, output_depth);
}

// Utility function to create camera matrix from pose
torch::Tensor create_camera_matrix_cuda(torch::Tensor camera_pose)
{
  // Simply return the pose matrix as camera matrix
  return camera_pose.contiguous().cuda();
}

// Utility function to estimate camera parameters from rays
std::tuple<torch::Tensor, torch::Tensor> rays_to_camera_matrix_cuda(
    torch::Tensor ray_origins,
    torch::Tensor ray_directions)
{
  // Simplified implementation - in practice you'd want more sophisticated estimation

  ray_origins = ray_origins.contiguous().cuda();
  ray_directions = ray_directions.contiguous().cuda();

  // Estimate camera center as mean of ray origins
  auto camera_center = torch::mean(ray_origins, 0);

  // Estimate camera direction as negative mean of ray directions
  auto mean_direction = torch::mean(ray_directions, 0);
  mean_direction = mean_direction / torch::norm(mean_direction);

  // Build camera coordinate system
  auto forward = -mean_direction;
  auto up_vec = torch::tensor({0.0f, 1.0f, 0.0f},
                              torch::dtype(torch::kFloat32).device(torch::kCUDA));
  auto right = torch::cross(forward, up_vec);
  right = right / torch::norm(right);
  auto up = torch::cross(right, forward);

  // Build camera transformation matrix
  auto rotation = torch::stack({right, up, forward}, 1); // [3, 3]
  auto translation = -torch::matmul(rotation.t(), camera_center.unsqueeze(1)).squeeze(1);

  auto camera_matrix = torch::zeros({4, 4},
                                    torch::dtype(torch::kFloat32).device(torch::kCUDA));
  camera_matrix.index_put_({torch::indexing::Slice(0, 3), torch::indexing::Slice(0, 3)}, rotation.t());
  camera_matrix.index_put_({torch::indexing::Slice(0, 3), 3}, translation);
  camera_matrix.index_put_({3, 3}, torch::tensor(1.0f, torch::dtype(torch::kFloat32).device(torch::kCUDA)));

  // Simplified intrinsics matrix
  auto intrinsics = torch::tensor({{800.0f, 0.0f, 400.0f},
                                   {0.0f, 800.0f, 300.0f},
                                   {0.0f, 0.0f, 1.0f}},
                                  torch::dtype(torch::kFloat32).device(torch::kCUDA));

  return std::make_tuple(camera_matrix, intrinsics);
}

// Performance benchmarking function
std::map<std::string, float> benchmark_voxel_rasterizer_cuda(
    torch::Tensor voxel_positions,
    torch::Tensor voxel_sizes,
    torch::Tensor voxel_densities,
    torch::Tensor voxel_colors,
    torch::Tensor camera_matrix,
    torch::Tensor intrinsics,
    torch::Tensor viewport_size,
    int num_iterations)
{
  std::map<std::string, float> timings;

  // Warmup
  for (int i = 0; i < 3; i++)
  {
    voxel_rasterization_cuda(
        voxel_positions, voxel_sizes, voxel_densities, voxel_colors,
        camera_matrix, intrinsics, viewport_size,
        0.1f, 100.0f, torch::tensor({0.0f, 0.0f, 0.0f}, torch::dtype(torch::kFloat32).device(torch::kCUDA)),
        "exp", "sigmoid", 2);
  }

  CUDA_CHECK(cudaDeviceSynchronize());

  // Benchmark
  auto start_time = std::chrono::high_resolution_clock::now();

  for (int i = 0; i < num_iterations; i++)
  {
    voxel_rasterization_cuda(
        voxel_positions, voxel_sizes, voxel_densities, voxel_colors,
        camera_matrix, intrinsics, viewport_size,
        0.1f, 100.0f, torch::tensor({0.0f, 0.0f, 0.0f}, torch::dtype(torch::kFloat32).device(torch::kCUDA)),
        "exp", "sigmoid", 2);
  }

  CUDA_CHECK(cudaDeviceSynchronize());

  auto end_time = std::chrono::high_resolution_clock::now();
  auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);

  timings["total_time_ms"] = duration.count() / 1000.0f;
  timings["avg_time_ms"] = timings["total_time_ms"] / num_iterations;
  timings["fps"] = 1000.0f / timings["avg_time_ms"];

  return timings;
}

// Python module bindings
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
  m.doc() = "CUDA-accelerated voxel rasterizer for SVRaster";

  m.def("voxel_rasterization", &voxel_rasterization_cuda,
        "CUDA-accelerated voxel rasterization",
        py::arg("voxel_positions"),
        py::arg("voxel_sizes"),
        py::arg("voxel_densities"),
        py::arg("voxel_colors"),
        py::arg("camera_matrix"),
        py::arg("intrinsics"),
        py::arg("viewport_size"),
        py::arg("near_plane") = 0.1f,
        py::arg("far_plane") = 100.0f,
        py::arg("background_color") = py::none(),
        py::arg("density_activation") = "exp",
        py::arg("color_activation") = "sigmoid",
        py::arg("sh_degree") = 2);

  m.def("create_camera_matrix", &create_camera_matrix_cuda,
        "Create camera matrix from pose",
        py::arg("camera_pose"));

  m.def("rays_to_camera_matrix", &rays_to_camera_matrix_cuda,
        "Estimate camera matrix from rays",
        py::arg("ray_origins"),
        py::arg("ray_directions"));

  m.def("benchmark", &benchmark_voxel_rasterizer_cuda,
        "Benchmark voxel rasterizer performance",
        py::arg("voxel_positions"),
        py::arg("voxel_sizes"),
        py::arg("voxel_densities"),
        py::arg("voxel_colors"),
        py::arg("camera_matrix"),
        py::arg("intrinsics"),
        py::arg("viewport_size"),
        py::arg("num_iterations") = 100);
}