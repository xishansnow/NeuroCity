/**
 * Block-NeRF CUDA C++ Bindings - Fixed Version
 * 
 * Based on "Block-NeRF: Scalable Large Scene Neural View Synthesis" (CVPR 2022)
 * Optimized for GTX 1080 Ti (Compute Capability 6.1)
 */

#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <vector>
#include <iostream>

// Structure definitions (must match CUDA kernels)
struct BlockInfo {
    float3 center;
    float radius;
    int active;
    int block_id;
};

struct Ray {
    float3 origin;
    float3 direction;
    float near_plane;
    float far_plane;
};

struct SamplePoint {
    float3 position;
    float t_val;
    int block_id;
};

// CUDA kernel launchers (declared in CUDA kernel file)
extern "C" {
    void launch_memory_bandwidth_test_kernel(
        const float* input_data,
        float* output_data,
        const int num_elements
    );
    
    void launch_block_visibility_kernel(
        const float3* camera_positions,
        const BlockInfo* blocks,
        float* visibility_scores,
        const int num_cameras,
        const int num_blocks,
        const float visibility_threshold
    );
    
    void launch_block_selection_kernel(
        const Ray* rays,
        const BlockInfo* blocks,
        int* selected_blocks,
        int* num_selected_blocks,
        const int num_rays,
        const int num_blocks
    );
}

// Wrapper functions for PyTorch tensors

torch::Tensor memory_bandwidth_test_cuda(torch::Tensor input_data) {
    // Check input tensor
    TORCH_CHECK(input_data.is_cuda(), "Input tensor must be on CUDA device");
    TORCH_CHECK(input_data.dtype() == torch::kFloat32, "Input tensor must be float32");
    
    const int num_elements = input_data.numel();
    auto output_data = torch::empty_like(input_data);
    
    const int threads_per_block = 256;
    const int num_blocks = (num_elements + threads_per_block - 1) / threads_per_block;
    
    // Launch CUDA kernel
    launch_memory_bandwidth_test_kernel(
        input_data.data_ptr<float>(),
        output_data.data_ptr<float>(),
        num_elements
    );
    
    // Check for CUDA errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "CUDA kernel error: " << cudaGetErrorString(err) << std::endl;
    }
    
    return output_data;
}

torch::Tensor block_visibility_cuda(
    torch::Tensor camera_positions,
    torch::Tensor block_centers,
    torch::Tensor block_radii,
    torch::Tensor block_active,
    float visibility_threshold
) {
    // Check input tensors
    TORCH_CHECK(camera_positions.is_cuda(), "camera_positions must be on CUDA device");
    TORCH_CHECK(block_centers.is_cuda(), "block_centers must be on CUDA device");
    TORCH_CHECK(block_radii.is_cuda(), "block_radii must be on CUDA device");
    TORCH_CHECK(block_active.is_cuda(), "block_active must be on CUDA device");
    
    const int num_cameras = camera_positions.size(0);
    const int num_blocks = block_centers.size(0);
    
    // Create output tensor
    auto visibility_scores = torch::zeros({num_blocks}, 
                                        torch::TensorOptions().dtype(torch::kFloat32).device(camera_positions.device()));
    
    // Convert input data to CUDA structures
    std::vector<BlockInfo> host_blocks(num_blocks);
    auto block_centers_cpu = block_centers.cpu();
    auto block_radii_cpu = block_radii.cpu();
    auto block_active_cpu = block_active.cpu();
    
    for (int i = 0; i < num_blocks; i++) {
        host_blocks[i].center = make_float3(
            block_centers_cpu[i][0].item<float>(),
            block_centers_cpu[i][1].item<float>(),
            block_centers_cpu[i][2].item<float>()
        );
        host_blocks[i].radius = block_radii_cpu[i].item<float>();
        host_blocks[i].active = block_active_cpu[i].item<int>();
        host_blocks[i].block_id = i;
    }
    
    // Allocate GPU memory for blocks
    BlockInfo* device_blocks;
    cudaMalloc(&device_blocks, num_blocks * sizeof(BlockInfo));
    cudaMemcpy(device_blocks, host_blocks.data(), num_blocks * sizeof(BlockInfo), cudaMemcpyHostToDevice);
    
    const int threads_per_block = 256;
    const int num_cuda_blocks = (num_blocks + threads_per_block - 1) / threads_per_block;
    
    // Launch CUDA kernel
    launch_block_visibility_kernel(
        reinterpret_cast<const float3*>(camera_positions.data_ptr<float>()),
        device_blocks,
        visibility_scores.data_ptr<float>(),
        num_cameras,
        num_blocks,
        visibility_threshold
    );
    
    // Check for CUDA errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "CUDA kernel error: " << cudaGetErrorString(err) << std::endl;
    }
    
    // Cleanup
    cudaFree(device_blocks);
    
    return visibility_scores;
}

std::vector<torch::Tensor> block_selection_cuda(
    torch::Tensor ray_origins,
    torch::Tensor ray_directions,
    torch::Tensor ray_near,
    torch::Tensor ray_far,
    torch::Tensor block_centers,
    torch::Tensor block_radii,
    torch::Tensor block_active,
    int max_blocks_per_ray
) {
    // Check input tensors
    TORCH_CHECK(ray_origins.is_cuda(), "ray_origins must be on CUDA device");
    TORCH_CHECK(ray_directions.is_cuda(), "ray_directions must be on CUDA device");
    
    const int num_rays = ray_origins.size(0);
    const int num_blocks = block_centers.size(0);
    
    // Create output tensors
    auto selected_blocks = torch::zeros({num_rays, max_blocks_per_ray}, 
                                      torch::TensorOptions().dtype(torch::kInt32).device(ray_origins.device()));
    auto num_selected_blocks = torch::zeros({num_rays}, 
                                          torch::TensorOptions().dtype(torch::kInt32).device(ray_origins.device()));
    
    // Convert input data to CUDA structures
    std::vector<Ray> host_rays(num_rays);
    auto ray_origins_cpu = ray_origins.cpu();
    auto ray_directions_cpu = ray_directions.cpu();
    auto ray_near_cpu = ray_near.cpu();
    auto ray_far_cpu = ray_far.cpu();
    
    for (int i = 0; i < num_rays; i++) {
        host_rays[i].origin = make_float3(
            ray_origins_cpu[i][0].item<float>(),
            ray_origins_cpu[i][1].item<float>(),
            ray_origins_cpu[i][2].item<float>()
        );
        host_rays[i].direction = make_float3(
            ray_directions_cpu[i][0].item<float>(),
            ray_directions_cpu[i][1].item<float>(),
            ray_directions_cpu[i][2].item<float>()
        );
        host_rays[i].near_plane = ray_near_cpu[i].item<float>();
        host_rays[i].far_plane = ray_far_cpu[i].item<float>();
    }
    
    std::vector<BlockInfo> host_blocks(num_blocks);
    auto block_centers_cpu = block_centers.cpu();
    auto block_radii_cpu = block_radii.cpu();
    auto block_active_cpu = block_active.cpu();
    
    for (int i = 0; i < num_blocks; i++) {
        host_blocks[i].center = make_float3(
            block_centers_cpu[i][0].item<float>(),
            block_centers_cpu[i][1].item<float>(),
            block_centers_cpu[i][2].item<float>()
        );
        host_blocks[i].radius = block_radii_cpu[i].item<float>();
        host_blocks[i].active = block_active_cpu[i].item<int>();
        host_blocks[i].block_id = i;
    }
    
    // Allocate GPU memory
    Ray* device_rays;
    BlockInfo* device_blocks;
    cudaMalloc(&device_rays, num_rays * sizeof(Ray));
    cudaMalloc(&device_blocks, num_blocks * sizeof(BlockInfo));
    
    cudaMemcpy(device_rays, host_rays.data(), num_rays * sizeof(Ray), cudaMemcpyHostToDevice);
    cudaMemcpy(device_blocks, host_blocks.data(), num_blocks * sizeof(BlockInfo), cudaMemcpyHostToDevice);
    
    const int threads_per_block = 256;
    const int num_cuda_blocks = (num_rays + threads_per_block - 1) / threads_per_block;
    
    // Launch CUDA kernel
    launch_block_selection_kernel(
        device_rays,
        device_blocks,
        selected_blocks.data_ptr<int>(),
        num_selected_blocks.data_ptr<int>(),
        num_rays,
        num_blocks
    );
    
    // Check for CUDA errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "CUDA kernel error: " << cudaGetErrorString(err) << std::endl;
    }
    
    // Cleanup
    cudaFree(device_rays);
    cudaFree(device_blocks);
    
    return {selected_blocks, num_selected_blocks};
}

// PyBind11 module definition
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("memory_bandwidth_test", &memory_bandwidth_test_cuda, "Memory bandwidth test (CUDA)");
    m.def("block_visibility", &block_visibility_cuda, "Block visibility computation (CUDA)");
    m.def("block_selection", &block_selection_cuda, "Block selection for rays (CUDA)");
}
