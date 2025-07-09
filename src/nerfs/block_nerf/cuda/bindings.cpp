/**
 * Simple C++ Bindings for Block-NeRF CUDA Extension
 * 
 * This file provides Python bindings for the CUDA kernels
 * defined in kernels.cu
 */

#include <torch/extension.h>
#include <vector>

// Forward declarations of CUDA functions
torch::Tensor memory_bandwidth_test_cuda(torch::Tensor input);
torch::Tensor simple_add_cuda(torch::Tensor a, torch::Tensor b);
torch::Tensor simple_multiply_cuda(torch::Tensor a, torch::Tensor b);
torch::Tensor block_visibility_cuda(
    torch::Tensor camera_positions,
    torch::Tensor block_centers,
    torch::Tensor block_radii,
    torch::Tensor view_directions,
    float visibility_threshold
);

// C++ interface functions
torch::Tensor memory_bandwidth_test(torch::Tensor input) {
    TORCH_CHECK(input.is_cuda(), "Input tensor must be on CUDA device");
    TORCH_CHECK(input.dtype() == torch::kFloat32, "Input tensor must be float32");
    return memory_bandwidth_test_cuda(input);
}

torch::Tensor simple_add(torch::Tensor a, torch::Tensor b) {
    TORCH_CHECK(a.is_cuda() && b.is_cuda(), "Input tensors must be on CUDA device");
    TORCH_CHECK(a.dtype() == torch::kFloat32 && b.dtype() == torch::kFloat32, 
                "Input tensors must be float32");
    TORCH_CHECK(a.sizes() == b.sizes(), "Input tensors must have same shape");
    return simple_add_cuda(a, b);
}

torch::Tensor simple_multiply(torch::Tensor a, torch::Tensor b) {
    TORCH_CHECK(a.is_cuda() && b.is_cuda(), "Input tensors must be on CUDA device");
    TORCH_CHECK(a.dtype() == torch::kFloat32 && b.dtype() == torch::kFloat32, 
                "Input tensors must be float32");
    TORCH_CHECK(a.sizes() == b.sizes(), "Input tensors must have same shape");
    return simple_multiply_cuda(a, b);
}

torch::Tensor block_visibility(
    torch::Tensor camera_positions,    // [N, 3]
    torch::Tensor block_centers,       // [M, 3] 
    torch::Tensor block_radii,         // [M]
    torch::Tensor view_directions,     // [N, 3]
    float visibility_threshold = 0.5
) {
    TORCH_CHECK(camera_positions.is_cuda() && block_centers.is_cuda() && 
                block_radii.is_cuda() && view_directions.is_cuda(),
                "All input tensors must be on CUDA device");
    
    TORCH_CHECK(camera_positions.dtype() == torch::kFloat32 && 
                block_centers.dtype() == torch::kFloat32 &&
                block_radii.dtype() == torch::kFloat32 &&
                view_directions.dtype() == torch::kFloat32,
                "All input tensors must be float32");
    
    TORCH_CHECK(camera_positions.dim() == 2 && camera_positions.size(1) == 3, 
                "Camera positions must be [N, 3]");
    TORCH_CHECK(block_centers.dim() == 2 && block_centers.size(1) == 3,
                "Block centers must be [M, 3]");
    TORCH_CHECK(block_radii.dim() == 1,
                "Block radii must be [M]");
    TORCH_CHECK(view_directions.dim() == 2 && view_directions.size(1) == 3,
                "View directions must be [N, 3]");
    
    return block_visibility_cuda(camera_positions, block_centers, block_radii, 
                                view_directions, visibility_threshold);
}

// Simple block selection based on visibility (CPU implementation)
std::vector<torch::Tensor> block_selection(
    torch::Tensor rays_o,           // [N, 3] Ray origins
    torch::Tensor rays_d,           // [N, 3] Ray directions  
    torch::Tensor block_centers,    // [M, 3] Block centers
    torch::Tensor block_radii,      // [M] Block radii
    int max_blocks = 8
) {
    // Input validation
    TORCH_CHECK(rays_o.dim() == 2 && rays_o.size(1) == 3, "rays_o must be [N, 3]");
    TORCH_CHECK(rays_d.dim() == 2 && rays_d.size(1) == 3, "rays_d must be [N, 3]");
    TORCH_CHECK(block_centers.dim() == 2 && block_centers.size(1) == 3, "block_centers must be [M, 3]");
    TORCH_CHECK(block_radii.dim() == 1, "block_radii must be [M]");
    TORCH_CHECK(max_blocks > 0, "max_blocks must be positive");
    
    int num_rays = rays_o.size(0);
    int num_blocks = block_centers.size(0);
    
    // Move all tensors to CPU for safe processing
    auto rays_o_cpu = rays_o.cpu().contiguous();
    auto block_centers_cpu = block_centers.cpu().contiguous();
    auto block_radii_cpu = block_radii.cpu().contiguous();
    
    // Create output tensors on CPU first
    auto selected_blocks_cpu = torch::zeros({num_rays, max_blocks}, torch::kInt32);
    auto num_selected_cpu = torch::zeros({num_rays}, torch::kInt32);
    
    // Get accessors for CPU tensors
    auto rays_o_acc = rays_o_cpu.accessor<float, 2>();
    auto block_centers_acc = block_centers_cpu.accessor<float, 2>();
    auto block_radii_acc = block_radii_cpu.accessor<float, 1>();
    auto selected_acc = selected_blocks_cpu.accessor<int, 2>();
    auto num_selected_acc = num_selected_cpu.accessor<int, 1>();
    
    // Process each ray
    for (int i = 0; i < num_rays; i++) {
        std::vector<std::pair<float, int>> distances;
        distances.reserve(num_blocks);  // Pre-allocate for efficiency
        
        for (int j = 0; j < num_blocks; j++) {
            float dx = rays_o_acc[i][0] - block_centers_acc[j][0];
            float dy = rays_o_acc[i][1] - block_centers_acc[j][1]; 
            float dz = rays_o_acc[i][2] - block_centers_acc[j][2];
            float distance = std::sqrt(dx*dx + dy*dy + dz*dz);
            
            // Only consider blocks within certain range
            if (distance <= block_radii_acc[j] * 3.0f) {
                distances.push_back({distance, j});
            }
        }
        
        // Sort by distance and select closest
        std::sort(distances.begin(), distances.end());
        int num_select = std::min(static_cast<int>(distances.size()), max_blocks);
        
        // Fill selected blocks
        for (int k = 0; k < num_select; k++) {
            selected_acc[i][k] = distances[k].second;
        }
        // Fill remaining slots with -1 (invalid block)
        for (int k = num_select; k < max_blocks; k++) {
            selected_acc[i][k] = -1;
        }
        
        num_selected_acc[i] = num_select;
    }
    
    // Move results back to original device
    auto selected_blocks = selected_blocks_cpu.to(rays_o.device());
    auto num_selected = num_selected_cpu.to(rays_o.device());
    
    return {selected_blocks, num_selected};
}

// Python bindings
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("memory_bandwidth_test", &memory_bandwidth_test, 
          "Memory bandwidth test for CUDA performance");
    m.def("simple_add", &simple_add, 
          "Simple CUDA addition operation");
    m.def("simple_multiply", &simple_multiply, 
          "Simple CUDA multiplication operation");
    m.def("block_visibility", &block_visibility, 
          "Block visibility computation using CUDA",
          py::arg("camera_positions"), py::arg("block_centers"), 
          py::arg("block_radii"), py::arg("view_directions"),
          py::arg("visibility_threshold") = 0.5f);
    m.def("block_selection", &block_selection, 
          "Block selection for rays based on proximity",
          py::arg("rays_o"), py::arg("rays_d"), py::arg("block_centers"), 
          py::arg("block_radii"), py::arg("max_blocks") = 8);
}
