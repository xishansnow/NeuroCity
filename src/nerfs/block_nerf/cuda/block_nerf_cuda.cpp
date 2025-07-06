/**
 * Block-NeRF CUDA C++ Bindings for PyTorch
 * 
 * Based on "Block-NeRF: Scalable Large Scene Neural View Synthesis" (CVPR 2022)
 * Optimized for GTX 1080 Ti (Compute Capability 6.1)
 */

#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>
#include <iostream>

// CUDA kernel declarations
extern "C" {
    void block_visibility_kernel(
        const float3* camera_positions,
        const struct BlockInfo* blocks,
        float* visibility_scores,
        const int num_cameras,
        const int num_blocks,
        const float visibility_threshold
    );
    
    void block_selection_kernel(
        const struct Ray* rays,
        const struct BlockInfo* blocks,
        int* selected_blocks,
        int* num_selected_blocks,
        const int num_rays,
        const int num_blocks
    );
    
    void ray_sampling_kernel(
        const struct Ray* rays,
        const int* selected_blocks,
        const int* num_selected_blocks,
        const struct BlockInfo* blocks,
        struct SamplePoint* sample_points,
        int* num_samples_per_ray,
        const int num_rays,
        const int samples_per_block,
        curandState* rand_states
    );
    
    void block_interpolation_weights_kernel(
        const float3* sample_positions,
        const int* block_ids,
        const struct BlockInfo* blocks,
        float* interpolation_weights,
        const int num_samples,
        const float overlap_factor
    );
    
    void block_volume_rendering_kernel(
        const struct SamplePoint* sample_points,
        const float* densities,
        const float3* colors,
        const float* interpolation_weights,
        const int* num_samples_per_ray,
        float3* output_colors,
        float* output_depths,
        float* output_alphas,
        const int num_rays,
        const float sigma_factor
    );
    
    void appearance_embedding_kernel(
        const int* appearance_ids,
        const int* block_ids,
        const float* appearance_embeddings,
        float* output_embeddings,
        const int num_samples,
        const int embed_dim,
        const int num_appearances
    );
    
    void block_gradient_kernel(
        const float3* grad_colors,
        const float* grad_depths,
        const struct SamplePoint* sample_points,
        const float* densities,
        const float3* colors,
        const float* interpolation_weights,
        const int* num_samples_per_ray,
        float* grad_densities,
        float3* grad_colors_out,
        const int num_rays,
        const float sigma_factor
    );
    
    void memory_bandwidth_test_kernel(
        const float* input_data,
        float* output_data,
        const int num_elements
    );
}

// Helper structures (matching CUDA kernel definitions)
struct BlockInfo {
    float3 center;
    float radius;
    int block_id;
    bool active;
};

struct Ray {
    float3 origin;
    float3 direction;
    float near;
    float far;
};

struct SamplePoint {
    float3 position;
    float3 direction;
    float t_val;
    int block_id;
};

// CUDA error checking macro
#define CHECK_CUDA(x) TORCH_CHECK(x.device().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

// Utility functions
inline __host__ __device__ float3 make_float3_from_tensor(const torch::Tensor& t, int idx) {
    return make_float3(
        t[idx][0].item<float>(),
        t[idx][1].item<float>(),
        t[idx][2].item<float>()
    );
}

inline void check_cuda_error(const char* msg) {
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "CUDA Error: " << msg << " - " << cudaGetErrorString(err) << std::endl;
        throw std::runtime_error("CUDA Error");
    }
}

/**
 * Block visibility computation
 * Determines which blocks are visible from given camera positions
 */
torch::Tensor block_visibility_cuda(
    torch::Tensor camera_positions,    // [N, 3] Camera positions
    torch::Tensor block_centers,       // [M, 3] Block centers  
    torch::Tensor block_radii,         // [M] Block radii
    torch::Tensor block_active,        // [M] Block active flags
    float visibility_threshold = 0.1f
) {
    CHECK_INPUT(camera_positions);
    CHECK_INPUT(block_centers);
    CHECK_INPUT(block_radii);
    CHECK_INPUT(block_active);
    
    int num_cameras = camera_positions.size(0);
    int num_blocks = block_centers.size(0);
    
    // Create output tensor
    auto visibility_scores = torch::zeros({num_cameras, num_blocks}, 
                                        torch::dtype(torch::kFloat32).device(camera_positions.device()));
    
    // Prepare block info
    std::vector<BlockInfo> h_blocks(num_blocks);
    for (int i = 0; i < num_blocks; i++) {
        h_blocks[i].center = make_float3(
            block_centers[i][0].item<float>(),
            block_centers[i][1].item<float>(),
            block_centers[i][2].item<float>()
        );
        h_blocks[i].radius = block_radii[i].item<float>();
        h_blocks[i].block_id = i;
        h_blocks[i].active = block_active[i].item<bool>();
    }
    
    // Copy to device
    BlockInfo* d_blocks;
    cudaMalloc(&d_blocks, num_blocks * sizeof(BlockInfo));
    cudaMemcpy(d_blocks, h_blocks.data(), num_blocks * sizeof(BlockInfo), cudaMemcpyHostToDevice);
    
    // Launch kernel
    const int block_size = 256;
    const int num_threads = num_cameras * num_blocks;
    const int grid_size = (num_threads + block_size - 1) / block_size;
    
    block_visibility_kernel<<<grid_size, block_size>>>(
        reinterpret_cast<const float3*>(camera_positions.data_ptr<float>()),
        d_blocks,
        visibility_scores.data_ptr<float>(),
        num_cameras,
        num_blocks,
        visibility_threshold
    );
    
    check_cuda_error("block_visibility_kernel");
    cudaFree(d_blocks);
    
    return visibility_scores;
}

/**
 * Block selection for ray tracing
 * Selects relevant blocks for each ray based on spatial intersection
 */
std::vector<torch::Tensor> block_selection_cuda(
    torch::Tensor ray_origins,         // [N, 3] Ray origins
    torch::Tensor ray_directions,      // [N, 3] Ray directions
    torch::Tensor ray_near,            // [N] Near bounds
    torch::Tensor ray_far,             // [N] Far bounds
    torch::Tensor block_centers,       // [M, 3] Block centers
    torch::Tensor block_radii,         // [M] Block radii
    torch::Tensor block_active,        // [M] Block active flags
    int max_blocks_per_ray = 8
) {
    CHECK_INPUT(ray_origins);
    CHECK_INPUT(ray_directions);
    CHECK_INPUT(ray_near);
    CHECK_INPUT(ray_far);
    CHECK_INPUT(block_centers);
    CHECK_INPUT(block_radii);
    CHECK_INPUT(block_active);
    
    int num_rays = ray_origins.size(0);
    int num_blocks = block_centers.size(0);
    
    // Create output tensors
    auto selected_blocks = torch::full({num_rays, max_blocks_per_ray}, -1, 
                                     torch::dtype(torch::kInt32).device(ray_origins.device()));
    auto num_selected_blocks = torch::zeros({num_rays}, 
                                          torch::dtype(torch::kInt32).device(ray_origins.device()));
    
    // Prepare ray data
    std::vector<Ray> h_rays(num_rays);
    for (int i = 0; i < num_rays; i++) {
        h_rays[i].origin = make_float3(
            ray_origins[i][0].item<float>(),
            ray_origins[i][1].item<float>(),
            ray_origins[i][2].item<float>()
        );
        h_rays[i].direction = make_float3(
            ray_directions[i][0].item<float>(),
            ray_directions[i][1].item<float>(),
            ray_directions[i][2].item<float>()
        );
        h_rays[i].near = ray_near[i].item<float>();
        h_rays[i].far = ray_far[i].item<float>();
    }
    
    // Prepare block info
    std::vector<BlockInfo> h_blocks(num_blocks);
    for (int i = 0; i < num_blocks; i++) {
        h_blocks[i].center = make_float3(
            block_centers[i][0].item<float>(),
            block_centers[i][1].item<float>(),
            block_centers[i][2].item<float>()
        );
        h_blocks[i].radius = block_radii[i].item<float>();
        h_blocks[i].block_id = i;
        h_blocks[i].active = block_active[i].item<bool>();
    }
    
    // Copy to device
    Ray* d_rays;
    BlockInfo* d_blocks;
    cudaMalloc(&d_rays, num_rays * sizeof(Ray));
    cudaMalloc(&d_blocks, num_blocks * sizeof(BlockInfo));
    cudaMemcpy(d_rays, h_rays.data(), num_rays * sizeof(Ray), cudaMemcpyHostToDevice);
    cudaMemcpy(d_blocks, h_blocks.data(), num_blocks * sizeof(BlockInfo), cudaMemcpyHostToDevice);
    
    // Launch kernel
    const int block_size = 256;
    const int grid_size = (num_rays + block_size - 1) / block_size;
    
    block_selection_kernel<<<grid_size, block_size>>>(
        d_rays,
        d_blocks,
        selected_blocks.data_ptr<int>(),
        num_selected_blocks.data_ptr<int>(),
        num_rays,
        num_blocks
    );
    
    check_cuda_error("block_selection_kernel");
    cudaFree(d_rays);
    cudaFree(d_blocks);
    
    return {selected_blocks, num_selected_blocks};
}

/**
 * Block volume rendering
 * Performs volume rendering with block-specific density and color
 */
std::vector<torch::Tensor> block_volume_rendering_cuda(
    torch::Tensor sample_positions,    // [N, max_samples, 3] Sample positions
    torch::Tensor sample_directions,   // [N, max_samples, 3] Sample directions  
    torch::Tensor sample_t_vals,       // [N, max_samples] Sample t values
    torch::Tensor sample_block_ids,    // [N, max_samples] Block IDs for samples
    torch::Tensor densities,           // [N, max_samples] Density values
    torch::Tensor colors,              // [N, max_samples, 3] Color values
    torch::Tensor interpolation_weights, // [N, max_samples] Interpolation weights
    torch::Tensor num_samples_per_ray, // [N] Number of samples per ray
    float sigma_factor = 1.0f
) {
    CHECK_INPUT(sample_positions);
    CHECK_INPUT(sample_directions);
    CHECK_INPUT(sample_t_vals);
    CHECK_INPUT(sample_block_ids);
    CHECK_INPUT(densities);
    CHECK_INPUT(colors);
    CHECK_INPUT(interpolation_weights);
    CHECK_INPUT(num_samples_per_ray);
    
    int num_rays = sample_positions.size(0);
    int max_samples = sample_positions.size(1);
    
    // Create output tensors
    auto output_colors = torch::zeros({num_rays, 3}, 
                                    torch::dtype(torch::kFloat32).device(sample_positions.device()));
    auto output_depths = torch::zeros({num_rays}, 
                                    torch::dtype(torch::kFloat32).device(sample_positions.device()));
    auto output_alphas = torch::zeros({num_rays}, 
                                    torch::dtype(torch::kFloat32).device(sample_positions.device()));
    
    // Prepare sample points
    std::vector<SamplePoint> h_sample_points(num_rays * max_samples);
    for (int i = 0; i < num_rays; i++) {
        for (int j = 0; j < max_samples; j++) {
            int idx = i * max_samples + j;
            h_sample_points[idx].position = make_float3(
                sample_positions[i][j][0].item<float>(),
                sample_positions[i][j][1].item<float>(),
                sample_positions[i][j][2].item<float>()
            );
            h_sample_points[idx].direction = make_float3(
                sample_directions[i][j][0].item<float>(),
                sample_directions[i][j][1].item<float>(),
                sample_directions[i][j][2].item<float>()
            );
            h_sample_points[idx].t_val = sample_t_vals[i][j].item<float>();
            h_sample_points[idx].block_id = sample_block_ids[i][j].item<int>();
        }
    }
    
    // Copy to device
    SamplePoint* d_sample_points;
    cudaMalloc(&d_sample_points, num_rays * max_samples * sizeof(SamplePoint));
    cudaMemcpy(d_sample_points, h_sample_points.data(), 
               num_rays * max_samples * sizeof(SamplePoint), cudaMemcpyHostToDevice);
    
    // Launch kernel
    const int block_size = 256;
    const int grid_size = (num_rays + block_size - 1) / block_size;
    
    block_volume_rendering_kernel<<<grid_size, block_size>>>(
        d_sample_points,
        densities.data_ptr<float>(),
        reinterpret_cast<const float3*>(colors.data_ptr<float>()),
        interpolation_weights.data_ptr<float>(),
        num_samples_per_ray.data_ptr<int>(),
        reinterpret_cast<float3*>(output_colors.data_ptr<float>()),
        output_depths.data_ptr<float>(),
        output_alphas.data_ptr<float>(),
        num_rays,
        sigma_factor
    );
    
    check_cuda_error("block_volume_rendering_kernel");
    cudaFree(d_sample_points);
    
    return {output_colors, output_depths, output_alphas};
}

/**
 * Appearance embedding lookup
 * Retrieves appearance embeddings for block-specific appearance control
 */
torch::Tensor appearance_embedding_cuda(
    torch::Tensor appearance_ids,      // [N] Appearance embedding IDs
    torch::Tensor block_ids,           // [N] Block IDs
    torch::Tensor appearance_embeddings, // [num_appearances, embed_dim] Embeddings
    int embed_dim
) {
    CHECK_INPUT(appearance_ids);
    CHECK_INPUT(block_ids);
    CHECK_INPUT(appearance_embeddings);
    
    int num_samples = appearance_ids.size(0);
    int num_appearances = appearance_embeddings.size(0);
    
    // Create output tensor
    auto output_embeddings = torch::zeros({num_samples, embed_dim}, 
                                        torch::dtype(torch::kFloat32).device(appearance_ids.device()));
    
    // Launch kernel
    const int block_size = 256;
    const int grid_size = (num_samples + block_size - 1) / block_size;
    
    appearance_embedding_kernel<<<grid_size, block_size>>>(
        appearance_ids.data_ptr<int>(),
        block_ids.data_ptr<int>(),
        appearance_embeddings.data_ptr<float>(),
        output_embeddings.data_ptr<float>(),
        num_samples,
        embed_dim,
        num_appearances
    );
    
    check_cuda_error("appearance_embedding_kernel");
    
    return output_embeddings;
}

/**
 * Block interpolation weights computation
 * Computes interpolation weights for blending between overlapping blocks
 */
torch::Tensor block_interpolation_weights_cuda(
    torch::Tensor sample_positions,    // [N, 3] Sample positions
    torch::Tensor block_ids,           // [N] Block IDs for each sample
    torch::Tensor block_centers,       // [M, 3] Block centers
    torch::Tensor block_radii,         // [M] Block radii
    float overlap_factor = 2.0f
) {
    CHECK_INPUT(sample_positions);
    CHECK_INPUT(block_ids);
    CHECK_INPUT(block_centers);
    CHECK_INPUT(block_radii);
    
    int num_samples = sample_positions.size(0);
    int num_blocks = block_centers.size(0);
    
    // Create output tensor
    auto interpolation_weights = torch::ones({num_samples}, 
                                           torch::dtype(torch::kFloat32).device(sample_positions.device()));
    
    // Prepare block info
    std::vector<BlockInfo> h_blocks(num_blocks);
    for (int i = 0; i < num_blocks; i++) {
        h_blocks[i].center = make_float3(
            block_centers[i][0].item<float>(),
            block_centers[i][1].item<float>(),
            block_centers[i][2].item<float>()
        );
        h_blocks[i].radius = block_radii[i].item<float>();
        h_blocks[i].block_id = i;
        h_blocks[i].active = true;
    }
    
    // Copy to device
    BlockInfo* d_blocks;
    cudaMalloc(&d_blocks, num_blocks * sizeof(BlockInfo));
    cudaMemcpy(d_blocks, h_blocks.data(), num_blocks * sizeof(BlockInfo), cudaMemcpyHostToDevice);
    
    // Launch kernel
    const int block_size = 256;
    const int grid_size = (num_samples + block_size - 1) / block_size;
    
    block_interpolation_weights_kernel<<<grid_size, block_size>>>(
        reinterpret_cast<const float3*>(sample_positions.data_ptr<float>()),
        block_ids.data_ptr<int>(),
        d_blocks,
        interpolation_weights.data_ptr<float>(),
        num_samples,
        overlap_factor
    );
    
    check_cuda_error("block_interpolation_weights_kernel");
    cudaFree(d_blocks);
    
    return interpolation_weights;
}

/**
 * Memory bandwidth test for performance analysis
 */
torch::Tensor memory_bandwidth_test_cuda(
    torch::Tensor input_data           // [N] Input data for bandwidth test
) {
    CHECK_INPUT(input_data);
    
    int num_elements = input_data.numel();
    
    // Create output tensor
    auto output_data = torch::zeros_like(input_data);
    
    // Launch kernel
    const int block_size = 256;
    const int grid_size = (num_elements + block_size - 1) / block_size;
    
    memory_bandwidth_test_kernel<<<grid_size, block_size>>>(
        input_data.data_ptr<float>(),
        output_data.data_ptr<float>(),
        num_elements
    );
    
    check_cuda_error("memory_bandwidth_test_kernel");
    
    return output_data;
}

// PyBind11 module definition
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("block_visibility", &block_visibility_cuda, 
          "Block-NeRF visibility computation (CUDA)");
    m.def("block_selection", &block_selection_cuda, 
          "Block-NeRF block selection for ray tracing (CUDA)");
    m.def("block_volume_rendering", &block_volume_rendering_cuda, 
          "Block-NeRF volume rendering (CUDA)");
    m.def("appearance_embedding", &appearance_embedding_cuda, 
          "Block-NeRF appearance embedding lookup (CUDA)");
    m.def("block_interpolation_weights", &block_interpolation_weights_cuda, 
          "Block-NeRF interpolation weights computation (CUDA)");
    m.def("memory_bandwidth_test", &memory_bandwidth_test_cuda, 
          "Memory bandwidth test for performance analysis (CUDA)");
}
