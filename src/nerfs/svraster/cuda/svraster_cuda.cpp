#include <torch/extension.h>
#include <cuda_runtime.h>
#include "svraster_cuda_kernel.h"

// Error checking macro
#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            printf("CUDA error at %s:%d - %s\n", __FILE__, __LINE__, \
                   cudaGetErrorString(err)); \
            throw std::runtime_error("CUDA error"); \
        } \
    } while(0)

// Convert torch tensors to CUDA data structures
torch::Tensor ray_voxel_intersection_cuda(
    torch::Tensor ray_origins,
    torch::Tensor ray_directions,
    torch::Tensor voxel_positions,
    torch::Tensor voxel_sizes,
    torch::Tensor voxel_densities,
    torch::Tensor voxel_colors
) {
    // Get tensor dimensions
    int num_rays = ray_origins.size(0);
    int num_voxels = voxel_positions.size(0);
    int max_intersections = 100;  // Configurable
    
    // Ensure tensors are on GPU and contiguous
    ray_origins = ray_origins.contiguous().cuda();
    ray_directions = ray_directions.contiguous().cuda();
    voxel_positions = voxel_positions.contiguous().cuda();
    voxel_sizes = voxel_sizes.contiguous().cuda();
    voxel_densities = voxel_densities.contiguous().cuda();
    voxel_colors = voxel_colors.contiguous().cuda();
    
    // Allocate output tensors
    auto intersection_counts = torch::zeros({num_rays}, 
        torch::dtype(torch::kInt32).device(torch::kCUDA));
    auto intersection_indices = torch::zeros({num_rays, max_intersections}, 
        torch::dtype(torch::kInt32).device(torch::kCUDA));
    auto intersection_t_near = torch::zeros({num_rays, max_intersections}, 
        torch::dtype(torch::kFloat32).device(torch::kCUDA));
    auto intersection_t_far = torch::zeros({num_rays, max_intersections}, 
        torch::dtype(torch::kFloat32).device(torch::kCUDA));
    
    // Convert ray data to CUDA format
    Ray* d_rays;
    CUDA_CHECK(cudaMalloc(&d_rays, num_rays * sizeof(Ray)));
    
    // Copy ray data
    for (int i = 0; i < num_rays; i++) {
        Ray ray;
        ray.origin = make_float3(
            ray_origins[i][0].item<float>(),
            ray_origins[i][1].item<float>(),
            ray_origins[i][2].item<float>()
        );
        ray.direction = make_float3(
            ray_directions[i][0].item<float>(),
            ray_directions[i][1].item<float>(),
            ray_directions[i][2].item<float>()
        );
        CUDA_CHECK(cudaMemcpy(d_rays + i, &ray, sizeof(Ray), cudaMemcpyHostToDevice));
    }
    
    // Convert voxel data to CUDA format
    Voxel* d_voxels;
    CUDA_CHECK(cudaMalloc(&d_voxels, num_voxels * sizeof(Voxel)));
    
    // Copy voxel data
    for (int i = 0; i < num_voxels; i++) {
        Voxel voxel;
        voxel.position = make_float3(
            voxel_positions[i][0].item<float>(),
            voxel_positions[i][1].item<float>(),
            voxel_positions[i][2].item<float>()
        );
        voxel.size = voxel_sizes[i].item<float>();
        voxel.density = voxel_densities[i].item<float>();
        voxel.color = make_float3(
            voxel_colors[i][0].item<float>(),
            voxel_colors[i][1].item<float>(),
            voxel_colors[i][2].item<float>()
        );
        voxel.level = 0;  // Default level
        voxel.morton_code = 0;  // Will be computed later
        
        CUDA_CHECK(cudaMemcpy(d_voxels + i, &voxel, sizeof(Voxel), cudaMemcpyHostToDevice));
    }
    
    // Launch kernel
    int block_size = 256;
    int grid_size = (num_rays + block_size - 1) / block_size;
    
    ray_voxel_intersection_kernel<<<grid_size, block_size>>>(
        d_rays,
        d_voxels,
        (int*)intersection_counts.data_ptr(),
        (int*)intersection_indices.data_ptr(),
        (float*)intersection_t_near.data_ptr(),
        (float*)intersection_t_far.data_ptr(),
        num_rays,
        num_voxels,
        max_intersections
    );
    
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    
    // Cleanup
    cudaFree(d_rays);
    cudaFree(d_voxels);
    
    // Return results as a tuple
    return torch::make_tuple(intersection_counts, intersection_indices, 
                           intersection_t_near, intersection_t_far);
}

torch::Tensor voxel_rasterization_cuda(
    torch::Tensor ray_origins,
    torch::Tensor ray_directions,
    torch::Tensor voxel_positions,
    torch::Tensor voxel_sizes,
    torch::Tensor voxel_densities,
    torch::Tensor voxel_colors,
    torch::Tensor intersection_counts,
    torch::Tensor intersection_indices,
    torch::Tensor intersection_t_near,
    torch::Tensor intersection_t_far
) {
    int num_rays = ray_origins.size(0);
    int max_intersections = intersection_indices.size(1);
    
    // Ensure tensors are on GPU and contiguous
    ray_origins = ray_origins.contiguous().cuda();
    ray_directions = ray_directions.contiguous().cuda();
    voxel_positions = voxel_positions.contiguous().cuda();
    voxel_sizes = voxel_sizes.contiguous().cuda();
    voxel_densities = voxel_densities.contiguous().cuda();
    voxel_colors = voxel_colors.contiguous().cuda();
    intersection_counts = intersection_counts.contiguous().cuda();
    intersection_indices = intersection_indices.contiguous().cuda();
    intersection_t_near = intersection_t_near.contiguous().cuda();
    intersection_t_far = intersection_t_far.contiguous().cuda();
    
    // Allocate output tensors
    auto output_colors = torch::zeros({num_rays, 3}, 
        torch::dtype(torch::kFloat32).device(torch::kCUDA));
    auto output_depths = torch::zeros({num_rays}, 
        torch::dtype(torch::kFloat32).device(torch::kCUDA));
    
    // Convert ray data to CUDA format
    Ray* d_rays;
    CUDA_CHECK(cudaMalloc(&d_rays, num_rays * sizeof(Ray)));
    
    // Copy ray data
    for (int i = 0; i < num_rays; i++) {
        Ray ray;
        ray.origin = make_float3(
            ray_origins[i][0].item<float>(),
            ray_origins[i][1].item<float>(),
            ray_origins[i][2].item<float>()
        );
        ray.direction = make_float3(
            ray_directions[i][0].item<float>(),
            ray_directions[i][1].item<float>(),
            ray_directions[i][2].item<float>()
        );
        CUDA_CHECK(cudaMemcpy(d_rays + i, &ray, sizeof(Ray), cudaMemcpyHostToDevice));
    }
    
    // Convert voxel data to CUDA format
    Voxel* d_voxels;
    int num_voxels = voxel_positions.size(0);
    CUDA_CHECK(cudaMalloc(&d_voxels, num_voxels * sizeof(Voxel)));
    
    // Copy voxel data
    for (int i = 0; i < num_voxels; i++) {
        Voxel voxel;
        voxel.position = make_float3(
            voxel_positions[i][0].item<float>(),
            voxel_positions[i][1].item<float>(),
            voxel_positions[i][2].item<float>()
        );
        voxel.size = voxel_sizes[i].item<float>();
        voxel.density = voxel_densities[i].item<float>();
        voxel.color = make_float3(
            voxel_colors[i][0].item<float>(),
            voxel_colors[i][1].item<float>(),
            voxel_colors[i][2].item<float>()
        );
        voxel.level = 0;
        voxel.morton_code = 0;
        
        CUDA_CHECK(cudaMemcpy(d_voxels + i, &voxel, sizeof(Voxel), cudaMemcpyHostToDevice));
    }
    
    // Launch kernel
    int block_size = 256;
    int grid_size = (num_rays + block_size - 1) / block_size;
    
    float3 background_color = make_float3(0.0f, 0.0f, 0.0f);
    
    voxel_rasterization_kernel<<<grid_size, block_size>>>(
        d_rays,
        d_voxels,
        (int*)intersection_counts.data_ptr(),
        (int*)intersection_indices.data_ptr(),
        (float*)intersection_t_near.data_ptr(),
        (float*)intersection_t_far.data_ptr(),
        (float3*)output_colors.data_ptr(),
        (float*)output_depths.data_ptr(),
        num_rays,
        max_intersections,
        background_color
    );
    
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    
    // Cleanup
    cudaFree(d_rays);
    cudaFree(d_voxels);
    
    return torch::make_tuple(output_colors, output_depths);
}

torch::Tensor compute_morton_codes_cuda(
    torch::Tensor voxel_positions,
    torch::Tensor scene_bounds
) {
    int num_voxels = voxel_positions.size(0);
    
    // Ensure tensors are on GPU and contiguous
    voxel_positions = voxel_positions.contiguous().cuda();
    scene_bounds = scene_bounds.contiguous().cuda();
    
    // Convert voxel data to CUDA format
    Voxel* d_voxels;
    CUDA_CHECK(cudaMalloc(&d_voxels, num_voxels * sizeof(Voxel)));
    
    // Copy voxel data
    for (int i = 0; i < num_voxels; i++) {
        Voxel voxel;
        voxel.position = make_float3(
            voxel_positions[i][0].item<float>(),
            voxel_positions[i][1].item<float>(),
            voxel_positions[i][2].item<float>()
        );
        voxel.size = 0.1f;  // Default size
        voxel.density = 0.0f;
        voxel.color = make_float3(0.5f, 0.5f, 0.5f);
        voxel.level = 0;
        voxel.morton_code = 0;
        
        CUDA_CHECK(cudaMemcpy(d_voxels + i, &voxel, sizeof(Voxel), cudaMemcpyHostToDevice));
    }
    
    // Extract scene bounds
    float3 scene_min = make_float3(
        scene_bounds[0].item<float>(),
        scene_bounds[1].item<float>(),
        scene_bounds[2].item<float>()
    );
    float3 scene_max = make_float3(
        scene_bounds[3].item<float>(),
        scene_bounds[4].item<float>(),
        scene_bounds[5].item<float>()
    );
    float3 scene_size = scene_max - scene_min;
    
    // Launch kernel
    int block_size = 256;
    int grid_size = (num_voxels + block_size - 1) / block_size;
    
    compute_morton_codes_kernel<<<grid_size, block_size>>>(
        d_voxels, scene_min, scene_size, num_voxels
    );
    
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    
    // Copy results back
    auto morton_codes = torch::zeros({num_voxels}, 
        torch::dtype(torch::kInt32).device(torch::kCUDA));
    
    for (int i = 0; i < num_voxels; i++) {
        Voxel voxel;
        CUDA_CHECK(cudaMemcpy(&voxel, d_voxels + i, sizeof(Voxel), cudaMemcpyDeviceToHost));
        morton_codes[i] = voxel.morton_code;
    }
    
    // Cleanup
    cudaFree(d_voxels);
    
    return morton_codes;
}

torch::Tensor adaptive_subdivision_cuda(
    torch::Tensor voxel_positions,
    torch::Tensor voxel_sizes,
    torch::Tensor voxel_densities,
    torch::Tensor voxel_colors,
    torch::Tensor voxel_gradients,
    float subdivision_threshold,
    int max_level
) {
    int num_voxels = voxel_positions.size(0);
    
    // Ensure tensors are on GPU and contiguous
    voxel_positions = voxel_positions.contiguous().cuda();
    voxel_sizes = voxel_sizes.contiguous().cuda();
    voxel_densities = voxel_densities.contiguous().cuda();
    voxel_colors = voxel_colors.contiguous().cuda();
    voxel_gradients = voxel_gradients.contiguous().cuda();
    
    // Convert voxel data to CUDA format
    Voxel* d_voxels;
    CUDA_CHECK(cudaMalloc(&d_voxels, num_voxels * sizeof(Voxel)));
    
    // Copy voxel data
    for (int i = 0; i < num_voxels; i++) {
        Voxel voxel;
        voxel.position = make_float3(
            voxel_positions[i][0].item<float>(),
            voxel_positions[i][1].item<float>(),
            voxel_positions[i][2].item<float>()
        );
        voxel.size = voxel_sizes[i].item<float>();
        voxel.density = voxel_densities[i].item<float>();
        voxel.color = make_float3(
            voxel_colors[i][0].item<float>(),
            voxel_colors[i][1].item<float>(),
            voxel_colors[i][2].item<float>()
        );
        voxel.level = 0;  // Default level
        voxel.morton_code = 0;
        
        CUDA_CHECK(cudaMemcpy(d_voxels + i, &voxel, sizeof(Voxel), cudaMemcpyHostToDevice));
    }
    
    // Allocate subdivision flags
    int* d_subdivision_flags;
    CUDA_CHECK(cudaMalloc(&d_subdivision_flags, num_voxels * sizeof(int)));
    
    int* d_new_voxel_count;
    CUDA_CHECK(cudaMalloc(&d_new_voxel_count, sizeof(int)));
    CUDA_CHECK(cudaMemset(d_new_voxel_count, 0, sizeof(int)));
    
    // Launch kernel
    int block_size = 256;
    int grid_size = (num_voxels + block_size - 1) / block_size;
    
    adaptive_subdivision_kernel<<<grid_size, block_size>>>(
        d_voxels,
        (float*)voxel_gradients.data_ptr(),
        d_subdivision_flags,
        d_new_voxel_count,
        subdivision_threshold,
        max_level,
        num_voxels
    );
    
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    
    // Get new voxel count
    int new_voxel_count;
    CUDA_CHECK(cudaMemcpy(&new_voxel_count, d_new_voxel_count, sizeof(int), cudaMemcpyDeviceToHost));
    
    // Copy subdivision flags back
    auto subdivision_flags = torch::zeros({num_voxels}, 
        torch::dtype(torch::kInt32).device(torch::kCUDA));
    CUDA_CHECK(cudaMemcpy(subdivision_flags.data_ptr(), d_subdivision_flags, 
                         num_voxels * sizeof(int), cudaMemcpyDeviceToHost));
    
    // Cleanup
    cudaFree(d_voxels);
    cudaFree(d_subdivision_flags);
    cudaFree(d_new_voxel_count);
    
    return torch::make_tuple(subdivision_flags, new_voxel_count);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("ray_voxel_intersection", &ray_voxel_intersection_cuda, "Ray-voxel intersection (CUDA)");
    m.def("voxel_rasterization", &voxel_rasterization_cuda, "Voxel rasterization (CUDA)");
    m.def("compute_morton_codes", &compute_morton_codes_cuda, "Compute Morton codes (CUDA)");
    m.def("adaptive_subdivision", &adaptive_subdivision_cuda, "Adaptive subdivision (CUDA)");
} 