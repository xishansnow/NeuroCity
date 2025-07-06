/**
 * Block-NeRF CUDA Header File
 * 
 * Based on "Block-NeRF: Scalable Large Scene Neural View Synthesis" (CVPR 2022)
 * Optimized for GTX 1080 Ti (Compute Capability 6.1)
 */

#ifndef BLOCK_NERF_CUDA_H
#define BLOCK_NERF_CUDA_H

#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>

// Forward declarations for CUDA functions
torch::Tensor block_visibility_cuda(
    torch::Tensor camera_positions,
    torch::Tensor block_centers,
    torch::Tensor block_radii,
    torch::Tensor block_active,
    float visibility_threshold
);

std::vector<torch::Tensor> block_selection_cuda(
    torch::Tensor ray_origins,
    torch::Tensor ray_directions,
    torch::Tensor ray_near,
    torch::Tensor ray_far,
    torch::Tensor block_centers,
    torch::Tensor block_radii,
    torch::Tensor block_active,
    int max_blocks_per_ray
);

std::vector<torch::Tensor> block_volume_rendering_cuda(
    torch::Tensor sample_positions,
    torch::Tensor sample_directions,
    torch::Tensor sample_t_vals,
    torch::Tensor sample_block_ids,
    torch::Tensor densities,
    torch::Tensor colors,
    torch::Tensor interpolation_weights,
    torch::Tensor num_samples_per_ray,
    float sigma_factor
);

torch::Tensor appearance_embedding_cuda(
    torch::Tensor appearance_ids,
    torch::Tensor block_ids,
    torch::Tensor appearance_embeddings,
    int embed_dim
);

torch::Tensor block_interpolation_weights_cuda(
    torch::Tensor sample_positions,
    torch::Tensor block_ids,
    torch::Tensor block_centers,
    torch::Tensor block_radii,
    float overlap_factor
);

torch::Tensor memory_bandwidth_test_cuda(
    torch::Tensor input_data
);

#endif // BLOCK_NERF_CUDA_H
