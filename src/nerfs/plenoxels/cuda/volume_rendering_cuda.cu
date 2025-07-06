#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// CUDA kernel for volume rendering
__global__ void volume_render_kernel(
    const float* __restrict__ densities,    // [N, S]
    const float* __restrict__ colors,       // [N, S, 3]
    const float* __restrict__ deltas,       // [N, S]
    float* __restrict__ weights,            // [N, S]
    float* __restrict__ rgb,                // [N, 3]
    float* __restrict__ depth,              // [N]
    const int N,
    const int S
) {
    const int ray_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (ray_idx >= N) return;
    
    // Initialize accumulation variables
    float transmittance = 1.0f;
    float3 acc_color = make_float3(0.0f, 0.0f, 0.0f);
    float acc_depth = 0.0f;
    
    // Compute alpha values and accumulate colors
    for (int i = 0; i < S; i++) {
        const float density = densities[ray_idx * S + i];
        const float delta = deltas[ray_idx * S + i];
        const float alpha = 1.0f - __expf(-density * delta);
        const float weight = alpha * transmittance;
        
        // Store weights
        weights[ray_idx * S + i] = weight;
        
        // Accumulate color and depth
        const int color_offset = ray_idx * S * 3 + i * 3;
        acc_color.x += weight * colors[color_offset];
        acc_color.y += weight * colors[color_offset + 1];
        acc_color.z += weight * colors[color_offset + 2];
        
        acc_depth += weight * delta;
        
        // Update transmittance
        transmittance *= (1.0f - alpha);
    }
    
    // Store final results
    rgb[ray_idx * 3] = acc_color.x;
    rgb[ray_idx * 3 + 1] = acc_color.y;
    rgb[ray_idx * 3 + 2] = acc_color.z;
    depth[ray_idx] = acc_depth;
}

// C++ interface
std::vector<torch::Tensor> volume_render_cuda(
    torch::Tensor densities,
    torch::Tensor colors,
    torch::Tensor deltas
) {
    // Ensure input tensors are on CUDA and contiguous
    densities = densities.to(torch::kCUDA).contiguous();
    colors = colors.to(torch::kCUDA).contiguous();
    deltas = deltas.to(torch::kCUDA).contiguous();
    
    const int N = densities.size(0);  // Number of rays
    const int S = densities.size(1);  // Samples per ray
    
    auto weights = torch::zeros({N, S}, densities.options());
    auto rgb = torch::zeros({N, 3}, densities.options());
    auto depth = torch::zeros({N}, densities.options());
    
    const int threads = 256;
    const int blocks = (N + threads - 1) / threads;
    
    volume_render_kernel<<<blocks, threads>>>(
        densities.data_ptr<float>(),
        colors.data_ptr<float>(),
        deltas.data_ptr<float>(),
        weights.data_ptr<float>(),
        rgb.data_ptr<float>(),
        depth.data_ptr<float>(),
        N,
        S
    );
    
    // Check for CUDA errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA kernel error: %s\n", cudaGetErrorString(err));
    }
    
    return {rgb, depth, weights};
} 