#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include "instant_ngp_cuda.h"

// C++ implementations that call CUDA kernels
torch::Tensor hash_encode_forward_cuda(
    torch::Tensor positions,
    torch::Tensor embeddings,
    torch::Tensor resolutions,
    torch::Tensor offsets,
    int num_levels,
    int feature_dim,
    uint32_t hashmap_size,
    float scale,
    torch::Tensor aabb_min,
    torch::Tensor aabb_max
) {
    // Ensure inputs are on CUDA and contiguous
    positions = positions.to(torch::kCUDA).contiguous();
    embeddings = embeddings.to(torch::kCUDA).contiguous();
    resolutions = resolutions.to(torch::kCUDA).contiguous();
    offsets = offsets.to(torch::kCUDA).contiguous();
    aabb_min = aabb_min.to(torch::kCUDA).contiguous();
    aabb_max = aabb_max.to(torch::kCUDA).contiguous();
    
    const int N = positions.size(0);
    auto encoded = torch::zeros({N, num_levels * feature_dim}, 
                               torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA));
    
    // Extract aabb values
    float3 aabb_min_val = make_float3(
        aabb_min[0].item<float>(),
        aabb_min[1].item<float>(),
        aabb_min[2].item<float>()
    );
    float3 aabb_max_val = make_float3(
        aabb_max[0].item<float>(),
        aabb_max[1].item<float>(),
        aabb_max[2].item<float>()
    );
    
    hash_encode_forward_wrapper(
        positions.data_ptr<float>(),
        embeddings.data_ptr<float>(),
        encoded.data_ptr<float>(),
        resolutions.data_ptr<int>(),
        offsets.data_ptr<uint32_t>(),
        N,
        num_levels,
        feature_dim,
        hashmap_size,
        scale,
        aabb_min_val,
        aabb_max_val
    );
    
    // Check for CUDA errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA kernel error in hash_encode_forward: %s\n", cudaGetErrorString(err));
    }
    
    return encoded;
}

torch::Tensor hash_encode_backward_cuda(
    torch::Tensor positions,
    torch::Tensor grad_encoded,
    torch::Tensor embeddings_shape,
    torch::Tensor resolutions,
    torch::Tensor offsets,
    int num_levels,
    int feature_dim,
    uint32_t hashmap_size,
    float scale,
    torch::Tensor aabb_min,
    torch::Tensor aabb_max
) {
    // Ensure inputs are on CUDA and contiguous
    positions = positions.to(torch::kCUDA).contiguous();
    grad_encoded = grad_encoded.to(torch::kCUDA).contiguous();
    resolutions = resolutions.to(torch::kCUDA).contiguous();
    offsets = offsets.to(torch::kCUDA).contiguous();
    aabb_min = aabb_min.to(torch::kCUDA).contiguous();
    aabb_max = aabb_max.to(torch::kCUDA).contiguous();
    
    const int N = positions.size(0);
    
    // Create grad_embeddings tensor with proper shape
    std::vector<int64_t> shape_vec;
    for (int i = 0; i < embeddings_shape.size(0); i++) {
        shape_vec.push_back(embeddings_shape[i].item<int64_t>());
    }
    auto grad_embeddings = torch::zeros(shape_vec, 
                                       torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA));
    
    // Extract aabb values
    float3 aabb_min_val = make_float3(
        aabb_min[0].item<float>(),
        aabb_min[1].item<float>(),
        aabb_min[2].item<float>()
    );
    float3 aabb_max_val = make_float3(
        aabb_max[0].item<float>(),
        aabb_max[1].item<float>(),
        aabb_max[2].item<float>()
    );
    
    hash_encode_backward_wrapper(
        positions.data_ptr<float>(),
        grad_encoded.data_ptr<float>(),
        grad_embeddings.data_ptr<float>(),
        resolutions.data_ptr<int>(),
        offsets.data_ptr<uint32_t>(),
        N,
        num_levels,
        feature_dim,
        hashmap_size,
        scale,
        aabb_min_val,
        aabb_max_val
    );
    
    // Check for CUDA errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA kernel error in hash_encode_backward: %s\n", cudaGetErrorString(err));
    }
    
    return grad_embeddings;
}

torch::Tensor sh_encode_cuda(
    torch::Tensor directions,
    int degree
) {
    // Ensure inputs are on CUDA and contiguous
    directions = directions.to(torch::kCUDA).contiguous();
    
    const int N = directions.size(0);
    const int sh_dim = (degree + 1) * (degree + 1);
    auto encoded = torch::zeros({N, sh_dim}, 
                               torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA));
    
    sh_encode_wrapper(
        directions.data_ptr<float>(),
        encoded.data_ptr<float>(),
        N,
        degree
    );
    
    // Check for CUDA errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA kernel error in sh_encode: %s\n", cudaGetErrorString(err));
    }
    
    return encoded;
}

// Python binding
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.doc() = "Instant NGP CUDA extensions for GTX 1080 Ti";
    
    m.def("hash_encode_forward", &hash_encode_forward_cuda, 
          "Hash encoding forward pass (CUDA)");
    
    m.def("hash_encode_backward", &hash_encode_backward_cuda, 
          "Hash encoding backward pass (CUDA)");
    
    m.def("sh_encode", &sh_encode_cuda, 
          "Spherical harmonics encoding (CUDA)");
}
