#include <torch/extension.h>
#include <vector>

// Forward declarations
torch::Tensor ray_voxel_intersect_cuda(
    torch::Tensor ray_origins,
    torch::Tensor ray_directions,
    torch::Tensor voxel_centers,
    float voxel_size
);

std::vector<torch::Tensor> volume_render_cuda(
    torch::Tensor densities,
    torch::Tensor colors,
    torch::Tensor deltas
);

torch::Tensor trilinear_interpolate_cuda(
    torch::Tensor features,
    torch::Tensor points,
    torch::Tensor voxel_indices,
    torch::Tensor weights
);

torch::Tensor compute_interpolation_weights_cuda(
    torch::Tensor points,
    torch::Tensor voxel_coords
);

// Python binding
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.doc() = "Plenoxels CUDA extensions";
    
    m.def("ray_voxel_intersect", &ray_voxel_intersect_cuda, 
          "Ray-voxel intersection (CUDA)");
    
    m.def("volume_render", &volume_render_cuda, 
          "Volume rendering (CUDA)");
    
    m.def("trilinear_interpolate", &trilinear_interpolate_cuda, 
          "Trilinear interpolation (CUDA)");
    
    m.def("compute_interpolation_weights", &compute_interpolation_weights_cuda, 
          "Compute interpolation weights (CUDA)");
}
