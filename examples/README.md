# NeRF Examples

This directory contains example usage scripts for various NeRF implementations in the NeuroCity project.

## Available Examples

### Core NeRF Variants

### 1. Classic NeRF (`classic_nerf_example.py`)
- **Description**: Original NeRF implementation for novel view synthesis
- **Best for**: Small-scale objects and scenes, research baselines
- **Key features**: MLP-based implicit representation, positional encoding
- **Usage**: `python classic_nerf_example.py --example basic`

### 2. Grid-NeRF (`grid_nerf_example.py`)  
- **Description**: Grid-guided NeRF for large urban scenes
- **Best for**: Large-scale urban environments, city-scale reconstruction
- **Key features**: Hierarchical voxel grids, spatial guidance
- **Usage**: `python grid_nerf_example.py --example kitti360 --data_path /path/to/kitti360`

### 3. Instant-NGP (`instant_ngp_example.py`)
- **Description**: Ultra-fast NeRF training with hash encoding
- **Best for**: Quick prototyping, real-time applications
- **Key features**: Multiresolution hash encoding, CUDA optimization
- **Usage**: `python instant_ngp_example.py --example synthetic`

### 4. Nerfacto (`nerfacto_example.py`)
- **Description**: Streamlined NeRF with improved efficiency
- **Best for**: General-purpose NeRF applications, production use
- **Key features**: Proposal networks, hash encoding, camera optimization
- **Usage**: `python nerfacto_example.py --example real_world --data_path /path/to/data`

### 5. Plenoxels (`plenoxels_example.py`)
- **Description**: Sparse voxel grid representation without neural networks
- **Best for**: Fast training, memory-efficient rendering
- **Key features**: Spherical harmonics, sparse voxel grids, no MLPs
- **Usage**: `python plenoxels_example.py --example synthetic`

### 6. SVRaster (`svraster_example.py`)
- **Description**: Sparse voxel rasterization for efficient rendering
- **Best for**: Real-time applications, large scenes
- **Key features**: Octree-based voxels, adaptive subdivision
- **Usage**: `python svraster_example.py --example train --data_path /path/to/data`

### Advanced NeRF Variants

### 7. Mega-NeRF (`mega_nerf_example.py`)
- **Description**: Large-scale scene reconstruction with spatial decomposition
- **Best for**: City-scale and very large environments
- **Key features**: Spatial clustering, multiple sub-NeRFs
- **Usage**: `python mega_nerf_example.py --example city_scale --data_path /path/to/city_data`

### 8. Mega-NeRF+ (`mega_nerf_plus_example.py`)
- **Description**: Enhanced Mega-NeRF with improved clustering
- **Best for**: Large-scale scenes requiring high quality
- **Key features**: Adaptive clustering, attention mechanisms, feature fusion
- **Usage**: `python mega_nerf_plus_example.py --example enhanced_large_scale`

### 9. Block-NeRF (`block_nerf_example.py`)
- **Description**: City-scale reconstruction with block decomposition
- **Best for**: Very large urban environments, city blocks
- **Key features**: Block-based decomposition, appearance embeddings
- **Usage**: `python block_nerf_example.py --example city_scale`

### 10. Bungee-NeRF (`bungee_nerf_example.py`)
- **Description**: Progressive neural radiance fields with adaptive sampling
- **Best for**: Training efficiency, progressive quality improvement
- **Key features**: Multi-stage training, adaptive sampling
- **Usage**: `python bungee_nerf_example.py --example progressive`

### 11. Mip-NeRF (`mip_nerf_example.py`)
- **Description**: Multiscale neural radiance fields with anti-aliasing
- **Best for**: High-quality rendering, anti-aliasing
- **Key features**: Integrated positional encoding, multiscale representation
- **Usage**: `python mip_nerf_example.py --example anti_aliasing`

### 12. Pyramid-NeRF (`pyramid_nerf_example.py`)
- **Description**: Hierarchical neural radiance fields with pyramid sampling
- **Best for**: Multi-resolution scenes, hierarchical modeling
- **Key features**: Pyramid structure, hierarchical sampling
- **Usage**: `python pyramid_nerf_example.py --example hierarchical`

### Specialized NeRF Variants

### 13. CNC-NeRF (`cnc_nerf_example.py`)
- **Description**: Controllable neural radiance fields
- **Best for**: Interactive rendering, conditional generation
- **Key features**: Control codes, conditional rendering, style transfer
- **Usage**: `python cnc_nerf_example.py --example controllable`

### 14. DNMP-NeRF (`dnmp_nerf_example.py`)
- **Description**: Dynamic neural radiance fields with motion modeling
- **Best for**: Dynamic scenes, temporal modeling
- **Key features**: Temporal encoding, motion fields, deformation modeling
- **Usage**: `python dnmp_nerf_example.py --example dynamic_scene`

### 15. Inf-NeRF (`inf_nerf_example.py`)
- **Description**: Infinite neural radiance fields for unbounded scenes
- **Best for**: Outdoor scenes, unbounded environments
- **Key features**: Scene contraction, infinite backgrounds
- **Usage**: `python inf_nerf_example.py --example unbounded_scene`

### 3D Representation Methods

### 16. SDF-Net (`sdf_net_example.py`)
- **Description**: Signed distance field learning for surface reconstruction
- **Best for**: Surface reconstruction, 3D shape modeling
- **Key features**: SDF representation, geometric initialization, Eikonal loss
- **Usage**: `python sdf_net_example.py --example surface_reconstruction`

### 17. Occupancy Networks (`occupancy_net_example.py`)
- **Description**: 3D shape representation with occupancy fields
- **Best for**: 3D shape completion, mesh generation
- **Key features**: Occupancy prediction, mesh extraction
- **Usage**: `python occupancy_net_example.py --example shape_reconstruction`

## Unified Runner

Use the unified runner to easily execute any example:

```bash
# Basic usage
python examples/run_example.py <method> --example_type basic

# With custom data path
python examples/run_example.py <method> --example_type <type> --data_path /path/to/data --output_dir /path/to/output

# Examples
python examples/run_example.py classic_nerf --example_type basic
python examples/run_example.py instant_ngp --example_type synthetic --data_path ./data/nerf_synthetic/lego
python examples/run_example.py grid_nerf --example_type kitti360 --data_path ./data/kitti360
```

## Quick Start

1. **Install dependencies**:
   ```bash
   pip install torch torchvision torchaudio
   pip install numpy opencv-python pillow
   pip install matplotlib tqdm tensorboard
   ```

2. **Download sample data**:
   ```bash
   # For synthetic scenes
   wget http://cseweb.ucsd.edu/~viscomp/projects/LF/papers/ECCV20/nerf/nerf_synthetic.zip
   unzip nerf_synthetic.zip
   
   # For real scenes (LLFF format)
   wget http://cseweb.ucsd.edu/~viscomp/projects/LF/papers/ECCV20/nerf/nerf_llff_data.zip  
   unzip nerf_llff_data.zip
   ```

3. **Run a basic example**:
   ```bash
   cd /home/xishansnow/uav_planner_ws/NeuroCity
   python examples/instant_ngp_example.py --example basic
   ```

## Data Formats

### Synthetic Data (Blender format)
```