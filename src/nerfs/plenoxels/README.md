# Plenoxels: Radiance Fields without Neural Networks

This package implements **Plenoxels**, a revolutionary approach to neural radiance fields that replaces neural networks with sparse voxel grids and spherical harmonics. Based on the paper "Plenoxels: Radiance Fields without Neural Networks" by Alex Yu et al.

## Overview

Plenoxels represents a paradigm shift in neural radiance field (NeRF) methods by:

- **No Neural Networks**: Uses sparse voxel grids instead of MLPs
- **Spherical Harmonics**: Represents view-dependent colors with SH coefficients
- **Fast Training**: Achieves 100x speedup compared to vanilla NeRF
- **High Quality**: Maintains comparable or superior rendering quality
- **Memory Efficient**: Sparse representation reduces memory usage

## Key Features

### 🚀 Fast Training
- Direct optimization of voxel parameters
- No forward/backward passes through neural networks
- Coarse-to-fine training strategy

### 🎯 High Quality Rendering
- Trilinear interpolation for smooth sampling
- Spherical harmonics for view-dependent appearance
- Volume rendering with proper alpha compositing

### 💾 Memory Efficient
- Sparse voxel grid representation
- Automatic pruning of low-density voxels
- Configurable resolution levels

### 🔧 Flexible Configuration
- Multiple dataset formats supported
- Customizable training parameters
- Easy integration with existing pipelines

## Architecture

```
Plenoxels Architecture:
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Input Rays    │ -> │   Voxel Grid     │ -> │ Volume Rendering│
│  (origins,dirs) │    │ (density + SH)   │    │   (RGB, depth)  │
└─────────────────┘    └──────────────────┘    └─────────────────┘
                              │
                              ▼
                    ┌──────────────────┐
                    │ Trilinear Interp │
                    │ + SH Evaluation  │
                    └──────────────────┘
```

## Installation

```bash
# Install dependencies
pip install torch torchvision numpy opencv-python imageio tqdm tensorboard

# Install the package
cd NeuroCity/src/plenoxels
python -m pip install -e .
```

## Quick Start

### Basic Usage

```python
from src.plenoxels import PlenoxelConfig, PlenoxelModel

# Create model configuration
config = PlenoxelConfig(
    grid_resolution=(256, 256, 256),
    scene_bounds=(-1.0, -1.0, -1.0, 1.0, 1.0, 1.0),
    sh_degree=2,
    use_coarse_to_fine=True
)

# Initialize model
model = PlenoxelModel(config)

# Forward pass
outputs = model(ray_origins, ray_directions)
rgb = outputs['rgb']      # Rendered colors
depth = outputs['depth']  # Depth values
```

### Training Example

```python
from src.plenoxels import (
    PlenoxelConfig, PlenoxelDatasetConfig, PlenoxelTrainerConfig,
    create_plenoxel_trainer
)

# Configuration
model_config = PlenoxelConfig(
    grid_resolution=(256, 256, 256),
    sh_degree=2,
    use_coarse_to_fine=True
)

dataset_config = PlenoxelDatasetConfig(
    data_dir="data/nerf_synthetic/lego",
    dataset_type="blender",
    num_rays_train=1024
)

trainer_config = PlenoxelTrainerConfig(
    max_epochs=10000,
    learning_rate=0.1,
    experiment_name="plenoxel_lego"
)

# Train model
trainer = create_plenoxel_trainer(model_config, trainer_config, dataset_config)
trainer.train()
```

## Dataset Support

### Blender Synthetic Dataset
```python
dataset_config = PlenoxelDatasetConfig(
    data_dir="path/to/nerf_synthetic/scene",
    dataset_type="blender",
    white_background=True,
    downsample_factor=1
)
```

### COLMAP Real Dataset
```python
dataset_config = PlenoxelDatasetConfig(
    data_dir="path/to/colmap/scene",
    dataset_type="colmap",
    downsample_factor=4
)
```

### LLFF Forward-Facing
```python
dataset_config = PlenoxelDatasetConfig(
    data_dir="path/to/llff/scene",
    dataset_type="llff",
    downsample_factor=8
)
```

## Configuration Options

### Model Configuration

```python
@dataclass
class PlenoxelConfig:
    # Voxel grid settings
    grid_resolution: Tuple[int, int, int] = (256, 256, 256)
    scene_bounds: Tuple[float, ...] = (-1.0, -1.0, -1.0, 1.0, 1.0, 1.0)
    
    # Spherical harmonics
    sh_degree: int = 2  # 0-3, higher = more view-dependent effects
    
    # Coarse-to-fine training
    use_coarse_to_fine: bool = True
    coarse_resolutions: List = [(64, 64, 64), (128, 128, 128), (256, 256, 256)]
    coarse_epochs: List[int] = [2000, 5000, 10000]
    
    # Regularization
    sparsity_threshold: float = 0.01
    tv_lambda: float = 1e-6      # Total variation
    l1_lambda: float = 1e-8      # L1 sparsity
    
    # Rendering
    near_plane: float = 0.1
    far_plane: float = 10.0
```

### Training Configuration

```python
@dataclass
class PlenoxelTrainerConfig:
    # Training
    max_epochs: int = 10000
    learning_rate: float = 0.1
    weight_decay: float = 0.0
    
    # Loss weights
    color_loss_weight: float = 1.0
    tv_loss_weight: float = 1e-6
    l1_loss_weight: float = 1e-8
    
    # Pruning
    pruning_threshold: float = 0.01
    pruning_interval: int = 1000
    
    # Logging and evaluation
    eval_interval: int = 1000
    save_interval: int = 5000
    log_interval: int = 100
    use_tensorboard: bool = True
```

## Advanced Features

### Coarse-to-Fine Training

Plenoxels supports progressive training with increasing voxel resolution:

```python
config = PlenoxelConfig(
    use_coarse_to_fine=True,
    coarse_resolutions=[(64, 64, 64), (128, 128, 128), (256, 256, 256)],
    coarse_epochs=[2000, 5000, 10000]
)
```

### Sparsity Regularization

Automatic pruning of low-density voxels:

```python
# During training, voxels below threshold are pruned
model.prune_voxels(threshold=0.01)

# Get occupancy statistics
stats = model.get_occupancy_stats()
print(f"Sparsity: {stats['sparsity_ratio']:.2%}")
```

### Custom Loss Functions

```python
from src.plenoxels import PlenoxelLoss

class CustomPlenoxelLoss(PlenoxelLoss):
    def forward(self, outputs, targets):
        losses = super().forward(outputs, targets)
        
        # Add custom losses
        if 'depth' in outputs and 'depths' in targets:
            depth_loss = F.mse_loss(outputs['depth'], targets['depths'])
            losses['depth_loss'] = depth_loss
        
        return losses
```

## Utilities

### Voxel Grid Operations

```python
from src.plenoxels.utils import (
    create_voxel_grid,
    prune_voxel_grid,
    compute_voxel_occupancy_stats
)

# Create voxel grid
grid_info = create_voxel_grid(
    resolution=(128, 128, 128),
    scene_bounds=(-1, -1, -1, 1, 1, 1)
)

# Compute statistics
stats = compute_voxel_occupancy_stats(density_grid)
```

### Rendering Utilities

```python
from src.plenoxels.utils import (
    generate_rays,
    sample_points_along_rays,
    volume_render
)

# Generate rays from camera poses
rays_o, rays_d = generate_rays(poses, focal, H, W)

# Sample points along rays
points, t_vals = sample_points_along_rays(
    rays_o, rays_d, near=0.1, far=10.0, num_samples=192
)
```

## Performance Optimization

### GPU Memory Management

```python
# Use smaller batch sizes for high-resolution scenes
dataset_config.num_rays_train = 512  # Reduce if GPU memory limited

# Use mixed precision training
trainer_config.use_amp = True
```

### Training Speed

```python
# Start with coarse resolution
config.grid_resolution = (128, 128, 128)  # Faster initial training

# Reduce number of samples for speed
model(rays_o, rays_d, num_samples=64)  # vs 192 for quality
```

## Evaluation Metrics

```python
# Compute PSNR
mse = torch.mean((pred_rgb - gt_rgb) ** 2)
psnr = -10.0 * torch.log10(mse)

# Compute SSIM (requires additional dependencies)
from skimage.metrics import structural_similarity as ssim
ssim_val = ssim(pred_img, gt_img, multichannel=True)
```

## Troubleshooting

### Common Issues

1. **Out of Memory**
   - Reduce `grid_resolution`
   - Decrease `num_rays_train`
   - Use gradient checkpointing

2. **Slow Training**
   - Enable coarse-to-fine training
   - Use lower SH degree initially
   - Prune voxels more frequently

3. **Poor Quality**
   - Increase `grid_resolution`
   - Higher `sh_degree` for view-dependent effects
   - Adjust scene bounds properly

### Debug Mode

```python
# Enable debug logging
import logging
logging.basicConfig(level=logging.DEBUG)

# Check occupancy statistics
stats = model.get_occupancy_stats()
print(f"Occupied voxels: {stats['occupied_voxels']}/{stats['total_voxels']}")
```

## Examples

See the `example_usage.py` file for complete examples:

```bash
# Run demo
python -m src.plenoxels.example_usage --mode demo

# Train on Blender dataset
python -m src.plenoxels.example_usage --mode train \
    --data_dir data/nerf_synthetic/lego \
    --dataset_type blender \
    --max_epochs 10000

# Render novel views
python -m src.plenoxels.example_usage --mode render \
    --checkpoint outputs/plenoxel_exp/best.pth \
    --num_renders 40
```

## Testing

Run the test suite:

```bash
python -m src.plenoxels.test_plenoxels
```

## Citation

If you use this implementation, please cite the original paper:

```bibtex
@article{yu2021plenoxels,
  title={Plenoxels: Radiance fields without neural networks},
  author={Yu, Alex and Fridovich-Keil, Sara and Tancik, Matthew and Chen, Qinhong and Recht, Benjamin and Kanazawa, Angjoo},
  journal={arXiv preprint arXiv:2112.05131},
  year={2021}
}
```

## License

This implementation is provided for research and educational purposes. Please refer to the original paper and code for licensing details.

## Contributing

Contributions are welcome! Please feel free to submit issues and pull requests.

## Acknowledgments

This implementation is based on the excellent work by Yu et al. on Plenoxels. Special thanks to the original authors for their groundbreaking research in neural radiance fields. 