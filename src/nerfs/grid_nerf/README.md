# Grid-NeRF: Grid-guided Neural Radiance Fields for Large Urban Scenes

This package implements Grid-guided Neural Radiance Fields (Grid-NeRF), a scalable approach for neural rendering of large urban environments using hierarchical voxel grids to guide the rendering process.

## Overview

Grid-NeRF addresses the challenges of rendering large urban scenes by:
- Using hierarchical voxel grids to organize scene geometry
- Employing multi-resolution grid structures for efficient sampling
- Optimizing neural networks with grid-guided features
- Supporting distributed training for large-scale datasets

## Key Features

- **Hierarchical Grid Structure**: Multi-level voxel grids with increasing resolution
- **Grid-Guided Neural Networks**: MLPs that leverage grid features for efficient rendering
- **Scalable Training**: Support for large urban datasets with distributed training
- **Fast Rendering**: Efficient volume rendering with grid-based sampling
- **Multiple Dataset Support**: Built-in support for KITTI-360 and custom datasets
- **Comprehensive Evaluation**: Metrics including PSNR, SSIM, and LPIPS

## Installation

### Prerequisites

- Python 3.8+
- PyTorch 1.12+
- CUDA 11.0+ (for GPU acceleration)

### Install Dependencies

```bash
pip install torch torchvision torchaudio
pip install opencv-python matplotlib numpy
pip install pyyaml tqdm tensorboard
pip install lpips  # Optional, for LPIPS metric
```

### Package Installation

```bash
# From the project root
cd src/grid_nerf
pip install -e .
```

## Quick Start

### Basic Usage

```python
from src.grid_nerf import GridNeRF, GridNeRFConfig, GridNeRFTrainer
from src.grid_nerf import create_dataset, quick_setup

# Quick setup with defaults
model, trainer, dataset = quick_setup(
    data_path="path/to/your/data",
    output_dir="./outputs",
    device="cuda"
)

# Start training
trainer.train(train_dataset=dataset)
```

### Custom Configuration

```python
from src.grid_nerf import GridNeRFConfig, GridNeRF

# Create custom configuration
config = GridNeRFConfig(
    # Scene bounds
    scene_bounds=(-100, -100, -10, 100, 100, 50),
    
    # Grid configuration
    grid_levels=4,
    base_resolution=64,
    resolution_multiplier=2,
    grid_feature_dim=32,
    
    # Network architecture
    density_layers=3,
    density_hidden_dim=256,
    color_layers=2,
    color_hidden_dim=128,
    
    # Training parameters
    batch_size=1024,
    num_epochs=200,
    grid_lr=1e-2,
    mlp_lr=5e-4
)

# Create model
model = GridNeRF(config)
```

## Architecture

### Hierarchical Grid Structure

The hierarchical grid consists of multiple levels with increasing resolution:

```
Level 0: 64³ voxels (coarse)
Level 1: 128³ voxels 
Level 2: 256³ voxels
Level 3: 512³ voxels (fine)
```

Each level stores feature vectors that are combined to guide the neural network.

### Grid-Guided MLP

The neural network consists of:
- **Density Network**: Predicts volume density from grid features and positions
- **Color Network**: Predicts RGB colors from grid features, positions, and view directions
- **Position Encoding**: Sinusoidal encoding for spatial coordinates
- **Direction Encoding**: Encoding for view directions

### Volume Rendering

The renderer performs:
1. **Ray Sampling**: Sample points along camera rays
2. **Grid Sampling**: Extract features from hierarchical grids
3. **Network Evaluation**: Predict density and color at sample points
4. **Volume Integration**: Composite final pixel colors using alpha blending

## Training

### Single GPU Training

```python
from src.grid_nerf import GridNeRFTrainer, GridNeRFConfig
from src.grid_nerf.dataset import create_dataset

# Setup configuration
config = GridNeRFConfig(
    batch_size=1024,
    num_epochs=100,
    grid_lr=1e-2,
    mlp_lr=5e-4
)

# Create trainer
trainer = GridNeRFTrainer(
    config=config,
    output_dir="./outputs",
    device=torch.device("cuda")
)

# Load dataset
train_dataset = create_dataset(
    data_path="path/to/data",
    split='train',
    config=config
)

# Start training
trainer.train(train_dataset=train_dataset)
```

### Multi-GPU Training

```python
import torch.multiprocessing as mp
from src.grid_nerf.trainer import main_worker

# Configuration
config = GridNeRFConfig(batch_size=2048)  # Scale for multiple GPUs
data_config = {
    'train_data_path': 'path/to/data',
    'train_kwargs': {}
}

# Launch distributed training
num_gpus = 4
mp.spawn(
    main_worker,
    args=(num_gpus, config, "./outputs", data_config),
    nprocs=num_gpus,
    join=True
)
```

## Dataset Format

### Directory Structure

```
data/
├── images/
│   ├── 000000.png
│   ├── 000001.png
│   └── ...
├── poses/
│   ├── 000000.npy  # Camera pose matrices (4x4)
│   ├── 000001.npy
│   └── ...
├── intrinsics.npy  # Camera intrinsic matrix (3x3)
└── scene_bounds.npy  # Scene boundaries (optional)
```

### KITTI-360 Dataset

For KITTI-360 data:

```python
from src.grid_nerf.dataset import KITTI360GridDataset

dataset = KITTI360GridDataset(
    data_path="path/to/kitti360",
    sequence="2013_05_28_drive_0000_sync",
    split='train',
    config=config
)
```

### Custom Dataset

```python
from src.grid_nerf.dataset import GridNeRFDataset

dataset = GridNeRFDataset(
    data_path="path/to/custom/data",
    split='train',
    config=config,
    image_extension='.jpg',
    load_depth=True  # If depth data available
)
```

## Evaluation

### Rendering Test Images

```python
# Load trained model
checkpoint = torch.load("path/to/checkpoint.pth")
model.load_state_dict(checkpoint['model_state_dict'])

# Render test images
with torch.no_grad():
    for i, sample in enumerate(test_dataset):
        rays_o, rays_d = sample['rays_o'], sample['rays_d']
        outputs = model(rays_o, rays_d)
        
        # Save rendered image
        rgb_pred = outputs['rgb'].reshape(H, W, 3)
        save_image(rgb_pred, f"render_{i:03d}.png")
```

### Computing Metrics

```python
from src.grid_nerf.utils import compute_psnr, compute_ssim, compute_lpips

# Compute metrics
psnr = compute_psnr(pred_image, gt_image)
ssim = compute_ssim(pred_image, gt_image)
lpips = compute_lpips(pred_image, gt_image)

print(f"PSNR: {psnr:.2f} dB")
print(f"SSIM: {ssim:.4f}")
print(f"LPIPS: {lpips:.4f}")
```

## API Reference

### Core Classes

#### GridNeRFConfig

Configuration class for Grid-NeRF parameters.

```python
config = GridNeRFConfig(
    # Scene bounds
    scene_bounds=(-100, -100, -10, 100, 100, 50),
    
    # Grid configuration
    grid_levels=4,
    base_resolution=64,
    resolution_multiplier=2,
    grid_feature_dim=32,
    
    # Network architecture
    density_layers=3,
    density_hidden_dim=256,
    color_layers=2,
    color_hidden_dim=128,
    position_encoding_levels=10,
    direction_encoding_levels=4,
    
    # Rendering
    num_samples=64,
    num_importance_samples=128,
    perturb=True,
    white_background=False,
    
    # Training
    batch_size=1024,
    num_epochs=200,
    grid_lr=1e-2,
    mlp_lr=5e-4,
    weight_decay=1e-6,
    grad_clip_norm=1.0,
    
    # Loss weights
    color_weight=1.0,
    depth_weight=0.1,
    grid_regularization_weight=1e-4,
    
    # Scheduling
    scheduler_type="cosine",
    max_steps=200000,
    warmup_steps=2000,
    
    # Evaluation
    eval_batch_size=256,
    chunk_size=1024,
    
    # Logging
    log_every_n_steps=100,
    eval_every_n_epochs=5,
    save_every_n_epochs=10,
    render_every_n_epochs=20
)
```

#### GridNeRF

Main model class implementing Grid-guided Neural Radiance Fields.

```python
model = GridNeRF(config)

# Forward pass
outputs = model(rays_o, rays_d)
# Returns: {'rgb': tensor, 'depth': tensor, 'weights': tensor, ...}
```

#### GridNeRFTrainer

Training pipeline with support for distributed training and evaluation.

```python
trainer = GridNeRFTrainer(
    config=config,
    output_dir="./outputs",
    device=device,
    rank=0,
    world_size=1,
    use_tensorboard=True
)

trainer.train(
    train_dataset=train_dataset,
    val_dataset=val_dataset,
    test_dataset=test_dataset,
    resume_from="path/to/checkpoint.pth"  # Optional
)
```

### Dataset Classes

#### GridNeRFDataset

Base dataset class for Grid-NeRF training data.

```python
dataset = GridNeRFDataset(
    data_path="path/to/data",
    split='train',
    config=config,
    image_extension='.png',
    load_depth=False,
    transform=None
)
```

#### KITTI360GridDataset

Specialized dataset for KITTI-360 data.

```python
dataset = KITTI360GridDataset(
    data_path="path/to/kitti360",
    sequence="2013_05_28_drive_0000_sync",
    split='train',
    config=config,
    start_frame=0,
    end_frame=None
)
```

### Utility Functions

#### Image Metrics

```python
from src.grid_nerf.utils import compute_psnr, compute_ssim, compute_lpips

psnr = compute_psnr(pred, target, max_val=1.0)
ssim = compute_ssim(pred, target, window_size=11)
lpips = compute_lpips(pred, target, net='alex')
```

#### Visualization

```python
from src.grid_nerf.utils import save_image, create_video_from_images

save_image(image_tensor, "output.png", normalize=True)
create_video_from_images("frames/", "video.mp4", fps=30)
```

#### Mathematical Utilities

```python
from src.grid_nerf.utils import (
    positional_encoding, get_ray_directions, 
    sample_along_rays, volume_rendering
)

# Positional encoding
encoded = positional_encoding(coordinates, L=10)

# Ray generation
directions = get_ray_directions(H, W, focal)

# Ray sampling
points, z_vals = sample_along_rays(rays_o, rays_d, near, far, n_samples)

# Volume rendering
outputs = volume_rendering(rgb, density, z_vals, rays_d)
```

## Examples

### Example 1: Basic Training

```python
from src.grid_nerf import quick_setup

# Quick setup and training
model, trainer, dataset = quick_setup(
    data_path="data/urban_scene",
    output_dir="outputs/experiment_1"
)

trainer.train(train_dataset=dataset)
```

### Example 2: Custom Configuration

```python
from src.grid_nerf import GridNeRFConfig, GridNeRF, GridNeRFTrainer
from src.grid_nerf.dataset import create_dataset

# Custom configuration for large scene
config = GridNeRFConfig(
    scene_bounds=(-200, -200, -20, 200, 200, 80),
    grid_levels=5,
    base_resolution=128,
    batch_size=2048,
    num_epochs=300
)

# Create components
model = GridNeRF(config)
dataset = create_dataset("data/large_city", 'train', config)
trainer = GridNeRFTrainer(config, "outputs/large_city", device)

# Train
trainer.train(train_dataset=dataset)
```

### Example 3: Evaluation and Visualization

```python
import torch
from src.grid_nerf import GridNeRF, GridNeRFConfig
from src.grid_nerf.utils import save_image, compute_psnr

# Load model
checkpoint = torch.load("outputs/best.pth")
config = GridNeRFConfig(**checkpoint['config'])
model = GridNeRF(config)
model.load_state_dict(checkpoint['model_state_dict'])

# Evaluation loop
total_psnr = 0
for i, sample in enumerate(test_dataset):
    with torch.no_grad():
        outputs = model(sample['rays_o'], sample['rays_d'])
    
    # Compute metrics
    psnr = compute_psnr(outputs['rgb'], sample['target_rgb'])
    total_psnr += psnr
    
    # Save rendered image
    H, W = sample['image_height'], sample['image_width']
    rgb_image = outputs['rgb'].reshape(H, W, 3)
    save_image(rgb_image, f"renders/test_{i:03d}.png")

print(f"Average PSNR: {total_psnr / len(test_dataset):.2f}")
```

## Performance Optimization

### Memory Optimization

- Use smaller batch sizes for limited GPU memory
- Reduce grid resolution or feature dimensions
- Enable gradient checkpointing for large models

```python
config = GridNeRFConfig(
    batch_size=512,          # Reduce batch size
    base_resolution=32,      # Lower resolution
    grid_feature_dim=16,     # Fewer features
    chunk_size=1024         # Smaller rendering chunks
)
```

### Training Speed

- Use multiple GPUs for distributed training
- Increase batch size for better GPU utilization
- Use mixed precision training (FP16)

```python
# Enable mixed precision
from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler()
with autocast():
    outputs = model(rays_o, rays_d)
    loss = loss_fn(outputs, targets)

scaler.scale(loss).backward()
scaler.step(optimizer)
scaler.update()
```

## Troubleshooting

### Common Issues

1. **Out of Memory Error**
   - Reduce batch size or chunk size
   - Lower grid resolution
   - Use gradient checkpointing

2. **Slow Training**
   - Increase batch size if memory allows
   - Use multiple GPUs
   - Check data loading bottlenecks

3. **Poor Rendering Quality**
   - Increase grid resolution
   - More network layers or hidden dimensions
   - Adjust loss weights
   - Longer training

4. **Dataset Loading Issues**
   - Check file paths and permissions
   - Verify image and pose file formats
   - Ensure camera intrinsics are provided

### Debug Mode

```python
# Enable debug logging
import logging
logging.basicConfig(level=logging.DEBUG)

# Check model gradients
for name, param in model.named_parameters():
    if param.grad is not None:
        print(f"{name}: {param.grad.norm().item()}")
```

## Contributing

We welcome contributions! Please see the main project documentation for contribution guidelines.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Citation

If you use this code in your research, please cite:

```bibtex
@article{gridnerf2023,
  title={Grid-guided neural radiance fields for large urban scenes},
  author={Author Name},
  journal={Conference/Journal Name},
  year={2023}
}
```

## Acknowledgments

- Based on the Neural Radiance Fields (NeRF) framework
- Inspired by grid-based neural representations
- Built with PyTorch and modern deep learning practices 