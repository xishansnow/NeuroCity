# SVRaster: Sparse Voxels Rasterization

SVRaster is a high-performance implementation of sparse voxel rasterization for real-time high-fidelity radiance field rendering. This package implements the method described in "Sparse Voxels Rasterization: Real-time High-fidelity Radiance Field Rendering" without neural networks or 3D Gaussians.

## Key Features

- **Adaptive Sparse Voxels**: Hierarchical voxel allocation with octree-based level-of-detail
- **Ray Direction-Dependent Morton Ordering**: Correct depth sorting without popping artifacts
- **Real-time Performance**: Efficient rasterization achieving high frame rates
- **High Fidelity**: Support for up to 65536³ grid resolution
- **Grid Compatibility**: Seamless integration with Volume Fusion, Voxel Pooling, and Marching Cubes

## Architecture

### Core Components

1. **AdaptiveSparseVoxels**: Manages sparse voxel representation with octree-based LOD
2. **VoxelRasterizer**: Custom rasterizer for efficient sparse voxel rendering
3. **SVRasterModel**: Main model combining sparse voxels and rasterization
4. **SVRasterLoss**: Loss functions for training

### Key Innovations

- **Adaptive Allocation**: Explicitly allocates sparse voxels to different levels of detail
- **Morton Ordering**: Uses ray direction-dependent Morton ordering for correct primitive blending
- **No Neural Networks**: Direct voxel representation without MLPs or 3D Gaussians
- **Efficient Storage**: Keeps only leaf nodes without full octree structure

## Installation

```bash
# Install dependencies
pip install torch torchvision numpy pillow tqdm tensorboard

# Install SVRaster (if packaged)
pip install svraster

# Or use directly from source
python -m src.svraster.example_usage --help
```

## Quick Start

### Basic Usage

```python
from src.svraster import SVRasterConfig, SVRasterModel
from src.svraster.dataset import SVRasterDatasetConfig, create_svraster_dataset
from src.svraster.trainer import SVRasterTrainerConfig, create_svraster_trainer

# Create model configuration
model_config = SVRasterConfig(
    max_octree_levels=12,
    base_resolution=64,
    scene_bounds=(-2.0, -2.0, -2.0, 2.0, 2.0, 2.0),
    density_activation="exp",
    color_activation="sigmoid"
)

# Create model
model = SVRasterModel(model_config)

# Render rays
ray_origins = torch.randn(1000, 3)
ray_directions = torch.randn(1000, 3)
outputs = model(ray_origins, ray_directions)
```

### Training

```python
# Dataset configuration
dataset_config = SVRasterDatasetConfig(
    data_dir="./data/nerf_synthetic/lego",
    dataset_type="blender",
    image_height=800,
    image_width=800,
    white_background=True
)

# Trainer configuration
trainer_config = SVRasterTrainerConfig(
    num_epochs=100,
    learning_rate=1e-3,
    enable_subdivision=True,
    enable_pruning=True
)

# Create datasets
train_dataset = create_svraster_dataset(dataset_config, split="train")
val_dataset = create_svraster_dataset(dataset_config, split="val")

# Create and run trainer
trainer = create_svraster_trainer(model_config, trainer_config, train_dataset, val_dataset)
trainer.train()
```

### Command Line Usage

```bash
# Training
python -m src.svraster.example_usage --mode train \
    --data_dir ./data/nerf_synthetic/lego \
    --output_dir ./outputs/svraster_lego

# Rendering
python -m src.svraster.example_usage --mode render \
    --data_dir ./data/nerf_synthetic/lego \
    --checkpoint ./outputs/svraster_lego/checkpoints/best_model.pth \
    --output_dir ./outputs/svraster_lego/renders
```

## Configuration

### Model Configuration

```python
SVRasterConfig(
    # Scene representation
    max_octree_levels=16,        # Maximum octree levels (65536³ resolution)
    base_resolution=64,          # Base grid resolution
    scene_bounds=(-1, -1, -1, 1, 1, 1),  # Scene bounding box
    
    # Voxel properties
    density_activation="exp",     # Density activation function
    color_activation="sigmoid",   # Color activation function
    sh_degree=2,                 # Spherical harmonics degree
    
    # Adaptive allocation
    subdivision_threshold=0.01,   # Threshold for voxel subdivision
    pruning_threshold=0.001,     # Threshold for voxel pruning
    
    # Rasterization
    ray_samples_per_voxel=8,     # Samples per voxel along ray
    morton_ordering=True,        # Use Morton ordering
    
    # Rendering
    background_color=(0, 0, 0),  # Background color
    use_view_dependent_color=True,
    use_opacity_regularization=True
)
```

### Dataset Configuration

```python
SVRasterDatasetConfig(
    # Data paths
    data_dir="./data",
    images_dir="images",
    
    # Data format
    dataset_type="blender",      # blender, colmap
    image_height=800,
    image_width=800,
    
    # Data splits
    train_split=0.8,
    val_split=0.1,
    test_split=0.1,
    
    # Ray sampling
    num_rays_train=1024,
    num_rays_val=512,
    
    # Background handling
    white_background=False,
    black_background=False
)
```

### Trainer Configuration

```python
SVRasterTrainerConfig(
    # Training parameters
    num_epochs=100,
    batch_size=1,
    learning_rate=1e-3,
    
    # Adaptive subdivision
    enable_subdivision=True,
    subdivision_start_epoch=10,
    subdivision_interval=5,
    
    # Pruning
    enable_pruning=True,
    pruning_start_epoch=20,
    pruning_interval=10,
    
    # Logging and saving
    val_interval=5,
    log_interval=100,
    save_interval=1000,
    
    # Hardware
    device="cuda",
    use_mixed_precision=True
)
```

## Data Formats

### Supported Dataset Types

1. **Blender Synthetic**: NeRF synthetic dataset format
2. **COLMAP**: Real-world captures processed with COLMAP

### Directory Structure

```
data/
├── images/              # Input images
│   ├── image_001.png
│   └── ...
├── transforms_train.json # Camera poses (Blender format)
├── transforms_val.json
└── transforms_test.json
```

## Advanced Features

### Adaptive Subdivision

SVRaster automatically subdivides voxels based on rendering gradients:

```python
# Enable adaptive subdivision
trainer_config.enable_subdivision = True
trainer_config.subdivision_start_epoch = 10
trainer_config.subdivision_interval = 5
trainer_config.subdivision_threshold = 0.01
```

### Voxel Pruning

Remove low-density voxels to maintain efficiency:

```python
# Enable pruning
trainer_config.enable_pruning = True
trainer_config.pruning_start_epoch = 20
trainer_config.pruning_interval = 10
trainer_config.pruning_threshold = 0.001
```

### Morton Ordering

Ray direction-dependent Morton ordering prevents popping artifacts:

```python
# Morton ordering is enabled by default
model_config.morton_ordering = True
```

## Performance Optimization

### Memory Efficiency

- Use chunked rendering for large images
- Enable mixed precision training
- Adjust render_chunk_size based on GPU memory

```python
trainer_config.use_mixed_precision = True
trainer_config.render_chunk_size = 1024  # Adjust based on GPU memory
```

### Speed Optimization

- Use appropriate octree levels for your scene
- Enable voxel pruning to remove unnecessary voxels
- Use larger batch sizes if memory allows

## Evaluation Metrics

SVRaster supports standard NeRF evaluation metrics:

- **PSNR**: Peak Signal-to-Noise Ratio
- **SSIM**: Structural Similarity Index
- **LPIPS**: Learned Perceptual Image Patch Similarity

## Troubleshooting

### Common Issues

1. **Out of Memory**: Reduce render_chunk_size or image resolution
2. **Slow Training**: Enable mixed precision and check GPU utilization
3. **Poor Quality**: Increase max_octree_levels or adjust subdivision threshold
4. **Artifacts**: Ensure Morton ordering is enabled

### Performance Tips

- Start with lower resolution for quick prototyping
- Use adaptive subdivision for better detail preservation
- Enable pruning to maintain efficiency during training
- Monitor voxel count to prevent excessive memory usage

## Citation

If you use SVRaster in your research, please cite:

```bibtex
@article{sun2024svraster,
  title={Sparse Voxels Rasterization: Real-time High-fidelity Radiance Field Rendering},
  author={Sun, Cheng and Choe, Jaesung and Loop, Charles and Ma, Wei-Chiu and Wang, Yu-Chiang Frank},
  journal={arXiv preprint arXiv:2412.04459},
  year={2024}
}
```

## License

This implementation is provided for research purposes. Please refer to the original paper and official implementation for licensing details.

## Contributing

Contributions are welcome! Please feel free to submit issues and pull requests.

## Acknowledgments

This implementation is based on the paper "Sparse Voxels Rasterization: Real-time High-fidelity Radiance Field Rendering" by Sun et al. We thank the authors for their excellent work and the research community for advancing neural radiance fields. 


## See Also

For more detailed technical documentation:

- [Training Implementation](SVRaster_Training_Implementation_cn.md): Detailed explanation of the training pipeline, loss functions, and optimization strategies
- [Rasterization Implementation](SVRaster_Rasterization_Implementation_cn.md): In-depth coverage of the sparse voxel rasterization algorithm and implementation details


