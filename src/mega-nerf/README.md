# Mega-NeRF: Scalable Construction of Large-Scale NeRFs for Virtual Fly-Throughs

This package implements Mega-NeRF, a method for scalable construction of large-scale Neural Radiance Fields that enables virtual fly-throughs of real-world environments.

## Overview

Mega-NeRF addresses the challenges of applying NeRF to large-scale scenes by:
- **Spatial Partitioning**: Decomposing large scenes into manageable subregions
- **Geometry-Aware Data Partitioning**: Intelligently distributing training data across submodules
- **Parallel Training**: Training multiple submodules simultaneously for efficiency
- **Temporal Coherence**: Ensuring smooth rendering across submodule boundaries

## Features

### Core Components

- **MegaNeRF Model**: Main model with spatial decomposition
- **MegaNeRFSubmodule**: Individual NeRF networks for scene regions
- **Spatial Partitioners**: Grid-based and geometry-aware partitioning strategies
- **Volumetric Renderer**: Efficient ray-based rendering with hierarchical sampling
- **Training Pipeline**: Sequential and parallel training modes

### Key Capabilities

- ✅ Large-scale scene reconstruction (city-scale)
- ✅ Spatial decomposition with configurable grid sizes
- ✅ Geometry-aware partitioning using camera positions
- ✅ Parallel training of submodules
- ✅ Appearance embedding for varying lighting conditions
- ✅ Interactive rendering with caching
- ✅ Video generation for fly-through sequences
- ✅ Multiple data format support (COLMAP, LLFF, NeRF format)

## Installation

```bash
# Clone the repository
git clone <repository-url>
cd NeuroCity

# Install dependencies
pip install -r requirements.txt

# Install the package in development mode
pip install -e .
```

## Quick Start

### 1. Training a Mega-NeRF Model

```bash
python src/mega-nerf/train_mega_nerf.py \
    --data_root /path/to/your/dataset \
    --exp_name my_meganerf_experiment \
    --num_submodules 8 \
    --grid_size 4,2 \
    --iterations_per_submodule 10000 \
    --training_mode sequential
```

### 2. Rendering Novel Views

```bash
python src/mega-nerf/render_mega_nerf.py \
    --model_path experiments/my_meganerf_experiment/final_model.pth \
    --output_dir renders/spiral_path \
    --render_type spiral \
    --num_frames 120 \
    --create_video
```

### 3. Using the Python API

```python
from src.mega_nerf import (
    MegaNeRF, MegaNeRFConfig, MegaNeRFTrainer,
    MegaNeRFDataset, GridPartitioner
)
Urban radiance field representation with deformable neural mesh primitives
# Setup configuration
config = MegaNeRFConfig(
    num_submodules=8,
    grid_size=(4, 2),
    hidden_dim=256,
    num_layers=8
)

# Create spatial partitioner
partitioner = GridPartitioner(
    scene_bounds=(-100, -100, -10, 100, 100, 50),
    grid_size=(4, 2),
    overlap_factor=0.15
)

# Create dataset
dataset = MegaNeRFDataset(
    data_root="path/to/data",
    partitioner=partitioner
)

# Create and train model
model = MegaNeRF(config)
trainer = MegaNeRFTrainer(config, model, dataset, "output_dir")
trainer.train_sequential()
```

## Configuration

### Model Configuration

```python
config = MegaNeRFConfig(
    # Scene decomposition
    num_submodules=8,           # Number of submodules
    grid_size=(4, 2),           # 2D grid decomposition
    overlap_factor=0.15,        # Overlap between submodules
    
    # Network architecture
    hidden_dim=256,             # Hidden layer dimension
    num_layers=8,               # Number of network layers
    use_viewdirs=True,          # Use view directions
    
    # Training parameters
    batch_size=1024,            # Ray batch size
    learning_rate=5e-4,         # Learning rate
    max_iterations=500000,      # Maximum iterations
    
    # Sampling parameters
    num_coarse=256,             # Coarse samples per ray
    num_fine=512,               # Fine samples per ray
    near=0.1,                   # Near plane
    far=1000.0,                 # Far plane
    
    # Scene bounds (x_min, y_min, z_min, x_max, y_max, z_max)
    scene_bounds=(-100, -100, -10, 100, 100, 50)
)
```

### Partitioning Strategies

#### Grid Partitioner
```python
partitioner = GridPartitioner(
    scene_bounds=scene_bounds,
    grid_size=(4, 2),           # 4x2 grid
    overlap_factor=0.15         # 15% overlap
)
```

#### Geometry-Aware Partitioner
```python
partitioner = GeometryAwarePartitioner(
    scene_bounds=scene_bounds,
    camera_positions=camera_positions,
    num_partitions=8,
    use_kmeans=True             # Use k-means clustering
)
```

## Training Modes

### Sequential Training
Trains submodules one after another:
```bash
python train_mega_nerf.py --training_mode sequential
```

### Parallel Training
Trains multiple submodules simultaneously:
```bash
python train_mega_nerf.py --training_mode parallel --num_parallel_workers 4
```

## Data Formats

### Supported Formats

1. **COLMAP**: Standard COLMAP sparse reconstruction
2. **LLFF**: Local Light Field Fusion format
3. **NeRF**: Original NeRF transforms.json format
4. **Synthetic**: Generated synthetic data

### Data Structure
```
dataset/
├── images/              # Input images
├── sparse/             # COLMAP sparse reconstruction (optional)
├── transforms.json     # NeRF format (optional)
├── poses_bounds.npy    # LLFF format (optional)
└── train.txt          # Training split (optional)
```

## Rendering Options

### Render Types

1. **Single View**: Render a single viewpoint
2. **Spiral Path**: Circular camera path around scene center
3. **Custom Path**: User-defined camera trajectory
4. **Dataset Views**: Render dataset test views

### Example Commands

```bash
# Spiral path rendering
python render_mega_nerf.py \
    --model_path model.pth \
    --render_type spiral \
    --radius 50 \
    --center 0,0,20 \
    --num_frames 120

# Dataset view rendUrban radiance field representation with deformable neural mesh primitivesering
python render_mega_nerf.py \
    --model_path model.pth \
    --render_type dataset \
    --data_root /path/to/dataset \
    --split test
```

## Performance Optimization

### Memory Management
- Use ray batching to control memory usage
- Enable caching for repeated rendering
- Adjust chunk size based on GPU memory

### Training Acceleration
- Use parallel training for multiple GPUs
- Enable mixed precision training
- Use appearance embeddings for varying conditions

### Rendering Speed
- Interactive renderer with caching
- Hierarchical sampling for efficiency
- Submodule-level culling for large scenes

## Advanced Usage

### Custom Partitioning
```python
class CustomPartitioner(SpatialPartitioner):
    def create_partitions(self):
        # Implement custom partitioning logic
        pass
    
    def assign_points_to_partitions(self, points):
        # Implement point assignment
        pass
```

### Custom Loss Functions
```python
class CustomTrainer(MegaNeRFTrainer):
    def _compute_loss(self, outputs, target_rgb):
        # Implement custom loss
        rgb_loss = F.mse_loss(outputs['rgb'], target_rgb)
        # Add additional loss terms
        return rgb_loss
```

### Appearance Control
```python
# Train with appearance embeddings
config.use_appearance_embedding = True
config.appearance_dim = 48

# Render with specific appearance
renderer.render_view(camera_info, appearance_id=5)
```

## Monitoring and Logging

### Weights & Biases Integration
```bash
python train_mega_nerf.py --use_wandb --wandb_project mega-nerf-experiments
```

### Tensorboard Logging
```python
# Enable tensorboard logging in trainer
trainer.setup_logging(use_tensorboard=True)
```

## Troubleshooting

### Common Issues

1. **Out of Memory**: Reduce batch size or chunk size
2. **Slow Training**: Enable parallel training or reduce resolution
3. **Poor Quality**: Increase number of samples or training iterations
4. **Boundary Artifacts**: Adjust overlap factor between submodules

### Debug Mode
```bash
python train_mega_nerf.py --debug --log_interval 10
```

## Citation

If you use this implementation, please cite:

```bibtex
@inproceedings{turki2022mega,
  title={Mega-nerf: Scalable construction of large-scale nerfs for virtual fly-throughs},
  author={Turki, Haithem and Ramanan, Deva and Satyanarayanan, Mahadev},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={12922--12931},
  year={2022}
}
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests for new functionality
5. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Original Mega-NeRF paper and implementation
- NeRF community for foundational work
- PyTorch team for the deep learning framework 