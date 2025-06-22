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
python src/nerfs/mega_nerf/train_mega_nerf.py \
    --dataset_path /path/to/dataset \
    --output_dir /path/to/output \
    --config_path configs/mega_nerf_config.yaml
```

### 2. Rendering Views

```bash  
python src/nerfs/mega_nerf/render_mega_nerf.py \
    --model_path /path/to/trained/model \
    --output_dir /path/to/renders \
    --camera_path /path/to/camera/poses
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

```