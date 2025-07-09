# Block-NeRF: Scalable Large Scene Neural View Synthesis

[![Version](https://img.shields.io/badge/version-1.0.0-blue.svg)](https://github.com/neurocity/block_nerf)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![CUDA](https://img.shields.io/badge/CUDA-supported-brightgreen.svg)]()
[![Python](https://img.shields.io/badge/python-3.8%2B-blue.svg)](https://python.org)

**Block-NeRF** implements scalable neural view synthesis for large-scale scenes through spatial decomposition and block-based training. This implementation enables city-scale reconstruction and rendering with state-of-the-art quality and performance.

## ✨ Features

- 🏗️ **Scalable Architecture**: Handles city-scale environments through block decomposition
- 🎨 **High-Quality Rendering**: Dual rendering pipeline (volume + rasterization)
- ⚡ **CUDA Acceleration**: GPU-optimized operations for maximum performance
- 🔧 **Modular Design**: Clean separation of training and inference components
- 📊 **Comprehensive Tools**: Full pipeline from data processing to final rendering
- 🌟 **Production Ready**: Robust error handling and extensive testing

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/neurocity/block_nerf.git
cd block_nerf

# Install dependencies
pip install -r requirements.txt

# Install package
pip install -e .

# Optional: Build CUDA extension for acceleration
cd cuda
source setup_environment.sh
python3 install_cuda_extension.py
```

### Basic Usage

```python
import torch
from block_nerf import (
    BlockNeRFConfig,
    BlockNeRFModel,
    BlockNeRFTrainer,
    BlockNeRFRenderer,
    BlockManager
)

# Create configuration
config = BlockNeRFConfig(
    block_size=64,
    max_blocks=100,
    appearance_embedding_dim=32
)

# Initialize components
model = BlockNeRFModel(config)
trainer = BlockNeRFTrainer(model, config)
renderer = BlockNeRFRenderer(model, config)

# Train on your data
trainer.train(dataset)

# Render novel views
rendered_image = renderer.render(camera_pose)
```

## 🏗️ Architecture

This refactored implementation follows a modern, dual-rendering architecture:

```
Training Phase:   BlockNeRFTrainer ↔ VolumeRenderer (stable volume rendering)
Inference Phase:  BlockNeRFRenderer ↔ BlockRasterizer (efficient rasterization)
```

### Key Design Principles

- **Modular Components**: Clean separation between training and inference
- **Tightly Coupled Systems**: Optimal performance through specialized rendering
- **Comprehensive Configuration**: Unified config system for all components
- **CUDA Acceleration**: Performance-critical operations in CUDA
- **Production Ready**: Robust error handling and comprehensive testing

## 🎯 Model Characteristics

### 🎨 Representation Method
- **Block Decomposition**: Divides large scenes into smaller, manageable NeRF blocks
- **Individual NeRFs**: Each block contains its own MLP-based NeRF model
- **Appearance Embeddings**: Per-block appearance codes for environmental variations
- **Visibility Network**: Predicts which blocks are visible from each viewpoint
- **Spatial Indexing**: Efficient spatial organization for block selection

### ⚡ Training Performance
- **Training Time**: 3-7 days for city-scale scenes (distributed training)
- **Training Speed**: ~2,000-5,000 rays/second per block on RTX 3080
- **Convergence**: Parallel training of multiple blocks
- **GPU Memory**: 6-12GB per block during training
- **Scalability**: Excellent scaling to very large scenes

### 🎬 Rendering Mechanism
- **Block Selection**: Visibility network determines relevant blocks
- **Parallel Rendering**: Multiple blocks rendered simultaneously
- **Block Compositing**: Smooth blending between overlapping blocks
- **Appearance Control**: Dynamic appearance adjustment per block
- **Pose Refinement**: Real-time pose optimization during rendering

### 🚀 Rendering Speed
- **Inference Speed**: 30-120 seconds per 800×800 image (depends on scene size)
- **Ray Processing**: ~1,000-3,000 rays/second during inference
- **Block Overhead**: Additional computation for block selection and blending
- **Scalability**: Rendering time scales with visible blocks, not total scene size
- **Parallel Processing**: Efficient multi-GPU rendering

### 💾 Storage Requirements
- **Model Size**: 1-10 GB for city-scale scenes (multiple blocks)
- **Per-block Size**: 100-500 MB per individual block
- **Scene Representation**: Total size scales with scene coverage
- **Appearance Codes**: Additional storage for environmental variations
- **Metadata**: Block boundaries, visibility networks, pose refinements

### 📊 Performance Comparison

| Metric | Classic NeRF | Block-NeRF | Advantage |
|--------|--------------|------------|-----------|
| Scene Scale | Room-scale | City-scale | **1000x larger** |
| Training Time | 1-2 days | 3-7 days | **Enables large scenes** |
| Model Size | 100-500 MB | 1-10 GB | **Scales with scene** |
| Rendering Quality | High | High | **Maintains quality** |
| Memory Efficiency | Fixed | Scales | **Handles any size** |

### 🎯 Use Cases
- **City-scale Reconstruction**: Large urban environment modeling
- **Autonomous Driving**: Large-scale scene understanding
- **Urban Planning**: City-wide visualization and simulation
- **Mapping Applications**: Large-area 3D mapping
- **Virtual Tourism**: Immersive city exploration

## Features

### Core Components

- **Core**: `BlockNeRFConfig`, `BlockNeRFModel`, `BlockNeRFLoss`
- **Training**: `BlockNeRFTrainer` + `VolumeRenderer` (stable training)
- **Inference**: `BlockNeRFRenderer` + `BlockRasterizer` (efficient rendering)
- **Data**: `BlockNeRFDataset` with comprehensive data loading
- **Management**: `BlockManager` for spatial decomposition
- **CLI**: Command-line interface for training and rendering

### Modern Features

- ✅ **Dual Rendering Architecture**: Volume rendering for training, rasterization for inference
- ✅ **Tightly Coupled Components**: Optimized performance through specialized coupling
- ✅ **Comprehensive Configuration**: Unified config system with YAML support
- ✅ **CUDA Acceleration**: Performance-critical kernels in CUDA
- ✅ **Block Caching**: Intelligent memory management for large scenes
- ✅ **Appearance Control**: Dynamic environmental variation handling
- ✅ **Pose Refinement**: Automatic camera pose improvement
- ✅ **Production Ready**: Robust error handling and validation

## Installation
```bash
# Install dependencies
pip install torch torchvision numpy opencv-python imageio tqdm tensorboard

# Install NeuroCity package
cd /path/to/NeuroCity
pip install -e .
```

## Quick Start

### 1. Training Example

```python
from src.nerfs.block_nerf import (
    BlockNeRFConfig, 
    create_block_nerf_trainer,
    create_block_nerf_dataset
)

# Configure the model
config = BlockNeRFConfig(
    scene_bounds=(-100, -100, -10, 100, 100, 30),  # City bounds (x_min, y_min, z_min, x_max, y_max, z_max)
    block_size=50.0,                                # Size of each block
    overlap_ratio=0.1,                              # Overlap between blocks
    hidden_dim=256,                                 # Network hidden dimension
    num_layers=8,                                   # Number of layers
    use_appearance_embedding=True,                  # Handle lighting variations
    use_pose_refinement=True,                       # Improve pose accuracy
)

# Create dataset
dataset = create_block_nerf_dataset(
    data_root="data/city_scene/",
    split="train",
    format="colmap"  # Support COLMAP, LLFF, custom formats
)

# Create trainer with dual architecture
trainer = create_block_nerf_trainer(
    model_config=config,
    scene_bounds=config.scene_bounds,
    device="cuda"
)

# Start training
trainer.train(dataset, num_epochs=1000)
```

### 2. Inference Example

```python
from src.nerfs.block_nerf import (
    BlockNeRFConfig,
    create_block_nerf_renderer,
    create_block_rasterizer
)

# Load same configuration as training
config = BlockNeRFConfig(...)

# Create renderer with rasterization
rasterizer = create_block_rasterizer()
renderer = create_block_nerf_renderer(
    model_config=config,
    rasterizer=rasterizer,
    device="cuda"
)

# Load trained blocks
renderer.load_blocks("checkpoints/city_scene/")

# Render novel view
camera_pose = torch.eye(4)  # 4x4 camera-to-world matrix
intrinsics = torch.tensor([[focal, 0, cx], [0, focal, cy], [0, 0, 1]])

output = renderer.render_image(
    camera_pose=camera_pose,
    intrinsics=intrinsics,
    width=800,
    height=600,
    appearance_id=0,     # Control lighting/appearance
    exposure_value=1.0   # Control exposure
)

# Output contains 'rgb', 'depth', 'alpha' tensors
rgb_image = output['rgb']      # [H, W, 3] rendered image
depth_map = output['depth']    # [H, W] depth map
alpha_mask = output['alpha']   # [H, W] alpha mask
```

### 3. Command Line Interface

```bash
# Train a Block-NeRF model
python -m src.nerfs.block_nerf.cli train \
    --config configs/city_scene.yaml \
    --data-root data/city_scene/ \
    --output-dir outputs/city_scene/ \
    --num-epochs 1000 \
    --validate-every 100

# Render novel views
python -m src.nerfs.block_nerf.cli render \
    --checkpoint outputs/city_scene/checkpoints/ \
    --poses data/city_scene/test_poses.txt \
    --output-dir renders/city_scene/ \
    --width 1920 \
    --height 1080 \
    --format mp4
```

### 4. Configuration Files

```yaml
# config.yaml
model:
  scene_bounds: [-100, -100, -10, 100, 100, 30]
  block_size: 50.0
  overlap_ratio: 0.1
  hidden_dim: 256
  num_layers: 8
  use_appearance_embedding: true
  use_pose_refinement: true

training:
  learning_rate: 5e-4
  batch_size: 1024
  num_epochs: 1000
  validate_every: 100
  save_every: 500

data:
  format: "colmap"
  img_scale: 1.0
  ray_batch_size: 1024
  use_cache: true

rendering:
  chunk_size: 4096
  use_cached_blocks: true
  max_cached_blocks: 8
  antialiasing: true
```

## Architecture Details

### Dual Rendering Design

**Training Phase** (Stable & Accurate):
```
BlockNeRFTrainer → VolumeRenderer → Volume Integration → Loss Computation
```

**Inference Phase** (Fast & Efficient):
```
BlockNeRFRenderer → BlockRasterizer → Block Selection → Rasterization
```

### Component Responsibilities

- **Core**: Model definitions, configurations, loss functions
- **Trainer + VolumeRenderer**: Stable training with volume rendering
- **Renderer + BlockRasterizer**: Fast inference with block-based rasterization
- **BlockManager**: Spatial decomposition and block coordination
- **Dataset**: Comprehensive data loading and preprocessing
- **CLI**: User-friendly command-line interface

## Performance

### Training Performance
- **Speed**: ~2,000-5,000 rays/second per block
- **Memory**: 6-12GB GPU memory per block
- **Time**: 3-7 days for city-scale scenes (distributed)
- **Scalability**: Parallel training across multiple blocks

### Inference Performance  
- **Speed**: 30-120 seconds per 800×800 image
- **Quality**: Maintains high fidelity
- **Memory**: Dynamic block caching
- **Scalability**: Scales with visible blocks, not total scene size

## Documentation

- **[QUICK_START_GUIDE.md](QUICK_START_GUIDE.md)**: Complete quick start guide
- **[API_REFERENCE.md](API_REFERENCE.md)**: Comprehensive API documentation
- **[TRAINING_MECHANISM_cn.md](TRAINING_MECHANISM_cn.md)**: Detailed training mechanisms
- **[CUDA Usage Guide](cuda/README_CUDA_USAGE.md)**: CUDA optimization setup

## Examples

See `examples/block_nerf_example.py` for comprehensive usage examples including:
- Basic training and inference
- Advanced configuration
- Multi-GPU training
- Custom data formats
- Performance optimization

## Troubleshooting

### Common Issues

1. **CUDA Out of Memory**
   ```python
   # Reduce batch size or block size
   config = BlockNeRFConfig(batch_size=512, block_size=25.0)
   ```

2. **Training Instability**
   ```python
   # Start with smaller scenes and gradually increase
   config = BlockNeRFConfig(scene_bounds=(-50, -50, -5, 50, 50, 15))
   ```

3. **Slow Rendering**
   ```python
   # Enable block caching and tune chunk size
   renderer_config = BlockNeRFRendererConfig(
       use_cached_blocks=True,
       chunk_size=2048
   )
   ```

For more details, see [TRAINING_FAQ_cn.md](TRAINING_FAQ_cn.md).

## Citation

```bibtex
@inproceedings{tancik2022block,
  title={Block-NeRF: Scalable Large Scene Neural View Synthesis},
  author={Tancik, Matthew and Casser, Vincent and Yan, Xinchen and Pradhan, Sabeek and Mildenhall, Ben and Srinivasan, Pratul P and Barron, Jonathan T and Kretzschmar, Henrik},
  booktitle={CVPR},
  year={2022}
}
```

## License

This implementation is provided under the MIT license. See [LICENSE](../../../LICENSE) for details.





