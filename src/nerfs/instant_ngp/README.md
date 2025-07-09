# Instant NGP Implementation

A PyTorch implementation of **Instant Neural Graphics Primitives with Multiresolution Hash Encoding** from SIGGRAPH 2022.

This implementation provides a fast, efficient, and easy-to-use version of Instant NGP for neural radiance fields (NeRF) and other neural graphics applications.

## 🚀 Key Features

- **⚡ 10-100x Faster**: Hash-based encoding dramatically reduces training time
- **🎯 High Quality**: Maintains rendering quality while being much faster
- **🔧 Easy to Use**: Simple API for training and inference
- **📦 Complete Package**: Includes dataset loading, training, and rendering
- **🧪 CUDA Accelerated**: Optimized CUDA kernels for maximum performance

## 🏗️ Architecture Overview

```
Input Position (x,y,z) → Hash Encoding → Small MLP → Density σ
                                    ↘
Input Direction (θ,φ) → SH Encoding → Color MLP → RGB Color
```

### Key Components

1. **Multiresolution Hash Encoding**: Efficient spatial feature lookup using hash tables
2. **Spherical Harmonics**: View-dependent appearance encoding
3. **Small MLPs**: Compact networks for fast inference
4. **Volume Rendering**: Standard NeRF-style ray marching and integration

## 📦 Installation

```bash
pip install -r requirements.txt
```

For CUDA support:
```bash
cd cuda
python setup.py build_ext --inplace
```

## 🚀 Quick Start

### Basic Training

```python
from nerfs.instant_ngp import (
    InstantNGPConfig, InstantNGPModel,
    InstantNGPTrainer, InstantNGPTrainerConfig,
    InstantNGPDataset, InstantNGPDatasetConfig
)

# Create model
config = InstantNGPConfig(
    num_levels=16,
    base_resolution=16,
    finest_resolution=512
)
model = InstantNGPModel(config)

# Create trainer
trainer_config = InstantNGPTrainerConfig(
    num_epochs=20,
    batch_size=8192,
    learning_rate=1e-2
)
trainer = InstantNGPTrainer(model, trainer_config)

# Create dataset
dataset_config = InstantNGPDatasetConfig(
    data_root="data/nerf_synthetic/lego",
    dataset_type="blender"
)
train_dataset = InstantNGPDataset(dataset_config, split="train")
val_dataset = InstantNGPDataset(dataset_config, split="val")

# Train
trainer.train(train_dataset, val_dataset)
```

### Inference

```python
from nerfs.instant_ngp import (
    InstantNGPInferenceRenderer, InstantNGPRendererConfig
)

# Create renderer
renderer_config = InstantNGPRendererConfig(
    num_samples=64,
    batch_size=4096
)
renderer = InstantNGPInferenceRenderer(model, renderer_config)

# Render image
result = renderer.render_image(
    camera_pose=camera_pose,
    intrinsics=intrinsics,
    width=800,
    height=800
)
```

### Command Line Interface

```bash
# Training
python -m nerfs.instant_ngp.cli train \
    --data-dir data/nerf_synthetic/lego \
    --output-dir outputs \
    --num-epochs 20

# Rendering
python -m nerfs.instant_ngp.cli render \
    --checkpoint outputs/model.pth \
    --output-dir renders \
    --width 800 --height 800
```

## 🔧 Configuration

Key configuration parameters:

```python
config = InstantNGPConfig(
    # Hash encoding
    num_levels=16,           # Number of resolution levels
    level_dim=2,             # Features per level
    base_resolution=16,      # Base grid resolution
    finest_resolution=512,   # Finest resolution
    log2_hashmap_size=19,    # Hash table size
    
    # Network architecture
    hidden_dim=64,           # MLP hidden dimension
    num_layers=2,            # Number of MLP layers
    
    # Training
    learning_rate=1e-2,      # Learning rate
    batch_size=8192,         # Ray batch size
)
```

## 📊 Performance

| Metric | Classic NeRF | Instant NGP | Improvement |
|--------|--------------|-------------|-------------|
| Training Time | 1-2 days | 20-60 min | **20-50x faster** |
| Inference Speed | 30 sec/image | Real-time | **>100x faster** |
| Model Size | 100-500 MB | 10-50 MB | **5-10x smaller** |
| GPU Memory | 8-16 GB | 2-4 GB | **2-4x less** |

## 🧪 Testing

Run the test suite:

```bash
python run_tests.py
```

## 📄 Citation

```bibtex
@article{mueller2022instant,
    title={Instant Neural Graphics Primitives with a Multiresolution Hash Encoding},
    author={Thomas M{\"u}ller and Alex Evans and Christoph Schied and Alexander Keller},
    journal={ACM Transactions on Graphics (SIGGRAPH)},
    year={2022},
    volume={41},
    number={4},
    pages={102:1--102:15}
}
```

## 📧 Support

For issues and questions:
- 🐛 **Bug reports**: Open an issue with reproduction steps
- 💡 **Feature requests**: Describe the desired functionality  
- ❓ **Usage questions**: Check examples and API documentation first

## 📜 License

MIT License - see LICENSE file for details.

---

**Happy Neural Rendering! 🎨🚀**

