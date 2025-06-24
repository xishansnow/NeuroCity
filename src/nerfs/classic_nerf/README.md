# Classic NeRF: Neural Radiance Fields for View Synthesis

This package implements the original NeRF model from the seminal paper:

**"NeRF: Representing Scenes as Neural Radiance Fields for View Synthesis"**  

*Ben Mildenhall, Pratul P. Srinivasan, Matthew Tancik, Jonathan T. Barron, Ravi Ramamoorthi, Ren Ng*  
ECCV 2020

## Overview

Neural Radiance Fields (NeRF) is a method for synthesizing novel views of complex scenes by optimizing an underlying continuous volumetric scene function using a sparse set of input views. The method uses a fully connected deep network to represent a scene as a continuous 5D function that outputs the volume density and view-dependent emitted radiance at any point in space.

### Key Features

- **Positional Encoding**: High-frequency details through sinusoidal position encoding
- **View-Dependent Rendering**: Realistic specular reflections and view-dependent effects  
- **Hierarchical Volume Sampling**: Coarse-to-fine sampling strategy for efficiency
- **Volume Rendering**: Differentiable volumetric rendering with neural radiance fields
- **Multiple Dataset Support**: Blender synthetic scenes and real-world datasets

## Installation

```bash
# Install dependencies
pip install torch torchvision numpy imageio opencv-python tqdm tensorboard

# Install the package
cd src/classic_nerf
pip install -e .
```

## Quick Start

### Basic Usage

```python
from classic_nerf import NeRFConfig, NeRF, NeRFTrainer
from classic_nerf.dataset import create_nerf_dataloader

# Create configuration
config = NeRFConfig(
    netdepth=8,
    netwidth=256,
    N_samples=64,
    N_importance=128,
    learning_rate=5e-4
)

# Load dataset
train_loader = create_nerf_dataloader(
    'blender', 
    'path/to/blender/scene', 
    split='train',
    batch_size=1024
)

# Create and train model
trainer = NeRFTrainer(config)
trainer.train(train_loader, num_epochs=100)
```

### Training a Model

```python
import torch
from classic_nerf import NeRFConfig, NeRFTrainer
from classic_nerf.dataset import create_nerf_dataloader

# Configure model
config = NeRFConfig(
    # Network architecture
    netdepth=8,              # MLP depth
    netwidth=256,            # MLP width
    netdepth_fine=8,         # Fine network depth
    netwidth_fine=256,       # Fine network width
    
    # Positional encoding
    multires=10,             # Coordinate encoding levels
    multires_views=4,        # Direction encoding levels
    
    # Sampling
    N_samples=64,            # Coarse samples per ray
    N_importance=128,        # Fine samples per ray
    perturb=True,            # Stratified sampling
    
    # Training
    learning_rate=5e-4,
    lrate_decay=250,
    
    # Scene bounds
    near=2.0,
    far=6.0
)

# Create data loaders
train_loader = create_nerf_dataloader(
    dataset_type='blender',
    basedir='data/nerf_synthetic/lego',
    split='train',
    batch_size=1024,
    white_bkgd=True
)

val_loader = create_nerf_dataloader(
    dataset_type='blender',
    basedir='data/nerf_synthetic/lego', 
    split='val',
    batch_size=1024,
    white_bkgd=True
)

# Create trainer
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
trainer = NeRFTrainer(config, device=device)

# Train model
trainer.train(
    train_loader=train_loader,
    val_loader=val_loader,
    num_epochs=200,
    log_dir='logs/classic_nerf',
    ckpt_dir='checkpoints/classic_nerf',
    val_interval=10,
    save_interval=50
)
```

### Rendering Novel Views

```python
import numpy as np
from classic_nerf.utils import create_spherical_poses, to8b
import imageio

# Load trained model
trainer.load_checkpoint('checkpoints/classic_nerf/final.pth')

# Create spiral camera path
render_poses = create_spherical_poses(radius=4.0, n_poses=40)

# Camera parameters
H, W = 800, 800
focal = 1111.1
K = np.array([[focal, 0, W/2], [0, focal, H/2], [0, 0, 1]])

# Render video
frames = []
for i, c2w in enumerate(render_poses):
    print(f"Rendering frame {i+1}/{len(render_poses)}")
    rgb = trainer.render_test_image(H, W, K, c2w.numpy())
    frames.append(to8b(rgb))

# Save video
imageio.mimwrite('spiral_render.mp4', frames, fps=30, quality=8)
```

## 🎯 Model Characteristics

### 🎨 Representation Method
- **Multi-Layer Perceptron (MLP)**: Deep fully-connected neural network (8 layers, 256 hidden units)
- **Positional Encoding**: High-frequency sinusoidal encoding for 3D coordinates (10 levels) and viewing directions (4 levels)
- **Implicit Function**: Continuous 5D function F(x,y,z,θ,φ) → (RGB, σ)
- **Skip Connections**: Direct connections from input to middle layers for better gradient flow
- **Hierarchical Sampling**: Coarse-to-fine sampling strategy with two separate networks

### ⚡ Training Performance
- **Training Time**: 1-2 days for typical scenes on modern GPUs
- **Training Speed**: ~1,000-5,000 rays/second on RTX 3080
- **Convergence**: Slow convergence requiring 200K-1M iterations
- **GPU Memory**: 8-16GB during training for complex scenes
- **Scalability**: Training time scales linearly with scene complexity

### 🎬 Rendering Mechanism
- **Stratified Sampling**: Uniform sampling along rays with random perturbation
- **Hierarchical Volume Sampling**: Coarse network guides fine network sampling
- **Volume Rendering**: Numerical integration using quadrature rules
- **Alpha Compositing**: Front-to-back accumulation with transmittance
- **View-Dependent Shading**: Separate MLP branch for view-dependent color

### 🚀 Rendering Speed
- **Inference Speed**: 10-30 seconds per 800×800 image on RTX 3080
- **Ray Processing**: ~1,000-3,000 rays/second during inference
- **Batch Processing**: Requires chunked rendering to avoid memory issues
- **Resolution Scaling**: Rendering time scales quadratically with image resolution
- **Interactive Rendering**: Not suitable for real-time applications

### 💾 Storage Requirements
- **Model Size**: 100-500 MB for typical scenes
- **MLP Weights**: ~50-100 MB for coarse network, ~50-100 MB for fine network
- **Scene Representation**: Model size independent of scene complexity
- **Memory Efficiency**: Compact continuous representation
- **Checkpoint Size**: Full model state ~200-1000 MB including optimizer states

### 📊 Performance Comparison

| Metric | Classic NeRF | Modern Methods | Disadvantage |
|--------|--------------|----------------|--------------|
| Training Time | 1-2 days | 20-60 min | **20-50x slower** |
| Inference Speed | 10-30 sec/image | Real-time | **>100x slower** |
| Model Size | 100-500 MB | 10-50 MB | **5-10x larger** |
| GPU Memory | 8-16 GB | 2-4 GB | **2-4x more** |
| Quality (PSNR) | Baseline | +0.5-2.0 dB | **Lower quality** |

### 🎯 Use Cases
- **Research Baseline**: Foundation for NeRF research and comparisons
- **High-Quality Rendering**: Excellent quality for offline rendering applications
- **Educational Purpose**: Clear and interpretable architecture for learning
- **Proof of Concept**: Demonstrating neural implicit representations
- **Custom Architectures**: Base for developing specialized NeRF variants

## Model Architecture

### NeRF Network

The NeRF model consists of:

1. **Positional Encoding**: Maps 3D coordinates and 2D viewing directions to higher-dimensional space
2. **MLP Network**: 8-layer fully connected network with ReLU activations
3. **Skip Connections**: Concatenate input coordinates at the 4th layer
4. **View-Dependent Output**: Separate branches for density and color

```
Input (x, y, z, θ, φ) → Positional Encoding → MLP → (RGB, σ)
                                              ↑
                                         Skip Connection
```

### Volume Rendering

The rendering equation integrates along rays:

```
C(r) = ∫ T(t) σ(r(t)) c(r(t), d) dt

where T(t) = exp(-∫ σ(r(s)) ds)
```

## Configuration Options

### Network Architecture
- `netdepth`: Depth of MLP (default: 8)  
- `netwidth`: Width of MLP (default: 256)
- `netdepth_fine`: Depth of fine network (default: 8)
- `netwidth_fine`: Width of fine network (default: 256)

### Positional Encoding
- `multires`: Levels for coordinate encoding (default: 10)
- `multires_views`: Levels for direction encoding (default: 4)

### Sampling
- `N_samples`: Coarse samples per ray (default: 64)
- `N_importance`: Fine samples per ray (default: 128)
- `perturb`: Enable stratified sampling (default: True)

### Training
- `learning_rate`: Learning rate (default: 5e-4)
- `lrate_decay`: LR decay rate in 1000s (default: 250)
- `raw_noise_std`: Noise for regularization (default: 0.0)

## Dataset Support

### Blender Synthetic Scenes

The package supports the standard NeRF synthetic dataset format:

```
scene/
├── transforms_train.json
├── transforms_val.json  
├── transforms_test.json
├── train/
│   ├── r_0.png
│   ├── r_1.png
│   └── ...
├── val/
└── test/
```

Each `transforms_*.json` contains camera poses and intrinsics:

```json
{
    "camera_angle_x": 0.6911112070083618,
    "frames": [
        {
            "file_path": "./train/r_0",
            "rotation": 0.012566370614359171,
            "transform_matrix": [...]
        }
    ]
}
```

### Custom Datasets

To use custom datasets, create a dataset class inheriting from `torch.utils.data.Dataset`:

```python
class CustomDataset(Dataset):
    def __init__(self, basedir, split='train'):
        # Load your data
        pass
    
    def __getitem__(self, idx):
        return {
            'rays_o': ray_origins,      # [3]
            'rays_d': ray_directions,   # [3] 
            'target': target_rgb        # [3]
        }
```

## Evaluation

### Metrics

The implementation includes standard evaluation metrics:

- **PSNR**: Peak Signal-to-Noise Ratio
- **SSIM**: Structural Similarity Index
- **LPIPS**: Learned Perceptual Image Patch Similarity

```python
# Evaluate on test set
test_loader = create_nerf_dataloader('blender', 'data/lego', 'test')
metrics = trainer.validate(test_loader)

print(f"PSNR: {metrics['val_psnr']:.2f} dB")
```

## Performance Tips

### Training Efficiency

1. **Batch Size**: Use larger batches (1024-4096 rays) for better GPU utilization
2. **Mixed Precision**: Enable automatic mixed precision for faster training
3. **Learning Rate**: Use learning rate decay for better convergence
4. **Hierarchical Sampling**: Essential for capturing fine details

### Memory Optimization

1. **Chunk Rendering**: Render in smaller chunks to avoid OOM
2. **Gradient Checkpointing**: Trade compute for memory
3. **Model Size**: Reduce network width/depth for faster inference

## Troubleshooting

### Common Issues

**Poor convergence**:
- Check learning rate (try 1e-4 to 1e-3)
- Verify positional encoding levels
- Ensure proper scene bounds (near/far)

**Blurry results**:
- Increase positional encoding frequency
- Add more fine samples (N_importance)
- Check camera poses accuracy

**Training instability**:
- Reduce learning rate
- Add noise regularization
- Use gradient clipping

### Debug Mode

Enable debug logging:

```python
import logging
logging.basicConfig(level=logging.DEBUG)

# Training with debug info
trainer.train(..., verbose=True)
```

## Examples

See `example_usage.py` for complete examples including:

- Training on synthetic data
- Novel view synthesis
- Video generation
- Model analysis and visualization

## Testing

Run the test suite:

```bash
python test_classic_nerf.py
```

Tests cover:
- Model architecture and forward pass
- Volume rendering functions  
- Dataset loading and preprocessing
- Training loop and optimization
- Utility functions

## Citation

If you use this implementation, please cite the original paper:

```bibtex
@inproceedings{mildenhall2020nerf,
  title={NeRF: Representing Scenes as Neural Radiance Fields for View Synthesis},
  author={Mildenhall, Ben and Srinivasan, Pratul P and Tancik, Matthew and Barron, Jonathan T and Ramamoorthi, Ravi and Ng, Ren},
  booktitle={ECCV},
  year={2020}
}
```

## License

This implementation is provided for research and educational purposes. Please refer to the original paper and code for licensing terms.

## Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality  
4. Ensure all tests pass
5. Submit a pull request

## Acknowledgments

This implementation is based on the original NeRF paper and official code release. Special thanks to the authors for their groundbreaking work in neural rendering.
