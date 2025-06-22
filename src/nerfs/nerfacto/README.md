# Nerfacto - Neural Radiance Fields

Nerfacto is a modern implementation of Neural Radiance Fields (NeRF) that combines the best practices from recent NeRF research, including Instant-NGP's hash encoding, proposal networks, and advanced training techniques.

## Features

- **Fast Training**: Based on Instant-NGP with hash-based spatial encoding
- **High Quality**: State-of-the-art rendering quality for real-world scenes
- **Multiple Data Formats**: Support for COLMAP, Blender, and Instant-NGP data formats
- **Modern Training**: Mixed precision, gradient accumulation, progressive training
- **Flexible Architecture**: Configurable network architecture and training parameters
- **Comprehensive Evaluation**: Built-in metrics (PSNR, SSIM, LPIPS) and visualization tools

## Installation

### Prerequisites

- Python 3.8+
- PyTorch 1.12+
- CUDA 11.3+ (for GPU acceleration)

### Dependencies

```bash
pip install torch torchvision torchaudio
pip install numpy opencv-python pillow
pip install tqdm tensorboard wandb
pip install scipy matplotlib
```

## Quick Start

### 1. Basic Usage

```python
from src.nerfacto import NerfactoModel, NerfactoConfig, NerfactoTrainer
from src.nerfacto.dataset import NerfactoDatasetConfig

# Create model configuration
model_config = NerfactoConfig(
    num_levels=16,
    base_resolution=16,
    max_resolution=2048,
    features_per_level=2
)

# Create dataset configuration
dataset_config = NerfactoDatasetConfig(
    data_dir="path/to/your/data",
    data_format="colmap"  # or "blender", "instant_ngp"
)

# Create and train model
trainer = NerfactoTrainer(model_config, dataset_config)
trainer.train()
```

### 2. Command Line Training

```bash
python -m src.nerfacto.example_usage \
    --data_dir /path/to/data \
    --data_format colmap \
    --output_dir outputs \
    --experiment_name my_scene \
    --max_epochs 30000
```

### 3. Evaluation Only

```bash
python -m src.nerfacto.example_usage \
    --data_dir /path/to/data \
    --eval_only \
    --checkpoint_path outputs/my_scene/checkpoints/best_model.pth
```

## Data Formats

### COLMAP Format

Your data directory should contain:
```
data/
├── images/           # RGB images
├── cameras.txt       # Camera intrinsics
├── images.txt        # Camera poses
└── points3D.txt      # 3D points (optional)
```

### Blender Format

Your data directory should contain:
```
data/
├── images/           # RGB images
├── transforms_train.json  # Training camera poses
├── transforms_val.json    # Validation camera poses
└── transforms_test.json   # Test camera poses
```

### Instant-NGP Format

Your data directory should contain:
```
data/
├── images/           # RGB images
└── transforms.json   # Camera poses and intrinsics
```

## Model Architecture

Nerfacto uses a modern NeRF architecture with:

- **Hash Encoding**: Multi-resolution hash grids for spatial features
- **Proposal Networks**: Coarse-to-fine sampling strategy
- **View-Dependent Colors**: Spherical harmonics or MLP-based view dependence
- **Regularization**: Various regularization techniques for stable training

### Key Components

1. **HashEncoder**: Multi-level hash encoding for spatial coordinates
2. **MLPHead**: Neural network for density and feature prediction
3. **ColorNet**: View-dependent color prediction
4. **ProposalNetworks**: Hierarchical sampling guidance
5. **VolumetricRenderer**: Ray marching and alpha compositing

## Configuration Options

### Model Configuration

```python
@dataclass
class NerfactoConfig:
    # Hash encoding
    num_levels: int = 16
    base_resolution: int = 16
    max_resolution: int = 2048
    features_per_level: int = 2
    
    # MLP architecture
    hidden_dim: int = 64
    num_layers: int = 2
    
    # Rendering
    num_samples_coarse: int = 48
    num_samples_fine: int = 48
    
    # Loss configuration
    use_proposal_loss: bool = True
    proposal_loss_weight: float = 1.0
```

### Training Configuration

```python
@dataclass
class NerfactoTrainerConfig:
    # Training settings
    max_epochs: int = 30000
    learning_rate: float = 5e-4
    batch_size: int = 1
    
    # Mixed precision
    use_mixed_precision: bool = True
    gradient_clip_val: float = 1.0
    
    # Progressive training
    use_progressive_training: bool = True
    progressive_levels: List[int] = [64, 128, 256, 512]
    
    # Evaluation
    eval_every_n_epochs: int = 1000
    save_every_n_epochs: int = 5000
```

## Training Tips

### 1. Data Preparation

- Ensure images are properly calibrated
- Use sufficient camera pose diversity
- Consider image resolution vs. training time trade-off

### 2. Hyperparameter Tuning

- Start with default parameters
- Adjust `max_resolution` based on scene complexity
- Increase `num_levels` for very detailed scenes
- Use `progressive_training` for faster convergence

### 3. Performance Optimization

- Use mixed precision training (`use_mixed_precision=True`)
- Adjust batch size based on GPU memory
- Enable gradient accumulation for larger effective batch sizes

## Evaluation Metrics

Nerfacto provides comprehensive evaluation:

- **PSNR**: Peak Signal-to-Noise Ratio
- **SSIM**: Structural Similarity Index
- **LPIPS**: Learned Perceptual Image Patch Similarity
- **Depth Metrics**: For depth supervision (if available)

## Rendering

### Novel View Synthesis

```python
# Load trained model
model = NerfactoModel.load_from_checkpoint("path/to/checkpoint.pth")

# Generate novel views
camera_poses = create_spiral_path(center, radius, num_views)
rendered_images = model.render_views(camera_poses, intrinsics)
```

### Export Results

The trainer automatically saves:
- Model checkpoints
- Training logs (TensorBoard)
- Evaluation metrics
- Rendered validation images

## Advanced Features

### 1. Custom Data Loaders

```python
class CustomDataset(NerfactoDataset):
    def __init__(self, config):
        super().__init__(config)
        # Custom implementation
    
    def _load_data(self):
        # Load your custom data format
        pass
```

### 2. Model Customization

```python
class CustomNerfacto(NerfactoModel):
    def __init__(self, config):
        super().__init__(config)
        # Add custom components
        self.custom_module = CustomModule()
    
    def forward(self, ray_origins, ray_directions):
        # Custom forward pass
        pass
```

### 3. Loss Function Modification

```python
class CustomLoss(NerfactoLoss):
    def forward(self, outputs, targets):
        losses = super().forward(outputs, targets)
        # Add custom loss terms
        losses['custom_loss'] = self.compute_custom_loss(outputs, targets)
        return losses
```

## Troubleshooting

### Common Issues

1. **CUDA Out of Memory**
   - Reduce batch size or image resolution
   - Use gradient accumulation
   - Enable mixed precision training

2. **Poor Convergence**
   - Check camera pose quality
   - Adjust learning rate
   - Enable progressive training

3. **Blurry Results**
   - Increase model capacity (`hidden_dim`, `num_layers`)
   - Use higher resolution hash grids
   - Check data quality

### Performance Tips

- Use SSD for data storage
- Optimize data loading (`num_workers`)
- Monitor GPU utilization
- Use appropriate precision (FP16/FP32)

## Citation

If you use Nerfacto in your research, please cite:

```bibtex
@article{nerfacto2023,
  title={Nerfacto: Modern Neural Radiance Fields},
  author={Your Name},
  year={2023}
}
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Instant-NGP for the hash encoding implementation
- NeRF for the original neural radiance fields concept
- Nerfstudio for inspiration and best practices
- The broader NeRF research community 