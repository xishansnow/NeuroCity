"""
Plenoxels Refactored Package

This module provides a clean, user-friendly interface for Plenoxels with
separate training and inference classes, following the same pattern as SVRaster.

Key Features:
- PlenoxelTrainer: Dedicated training class with advanced features
- PlenoxelRenderer: Optimized inference class for high-quality rendering
- Clean separation of concerns between training and inference
- Consistent API design with other NeRF implementations
- Professional configuration management
- Comprehensive documentation and examples

Usage Examples:

Training:
```python
from nerfs.plenoxels import PlenoxelTrainer, PlenoxelTrainingConfig

# Configure training
config = PlenoxelTrainingConfig(
    grid_resolution=(256, 256, 256),
    num_epochs=10000,
    use_coarse_to_fine=True,
)

# Create and train
trainer = PlenoxelTrainer(config, train_dataset, val_dataset)
renderer = trainer.train()  # Returns trained renderer
```

Inference:
```python
from nerfs.plenoxels import PlenoxelRenderer

# Load from checkpoint
renderer = PlenoxelRenderer.from_checkpoint("path/to/checkpoint.pth")

# Render image
image = renderer.render_image(height=512, width=512, camera_matrix, camera_pose)

# Render video
frames = renderer.render_video(camera_trajectory, height=512, width=512)
```
"""

from .config import (
    PlenoxelConfig,
    PlenoxelTrainingConfig,
    PlenoxelInferenceConfig,
    TrainingConfig,  # Alias
    InferenceConfig,  # Alias
)

from .trainer import (
    PlenoxelTrainer,
    create_plenoxel_trainer,
)

from .renderer import (
    PlenoxelRenderer,
    create_plenoxel_renderer,
)

# Keep existing imports for backward compatibility
from .core import (
    PlenoxelModel,
    PlenoxelLoss,
    VoxelGrid,
    VolumetricRenderer,
    SphericalHarmonics,
    trilinear_interpolation,
)

from .dataset import (
    PlenoxelDataset,
    PlenoxelDatasetConfig,
    create_plenoxel_dataloader,
    create_plenoxel_dataset,
)

# Version information
__version__ = "2.0.0"
__author__ = "NeuroCity Development Team"

# Main exports - new interface
__all__ = [
    # New unified interface
    "PlenoxelTrainer",
    "PlenoxelRenderer", 
    "create_plenoxel_trainer",
    "create_plenoxel_renderer",
    
    # Configuration classes
    "PlenoxelConfig",
    "PlenoxelTrainingConfig", 
    "PlenoxelInferenceConfig",
    "TrainingConfig",
    "InferenceConfig",
    
    # Core components
    "PlenoxelModel",
    "PlenoxelLoss",
    "VoxelGrid",
    "VolumetricRenderer",
    "SphericalHarmonics",
    "trilinear_interpolation",
    
    # Dataset utilities
    "PlenoxelDataset",
    "PlenoxelDatasetConfig", 
    "create_plenoxel_dataloader",
    "create_plenoxel_dataset",
]

# Configuration for easy access
DEFAULT_TRAINING_CONFIG = PlenoxelTrainingConfig()
DEFAULT_INFERENCE_CONFIG = PlenoxelInferenceConfig()


def quick_train(
    train_dataset,
    val_dataset=None,
    config=None,
    **kwargs
) -> PlenoxelRenderer:
    """Quick training function with sensible defaults.
    
    Args:
        train_dataset: Training dataset
        val_dataset: Optional validation dataset
        config: Optional training configuration
        **kwargs: Additional configuration parameters
    
    Returns:
        PlenoxelRenderer: Trained renderer ready for inference
    """
    if config is None:
        config = PlenoxelTrainingConfig(**kwargs)
    
    trainer = PlenoxelTrainer(config, train_dataset, val_dataset)
    return trainer.train()


def quick_render(
    checkpoint_path: str,
    height: int = 512,
    width: int = 512,
    camera_matrix=None,
    camera_pose=None,
    config=None,
) -> dict:
    """Quick rendering function with sensible defaults.
    
    Args:
        checkpoint_path: Path to trained model checkpoint
        height: Image height
        width: Image width  
        camera_matrix: Camera intrinsic matrix
        camera_pose: Camera pose matrix
        config: Optional inference configuration
    
    Returns:
        Dictionary containing rendered outputs
    """
    renderer = PlenoxelRenderer.from_checkpoint(checkpoint_path, config)
    
    if camera_matrix is None or camera_pose is None:
        raise ValueError("Camera parameters required for rendering")
    
    return renderer.render_image(height, width, camera_matrix, camera_pose)


# Example configurations for common use cases
class ExampleConfigs:
    """Pre-defined configurations for common scenarios."""
    
    @staticmethod
    def fast_training() -> PlenoxelTrainingConfig:
        """Configuration optimized for fast training."""
        return PlenoxelTrainingConfig(
            grid_resolution=(128, 128, 128),
            num_epochs=5000,
            use_coarse_to_fine=True,
            coarse_resolutions=[(64, 64, 64), (128, 128, 128)],
            coarse_epochs=[1000, 5000],
            batch_size=8192,
            pruning_interval=500,
        )
    
    @staticmethod
    def high_quality_training() -> PlenoxelTrainingConfig:
        """Configuration optimized for high quality results."""
        return PlenoxelTrainingConfig(
            grid_resolution=(512, 512, 512),
            num_epochs=20000,
            use_coarse_to_fine=True,
            coarse_resolutions=[(128, 128, 128), (256, 256, 256), (512, 512, 512)],
            coarse_epochs=[2000, 8000, 20000],
            batch_size=4096,
            tv_lambda=1e-5,
            l1_lambda=1e-7,
        )
    
    @staticmethod
    def fast_inference() -> PlenoxelInferenceConfig:
        """Configuration optimized for fast inference."""
        return PlenoxelInferenceConfig(
            high_quality=False,
            num_samples=32,
            chunk_size=16384,
            use_half_precision=True,
        )
    
    @staticmethod
    def high_quality_inference() -> PlenoxelInferenceConfig:
        """Configuration optimized for high quality inference."""
        return PlenoxelInferenceConfig(
            high_quality=True,
            max_samples_per_ray=128,
            adaptive_sampling=True,
            render_depth=True,
            render_normals=True,
            tile_size=256,  # Smaller tiles for high quality
        )


# Convenience imports at package level
ExampleConfigs = ExampleConfigs 