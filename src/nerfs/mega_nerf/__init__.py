"""
Mega-NeRF: Scalable Construction of Large-Scale NeRFs for Virtual Fly-Throughs

A scalable implementation of Neural Radiance Fields for large-scale scenes.
This package provides both training and inference capabilities with spatial partitioning.

Key Components:
    - Core: Main model and configuration classes
    - Trainer: Training pipeline with parallel submodule training
    - Renderer: Inference pipeline with volume rendering
    - Dataset: Data loading and spatial partitioning utilities
    - Utils: Helper functions and utilities

Example Usage:
    ```python
    import mega_nerf

    # Configure and create model
    config = mega_nerf.MegaNeRFConfig(
        num_submodules=8,
        grid_size=(4, 2),
        hidden_dim=256
    )
    model = mega_nerf.MegaNeRF(config)

    # Training
    dataset_config = mega_nerf.MegaNeRFDatasetConfig(data_root="path/to/data")
    dataset = mega_nerf.MegaNeRFDataset(dataset_config)

    trainer_config = mega_nerf.MegaNeRFTrainerConfig()
    trainer = mega_nerf.MegaNeRFTrainer(model, dataset, trainer_config)
    trainer.train_sequential()

    # Inference
    renderer_config = mega_nerf.MegaNeRFRendererConfig()
    renderer = mega_nerf.MegaNeRFRenderer(model, renderer_config)
    rendered_image = renderer.render_image(camera_pose, intrinsics)
    ```
"""

__version__ = "1.0.0"
__author__ = "NeuroCity Team"

# Core components
from .core import (
    MegaNeRF,
    MegaNeRFConfig,
    MegaNeRFSubmodule,
    PositionalEncoding,
)

# Training components
from .trainer import (
    MegaNeRFTrainer,
    MegaNeRFTrainerConfig,
)

# Rendering components
from .renderer import (
    MegaNeRFRenderer,
    MegaNeRFRendererConfig,
    create_mega_nerf_renderer,
    render_demo_images,
)

# Dataset components
from .dataset import (
    MegaNeRFDataset,
    MegaNeRFDatasetConfig,
    CameraDataset,
    CameraInfo,
)

# Spatial partitioning
from .spatial_partitioner import (
    SpatialPartitioner,
    GeometryAwarePartitioner,
)

# Volume rendering
from .volumetric_renderer import (
    VolumetricRenderer,
    BatchRenderer,
)

# Utilities
from .utils import (
    # Camera utilities
    generate_rays,
    create_spiral_path,
    interpolate_poses,
    generate_random_poses,
    # Rendering utilities
    save_image,
    save_video,
    create_depth_visualization,
    compute_psnr,
    compute_ssim,
    compute_lpips,
    # I/O utilities
    save_checkpoint,
    load_checkpoint,
    save_config,
    load_config,
    # Spatial partitioning utilities
    compute_scene_bounds,
    create_spatial_grid,
    assign_points_to_cells,
)

# All public API exports
__all__ = [
    # Core
    "MegaNeRF",
    "MegaNeRFConfig",
    "MegaNeRFSubmodule",
    "PositionalEncoding",
    # Training
    "MegaNeRFTrainer",
    "MegaNeRFTrainerConfig",
    # Rendering
    "MegaNeRFRenderer",
    "MegaNeRFRendererConfig",
    "create_mega_nerf_renderer",
    "render_demo_images",
    # Dataset
    "MegaNeRFDataset",
    "MegaNeRFDatasetConfig",
    "CameraDataset",
    "CameraInfo",
    # Spatial partitioning
    "SpatialPartitioner",
    "GeometryAwarePartitioner",
    # Volume rendering
    "VolumetricRenderer",
    "BatchRenderer",
    # Camera utilities
    "generate_rays",
    "create_spiral_path",
    "interpolate_poses",
    "generate_random_poses",
    # Rendering utilities
    "save_image",
    "save_video",
    "create_depth_visualization",
    "compute_psnr",
    "compute_ssim",
    "compute_lpips",
    # I/O utilities
    "save_checkpoint",
    "load_checkpoint",
    "save_config",
    "load_config",
    # Spatial partitioning utilities
    "compute_scene_bounds",
    "create_spatial_grid",
    "assign_points_to_cells",
    # Package info
    "__version__",
]


def get_device_info():
    """Get information about available compute devices."""
    import torch

    info = {
        "torch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": torch.version.cuda if torch.cuda.is_available() else None,
        "device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
    }

    if torch.cuda.is_available():
        info["devices"] = []
        for i in range(torch.cuda.device_count()):
            device_props = torch.cuda.get_device_properties(i)
            info["devices"].append(
                {
                    "name": device_props.name,
                    "memory_total": device_props.total_memory // (1024**3),  # GB
                    "compute_capability": f"{device_props.major}.{device_props.minor}",
                }
            )

    return info


def quick_start_guide():
    """Print a quick start guide for Mega-NeRF."""
    guide = """
Mega-NeRF Quick Start Guide
===========================

1. Basic Training:
   ```python
   import mega_nerf

   # Load data
   dataset_config = mega_nerf.MegaNeRFDatasetConfig(data_root="path/to/data")
   dataset = mega_nerf.MegaNeRFDataset(dataset_config)
   
   # Configure model
   model_config = mega_nerf.MegaNeRFConfig(
       num_submodules=8,
       grid_size=(4, 2),
       hidden_dim=256
   )
   model = mega_nerf.MegaNeRF(model_config)
   
   # Train
   trainer_config = mega_nerf.MegaNeRFTrainerConfig()
   trainer = mega_nerf.MegaNeRFTrainer(model, dataset, trainer_config)
   trainer.train_sequential()
   ```

2. Inference/Rendering:
   ```python
   # Load trained model
   renderer = mega_nerf.create_mega_nerf_renderer("checkpoint.pth")
   
   # Render single image
   camera_pose = torch.eye(4)
   intrinsics = torch.tensor([[800, 0, 400], [0, 800, 300], [0, 0, 1]])
   result = renderer.render_image(camera_pose, intrinsics)
   
   # Render video
   center = torch.tensor([0.0, 0.0, 0.0])
   renderer.render_spiral_video(center, radius=2.0, num_frames=100, 
                               intrinsics=intrinsics, output_path="video.mp4")
   ```

3. Spatial Partitioning:
   ```python
   # Create spatial partitioner
   partitioner = mega_nerf.SpatialPartitioner(
       num_partitions=8,
       overlap_factor=0.1
   )
   
   # Use with dataset
   dataset = mega_nerf.MegaNeRFDataset(dataset_config, partitioner=partitioner)
   ```

For more examples and documentation, see the examples/ directory.
"""
    print(guide)


def check_compatibility():
    """Check system compatibility for Mega-NeRF."""
    import torch
    import numpy as np

    issues = []
    warnings = []

    # Check PyTorch version
    if torch.__version__ < "1.9.0":
        issues.append(f"PyTorch version {torch.__version__} is too old. Need >= 1.9.0")

    # Check CUDA availability
    if not torch.cuda.is_available():
        warnings.append("CUDA not available. Training will be slow on CPU.")
    else:
        # Check CUDA version
        cuda_version = torch.version.cuda
        if cuda_version and cuda_version < "11.0":
            warnings.append(f"CUDA version {cuda_version} is old. Consider upgrading to 11.0+")

    # Check NumPy version
    if np.__version__ < "1.19.0":
        issues.append(f"NumPy version {np.__version__} is too old. Need >= 1.19.0")

    # Check optional dependencies
    try:
        import imageio
    except ImportError:
        warnings.append("imageio not available. Video rendering will not work.")

    try:
        import cv2
    except ImportError:
        warnings.append("cv2 not available. Some image processing features will be limited.")

    # Report results
    if issues:
        print("❌ Compatibility Issues:")
        for issue in issues:
            print(f"  - {issue}")

    if warnings:
        print("⚠️  Warnings:")
        for warning in warnings:
            print(f"  - {warning}")

    if not issues and not warnings:
        print("✅ System is compatible with Mega-NeRF")

    return len(issues) == 0
