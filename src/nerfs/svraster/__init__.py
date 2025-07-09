"""
SVRaster: Sparse Voxel Radiance Fields

A fast and efficient implementation of neural radiance fields using sparse voxel grids.
This package provides both training and inference capabilities with CUDA acceleration.

Key Components:
    - Core: Main model and configuration classes
    - Trainer: Training pipeline for volume rendering
    - Renderer: Inference pipeline with rasterization
    - Dataset: Data loading and preprocessing utilities
    - Utils: Helper functions and utilities
    - CUDA: GPU-accelerated implementations

Example Usage:
    ```python
    import svraster

    # Configure and create model
    config = svraster.SVRasterConfig(
        max_octree_levels=8,
        base_resolution=128,
        sh_degree=2
    )
    model = svraster.SVRasterModel(config)

    # Training
    trainer_config = svraster.SVRasterTrainerConfig()
    trainer = svraster.SVRasterTrainer(model, trainer_config)
    trainer.train(dataset)

    # Inference
    renderer_config = svraster.SVRasterRendererConfig()
    renderer = svraster.SVRasterRenderer(model, renderer_config)
    rendered_image = renderer.render(camera_pose)
    ```
"""

__version__ = "1.0.0"
__author__ = "NeuroCity Team"

# Core components
from .core import (
    SVRasterModel,
    SVRasterConfig,
    SVRasterLoss,
)

# Training components
from .trainer import (
    SVRasterTrainer,
    SVRasterTrainerConfig,
)

# Rendering components
from .renderer import (
    SVRasterRenderer,
    SVRasterRendererConfig,
    VoxelRasterizerConfig,
)

# Volume rendering (for training)
from .volume_renderer import VolumeRenderer

# True rasterization (for inference)
from .voxel_rasterizer import VoxelRasterizer

# Dataset utilities
from .dataset import (
    SVRasterDataset,
    SVRasterDatasetConfig,
)

# Spherical harmonics
from .spherical_harmonics import eval_sh_basis

# Utilities
from .utils import (
    morton_encode_3d,
    morton_decode_3d,
    octree_subdivision,
    octree_pruning,
    ray_direction_dependent_ordering,
    depth_peeling,
    voxel_pruning,
)

# CUDA/GPU acceleration (optional)
try:
    from .cuda import (
        SVRasterGPU,
        SVRasterGPUTrainer,
        EMAModel,
    )
    CUDA_AVAILABLE = True
    _cuda_components = ["SVRasterGPU", "SVRasterGPUTrainer", "EMAModel"]
except ImportError:
    CUDA_AVAILABLE = False
    _cuda_components = []

# All public API exports
__all__ = [
    # Core
    "SVRasterModel",
    "SVRasterConfig", 
    "SVRasterLoss",
    
    # Training
    "SVRasterTrainer",
    "SVRasterTrainerConfig",
    "VolumeRenderer",
    
    # Rendering
    "SVRasterRenderer",
    "SVRasterRendererConfig",
    "VoxelRasterizer",
    "VoxelRasterizerConfig",
    
    # Dataset
    "SVRasterDataset",
    "SVRasterDatasetConfig",
    
    # Spherical harmonics
    "eval_sh_basis",
    
    # Utilities
    "morton_encode_3d",
    "morton_decode_3d",
    "octree_subdivision",
    "octree_pruning", 
    "ray_direction_dependent_ordering",
    "depth_peeling",
    "voxel_pruning",
    
    # Package info
    "__version__",
    "CUDA_AVAILABLE",
] + _cuda_components


def get_device_info():
    """Get information about available compute devices."""
    import torch
    
    info = {
        "torch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": torch.version.cuda if torch.cuda.is_available() else None,
        "device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
        "svraster_cuda": CUDA_AVAILABLE,
    }
    
    if torch.cuda.is_available():
        info["devices"] = []
        for i in range(torch.cuda.device_count()):
            device_props = torch.cuda.get_device_properties(i)
            info["devices"].append({
                "name": device_props.name,
                "memory_total": device_props.total_memory // (1024**3),  # GB
                "compute_capability": f"{device_props.major}.{device_props.minor}",
            })
    
    return info


def quick_start_guide():
    """Print a quick start guide for SVRaster."""
    guide = """
SVRaster Quick Start Guide
==========================

1. Basic Training:
   ```python
   import svraster

   # Load data
   dataset_config = svraster.SVRasterDatasetConfig(data_dir="path/to/data")
   dataset = svraster.SVRasterDataset(dataset_config)
   
   # Configure model
   model_config = svraster.SVRasterConfig(
       max_octree_levels=8,
       base_resolution=128,
       sh_degree=2
   )
   model = svraster.SVRasterModel(model_config)
   
   # Train
   trainer_config = svraster.SVRasterTrainerConfig(num_epochs=100)
   trainer = svraster.SVRasterTrainer(model, trainer_config)
   trainer.train(dataset)
   ```

2. Inference/Rendering:
   ```python
   # Load trained model
   model = svraster.SVRasterModel.load("checkpoint.pth")
   
   # Render
   renderer_config = svraster.SVRasterRendererConfig()
   renderer = svraster.SVRasterRenderer(model, renderer_config)
   image = renderer.render(camera_pose, image_size=(800, 800))
   ```

3. GPU Acceleration (if available):
   ```python
   if svraster.CUDA_AVAILABLE:
       gpu_model = svraster.SVRasterGPU(model_config)
       gpu_trainer = svraster.SVRasterGPUTrainer(gpu_model, trainer_config)
   ```

For more examples, see the demos/ directory.
"""
    print(guide)


# Convenience function for checking compatibility
def check_compatibility():
    """Check system compatibility and suggest optimizations."""
    import torch
    
    print("SVRaster Compatibility Check")
    print("=" * 30)
    
    device_info = get_device_info()
    
    print(f"PyTorch version: {device_info['torch_version']}")
    print(f"CUDA available: {device_info['cuda_available']}")
    
    if device_info['cuda_available']:
        print(f"CUDA version: {device_info['cuda_version']}")
        print(f"GPU devices: {device_info['device_count']}")
        for i, device in enumerate(device_info['devices']):
            print(f"  Device {i}: {device['name']} ({device['memory_total']}GB)")
        print(f"SVRaster CUDA: {device_info['svraster_cuda']}")
        
        if not device_info['svraster_cuda']:
            print("\nWarning: SVRaster CUDA extensions not available.")
            print("To enable GPU acceleration, run: python setup.py build_ext --inplace")
    else:
        print("No CUDA devices found. SVRaster will run on CPU.")
        
    print("\nRecommendations:")
    if device_info['cuda_available'] and device_info['svraster_cuda']:
        print("✓ GPU acceleration fully available")
    elif device_info['cuda_available']:
        print("⚠ GPU detected but CUDA extensions not built")
        print("  Run: cd cuda && python setup.py build_ext --inplace")
    else:
        print("⚠ No GPU acceleration available")
        print("  Consider using a CUDA-enabled environment for better performance")