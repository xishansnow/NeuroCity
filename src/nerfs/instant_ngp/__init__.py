"""
Instant NGP: Instant Neural Graphics Primitives

This package implements the Instant NGP model with clear separation between training and inference:

Training Pipeline:
    - Uses volume rendering for high-quality training
    - Optimized for gradient computation and parameter updates
    - Supports mixed precision training and advanced optimizers

Inference Pipeline:
    - Uses optimized rendering for fast inference
    - Supports hierarchical sampling and early termination
    - Optimized for real-time rendering and batch processing

Key Components:
    - Core: Model definitions and encoders
    - Training: Specialized training pipeline with volume rendering
    - Inference: Optimized inference pipeline for fast rendering
    - Dataset: Data loading and preprocessing utilities
    - Utils: Helper functions and utilities

Example Usage:
    ```python
    import instant_ngp
    
    # Create configuration
    config = instant_ngp.InstantNGPConfig(
        num_levels=16, level_dim=2, base_resolution=16, desired_resolution=2048
    )
    
    # Create model
    model = instant_ngp.InstantNGPModel(config)
    
    # Training
    trainer_config = instant_ngp.InstantNGPTrainerConfig()
    trainer = instant_ngp.InstantNGPTrainer(model, trainer_config)
    trainer.train(train_loader, num_epochs=20)
    
    # Inference
    renderer_config = instant_ngp.InstantNGPRendererConfig()
    renderer = instant_ngp.InstantNGPRenderer(model, renderer_config)
    rendered_image = renderer.render_image(camera_pose, intrinsics, width, height)
    ```
"""

__version__ = "1.0.0"
__author__ = "NeuroCity Team"
__email__ = "team@neurocity.ai"

# Core components
from .core import (
    InstantNGPConfig,
    InstantNGPModel,
    HashEncoder,
    SHEncoder,
    InstantNGPLoss,
    InstantNGPRenderer,  # Legacy renderer from core
)

# Training components
from .trainer import (
    InstantNGPTrainer,
    InstantNGPTrainerConfig,
)

# Inference components
from .renderer import (
    InstantNGPRenderer as InstantNGPInferenceRenderer,
    InstantNGPRendererConfig,
)

# Dataset utilities
from .dataset import (
    InstantNGPDataset,
    InstantNGPDatasetConfig,
    create_instant_ngp_dataloader
)

# Note: Legacy trainer has been removed. Use InstantNGPTrainer from trainer_new instead.

# Utility functions
from .utils import (
    contract_to_unisphere,
    uncontract_from_unisphere,
    morton_encode_3d,
    compute_tv_loss,
    adaptive_sampling,
    estimate_normals,
    compute_hash_grid_size,
)

# CLI tools
from . import cli

# Add alias for backward compatibility
InstantNGP = InstantNGPModel

# CUDA acceleration (optional)
try:
    from .cuda import (
        InstantNGPCUDA,
        hash_encode_cuda,
        volume_render_cuda,
    )
    CUDA_AVAILABLE = True
    _cuda_components = ["InstantNGPCUDA", "hash_encode_cuda", "volume_render_cuda"]
except ImportError:
    CUDA_AVAILABLE = False
    _cuda_components = []

__all__ = [
    # Core components
    "InstantNGPConfig",
    "InstantNGPModel",
    "InstantNGP",  # Alias for backward compatibility
    "HashEncoder", 
    "SHEncoder",
    "InstantNGPLoss",
    "InstantNGPRenderer",  # Legacy renderer
    
    # Training components (new architecture)
    "InstantNGPTrainer",
    "InstantNGPTrainerConfig", 
    
    # Inference components (new architecture)
    "InstantNGPInferenceRenderer",
    "InstantNGPRendererConfig",
    
    # Dataset utilities
    "InstantNGPDataset",
    "InstantNGPDatasetConfig",
    "create_instant_ngp_dataloader",
    
    # Utility functions
    "contract_to_unisphere",
    "uncontract_from_unisphere", 
    "morton_encode_3d",
    "compute_tv_loss",
    "adaptive_sampling",
    "estimate_normals",
    "compute_hash_grid_size",
]

# Add CUDA components if available
if CUDA_AVAILABLE:
    __all__.extend(_cuda_components)

# Expose availability flags
__all__.extend(["CUDA_AVAILABLE"])
