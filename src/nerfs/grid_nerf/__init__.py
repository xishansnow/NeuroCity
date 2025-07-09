"""
from __future__ import annotations

Grid-NeRF: Grid-guided Neural Radiance Fields for Large Urban Scenes

This package implements Grid-guided Neural Radiance Fields, a scalable approach
for rendering large urban environments using hierarchical voxel grids to guide
neural rendering.

Main Components:
- GridNeRF: Main model class
- GridNeRFConfig: Configuration class
- GridNeRFTrainer: Training pipeline
- GridNeRFDataset: Dataset handling
- Utilities for rendering, evaluation, and visualization

Based on the paper: "Grid-guided neural radiance fields for large urban scenes"
"""

from .core import GridNeRF, GridNeRFConfig, GridNeRFRenderer, GridGuidedMLP, HierarchicalGrid
from typing import Tuple


from .dataset import GridNeRFDataset

from .trainer import GridNeRFTrainer

from .logging_utils import setup_logging

# Import additional functions from main utils module
from . import utils as grid_utils

load_config = grid_utils.load_config
save_config = grid_utils.save_config

# Version info
__version__ = "1.0.0"
__author__ = "Grid-NeRF Implementation Team"
__description__ = "Grid-guided Neural Radiance Fields for Large Urban Scenes"

# Package metadata
__all__ = [
    # Core classes
    "GridNeRF",
    "GridNeRFConfig",
    "GridNeRFRenderer",
    "GridGuidedMLP",
    "HierarchicalGrid",
    # Dataset classes
    "GridNeRFDataset",
    # Training
    "GridNeRFTrainer",
    # Utils
    "setup_logging",
    # Utilities - Config
    "load_config",
    "save_config",
]

# Default configuration with CORRECT parameter names
DEFAULT_CONFIG = {
    # Scene bounds (tuple format)
    "scene_bounds": (-100, -100, -10, 100, 100, 50),
    # Grid configuration
    "base_grid_resolution": 64,
    "max_grid_resolution": 512,
    "num_grid_levels": 4,
    "grid_feature_dim": 32,
    # Network architecture
    "mlp_hidden_dim": 256,
    "mlp_num_layers": 4,
    "view_dependent": True,
    "view_embed_dim": 27,
    # Training settings
    "learning_rate": 5e-4,
    "weight_decay": 1e-6,
    # Rendering settings
    "near_plane": 0.1,
    "far_plane": 1000.0,
    "num_samples_coarse": 64,
    "num_samples_fine": 128,
    # Grid update settings
    "grid_update_freq": 100,
    "density_threshold": 0.01,
    # Loss weights
    "color_loss_weight": 1.0,
    "depth_loss_weight": 0.1,
    "grid_regularization_weight": 0.001,
}


def get_default_config() -> dict:
    """Get default Grid-NeRF configuration."""
    return DEFAULT_CONFIG.copy()


def create_grid_nerf_model(config: dict = None, device: str = "cuda") -> GridNeRF:
    """
    Create a Grid-NeRF model with default or custom configuration.

    Args:
        config: Configuration dictionary (uses defaults if None)
        device: Device to create model on

    Returns:
        GridNeRF model instance
    """
    if config is None:
        config = get_default_config()

    grid_config = GridNeRFConfig(**config)
    model = GridNeRF(grid_config)

    if device:
        model = model.to(device)

    return model


def quick_setup(
    data_path: str, output_dir: str, config: dict = None, device: str = "cuda"
) -> tuple[GridNeRF, GridNeRFTrainer, GridNeRFDataset]:
    """
    Quick setup for Grid-NeRF training.

    Args:
        data_path: Path to training data
        output_dir: Output directory for results
        config: Configuration dictionary
        device: Device to use

    Returns:
        Tuple of (model, trainer, dataset)
    """
    import torch

    # Setup configuration
    if config is None:
        config = get_default_config()

    grid_config = GridNeRFConfig(**config)

    # Create model
    model = create_grid_nerf_model(config, device)

    # Create dataset
    dataset = GridNeRFDataset(data_path=data_path, config=grid_config)

    # Create trainer
    trainer = GridNeRFTrainer(
        config=grid_config, output_dir=output_dir, device=torch.device(device)
    )

    return model, trainer, dataset


# Import checks
def check_dependencies():
    """Check if all required dependencies are available."""
    missing_deps = []

    try:
        import torch

        if not torch.cuda.is_available():
            print("Warning: CUDA not available, will use CPU")
    except ImportError:
        missing_deps.append("torch")

    try:
        import numpy
    except ImportError:
        missing_deps.append("numpy")

    try:
        import cv2
    except ImportError:
        missing_deps.append("opencv-python")

    try:
        import matplotlib
    except ImportError:
        missing_deps.append("matplotlib")

    try:
        import yaml
    except ImportError:
        missing_deps.append("pyyaml")

    if missing_deps:
        print(f"Warning: Missing dependencies: {', '.join(missing_deps)}")
        print("Please install with: pip install " + " ".join(missing_deps))

    return len(missing_deps) == 0


# Run dependency check on import
check_dependencies()
