"""
Plenoxels Refactored Configuration

This module provides configuration classes for both training and inference modes.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
import torch
from typing import Optional


@dataclass
class PlenoxelConfig:
    """Base configuration for Plenoxels model.

    This configuration is shared between training and inference modes,
    containing all the essential parameters for the voxel grid and rendering.
    """

    # Grid parameters
    grid_resolution: tuple[int, int, int] = (256, 256, 256)
    scene_bounds: torch.Tensor | None = None

    # Spherical harmonics parameters
    sh_degree: int = 2

    # Rendering parameters
    near_plane: float = 0.1
    far_plane: float = 10.0
    step_size: float = 0.01
    sigma_thresh: float = 1e-8
    stop_thresh: float = 1e-7
    num_samples: int = 64

    # Device parameters
    use_cuda: bool = True
    device: torch.device = field(
        default_factory=lambda: torch.device("cuda" if torch.cuda.is_available() else "cpu")
    )

    # Memory optimization
    max_batch_size: int = 4096
    chunk_size: int = 8192

    def __post_init__(self):
        """Initialize default scene bounds if not provided."""
        if self.scene_bounds is None:
            self.scene_bounds = torch.tensor([-1.0, -1.0, -1.0, 1.0, 1.0, 1.0], device=self.device)

        # Validate parameters
        assert len(self.grid_resolution) == 3, "Grid resolution must be a 3-tuple"
        assert self.sh_degree >= 0, "SH degree must be non-negative"
        assert self.near_plane > 0, "Near plane must be positive"
        assert self.far_plane > self.near_plane, "Far plane must be greater than near plane"


@dataclass
class PlenoxelTrainingConfig(PlenoxelConfig):
    """Configuration for Plenoxels training mode.

    Extends the base configuration with training-specific parameters.
    """

    # Training parameters
    num_epochs: int = 10000
    learning_rate: float = 0.1
    weight_decay: float = 0.0
    batch_size: int = 4096

    # Coarse-to-fine training
    use_coarse_to_fine: bool = True
    coarse_resolutions: list[tuple[int, int, int]] = field(
        default_factory=lambda: [(128, 128, 128), (256, 256, 256), (512, 512, 512)]
    )
    coarse_epochs: list[int] = field(default_factory=lambda: [2000, 5000, 10000])

    # Regularization
    tv_lambda: float = 1e-6  # Total variation
    l1_lambda: float = 1e-8  # L1 sparsity
    sparsity_threshold: float = 0.01

    # Pruning
    pruning_interval: int = 1000
    pruning_threshold: float = 0.01

    # Loss parameters
    rgb_loss_type: str = "mse"
    loss_reduction: str = "mean"  # Added missing parameter
    depth_loss_type: str = "l1"
    normal_loss_type: str = "l1"
    density_loss_type: str = "l1"
    depth_lambda: float = 1.0
    normal_lambda: float = 1.0

    # Cache parameters
    cache_size: int = 1000

    # Optimization
    use_adam: bool = False
    lr_decay_steps: list[int] | None = None
    lr_decay_gamma: float = 0.1

    # Monitoring and saving
    log_interval: int = 100
    eval_interval: int = 1000
    save_interval: int = 5000
    checkpoint_dir: str = "checkpoints"
    log_dir: str = "logs"

    # Experiment tracking
    experiment_name: str = "plenoxel_experiment"
    use_wandb: bool = False
    use_tensorboard: bool = True

    def __post_init__(self):
        super().__post_init__()
        # Validate training-specific parameters
        assert len(self.coarse_resolutions) == len(
            self.coarse_epochs
        ), "Mismatched coarse-to-fine parameters"


@dataclass
class PlenoxelInferenceConfig(PlenoxelConfig):
    """Configuration for Plenoxels inference mode.

    Extends the base configuration with inference-specific parameters.
    """

    # Rendering quality
    high_quality: bool = True
    adaptive_sampling: bool = True
    max_samples_per_ray: int = 128

    # Memory optimization for inference
    tile_size: int = 512  # For tiled rendering
    overlap: int = 64  # Overlap between tiles

    # Output options
    render_depth: bool = True
    render_normals: bool = False
    render_weights: bool = False

    # Performance optimization
    use_half_precision: bool = False
    optimize_for_inference: bool = True

    def __post_init__(self):
        super().__post_init__()
        # Optimize settings for inference
        if self.optimize_for_inference:
            self.chunk_size = min(self.chunk_size, 16384)  # Larger chunks for inference
            if self.high_quality:
                self.num_samples = max(self.num_samples, 64)


# Type aliases for convenience
TrainingConfig = PlenoxelTrainingConfig
InferenceConfig = PlenoxelInferenceConfig
