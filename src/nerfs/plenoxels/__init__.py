"""
Plenoxels: Radiance Fields without Neural Networks

This package implements Plenoxels, a method for neural radiance fields that replaces
neural networks with sparse voxel grids and spherical harmonics.

Key features:
- Sparse voxel grid representation
- Spherical harmonics for view-dependent colors
- Fast training without neural networks
- Trilinear interpolation for smooth rendering
- Coarse-to-fine optimization
"""

from .core import (
    PlenoxelConfig, VoxelGrid, SphericalHarmonics, PlenoxelModel, PlenoxelLoss
)

from .dataset import (
    PlenoxelDataset, PlenoxelDatasetConfig, create_plenoxel_dataloader, create_plenoxel_dataset
)

from .trainer import (
    PlenoxelTrainer, PlenoxelTrainerConfig, create_plenoxel_trainer
)

__version__ = "1.0.0"
__author__ = "NeuroCity Team"

__all__ = [
    # Core components
    'PlenoxelConfig', 'VoxelGrid', 'SphericalHarmonics', 'PlenoxelModel', 'PlenoxelLoss',
    # Dataset components
    'PlenoxelDataset', 'PlenoxelDatasetConfig', 'create_plenoxel_dataloader', 'create_plenoxel_dataset',
    # Training components
    'PlenoxelTrainer', 'PlenoxelTrainerConfig', 'create_plenoxel_trainer'
] 