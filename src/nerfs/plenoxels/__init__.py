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
- NeuralVDB integration for efficient storage
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

# NeuralVDB integration (optional)
try:
    from .neuralvdb_interface import (
        NeuralVDBManager, NeuralVDBConfig, save_plenoxel_as_neuralvdb, load_plenoxel_from_neuralvdb
    )
    from .trainer_neuralvdb import (
        NeuralVDBPlenoxelTrainer, NeuralVDBTrainerConfig, create_neuralvdb_trainer
    )
    NEURALVDB_AVAILABLE = True
except ImportError:
    NEURALVDB_AVAILABLE = False

__version__ = "1.0.0"
__author__ = "NeuroCity Team"

__all__ = [
    # Core components
    'PlenoxelConfig', 'VoxelGrid', 'SphericalHarmonics', 'PlenoxelModel', 'PlenoxelLoss', # Dataset components
    'PlenoxelDataset', 'PlenoxelDatasetConfig', 'create_plenoxel_dataloader', 'create_plenoxel_dataset', # Training components
    'PlenoxelTrainer', 'PlenoxelTrainerConfig', 'create_plenoxel_trainer'
]

# Add NeuralVDB components if available
if NEURALVDB_AVAILABLE:
    __all__.extend([
        # NeuralVDB components
        'NeuralVDBManager', 'NeuralVDBConfig', 'save_plenoxel_as_neuralvdb', 'load_plenoxel_from_neuralvdb', 'NeuralVDBPlenoxelTrainer', 'NeuralVDBTrainerConfig', 'create_neuralvdb_trainer'
    ]) 