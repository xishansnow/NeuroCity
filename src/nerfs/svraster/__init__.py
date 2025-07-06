"""
SVRaster: Sparse Voxels Rasterization for Real-time High-fidelity Radiance Field Rendering

This package implements the SVRaster method for real-time neural radiance field rendering
using adaptive sparse voxels and efficient rasterization without neural networks.

Key features:
- Adaptive sparse voxel allocation with octree-based level-of-detail
- Ray direction-dependent Morton ordering for correct depth sorting
- Custom rasterizer for efficient sparse voxel rendering
- Real-time performance with high-fidelity quality
- Compatibility with grid-based 3D processing techniques
"""

from __future__ import annotations

from .core import (
    SVRasterConfig, AdaptiveSparseVoxels, SVRasterModel, SVRasterLoss
)

from .volume_renderer import (
    VolumeRenderer
)

from .true_rasterizer import (
    TrueVoxelRasterizer
)

from .dataset import (
    SVRasterDataset, SVRasterDatasetConfig, create_svraster_dataloader, create_svraster_dataset
)

from .trainer import (
    SVRasterTrainer, SVRasterTrainerConfig, create_svraster_trainer
)

from .renderer import (
    SVRasterRenderer, SVRasterRendererConfig, TrueVoxelRasterizerConfig, create_svraster_renderer
)

from .utils import (
    morton_encode_3d, morton_decode_3d, ray_direction_dependent_ordering, octree_subdivision, voxel_pruning, depth_peeling
)

# Try to import CUDA modules (optional)
try:
    from .cuda import SVRasterGPU, SVRasterGPUTrainer
    CUDA_AVAILABLE = True
except ImportError:
    CUDA_AVAILABLE = False
    SVRasterGPU = None
    SVRasterGPUTrainer = None

__version__ = "1.0.0"
__author__ = "NeuroCity Team"

__all__ = [
    # Core components
    'SVRasterConfig', 'AdaptiveSparseVoxels', 'VolumeRenderer', 'TrueVoxelRasterizer', 'SVRasterModel', 'SVRasterLoss', 
    # Dataset components
    'SVRasterDataset', 'SVRasterDatasetConfig', 'create_svraster_dataloader', 'create_svraster_dataset', 
    # Training components
    'SVRasterTrainer', 'SVRasterTrainerConfig', 'create_svraster_trainer', 
    # Rendering components
    'SVRasterRenderer', 'SVRasterRendererConfig', 'TrueVoxelRasterizerConfig', 'create_svraster_renderer',
    # Utility functions
    'morton_encode_3d', 'morton_decode_3d', 'ray_direction_dependent_ordering', 'octree_subdivision', 'voxel_pruning', 'depth_peeling',
    # CUDA support info
    'CUDA_AVAILABLE', 'SVRasterGPU', 'SVRasterGPUTrainer'
] 