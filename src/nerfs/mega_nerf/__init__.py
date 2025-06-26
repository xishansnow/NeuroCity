"""
Mega-NeRF: Scalable Construction of Large-Scale NeRFs for Virtual Fly-Throughs

This module implements Mega-NeRF, a method for scalable construction of 
large-scale Neural Radiance Fields that enables virtual fly-throughs
of real-world environments.

Key Features:
- Spatial partitioning for large-scale scene decomposition
- Geometry-aware data partitioning
- Parallel training of submodules
- Temporal coherence for smooth rendering
- Efficient memory management for city-scale scenes

References:
- Mega-NeRF: Scalable Construction of Large-Scale NeRFs for Virtual Fly-Throughs (CVPR 2022)
- https://meganerf.cmusatyalab.org/
"""

from .mega_nerf_model import MegaNeRF, MegaNeRFSubmodule, MegaNeRFConfig
from .spatial_partitioner import SpatialPartitioner, GeometryAwarePartitioner
from .mega_nerf_trainer import MegaNeRFTrainer, ParallelTrainer
from .mega_nerf_dataset import MegaNeRFDataset, CameraDataset
from .volumetric_renderer import VolumetricRenderer, BatchRenderer
from .utils import *

__version__ = "1.0.0"

__all__ = [
    # Core models
    'MegaNeRF', 'MegaNeRFSubmodule', 'MegaNeRFConfig', # Spatial partitioning
    'SpatialPartitioner', 'GeometryAwarePartitioner', # Training
    'MegaNeRFTrainer', 'ParallelTrainer', # Data handling
    'MegaNeRFDataset', 'CameraDataset', # Rendering
    'VolumetricRenderer', 'BatchRenderer', # Utilities
    'PositionalEncoding', 'create_camera_path', 'load_colmap_data', 'save_checkpoint', 'load_checkpoint'
] 