"""
Mega-NeRF++: An Improved Scalable NeRFs for High-resolution Photogrammetric Images

This package implements Mega-NeRF++, an advanced neural radiance field approach 
specifically designed for large-scale scenes with high-resolution photogrammetric images.

Key Features:
- Scalable architecture for large scenes
- High-resolution photogrammetric image support
- Improved spatial partitioning strategies
- Enhanced memory management
- Multi-resolution training and rendering
- Photogrammetric data optimization
- Efficient hierarchical representation
"""

from .core import (
    MegaNeRFPlusConfig,
    ScalableNeRFModel,
    HierarchicalSpatialEncoder,
    MultiResolutionMLP,
    PhotogrammetricRenderer,
    MegaNeRFPlus
)

from .spatial_partitioner import (
    SpatialPartitioner,
    AdaptiveOctree,
    HierarchicalGridPartitioner,
    PhotogrammetricPartitioner
)

from .multires_renderer import (
    MultiResolutionRenderer,
    AdaptiveLODRenderer,
    PhotogrammetricVolumetricRenderer
)

from .dataset import (
    MegaNeRFPlusDataset,
    PhotogrammetricDataset,
    LargeSceneDataset,
    create_meganerf_plus_dataset,
    create_photogrammetric_dataloader
)

from .trainer import (
    MegaNeRFPlusTrainer,
    MultiScaleTrainer,
    DistributedTrainer
)

from .memory_manager import (
    MemoryManager,
    CacheManager,
    StreamingDataLoader
)

__version__ = "1.0.0"
__author__ = "NeuroCity Team"
__description__ = "Mega-NeRF++: Scalable NeRFs for High-resolution Photogrammetric Images"

__all__ = [
    # Core components
    "MegaNeRFPlusConfig",
    "ScalableNeRFModel", 
    "HierarchicalSpatialEncoder",
    "MultiResolutionMLP",
    "PhotogrammetricRenderer",
    "MegaNeRFPlus",
    
    # Spatial partitioning
    "SpatialPartitioner",
    "AdaptiveOctree",
    "HierarchicalGridPartitioner", 
    "PhotogrammetricPartitioner",
    
    # Multi-resolution rendering
    "MultiResolutionRenderer",
    "AdaptiveLODRenderer",
    "PhotogrammetricVolumetricRenderer",
    
    # Dataset and data loading
    "MegaNeRFPlusDataset",
    "PhotogrammetricDataset",
    "LargeSceneDataset",
    "create_meganerf_plus_dataset",
    "create_photogrammetric_dataloader",
    
    # Training
    "MegaNeRFPlusTrainer",
    "MultiScaleTrainer",
    "DistributedTrainer",
    
    # Memory management
    "MemoryManager",
    "CacheManager", 
    "StreamingDataLoader"
] 