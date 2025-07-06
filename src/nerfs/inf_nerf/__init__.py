"""
InfNeRF: Towards Infinite Scale NeRF Rendering with O(log n) Space Complexity

This module implements InfNeRF as described in:
"InfNeRF: Towards Infinite Scale NeRF Rendering with O(log n) Space Complexity"
by Jiabin Liang et al. (SIGGRAPH Asia 2024)

Key Features:
- Octree-based Level of Detail (LoD) structure
- O(log n) space complexity for rendering
- Anti-aliasing rendering through hierarchical structure
- Scalable large-scale scene representation
- Distributed training support
"""

from .core import InfNeRF, InfNeRFConfig, OctreeNode, InfNeRFRenderer, LoDAwareNeRF

from .dataset import InfNeRFDataset, InfNeRFDatasetConfig

from .trainer import InfNeRFTrainer, InfNeRFTrainerConfig, create_inf_nerf_trainer

# Utils
from .utils.octree_utils import OctreeBuilder, OctreePruner, calculate_gsd
from .utils.lod_utils import (
    LoDManager,
    anti_aliasing_sampling,
    determine_lod_level,
    pyramid_supervision,
)
from .utils.rendering_utils import distributed_rendering, memory_efficient_rendering

__version__ = "1.0.0"

__all__ = [
    # Core components
    "InfNeRF",
    "InfNeRFConfig",
    "OctreeNode",
    "InfNeRFRenderer",
    "LoDAwareNeRF",
    # Dataset and training
    "InfNeRFDataset",
    "InfNeRFDatasetConfig",
    "InfNeRFTrainer",
    "InfNeRFTrainerConfig",
    "create_inf_nerf_trainer",
    # Utils
    "OctreeBuilder",
    "OctreePruner",
    "calculate_gsd",
    "LoDManager",
    "anti_aliasing_sampling",
    "determine_lod_level",
    "pyramid_supervision",
    "distributed_rendering",
    "memory_efficient_rendering",
]
