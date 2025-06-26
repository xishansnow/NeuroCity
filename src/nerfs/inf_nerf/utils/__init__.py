"""
Utility modules for InfNeRF.

This package contains utility functions and classes for:
- Octree construction and pruning
- Level of Detail management
- Rendering utilities
- Distributed training support
"""

from .octree_utils import (
    OctreeBuilder, OctreePruner, calculate_gsd, visualize_octree
)

from .lod_utils import (
    LoDManager, anti_aliasing_sampling, determine_lod_level, pyramid_supervision
)

from .rendering_utils import (
    distributed_rendering, memory_efficient_rendering, batch_ray_sampling
)

__all__ = [
    # Octree utilities
    "OctreeBuilder", "OctreePruner", "calculate_gsd", "visualize_octree", # LoD utilities
    "LoDManager", "anti_aliasing_sampling", "determine_lod_level", "pyramid_supervision", # Rendering utilities
    "distributed_rendering", "memory_efficient_rendering", "batch_ray_sampling", ] 