"""
NeRFs Package - Neural Radiance Fields Collection

This package contains various Neural Radiance Fields (NeRF) implementations
and related methods for neural scene representation and rendering.

Available modules:
- block_nerf: Block-NeRF for large-scale scene representation
- classic_nerf: Original NeRF implementation
- dnmp_nerf: Differentiable Neural Mesh Primitives
- grid_nerf: Grid-based NeRF variants
- mega_nerf: Mega-NeRF for large-scale scenes
- mip_nerf: Mip-NeRF with anti-aliasing
- nerfacto: Nerfacto implementation
- plenoxels: Plenoxels sparse voxel representation
- svraster: Sparse Voxel Rasterization
- bungee_nerf: Bungee-NeRF progressive training
- instant_ngp: Instant Neural Graphics Primitives
- mega_nerf_plus: Enhanced Mega-NeRF
- pyramid_nerf: Pyramid NeRF multi-scale representation
- occupancy_net: Occupancy Networks for 3D reconstruction
- sdf_net: Signed Distance Function networks for shape representation

Author: NeuroCity Team
Version: 1.0.0
"""

import warnings
import sys
from typing import Any

# Suppress warnings for optional dependencies
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    
    # Try importing optional modules
    try:
        import openvdb
        OPENVDB_AVAILABLE = True
    except ImportError:
        OPENVDB_AVAILABLE = False
    
    try:
        import trimesh
        TRIMESH_AVAILABLE = True
    except ImportError:
        TRIMESH_AVAILABLE = False

# Only show warnings once if running as main script
if __name__ != "__main__":
    if not OPENVDB_AVAILABLE:
        print("Note: OpenVDB not available. NeuralVDB functionality will be limited.")
    if not TRIMESH_AVAILABLE:
        print("Note: trimesh not available. Some mesh processing features disabled.")

__version__ = "1.0.0"
__author__ = "NeuroCity Team"

# Import all NeRF modules
try:
    from . import block_nerf
    from . import classic_nerf
    from . import dnmp_nerf
    from . import grid_nerf
    from . import mega_nerf
    from . import mip_nerf
    from . import nerfacto
    from . import plenoxels
    from . import svraster
    from . import bungee_nerf
    from . import instant_ngp
    from . import mega_nerf_plus
    from . import pyramid_nerf
    from . import occupancy_net
    from . import sdf_net
except ImportError as e:
    print(f"Warning: Some NeRF modules could not be imported: {e}")

# Available NeRF implementations
AVAILABLE_NERFS = [
    'block_nerf', 'classic_nerf', 'dnmp_nerf', 'grid_nerf', 'mega_nerf', 'mip_nerf', 'nerfacto', 'plenoxels', 'svraster', 'bungee_nerf', 'instant_ngp', 'mega_nerf_plus', 'pyramid_nerf', 'occupancy_net', 'sdf_net'
]

def list_available_nerfs() -> list[str]:
    """List all available NeRF implementations."""
    return AVAILABLE_NERFS

def get_nerf_info() -> dict[str, str]:
    """Get information about all NeRF implementations."""
    info = {
        'block_nerf': 'Block-NeRF for large-scale scene representation with spatial partitioning', 'classic_nerf': 'Original Neural Radiance Fields implementation', 'dnmp_nerf': 'Differentiable Neural Mesh Primitives with mesh-based representation', 'grid_nerf': 'Grid-based NeRF variants for efficient rendering', 'mega_nerf': 'Mega-NeRF for large-scale outdoor scene reconstruction', 'mip_nerf': 'Mip-NeRF with anti-aliasing and multiscale representation', 'nerfacto': 'Nerfacto - practical NeRF implementation', 'plenoxels': 'Plenoxels sparse voxel representation without neural networks', 'svraster': 'Sparse Voxel Rasterization for efficient rendering', 'bungee_nerf': 'Bungee-NeRF with progressive training strategy', 'instant_ngp': 'Instant Neural Graphics Primitives with hash encoding', 'mega_nerf_plus': 'Enhanced Mega-NeRF with improved memory management', 'pyramid_nerf': 'Pyramid NeRF with multi-scale hierarchical representation', 'occupancy_net': 'Occupancy Networks for 3D reconstruction', 'sdf_net': 'Signed Distance Function networks for shape representation'
    }
    return info

def get_nerf_module(name: str) -> Any:
    """Get a specific NeRF module by name."""
    if name not in AVAILABLE_NERFS:
        raise ValueError(f"NeRF '{name}' not available. Available: {AVAILABLE_NERFS}")
    
    # Import the requested module
    if name == 'block_nerf':
        from . import block_nerf
        return block_nerf
    elif name == 'classic_nerf':
        from . import classic_nerf
        return classic_nerf
    elif name == 'dnmp_nerf':
        from . import dnmp_nerf
        return dnmp_nerf
    elif name == 'grid_nerf':
        from . import grid_nerf
        return grid_nerf
    elif name == 'mega_nerf':
        from . import mega_nerf
        return mega_nerf
    elif name == 'mip_nerf':
        from . import mip_nerf
        return mip_nerf
    elif name == 'nerfacto':
        from . import nerfacto
        return nerfacto
    elif name == 'plenoxels':
        from . import plenoxels
        return plenoxels
    elif name == 'svraster':
        from . import svraster
        return svraster
    elif name == 'bungee_nerf':
        from . import bungee_nerf
        return bungee_nerf
    elif name == 'instant_ngp':
        from . import instant_ngp
        return instant_ngp
    elif name == 'mega_nerf_plus':
        from . import mega_nerf_plus
        return mega_nerf_plus
    elif name == 'pyramid_nerf':
        from . import pyramid_nerf
        return pyramid_nerf
    elif name == 'occupancy_net':
        from . import occupancy_net
        return occupancy_net
    elif name == 'sdf_net':
        from . import sdf_net
        return sdf_net
    else:
        raise ValueError(f"Unknown NeRF module: {name}")

# Expose main functions
__all__ = [
    'AVAILABLE_NERFS', 'list_available_nerfs', 'get_nerf_info', 'get_nerf_module', # Module imports
    'block_nerf', 'classic_nerf', 'dnmp_nerf', 'grid_nerf', 'mega_nerf', 'mip_nerf', 'nerfacto', 'plenoxels', 'svraster', 'bungee_nerf', 'instant_ngp', 'mega_nerf_plus', 'pyramid_nerf', 'occupancy_net', 'sdf_net'
] 