"""
NeuroCity NeRF Models Package

This package contains various NeRF model implementations:
- Classic NeRF
- Instant NGP
- Plenoxels
- Block NeRF
- Mega NeRF
- Grid NeRF
- SVRaster
- CNC NeRF
- Inf NeRF
- Bungee NeRF
- Pyramid NeRF
- MIP NeRF
"""

import logging
import sys
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# Add the parent directory to allow importing from utils
_current_dir = Path(__file__).parent
_parent_dir = _current_dir.parent
if str(_parent_dir) not in sys.path:
    sys.path.insert(0, str(_parent_dir))

__version__ = "1.0.0"

__all__ = [
    'classic_nerf',
    'instant_ngp',
    'plenoxels',
    'block_nerf',
    'mega_nerf',
    'grid_nerf',
    'svraster',
    'cnc_nerf',
    'inf_nerf',
    'bungee_nerf',
    'pyramid_nerf',
    'mip_nerf'
] 