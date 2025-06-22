"""
Block-NeRF: Scalable Large Scene Neural View Synthesis

This module implements Block-NeRF, a variant of Neural Radiance Fields
that can represent large-scale environments by decomposing scenes into
individually trained NeRFs.

Key Features:
- Block decomposition for scalable city-scale reconstruction
- Appearance embeddings for handling environmental variations
- Learned pose refinement for improved alignment
- Exposure conditioning for controllable rendering
- Visibility prediction for efficient block selection
- Seamless block compositing for smooth transitions

References:
- Block-NeRF: Scalable Large Scene Neural View Synthesis (CVPR 2022)
- https://waymo.com/research/block-nerf/
"""

from .block_nerf_model import BlockNeRF, BlockNeRFNetwork
from .block_manager import BlockManager
from .visibility_network import VisibilityNetwork
from .appearance_embedding import AppearanceEmbedding
from .pose_refinement import PoseRefinement
from .block_compositor import BlockCompositor
from .trainer import BlockNeRFTrainer
from .renderer import BlockNeRFRenderer
from .dataset import BlockNeRFDataset
from .utils import *

__version__ = "1.0.0"
__author__ = "NeuroCity Team"

__all__ = [
    "BlockNeRF",
    "BlockNeRFNetwork", 
    "BlockManager",
    "VisibilityNetwork",
    "AppearanceEmbedding",
    "PoseRefinement",
    "BlockCompositor",
    "BlockNeRFTrainer",
    "BlockNeRFRenderer",
    "BlockNeRFDataset"
] 