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

# Core model components - these can be imported immediately
from .block_nerf_model import BlockNeRF, BlockNeRFNetwork
from .block_manager import BlockManager
from .visibility_network import VisibilityNetwork
from .block_compositor import BlockCompositor
from .appearance_embedding import AppearanceEmbedding
from .pose_refinement import PoseRefinement

# Import renderer
from .renderer import BlockNeRFRenderer

__version__ = "1.0.0"
__author__ = "NeuroCity Team"

# Define what gets imported with "from block_nerf import *"
__all__ = [
    "BlockNeRF", "BlockNeRFNetwork", "BlockManager", "VisibilityNetwork", "AppearanceEmbedding", "PoseRefinement", "BlockCompositor", "BlockNeRFRenderer"
]

# Lazy imports to avoid circular dependencies
def get_trainer():
    """Get Block-NeRF trainer."""
    from .trainer import BlockNeRFTrainer
    return BlockNeRFTrainer

def get_dataset():
    """Get Block-NeRF dataset."""
    from .dataset import BlockNeRFDataset
    return BlockNeRFDataset

# Add lazy-loaded components to __all__
BlockNeRFTrainer = property(lambda self: get_trainer())
BlockNeRFDataset = property(lambda self: get_dataset()) 