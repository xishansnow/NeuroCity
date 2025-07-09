"""
Block-NeRF: Scalable Large Scene Neural View Synthesis

This module implements Block-NeRF, a variant of Neural Radiance Fields
that can represent large-scale environments by decomposing scenes into
individually trained NeRFs.

This package has been refactored following the SVRaster pattern with:
- Dual rendering architecture (volume rendering for training, rasterization for inference)
- Tightly coupled components for optimal performance
- Clear separation of training and inference pipelines
- Modular configuration system

Key Features:
- Block decomposition for scalable city-scale reconstruction
- Appearance embeddings for handling environmental variations
- Learned pose refinement for improved alignment
- Exposure conditioning for controllable rendering
- Visibility prediction for efficient block selection
- Seamless block compositing for smooth transitions

Architecture:
- Training Phase: BlockNeRFTrainer ↔ VolumeRenderer (stable volume rendering)
- Inference Phase: BlockNeRFRenderer ↔ BlockRasterizer (efficient rasterization)

References:
- Block-NeRF: Scalable Large Scene Neural View Synthesis (CVPR 2022)
- https://waymo.com/research/block-nerf/
"""

# Version information
from ._version import __version__, __author__, __email__, __description__, __url__

__all__ = [
    # Version info
    "__version__",
    "__author__", 
    "__email__",
    "__description__",
    "__url__",
    
    # Core components
    "BlockNeRFConfig",
    "BlockNeRF",
    "BlockNeRFNetwork",
    "BlockNeRFLoss",
    "check_compatibility",
    "get_device_info",
    
    # Training components
    "BlockNeRFTrainer",
    "BlockNeRFTrainerConfig", 
    "create_block_nerf_trainer",
    
    # Rendering components
    "VolumeRenderer",
    "VolumeRendererConfig",
    "BlockRasterizer",
    "BlockRasterizerConfig",
    "BlockNeRFRenderer",
    "BlockNeRFRendererConfig",
    
    # Utilities
    "BlockManager",
    "BlockManagerConfig",
    "BlockNeRFDataset",
    "AppearanceEmbedding",
    "VisibilityNetwork",
    "PoseRefinement",
]

# Core components
from .core import (
    BlockNeRFConfig,
    BlockNeRFLoss,
    check_compatibility,
    get_device_info,
)

# Model components
from .block_nerf_model import (
    BlockNeRF,
    BlockNeRFNetwork,
)

# Training components (tightly coupled with volume rendering)
from .trainer import (
    BlockNeRFTrainer,
    BlockNeRFTrainerConfig,
    create_block_nerf_trainer,
)

# Volume rendering (for training)
from .volume_renderer import (
    VolumeRenderer,
    VolumeRendererConfig,
)

# Block rasterization (for inference)
from .block_rasterizer import (
    BlockRasterizer,
    BlockRasterizerConfig,
)

# Inference renderer (couples rasterizer with model)
from .renderer import (
    BlockNeRFRenderer,
    BlockNeRFRendererConfig,
)

# Utility components
from .block_manager import (
    BlockManager,
    BlockManagerConfig,
)

from .dataset import (
    BlockNeRFDataset,
)

from .appearance_embedding import (
    AppearanceEmbedding,
)

from .visibility_network import (
    VisibilityNetwork,
)

from .pose_refinement import (
    PoseRefinement,
)
    create_volume_renderer,
)

# Inference components (tightly coupled with rasterization)
from .renderer import (
    BlockNeRFRenderer,
    BlockNeRFRendererConfig,
    create_block_nerf_renderer,
)

# Block rasterization (for inference)
from .block_rasterizer import (
    BlockRasterizer,
    BlockRasterizerConfig,
    create_block_rasterizer,
)

# Dataset utilities
from .dataset import (
    BlockNeRFDataset,
    BlockNeRFDatasetConfig,
    create_block_nerf_dataloader,
    create_block_nerf_dataset,
)

# Block management
from .block_manager import (
    BlockManager,
    create_block_manager,
)

# Legacy components (for compatibility)
from .visibility_network import VisibilityNetwork
from .appearance_embedding import AppearanceEmbedding
from .pose_refinement import PoseRefinement

# Package metadata
__version__ = "1.0.0"
__author__ = "NeuroCity Team"

# All public API exports
__all__ = [
    # Core
    "BlockNeRFConfig",
    "BlockNeRFModel", 
    "BlockNeRFLoss",
    
    # Training (with volume rendering)
    "BlockNeRFTrainer",
    "BlockNeRFTrainerConfig",
    "VolumeRenderer",
    "VolumeRendererConfig",
    "create_block_nerf_trainer",
    "create_volume_renderer",
    
    # Inference (with rasterization)
    "BlockNeRFRenderer",
    "BlockNeRFRendererConfig", 
    "BlockRasterizer",
    "BlockRasterizerConfig",
    "create_block_nerf_renderer",
    "create_block_rasterizer",
    
    # Dataset
    "BlockNeRFDataset",
    "BlockNeRFDatasetConfig",
    "create_block_nerf_dataloader",
    "create_block_nerf_dataset",
    
    # Block management
    "BlockManager",
    "create_block_manager",
    
    # Legacy components
    "VisibilityNetwork",
    "AppearanceEmbedding", 
    "PoseRefinement",
    
    # Utilities
    "check_compatibility",
    "get_device_info",
]

# Quick start guide
def quick_start_guide():
    """Display quick start guide for Block-NeRF."""
    print("""
🚀 Block-NeRF Quick Start Guide
================================

📦 Basic Usage:
```python
import nerfs.block_nerf as block_nerf

# 1. Create configuration
config = block_nerf.BlockNeRFConfig(
    scene_bounds=(-100, -100, -10, 100, 100, 10),
    block_size=75.0,
    max_blocks=64
)

# 2. Training setup
trainer_config = block_nerf.BlockNeRFTrainerConfig(num_epochs=100)
volume_renderer = block_nerf.create_volume_renderer()
trainer = block_nerf.create_block_nerf_trainer(config, trainer_config)

# 3. Dataset
dataset_config = block_nerf.BlockNeRFDatasetConfig(data_dir="./data")
train_loader = block_nerf.create_block_nerf_dataloader(dataset_config, "train")

# 4. Training
trainer.train(train_loader)

# 5. Inference
renderer_config = block_nerf.BlockNeRFRendererConfig()
rasterizer = block_nerf.create_block_rasterizer() 
renderer = block_nerf.create_block_nerf_renderer(config, renderer_config)
rendered_image = renderer.render_image(camera_pose, intrinsics)
```

🏗️ Architecture:
- Training: Trainer ↔ VolumeRenderer (stable training)
- Inference: Renderer ↔ BlockRasterizer (fast inference)

📖 More info: Check the documentation and examples
""")

# System compatibility check
try:
    import torch
    CUDA_AVAILABLE = torch.cuda.is_available()
except ImportError:
    CUDA_AVAILABLE = False 