"""
Nerfacto - Advanced NeRF Implementation
=====================================

A state-of-the-art NeRF implementation based on nerfstudio's nerfacto model,
combining the best practices and optimizations from various NeRF variants.

Key Features:
- Hash encoding for efficient scene representation
- Proposal networks for better sampling
- Appearance embeddings for varying lighting conditions
- Distortion loss for improved geometry
- Orientation loss for better surface normals
- Background modeling with learnable backgrounds
"""

from .core import (
    NerfactoConfig,
    NerfactoModel,
    NerfactoFieldConfig,
    NerfactoField,
    HashEncoding,
    ProposalNetwork,
    AppearanceEmbedding,
    NerfactoRenderer,
    NerfactoLoss
)

from .dataset import (
    NerfactoDataset,
    NerfactoDataManager,
    create_nerfacto_dataloader,
    create_nerfacto_dataset,
    parse_transforms_json,
    NerfactoRayBundle
)

from .trainer import (
    NerfactoTrainer,
    NerfactoConfig as TrainerConfig,
    NerfactoOptimizer,
    NerfactoScheduler
)

from .utils import (
    NerfactoUtils,
    camera_utils,
    geometry_utils,
    loss_utils,
    rendering_utils,
    visualization_utils
)

__version__ = "1.0.0"
__author__ = "NeuroCity Development Team"

__all__ = [
    # Core components
    "NerfactoConfig",
    "NerfactoModel", 
    "NerfactoFieldConfig",
    "NerfactoField",
    "HashEncoding",
    "ProposalNetwork",
    "AppearanceEmbedding",
    "NerfactoRenderer",
    "NerfactoLoss",
    
    # Dataset components
    "NerfactoDataset",
    "NerfactoDataManager",
    "create_nerfacto_dataloader", 
    "create_nerfacto_dataset",
    "parse_transforms_json",
    "NerfactoRayBundle",
    
    # Training components
    "NerfactoTrainer",
    "TrainerConfig",
    "NerfactoOptimizer",
    "NerfactoScheduler",
    
    # Utilities
    "NerfactoUtils",
    "camera_utils",
    "geometry_utils", 
    "loss_utils",
    "rendering_utils",
    "visualization_utils"
] 