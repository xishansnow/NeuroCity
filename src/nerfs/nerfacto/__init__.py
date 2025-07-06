"""
Nerfacto - Advanced NeRF Implementation
=====================================

A state-of-the-art NeRF implementation based on nerfstudio's nerfacto model, combining the best practices and optimizations from various NeRF variants.

Key Features:
- Hash encoding for efficient scene representation
- Proposal networks for better sampling
- Appearance embeddings for varying lighting conditions
- Distortion loss for improved geometry
- Orientation loss for better surface normals
- Background modeling with learnable backgrounds
"""

from .core import (
    NeRFactoConfig,
    NerfactoModel,
    NerfactoFieldConfig,
    NerfactoField,
    HashEncoding,
    ProposalNetwork,
    AppearanceEmbedding,
    NerfactoRenderer,
    NerfactoLoss,
)

from .dataset import NerfactoDataset, create_nerfacto_dataloader, create_nerfacto_dataset

from .trainer import NerfactoTrainer

from .utils import camera_utils

__version__ = "1.0.0"
__author__ = "NeuroCity Development Team"

__all__ = [
    # Core components
    "NeRFactoConfig",
    "NerfactoModel",
    "NerfactoFieldConfig",
    "NerfactoField",
    "HashEncoding",
    "ProposalNetwork",
    "AppearanceEmbedding",
    "NerfactoRenderer",
    "NerfactoLoss",  # Dataset components
    "NerfactoDataset",
    "create_nerfacto_dataloader",
    "create_nerfacto_dataset",  # Training components
    "NerfactoTrainer",  # Utilities
    "camera_utils",
]
