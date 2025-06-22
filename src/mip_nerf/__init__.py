"""
Mip-NeRF: A Multiscale Representation for Anti-Aliasing Neural Radiance Fields

This package implements Mip-NeRF, which addresses aliasing artifacts in Neural Radiance Fields
by representing each pixel as a cone rather than a ray, and using integrated positional encoding.

Key Features:
- Integrated Positional Encoding (IPE)
- Conical frustum representation
- Multi-scale anti-aliasing
- Improved rendering quality for varying viewing distances
"""

from .core import (
    MipNeRFConfig,
    IntegratedPositionalEncoder,
    ConicalFrustum,
    MipNeRFMLP,
    MipNeRFRenderer,
    MipNeRF
)

from .dataset import (
    MipNeRFDataset,
    BlenderMipNeRFDataset,
    create_mip_nerf_dataset,
    create_mip_nerf_dataloader
)

from .trainer import MipNeRFTrainer

__version__ = "1.0.0"
__author__ = "NeuroCity Team"
__all__ = [
    "MipNeRFConfig",
    "IntegratedPositionalEncoder", 
    "ConicalFrustum",
    "MipNeRFMLP",
    "MipNeRFRenderer",
    "MipNeRF",
    "MipNeRFDataset",
    "BlenderMipNeRFDataset",
    "create_mip_nerf_dataset",
    "create_mip_nerf_dataloader",
    "MipNeRFTrainer"
] 