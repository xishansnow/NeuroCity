"""
InfNeRF: Infinite Scale NeRF with O(log n) Space Complexity

This package implements InfNeRF with decoupled trainer and renderer architecture.
"""

from .core import (
    InfNeRF,
    InfNeRFConfig,
    OctreeNode,
    LoDAwareNeRF,
    HashEncoder,
    SphericalHarmonicsEncoder,
)

from .trainer import InfNeRFTrainer, InfNeRFTrainerConfig, create_inf_nerf_trainer

from .renderer import (
    InfNeRFRenderer,
    InfNeRFRendererConfig,
    create_inf_nerf_renderer,
    render_demo_images,
)

from .utils.volume_renderer import VolumeRenderer, VolumeRendererConfig, create_volume_renderer

__version__ = "1.0.0"

__all__ = [
    # Core models
    "InfNeRF",
    "InfNeRFConfig",
    "OctreeNode",
    "LoDAwareNeRF",
    "HashEncoder",
    "SphericalHarmonicsEncoder",
    # Trainer
    "InfNeRFTrainer",
    "InfNeRFTrainerConfig",
    "create_inf_nerf_trainer",
    # Renderer
    "InfNeRFRenderer",
    "InfNeRFRendererConfig",
    "create_inf_nerf_renderer",
    "render_demo_images",
    # Volume renderer
    "VolumeRenderer",
    "VolumeRendererConfig",
    "create_volume_renderer",
]
