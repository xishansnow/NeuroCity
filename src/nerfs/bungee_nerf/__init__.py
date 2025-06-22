# BungeeNeRF: Progressive Neural Radiance Field for Extreme Multi-scale Scene Rendering
# Based on "BungeeNeRF: Progressive Neural Radiance Field for Extreme Multi-scale Scene Rendering" by Xiangli et al.

from .core import BungeeNeRF, BungeeNeRFConfig, ProgressiveBlock
from .progressive_encoder import ProgressivePositionalEncoder, MultiScaleEncoder
from .multiscale_renderer import MultiScaleRenderer, LevelOfDetailRenderer
from .dataset import BungeeNeRFDataset, MultiScaleDataset, GoogleEarthDataset
from .trainer import BungeeNeRFTrainer, ProgressiveTrainer
from .utils import (
    compute_scale_factor,
    get_level_of_detail,
    progressive_positional_encoding,
    multiscale_sampling,
    compute_multiscale_loss,
    save_bungee_model,
    load_bungee_model,
    compute_psnr,
    compute_ssim,
    create_progressive_schedule
)

__version__ = "1.0.0"
__author__ = "NeuroCity Team"
__description__ = "BungeeNeRF: Progressive Neural Radiance Field for Extreme Multi-scale Scene Rendering"

__all__ = [
    # Core components
    "BungeeNeRF",
    "BungeeNeRFConfig", 
    "ProgressiveBlock",
    
    # Progressive encoding
    "ProgressivePositionalEncoder",
    "MultiScaleEncoder",
    
    # Rendering
    "MultiScaleRenderer",
    "LevelOfDetailRenderer",
    
    # Data handling
    "BungeeNeRFDataset",
    "MultiScaleDataset",
    "GoogleEarthDataset",
    
    # Training
    "BungeeNeRFTrainer",
    "ProgressiveTrainer",
    
    # Utilities
    "compute_scale_factor",
    "get_level_of_detail",
    "progressive_positional_encoding",
    "multiscale_sampling",
    "compute_multiscale_loss",
    "save_bungee_model",
    "load_bungee_model",
    "compute_psnr",
    "compute_ssim",
    "create_progressive_schedule"
]
