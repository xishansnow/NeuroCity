# PyNeRF: Pyramidal Neural Radiance Fields
# Based on "PyNeRF: Pyramidal Neural Radiance Fields" by Turki et al.

from .core import PyNeRF, PyNeRFConfig
from .pyramid_encoder import PyramidEncoder, HashEncoder
from .pyramid_renderer import PyramidRenderer, VolumetricRenderer
from .dataset import PyNeRFDataset, MultiScaleDataset
from .trainer import PyNeRFTrainer
from .utils import (
    compute_sample_area,
    get_pyramid_level,
    interpolate_pyramid_outputs,
    create_pyramid_hierarchy,
    save_pyramid_model,
    load_pyramid_model,
    compute_psnr,
    compute_ssim,
    log_pyramid_stats,
    create_training_schedule,
    apply_learning_rate_schedule
)

__version__ = "1.0.0"
__author__ = "NeuroCity Team"
__description__ = "PyNeRF: Pyramidal Neural Radiance Fields implementation"

__all__ = [
    # Core components
    "PyNeRF",
    "PyNeRFConfig",
    
    # Pyramid encoding
    "PyramidEncoder", 
    "HashEncoder",
    
    # Rendering
    "PyramidRenderer",
    "VolumetricRenderer",
    
    # Data handling
    "PyNeRFDataset",
    "MultiScaleDataset",
    
    # Training
    "PyNeRFTrainer",
    
    # Utilities
    "compute_sample_area",
    "get_pyramid_level", 
    "interpolate_pyramid_outputs",
    "create_pyramid_hierarchy",
    "save_pyramid_model",
    "load_pyramid_model",
    "compute_psnr",
    "compute_ssim",
    "log_pyramid_stats",
    "create_training_schedule",
    "apply_learning_rate_schedule"
] 