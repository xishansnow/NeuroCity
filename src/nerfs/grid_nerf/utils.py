"""
Grid-NeRF Utilities Module

This module provides utility functions for Grid-NeRF including:
- Image metrics (PSNR, SSIM, LPIPS)
- Visualization and rendering utilities  
- I/O operations
- Logging setup
- Learning rate scheduling
- Mathematical utilities
"""

import os
import cv2
import torch
import torch.nn.functional as F
import numpy as np
import logging
from pathlib import Path
import matplotlib.pyplot as plt
from PIL import Image
import json
import yaml
from torchvision.utils import save_image as torch_save_image
from torch.optim.lr_scheduler import _LRScheduler

# Import from submodules
from .utils.metrics_utils import compute_psnr, compute_ssim, compute_lpips
from .utils.io_utils import save_image, load_image, create_video_from_images, create_comparison_grid
from .utils.training_utils import setup_logging, load_config, save_config, get_learning_rate_scheduler
from .utils.math_utils import positional_encoding, safe_normalize
from .utils.ray_utils import get_ray_directions, sample_along_rays, volume_rendering

__all__ = [
    # Metrics
    'compute_psnr', 'compute_ssim', 'compute_lpips',
    # I/O
    'save_image', 'load_image', 'create_video_from_images', 'create_comparison_grid',
    # Training
    'setup_logging', 'load_config', 'save_config', 'get_learning_rate_scheduler',
    # Math
    'positional_encoding', 'safe_normalize',
    # Ray operations
    'get_ray_directions', 'sample_along_rays', 'volume_rendering'
]