"""
Grid-NeRF Utilities Package

This package provides utility modules for Grid-NeRF including:
- Image metrics (PSNR, SSIM, LPIPS)
- Visualization and rendering utilities
- I/O operations
- Logging setup
- Learning rate scheduling
- Mathematical utilities
- Ray operations
"""

from .metrics_utils import compute_psnr, compute_ssim, compute_lpips
from .io_utils import save_image, load_image, create_video_from_images, create_comparison_grid
from .training_utils import setup_logging, load_config, save_config, get_learning_rate_scheduler
from .math_utils import positional_encoding, safe_normalize
from .ray_utils import (
    get_ray_directions,
    sample_points_along_rays as sample_along_rays,
    compute_ray_weights as volume_rendering,
)

__all__ = [
    # Metrics
    "compute_psnr",
    "compute_ssim",
    "compute_lpips",
    # I/O
    "save_image",
    "load_image",
    "create_video_from_images",
    "create_comparison_grid",
    # Training
    "setup_logging",
    "load_config",
    "save_config",
    "get_learning_rate_scheduler",
    # Math
    "positional_encoding",
    "safe_normalize",
    # Ray operations
    "get_ray_directions",
    "sample_along_rays",
    "volume_rendering",
]
