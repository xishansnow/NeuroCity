from typing import Any, Optional
"""
Grid-NeRF Utils Package

This package contains utility modules for Grid-NeRF including:
- grid_utils: Grid operations and spatial computations
- ray_utils: Ray operations and sampling strategies
- metrics_utils: Evaluation metrics and visualization functions

Additional utilities like setup_logging are available directly from nerfs.grid_nerf.utils
"""

# Grid utilities
from .grid_utils import (
    compute_grid_bounds, world_to_grid_coords, grid_to_world_coords, trilinear_interpolation, create_voxel_grid, dilate_grid, prune_grid, compute_grid_occupancy, adaptive_grid_subdivision, compute_grid_gradient, smooth_grid, compute_grid_statistics
)

# Ray utilities
from .ray_utils import (
    generate_rays_from_camera, sample_points_along_rays, hierarchical_sampling, ray_aabb_intersection, compute_ray_grid_intersections, voxel_traversal_3d, importance_sampling_from_grid, compute_ray_weights, sample_stratified_rays
)

# Metrics utilities  
from .metrics_utils import (
    compute_psnr, compute_ssim, compute_lpips, compute_depth_metrics, compute_novel_view_metrics, visualize_grid, create_error_map, save_rendering_comparison, compute_rendering_statistics, evaluate_model_performance, plot_training_curves, save_image, load_image, create_video_from_images, create_comparison_grid
)

# Re-export setup_logging and other functions from the main utils module
# We import them at the module level to avoid circular imports
import sys
from pathlib import Path

# Add the parent directory to allow importing from utils.py
_current_dir = Path(__file__).parent
_parent_dir = _current_dir.parent
if str(_parent_dir) not in sys.path:
    sys.path.insert(0, str(_parent_dir))

# Import from utils.py without creating circular import
try:
    from nerfs.grid_nerf.utils import (
        setup_logging, load_config, save_config, get_learning_rate_scheduler, positional_encoding, safe_normalize, get_ray_directions, sample_along_rays, volume_rendering
    )
except ImportError:
    # If import fails, we'll define these functions at runtime
    def setup_logging(*args, **kwargs) -> None:
        """Set up logging configuration."""
        import logging
        if args and args[0]:
            from pathlib import Path
            log_file = Path(args[0])
            log_file.parent.mkdir(parents=True, exist_ok=True)
            logging.basicConfig(
                level=kwargs.get('level', logging.INFO),
                handlers=[logging.StreamHandler(), logging.FileHandler(log_file)]
            )
        else:
            logging.basicConfig(
                level=kwargs.get(
                    'level',
                    logging.INFO,
                )
            )
    
    def load_config(*args, **kwargs) -> dict[str, Any]:
        """Load configuration from file."""
        return {}
    
    def save_config(*args, **kwargs) -> None:
        """Save configuration to file."""
        pass
    
    def get_learning_rate_scheduler(*args, **kwargs) -> Optional[Any]:
        """Get learning rate scheduler."""
        return None
    
    def positional_encoding(*args, **kwargs) -> Any:
        """Apply positional encoding."""
        return args[0] if args else None
    
    def safe_normalize(*args, **kwargs) -> Any:
        """Safely normalize tensor."""
        return args[0] if args else None
    
    def get_ray_directions(*args, **kwargs) -> Optional[Any]:
        """Get ray directions."""
        return None
    
    def sample_along_rays(*args, **kwargs) -> tuple[Optional[Any], Optional[Any]]:
        """Sample points along rays."""
        return None, None
    
    def volume_rendering(*args, **kwargs) -> dict[str, Any]:
        """Perform volume rendering."""
        return {}

__version__ = "1.0.0"

__all__ = [
    # Grid utilities
    'compute_grid_bounds', 'world_to_grid_coords', 'grid_to_world_coords', 'trilinear_interpolation', 'create_voxel_grid', 'dilate_grid', 'prune_grid', 'compute_grid_occupancy', 'adaptive_grid_subdivision', 'compute_grid_gradient', 'smooth_grid', 'compute_grid_statistics', # Ray utilities
    'generate_rays_from_camera', 'sample_points_along_rays', 'hierarchical_sampling', 'ray_aabb_intersection', 'compute_ray_grid_intersections', 'voxel_traversal_3d', 'importance_sampling_from_grid', 'compute_ray_weights', 'sample_stratified_rays', # Metrics utilities
    'compute_psnr', 'compute_ssim', 'compute_lpips', 'compute_depth_metrics', 'compute_novel_view_metrics', 'visualize_grid', 'create_error_map', 'save_rendering_comparison', 'compute_rendering_statistics', 'evaluate_model_performance', 'plot_training_curves', 'save_image', 'load_image', 'create_video_from_images', 'create_comparison_grid', # Additional utilities from main utils.py
    'setup_logging', 'load_config', 'save_config', 'get_learning_rate_scheduler', 'positional_encoding', 'safe_normalize', 'get_ray_directions', 'sample_along_rays', 'volume_rendering'
] 