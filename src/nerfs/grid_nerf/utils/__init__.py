"""
Grid-NeRF Utils Package

This package contains utility modules for Grid-NeRF including:
- grid_utils: Grid operations and spatial computations
- ray_utils: Ray operations and sampling strategies
- metrics_utils: Evaluation metrics and visualization functions
"""

# Grid utilities
from .grid_utils import (
    compute_grid_bounds,
    world_to_grid_coords,
    grid_to_world_coords,
    trilinear_interpolation,
    create_voxel_grid,
    dilate_grid,
    prune_grid,
    compute_grid_occupancy,
    adaptive_grid_subdivision,
    compute_grid_gradient,
    smooth_grid,
    compute_grid_statistics
)

# Ray utilities
from .ray_utils import (
    generate_rays_from_camera,
    sample_points_along_rays,
    hierarchical_sampling,
    ray_aabb_intersection,
    compute_ray_grid_intersections,
    voxel_traversal_3d,
    importance_sampling_from_grid,
    compute_ray_weights,
    sample_stratified_rays
)

# Metrics utilities
from .metrics_utils import (
    compute_psnr,
    compute_ssim,
    compute_lpips,
    compute_depth_metrics,
    compute_novel_view_metrics,
    visualize_grid,
    create_error_map,
    save_rendering_comparison,
    compute_rendering_statistics,
    evaluate_model_performance,
    plot_training_curves,
    save_image,
    load_image,
    create_video_from_images,
    create_comparison_grid,
    load_config,
    save_config,
    setup_logging,
    get_learning_rate_scheduler,
    CosineAnnealingWarmRestarts,
    positional_encoding,
    safe_normalize,
    get_ray_directions,
    sample_along_rays,
    volume_rendering
)

__version__ = "1.0.0"

__all__ = [
    # Grid utilities
    'compute_grid_bounds',
    'world_to_grid_coords',
    'grid_to_world_coords',
    'trilinear_interpolation',
    'create_voxel_grid',
    'dilate_grid',
    'prune_grid',
    'compute_grid_occupancy',
    'adaptive_grid_subdivision',
    'compute_grid_gradient',
    'smooth_grid',
    'compute_grid_statistics',
    
    # Ray utilities
    'generate_rays_from_camera',
    'sample_points_along_rays',
    'hierarchical_sampling',
    'ray_aabb_intersection',
    'compute_ray_grid_intersections',
    'voxel_traversal_3d',
    'importance_sampling_from_grid',
    'compute_ray_weights',
    'sample_stratified_rays',
    
    # Metrics utilities
    'compute_psnr',
    'compute_ssim',
    'compute_lpips',
    'compute_depth_metrics',
    'compute_novel_view_metrics',
    'visualize_grid',
    'create_error_map',
    'save_rendering_comparison',
    'compute_rendering_statistics',
    'evaluate_model_performance',
    'plot_training_curves',
    'save_image',
    'load_image',
    'create_video_from_images',
    'create_comparison_grid',
    'load_config',
    'save_config',
    'setup_logging',
    'get_learning_rate_scheduler',
    'CosineAnnealingWarmRestarts',
    'positional_encoding',
    'safe_normalize',
    'get_ray_directions',
    'sample_along_rays',
    'volume_rendering'
] 