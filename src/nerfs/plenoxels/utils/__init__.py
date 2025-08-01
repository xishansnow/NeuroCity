"""
Plenoxels Utilities Package

This package contains utility functions for Plenoxels operations.
"""

from .voxel_utils import (
    create_voxel_grid,
    prune_voxel_grid,
    compute_voxel_bounds,
    voxel_to_world_coords,
    world_to_voxel_coords,
)

from .rendering_utils import (
    generate_rays,
    sample_points_along_rays,
    volume_render,
    compute_ray_aabb_intersection,
    compute_ray_bundle_intersections,
    ray_sphere_intersection,
    compute_optical_depth,
    alpha_composite,
    compute_expected_depth,
    compute_depth_variance,
)

from .visualization_utils import (
    visualize_voxel_grid,
    visualize_density_field,
    create_occupancy_visualization,
    render_novel_view_video,
)

from .metrics_utils import (
    compute_psnr,
    compute_ssim,
    compute_lpips,
    compute_all_metrics,
)

__all__ = [
    # Voxel utilities
    "create_voxel_grid",
    "prune_voxel_grid",
    "compute_voxel_bounds",
    "voxel_to_world_coords",
    "world_to_voxel_coords",  # Rendering utilities
    "generate_rays",
    "sample_points_along_rays",
    "volume_render",
    "compute_ray_aabb_intersection",  # Visualization utilities
    "visualize_voxel_grid",
    "visualize_density_field",
    "create_occupancy_visualization",
    "render_novel_view_video",
    "compute_psnr",
    "compute_ssim",
    "compute_lpips",
    "compute_all_metrics",
]
