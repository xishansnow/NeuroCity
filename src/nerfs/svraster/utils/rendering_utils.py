from __future__ import annotations

from typing import Optional, List, Tuple

"""
Rendering utilities for SVRaster.
"""

import torch
import numpy as np


def ray_direction_dependent_ordering(
    voxel_positions_or_origins: torch.Tensor,
    morton_codes_or_directions: torch.Tensor,
    ray_direction: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Reorder voxels based on ray direction for correct depth ordering.

    Args:
        voxel_positions_or_origins: Voxel positions [N, 3] or ray origins [N, 3]
        morton_codes_or_directions: Morton codes [N] or ray directions [N, 3]
        ray_direction: Mean ray direction [3] (optional)

    Returns:
        Sort indices for correct depth ordering
    """
    device = voxel_positions_or_origins.device

    # Handle different call signatures
    if ray_direction is None:
        # Called with (ray_origins, ray_directions)
        ray_origins = voxel_positions_or_origins
        ray_directions = morton_codes_or_directions

        # Compute distances from origin for ordering
        distances = torch.norm(ray_origins, dim=1)

        # Create ordering indices
        sort_indices = torch.argsort(distances)

        return sort_indices
    else:
        # Called with (voxel_positions, morton_codes, ray_direction)
        voxel_positions = voxel_positions_or_origins
        morton_codes = morton_codes_or_directions
        ray_direction = ray_direction.to(device)
        morton_codes = morton_codes.to(device)

        # Compute dot product with ray direction for secondary sorting
        dots = torch.sum(voxel_positions * ray_direction, dim=1)

        # Primary sort by Morton code, secondary by ray direction
        # This ensures correct depth order while maintaining spatial coherence
        sort_keys = morton_codes.float() + dots * 1e-6
        sort_indices = torch.argsort(sort_keys)

        return sort_indices


def depth_peeling(
    depths_or_positions: torch.Tensor,
    colors_or_sizes: torch.Tensor,
    ray_origin: Optional[torch.Tensor] = None,
    ray_direction: Optional[torch.Tensor] = None,
    num_layers: int = 4,
) -> list[torch.Tensor]:
    """
    Perform depth peeling for correct transparency rendering.

    Args:
        depths_or_positions: Depth values [N] or voxel positions [N, 3]
        colors_or_sizes: Color values [N, 3] or voxel sizes [N]
        ray_origin: Ray origin [3] (optional)
        ray_direction: Ray direction [3] (optional)
        num_layers: Number of depth layers

    Returns:
        List of sorted indices or data for each layer
    """
    if ray_origin is None and ray_direction is None:
        # Called with (depths, colors) - simple depth-based peeling
        depths = depths_or_positions
        colors = colors_or_sizes

        # Sort by depth
        sorted_indices = torch.argsort(depths)

        # Split into layers
        layer_size = len(sorted_indices) // num_layers
        layers = []

        for i in range(num_layers):
            start_idx = i * layer_size
            end_idx = (i + 1) * layer_size if i < num_layers - 1 else len(sorted_indices)
            layer_indices = sorted_indices[start_idx:end_idx]
            # Return tuples of (depth, color) for each layer
            layer_data = (depths[layer_indices], colors[layer_indices])
            layers.append(layer_data)

        return layers
    else:
        # Called with full arguments 将- position-based peeling
        voxel_positions = depths_or_positions
        voxel_sizes = colors_or_sizes

        # Ensure ray_origin and ray_direction are not None
        if ray_origin is not None and ray_direction is not None:
            # Compute distances from ray origin
            to_voxels = voxel_positions - ray_origin
            distances = torch.sum(to_voxels * ray_direction, dim=1)

            # Sort by distance
            sorted_indices = torch.argsort(distances)
        else:
            raise ValueError(
                "ray_origin and ray_direction must not be None for position-based peeling."
            )

        # Split into layers
        layer_size = len(sorted_indices) // num_layers
        layers = []

        for i in range(num_layers):
            start_idx = i * layer_size
            end_idx = (i + 1) * layer_size if i < num_layers - 1 else len(sorted_indices)
            layers.append(sorted_indices[start_idx:end_idx])

        return layers


def adaptive_sampling(
    ray_origin: torch.Tensor,
    ray_direction: torch.Tensor,
    near: float,
    far: float,
    num_samples: int,
    voxel_positions: torch.Tensor,
    voxel_sizes: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Adaptive sampling along rays based on voxel density.

    Args:
        ray_origin: Ray origin [3]
        ray_direction: Ray direction [3]
        near: Near plane distance
        far: Far plane distance
        num_samples: Number of samples
        voxel_positions: Voxel positions [N, 3]
        voxel_sizes: Voxel sizes [N]

    Returns:
        Sample points, distances, and weights
    """
    # Uniform sampling as baseline
    t_vals = torch.linspace(near, far, num_samples, device=ray_origin.device)
    sample_points = ray_origin + ray_direction * t_vals.unsqueeze(-1)

    # Compute weights based on nearby voxel density
    weights = torch.ones_like(t_vals)

    for i, point in enumerate(sample_points):
        # Find nearby voxels
        distances_to_voxels = torch.norm(voxel_positions - point, dim=1)
        nearby_mask = distances_to_voxels < (voxel_sizes * 2)

        if nearby_mask.any():
            # Increase weight where voxels are present
            weights[i] = 1.0 + nearby_mask.sum().float() * 0.1

    return sample_points, t_vals, weights


def hierarchical_sampling(
    ray_origin: torch.Tensor,
    ray_direction: torch.Tensor,
    coarse_samples: torch.Tensor,
    coarse_weights: torch.Tensor,
    num_fine_samples: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Hierarchical sampling for fine detail rendering.

    Args:
        ray_origin: Ray origin [3]
        ray_direction: Ray direction [3]
        coarse_samples: Coarse sample distances [N]
        coarse_weights: Coarse sample weights [N]
        num_fine_samples: Number of fine samples

    Returns:
        Fine sample points and distances
    """
    # Convert weights to PDF
    weights = coarse_weights + 1e-5  # Prevent division by zero
    pdf = weights / torch.sum(weights)
    cdf = torch.cumsum(pdf, dim=0)

    # Stratified sampling from CDF
    u = torch.rand(num_fine_samples, device=ray_origin.device)
    indices = torch.searchsorted(cdf, u, right=True)
    indices = torch.clamp(indices, 0, len(coarse_samples) - 1)

    # Interpolate sample positions
    fine_t_vals = coarse_samples[indices]
    fine_sample_points = ray_origin + ray_direction * fine_t_vals.unsqueeze(-1)

    return fine_sample_points, fine_t_vals


def compute_ray_aabb_intersection(
    ray_origin: torch.Tensor,
    ray_direction: torch.Tensor,
    aabb_min: torch.Tensor,
    aabb_max: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Compute ray-AABB intersection.

    Args:
        ray_origin: Ray origin [3]
        ray_direction: Ray direction [3]
        aabb_min: AABB minimum corner [3]
        aabb_max: AABB maximum corner [3]

    Returns:
        Intersection flag, t_near, t_far
    """
    # Compute intersection with each slab
    t1 = (aabb_min - ray_origin) / ray_direction
    t2 = (aabb_max - ray_origin) / ray_direction

    # Handle parallel rays
    t_min = torch.min(t1, t2)
    t_max = torch.max(t1, t2)

    # Find intersection interval
    t_near = torch.max(t_min)
    t_far = torch.min(t_max)

    # Check if intersection exists
    intersects = (t_far > t_near) & (t_far > 0)

    return intersects, t_near, t_far
