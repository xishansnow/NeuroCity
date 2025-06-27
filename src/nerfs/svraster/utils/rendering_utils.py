from typing import Optional
"""
Rendering utilities for SVRaster.
"""

import torch
import numpy as np

def ray_direction_dependent_ordering(
    voxel_positions: torch.Tensor,
    morton_codes: torch.Tensor,
    ray_direction: torch.Tensor,
) -> torch.Tensor:
    """
    Sort voxels using ray direction-dependent Morton ordering.
    
    Args:
        voxel_positions: Voxel positions [N, 3]
        morton_codes: Morton codes for voxels [N]
        ray_direction: Mean ray direction [3]
        
    Returns:
        Sort indices for correct depth ordering
    """
    # Compute dot product with ray direction for secondary sorting
    dots = torch.sum(voxel_positions * ray_direction, dim=1)
    
    # Primary sort by Morton code, secondary by ray direction
    # This ensures correct depth order while maintaining spatial coherence
    sort_keys = morton_codes.float() + dots * 1e-6
    sort_indices = torch.argsort(sort_keys)
    
    return sort_indices

def depth_peeling(
    voxel_positions: torch.Tensor,
    voxel_sizes: torch.Tensor,
    ray_origin: torch.Tensor,
    ray_direction: torch.Tensor,
    num_layers: int = 4,
) -> list[torch.Tensor]:
    """
    Perform depth peeling for correct transparency rendering.
    
    Args:
        voxel_positions: Voxel positions [N, 3]
        voxel_sizes: Voxel sizes [N]
        ray_origin: Ray origin [3]
        ray_direction: Ray direction [3]
        num_layers: Number of depth layers to peel
        
    Returns:
        list of voxel indices for each depth layer
    """
    # Compute distances from ray origin
    to_voxels = voxel_positions - ray_origin
    distances = torch.sum(to_voxels * ray_direction, dim=1)
    
    # Sort by distance
    sorted_indices = torch.argsort(distances)
    
    # Divide into layers
    num_voxels = voxel_positions.shape[0]
    layer_size = num_voxels // num_layers
    
    layers = []
    for i in range(num_layers):
        start_idx = i * layer_size
        end_idx = (i + 1) * layer_size if i < num_layers - 1 else num_voxels
        layer_indices = sorted_indices[start_idx:end_idx]
        layers.append(layer_indices)
    
    return layers

def compute_ray_aabb_intersection(
    ray_origin: torch.Tensor,
    ray_direction: torch.Tensor,
    box_min: torch.Tensor,
    box_max: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Compute ray-AABB intersection.
    
    Args:
        ray_origin: Ray origin [3] or [N, 3]
        ray_direction: Ray direction [3] or [N, 3]
        box_min: Box minimum corner [3] or [M, 3]
        box_max: Box maximum corner [3] or [M, 3]
        
    Returns:
        tuple of (t_near, t_far, valid_mask)
    """
    # Handle broadcasting
    if ray_origin.dim() == 1:
        ray_origin = ray_origin.unsqueeze(0)
    if ray_direction.dim() == 1:
        ray_direction = ray_direction.unsqueeze(0)
    if box_min.dim() == 1:
        box_min = box_min.unsqueeze(0)
    if box_max.dim() == 1:
        box_max = box_max.unsqueeze(0)
    
    # Compute intersection
    inv_dir = 1.0 / (ray_direction + 1e-8)
    
    t1 = (box_min - ray_origin) * inv_dir
    t2 = (box_max - ray_origin) * inv_dir
    
    t_near = torch.max(torch.min(t1, t2), dim=-1)[0]
    t_far = torch.min(torch.max(t1, t2), dim=-1)[0]
    
    # Valid intersections
    valid_mask = (t_near <= t_far) & (t_far > 0)
    
    return t_near, t_far, valid_mask

def volume_rendering_integration(
    densities: torch.Tensor,
    colors: torch.Tensor,
    distances: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Perform volume rendering integration.
    
    Args:
        densities: Density values along ray [N]
        colors: Color values along ray [N, 3]
        distances: Distance intervals [N]
        
    Returns:
        tuple of (integrated_color, integrated_alpha)
    """
    # Compute alpha values
    alphas = 1.0 - torch.exp(-densities * distances)
    
    # Compute transmittance
    transmittance = torch.cumprod(1.0 - alphas + 1e-8, dim=0)
    transmittance = torch.cat([torch.ones(1, device=transmittance.device), transmittance[:-1]])
    
    # Compute weights
    weights = alphas * transmittance
    
    # Integrate color
    integrated_color = torch.sum(weights.unsqueeze(-1) * colors, dim=0)
    
    # Compute total alpha
    integrated_alpha = 1.0 - transmittance[-1]
    
    return integrated_color, integrated_alpha

def render_rays(
    ray_origins: torch.Tensor, ray_directions: torch.Tensor, near: float, far: float, num_samples: int = 64, perturb: bool = True
) -> dict[str, torch.Tensor]:
    """Render rays using volume rendering."""
    # Get sample points and intervals
    samples = compute_ray_samples(ray_origins, ray_directions, near, far, num_samples, perturb)
    
    return {
        "samples": samples["points"], "intervals": samples["intervals"], "ray_indices": samples["ray_indices"]
    }

def compute_ray_samples(
    ray_origins: torch.Tensor, ray_directions: torch.Tensor, near: float, far: float, num_samples: int = 64, perturb: bool = True
) -> dict[str, torch.Tensor]:
    """Compute sample points along rays."""
    # Generate sample points
    t_vals = torch.linspace(0., 1., num_samples, device=ray_origins.device)
    z_vals = near * (1. - t_vals) + far * t_vals
    
    if perturb:
        # Random sampling
        mids = .5 * (z_vals[..., 1:] + z_vals[..., :-1])
        upper = torch.cat([mids, z_vals[..., -1:]], dim=-1)
        lower = torch.cat([z_vals[..., :1], mids], dim=-1)
        t_rand = torch.rand_like(z_vals)
        z_vals = lower + (upper - lower) * t_rand
    
    # Compute intervals
    intervals = z_vals[..., 1:] - z_vals[..., :-1]
    intervals = torch.cat([intervals, torch.tensor([1e10])])
    
    # Compute sample points
    points = ray_origins[..., None, :] + ray_directions[..., None, :] * z_vals[..., :, None]
    
    return {"points": points, "intervals": intervals, "z_vals": z_vals, "ray_indices": torch.arange(ray_origins.shape[0], device=ray_origins.device)} 