"""
Ray utilities for Grid-NeRF.

This module provides utility functions for ray operations, sampling strategies,
and ray-grid intersection computations specific to Grid-NeRF.
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import Tuple, List, Optional, Union, Dict
import math


def generate_rays_from_camera(camera_poses: torch.Tensor,
                             camera_intrinsics: torch.Tensor,
                             image_height: int,
                             image_width: int,
                             device: str = 'cuda') -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Generate rays from camera parameters.
    
    Args:
        camera_poses: Camera poses [N, 4, 4] (camera to world transform)
        camera_intrinsics: Camera intrinsics [N, 3, 3]
        image_height: Image height
        image_width: Image width
        device: Device to use
        
    Returns:
        Tuple of (ray_origins, ray_directions) each [N, H, W, 3]
    """
    N = camera_poses.shape[0]
    
    # Create pixel coordinates
    i, j = torch.meshgrid(torch.arange(image_width, device=device),
                         torch.arange(image_height, device=device),
                         indexing='ij')
    i = i.t().float()  # [H, W]
    j = j.t().float()  # [H, W]
    
    # Homogeneous pixel coordinates
    pixel_coords = torch.stack([i, j, torch.ones_like(i)], dim=-1)  # [H, W, 3]
    
    ray_origins_list = []
    ray_directions_list = []
    
    for n in range(N):
        # Get camera intrinsics and pose
        K = camera_intrinsics[n]  # [3, 3]
        c2w = camera_poses[n]     # [4, 4]
        
        # Convert pixel coordinates to camera coordinates
        K_inv = torch.inverse(K)
        camera_coords = torch.matmul(pixel_coords, K_inv.T)  # [H, W, 3]
        
        # Normalize direction vectors
        camera_dirs = camera_coords / torch.norm(camera_coords, dim=-1, keepdim=True)
        
        # Transform to world coordinates
        R = c2w[:3, :3]  # [3, 3]
        t = c2w[:3, 3]   # [3]
        
        # Ray directions in world coordinates
        world_dirs = torch.matmul(camera_dirs, R.T)  # [H, W, 3]
        
        # Ray origins (camera center in world coordinates)
        world_origins = t.expand_as(world_dirs)  # [H, W, 3]
        
        ray_origins_list.append(world_origins)
        ray_directions_list.append(world_dirs)
    
    ray_origins = torch.stack(ray_origins_list, dim=0)    # [N, H, W, 3]
    ray_directions = torch.stack(ray_directions_list, dim=0)  # [N, H, W, 3]
    
    return ray_origins, ray_directions


def sample_points_along_rays(ray_origins: torch.Tensor,
                           ray_directions: torch.Tensor,
                           near: float,
                           far: float,
                           num_samples: int,
                           stratified: bool = True,
                           perturb: bool = True) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Sample points along rays.
    
    Args:
        ray_origins: Ray origins [..., 3]
        ray_directions: Ray directions [..., 3]
        near: Near plane distance
        far: Far plane distance
        num_samples: Number of samples per ray
        stratified: Whether to use stratified sampling
        perturb: Whether to add random perturbation
        
    Returns:
        Tuple of (sample_points, t_vals) where:
        - sample_points: [..., num_samples, 3]
        - t_vals: [..., num_samples]
    """
    batch_shape = ray_origins.shape[:-1]
    device = ray_origins.device
    
    if stratified:
        # Stratified sampling
        t_vals = torch.linspace(near, far, num_samples, device=device)
        t_vals = t_vals.expand(*batch_shape, num_samples)
        
        if perturb:
            # Add random perturbation
            mids = 0.5 * (t_vals[..., 1:] + t_vals[..., :-1])
            upper = torch.cat([mids, t_vals[..., -1:]], dim=-1)
            lower = torch.cat([t_vals[..., :1], mids], dim=-1)
            
            # Random uniform samples
            t_rand = torch.rand_like(t_vals)
            t_vals = lower + (upper - lower) * t_rand
    else:
        # Uniform sampling
        t_vals = torch.linspace(near, far, num_samples, device=device)
        t_vals = t_vals.expand(*batch_shape, num_samples)
    
    # Compute sample points
    sample_points = ray_origins.unsqueeze(-2) + t_vals.unsqueeze(-1) * ray_directions.unsqueeze(-2)
    
    return sample_points, t_vals


def hierarchical_sampling(ray_origins: torch.Tensor,
                         ray_directions: torch.Tensor,
                         t_vals_coarse: torch.Tensor,
                         weights_coarse: torch.Tensor,
                         num_fine_samples: int,
                         perturb: bool = True) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Perform hierarchical sampling based on coarse weights.
    
    Args:
        ray_origins: Ray origins [..., 3]
        ray_directions: Ray directions [..., 3]
        t_vals_coarse: Coarse t values [..., N_coarse]
        weights_coarse: Coarse weights [..., N_coarse]
        num_fine_samples: Number of fine samples
        perturb: Whether to add perturbation
        
    Returns:
        Tuple of (fine_points, t_vals_fine)
    """
    batch_shape = ray_origins.shape[:-1]
    device = ray_origins.device
    
    # Get PDF from weights
    weights = weights_coarse + 1e-5  # Prevent division by zero
    pdf = weights / torch.sum(weights, dim=-1, keepdim=True)
    cdf = torch.cumsum(pdf, dim=-1)
    cdf = torch.cat([torch.zeros_like(cdf[..., :1]), cdf], dim=-1)
    
    # Sample from CDF
    u = torch.rand(*batch_shape, num_fine_samples, device=device)
    
    if perturb:
        u = u.contiguous()
    
    # Invert CDF
    indices = torch.searchsorted(cdf, u, right=True)
    below = torch.clamp(indices - 1, 0, cdf.shape[-1] - 1)
    above = torch.clamp(indices, 0, cdf.shape[-1] - 1)
    
    indices_g = torch.stack([below, above], dim=-1)
    matched_shape = list(indices_g.shape[:-1]) + [cdf.shape[-1]]
    
    cdf_g = torch.gather(cdf.unsqueeze(-2).expand(matched_shape), dim=-1, 
                        index=indices_g)
    bins_g = torch.gather(t_vals_coarse.unsqueeze(-2).expand(matched_shape), dim=-1,
                         index=indices_g)
    
    denom = cdf_g[..., 1] - cdf_g[..., 0]
    denom = torch.where(denom < 1e-5, torch.ones_like(denom), denom)
    
    t = (u - cdf_g[..., 0]) / denom
    t_vals_fine = bins_g[..., 0] + t * (bins_g[..., 1] - bins_g[..., 0])
    
    # Combine coarse and fine samples
    t_vals_combined, _ = torch.sort(torch.cat([t_vals_coarse, t_vals_fine], dim=-1), dim=-1)
    
    # Compute fine sample points
    fine_points = ray_origins.unsqueeze(-2) + t_vals_combined.unsqueeze(-1) * ray_directions.unsqueeze(-2)
    
    return fine_points, t_vals_combined


def ray_aabb_intersection(ray_origins: torch.Tensor,
                         ray_directions: torch.Tensor,
                         aabb_min: torch.Tensor,
                         aabb_max: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute ray-AABB intersection.
    
    Args:
        ray_origins: Ray origins [..., 3]
        ray_directions: Ray directions [..., 3]
        aabb_min: AABB minimum bounds [3]
        aabb_max: AABB maximum bounds [3]
        
    Returns:
        Tuple of (t_near, t_far) for intersection points
    """
    # Compute intersection with each slab
    t_min = (aabb_min - ray_origins) / (ray_directions + 1e-8)
    t_max = (aabb_max - ray_origins) / (ray_directions + 1e-8)
    
    # Ensure t_min < t_max
    t1 = torch.minimum(t_min, t_max)
    t2 = torch.maximum(t_min, t_max)
    
    # Compute near and far intersection points
    t_near = torch.maximum(torch.maximum(t1[..., 0], t1[..., 1]), t1[..., 2])
    t_far = torch.minimum(torch.minimum(t2[..., 0], t2[..., 1]), t2[..., 2])
    
    return t_near, t_far


def compute_ray_grid_intersections(ray_origins: torch.Tensor,
                                  ray_directions: torch.Tensor,
                                  grid_bounds: torch.Tensor,
                                  grid_resolution: int) -> Dict[str, torch.Tensor]:
    """
    Compute which grid cells each ray intersects.
    
    Args:
        ray_origins: Ray origins [..., 3]
        ray_directions: Ray directions [..., 3]
        grid_bounds: Grid bounds [6] (x_min, y_min, z_min, x_max, y_max, z_max)
        grid_resolution: Grid resolution
        
    Returns:
        Dictionary containing intersection information
    """
    batch_shape = ray_origins.shape[:-1]
    device = ray_origins.device
    
    # Compute ray-grid intersection
    aabb_min = grid_bounds[:3]
    aabb_max = grid_bounds[3:]
    
    t_near, t_far = ray_aabb_intersection(ray_origins, ray_directions, aabb_min, aabb_max)
    
    # Check if ray intersects grid
    valid_rays = (t_near <= t_far) & (t_far > 0)
    
    # Compute entry and exit points
    t_near = torch.clamp(t_near, min=0)
    entry_points = ray_origins + t_near.unsqueeze(-1) * ray_directions
    exit_points = ray_origins + t_far.unsqueeze(-1) * ray_directions
    
    # Convert to grid coordinates
    grid_size = aabb_max - aabb_min
    entry_grid = (entry_points - aabb_min) / grid_size * (grid_resolution - 1)
    exit_grid = (exit_points - aabb_min) / grid_size * (grid_resolution - 1)
    
    return {
        'valid_rays': valid_rays,
        't_near': t_near,
        't_far': t_far,
        'entry_points': entry_points,
        'exit_points': exit_points,
        'entry_grid': entry_grid,
        'exit_grid': exit_grid
    }


def voxel_traversal_3d(ray_origins: torch.Tensor,
                      ray_directions: torch.Tensor,
                      grid_bounds: torch.Tensor,
                      grid_resolution: int,
                      max_steps: int = 1000) -> Dict[str, torch.Tensor]:
    """
    3D voxel traversal algorithm (3D DDA).
    
    Args:
        ray_origins: Ray origins [..., 3]
        ray_directions: Ray directions [..., 3]
        grid_bounds: Grid bounds [6]
        grid_resolution: Grid resolution
        max_steps: Maximum traversal steps
        
    Returns:
        Dictionary containing traversal information
    """
    batch_shape = ray_origins.shape[:-1]
    device = ray_origins.device
    
    # Convert to grid coordinates
    grid_size = grid_bounds[3:] - grid_bounds[:3]
    grid_origins = (ray_origins - grid_bounds[:3]) / grid_size * (grid_resolution - 1)
    
    # Normalize ray directions
    ray_dirs_norm = ray_directions / torch.norm(ray_directions, dim=-1, keepdim=True)
    
    # Current voxel indices
    current_voxel = torch.floor(grid_origins).long()
    
    # Direction signs
    step = torch.sign(ray_dirs_norm).long()
    
    # Distance to next voxel boundary
    next_boundary = torch.where(step > 0, 
                               current_voxel + 1, 
                               current_voxel).float()
    
    # Distance along ray to next boundary
    t_max = (next_boundary - grid_origins) / (ray_dirs_norm + 1e-8)
    
    # Distance between voxel boundaries along ray
    t_delta = torch.abs(1.0 / (ray_dirs_norm + 1e-8))
    
    # Initialize traversal
    traversed_voxels = []
    t_values = []
    
    for _ in range(max_steps):
        # Check bounds
        valid = ((current_voxel >= 0) & (current_voxel < grid_resolution)).all(dim=-1)
        
        if not valid.any():
            break
        
        # Store current voxel
        traversed_voxels.append(current_voxel.clone())
        t_values.append(torch.min(t_max, dim=-1)[0])
        
        # Find which axis to step along
        min_axis = torch.argmin(t_max, dim=-1)
        
        # Step to next voxel
        for i in range(3):
            mask = (min_axis == i)
            current_voxel[mask, i] += step[mask, i]
            t_max[mask, i] += t_delta[mask, i]
    
    return {
        'traversed_voxels': torch.stack(traversed_voxels, dim=-2) if traversed_voxels else torch.empty(*batch_shape, 0, 3),
        't_values': torch.stack(t_values, dim=-1) if t_values else torch.empty(*batch_shape, 0)
    }


def importance_sampling_from_grid(ray_origins: torch.Tensor,
                                 ray_directions: torch.Tensor,
                                 grid: torch.Tensor,
                                 grid_bounds: torch.Tensor,
                                 num_samples: int,
                                 near: float = 0.1,
                                 far: float = 10.0) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Perform importance sampling based on grid occupancy.
    
    Args:
        ray_origins: Ray origins [..., 3]
        ray_directions: Ray directions [..., 3]
        grid: Occupancy grid [D, H, W]
        grid_bounds: Grid bounds [6]
        num_samples: Number of samples
        near: Near plane
        far: Far plane
        
    Returns:
        Tuple of (sample_points, t_vals)
    """
    batch_shape = ray_origins.shape[:-1]
    device = ray_origins.device
    
    # Initial uniform sampling
    t_vals_uniform = torch.linspace(near, far, num_samples * 2, device=device)
    t_vals_uniform = t_vals_uniform.expand(*batch_shape, num_samples * 2)
    
    # Sample points
    sample_points = ray_origins.unsqueeze(-2) + t_vals_uniform.unsqueeze(-1) * ray_directions.unsqueeze(-2)
    
    # Convert to grid coordinates
    grid_size = grid_bounds[3:] - grid_bounds[:3]
    grid_coords = (sample_points - grid_bounds[:3]) / grid_size * (grid.shape[0] - 1)
    
    # Sample grid occupancy
    grid_expanded = grid.unsqueeze(0).unsqueeze(0).float()  # [1, 1, D, H, W]
    
    # Normalize coordinates for grid_sample
    normalized_coords = grid_coords.clone()
    for i in range(3):
        normalized_coords[..., i] = (normalized_coords[..., i] / (grid.shape[i] - 1)) * 2 - 1
    
    # Sample occupancy
    occupancy = F.grid_sample(
        grid_expanded,
        normalized_coords.unsqueeze(-3),  # Add dimension for grid_sample
        mode='bilinear',
        padding_mode='border',
        align_corners=True
    ).squeeze(0).squeeze(0).squeeze(-2)
    
    # Convert to weights
    weights = occupancy + 1e-5
    weights = weights / torch.sum(weights, dim=-1, keepdim=True)
    
    # Sample based on importance
    cdf = torch.cumsum(weights, dim=-1)
    cdf = torch.cat([torch.zeros_like(cdf[..., :1]), cdf], dim=-1)
    
    # Random sampling
    u = torch.rand(*batch_shape, num_samples, device=device)
    
    # Invert CDF
    indices = torch.searchsorted(cdf, u, right=True)
    indices = torch.clamp(indices, 0, len(t_vals_uniform[0]) - 1)
    
    # Get corresponding t values
    t_vals_sampled = torch.gather(t_vals_uniform, -1, indices)
    
    # Compute final sample points
    final_sample_points = ray_origins.unsqueeze(-2) + t_vals_sampled.unsqueeze(-1) * ray_directions.unsqueeze(-2)
    
    return final_sample_points, t_vals_sampled


def compute_ray_weights(densities: torch.Tensor,
                       t_vals: torch.Tensor,
                       ray_directions: torch.Tensor) -> torch.Tensor:
    """
    Compute volumetric rendering weights from densities.
    
    Args:
        densities: Density values [..., N_samples]
        t_vals: t values along rays [..., N_samples]
        ray_directions: Ray directions [..., 3]
        
    Returns:
        Weights [..., N_samples]
    """
    # Compute distances between adjacent samples
    dists = t_vals[..., 1:] - t_vals[..., :-1]
    
    # Add distance for last sample
    last_dist = torch.full_like(dists[..., :1], 1e10)
    dists = torch.cat([dists, last_dist], dim=-1)
    
    # Account for ray direction magnitude
    dists = dists * torch.norm(ray_directions, dim=-1, keepdim=True)
    
    # Compute alpha values
    alpha = 1.0 - torch.exp(-densities * dists)
    
    # Compute transmittance
    transmittance = torch.cumprod(1.0 - alpha + 1e-10, dim=-1)
    transmittance = torch.cat([torch.ones_like(transmittance[..., :1]), 
                              transmittance[..., :-1]], dim=-1)
    
    # Compute weights
    weights = alpha * transmittance
    
    return weights


def sample_stratified_rays(num_rays: int,
                          image_height: int,
                          image_width: int,
                          device: str = 'cuda') -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Sample rays with stratified sampling.
    
    Args:
        num_rays: Number of rays to sample
        image_height: Image height
        image_width: Image width
        device: Device to use
        
    Returns:
        Tuple of (i_indices, j_indices) for pixel coordinates
    """
    # Stratified sampling
    i_coords = torch.rand(num_rays, device=device) * image_width
    j_coords = torch.rand(num_rays, device=device) * image_height
    
    i_indices = torch.floor(i_coords).long()
    j_indices = torch.floor(j_coords).long()
    
    # Clamp to valid range
    i_indices = torch.clamp(i_indices, 0, image_width - 1)
    j_indices = torch.clamp(j_indices, 0, image_height - 1)
    
    return i_indices, j_indices 