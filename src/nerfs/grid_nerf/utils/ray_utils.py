"""
Ray utilities for Grid-NeRF.

This module provides utility functions for ray operations, sampling strategies, and ray-grid intersection computations specific to Grid-NeRF.
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import List, Optional, Tuple, Union 
import math


def generate_rays_from_camera(
    camera_poses: torch.Tensor,
    camera_intrinsics: torch.Tensor,
    image_height: int,
    image_width: int,
    device: str = 'cuda'
) -> tuple[torch.Tensor, torch.Tensor]:
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
    i, j = torch.meshgrid(
        torch.arange(image_height),
        torch.arange(image_width),
        indexing='ij'
    )
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


def sample_points_along_rays(
    ray_origins: torch.Tensor,
    ray_directions: torch.Tensor,
    near: float,
    far: float,
    num_samples: int,
    stratified: bool = True,
    perturb: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
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


def hierarchical_sampling(
    ray_origins: torch.Tensor,
    ray_directions: torch.Tensor,
    t_vals_coarse: torch.Tensor,
    weights_coarse: torch.Tensor,
    num_fine_samples: int,
    perturb: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
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
    
    cdf_g = torch.gather(cdf.unsqueeze(-2).expand(matched_shape), dim=-1, index=indices_g)
    bins_g = torch.gather(
        t_vals_coarse.unsqueeze(-2),
        dim=-1,
        index=indices.unsqueeze(-1)
    )
    
    denom = cdf_g[..., 1] - cdf_g[..., 0]
    denom = torch.where(denom < 1e-5, torch.ones_like(denom), denom)
    
    t = (u - cdf_g[..., 0]) / denom
    t_vals_fine = bins_g[..., 0] + t * (bins_g[..., 1] - bins_g[..., 0])
    
    # Combine coarse and fine samples
    t_vals_combined, _ = torch.sort(torch.cat([t_vals_coarse, t_vals_fine], dim=-1), dim=-1)
    
    # Compute fine sample points
    fine_points = ray_origins.unsqueeze(-2) + t_vals_combined.unsqueeze(-1) * ray_directions.unsqueeze(-2)
    
    return fine_points, t_vals_combined


def ray_aabb_intersection(
    ray_origins: torch.Tensor,
    ray_directions: torch.Tensor,
    aabb_min: torch.Tensor,
    aabb_max: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
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


def compute_ray_grid_intersections(
    ray_origins: torch.Tensor,
    ray_directions: torch.Tensor,
    grid_bounds: torch.Tensor,
    grid_resolution: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Compute which grid cells each ray intersects.
    
    Args:
        ray_origins: Ray origins [..., 3]
        ray_directions: Ray directions [..., 3]
        grid_bounds: Grid bounds [6] (x_min, y_min, z_min, x_max, y_max, z_max)
        grid_resolution: Grid resolution
        
    Returns:
        Tuple of (cell_indices, t_starts, t_ends) where:
        - cell_indices: [..., num_cells, 3] indices of intersected cells
        - t_starts: [..., num_cells] start distances along rays
        - t_ends: [..., num_cells] end distances along rays
    """
    # Get grid bounds
    grid_min = grid_bounds[:3]
    grid_max = grid_bounds[3:]
    
    # Compute ray-grid intersection
    t_near, t_far = ray_aabb_intersection(ray_origins, ray_directions, grid_min, grid_max)
    
    # Early exit if no intersection
    valid_rays = t_far > t_near
    if not valid_rays.any():
        return torch.empty(0, 3), torch.empty(0), torch.empty(0)
    
    # Compute cell size
    cell_size = (grid_max - grid_min) / grid_resolution
    
    # Initialize DDA algorithm
    pos = ray_origins + t_near.unsqueeze(-1) * ray_directions
    cell_indices = ((pos - grid_min) / cell_size).long()
    cell_indices = torch.clamp(cell_indices, 0, grid_resolution - 1)
    
    # Compute t values at cell boundaries
    next_t = torch.zeros_like(t_near)
    for dim in range(3):
        cell_boundary = grid_min[dim] + (cell_indices[..., dim] + 1) * cell_size[dim]
        t = (cell_boundary - ray_origins[..., dim]) / ray_directions[..., dim]
        next_t = torch.where(t > t_near, t, next_t)
    
    return cell_indices, t_near, next_t


def voxel_traversal_3d(
    ray_origins: torch.Tensor,
    ray_directions: torch.Tensor,
    grid_bounds: torch.Tensor,
    grid_resolution: int,
    max_steps: int = 1000,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Perform 3D voxel traversal using DDA algorithm.
    
    Args:
        ray_origins: Ray origins [..., 3]
        ray_directions: Ray directions [..., 3]
        grid_bounds: Grid bounds [6]
        grid_resolution: Grid resolution
        max_steps: Maximum number of steps
        
    Returns:
        Tuple of (voxel_indices, t_starts, t_ends) where:
        - voxel_indices: [..., num_steps, 3] indices of traversed voxels
        - t_starts: [..., num_steps] start distances along rays
        - t_ends: [..., num_steps] end distances along rays
    """
    # Get grid bounds
    grid_min = grid_bounds[:3]
    grid_max = grid_bounds[3:]
    
    # Compute cell size
    cell_size = (grid_max - grid_min) / grid_resolution
    
    # Initialize DDA
    t_near, t_far = ray_aabb_intersection(ray_origins, ray_directions, grid_min, grid_max)
    pos = ray_origins + t_near.unsqueeze(-1) * ray_directions
    
    # Get initial voxel
    voxel = ((pos - grid_min) / cell_size).long()
    voxel = torch.clamp(voxel, 0, grid_resolution - 1)
    
    # Compute step direction and t_delta
    step = torch.sign(ray_directions)
    t_delta = cell_size / (ray_directions + 1e-8)
    t_delta = torch.abs(t_delta)
    
    # Initialize outputs
    voxel_indices = []
    t_starts = []
    t_ends = []
    
    # Traverse grid
    t = t_near
    for _ in range(max_steps):
        voxel_indices.append(voxel)
        t_starts.append(t)
        
        # Compute next intersection
        t_next = torch.zeros_like(t)
        next_voxel = voxel.clone()
        
        for dim in range(3):
            if step[..., dim] != 0:
                t_dim = ((voxel[..., dim] + step[..., dim]) * cell_size[dim] + grid_min[dim] - ray_origins[..., dim]) / ray_directions[..., dim]
                mask = t_dim < t_next
                t_next = torch.where(mask, t_dim, t_next)
                next_voxel = torch.where(mask.unsqueeze(-1), voxel + step * torch.eye(3)[dim], next_voxel)
        
        t_ends.append(t_next)
        
        # Update current voxel and time
        voxel = next_voxel
        t = t_next
        
        # Check if we're outside grid
        outside = (voxel < 0).any(-1) | (voxel >= grid_resolution).any(-1)
        if outside.all():
            break
    
    # Stack outputs
    voxel_indices = torch.stack(voxel_indices, dim=-2)
    t_starts = torch.stack(t_starts, dim=-1)
    t_ends = torch.stack(t_ends, dim=-1)
    
    return voxel_indices, t_starts, t_ends


def importance_sampling_from_grid(
    ray_origins: torch.Tensor,
    ray_directions: torch.Tensor,
    grid: torch.Tensor,
    grid_bounds: torch.Tensor,
    num_samples: int,
    near: float = 0.1,
    far: float = 10.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Sample points along rays based on grid importance.
    
    Args:
        ray_origins: Ray origins [..., 3]
        ray_directions: Ray directions [..., 3]
        grid: Importance grid [D, H, W]
        grid_bounds: Grid bounds [6]
        num_samples: Number of samples per ray
        near: Near plane distance
        far: Far plane distance
        
    Returns:
        Tuple of (sample_points, t_vals)
    """
    # Get grid resolution
    D, H, W = grid.shape
    grid_resolution = max(D, H, W)
    
    # Compute ray-grid intersections
    voxel_indices, t_starts, t_ends = voxel_traversal_3d(
        ray_origins,
        ray_directions,
        grid_bounds,
        grid_resolution
    )
    
    # Get importance values for intersected voxels
    importance = grid[voxel_indices[..., 0], voxel_indices[..., 1], voxel_indices[..., 2]]
    
    # Normalize importance values
    importance = importance / (importance.sum(-1, keepdim=True) + 1e-8)
    
    # Sample points proportional to importance
    u = torch.rand(*ray_origins.shape[:-1], num_samples, device=ray_origins.device)
    cdf = torch.cumsum(importance, dim=-1)
    cdf = torch.cat([torch.zeros_like(cdf[..., :1]), cdf], dim=-1)
    
    # Invert CDF
    indices = torch.searchsorted(cdf, u, right=True)
    below = torch.clamp(indices - 1, 0, cdf.shape[-1] - 1)
    above = torch.clamp(indices, 0, cdf.shape[-1] - 1)
    
    # Sample within selected intervals
    t_low = torch.gather(t_starts, dim=-1, index=below)
    t_high = torch.gather(t_ends, dim=-1, index=above)
    t_vals = t_low + torch.rand_like(t_low) * (t_high - t_low)
    
    # Compute sample points
    sample_points = ray_origins.unsqueeze(-2) + t_vals.unsqueeze(-1) * ray_directions.unsqueeze(-2)
    
    return sample_points, t_vals


def compute_ray_weights(
    densities: torch.Tensor,
    t_vals: torch.Tensor,
    ray_directions: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Compute weights for volume rendering.
    
    Args:
        densities: Density values [..., num_samples]
        t_vals: Sample distances [..., num_samples]
        ray_directions: Ray directions [..., 3]
        
    Returns:
        Tuple of (weights, transmittance)
    """
    # Compute delta distances
    dists = t_vals[..., 1:] - t_vals[..., :-1]
    dists = torch.cat([dists, torch.ones_like(dists[..., :1]) * 1e10], dim=-1)
    
    # Scale by ray direction length
    dists = dists * torch.norm(ray_directions.unsqueeze(-1), dim=-2)
    
    # Compute alpha values
    alpha = 1.0 - torch.exp(-densities * dists)
    
    # Compute transmittance and weights
    transmittance = torch.cumprod(1.0 - alpha + 1e-10, dim=-1)
    transmittance = torch.cat([torch.ones_like(transmittance[..., :1]), transmittance[..., :-1]], dim=-1)
    weights = alpha * transmittance
    
    return weights, transmittance


def sample_stratified_rays(
    num_rays: int,
    image_height: int,
    image_width: int,
    device: str = 'cuda',
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Sample stratified rays from image plane.
    
    Args:
        num_rays: Number of rays to sample
        image_height: Image height
        image_width: Image width
        device: Device to use
        
    Returns:
        Tuple of (pixel_x, pixel_y) coordinates
    """
    # Create stratified grid
    num_h = int(math.sqrt(num_rays * image_height / image_width))
    num_w = int(num_rays / num_h)
    
    # Create pixel coordinates
    pixel_x = torch.linspace(0, image_width - 1, num_w, device=device)
    pixel_y = torch.linspace(0, image_height - 1, num_h, device=device)
    
    # Add random offset
    pixel_x = pixel_x + torch.rand_like(pixel_x)
    pixel_y = pixel_y + torch.rand_like(pixel_y)
    
    # Create meshgrid
    pixel_x, pixel_y = torch.meshgrid(pixel_x, pixel_y, indexing='ij')
    
    # Reshape to [num_rays, 2]
    pixel_x = pixel_x.reshape(-1)
    pixel_y = pixel_y.reshape(-1)
    
    return pixel_x, pixel_y 