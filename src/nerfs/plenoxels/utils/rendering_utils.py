from __future__ import annotations

"""
Rendering Utilities for Plenoxels

This module provides rendering utilities including ray generation, point sampling, volume rendering, and ray-voxel intersection computations.
"""

from typing import Any, Optional


import torch
import numpy as np
import math


def generate_rays(
    poses: torch.Tensor,
    focal: float,
    H: int,
    W: int,
    near: float = 0.1,
    far: float = 10.0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Generate rays from camera poses.

    Args:
        poses: Camera poses [N, 3, 4] or [N, 4, 4]
        focal: Camera focal length
        H: Image height
        W: Image width
        near: Near plane distance
        far: Far plane distance

    Returns:
        tuple of (ray_origins, ray_directions)
    """
    device = poses.device
    N = poses.shape[0]

    # Create pixel coordinates
    i, j = torch.meshgrid(
        torch.arange(W, dtype=torch.float32, device=device),
        torch.arange(H, dtype=torch.float32, device=device),
        indexing="xy",
    )

    # Convert to camera coordinates
    dirs = torch.stack(
        [(i - W * 0.5) / focal, -(j - H * 0.5) / focal, -torch.ones_like(i)], dim=-1
    )  # [H, W, 3]

    rays_o = []
    rays_d = []

    for n in range(N):
        pose = poses[n]
        if pose.shape[0] == 4:
            pose = pose[:3, :4]  # Remove homogeneous coordinate

        # Transform ray directions to world coordinates
        rays_d_world = dirs @ pose[:3, :3].T  # [H, W, 3]
        rays_o_world = pose[:3, -1].expand_as(rays_d_world)  # [H, W, 3]

        rays_o.append(rays_o_world)
        rays_d.append(rays_d_world)

    rays_o = torch.stack(rays_o, dim=0)  # [N, H, W, 3]
    rays_d = torch.stack(rays_d, dim=0)  # [N, H, W, 3]

    return rays_o, rays_d


def sample_points_along_rays(
    ray_origins: torch.Tensor,
    ray_directions: torch.Tensor,
    near: float,
    far: float,
    num_samples: int,
    perturb: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Sample points along rays for volume rendering.

    Args:
        ray_origins: Ray origins [N, 3]
        ray_directions: Ray directions [N, 3]
        near: Near plane distance
        far: Far plane distance
        num_samples: Number of samples per ray
        perturb: Whether to add random perturbation

    Returns:
        tuple of (sample_points, t_values)
    """
    device = ray_origins.device
    N = ray_origins.shape[0]

    # Linear sampling in depth
    t_vals = torch.linspace(near, far, num_samples, device=device)
    t_vals = t_vals.expand(N, num_samples)  # [N, num_samples]

    if perturb:
        # Add random perturbation
        mids = 0.5 * (t_vals[..., 1:] + t_vals[..., :-1])
        upper = torch.cat([mids, t_vals[..., -1:]], dim=-1)
        lower = torch.cat([t_vals[..., :1], mids], dim=-1)
        t_rand = torch.rand_like(t_vals)
        t_vals = lower + (upper - lower) * t_rand

    # Compute sample points
    points = ray_origins.unsqueeze(-2) + t_vals.unsqueeze(-1) * ray_directions.unsqueeze(-2)

    return points, t_vals


def volume_render(
    densities: torch.Tensor,
    colors: torch.Tensor,
    t_vals: torch.Tensor,
    ray_directions: torch.Tensor,
    white_background: bool = False,
) -> Dict[str, torch.Tensor]:
    """
    Perform volume rendering given densities and colors.

    Args:
        densities: Volume densities [N, num_samples]
        colors: RGB colors [N, num_samples, 3]
        t_vals: Sample distances along rays [N, num_samples]
        ray_directions: Ray directions [N, 3]
        white_background: Whether to use white background

    Returns:
        Dictionary with rendered outputs
    """
    # Compute distances between samples
    dists = t_vals[..., 1:] - t_vals[..., :-1]
    dists = torch.cat([dists, torch.full_like(dists[..., :1], 1e10)], dim=-1)

    # Apply ray direction norm to distances
    dists = dists * torch.norm(ray_directions, dim=-1, keepdim=True)

    # Convert density to alpha values
    alpha = 1.0 - torch.exp(-torch.relu(densities) * dists)

    # Compute transmittance
    transmittance = torch.cumprod(1.0 - alpha + 1e-10, dim=-1)
    transmittance = torch.cat(
        [torch.ones_like(alpha[..., :1]), alpha],
    )

    # Compute weights
    weights = alpha * transmittance

    # Composite colors
    rgb = torch.sum(weights.unsqueeze(-1) * colors, dim=-2)

    # Add background
    if white_background:
        acc_weights = torch.sum(weights, dim=-1, keepdim=True)
        rgb = rgb + (1.0 - acc_weights)

    # Compute depth
    depth = torch.sum(weights * t_vals, dim=-1)

    # Compute disparity
    disp = 1.0 / torch.max(1e-10 * torch.ones_like(depth), depth / torch.sum(weights, dim=-1))

    # Compute accumulated alpha
    acc = torch.sum(weights, dim=-1)

    return {
        "rgb": rgb,
        "depth": depth,
        "disp": disp,
        "acc": acc,
        "weights": weights,
        "alpha": alpha,
        "transmittance": transmittance,
    }


def compute_ray_aabb_intersection(
    ray_origins: torch.Tensor,
    ray_directions: torch.Tensor,
    aabb_min: torch.Tensor,
    aabb_max: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute ray-AABB intersection for efficient ray marching.

    Args:
        ray_origins: Ray origins [N, 3]
        ray_directions: Ray directions [N, 3]
        aabb_min: AABB minimum coordinates [3]
        aabb_max: AABB maximum coordinates [3]

    Returns:
        tuple of (t_near, t_far) intersection distances
    """
    # Compute intersection with each slab
    inv_dir = 1.0 / (ray_directions + 1e-8)

    t_min = (aabb_min - ray_origins) * inv_dir
    t_max = (aabb_max - ray_origins) * inv_dir

    # Ensure t_min <= t_max
    t_min, t_max = torch.min(t_min, t_max), torch.max(t_min, t_max)

    # Find intersection
    t_near = torch.max(t_min, dim=-1)[0]
    t_far = torch.min(t_max, dim=-1)[0]

    # Check for valid intersection
    valid = t_near <= t_far
    t_near = torch.where(valid, t_near, torch.full_like(t_near, float("inf")))
    t_far = torch.where(valid, t_far, torch.full_like(t_far, -float("inf")))

    return t_near, t_far


def hierarchical_sampling(
    ray_origins: torch.Tensor,
    ray_directions: torch.Tensor,
    t_vals_coarse: torch.Tensor,
    weights_coarse: torch.Tensor,
    num_fine_samples: int,
    perturb: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Perform hierarchical sampling based on coarse weights.

    Args:
        ray_origins: Ray origins [N, 3]
        ray_directions: Ray directions [N, 3]
        t_vals_coarse: Coarse sample distances [N, num_coarse]
        weights_coarse: Coarse weights [N, num_coarse]
        num_fine_samples: Number of fine samples
        perturb: Whether to add perturbation

    Returns:
        tuple of (fine_points, fine_t_vals)
    """
    device = ray_origins.device
    N = ray_origins.shape[0]

    # Compute PDF from weights
    weights = weights_coarse + 1e-5  # Prevent division by zero
    pdf = weights / torch.sum(weights, dim=-1, keepdim=True)
    cdf = torch.cumsum(pdf, dim=-1)
    cdf = torch.cat([torch.zeros_like(cdf[..., :1]), cdf], dim=-1)

    # Sample from CDF
    if perturb:
        u = torch.rand(N, num_fine_samples, device=device)
    else:
        u = torch.linspace(0.0, 1.0, num_fine_samples, device=device)
        u = u.expand(N, num_fine_samples)

    # Invert CDF
    indices = torch.searchsorted(cdf, u, right=True)
    below = torch.clamp(indices - 1, 0, cdf.shape[-1] - 1)
    above = torch.clamp(indices, 0, cdf.shape[-1] - 1)
    indices_g = torch.stack([below, above], dim=-1)  # [N, num_fine, 2]

    # Linear interpolation
    matched_shape = list(indices_g.shape[:-1]) + [cdf.shape[-1]]
    cdf_g = torch.gather(cdf.unsqueeze(-2).expand(matched_shape), -1, indices_g)
    bins_g = torch.gather(t_vals_coarse.unsqueeze(-2).expand(matched_shape), -1, indices_g)

    denom = cdf_g[..., 1] - cdf_g[..., 0]
    denom = torch.where(denom < 1e-5, torch.ones_like(denom), denom)
    t_fine = bins_g[..., 0] + (u - cdf_g[..., 0]) / denom * (bins_g[..., 1] - bins_g[..., 0])

    # Compute fine sample points
    points_fine = ray_origins.unsqueeze(-2) + t_fine.unsqueeze(-1) * ray_directions.unsqueeze(-2)

    return points_fine, t_fine


def compute_optical_depth(densities: torch.Tensor, dists: torch.Tensor) -> torch.Tensor:
    """
    Compute optical depth from densities and distances.

    Args:
        densities: Volume densities [N, num_samples]
        dists: Distances between samples [N, num_samples]

    Returns:
        Optical depth values [N, num_samples]
    """
    return torch.relu(densities) * dists


def alpha_composite(colors: torch.Tensor, alphas: torch.Tensor) -> torch.Tensor:
    """
    Perform alpha compositing of colors.

    Args:
        colors: RGB colors [N, num_samples, 3]
        alphas: Alpha values [N, num_samples]

    Returns:
        Composited RGB colors [N, 3]
    """
    # Compute transmittance
    transmittance = torch.cumprod(1.0 - alphas + 1e-10, dim=-1)
    transmittance = torch.cat(
        [torch.ones_like(colors[..., :1]), colors],
    )

    # Compute weights
    weights = alphas * transmittance

    # Composite colors
    rgb = torch.sum(weights.unsqueeze(-1) * colors, dim=-2)

    return rgb


def compute_expected_depth(t_vals: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    """
    Compute expected depth from sample positions and weights.

    Args:
        t_vals: Sample distances along rays [N, num_samples]
        weights: Rendering weights [N, num_samples]

    Returns:
        Expected depth values [N]
    """
    return torch.sum(weights * t_vals, dim=-1)


def compute_depth_variance(
    t_vals: torch.Tensor,
    weights: torch.Tensor,
    expected_depth: torch.Tensor,
) -> torch.Tensor:
    """
    Compute depth variance for uncertainty estimation.

    Args:
        t_vals: Sample distances along rays [N, num_samples]
        weights: Rendering weights [N, num_samples]
        expected_depth: Expected depth values [N]

    Returns:
        Depth variance values [N]
    """
    depth_diff = t_vals - expected_depth.unsqueeze(-1)
    variance = torch.sum(weights * depth_diff**2, dim=-1)
    return variance


def ray_sphere_intersection(
    ray_origins: torch.Tensor,
    ray_directions: torch.Tensor,
    sphere_center: torch.Tensor,
    sphere_radius: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute ray-sphere intersection.

    Args:
        ray_origins: Ray origins [N, 3]
        ray_directions: Ray directions [N, 3]
        sphere_center: Sphere center [3]
        sphere_radius: Sphere radius

    Returns:
        tuple of (t_near, t_far) intersection distances
    """
    # Vector from ray origin to sphere center
    oc = ray_origins - sphere_center

    # Quadratic equation coefficients
    a = torch.sum(ray_directions**2, dim=-1)
    b = 2.0 * torch.sum(oc * ray_directions, dim=-1)
    c = torch.sum(oc**2, dim=-1) - sphere_radius**2

    # Discriminant
    discriminant = b**2 - 4 * a * c

    # Check for intersection
    valid = discriminant >= 0
    sqrt_discriminant = torch.sqrt(torch.clamp(discriminant, min=0))

    # Compute intersection distances
    t_near = (-b - sqrt_discriminant) / (2 * a)
    t_far = (-b + sqrt_discriminant) / (2 * a)

    # Handle invalid intersections
    t_near = torch.where(valid, t_near, torch.full_like(t_near, float("inf")))
    t_far = torch.where(valid, t_far, torch.full_like(t_far, -float("inf")))

    return t_near, t_far


def compute_ray_bundle_intersections(
    ray_origins: torch.Tensor,
    ray_directions: torch.Tensor,
    voxel_centers: torch.Tensor,
    voxel_size: float,
) -> torch.Tensor:
    """
    Compute intersections between ray bundle and voxel grid.

    Args:
        ray_origins: Ray origins [N, 3]
        ray_directions: Ray directions [N, 3]
        voxel_centers: Voxel center coordinates [V, 3]
        voxel_size: Size of each voxel

    Returns:
        Intersection mask [N, V]
    """
    N = ray_origins.shape[0]
    V = voxel_centers.shape[0]

    # Expand dimensions for broadcasting
    origins = ray_origins.unsqueeze(1)  # [N, 1, 3]
    directions = ray_directions.unsqueeze(1)  # [N, 1, 3]
    centers = voxel_centers.unsqueeze(0)  # [1, V, 3]

    # Compute AABB bounds for each voxel
    half_size = voxel_size / 2
    aabb_min = centers - half_size
    aabb_max = centers + half_size

    # Ray-AABB intersection for all ray-voxel pairs
    inv_dir = 1.0 / (directions + 1e-8)

    t_min = (aabb_min - origins) * inv_dir
    t_max = (aabb_max - origins) * inv_dir

    t_min, t_max = torch.min(t_min, t_max), torch.max(t_min, t_max)

    t_near = torch.max(t_min, dim=-1)[0]  # [N, V]
    t_far = torch.min(t_max, dim=-1)[0]  # [N, V]

    # Check for valid intersection
    intersects = (t_near <= t_far) & (t_far > 0)

    return intersects
