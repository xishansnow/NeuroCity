from typing import Any, Optional, Union
"""
Grid utilities for Grid-NeRF.

This module provides utility functions for grid operations, spatial computations, and grid management specific to Grid-NeRF.
"""

import torch
import torch.nn.functional as F
import numpy as np
import math

def compute_grid_bounds(
    points: torch.Tensor,
    margin: float = 0.1
) -> tuple[float, float, float, float, float, float]:
    """
    Compute scene bounds from point cloud.
    
    Args:
        points: Point cloud tensor [N, 3]
        margin: Additional margin to add to bounds
        
    Returns:
        tuple of (x_min, y_min, z_min, x_max, y_max, z_max)
    """
    min_bounds = torch.min(points, dim=0)[0]
    max_bounds = torch.max(points, dim=0)[0]
    
    # Add margin
    scene_size = max_bounds - min_bounds
    margin_size = scene_size * margin
    
    min_bounds -= margin_size
    max_bounds += margin_size
    
    return (*min_bounds.tolist(), *max_bounds.tolist())

def world_to_grid_coords(
    world_coords: torch.Tensor,
    scene_bounds: torch.Tensor,
    grid_resolution: int
) -> torch.Tensor:
    """
    Convert world coordinates to grid coordinates.
    
    Args:
        world_coords: World coordinates [N, 3]
        scene_bounds: Scene bounds [6] (x_min, y_min, z_min, x_max, y_max, z_max)
        grid_resolution: Grid resolution
        
    Returns:
        Grid coordinates [N, 3] in range [0, grid_resolution-1]
    """
    scene_min = scene_bounds[:3]
    scene_size = scene_bounds[3:] - scene_bounds[:3]
    
    # Normalize to [0, 1]
    normalized = (world_coords - scene_min) / scene_size
    
    # Scale to grid resolution
    grid_coords = normalized * (grid_resolution - 1)
    
    return grid_coords

def grid_to_world_coords(
    grid_coords: torch.Tensor,
    scene_bounds: torch.Tensor,
    grid_resolution: int
) -> torch.Tensor:
    """
    Convert grid coordinates to world coordinates.
    
    Args:
        grid_coords: Grid coordinates [N, 3]
        scene_bounds: Scene bounds [6]
        grid_resolution: Grid resolution
        
    Returns:
        World coordinates [N, 3]
    """
    scene_min = scene_bounds[:3]
    scene_size = scene_bounds[3:] - scene_bounds[:3]
    
    # Normalize to [0, 1]
    normalized = grid_coords / (grid_resolution - 1)
    
    # Scale to world coordinates
    world_coords = normalized * scene_size + scene_min
    
    return world_coords

def trilinear_interpolation(grid_features: torch.Tensor, grid_coords: torch.Tensor) -> torch.Tensor:
    """
    Perform trilinear interpolation of grid features.
    
    Args:
        grid_features: Grid features [D, H, W, C]
        grid_coords: Grid coordinates [N, 3] in range [0, resolution-1]
        
    Returns:
        Interpolated features [N, C]
    """
    # Convert to normalized coordinates for grid_sample
    D, H, W, C = grid_features.shape
    
    # Normalize coordinates to [-1, 1] for grid_sample
    normalized_coords = grid_coords.clone()
    normalized_coords[:, 0] = (normalized_coords[:, 0] / (D - 1)) * 2 - 1
    normalized_coords[:, 1] = (normalized_coords[:, 1] / (H - 1)) * 2 - 1
    normalized_coords[:, 2] = (normalized_coords[:, 2] / (W - 1)) * 2 - 1
    
    # Reshape for grid_sample: [1, 1, 1, N, 3]
    grid_coords_reshaped = normalized_coords.unsqueeze(0).unsqueeze(0).unsqueeze(0)
    
    # Reshape grid features: [1, C, D, H, W]
    grid_features_reshaped = grid_features.permute(3, 0, 1, 2).unsqueeze(0)
    
    # Perform trilinear interpolation
    interpolated = F.grid_sample(
        grid_features_reshaped, grid_coords_reshaped, mode='bilinear', padding_mode='border', align_corners=True
    )
    
    # Reshape output: [N, C]
    interpolated = interpolated.squeeze(0).squeeze(1).squeeze(1).transpose(0, 1)
    
    return interpolated

def create_voxel_grid(
    points: torch.Tensor,
    features: Optional[torch.Tensor] = None,
    grid_resolution: int = 64,
    scene_bounds: Optional[tuple[float, float, float, float, float, float]] = None
) -> torch.Tensor:
    """
    Create voxel grid from point cloud.
    
    Args:
        points: Point cloud [N, 3]
        features: Point features [N, F] (optional)
        grid_resolution: Grid resolution
        scene_bounds: Scene bounds (optional, computed if None)
        
    Returns:
        Voxel grid [D, H, W, F] where F is feature dimension
    """
    if scene_bounds is None:
        scene_bounds = compute_grid_bounds(points)
    
    scene_bounds_tensor = torch.tensor(scene_bounds, device=points.device)
    
    # Convert points to grid coordinates
    grid_coords = world_to_grid_coords(points, scene_bounds_tensor, grid_resolution)
    
    # Initialize voxel grid
    if features is not None:
        feature_dim = features.shape[1]
    else:
        feature_dim = 1
        features = torch.ones(len(points), 1, device=points.device)
    
    voxel_grid = torch.zeros(
        grid_resolution,
        grid_resolution,
        grid_resolution,
        feature_dim,
        device=points.device,
    )
    
    # Round coordinates to nearest voxel
    voxel_indices = torch.round(grid_coords).long()
    
    # Clamp to valid range
    voxel_indices = torch.clamp(voxel_indices, 0, grid_resolution - 1)
    
    # Accumulate features in voxels
    for i in range(len(points)):
        x, y, z = voxel_indices[i]
        voxel_grid[x, y, z] += features[i]
    
    return voxel_grid

def dilate_grid(grid: torch.Tensor, kernel_size: int = 3) -> torch.Tensor:
    """
    Dilate a 3D grid using morphological dilation.
    
    Args:
        grid: Input grid [D, H, W, C]
        kernel_size: Dilation kernel size
        
    Returns:
        Dilated grid [D, H, W, C]
    """
    D, H, W, C = grid.shape
    
    # Create dilation kernel
    kernel = torch.ones(1, 1, kernel_size, kernel_size, kernel_size, device=grid.device)
    
    # Process each channel separately
    dilated_channels = []
    
    for c in range(C):
        channel = grid[:, :, :, c].unsqueeze(0).unsqueeze(0)  # [1, 1, D, H, W]
        
        # Apply 3D dilation
        dilated_channel = F.conv3d(
            channel, kernel, padding=kernel_size // 2
        )
        
        dilated_channels.append(dilated_channel.squeeze(0).squeeze(0))
    
    # Stack channels
    dilated_grid = torch.stack(dilated_channels, dim=-1)
    
    return dilated_grid

def prune_grid(grid: torch.Tensor, threshold: float = 0.01) -> torch.Tensor:
    """
    Prune grid cells below threshold.
    
    Args:
        grid: Input grid [D, H, W, C]
        threshold: Pruning threshold
        
    Returns:
        Pruned grid [D, H, W, C]
    """
    # Compute feature magnitude
    magnitude = torch.norm(grid, dim=-1, keepdim=True)
    
    # Create mask
    mask = magnitude > threshold
    
    # Apply mask
    pruned_grid = grid * mask
    
    return pruned_grid

def compute_grid_occupancy(grid: torch.Tensor, threshold: float = 0.01) -> torch.Tensor:
    """
    Compute occupancy mask for grid.
    
    Args:
        grid: Input grid [D, H, W, C]
        threshold: Occupancy threshold
        
    Returns:
        Binary occupancy mask [D, H, W]
    """
    # Compute feature magnitude
    magnitude = torch.norm(grid, dim=-1)
    
    # Create binary mask
    occupancy = magnitude > threshold
    
    return occupancy

def adaptive_grid_subdivision(
    grid: torch.Tensor,
    occupancy_threshold: float = 0.01,
    subdivision_threshold: float = 0.8
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Adaptively subdivide grid based on occupancy.
    
    Args:
        grid: Input grid [D, H, W, C]
        occupancy_threshold: Threshold for occupancy
        subdivision_threshold: Threshold for subdivision
        
    Returns:
        tuple of (subdivided grid [D*2, H*2, W*2, C], subdivision mask [D, H, W])
    """
    # Compute occupancy
    occupancy = compute_grid_occupancy(grid, occupancy_threshold)
    
    # Compute subdivision mask
    kernel_size = 3
    kernel = torch.ones(1, 1, kernel_size, kernel_size, kernel_size, device=grid.device)
    occupancy_count = F.conv3d(
        occupancy.float().unsqueeze(0).unsqueeze(0),
        kernel,
        padding=kernel_size // 2
    ).squeeze(0).squeeze(0)
    
    subdivision_mask = occupancy_count > (kernel_size ** 3 * subdivision_threshold)
    
    # Create subdivided grid
    D, H, W, C = grid.shape
    subdivided_grid = F.interpolate(
        grid.permute(3, 0, 1, 2).unsqueeze(0),
        size=(D*2, H*2, W*2),
        mode='trilinear',
        align_corners=True
    ).squeeze(0).permute(1, 2, 3, 0)
    
    return subdivided_grid, subdivision_mask

def compute_grid_gradient(grid: torch.Tensor) -> torch.Tensor:
    """
    Compute spatial gradients of grid.
    
    Args:
        grid: Input grid [D, H, W, C]
        
    Returns:
        Grid gradients [D, H, W, C, 3]
    """
    # Compute gradients along each dimension
    dx = torch.zeros_like(grid)
    dy = torch.zeros_like(grid)
    dz = torch.zeros_like(grid)
    
    dx[1:, :, :] = grid[1:, :, :] - grid[:-1, :, :]
    dy[:, 1:, :] = grid[:, 1:, :] - grid[:, :-1, :]
    dz[:, :, 1:] = grid[:, :, 1:] - grid[:, :, :-1]
    
    # Stack gradients
    gradients = torch.stack([dx, dy, dz], dim=-1)
    
    return gradients

def smooth_grid(grid: torch.Tensor, sigma: float = 1.0) -> torch.Tensor:
    """
    Apply Gaussian smoothing to grid.
    
    Args:
        grid: Input grid [D, H, W, C]
        sigma: Gaussian standard deviation
        
    Returns:
        Smoothed grid [D, H, W, C]
    """
    # Create Gaussian kernel
    kernel_size = int(2 * round(3 * sigma) + 1)
    kernel = torch.zeros(1, 1, kernel_size, kernel_size, kernel_size, device=grid.device)
    
    # Fill kernel with Gaussian values
    center = kernel_size // 2
    for i in range(kernel_size):
        for j in range(kernel_size):
            for k in range(kernel_size):
                x = i - center
                y = j - center
                z = k - center
                kernel[0, 0, i, j, k] = torch.exp(-(x*x + y*y + z*z) / (2*sigma*sigma))
    
    # Normalize kernel
    kernel = kernel / kernel.sum()
    
    # Process each channel separately
    smoothed_channels = []
    for c in range(grid.shape[-1]):
        channel = grid[:, :, :, c].unsqueeze(0).unsqueeze(0)
        smoothed_channel = F.conv3d(
            channel,
            kernel,
            padding=kernel_size // 2
        )
        smoothed_channels.append(smoothed_channel.squeeze(0).squeeze(0))
    
    # Stack channels
    smoothed_grid = torch.stack(smoothed_channels, dim=-1)
    
    return smoothed_grid

def compute_grid_statistics(grid: torch.Tensor) -> dict[str, Union[float, int]]:
    """
    Compute statistics of grid values.
    
    Args:
        grid: Input grid [D, H, W, C]
        
    Returns:
        Dictionary of statistics
    """
    stats = {
        'min': float(grid.min().item()),
        'max': float(grid.max().item()),
        'mean': float(grid.mean().item()),
        'std': float(grid.std().item()),
        'num_nonzero': int((grid != 0).sum().item()),
        'total_cells': int(grid.numel()),
        'sparsity': float((grid == 0).sum().item() / grid.numel())
    }
    
    return stats 