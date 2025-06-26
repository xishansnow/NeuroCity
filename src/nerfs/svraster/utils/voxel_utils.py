"""
Voxel utilities for SVRaster.
"""

import torch
import numpy as np
from typing import Optional, Tuple


def voxel_pruning(
    voxel_densities: torch.Tensor,
    voxel_positions: torch.Tensor,
    voxel_sizes: torch.Tensor,
    threshold: float = 0.001,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Prune voxels with low density.
    
    Args:
        voxel_densities: Voxel density values [N]
        voxel_positions: Voxel positions [N, 3]
        voxel_sizes: Voxel sizes [N]
        threshold: Pruning threshold
        
    Returns:
        Tuple of (pruned_densities, pruned_positions, pruned_sizes)
    """
    # Apply activation (assume exp activation)
    densities = torch.exp(voxel_densities)
    
    # Create keep mask
    keep_mask = densities > threshold
    
    # Apply pruning
    pruned_densities = voxel_densities[keep_mask]
    pruned_positions = voxel_positions[keep_mask]
    pruned_sizes = voxel_sizes[keep_mask]
    
    return pruned_densities, pruned_positions, pruned_sizes


def compute_voxel_bounds(
    voxel_positions: torch.Tensor,
    voxel_sizes: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Compute bounding boxes for voxels.
    
    Args:
        voxel_positions: Voxel center positions [N, 3]
        voxel_sizes: Voxel sizes [N]
        
    Returns:
        Tuple of (box_min, box_max) each [N, 3]
    """
    half_sizes = voxel_sizes.unsqueeze(-1) * 0.5
    box_min = voxel_positions - half_sizes
    box_max = voxel_positions + half_sizes
    
    return box_min, box_max


def voxel_to_world_coords(
    voxel_coords: torch.Tensor,
    scene_bounds: tuple[float,
    float,
    float,
    float,
    float,
    float],
    grid_resolution: int,
) -> torch.Tensor:
    """
    Convert voxel coordinates to world coordinates.
    
    Args:
        voxel_coords: Voxel coordinates [N, 3]
        scene_bounds: Scene bounds (min_x, min_y, min_z, max_x, max_y, max_z)
        grid_resolution: Grid resolution
        
    Returns:
        World coordinates [N, 3]
    """
    scene_min = torch.tensor(scene_bounds[:3], device=voxel_coords.device)
    scene_max = torch.tensor(scene_bounds[3:], device=voxel_coords.device)
    scene_size = scene_max - scene_min
    
    # Normalize voxel coordinates to [0, 1]
    normalized_coords = voxel_coords.float() / grid_resolution
    
    # Convert to world coordinates
    world_coords = normalized_coords * scene_size + scene_min
    
    return world_coords


def world_to_voxel_coords(
    world_coords: torch.Tensor,
    scene_bounds: tuple[float,
    float,
    float,
    float,
    float,
    float],
    grid_resolution: int,
) -> torch.Tensor:
    """
    Convert world coordinates to voxel coordinates.
    
    Args:
        world_coords: World coordinates [N, 3]
        scene_bounds: Scene bounds (min_x, min_y, min_z, max_x, max_y, max_z)
        grid_resolution: Grid resolution
        
    Returns:
        Voxel coordinates [N, 3]
    """
    scene_min = torch.tensor(scene_bounds[:3], device=world_coords.device)
    scene_max = torch.tensor(scene_bounds[3:], device=world_coords.device)
    scene_size = scene_max - scene_min
    
    # Normalize to [0, 1]
    normalized_coords = (world_coords - scene_min) / scene_size
    
    # Convert to voxel coordinates
    voxel_coords = (normalized_coords * grid_resolution).long()
    voxel_coords = torch.clamp(voxel_coords, 0, grid_resolution - 1)
    
    return voxel_coords 