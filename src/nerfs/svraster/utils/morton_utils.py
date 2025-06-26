"""
Morton code utilities for SVRaster.

This module provides functions for encoding and decoding 3D coordinates
using Morton codes (Z-order curves) for spatial ordering.
"""

import torch
import numpy as np
from typing import Tuple

def morton_encode_3d(x: int, y: int, z: int) -> int:
    """
    Encode 3D coordinates into Morton code.
    
    Args:
        x, y, z: 3D coordinates
        
    Returns:
        Morton code as integer
    """
    def part1by2(n):
        """Separate bits by inserting two zeros between each bit."""
        n &= 0x000003ff  # Mask to 10 bits
        n = (n ^ (n << 16)) & 0xff0000ff
        n = (n ^ (n << 8)) & 0x0300f00f
        n = (n ^ (n << 4)) & 0x030c30c3
        n = (n ^ (n << 2)) & 0x09249249
        return n
    
    return (part1by2(z) << 2) + (part1by2(y) << 1) + part1by2(x)


def morton_decode_3d(morton_code: int) -> tuple[int, int, int]:
    """
    Decode Morton code back to 3D coordinates.
    
    Args:
        morton_code: Morton code as integer
        
    Returns:
        Tuple of (x, y, z) coordinates
    """
    def compact1by2(n):
        """Compact bits by removing two zeros between each bit."""
        n &= 0x09249249
        n = (n ^ (n >> 2)) & 0x030c30c3
        n = (n ^ (n >> 4)) & 0x0300f00f
        n = (n ^ (n >> 8)) & 0x0300f00f
        n = (n ^ (n >> 16)) & 0x000003ff
        return n
    
    x = compact1by2(morton_code)
    y = compact1by2(morton_code >> 1)
    z = compact1by2(morton_code >> 2)
    
    return x, y, z


def morton_encode_batch(coords: torch.Tensor) -> torch.Tensor:
    """
    Encode batch of 3D coordinates into Morton codes.
    
    Args:
        coords: Tensor of shape [N, 3] with integer coordinates
        
    Returns:
        Tensor of Morton codes [N]
    """
    coords = coords.long()
    x, y, z = coords[:, 0], coords[:, 1], coords[:, 2]
    
    # Vectorized Morton encoding
    morton_codes = torch.zeros_like(x, dtype=torch.long)
    
    for i in range(coords.shape[0]):
        morton_codes[i] = morton_encode_3d(x[i].item(), y[i].item(), z[i].item())
    
    return morton_codes


def morton_decode_batch(morton_codes: torch.Tensor) -> torch.Tensor:
    """
    Decode batch of Morton codes back to 3D coordinates.
    
    Args:
        morton_codes: Tensor of Morton codes [N]
        
    Returns:
        Tensor of coordinates [N, 3]
    """
    coords = torch.zeros(morton_codes.shape[0], 3, dtype=torch.long, device=morton_codes.device)
    
    for i in range(morton_codes.shape[0]):
        x, y, z = morton_decode_3d(morton_codes[i].item())
        coords[i] = torch.tensor([x, y, z])
    
    return coords


def compute_morton_order(
    positions: torch.Tensor,
    scene_bounds: tuple[float,
    float,
    float,
    float,
    float,
    float],
    grid_resolution: int,
) -> torch.Tensor:
    """
    Compute Morton order for a set of 3D positions.
    
    Args:
        positions: Tensor of 3D positions [N, 3]
        scene_bounds: Scene bounds (min_x, min_y, min_z, max_x, max_y, max_z)
        grid_resolution: Grid resolution for discretization
        
    Returns:
        Morton codes for the positions [N]
    """
    # Normalize positions to [0, 1]
    scene_min = torch.tensor(scene_bounds[:3], device=positions.device)
    scene_max = torch.tensor(scene_bounds[3:], device=positions.device)
    scene_size = scene_max - scene_min
    
    normalized_pos = (positions - scene_min) / scene_size
    
    # Discretize to grid coordinates
    grid_coords = (normalized_pos * grid_resolution).long()
    grid_coords = torch.clamp(grid_coords, 0, grid_resolution - 1)
    
    # Compute Morton codes
    return morton_encode_batch(grid_coords)


def sort_by_morton_order(
    positions: torch.Tensor,
    scene_bounds: tuple[float,
    float,
    float,
    float,
    float,
    float],
    grid_resolution: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Sort positions by Morton order.
    
    Args:
        positions: Tensor of 3D positions [N, 3]
        scene_bounds: Scene bounds
        grid_resolution: Grid resolution for discretization
        
    Returns:
        Tuple of (sorted_positions, sort_indices)
    """
    morton_codes = compute_morton_order(positions, scene_bounds, grid_resolution)
    sort_indices = torch.argsort(morton_codes)
    sorted_positions = positions[sort_indices]
    
    return sorted_positions, sort_indices 