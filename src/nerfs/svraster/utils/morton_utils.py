"""
from __future__ import annotations

Morton code utilities for SVRaster.

This module provides functions for encoding and decoding 3D coordinates
using Morton codes (Z-order curves) for spatial ordering.
"""

import torch
import numpy as np
from typing import Tuple


def morton_encode_3d(coords_or_x, y=None, z=None) -> int | np.ndarray:
    """
    Encode 3D coordinates into Morton code.

    改进的 Morton 编码实现，支持更高的分辨率：
    - 每个坐标分量使用 21 位，总共支持 2097151³ 个位置
    - 完全满足 SVRaster 的 65536³ 分辨率需求

    Args:
        coords_or_x: Either 3D coordinates array of shape (N, 3) or x coordinate
        y: y coordinate (if coords_or_x is x)
        z: z coordinate (if coords_or_x is x)

    Returns:
        Morton code(s) as integer or array
    """

    def part1by2(n):
        """Separate bits by inserting two zeros between each bit."""
        # 支持更大的坐标值，使用更高效的位交错
        # 将 21 位输入扩展为 63 位输出（3 * 21 = 63）
        n &= 0x1FFFFF  # 21 位掩码，支持 0-2097151
        n = (n ^ (n << 32)) & 0x1F00000000FFFF
        n = (n ^ (n << 16)) & 0x1F0000FF0000FF
        n = (n ^ (n << 8)) & 0x100F00F00F00F00F
        n = (n ^ (n << 4)) & 0x10C30C30C30C30C3
        n = (n ^ (n << 2)) & 0x1249249249249249
        return n

    def encode_single(x, y, z):
        """Encode single coordinate triple."""
        # Validate inputs
        if x < 0 or y < 0 or z < 0:
            raise ValueError("Morton encoding requires non-negative coordinates")

        # 确保输入在合理范围内
        x = max(0, min(x, 0x1FFFFF))
        y = max(0, min(y, 0x1FFFFF))
        z = max(0, min(z, 0x1FFFFF))

        return (part1by2(z) << 2) + (part1by2(y) << 1) + part1by2(x)

    # Handle different input formats
    if y is None and z is None:
        # coords_or_x is an array of coordinates
        coords = np.asarray(coords_or_x)
        if coords.ndim == 1 and len(coords) == 3:
            # Single coordinate triple
            return encode_single(int(coords[0]), int(coords[1]), int(coords[2]))
        elif coords.ndim == 2 and coords.shape[1] == 3:
            # Multiple coordinate triples
            results = []
            for i in range(coords.shape[0]):
                results.append(
                    encode_single(int(coords[i, 0]), int(coords[i, 1]), int(coords[i, 2]))
                )
            return np.array(results)
        else:
            raise ValueError("Invalid coordinate array shape")
    else:
        # Individual x, y, z coordinates
        if y is None or z is None:
            raise ValueError("y and z coordinates must be provided")
        return encode_single(int(coords_or_x), int(y), int(z))


def morton_decode_3d(morton_codes) -> tuple[int, int, int] | np.ndarray:
    """
    Decode Morton code(s) back to 3D coordinates.

    Args:
        morton_codes: Morton code as integer or array of integers

    Returns:
        tuple of (x, y, z) coordinates or array of coordinates
    """

    def compact1by2(n):
        """Compact bits by removing two zeros between each bit."""
        n &= 0x1249249249249249
        n = (n ^ (n >> 2)) & 0x10C30C30C30C30C3
        n = (n ^ (n >> 4)) & 0x100F00F00F00F00F
        n = (n ^ (n >> 8)) & 0x1F0000FF0000FF
        n = (n ^ (n >> 16)) & 0x1F00000000FFFF
        n = (n ^ (n >> 32)) & 0x1FFFFF
        return n

    def decode_single(morton_code):
        """Decode single morton code."""
        x = compact1by2(morton_code)
        y = compact1by2(morton_code >> 1)
        z = compact1by2(morton_code >> 2)
        return x, y, z

    # Handle different input formats
    morton_codes = np.asarray(morton_codes)
    if morton_codes.ndim == 0:
        # Single morton code
        return decode_single(int(morton_codes))
    else:
        # Array of morton codes
        results = []
        for morton_code in morton_codes:
            x, y, z = decode_single(int(morton_code))
            results.append([x, y, z])
        return np.array(results)


def morton_encode_batch(coords: torch.Tensor) -> torch.Tensor:
    """
    Encode batch of 3D coordinates into Morton codes.

    使用向量化操作进行批量 Morton 码计算，提高性能。

    Args:
        coords: Tensor of shape [N, 3] with integer coordinates

    Returns:
        Tensor of Morton codes [N]
    """
    coords = coords.long()
    x, y, z = coords[:, 0], coords[:, 1], coords[:, 2]

    # 确保坐标在合理范围内
    max_coord = 0x1FFFFF  # 21 位最大值
    x = torch.clamp(x, 0, max_coord)
    y = torch.clamp(y, 0, max_coord)
    z = torch.clamp(z, 0, max_coord)

    # 向量化的位交错函数
    def part1by2_vectorized(n):
        n = n & 0x1FFFFF
        n = (n ^ (n << 32)) & 0x1F00000000FFFF
        n = (n ^ (n << 16)) & 0x1F0000FF0000FF
        n = (n ^ (n << 8)) & 0x100F00F00F00F00F
        n = (n ^ (n << 4)) & 0x10C30C30C30C30C3
        n = (n ^ (n << 2)) & 0x1249249249249249
        return n

    # 计算 Morton 码
    morton_x = part1by2_vectorized(x)
    morton_y = part1by2_vectorized(y)
    morton_z = part1by2_vectorized(z)

    return (morton_z << 2) + (morton_y << 1) + morton_x


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
        x, y, z = morton_decode_3d(int(morton_codes[i].item()))
        coords[i] = torch.tensor([x, y, z])

    return coords


def compute_morton_order(
    positions: torch.Tensor,
    scene_bounds: tuple[float, float, float, float, float, float],
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

    # Compute Morton codes using vectorized implementation
    return morton_encode_batch(grid_coords)


def sort_by_morton_order(
    positions: torch.Tensor,
    scene_bounds: tuple[float, float, float, float, float, float],
    grid_resolution: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Sort positions by Morton order.

    Args:
        positions: Tensor of 3D positions [N, 3]
        scene_bounds: Scene bounds
        grid_resolution: Grid resolution for discretization

    Returns:
        tuple of (sorted_positions, sort_indices)
    """
    morton_codes = compute_morton_order(positions, scene_bounds, grid_resolution)
    sort_indices = torch.argsort(morton_codes)
    sorted_positions = positions[sort_indices]

    return sorted_positions, sort_indices
