"""Geometry utilities for Instant NGP."""

import torch
import numpy as np


def estimate_normals(positions: torch.Tensor, density_fn, eps: float = 1e-3) -> torch.Tensor:
    """Estimate normals using finite differences."""
    device = positions.device
    normals = torch.zeros_like(positions)
    
    for i in range(3):
        offset = torch.zeros_like(positions)
        offset[:, i] = eps
        
        pos_plus = positions + offset
        pos_minus = positions - offset
        
        density_plus = density_fn(pos_plus)
        density_minus = density_fn(pos_minus)
        
        normals[:, i] = (density_plus - density_minus).squeeze() / (2 * eps)
    
    # Normalize
    normals = normals / (torch.norm(normals, dim=-1, keepdim=True) + 1e-8)
    
    return normals


def compute_surface_points(positions: torch.Tensor, density: torch.Tensor, threshold: float = 0.5) -> torch.Tensor:
    """Extract surface points based on density threshold."""
    surface_mask = density.squeeze() > threshold
    return positions[surface_mask]


def mesh_extraction(positions: torch.Tensor, density: torch.Tensor, threshold: float = 0.5):
    """Simple mesh extraction (placeholder)."""
    surface_points = compute_surface_points(positions, density, threshold)
    return surface_points, None  # vertices, faces 