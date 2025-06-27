"""
Utility functions for Instant NGP.

This file contains utility functions for the Instant NGP model.

The functions are used to:
- Contract infinite space to unit sphere
- Uncontract from unit sphere to infinite space
- Compute total variation loss on hash grids
"""

import torch
import numpy as np

def contract_to_unisphere(x: torch.Tensor) -> torch.Tensor:
    """Contract infinite space to unit sphere."""
    mag = torch.norm(x, dim=-1, keepdim=True)
    mask = mag > 1
    x_contracted = torch.where(
        mask, (2 - 1/mag) * (x / mag), x
    )
    return x_contracted

def uncontract_from_unisphere(x: torch.Tensor) -> torch.Tensor:
    """Uncontract from unit sphere to infinite space."""
    mag = torch.norm(x, dim=-1, keepdim=True)
    mask = mag > 1
    x_uncontracted = torch.where(
        mask, x / (2 - mag), x
    )
    return x_uncontracted

def morton_encode_3d(x: torch.Tensor, y: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
    """3D Morton encoding for spatial hashing."""
    def expand_bits(v):
        v = (v * 0x00010001) & 0xFF0000FF
        v = (v * 0x00000101) & 0x0F00F00F
        v = (v * 0x00000011) & 0xC30C30C3
        v = (v * 0x00000005) & 0x49249249
        return v
    
    # Assume coordinates are integers
    x_expanded = expand_bits(x.long())
    y_expanded = expand_bits(y.long())
    z_expanded = expand_bits(z.long())
    
    return x_expanded | (y_expanded << 1) | (z_expanded << 2)

def compute_tv_loss(hash_grids: list, grid_coords: torch.Tensor) -> torch.Tensor:
    """Compute total variation loss on hash grids."""
    tv_loss = 0.0
    
    for i, grid in enumerate(hash_grids):
        # Get neighboring coordinates
        neighbors = []
        for dim in range(3):
            offset = torch.zeros_like(grid_coords)
            offset[:, dim] = 1
            neighbors.append(grid_coords + offset)
        
        # Compute differences
        for neighbor_coords in neighbors:
            # Clamp to valid range
            neighbor_coords = torch.clamp(neighbor_coords, 0, grid.shape[0]-1)
            
            # Get features
            current_features = grid[grid_coords[:, 0], grid_coords[:, 1], grid_coords[:, 2]]
            neighbor_features = grid[neighbor_coords[:, 0], neighbor_coords[:, 1], neighbor_coords[:, 2]]
            
            # Compute TV loss
            tv_loss += torch.mean((current_features - neighbor_features) ** 2)
    
    return tv_loss

def adaptive_sampling(
    density: torch.Tensor,
    z_vals: torch.Tensor,
    rays_d: torch.Tensor,
    num_importance_samples: int = 64,
):
    """Adaptive sampling based on density."""
    # Compute weights
    dists = z_vals[..., 1:] - z_vals[..., :-1]
    dists = torch.cat([
        dists,
        torch.tensor([1e10], device=dists.device).expand(dists[..., :1].shape)
    ], -1)
    dists = dists * torch.norm(rays_d[..., None, :], dim=-1)
    
    alpha = 1. - torch.exp(-density.squeeze(-1) * dists)
    weights = alpha * torch.cumprod(
        torch.cat,
    )
    
    # Sample based on weights
    weights = weights + 1e-5  # Prevent nans
    pdf = weights / torch.sum(weights, -1, keepdim=True)
    cdf = torch.cumsum(pdf, -1)
    cdf = torch.cat([torch.zeros_like(cdf[..., :1]), cdf], -1)
    
    # Take uniform samples
    u = torch.rand(list(cdf.shape[:-1]) + [num_importance_samples], device=weights.device)
    
    # Invert CDF
    inds = torch.searchsorted(cdf, u, right=True)
    below = torch.max(torch.zeros_like(inds-1), inds-1)
    above = torch.min((cdf.shape[-1]-1) * torch.ones_like(inds), inds)
    inds_g = torch.stack([below, above], -1)
    
    matched_shape = [inds_g.shape[0], inds_g.shape[1], cdf.shape[-1]]
    cdf_g = torch.gather(cdf.unsqueeze(1).expand(matched_shape), 2, inds_g)
    bins_g = torch.gather(z_vals.unsqueeze(1).expand(matched_shape), 2, inds_g)
    
    denom = (cdf_g[..., 1] - cdf_g[..., 0])
    denom = torch.where(denom < 1e-5, torch.ones_like(denom), denom)
    t = (u - cdf_g[..., 0]) / denom
    samples = bins_g[..., 0] + t * (bins_g[..., 1] - bins_g[..., 0])
    
    return samples

def estimate_normals(model, positions: torch.Tensor, epsilon: float = 1e-3) -> torch.Tensor:
    """Estimate surface normals using finite differences."""
    positions.requires_grad_(True)
    
    # Get density
    density = model.get_density(positions)
    
    # Compute gradients
    grad_outputs = torch.ones_like(density)
    gradients = torch.autograd.grad(
        outputs=density, inputs=positions, grad_outputs=grad_outputs, create_graph=True, retain_graph=True, only_inputs=True
    )[0]
    
    # Normalize to get normals
    normals = torch.nn.functional.normalize(gradients, dim=-1)
    
    return normals

def compute_hash_grid_size(resolution: int, max_entries: int) -> int:
    """Compute optimal hash grid size."""
    full_size = resolution ** 3
    return min(full_size, max_entries)

def to8b(x: np.ndarray) -> np.ndarray:
    """Convert to 8-bit image."""
    return (255 * np.clip(x, 0, 1)).astype(np.uint8)

def create_spherical_poses(radius: float = 3.0, n_poses: int = 40) -> np.ndarray:
    """Create spherical camera poses."""
    poses = []
    
    for i in range(n_poses):
        theta = 2 * np.pi * i / n_poses
        phi = 0  # Elevation angle
        
        # Spherical to cartesian
        x = radius * np.cos(phi) * np.cos(theta)
        y = radius * np.cos(phi) * np.sin(theta)
        z = radius * np.sin(phi)
        
        # Look at origin
        look_at = np.array([0, 0, 0])
        up = np.array([0, 0, 1])
        
        forward = look_at - np.array([x, y, z])
        forward = forward / np.linalg.norm(forward)
        
        right = np.cross(forward, up)
        right = right / np.linalg.norm(right)
        
        up = np.cross(right, forward)
        
        pose = np.eye(4)
        pose[:3, 0] = right
        pose[:3, 1] = up
        pose[:3, 2] = -forward  
        pose[:3, 3] = [x, y, z]
        
        poses.append(pose)
    
    return np.array(poses)

def compute_psnr(pred: torch.Tensor, target: torch.Tensor) -> float:
    """Compute PSNR between predictions and targets."""
    mse = torch.mean((pred - target) ** 2)
    psnr = -10. * torch.log10(mse + 1e-8)
    return psnr.item()

def linear_to_srgb(linear: torch.Tensor) -> torch.Tensor:
    """Convert linear RGB to sRGB."""
    return torch.where(
        linear <= 0.0031308, 12.92 * linear, 1.055 * torch.pow(linear, 1.0/2.4) - 0.055
    )

def srgb_to_linear(srgb: torch.Tensor) -> torch.Tensor:
    """Convert sRGB to linear RGB."""
    return torch.where(
        srgb <= 0.04045, srgb / 12.92, torch.pow((srgb + 0.055) / 1.055, 2.4)
    )
