"""
Plenoxels Core Module

This module implements the core components of Plenoxels:
- Sparse voxel grids for density and color representation
- Spherical harmonics for view-dependent appearance
- Trilinear interpolation for smooth sampling
- Volume rendering without neural networks

Based on the paper: "Plenoxels: Radiance Fields without Neural Networks"
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, field


@dataclass
class PlenoxelConfig:
    """Configuration for Plenoxel model."""
    
    # Voxel grid settings
    grid_resolution: tuple[int, int, int] = (256, 256, 256)
    scene_bounds: tuple[float, float, float, float, float, float] = (
        -1.0,
        -1.0,
        -1.0,
        1.0,
        1.0,
        1.0,
    )
    
    # Spherical harmonics
    sh_degree: int = 2  # Degree of spherical harmonics (0-3)
    
    # Coarse-to-fine training
    use_coarse_to_fine: bool = True
    coarse_resolutions: list[tuple[int, int, int]] = field(
        default_factory=lambda: [(128, 128, 128), (256, 256, 256), (512, 512, 512)]
    )
    coarse_epochs: list[int] = field(default_factory=lambda: [2000, 5000, 10000])
    
    # Sparsity and regularization
    sparsity_threshold: float = 0.01
    tv_lambda: float = 1e-6  # Total variation regularization
    l1_lambda: float = 1e-8  # L1 sparsity regularization
    
    # Rendering settings
    step_size: float = 0.01
    sigma_thresh: float = 1e-8
    stop_thresh: float = 1e-4
    
    # Optimization
    learning_rate: float = 0.1
    weight_decay: float = 0.0
    
    # Near and far planes
    near_plane: float = 0.1
    far_plane: float = 10.0


class SphericalHarmonics:
    """Spherical harmonics utilities for view-dependent color representation."""
    
    @staticmethod
    def get_num_coeffs(degree: int) -> int:
        """Get number of SH coefficients for given degree."""
        return (degree + 1) ** 2
    
    @staticmethod
    def eval_sh(degree: int, dirs: torch.Tensor) -> torch.Tensor:
        """
        Evaluate spherical harmonics basis functions.
        
        Args:
            degree: SH degree (0-3)
            dirs: Direction vectors [N, 3] (normalized)
            
        Returns:
            SH basis values [N, (degree+1)^2]
        """
        dirs = dirs / (torch.norm(dirs, dim=-1, keepdim=True) + 1e-8)
        x, y, z = dirs[..., 0], dirs[..., 1], dirs[..., 2]
        
        result = []
        
        # Degree 0 (constant)
        result.append(torch.ones_like(x) * 0.28209479177387814)  # Y_0^0
        
        if degree >= 1:
            # Degree 1
            result.append(-0.48860251190291987 * y)  # Y_1^{-1}
            result.append(0.48860251190291987 * z)   # Y_1^0
            result.append(-0.48860251190291987 * x)  # Y_1^1
        
        if degree >= 2:
            # Degree 2
            xx, yy, zz = x * x, y * y, z * z
            xy, yz, xz = x * y, y * z, x * z
            
            result.append(1.0925484305920792 * xy)  # Y_2^{-2}
            result.append(-1.0925484305920792 * yz)  # Y_2^{-1}
            result.append(0.31539156525252005 * (2.0 * zz - xx - yy))  # Y_2^0
            result.append(-1.0925484305920792 * xz)  # Y_2^1
            result.append(0.5462742152960396 * (xx - yy))  # Y_2^2
        
        if degree >= 3:
            # Degree 3 (simplified)
            result.append(-0.5900435899266435 * y * (3.0 * xx - yy))  # Y_3^{-3}
            result.append(2.890611442640554 * xy * z)  # Y_3^{-2}
            result.append(-0.4570457994644658 * y * (4.0 * zz - xx - yy))  # Y_3^{-1}
            result.append(0.3731763325901154 * z * (2.0 * zz - 3.0 * xx - 3.0 * yy))  # Y_3^0
            result.append(-0.4570457994644658 * x * (4.0 * zz - xx - yy))  # Y_3^1
            result.append(1.445305721320277 * z * (xx - yy))  # Y_3^2
            result.append(-0.5900435899266435 * x * (xx - 3.0 * yy))  # Y_3^3
        
        return torch.stack(result, dim=-1)
    
    @staticmethod
    def eval_sh_color(sh_coeffs: torch.Tensor, dirs: torch.Tensor) -> torch.Tensor:
        """
        Evaluate color using spherical harmonics.
        
        Args:
            sh_coeffs: SH coefficients [N, 3, num_coeffs]
            dirs: View directions [N, 3]
            
        Returns:
            RGB colors [N, 3]
        """
        degree = int(math.sqrt(sh_coeffs.shape[-1])) - 1
        sh_basis = SphericalHarmonics.eval_sh(degree, dirs)  # [N, num_coeffs]
        
        # Compute RGB colors
        colors = torch.sum(sh_coeffs * sh_basis.unsqueeze(1), dim=-1)  # [N, 3]
        colors = torch.sigmoid(colors)  # Apply sigmoid activation
        
        return colors


class VoxelGrid(nn.Module):
    """Sparse voxel grid for storing density and spherical harmonic coefficients."""
    
    def __init__(
        self,
        resolution: tuple[int,
        int,
        int],
        scene_bounds: tuple[float,
        float,
        float,
        float,
        float,
        float],
        sh_degree: int = 2,
    ) -> None:
        super().__init__()
        
        self.resolution = resolution
        self.scene_bounds = torch.tensor(scene_bounds)
        self.sh_degree = sh_degree
        self.num_sh_coeffs = SphericalHarmonics.get_num_coeffs(sh_degree)
        
        # Voxel grid parameters
        # Density grid [D, H, W]
        self.density = nn.Parameter(torch.zeros(*resolution))
        
        # SH coefficients grid [D, H, W, 3, num_sh_coeffs]
        sh_shape = (*resolution, 3, self.num_sh_coeffs)
        self.sh_coeffs = nn.Parameter(torch.zeros(sh_shape))
        
        # Initialize parameters
        self.reset_parameters()
    
    def reset_parameters(self):
        """Initialize voxel grid parameters."""
        # Initialize density with small random values
        nn.init.uniform_(self.density, -0.1, 0.1)
        
        # Initialize SH coefficients
        nn.init.uniform_(self.sh_coeffs, -0.1, 0.1)
        
        # Set the 0th SH coefficient to a reasonable base color
        self.sh_coeffs.data[..., 0] = 0.5  # Gray base color
    
    def world_to_voxel_coords(self, world_coords: torch.Tensor) -> torch.Tensor:
        """Convert world coordinates to voxel grid coordinates."""
        scene_min = self.scene_bounds[:3]
        scene_max = self.scene_bounds[3:]
        scene_size = scene_max - scene_min
        
        # Normalize to [0, 1]
        normalized = (world_coords - scene_min) / scene_size
        
        # Scale to voxel resolution
        voxel_coords = normalized * torch.tensor(
            self.resolution,
            device=world_coords.device,
        )
        
        return voxel_coords
    
    def trilinear_interpolation(self, coords: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Perform trilinear interpolation to get density and SH coefficients.
        
        Args:
            coords: World coordinates [N, 3]
            
        Returns:
            Tuple of (densities, sh_coeffs)
        """
        # Convert to voxel coordinates
        voxel_coords = self.world_to_voxel_coords(coords)
        
        # Get resolution as tensor
        resolution_tensor = torch.tensor(self.resolution, device=coords.device)
        max_coords = resolution_tensor.float() - 1.0
        
        # Clamp coordinates to valid range
        voxel_coords = torch.clamp(voxel_coords, min=0.0, max=max_coords)
        
        # Get integer and fractional parts
        coords_floor = torch.floor(voxel_coords).long()
        coords_frac = voxel_coords - coords_floor.float()
        
        # Get neighboring voxel indices
        x0, y0, z0 = coords_floor[..., 0], coords_floor[..., 1], coords_floor[..., 2]
        x1 = torch.clamp(x0 + 1, min=0, max=self.resolution[0] - 1)
        y1 = torch.clamp(y0 + 1, min=0, max=self.resolution[1] - 1)
        z1 = torch.clamp(z0 + 1, min=0, max=self.resolution[2] - 1)
        
        # Get fractional weights
        dx, dy, dz = coords_frac[..., 0], coords_frac[..., 1], coords_frac[..., 2]
        
        # Trilinear interpolation weights
        w000 = (1 - dx) * (1 - dy) * (1 - dz)
        w001 = (1 - dx) * (1 - dy) * dz
        w010 = (1 - dx) * dy * (1 - dz)
        w011 = (1 - dx) * dy * dz
        w100 = dx * (1 - dy) * (1 - dz)
        w101 = dx * (1 - dy) * dz
        w110 = dx * dy * (1 - dz)
        w111 = dx * dy * dz
        
        # Interpolate density
        density_interp = (
            w000 * self.density[z0, y0, x0] +
            w001 * self.density[z1, y0, x0] +
            w010 * self.density[z0, y1, x0] +
            w011 * self.density[z1, y1, x0] +
            w100 * self.density[z0, y0, x1] +
            w101 * self.density[z1, y0, x1] +
            w110 * self.density[z0, y1, x1] +
            w111 * self.density[z1, y1, x1]
        )
        
        # Interpolate SH coefficients
        sh_interp = (
            w000.unsqueeze(-1).unsqueeze(-1) * self.sh_coeffs[z0, y0, x0] +
            w001.unsqueeze(-1).unsqueeze(-1) * self.sh_coeffs[z1, y0, x0] +
            w010.unsqueeze(-1).unsqueeze(-1) * self.sh_coeffs[z0, y1, x0] +
            w011.unsqueeze(-1).unsqueeze(-1) * self.sh_coeffs[z1, y1, x0] +
            w100.unsqueeze(-1).unsqueeze(-1) * self.sh_coeffs[z0, y0, x1] +
            w101.unsqueeze(-1).unsqueeze(-1) * self.sh_coeffs[z1, y0, x1] +
            w110.unsqueeze(-1).unsqueeze(-1) * self.sh_coeffs[z0, y1, x1] +
            w111.unsqueeze(-1).unsqueeze(-1) * self.sh_coeffs[z1, y1, x1]
        )
        
        return density_interp, sh_interp
    
    def get_occupancy_mask(self, threshold: float = 0.01) -> torch.Tensor:
        """Get occupancy mask for sparsity regularization."""
        return (torch.exp(self.density) > threshold).float()
    
    def prune_voxels(self, threshold: float = 0.01):
        """Prune voxels with low density."""
        mask = self.get_occupancy_mask(threshold)
        self.density.data *= mask
        self.sh_coeffs.data *= mask.unsqueeze(-1).unsqueeze(-1)
    
    def total_variation_loss(self) -> torch.Tensor:
        """Compute total variation loss for smoothness."""
        # TV loss for density
        tv_density = (
            torch.mean(torch.abs(self.density[1:, :, :] - self.density[:-1, :, :])) +
            torch.mean(torch.abs(self.density[:, 1:, :] - self.density[:, :-1, :])) +
            torch.mean(torch.abs(self.density[:, :, 1:] - self.density[:, :, :-1]))
        )
        
        # TV loss for SH coefficients
        tv_sh = (
            torch.mean(torch.abs(self.sh_coeffs[1:, :, :] - self.sh_coeffs[:-1, :, :])) +
            torch.mean(torch.abs(self.sh_coeffs[:, 1:, :] - self.sh_coeffs[:, :-1, :])) +
            torch.mean(torch.abs(self.sh_coeffs[:, :, 1:] - self.sh_coeffs[:, :, :-1]))
        )
        
        return tv_density + tv_sh
    
    def l1_loss(self) -> torch.Tensor:
        """Compute L1 sparsity loss."""
        return torch.mean(torch.abs(torch.exp(self.density))) + torch.mean(torch.abs(self.sh_coeffs))


class VolumetricRenderer(nn.Module):
    """Volume renderer for Plenoxels."""
    
    def __init__(self, config: PlenoxelConfig):
        super().__init__()
        self.config = config
    
    def sample_points_along_rays(
        self,
        ray_origins: torch.Tensor,
        ray_directions: torch.Tensor,
        near: float,
        far: float,
        num_samples: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Sample points along rays."""
        device = ray_origins.device
        batch_size = ray_origins.shape[0]
        
        # Linear sampling in depth
        t_vals = torch.linspace(near, far, num_samples, device=device)
        t_vals = t_vals.expand(batch_size, num_samples)
        
        # Add jitter for training
        if self.training:
            mids = 0.5 * (t_vals[..., 1:] + t_vals[..., :-1])
            upper = torch.cat([mids, t_vals[..., -1:]], dim=-1)
            lower = torch.cat([t_vals[..., :1], mids], dim=-1)
            t_rand = torch.rand_like(t_vals)
            t_vals = lower + (upper - lower) * t_rand
        
        # Compute sample points
        points = ray_origins.unsqueeze(-2) + t_vals.unsqueeze(-1) * ray_directions.unsqueeze(-2)
        
        return points, t_vals
    
    def volume_render(
        self,
        densities: torch.Tensor,
        colors: torch.Tensor,
        t_vals: torch.Tensor,
        ray_directions: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Perform volume rendering."""
        # Compute distances between samples
        dists = t_vals[..., 1:] - t_vals[..., :-1]
        dists = torch.cat([dists, torch.full_like(dists[..., :1], 1e10)], dim=-1)
        
        # Apply ray direction norm
        dists = dists * torch.norm(ray_directions, dim=-1, keepdim=True)
        
        # Convert density to alpha
        alpha = 1.0 - torch.exp(-torch.relu(densities) * dists)
        
        # Compute transmittance
        transmittance = torch.cumprod(1.0 - alpha + 1e-10, dim=-1)
        transmittance = torch.cat([torch.ones_like(transmittance[..., :1]), transmittance], dim=-1)
        
        # Compute weights
        weights = alpha * transmittance
        
        # Composite colors
        rgb = torch.sum(weights.unsqueeze(-1) * colors, dim=-2)
        
        # Compute depth
        depth = torch.sum(weights * t_vals, dim=-1)
        
        # Compute accumulated weights (for background)
        acc_weights = torch.sum(weights, dim=-1)
        
        return {
            'rgb': rgb, 'depth': depth, 'weights': weights, 'alpha': alpha, 'acc_weights': acc_weights
        }


class PlenoxelModel(nn.Module):
    """Main Plenoxel model."""
    
    def __init__(self, config: PlenoxelConfig):
        super().__init__()
        self.config = config
        
        # Initialize voxel grid
        self.voxel_grid = VoxelGrid(
            resolution=config.grid_resolution, scene_bounds=config.scene_bounds, sh_degree=config.sh_degree
        )
        
        # Volume renderer
        self.renderer = VolumetricRenderer(config)
        
        # Current resolution level for coarse-to-fine training
        self.current_resolution_level = 0
    
    def forward(
        self,
        ray_origins: torch.Tensor,
        ray_directions: torch.Tensor,
        num_samples: int = 192,
    ) -> dict[str, torch.Tensor]:
        """
        Forward pass of Plenoxel model.
        
        Args:
            ray_origins: Ray origins [N, 3]
            ray_directions: Ray directions [N, 3]
            num_samples: Number of samples per ray
            
        Returns:
            Dictionary with rendered outputs
        """
        # Sample points along rays
        points, t_vals = self.renderer.sample_points_along_rays(
            ray_origins, ray_directions, self.config.near_plane, self.config.far_plane, num_samples
        )
        
        # Flatten points for voxel grid query
        points_flat = points.reshape(-1, 3)
        
        # Query voxel grid
        densities_flat, sh_coeffs_flat = self.voxel_grid.trilinear_interpolation(points_flat)
        
        # Reshape back
        batch_size, n_samples = points.shape[:2]
        densities = densities_flat.reshape(batch_size, n_samples)
        sh_coeffs = sh_coeffs_flat.reshape(batch_size, n_samples, 3, -1)
        
        # Compute view directions for each sample
        view_dirs = ray_directions.unsqueeze(-2).expand(-1, n_samples, -1)
        view_dirs_flat = view_dirs.reshape(-1, 3)
        
        # Evaluate spherical harmonics for colors
        sh_coeffs_eval = sh_coeffs.reshape(-1, 3, sh_coeffs.shape[-1])
        colors_flat = SphericalHarmonics.eval_sh_color(sh_coeffs_eval, view_dirs_flat)
        colors = colors_flat.reshape(batch_size, n_samples, 3)
        
        # Volume rendering
        render_outputs = self.renderer.volume_render(
            densities, colors, t_vals, ray_directions
        )
        
        # Add additional outputs
        render_outputs.update({
            'densities': densities, 'colors': colors, 'points': points, 't_vals': t_vals
        })
        
        return render_outputs
    
    def update_resolution(self, new_resolution: tuple[int, int, int]):
        """Update voxel grid resolution for coarse-to-fine training."""
        if new_resolution != self.voxel_grid.resolution:
            print(f"Updating resolution from {self.voxel_grid.resolution} to {new_resolution}")
            
            # Create new voxel grid
            old_grid = self.voxel_grid
            new_grid = VoxelGrid(
                resolution=new_resolution, scene_bounds=self.config.scene_bounds, sh_degree=self.config.sh_degree
            ).to(old_grid.density.device)
            
            # Upsample existing grid if needed
            if hasattr(old_grid, 'density'):
                # Use trilinear interpolation to upsample
                new_grid = self._upsample_grid(old_grid, new_grid)
            
            self.voxel_grid = new_grid
    
    def _upsample_grid(self, old_grid: VoxelGrid, new_grid: VoxelGrid) -> VoxelGrid:
        """Upsample voxel grid to higher resolution."""
        old_res = old_grid.resolution
        new_res = new_grid.resolution
        
        # Create coordinate grids
        device = old_grid.density.device
        
        # Upsample density
        density_old = old_grid.density.unsqueeze(0).unsqueeze(0)  # [1, 1, D, H, W]
        density_new = F.interpolate(density_old, size=new_res, mode='trilinear', align_corners=True)
        new_grid.density.data = density_new.squeeze(0).squeeze(0)
        
        # Upsample SH coefficients
        sh_old = old_grid.sh_coeffs.permute(3, 4, 0, 1, 2)  # [3, n_coeffs, D, H, W]
        sh_old = sh_old.reshape(-1, *old_res).unsqueeze(0)  # [1, 3*n_coeffs, D, H, W]
        sh_new = F.interpolate(sh_old, size=new_res, mode='trilinear', align_corners=True)
        sh_new = sh_new.squeeze(0).reshape(3, new_grid.num_sh_coeffs, *new_res)
        new_grid.sh_coeffs.data = sh_new.permute(2, 3, 4, 0, 1)  # [D, H, W, 3, n_coeffs]
        
        return new_grid
    
    def prune_voxels(self, threshold: float | None = None):
        """Prune low-density voxels."""
        if threshold is None:
            threshold = self.config.sparsity_threshold
        self.voxel_grid.prune_voxels(threshold)
    
    def get_occupancy_stats(self) -> dict[str, float]:
        """Get statistics about voxel occupancy."""
        mask = self.voxel_grid.get_occupancy_mask(self.config.sparsity_threshold)
        total_voxels = mask.numel()
        occupied_voxels = mask.sum().item()
        
        return {
            'total_voxels': total_voxels, 'occupied_voxels': occupied_voxels, 'occupancy_ratio': occupied_voxels / total_voxels, 'sparsity_ratio': 1.0 - (
                occupied_voxels / total_voxels,
            )
        }


class PlenoxelLoss(nn.Module):
    """Loss function for Plenoxel training."""
    
    def __init__(self, config: PlenoxelConfig):
        super().__init__()
        self.config = config
    
    def forward(
        self,
        outputs: dict[str,
        torch.Tensor],
        targets: dict[str,
        torch.Tensor],
    ) -> dict[str, torch.Tensor]:   
        """
        Compute Plenoxel losses.
        
        Args:
            outputs: Model outputs
            targets: Target values
            
        Returns:
            Dictionary of losses
        """
        losses = {}
        
        # Color reconstruction loss (MSE)
        if 'rgb' in outputs and 'colors' in targets:
            color_loss = F.mse_loss(outputs['rgb'], targets['colors'])
            losses['color_loss'] = color_loss
        
        # Total variation loss for smoothness
        if self.config.tv_lambda > 0:
            tv_loss = self.config.tv_lambda * outputs.get('tv_loss', 0)
            losses['tv_loss'] = tv_loss
        
        # L1 sparsity loss
        if self.config.l1_lambda > 0:
            l1_loss = self.config.l1_lambda * outputs.get('l1_loss', 0)
            losses['l1_loss'] = l1_loss
        
        # Depth loss (if available)
        if 'depth' in outputs and 'depths' in targets:
            depth_loss = F.mse_loss(outputs['depth'], targets['depths'])
            losses['depth_loss'] = depth_loss
        
        return losses 