"""
Plenoxels Core Implementation

This module implements the core components of Plenoxels:
- VoxelGrid: Sparse voxel grid for density and color representation
- SphericalHarmonics: Utilities for spherical harmonics computation
- PlenoxelModel: Main model combining voxel grid and rendering
- PlenoxelLoss: Loss functions for training
- Utility functions for interpolation and coordinate conversion

The implementation follows the paper "Plenoxels: Radiance Fields without Neural Networks"
and provides a fast, efficient representation for neural radiance fields.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.amp.autocast_mode import autocast
from torch.amp.grad_scaler import GradScaler
import numpy as np
import math
from dataclasses import dataclass, field
import logging
from typing import Optional, Union
from pathlib import Path
import time
import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)

# Check CUDA availability
CUDA_AVAILABLE = torch.cuda.is_available()

try:
    from .cuda import plenoxels_cuda

    CUDA_EXTENSION_AVAILABLE = True
except ImportError:
    logger.warning("CUDA extensions not available, using CPU implementation")
    CUDA_EXTENSION_AVAILABLE = False


@dataclass
class PlenoxelConfig:
    """Configuration for Plenoxels model.

    Attributes:
        grid_resolution: Tuple of (depth, height, width) for voxel grid
        scene_bounds: Tensor of [min_x, min_y, min_z, max_x, max_y, max_z]
        sh_degree: Maximum degree of spherical harmonics
        near_plane: Minimum rendering distance
        far_plane: Maximum rendering distance
        step_size: Ray marching step size
        sigma_thresh: Density threshold for early stopping
        stop_thresh: Transmittance threshold for early stopping
        use_coarse_to_fine: Whether to use coarse-to-fine optimization
        coarse_to_fine_steps: Number of coarse-to-fine steps
        coarse_to_fine_iters: Iterations per coarse-to-fine step
        coarse_resolutions: List of grid resolutions for coarse-to-fine
        coarse_epochs: List of epochs for each coarse-to-fine step
        sparsity_threshold: Threshold for pruning low-density voxels
        tv_lambda: Total variation regularization weight
        l1_lambda: L1 sparsity regularization weight
        learning_rate: Base learning rate
        weight_decay: Weight decay for optimization
        use_cuda: Whether to use CUDA acceleration if available
        device: Device to use for computation
        rgb_loss_type: Type of RGB loss function
        loss_reduction: Loss reduction method
        depth_loss_type: Type of depth loss function
        normal_loss_type: Type of normal loss function
        density_loss_type: Type of density loss function
        depth_lambda: Weight for depth loss
        normal_lambda: Weight for normal loss
        cache_size: Maximum cache size
        batch_size: Training batch size
        log_every: Log frequency in steps
        checkpoint_dir: Directory for saving checkpoints
        log_dir: Directory for saving logs
        num_epochs: Number of training epochs
    """

    # Grid parameters
    grid_resolution: tuple[int, int, int] = (256, 256, 256)
    scene_bounds: torch.Tensor | None = None

    # Spherical harmonics parameters
    sh_degree: int = 2

    # Rendering parameters
    near_plane: float = 0.1
    far_plane: float = 10.0
    step_size: float = 0.01
    sigma_thresh: float = 1e-8
    stop_thresh: float = 1e-7
    num_samples: int = 64  # Number of samples along each ray

    # Training parameters
    use_coarse_to_fine: bool = True
    coarse_to_fine_steps: int = 4
    coarse_to_fine_iters: int = 1000

    # Coarse-to-fine resolutions
    coarse_resolutions: list[tuple[int, int, int]] = field(
        default_factory=lambda: [(128, 128, 128), (256, 256, 256), (512, 512, 512)]
    )
    coarse_epochs: list[int] = field(default_factory=lambda: [2000, 5000, 10000])

    # Regularization parameters
    sparsity_threshold: float = 0.01
    tv_lambda: float = 1e-6  # Total variation regularization
    l1_lambda: float = 1e-8  # L1 sparsity regularization

    # Optimization parameters
    learning_rate: float = 0.1
    weight_decay: float = 0.0

    # Device parameters
    use_cuda: bool = True
    device: torch.device = field(
        default_factory=lambda: torch.device("cuda" if torch.cuda.is_available() else "cpu")
    )

    # Loss parameters
    rgb_loss_type: str = "mse"
    loss_reduction: str = "mean"
    depth_loss_type: str = "l1"
    normal_loss_type: str = "l1"
    density_loss_type: str = "l1"
    depth_lambda: float = 1.0
    normal_lambda: float = 1.0

    # Cache parameters
    cache_size: int = 1000

    # Training parameters
    batch_size: int = 4096
    log_every: int = 100
    checkpoint_dir: str = "checkpoints"
    log_dir: str = "logs"
    num_epochs: int = 50

    def __post_init__(self):
        """Initialize default scene bounds if not provided."""
        if self.scene_bounds is None:
            self.scene_bounds = torch.tensor([-1.0, -1.0, -1.0, 1.0, 1.0, 1.0], device=self.device)

        # Validate parameters
        assert len(self.grid_resolution) == 3, "Grid resolution must be a 3-tuple"
        assert self.sh_degree >= 0, "SH degree must be non-negative"
        assert self.near_plane > 0, "Near plane must be positive"
        assert self.far_plane > self.near_plane, "Far plane must be greater than near plane"
        assert self.step_size > 0, "Step size must be positive"
        assert len(self.coarse_resolutions) == len(
            self.coarse_epochs
        ), "Mismatched coarse-to-fine parameters"


class CUDAManager:
    """CUDA resource manager for Plenoxels."""

    def __init__(self, config: PlenoxelConfig):
        """Initialize CUDA manager."""
        self.config = config
        self.use_cuda = config.use_cuda and CUDA_AVAILABLE

        if self.use_cuda:
            self.device = torch.device("cuda")
            self.streams = [torch.cuda.Stream() for _ in range(4)]  # Create multiple streams
        else:
            self.device = torch.device("cpu")
            self.streams = None

    def get_device(self) -> torch.device:
        """Get current device."""
        return self.device

    def get_stream(self, idx: int = 0) -> Optional[torch.cuda.Stream]:
        """Get CUDA stream by index."""
        if self.streams:
            return self.streams[idx % len(self.streams)]
        return None

    def synchronize(self):
        """Synchronize all CUDA streams."""
        if self.use_cuda:
            torch.cuda.synchronize()

    def clear_cache(self):
        """Clear CUDA cache."""
        if self.use_cuda:
            torch.cuda.empty_cache()

    @staticmethod
    def is_available() -> bool:
        """Check if CUDA is available."""
        return CUDA_AVAILABLE and torch.cuda.is_available()


class SphericalHarmonics:
    """Spherical harmonics utilities with improved edge case handling."""

    @staticmethod
    def get_num_coeffs(degree: int) -> int:
        """Get number of coefficients for given degree."""
        return (degree + 1) ** 2

    @staticmethod
    def eval_sh_basis(degree: int, dirs: torch.Tensor) -> torch.Tensor:
        """Evaluate spherical harmonics basis functions.

        Args:
            degree: Maximum degree of spherical harmonics
            dirs: Direction vectors [..., 3], will be normalized

        Returns:
            Basis function values [..., num_coeffs]

        Raises:
            ValueError: If degree is negative or dirs has wrong shape
        """
        if degree < 0:
            raise ValueError(f"Expected non-negative degree, got {degree}")
        if dirs.shape[-1] != 3:
            raise ValueError(f"Expected dirs to have shape [..., 3], got {dirs.shape}")

        # Handle edge cases
        dirs = dirs.clone()  # Avoid modifying input
        zero_mask = torch.sum(dirs * dirs, dim=-1) < 1e-8
        if zero_mask.any():
            # Replace zero vectors with arbitrary unit vector
            dirs[zero_mask] = torch.tensor([0.0, 0.0, 1.0], device=dirs.device)

        # Normalize directions
        dirs = F.normalize(dirs, dim=-1)
        x, y, z = dirs[..., 0], dirs[..., 1], dirs[..., 2]

        num_coeffs = SphericalHarmonics.get_num_coeffs(degree)
        result = torch.empty((*dirs.shape[:-1], num_coeffs), device=dirs.device)

        # l = 0 (constant)
        result[..., 0] = 0.282095  # 1/(2*sqrt(pi))

        if degree > 0:
            # l = 1 (linear)
            result[..., 1] = 0.488603 * y  # sqrt(3)/(2*sqrt(pi)) * y
            result[..., 2] = 0.488603 * z  # sqrt(3)/(2*sqrt(pi)) * z
            result[..., 3] = 0.488603 * x  # sqrt(3)/(2*sqrt(pi)) * x

        if degree > 1:
            # l = 2 (quadratic)
            result[..., 4] = 1.092548 * x * y  # sqrt(15)/(2*sqrt(pi)) * x * y
            result[..., 5] = 1.092548 * y * z  # sqrt(15)/(2*sqrt(pi)) * y * z
            result[..., 6] = 0.315392 * (3 * z * z - 1)  # sqrt(5)/(4*sqrt(pi)) * (3z^2 - 1)
            result[..., 7] = 1.092548 * x * z  # sqrt(15)/(2*sqrt(pi)) * x * z
            result[..., 8] = 0.546274 * (x * x - y * y)  # sqrt(15)/(4*sqrt(pi)) * (x^2 - y^2)

        if degree > 2:
            # l = 3 (cubic)
            result[..., 9] = 0.590044 * y * (3 * x * x - y * y)
            result[..., 10] = 2.890611 * x * y * z
            result[..., 11] = 0.457046 * y * (5 * z * z - 1)
            result[..., 12] = 0.373176 * z * (5 * z * z - 3)
            result[..., 13] = 0.457046 * x * (5 * z * z - 1)
            result[..., 14] = 1.445306 * z * (x * x - y * y)
            result[..., 15] = 0.590044 * x * (x * x - 3 * y * y)

        return result

    @staticmethod
    def eval_sh_color(sh_coeffs: torch.Tensor, dirs: torch.Tensor) -> torch.Tensor:
        """Evaluate spherical harmonics to get colors.

        Args:
            sh_coeffs: SH coefficients [..., 3, num_coeffs]
            dirs: Direction vectors [..., 3], will be normalized

        Returns:
            RGB colors [..., 3] in range [0, 1]

        Raises:
            ValueError: If inputs have invalid dimensions
        """
        if sh_coeffs.shape[-2] != 3:
            raise ValueError(
                f"Expected sh_coeffs to have shape [..., 3, num_coeffs], got {sh_coeffs.shape}"
            )
        if dirs.shape[-1] != 3:
            raise ValueError(f"Expected dirs to have shape [..., 3], got {dirs.shape}")

        # Get degree from number of coefficients
        num_coeffs = sh_coeffs.shape[-1]
        degree = int(math.sqrt(num_coeffs) - 1 + 1e-6)

        # Evaluate basis functions
        basis = SphericalHarmonics.eval_sh_basis(degree, dirs)  # [..., num_coeffs]

        # Compute colors with proper broadcasting
        # sh_coeffs: [..., 3, num_coeffs], basis: [..., num_coeffs]
        # Expand basis to [..., 1, num_coeffs] for broadcasting with sh_coeffs
        basis = basis.unsqueeze(-2)  # [..., 1, num_coeffs]
        colors = torch.sum(sh_coeffs * basis, dim=-1)  # [..., 3]

        # Clamp to valid range with smooth transition
        return torch.sigmoid(colors)


def trilinear_interpolation(
    grid: torch.Tensor,
    points: torch.Tensor,
    has_channels: bool = False,
    align_corners: bool = True,
) -> torch.Tensor:
    """Trilinear interpolation with proper shape handling.

    Args:
        grid: Grid of values [..., H, W, D] or [..., H, W, D, C, N] if has_channels
        points: Points to interpolate at [..., 3]
        has_channels: Whether grid has channel dimensions
        align_corners: Whether to align corners

    Returns:
        Interpolated values [...] or [..., C, N] if has_channels
    """
    # Get grid shape
    grid_shape = grid.shape

    # Handle empty input
    if points.numel() == 0:
        output_shape = points.shape[:-1]
        if has_channels:
            output_shape = (*output_shape, grid.shape[-2], grid.shape[-1])
        return torch.zeros(output_shape, device=grid.device)

    # Scale points from [-1, 1] to [0, size-1]
    if align_corners:
        points = (points + 1) * 0.5
        points = points * torch.tensor(
            [grid_shape[-3] - 1, grid_shape[-2] - 1, grid_shape[-1] - 1],
            device=points.device,
        )
    else:
        points = (
            (points + 1)
            * torch.tensor(
                [grid_shape[-3], grid_shape[-2], grid_shape[-1]],
                device=points.device,
            )
            - 1
        ) * 0.5

    # Get integer and fractional coordinates
    points_floor = torch.floor(points).long()  # Convert to long here
    points_ceil = torch.ceil(points).long()  # Convert to long here
    frac = points - points_floor.float()  # Convert back to float for interpolation

    # Ensure indices are in bounds
    def clamp_coords(coords: torch.Tensor) -> torch.Tensor:
        return torch.stack(
            [
                coords[..., 0].clamp(0, grid_shape[-3] - 1),
                coords[..., 1].clamp(0, grid_shape[-2] - 1),
                coords[..., 2].clamp(0, grid_shape[-1] - 1),
            ],
            dim=-1,
        )

    points_floor = clamp_coords(points_floor)
    points_ceil = clamp_coords(points_ceil)

    # Get corner indices
    idx000 = points_floor
    idx100 = torch.stack([points_ceil[..., 0], points_floor[..., 1], points_floor[..., 2]], dim=-1)
    idx010 = torch.stack([points_floor[..., 0], points_ceil[..., 1], points_floor[..., 2]], dim=-1)
    idx110 = torch.stack([points_ceil[..., 0], points_ceil[..., 1], points_floor[..., 2]], dim=-1)
    idx001 = torch.stack([points_floor[..., 0], points_floor[..., 1], points_ceil[..., 2]], dim=-1)
    idx101 = torch.stack([points_ceil[..., 0], points_floor[..., 1], points_ceil[..., 2]], dim=-1)
    idx011 = torch.stack([points_floor[..., 0], points_ceil[..., 1], points_ceil[..., 2]], dim=-1)
    idx111 = points_ceil

    # Compute weights with proper broadcasting
    x, y, z = frac[..., 0:1], frac[..., 1:2], frac[..., 2:3]
    w000 = (1 - x) * (1 - y) * (1 - z)  # [..., 1]
    w100 = x * (1 - y) * (1 - z)
    w010 = (1 - x) * y * (1 - z)
    w110 = x * y * (1 - z)
    w001 = (1 - x) * (1 - y) * z
    w101 = x * (1 - y) * z
    w011 = (1 - x) * y * z
    w111 = x * y * z

    # Gather values with proper shape handling
    def gather_values(idx: torch.Tensor) -> torch.Tensor:
        if has_channels:
            # Grid shape is [..., H, W, D, C, N], idx shape is [..., 3]
            gathered = grid[..., idx[..., 0], idx[..., 1], idx[..., 2], :, :]
        else:
            gathered = grid[..., idx[..., 0], idx[..., 1], idx[..., 2]]
            # Add a new axis for broadcasting with weights
            gathered = gathered.unsqueeze(-1) if gathered.dim() == points.dim() - 1 else gathered
        return gathered

    # Add channel and coefficient dimensions for broadcasting
    if has_channels:
        # Add dimensions for [..., 1, 1] to broadcast with [..., C, N]
        w000 = w000.view(*w000.shape[:-1], 1, 1)
        w100 = w100.view(*w100.shape[:-1], 1, 1)
        w010 = w010.view(*w010.shape[:-1], 1, 1)
        w110 = w110.view(*w110.shape[:-1], 1, 1)
        w001 = w001.view(*w001.shape[:-1], 1, 1)
        w101 = w101.view(*w101.shape[:-1], 1, 1)
        w011 = w011.view(*w011.shape[:-1], 1, 1)
        w111 = w111.view(*w111.shape[:-1], 1, 1)
    else:
        # Add single dimension for broadcasting with [..., 1]
        w000 = w000.view(*w000.shape[:-1], 1)
        w100 = w100.view(*w100.shape[:-1], 1)
        w010 = w010.view(*w010.shape[:-1], 1)
        w110 = w110.view(*w110.shape[:-1], 1)
        w001 = w001.view(*w001.shape[:-1], 1)
        w101 = w101.view(*w101.shape[:-1], 1)
        w011 = w011.view(*w011.shape[:-1], 1)
        w111 = w111.view(*w111.shape[:-1], 1)

    # Interpolate with proper broadcasting
    values = (
        w000 * gather_values(idx000)
        + w100 * gather_values(idx100)
        + w010 * gather_values(idx010)
        + w110 * gather_values(idx110)
        + w001 * gather_values(idx001)
        + w101 * gather_values(idx101)
        + w011 * gather_values(idx011)
        + w111 * gather_values(idx111)
    )

    # Remove the extra dimension we added for broadcasting if not using channels
    if not has_channels:
        values = values.squeeze(-1)

    return values


class VoxelGrid(nn.Module):
    """Sparse voxel grid for density and color representation.

    Features:
    - Memory-efficient sparse storage
    - Modern PyTorch optimizations
    - Efficient coordinate transforms
    - Automatic mixed precision support
    """

    def __init__(
        self,
        resolution: tuple[int, int, int],
        scene_bounds: torch.Tensor,
        num_sh_coeffs: int = 9,
        device: torch.device | None = None,
    ):
        """Initialize voxel grid.

        Args:
            resolution: Grid resolution (depth, height, width)
            scene_bounds: Scene bounds tensor [min_x, min_y, min_z, max_x, max_y, max_z]
            num_sh_coeffs: Number of spherical harmonics coefficients
            device: Device to place tensors on
        """
        super().__init__()

        self.resolution = resolution
        self.num_sh_coeffs = num_sh_coeffs

        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Move scene bounds to device efficiently
        self.register_buffer("scene_bounds", scene_bounds.to(device))

        # Initialize grid parameters with modern initialization
        self.density = nn.Parameter(torch.zeros(resolution, device=device), requires_grad=True)

        # Initialize spherical harmonics coefficients with correct shape
        self.sh_coeffs = nn.Parameter(
            torch.zeros(
                resolution[0], resolution[1], resolution[2], 3, num_sh_coeffs, device=device
            ),
            requires_grad=True,
        )

        # Initialize coordinate transforms efficiently
        self.register_buffer(
            "voxel_size",
            (self.scene_bounds[3:] - self.scene_bounds[:3])
            / torch.tensor(resolution, device=device),
        )
        self.register_buffer("min_coords", self.scene_bounds[:3])

        # Initialize occupancy grid for efficient ray marching
        self.register_buffer(
            "occupancy_grid", torch.zeros(resolution, dtype=torch.bool, device=device)
        )

    def world_to_voxel_coords(self, points: torch.Tensor) -> torch.Tensor:
        """Convert world coordinates to voxel coordinates efficiently.

        Args:
            points: World coordinates [..., 3]

        Returns:
            Voxel coordinates [..., 3] in range [0, resolution-1]
        """
        # Handle empty input efficiently
        if points.numel() == 0:
            return torch.zeros_like(points)

        # Get resolution as tensor and ensure it's on the same device as points
        resolution = torch.tensor(self.resolution, device=points.device, dtype=points.dtype)
        scene_bounds = self.scene_bounds.to(points.device)

        # Check bounds efficiently using broadcasting
        out_min = points < scene_bounds[:3].unsqueeze(0)
        out_max = points > scene_bounds[3:].unsqueeze(0)

        # Normalize to [0, 1] range efficiently
        points_norm = (points - scene_bounds[:3]) / (scene_bounds[3:] - scene_bounds[:3])

        # Scale to voxel coordinates efficiently
        voxel_coords = points_norm * (resolution - 1).to(points.dtype)

        # Handle out-of-bounds points efficiently using where
        zeros = torch.zeros_like(voxel_coords)
        max_coords = (resolution - 1).to(points.dtype).expand_as(voxel_coords)

        voxel_coords = torch.where(out_min, zeros, voxel_coords)
        voxel_coords = torch.where(out_max, max_coords, voxel_coords)

        return voxel_coords

    def voxel_to_world_coords(self, voxel_coords: torch.Tensor) -> torch.Tensor:
        """Convert voxel coordinates to world coordinates efficiently.

        Args:
            voxel_coords: Voxel coordinates [..., 3] in range [0, resolution-1]

        Returns:
            World coordinates [..., 3]
        """
        # Handle empty input efficiently
        if voxel_coords.numel() == 0:
            return torch.zeros_like(voxel_coords)

        # Get resolution as tensor and ensure it's on the same device as voxel_coords
        resolution = torch.tensor(
            self.resolution, device=voxel_coords.device, dtype=voxel_coords.dtype
        )
        scene_bounds = self.scene_bounds.to(voxel_coords.device)

        # Normalize to [0, 1] range efficiently
        points_norm = voxel_coords / (resolution - 1).to(voxel_coords.dtype)

        # Scale to world coordinates efficiently
        world_coords = points_norm * (scene_bounds[3:] - scene_bounds[:3]) + scene_bounds[:3]

        return world_coords

    def forward(self, points: torch.Tensor, dirs: torch.Tensor) -> dict[str, torch.Tensor]:
        """Forward pass with automatic mixed precision support.

        Args:
            points: World coordinates [..., 3]
            dirs: View directions [..., 3]

        Returns:
            Dictionary with density and color values
        """
        # Convert to voxel coordinates efficiently
        voxel_coords = self.world_to_voxel_coords(points)

        # Get density and SH coefficients through trilinear interpolation
        with autocast(device_type="cuda" if torch.cuda.is_available() else "cpu"):
            density = trilinear_interpolation(self.density, voxel_coords, has_channels=False)
            sh_coeffs = trilinear_interpolation(self.sh_coeffs, voxel_coords, has_channels=True)

            # Compute colors from SH coefficients efficiently
            colors = SphericalHarmonics.eval_sh_color(sh_coeffs, dirs)

        return {
            "density": F.relu(density),  # Non-negative density
            "rgb": torch.sigmoid(colors),  # Ensure valid RGB range
        }

    def update_occupancy_grid(self, density_threshold: float = 0.01) -> None:
        """Update occupancy grid for efficient ray marching.

        Args:
            density_threshold: Density threshold for occupancy
        """
        with torch.no_grad():
            self.occupancy_grid.copy_(self.density > density_threshold)

    def total_variation_loss(self) -> torch.Tensor:
        """Compute total variation loss efficiently."""
        with autocast(device_type="cuda" if torch.cuda.is_available() else "cpu"):
            tv_x = torch.mean(torch.abs(self.density[1:, :, :] - self.density[:-1, :, :]))
            tv_y = torch.mean(torch.abs(self.density[:, 1:, :] - self.density[:, :-1, :]))
            tv_z = torch.mean(torch.abs(self.density[:, :, 1:] - self.density[:, :, :-1]))

        return (tv_x + tv_y + tv_z) / 3.0

    def l1_loss(self) -> torch.Tensor:
        """Compute L1 sparsity loss efficiently."""
        with autocast(device_type="cuda" if torch.cuda.is_available() else "cpu"):
            return torch.mean(torch.abs(self.density))

    def prune_empty_voxels(self, density_threshold: float = 0.01) -> None:
        """Prune empty voxels for memory efficiency.

        Args:
            density_threshold: Density threshold for pruning
        """
        with torch.no_grad():
            mask = self.density > density_threshold
            self.density.data *= mask
            self.sh_coeffs.data *= mask.unsqueeze(-1).unsqueeze(-1)
            self.occupancy_grid.copy_(mask)

    @torch.jit.export
    def get_density_and_color(
        self, points: torch.Tensor, dirs: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Get density and color values efficiently (TorchScript compatible).

        Args:
            points: World coordinates [..., 3]
            dirs: View directions [..., 3]

        Returns:
            Tuple of density and color values
        """
        outputs = self.forward(points, dirs)
        return outputs["density"], outputs["rgb"]

    def get_density(self, points: torch.Tensor) -> torch.Tensor:
        """Get density values at world coordinates.

        Args:
            points: World coordinates [..., 3]

        Returns:
            Density values [...]
        """
        # Convert to voxel coordinates
        voxel_coords = self.world_to_voxel_coords(points)

        # Get density through trilinear interpolation
        density = trilinear_interpolation(self.density, voxel_coords, has_channels=False)

        # Ensure non-negative density
        return F.relu(density)

    def get_color(self, points: torch.Tensor, dirs: torch.Tensor) -> torch.Tensor:
        """Get color values at world coordinates.

        Args:
            points: World coordinates [..., 3]
            dirs: View directions [..., 3]

        Returns:
            RGB colors [..., 3]
        """
        # Convert to voxel coordinates
        voxel_coords = self.world_to_voxel_coords(points)

        # Get SH coefficients through trilinear interpolation
        sh_coeffs = trilinear_interpolation(self.sh_coeffs, voxel_coords, has_channels=True)

        # Compute colors from SH coefficients
        colors = SphericalHarmonics.eval_sh_color(sh_coeffs, dirs)

        # Ensure valid RGB range
        return torch.sigmoid(colors)

    def get_sh_coeffs(self, points: torch.Tensor) -> torch.Tensor:
        """Get spherical harmonics coefficients at sampled points.

        Args:
            points: Sampled points [N, num_samples, 3]

        Returns:
            Spherical harmonics coefficients [N, num_samples, 3, num_coeffs]
        """
        # Convert points to voxel coordinates
        voxel_coords = self.world_to_voxel_coords(points)

        # Get spherical harmonics coefficients through trilinear interpolation
        sh_coeffs = trilinear_interpolation(self.sh_coeffs, voxel_coords, has_channels=True)

        return sh_coeffs

    def get_occupied_voxel_count(self) -> int:
        """Get number of occupied voxels.

        Returns:
            Number of occupied voxels
        """
        return self.occupancy_grid.sum().item()

    def update_resolution(self, new_resolution: tuple[int, int, int]) -> None:
        """Update voxel grid resolution.

        Args:
            new_resolution: New grid resolution (X, Y, Z)
        """
        self.resolution = new_resolution
        self.voxel_size = (self.scene_bounds[3:] - self.scene_bounds[:3]) / torch.tensor(
            new_resolution
        )
        self.min_coords = self.scene_bounds[:3]
        self.occupancy_grid = torch.zeros(new_resolution, dtype=torch.bool, device=self.device)

    def prune_voxels(self, threshold: float = 0.01) -> None:
        """Prune low-density voxels.

        Args:
            threshold: Density threshold below which voxels are pruned
        """
        with torch.no_grad():
            mask = self.density > threshold
            self.density.data *= mask
            self.sh_coeffs.data *= mask.unsqueeze(-1).unsqueeze(-1)
            self.occupancy_grid.copy_(mask)


class VolumetricRenderer:
    """Volumetric renderer for Plenoxels.

    Features:
    - Adaptive sampling based on density distribution
    - Efficient batch processing with modern PyTorch features
    - Early ray termination
    - CUDA acceleration with CPU fallback
    - Automatic mixed precision support
    """

    def __init__(self, config: PlenoxelConfig):
        """Initialize renderer.

        Args:
            config: Renderer configuration
        """
        self.config = config
        self.device = config.device
        self.training = True  # Default to training mode
        self.use_cuda = config.use_cuda and CUDA_AVAILABLE and CUDA_EXTENSION_AVAILABLE

    def train(self, mode: bool = True):
        """Set training mode.

        Args:
            mode: Whether to set training mode
        """
        self.training = mode
        return self

    def eval(self):
        """Set evaluation mode."""
        return self.train(False)

    @torch.jit.export
    def render_rays(
        self,
        voxel_grid: VoxelGrid,
        rays_o: torch.Tensor,
        rays_d: torch.Tensor,
        num_samples: int,
        near: float = 2.0,
        far: float = 6.0,
        use_adaptive_sampling: bool = True,
        chunk_size: int = 4096,
    ) -> dict[str, torch.Tensor]:
        """Render rays through a voxel grid.

        Args:
            voxel_grid: Voxel grid to render
            rays_o: Ray origins [..., 3]
            rays_d: Ray directions [..., 3]
            num_samples: Number of samples per ray
            near: Near plane distance
            far: Far plane distance
            use_adaptive_sampling: Whether to use adaptive sampling
            chunk_size: Maximum number of rays to process at once

        Returns:
            Dictionary containing:
                - rgb: Rendered colors [..., 3]
                - depth: Rendered depths [..., 1]
                - weights: Ray weights [..., num_samples]
                - transmittance: Ray transmittance [..., num_samples]um_samples]

        Raises:
            ValueError: If inputs have invalid dimensions or parameters
        """
        # Validate inputs
        if rays_o.shape[-1] != 3:
            raise ValueError(f"Expected rays_o to have shape [..., 3], got {rays_o.shape}")
        if rays_d.shape[-1] != 3:
            raise ValueError(f"Expected rays_d to have shape [..., 3], got {rays_d.shape}")
        if rays_o.shape[:-1] != rays_d.shape[:-1]:
            raise ValueError(
                f"Expected rays_o and rays_d to have matching batch dimensions, "
                f"got {rays_o.shape[:-1]} and {rays_d.shape[:-1]}"
            )
        if num_samples <= 0:
            raise ValueError(f"Expected num_samples > 0, got {num_samples}")
        if near >= far:
            raise ValueError(f"Expected near < far, got near={near}, far={far}")

        # Move inputs to device and ensure contiguous memory
        rays_o = rays_o.to(self.device, non_blocking=True).contiguous()
        rays_d = rays_d.to(self.device, non_blocking=True).contiguous()

        # Normalize ray directions
        rays_d = F.normalize(rays_d, dim=-1)

        # Process rays in chunks to avoid OOM
        outputs = {}
        batch_shape = rays_o.shape[:-1]
        num_rays = np.prod(batch_shape)

        for i in range(0, num_rays, chunk_size):
            # Extract chunk
            chunk_rays_o = rays_o.view(-1, 3)[i : i + chunk_size]
            chunk_rays_d = rays_d.view(-1, 3)[i : i + chunk_size]

            # Sample points along rays
            points, deltas = self.sample_points_along_rays(
                chunk_rays_o,
                chunk_rays_d,
                num_samples,
                near,
                far,
                use_adaptive_sampling,
            )  # [chunk_size, num_samples, 3], [chunk_size, num_samples]

            # Get density and color at sampled points
            density = voxel_grid.get_density(points)  # [chunk_size, num_samples, 1]
            color = voxel_grid.get_color(
                points, chunk_rays_d.unsqueeze(1)
            )  # [chunk_size, num_samples, 3]

            # Compute weights for volume rendering
            weights = self._compute_weights(density, deltas)  # [chunk_size, num_samples]

            # Compute final outputs
            rgb = torch.sum(weights[..., None] * color, dim=-2)  # [chunk_size, 3]
            depth = torch.sum(weights * points[..., -1], dim=-1, keepdim=True)  # [chunk_size, 1]

            # Store chunk outputs
            if not outputs:
                # Initialize output tensors
                outputs = {
                    "rgb": torch.zeros(num_rays, 3, device=self.device),
                    "depth": torch.zeros(num_rays, 1, device=self.device),
                    "weights": torch.zeros(num_rays, num_samples, device=self.device),
                    "transmittance": torch.zeros(num_rays, num_samples, device=self.device),
                }

            # Update outputs
            outputs["rgb"][i : i + chunk_size] = rgb
            outputs["depth"][i : i + chunk_size] = depth
            outputs["weights"][i : i + chunk_size] = weights
            outputs["transmittance"][i : i + chunk_size] = torch.exp(
                -torch.cumsum(density.squeeze(-1) * deltas, dim=-1)
            )

        # Reshape outputs back to original batch dimensions
        outputs = {k: v.view(*batch_shape, *v.shape[1:]) for k, v in outputs.items()}

        return outputs

    def _render_rays_cpu(
        self,
        voxel_grid: VoxelGrid,
        rays_o: torch.Tensor,
        rays_d: torch.Tensor,
        num_samples: int,
        near: float,
        far: float,
        use_adaptive_sampling: bool,
        chunk_size: int,
    ) -> dict[str, torch.Tensor]:
        """CPU implementation of ray rendering."""
        # Sample points along rays
        points, deltas = self.sample_points_along_rays(
            rays_o, rays_d, num_samples, near, far, use_adaptive_sampling
        )

        # Get density and color at sampled points
        density = voxel_grid.get_density(points)
        colors = voxel_grid.get_color(points, rays_d)

        # Compute alpha compositing weights
        weights = self._compute_weights(density, deltas)

        # Compute final color and depth
        rgb = torch.sum(weights[..., None] * colors, dim=-2)
        depth = torch.sum(weights * points[..., -1], dim=-1)

        return {
            "rgb": rgb,
            "depth": depth,
            "weights": weights,
            "transmittance": torch.exp(-torch.cumsum(density * deltas, dim=-1)),
        }

    def ray_grid_intersect(
        self, rays_o: torch.Tensor, rays_d: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute ray-grid intersections.

        Args:
            rays_o: Ray origins [..., 3]
            rays_d: Ray directions [..., 3]

        Returns:
            Tuple of:
                - Entry points [..., 3]
                - Exit points [..., 3]
        """
        # Move inputs to device
        rays_o = rays_o.to(self.device)
        rays_d = rays_d.to(self.device)

        # Use CUDA implementation if available
        if self.use_cuda and hasattr(plenoxels_cuda, "ray_grid_intersect"):
            return plenoxels_cuda.ray_grid_intersect(rays_o, rays_d)

        # Fallback to CPU implementation
        return self._ray_grid_intersect_cpu(rays_o, rays_d)

    def _ray_grid_intersect_cpu(
        self, rays_o: torch.Tensor, rays_d: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """CPU implementation of ray-grid intersection."""
        # Compute intersections with axis-aligned bounding box
        inv_rays_d = 1.0 / (rays_d + 1e-10)
        t0 = (self.config.scene_bounds[None, :3] - rays_o) * inv_rays_d
        t1 = (self.config.scene_bounds[None, 3:] - rays_o) * inv_rays_d

        # Get entry and exit points
        t_near = torch.max(torch.min(t0, t1), dim=-1)[0]
        t_far = torch.min(torch.max(t0, t1), dim=-1)[0]

        # Compute intersection points
        entry_points = rays_o + t_near[..., None] * rays_d
        exit_points = rays_o + t_far[..., None] * rays_d

        return entry_points, exit_points

    def sample_points_along_rays(
        self,
        rays_o: torch.Tensor,
        rays_d: torch.Tensor,
        num_samples: int,
        near: float,
        far: float,
        use_adaptive_sampling: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Sample points along rays.

        Args:
            rays_o: Ray origins [..., 3]
            rays_d: Ray directions [..., 3]
            num_samples: Number of samples per ray
            near: Near plane distance
            far: Far plane distance
            use_adaptive_sampling: Whether to use adaptive sampling

        Returns:
            Tuple of:
                - Sample points [..., num_samples, 3]
                - Sample deltas [..., num_samples]
        """
        # Move inputs to device and ensure they have the same shape
        rays_o = rays_o.to(self.device)
        rays_d = rays_d.to(self.device)

        # Ensure rays_o and rays_d have at least 2 dimensions
        if rays_o.dim() == 1:
            rays_o = rays_o.unsqueeze(0)
        if rays_d.dim() == 1:
            rays_d = rays_d.unsqueeze(0)

        # Generate sampling points
        t = torch.linspace(0, 1, num_samples, device=self.device)
        if use_adaptive_sampling and self.training:
            # Add random offset for training
            t = t + torch.rand_like(t) * (1 / num_samples)

        # Scale to near-far range
        t = near + (far - near) * t

        # Expand t to match batch dimensions
        for _ in range(rays_o.dim() - 1):
            t = t.unsqueeze(0)
        t = t.expand(*rays_o.shape[:-1], num_samples)

        # Compute sample points and deltas
        points = rays_o[..., None, :] + rays_d[..., None, :] * t[..., :, None]
        deltas = torch.cat(
            [
                t[..., 1:] - t[..., :-1],
                torch.tensor([1e10], device=self.device).expand(*t.shape[:-1], 1),
            ],
            dim=-1,
        )

        return points, deltas

    def _compute_weights(self, density: torch.Tensor, deltas: torch.Tensor) -> torch.Tensor:
        """Compute alpha compositing weights.

        Args:
            density: Density values [..., num_samples]
            deltas: Sample deltas [..., num_samples]

        Returns:
            Alpha compositing weights [..., num_samples]
        """
        # Compute alpha values
        alpha = 1.0 - torch.exp(-density * deltas)

        # Compute transmittance
        transmittance = torch.cumprod(
            torch.cat(
                [
                    torch.ones_like(alpha[..., :1]),
                    1.0 - alpha[..., :-1] + 1e-10,
                ],
                dim=-1,
            ),
            dim=-1,
        )

        # Compute weights
        weights = alpha * transmittance

        return weights


class PlenoxelModel(nn.Module):
    """Plenoxels model implementation.

    This model represents a scene using a sparse voxel grid with spherical harmonics
    coefficients at each voxel. The model can be trained to learn a volumetric
    representation of a scene from a set of images.
    """

    def __init__(self, config: PlenoxelConfig):
        """Initialize model.

        Args:
            config: Model configuration
        """
        super().__init__()
        self.config = config

        # Initialize voxel grid
        self.voxel_grid = VoxelGrid(
            resolution=config.grid_resolution,
            scene_bounds=config.scene_bounds,
            num_sh_coeffs=(config.sh_degree + 1) ** 2,
            device=config.device,
        )

        # Initialize volumetric renderer
        self.renderer = VolumetricRenderer(config)

        # Initialize scaler for automatic mixed precision
        self.scaler = GradScaler()

    def forward(
        self,
        rays_o: torch.Tensor,
        rays_d: torch.Tensor,
        chunk_size: Optional[int] = None,
    ) -> dict[str, torch.Tensor]:
        """Forward pass.

        Args:
            rays_o: Ray origins [N, 3]
            rays_d: Ray directions [N, 3]
            chunk_size: Optional chunk size for memory efficiency

        Returns:
            Dictionary containing:
                - rgb: Rendered colors [N, 3]
                - depth: Rendered depths [N]
                - weights: Ray weights [N, num_samples]
                - points: Sampled points [N, num_samples, 3]
        """
        # Process rays in chunks if needed
        if chunk_size is not None and rays_o.shape[0] > chunk_size:
            outputs_chunks = []
            for i in range(0, rays_o.shape[0], chunk_size):
                chunk_rays_o = rays_o[i : i + chunk_size]
                chunk_rays_d = rays_d[i : i + chunk_size]
                outputs_chunks.append(self(chunk_rays_o, chunk_rays_d))
            return {
                k: torch.cat([chunk[k] for chunk in outputs_chunks], dim=0)
                for k in outputs_chunks[0].keys()
            }

        # Sample points along rays
        points, t_vals = self.renderer.sample_points_along_rays(
            rays_o,
            rays_d,
            num_samples=self.config.num_samples,
            near=self.config.near_plane,
            far=self.config.far_plane,
            use_adaptive_sampling=self.training,
        )

        # Get density and color at sampled points
        density = self.voxel_grid.get_density(points)  # [N, num_samples, 1]
        color = self.voxel_grid.get_color(points, rays_d.unsqueeze(1))  # [N, num_samples, 3]

        # Compute weights for volume rendering
        weights = self.renderer._compute_weights(density, t_vals)  # [N, num_samples]

        # Compute final outputs
        rgb = torch.sum(weights[..., None] * color, dim=-2)  # [N, 3]
        depth = torch.sum(weights * points[..., -1], dim=-1, keepdim=True)  # [N, 1]

        return {
            "rgb": rgb,
            "depth": depth,
            "weights": weights,
            "points": points,
        }

    def get_occupancy_stats(self) -> dict[str, float]:
        """Get occupancy statistics of the voxel grid.

        Returns:
            Dictionary containing:
                - total_voxels: Total number of voxels in the grid
                - occupied_voxels: Number of occupied voxels
                - sparsity: Ratio of occupied voxels to total voxels
                - memory_mb: Estimated memory usage in MB
        """
        total_voxels = torch.prod(torch.tensor(self.config.grid_resolution)).item()
        occupied_voxels = self.voxel_grid.get_occupied_voxel_count()
        sparsity = occupied_voxels / total_voxels

        # Estimate memory usage (rough approximation)
        bytes_per_voxel = (3 * (self.config.sh_degree + 1) ** 2) * 4  # 4 bytes per float32
        memory_mb = (occupied_voxels * bytes_per_voxel) / (1024 * 1024)

        return {
            "total_voxels": total_voxels,
            "occupied_voxels": occupied_voxels,
            "sparsity": sparsity,
            "memory_mb": memory_mb,
        }

    def update_resolution(self, new_resolution: tuple[int, int, int]) -> None:
        """Update voxel grid resolution.

        Args:
            new_resolution: New grid resolution (X, Y, Z)
        """
        self.voxel_grid.update_resolution(new_resolution)

    def prune_voxels(self, threshold: float = 0.01) -> None:
        """Prune low-density voxels.

        Args:
            threshold: Density threshold below which voxels are pruned
        """
        self.voxel_grid.prune_voxels(threshold)

    def save_checkpoint(
        self, path: str | Path, optimizer: torch.optim.Optimizer | None = None
    ) -> None:
        """Save model checkpoint.

        Args:
            path: Path to save checkpoint
            optimizer: Optional optimizer to save state
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        checkpoint = {
            "model_state": self.state_dict(),
            "config": self.config,
            "scaler": self.scaler.state_dict(),  # Save AMP scaler state
        }

        if optimizer is not None:
            checkpoint["optimizer_state"] = optimizer.state_dict()

        torch.save(checkpoint, path)

    @classmethod
    def load_checkpoint(
        cls, path: str | Path, map_location: Optional[torch.device] = None
    ) -> tuple["PlenoxelModel", Optional[Dict]]:
        """Load model from checkpoint.

        Args:
            path: Path to checkpoint
            map_location: Optional device to load checkpoint to

        Returns:
            Tuple of:
                - Loaded model
                - Optional optimizer state dict
        """
        checkpoint = torch.load(path, map_location=map_location)

        model = cls(checkpoint["config"])
        model.load_state_dict(checkpoint["model_state"])

        optimizer_state = checkpoint.get("optimizer_state")

        return model, optimizer_state


class PlenoxelLoss:
    """Modern loss functions for Plenoxels training.

    Features:
    - Modern loss function implementations
    - Efficient loss computation with AMP support
    - Flexible loss weighting and scheduling
    """

    def __init__(self, config: PlenoxelConfig):
        """Initialize loss functions.

        Args:
            config: Model configuration
        """
        self.config = config
        self.device = config.device

        # Initialize loss functions with modern implementations
        self._setup_criterion()

    def compute_rgb_loss(
        self, pred_rgb: torch.Tensor, gt_rgb: torch.Tensor, loss_type: str = "mse"
    ) -> torch.Tensor:
        """Compute RGB reconstruction loss.

        Args:
            pred_rgb: Predicted RGB values [..., 3]
            gt_rgb: Ground truth RGB values [..., 3]
            loss_type: Type of loss function

        Returns:
            RGB loss value
        """
        return self.rgb_criterion(pred_rgb, gt_rgb)

    def compute_depth_loss(
        self, pred_depth: torch.Tensor, gt_depth: torch.Tensor, mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        """Compute depth reconstruction loss.

        Args:
            pred_depth: Predicted depth values [...]
            gt_depth: Ground truth depth values [...]
            mask: Optional validity mask [...]

        Returns:
            Depth loss value
        """
        if mask is not None:
            pred_depth = pred_depth[mask]
            gt_depth = gt_depth[mask]

        if pred_depth.numel() == 0:
            return torch.tensor(0.0, device=self.device)

        return self.depth_criterion(pred_depth, gt_depth)

    def compute_normal_loss(
        self,
        pred_normals: torch.Tensor,
        gt_normals: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Compute normal reconstruction loss.

        Args:
            pred_normals: Predicted normal vectors [..., 3]
            gt_normals: Ground truth normal vectors [..., 3]
            mask: Optional validity mask [...]

        Returns:
            Normal loss value
        """
        if mask is not None:
            pred_normals = pred_normals[mask]
            gt_normals = gt_normals[mask]

        if pred_normals.numel() == 0:
            return torch.tensor(0.0, device=self.device)

        # Normalize vectors efficiently using F.normalize
        pred_normals = F.normalize(pred_normals, dim=-1, eps=1e-8)
        gt_normals = F.normalize(gt_normals, dim=-1, eps=1e-8)

        # Compute cosine similarity efficiently
        cos_sim = torch.sum(pred_normals * gt_normals, dim=-1)

        # Clamp for numerical stability
        cos_sim = torch.clamp(cos_sim, min=-1.0, max=1.0)

        return torch.mean(1 - cos_sim)

    def __call__(
        self,
        outputs: dict[str, torch.Tensor],
        batch: dict[str, torch.Tensor],
        model: PlenoxelModel,
        global_step: int,
    ) -> dict[str, torch.Tensor]:
        """Compute all losses efficiently.

        Args:
            outputs: Model outputs
            batch: Ground truth batch
            model: Plenoxels model
            global_step: Current training step

        Returns:
            Dictionary containing all loss terms and total loss
        """
        # Compute RGB loss with modern implementation
        rgb_loss = self.compute_rgb_loss(
            outputs["rgb"], batch["rgb"], loss_type=self.config.rgb_loss_type
        )

        # Initialize loss dict with modern type hints
        losses: dict[str, torch.Tensor] = {"rgb_loss": rgb_loss}

        # Add depth loss if available
        if "depth" in outputs and "depth" in batch:
            depth_loss = self.compute_depth_loss(
                outputs["depth"], batch["depth"], batch.get("depth_mask")
            )
            losses["depth_loss"] = depth_loss * self.config.depth_lambda

        # Add normal loss if available
        if "normals" in outputs and "normals" in batch:
            normal_loss = self.compute_normal_loss(
                outputs["normals"], batch["normals"], batch.get("normal_mask")
            )
            losses["normal_loss"] = normal_loss * self.config.normal_lambda

        # Add regularization losses efficiently
        reg_losses = self.compute_regularization_loss(model, global_step)
        losses.update(reg_losses)

        # Compute total loss efficiently
        total_loss = sum(losses.values())
        losses["total_loss"] = total_loss

        return losses

    @staticmethod
    def create_loss_function(loss_type: str = "mse", reduction: str = "mean") -> nn.Module:
        """Create a modern loss function with specified configuration.

        Args:
            loss_type: Type of loss function ('mse', 'l1', 'huber', etc.)
            reduction: Reduction method ('mean', 'sum', 'none')

        Returns:
            Configured loss function
        """
        loss_map = {
            "mse": lambda: nn.MSELoss(reduction=reduction),
            "l1": lambda: nn.L1Loss(reduction=reduction),
            "huber": lambda: nn.HuberLoss(reduction=reduction, delta=1.0),
            "smooth_l1": lambda: nn.SmoothL1Loss(reduction=reduction, beta=1.0),
            "ssim": lambda: SSIMLoss(reduction=reduction),  # Custom SSIM loss
            "perceptual": lambda: PerceptualLoss(reduction=reduction),  # Custom perceptual loss
        }

        if loss_type not in loss_map:
            raise ValueError(f"Unsupported loss type: {loss_type}")

        return loss_map[loss_type]()

    def _setup_criterion(self) -> None:
        """Setup training criterion with modern configuration."""
        # RGB loss
        self.rgb_criterion = self.create_loss_function(
            loss_type=self.config.rgb_loss_type, reduction=self.config.loss_reduction
        )

        # Depth loss
        self.depth_criterion = self.create_loss_function(
            loss_type=self.config.depth_loss_type, reduction=self.config.loss_reduction
        )

        # Normal loss
        self.normal_criterion = self.create_loss_function(
            loss_type=self.config.normal_loss_type, reduction=self.config.loss_reduction
        )

        # Density loss
        self.density_criterion = self.create_loss_function(
            loss_type=self.config.density_loss_type, reduction=self.config.loss_reduction
        )

        # Total variation loss
        self.tv_loss = self.config.tv_lambda

        # L1 sparsity loss
        self.l1_loss = self.config.l1_lambda

    def compute_regularization_loss(
        self, model: PlenoxelModel, global_step: int
    ) -> dict[str, torch.Tensor]:
        """Compute regularization losses.

        Args:
            model: Plenoxels model
            global_step: Current training step

        Returns:
            Dictionary containing regularization losses
        """
        # Total variation loss
        tv_loss = model.voxel_grid.total_variation_loss()

        # L1 sparsity loss
        l1_loss = model.voxel_grid.l1_loss()

        # Beta annealing for TV loss
        if self.config.use_coarse_to_fine:
            tv_weight = self.config.tv_lambda * (0.1 + 0.9 * min(1.0, global_step / 1000))
        else:
            tv_weight = self.config.tv_lambda

        return {"tv_loss": tv_loss * tv_weight, "l1_loss": l1_loss * self.config.l1_lambda}


class PlenoxelMetrics:
    """Metrics tracker for Plenoxels training.

    Features:
    - Automatic metric aggregation
    - Moving average computation
    - Periodic logging
    - TensorBoard support
    """

    def __init__(self, config: PlenoxelConfig):
        """Initialize metrics tracker.

        Args:
            config: Metrics configuration
        """
        self.config = config
        self.metrics = {}
        self.step_metrics = {}
        self.best_metrics = {}

        # Initialize TensorBoard writer if available
        try:
            from torch.utils.tensorboard import SummaryWriter

            self.writer = SummaryWriter(config.log_dir)
            self.use_tensorboard = True
        except ImportError:
            logger.warning("TensorBoard not available, logging disabled")
            self.use_tensorboard = False

    def update(self, metrics: dict[str, float], step: int):
        """Update metrics with new values.

        Args:
            metrics: Dictionary of metric values
            step: Current training step
        """
        # Update step metrics
        self.step_metrics = metrics

        # Update running averages
        for name, value in metrics.items():
            if name not in self.metrics:
                self.metrics[name] = []
            self.metrics[name].append(value)

            # Update best metrics
            if name not in self.best_metrics or value < self.best_metrics[name]:
                self.best_metrics[name] = value

        # Log to TensorBoard
        if self.use_tensorboard:
            for name, value in metrics.items():
                self.writer.add_scalar(f"train/{name}", value, step)

    def get_average(self, name: str, window_size: int = 100) -> float:
        """Get moving average of a metric.

        Args:
            name: Metric name
            window_size: Window size for moving average

        Returns:
            Moving average value
        """
        if name not in self.metrics:
            return 0.0
        values = self.metrics[name][-window_size:]
        return sum(values) / len(values)

    def get_best(self, name: str) -> float:
        """Get best value of a metric.

        Args:
            name: Metric name

        Returns:
            Best metric value
        """
        return self.best_metrics.get(name, float("inf"))

    def reset(self):
        """Reset all metrics."""
        self.metrics.clear()
        self.step_metrics.clear()
        self.best_metrics.clear()


class PlenoxelScheduler:
    """Learning rate scheduler for Plenoxels training.

    Features:
    - Multiple scheduling strategies
    - Warm-up phase
    - Coarse-to-fine scheduling
    - Automatic adjustment based on metrics
    """

    def __init__(self, optimizer: torch.optim.Optimizer, config: PlenoxelConfig, num_steps: int):
        """Initialize scheduler.

        Args:
            optimizer: PyTorch optimizer
            config: Scheduler configuration
            num_steps: Total number of training steps
        """
        self.optimizer = optimizer
        self.config = config
        self.num_steps = num_steps
        self.current_step = 0

        # Initialize learning rates
        self.initial_lr = config.learning_rate
        self.min_lr = config.learning_rate * 0.01

        # Warm-up parameters
        self.warmup_steps = min(1000, num_steps // 10)

        # Coarse-to-fine parameters
        if config.use_coarse_to_fine:
            self.stage_steps = [steps for steps in config.coarse_epochs]
            self.stage_lrs = [self.initial_lr * (0.5**i) for i in range(len(self.stage_steps))]

    def step(self, metrics: Optional[dict[str, float]] = None):
        """Update learning rate.

        Args:
            metrics: Optional metrics for adaptive scheduling
        """
        self.current_step += 1

        # Compute learning rate
        if self.current_step < self.warmup_steps:
            # Warm-up phase
            lr = self.initial_lr * (self.current_step / self.warmup_steps)
        elif self.config.use_coarse_to_fine:
            # Coarse-to-fine phase
            current_stage = 0
            steps_sum = 0
            for i, steps in enumerate(self.stage_steps):
                steps_sum += steps
                if self.current_step < steps_sum:
                    current_stage = i
                    break
            lr = self.stage_lrs[current_stage]
        else:
            # Cosine decay
            progress = (self.current_step - self.warmup_steps) / (
                self.num_steps - self.warmup_steps
            )
            lr = self.min_lr + 0.5 * (self.initial_lr - self.min_lr) * (
                1 + math.cos(math.pi * progress)
            )

        # Update optimizer
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = lr

        return lr


class PlenoxelTrainer:
    """Trainer for Plenoxels model.

    Features:
    - Automatic training loop
    - Validation and testing
    - Checkpoint management
    - Progress tracking
    - Multi-GPU support
    """

    def __init__(
        self,
        model: PlenoxelModel,
        train_dataloader: torch.utils.data.DataLoader,
        val_dataloader: Optional[torch.utils.data.DataLoader] = None,
        config: Optional[PlenoxelConfig] = None,
    ):
        """Initialize trainer.

        Args:
            model: Plenoxels model
            train_dataloader: Training data loader
            val_dataloader: Optional validation data loader
            config: Optional trainer configuration
        """
        self.model = model
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.config = config or model.config

        # Initialize components
        self.cuda_manager = CUDAManager(self.config)
        self.device = self.cuda_manager.get_device()

        # Move model to device
        self.model = self.model.to(self.device)

        # Initialize optimizer
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
        )

        # Initialize loss function
        self.criterion = PlenoxelLoss(self.config)

        # Initialize metrics tracker
        self.metrics = PlenoxelMetrics(self.config)

        # Initialize scheduler
        self.scheduler = PlenoxelScheduler(
            self.optimizer, self.config, len(train_dataloader) * self.config.num_epochs
        )

        # Initialize state
        self.current_epoch = 0
        self.global_step = 0
        self.best_val_loss = float("inf")

    def train_epoch(self, epoch: int | None = None) -> dict[str, float]:
        """Train for one epoch.

        Args:
            epoch: Optional epoch number for logging

        Returns:
            Dictionary containing training metrics
        """
        self.model.train()
        epoch_metrics = []

        for batch in self.train_dataloader:
            # Move batch to device
            batch = {k: v.to(self.device) for k, v in batch.items()}

            # Forward pass
            outputs = self.model(
                batch["rays_o"], batch["rays_d"], chunk_size=self.config.batch_size
            )

            # Compute losses
            losses = self.criterion(outputs, batch, self.model, self.global_step)

            # Optimization step
            self.optimizer.zero_grad()
            losses["total_loss"].backward()
            self.optimizer.step()

            # Update learning rate
            lr = self.scheduler.step()

            # Update metrics
            metrics = {
                name: value.item() if torch.is_tensor(value) else value
                for name, value in losses.items()
            }
            metrics["learning_rate"] = lr
            epoch_metrics.append(metrics)
            self.metrics.update(metrics, self.global_step)

            # Increment step counter
            self.global_step += 1

            # Log progress
            if self.global_step % self.config.log_every == 0:
                self._log_progress("train", epoch=epoch)

        # Compute average metrics for the epoch
        avg_metrics = {}
        for key in epoch_metrics[0].keys():
            avg_metrics[key] = sum(m[key] for m in epoch_metrics) / len(epoch_metrics)

        return avg_metrics

    @torch.no_grad()
    def validate(self):
        """Perform validation."""
        if not self.val_dataloader:
            return

        self.model.eval()
        val_metrics = []

        for batch in self.val_dataloader:
            # Move batch to device
            batch = {k: v.to(self.device) for k, v in batch.items()}

            # Forward pass
            outputs = self.model(
                batch["rays_o"], batch["rays_d"], chunk_size=self.config.batch_size * 2
            )

            # Compute metrics
            metrics = self.model.eval_step(batch)
            val_metrics.append(metrics)

        # Average metrics
        avg_metrics = {
            name: sum(m[name] for m in val_metrics) / len(val_metrics)
            for name in val_metrics[0].keys()
        }

        # Update best metrics
        if avg_metrics["rgb_loss"] < self.best_val_loss:
            self.best_val_loss = avg_metrics["rgb_loss"]
            self.save_checkpoint("best.pt")

        # Log progress
        self._log_progress("val", avg_metrics)

    def train(self, num_epochs: int):
        """Train for multiple epochs.

        Args:
            num_epochs: Number of epochs to train
        """
        for epoch in range(num_epochs):
            self.current_epoch = epoch

            # Training
            self.train_epoch()

            # Validation
            self.validate()

            # Save checkpoint
            if (epoch + 1) % self.config.save_every == 0:
                self.save_checkpoint(f"epoch_{epoch+1}.pt")

    def save_checkpoint(self, filename: str):
        """Save checkpoint.

        Args:
            filename: Checkpoint filename
        """
        checkpoint_dir = Path(self.config.checkpoint_dir)
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        self.model.save_checkpoint(checkpoint_dir / filename, self.optimizer)

    def load_checkpoint(self, filename: str):
        """Load checkpoint.

        Args:
            filename: Checkpoint filename
        """
        checkpoint_dir = Path(self.config.checkpoint_dir)
        model, optimizer_state = PlenoxelModel.load_checkpoint(
            checkpoint_dir / filename, self.device
        )

        self.model = model.to(self.device)
        if optimizer_state:
            self.optimizer.load_state_dict(optimizer_state)

    def _log_progress(
        self, phase: str, metrics: Optional[dict[str, float]] = None, epoch: Optional[int] = None
    ):
        """Log training progress.

        Args:
            phase: Training phase ('train' or 'val')
            metrics: Optional metrics to log
            epoch: Optional epoch number for logging
        """
        metrics = metrics or self.metrics.step_metrics

        # Create progress message
        msg = (
            f"[{phase.upper()}] Epoch {epoch + 1 if epoch is not None else self.current_epoch + 1}"
        )
        if phase == "train":
            msg += f" Step {self.global_step}"
        msg += " ||"

        # Add metrics
        for name, value in metrics.items():
            msg += f" {name}: {value:.4f} |"

        logger.info(msg)

    @torch.no_grad()
    def validate_epoch(self, epoch: int | None = None) -> dict[str, float]:
        """Perform validation for one epoch.

        Args:
            epoch: Optional epoch number for logging

        Returns:
            Dictionary containing validation metrics
        """
        if not self.val_dataloader:
            return {}

        self.model.eval()
        val_metrics = []

        for batch in self.val_dataloader:
            # Move batch to device
            batch = {k: v.to(self.device) for k, v in batch.items()}

            # Get image dimensions
            B, H, W = batch["rays_o"].shape[:3]  # [B, H, W, 3]

            # Process each image in the batch
            batch_metrics = []
            for b in range(B):
                # Get single image rays
                rays_o = batch["rays_o"][b]  # [H, W, 3]
                rays_d = batch["rays_d"][b]  # [H, W, 3]
                target_rgb = batch["rgb"][b]  # [H, W, 3]

                # Flatten rays for processing
                rays_o_flat = rays_o.reshape(-1, 3)
                rays_d_flat = rays_d.reshape(-1, 3)

                # Process in chunks to avoid memory issues
                chunk_size = self.config.batch_size * 2
                rgb_chunks = []

                for i in range(0, len(rays_o_flat), chunk_size):
                    # Forward pass with automatic mixed precision
                    with autocast(
                        device_type="cuda" if torch.cuda.is_available() else "cpu"
                    ):
                        outputs = self.model(
                            rays_o_flat[i : i + chunk_size],
                            rays_d_flat[i : i + chunk_size],
                            chunk_size=chunk_size,
                        )
                        rgb_chunks.append(outputs["rgb"])

                # Combine chunks and reshape back to image dimensions
                rgb_pred = torch.cat(rgb_chunks, dim=0).reshape(H, W, 3)

                # Compute metrics
                rgb_loss = F.mse_loss(rgb_pred, target_rgb)
                psnr = -10 * torch.log10(rgb_loss)

                batch_metrics.append({"rgb_loss": rgb_loss.item(), "psnr": psnr.item()})

            # Average metrics across batch
            metrics = {}
            for key in batch_metrics[0].keys():
                metrics[key] = sum(m[key] for m in batch_metrics) / len(batch_metrics)
            val_metrics.append(metrics)

            # Log progress
            if len(val_metrics) % self.config.log_every == 0:
                self._log_progress("val", metrics=metrics, epoch=epoch)

        # Compute average metrics
        avg_metrics = {}
        for key in val_metrics[0].keys():
            avg_metrics[f"val_{key}"] = sum(m[key] for m in val_metrics) / len(val_metrics)

        # Update best validation loss
        if avg_metrics["val_rgb_loss"] < self.best_val_loss:
            self.best_val_loss = avg_metrics["val_rgb_loss"]
            self.save_checkpoint("best.pt")

        return avg_metrics


class PlenoxelVisualizer:
    """Visualization tools for Plenoxels.

    Features:
    - Training metrics visualization
    - Model architecture visualization
    - Memory usage plots
    - Performance analysis plots
    """

    def __init__(
        self,
        config: PlenoxelConfig,
        profiler: Optional[PlenoxelProfiler] = None,
        auto_tuner: Optional[PlenoxelAutoTuner] = None,
    ):
        """Initialize visualizer.

        Args:
            config: Visualization configuration
            profiler: Optional performance profiler
            auto_tuner: Optional auto-tuner
        """
        self.config = config
        self.profiler = profiler
        self.auto_tuner = auto_tuner

        # Initialize matplotlib style
        plt.style.use("seaborn")

    def plot_training_metrics(
        self, metrics_history: list[dict[str, float]], save_path: Optional[str] = None
    ):
        """Plot training metrics history.

        Args:
            metrics_history: List of metric dictionaries
            save_path: Optional path to save plot
        """
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle("Training Metrics", fontsize=16)

        # Plot loss
        if "loss" in metrics_history[0]:
            losses = [m["loss"] for m in metrics_history]
            axes[0, 0].plot(losses, label="Loss")
            axes[0, 0].set_title("Loss History")
            axes[0, 0].set_xlabel("Iteration")
            axes[0, 0].set_ylabel("Loss")
            axes[0, 0].grid(True)

        # Plot learning rate
        if "learning_rate" in metrics_history[0]:
            lrs = [m["learning_rate"] for m in metrics_history]
            axes[0, 1].plot(lrs, label="Learning Rate")
            axes[0, 1].set_title("Learning Rate History")
            axes[0, 1].set_xlabel("Iteration")
            axes[0, 1].set_ylabel("Learning Rate")
            axes[0, 1].grid(True)

        # Plot PSNR if available
        if "psnr" in metrics_history[0]:
            psnr = [m["psnr"] for m in metrics_history]
            axes[1, 0].plot(psnr, label="PSNR")
            axes[1, 0].set_title("PSNR History")
            axes[1, 0].set_xlabel("Iteration")
            axes[1, 0].set_ylabel("PSNR")
            axes[1, 0].grid(True)

        # Plot SSIM if available
        if "ssim" in metrics_history[0]:
            ssim = [m["ssim"] for m in metrics_history]
            axes[1, 1].plot(ssim, label="SSIM")
            axes[1, 1].set_title("SSIM History")
            axes[1, 1].set_xlabel("Iteration")
            axes[1, 1].set_ylabel("SSIM")
            axes[1, 1].grid(True)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path)
        else:
            plt.show()

        plt.close()

    def plot_memory_usage(self, save_path: Optional[str] = None):
        """Plot memory usage statistics.

        Args:
            save_path: Optional path to save plot
        """
        if self.profiler is None:
            logger.warning("No profiler available for memory visualization")
            return

        memory_stats = self.profiler.memory_stats
        if not memory_stats:
            logger.warning("No memory statistics available")
            return

        fig, ax = plt.subplots(figsize=(10, 6))

        timestamps = [s["timestamp"] - memory_stats[0]["timestamp"] for s in memory_stats]
        allocated = [s["allocated"] / 1e9 for s in memory_stats]  # Convert to GB
        reserved = [s["reserved"] / 1e9 for s in memory_stats]

        ax.plot(timestamps, allocated, label="Allocated")
        ax.plot(timestamps, reserved, label="Reserved")

        ax.set_title("GPU Memory Usage Over Time")
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Memory (GB)")
        ax.grid(True)
        ax.legend()

        if save_path:
            plt.savefig(save_path)
        else:
            plt.show()

        plt.close()

    def plot_timing_breakdown(self, event_times: dict[str, float], save_path: Optional[str] = None):
        """Plot timing breakdown of different operations.

        Args:
            event_times: Dictionary of event names and times
            save_path: Optional path to save plot
        """
        if not event_times:
            logger.warning("No timing information available")
            return

        # Sort events by time
        sorted_events = sorted(event_times.items(), key=lambda x: x[1], reverse=True)

        names = [e[0] for e in sorted_events]
        times = [e[1] for e in sorted_events]

        fig, ax = plt.subplots(figsize=(10, 6))

        y_pos = range(len(names))
        ax.barh(y_pos, times)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(names)

        ax.set_title("Operation Timing Breakdown")
        ax.set_xlabel("Time (ms)")

        if save_path:
            plt.savefig(save_path)
        else:
            plt.show()

        plt.close()

    def plot_tuning_suggestions(self, save_path: Optional[str] = None):
        """Plot auto-tuning suggestions history.

        Args:
            save_path: Optional path to save plot
        """
        if self.auto_tuner is None:
            logger.warning("No auto-tuner available for visualization")
            return

        if not self.auto_tuner.history:
            logger.warning("No auto-tuning history available")
            return

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10))

        # Plot learning rate suggestions
        timestamps = [
            h["timestamp"] - self.auto_tuner.history[0]["timestamp"]
            for h in self.auto_tuner.history
        ]
        learning_rates = [h["metrics"].get("learning_rate", 0) for h in self.auto_tuner.history]

        ax1.plot(timestamps, learning_rates)
        ax1.set_title("Learning Rate History")
        ax1.set_xlabel("Time (s)")
        ax1.set_ylabel("Learning Rate")
        ax1.grid(True)

        # Plot batch size suggestions
        batch_sizes = [h["metrics"].get("batch_size", 0) for h in self.auto_tuner.history]

        ax2.plot(timestamps, batch_sizes)
        ax2.set_title("Batch Size History")
        ax2.set_xlabel("Time (s)")
        ax2.set_ylabel("Batch Size")
        ax2.grid(True)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path)
        else:
            plt.show()

        plt.close()


class PlenoxelProfiler:
    """Performance profiling for Plenoxels.

    Features:
    - Memory usage tracking
    - CUDA event timing
    - Operation profiling
    - Bottleneck detection
    """

    def __init__(self):
        """Initialize profiler."""
        self.memory_stats = []
        self.timing_stats = {}
        self.cuda_events = {}
        self.current_scope = None

    def start_event(self, name: str):
        """Start timing an event.

        Args:
            name: Event name
        """
        if torch.cuda.is_available():
            start_event = torch.cuda.Event(enable_timing=True)
            start_event.record()
            self.cuda_events[name] = {"start": start_event, "end": None}

    def end_event(self, name: str):
        """End timing an event.

        Args:
            name: Event name
        """
        if torch.cuda.is_available() and name in self.cuda_events:
            end_event = torch.cuda.Event(enable_timing=True)
            end_event.record()
            self.cuda_events[name]["end"] = end_event

    def get_event_time(self, name: str) -> float:
        """Get event execution time.

        Args:
            name: Event name

        Returns:
            Event time in milliseconds
        """
        if name in self.cuda_events:
            events = self.cuda_events[name]
            if events["start"] is not None and events["end"] is not None:
                torch.cuda.synchronize()
                return events["start"].elapsed_time(events["end"])
        return 0.0

    def track_memory(self):
        """Track current memory usage."""
        if torch.cuda.is_available():
            memory_allocated = torch.cuda.memory_allocated()
            memory_reserved = torch.cuda.memory_reserved()
            self.memory_stats.append(
                {
                    "allocated": memory_allocated,
                    "reserved": memory_reserved,
                    "timestamp": time.time(),
                }
            )

    def get_memory_stats(self) -> Dict:
        """Get memory usage statistics.

        Returns:
            Dictionary of memory statistics
        """
        if not self.memory_stats:
            return {}

        allocated = [stat["allocated"] for stat in self.memory_stats]
        reserved = [stat["reserved"] for stat in self.memory_stats]

        return {
            "peak_allocated": max(allocated),
            "peak_reserved": max(reserved),
            "avg_allocated": sum(allocated) / len(allocated),
            "avg_reserved": sum(reserved) / len(reserved),
        }

    def clear(self):
        """Clear profiling data."""
        self.memory_stats.clear()
        self.timing_stats.clear()
        self.cuda_events.clear()
