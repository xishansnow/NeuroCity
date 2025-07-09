from __future__ import annotations

"""
Instant NGP: Instant Neural Graphics Primitives with Multiresolution Hash Encoding.

This module implements the Instant NGP model as described in:
"Instant Neural Graphics Primitives with a Multiresolution Hash Encoding"
by Thomas Müller et al. (SIGGRAPH 2022)

Key components:
- Multiresolution hash encoding for efficient feature lookup
- Small MLP networks for fast inference
- CUDA-optimized hash table operations (fallback to PyTorch for compatibility)
- Support for NeRF and other neural graphics primitives
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from dataclasses import dataclass
from torch.amp import autocast, GradScaler
from pathlib import Path

# Type aliases for modern Python 3.10
Tensor = torch.Tensor
Device = torch.device | str
DType = torch.dtype
TensorDict = dict[str, Tensor]


@dataclass
class InstantNGPConfig:
    """Configuration for Instant-NGP model with modern features."""

    # Hash encoding settings
    num_levels: int = 16
    base_resolution: int = 16
    finest_resolution: int = 512
    log2_hashmap_size: int = 19
    feature_dim: int = 2

    # Network architecture
    hidden_dim: int = 64
    num_layers: int = 3
    geo_feature_dim: int = 15
    num_layers_color: int = 4
    hidden_dim_color: int = 64

    # Rendering settings
    num_samples: int = 128
    num_importance_samples: int = 64
    background_color: str = "white"
    near_plane: float = 0.1
    far_plane: float = 1000.0

    # Training settings
    learning_rate: float = 1e-2
    learning_rate_alpha: float = 1e-3
    learning_rate_decay_steps: int = 20000
    learning_rate_decay_mult: float = 0.1
    weight_decay: float = 0.0
    decay_step: int = 20000
    learning_rate_decay: float = 0.1

    # Training optimization
    use_amp: bool = True  # Automatic mixed precision
    grad_scaler_init_scale: float = 65536.0
    grad_scaler_growth_factor: float = 2.0
    grad_scaler_backoff_factor: float = 0.5
    grad_scaler_growth_interval: int = 2000

    # Memory optimization
    use_non_blocking: bool = True  # Non-blocking memory transfers
    set_grad_to_none: bool = True  # More efficient gradient clearing
    chunk_size: int = 8192  # Processing chunk size
    cache_grid_samples: bool = True  # Cache grid samples for faster training

    # Device settings
    default_device: str = "cuda"
    pin_memory: bool = True

    # Batch processing
    batch_size: int = 4096
    num_workers: int = 4

    # Checkpointing
    checkpoint_dir: str = "checkpoints"
    save_every_n_steps: int = 1000
    keep_last_n_checkpoints: int = 5

    def __post_init__(self):
        """Post-initialization validation and initialization."""
        # Validate settings
        assert self.num_levels > 0, "Number of levels must be positive"
        assert (
            self.base_resolution < self.finest_resolution
        ), "Base resolution must be less than finest resolution"

        # Initialize device
        self.device = torch.device(self.default_device if torch.cuda.is_available() else "cpu")

        # Initialize grad scaler for AMP
        if self.use_amp:
            self.grad_scaler = GradScaler(
                "cuda",
                init_scale=self.grad_scaler_init_scale,
                growth_factor=self.grad_scaler_growth_factor,
                backoff_factor=self.grad_scaler_backoff_factor,
                growth_interval=self.grad_scaler_growth_interval,
            )

        # Create checkpoint directory
        self.checkpoint_dir = Path(self.checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)


class HashEncoder(nn.Module):
    """Multiresolution hash encoding."""

    def __init__(self, config: InstantNGPConfig):
        super().__init__()
        self.config = config

        # Calculate encoding parameters
        self.num_levels = config.num_levels
        self.level_dim = config.feature_dim
        self.per_level_scale = 2.0  # Scale factor between levels
        self.base_resolution = config.base_resolution
        self.log2_hashmap_size = config.log2_hashmap_size
        self.desired_resolution = config.finest_resolution

        # Compute maximum resolution
        max_res = int(
            np.ceil(config.base_resolution * (self.per_level_scale ** (config.num_levels - 1)))
        )
        if max_res > config.finest_resolution:
            config.finest_resolution = max_res

        # Initialize hash tables for each level
        self.embeddings = nn.ModuleList()
        self.resolutions = []

        for i in range(self.num_levels):
            resolution = int(np.ceil(config.base_resolution * (self.per_level_scale**i)))
            params_in_level = min(resolution**3, 2**config.log2_hashmap_size)

            self.resolutions.append(resolution)
            embedding = nn.Embedding(params_in_level, config.feature_dim)

            # Xavier uniform initialization
            std = 1e-4
            nn.init.uniform_(embedding.weight, -std, std)

            self.embeddings.append(embedding)

        self.output_dim = self.num_levels * self.level_dim

    def hash_function(self, coords, resolution):
        """Simple hash function for grid coordinates."""
        # Use a simple hash function - in practice this would be optimized CUDA code
        primes = torch.tensor([1, 2654435761, 805459861], device=coords.device, dtype=torch.long)

        # Clamp coordinates to valid range
        coords = coords.clamp(0, resolution - 1).long()

        # Compute hash
        hash_val = torch.zeros(coords.shape[0], device=coords.device, dtype=torch.long)
        for i in range(3):
            hash_val ^= coords[:, i] * primes[i]

        return hash_val % self.embeddings[0].num_embeddings

    def get_grid_coordinates(self, positions, resolution):
        """Get grid coordinates and interpolation weights."""
        # Scale positions to grid resolution
        scaled_positions = (positions + 1) / 2 * (resolution - 1)  # [-1, 1] -> [0, res-1]

        # Get integer and fractional parts
        grid_coords = torch.floor(scaled_positions).long()
        weights = scaled_positions - grid_coords.float()

        return grid_coords, weights

    def trilinear_interpolation(self, features, weights):
        """Perform trilinear interpolation."""
        # features: [N, 8, D] - 8 corner features
        # weights: [N, 3] - interpolation weights

        # Unpack weights
        wx, wy, wz = weights.unbind(-1)

        # Expand weights for broadcasting with features
        wx = wx.unsqueeze(-1)  # [N, 1]
        wy = wy.unsqueeze(-1)  # [N, 1]
        wz = wz.unsqueeze(-1)  # [N, 1]

        # Bilinear interpolation in xy plane for z=0 and z=1
        c00 = features[:, 0] * (1 - wx) + features[:, 1] * wx  # (0, 0, 0) and (1, 0, 0)
        c01 = features[:, 2] * (1 - wx) + features[:, 3] * wx  # (0, 1, 0) and (1, 1, 0)
        c10 = features[:, 4] * (1 - wx) + features[:, 5] * wx  # (0, 0, 1) and (1, 0, 1)
        c11 = features[:, 6] * (1 - wx) + features[:, 7] * wx  # (0, 1, 1) and (1, 1, 1)

        # Interpolate in y direction
        c0 = c00 * (1 - wy) + c01 * wy  # z=0 plane
        c1 = c10 * (1 - wy) + c11 * wy  # z=1 plane

        # Interpolate in z direction
        result = c0 * (1 - wz) + c1 * wz

        return result

    def forward(self, positions):
        """
        Encode positions using multiresolution hash encoding.

        Args:
            positions: [N, 3] input positions in [-1, 1]

        Returns:
            encoded: [N, output_dim] encoded features
        """
        batch_size = positions.shape[0]
        device = positions.device

        # Collect features from all levels
        encoded_features = []

        for level_idx in range(self.num_levels):
            resolution = self.resolutions[level_idx]
            embedding = self.embeddings[level_idx]

            # Get grid coordinates and weights
            grid_coords, weights = self.get_grid_coordinates(positions, resolution)

            # Get 8 corner coordinates for trilinear interpolation
            corner_coords = []
            for dx in [0, 1]:
                for dy in [0, 1]:
                    for dz in [0, 1]:
                        corner = grid_coords + torch.tensor([dx, dy, dz], device=device)
                        corner = torch.clamp(corner, 0, resolution - 1)
                        corner_coords.append(corner)

            corner_coords = torch.stack(corner_coords, dim=1)  # [N, 8, 3]

            # Hash coordinates to get indices
            corner_indices = []
            for i in range(8):
                indices = self.hash_function(corner_coords[:, i], resolution)
                corner_indices.append(indices)

            corner_indices = torch.stack(corner_indices, dim=1)  # [N, 8]

            # Look up embeddings
            corner_features = embedding(corner_indices)  # [N, 8, level_dim]

            # Trilinear interpolation
            interpolated = self.trilinear_interpolation(corner_features, weights)

            encoded_features.append(interpolated)

        # Concatenate features from all levels
        encoded = torch.cat(encoded_features, dim=-1)
        return encoded


class SHEncoder(nn.Module):
    """Spherical harmonics encoder for view directions."""

    def __init__(self, degree: int = 4):
        super().__init__()
        self.degree = degree
        self.output_dim = degree**2

    def forward(self, directions):
        """Encode directions using spherical harmonics."""
        # Normalize directions
        directions = F.normalize(directions, dim=-1)

        x, y, z = directions.unbind(-1)

        # Compute spherical harmonics up to specified degree
        sh_features = []

        # Degree 0
        sh_features.append(0.28209479177387814 * torch.ones_like(x))  # Y_0^0

        if self.degree > 1:
            # Degree 1
            sh_features.append(-0.48860251190291987 * y)  # Y_1^{-1}
            sh_features.append(0.48860251190291987 * z)  # Y_1^0
            sh_features.append(-0.48860251190291987 * x)  # Y_1^1

        if self.degree > 2:
            # Degree 2
            sh_features.append(1.0925484305920792 * x * y)  # Y_2^{-2}
            sh_features.append(-1.0925484305920792 * y * z)  # Y_2^{-1}
            sh_features.append(0.31539156525252005 * (2 * z**2 - x**2 - y**2))  # Y_2^0
            sh_features.append(-1.0925484305920792 * x * z)  # Y_2^1
            sh_features.append(0.5462742152960396 * (x**2 - y**2))  # Y_2^2

        if self.degree > 3:
            # Degree 3
            sh_features.append(-0.5900435899266435 * y * (3 * x**2 - y**2))  # Y_3^{-3}
            sh_features.append(2.890611442640554 * x * y * z)  # Y_3^{-2}
            sh_features.append(-0.4570457994644658 * y * (4 * z**2 - x**2 - y**2))  # Y_3^{-1}
            sh_features.append(0.3731763325901154 * z * (2 * z**2 - 3 * x**2 - 3 * y**2))  # Y_3^0
            sh_features.append(-0.4570457994644658 * x * (4 * z**2 - x**2 - y**2))  # Y_3^1
            sh_features.append(1.445305721320277 * z * (x**2 - y**2))  # Y_3^2
            sh_features.append(-0.5900435899266435 * x * (x**2 - 3 * y**2))  # Y_3^3

        return torch.stack(sh_features[: self.output_dim], dim=-1)


class InstantNGPModel(nn.Module):
    """InstantNGP model with modern PyTorch features.

    Features:
    - Multi-resolution hash encoding
    - Efficient batch processing
    - Automatic mixed precision training
    - Memory-optimized operations
    - CUDA acceleration with CPU fallback
    """

    def __init__(self, config: InstantNGPConfig):
        super().__init__()
        self.config = config

        # Initialize encoders
        self.encoding = HashEncoder(config)
        self.direction_encoder = SHEncoder(degree=4)

        # Initialize networks
        self.sigma_net = nn.Sequential(
            nn.Linear(self.encoding.output_dim, config.hidden_dim),
            nn.ReLU(),
            nn.Linear(config.hidden_dim, config.hidden_dim),
            nn.ReLU(),
            nn.Linear(config.hidden_dim, 1 + config.geo_feature_dim),
        )

        self.color_net = nn.Sequential(
            nn.Linear(
                config.geo_feature_dim + self.direction_encoder.output_dim,
                config.hidden_dim_color,
            ),
            nn.ReLU(),
            nn.Linear(config.hidden_dim_color, config.hidden_dim_color),
            nn.ReLU(),
            nn.Linear(config.hidden_dim_color, 3),
            nn.Sigmoid(),
        )

        # Initialize renderer
        self.renderer = InstantNGPRenderer(config)

        # Move to device
        self.to(config.device)

        # Initialize AMP scaler
        if config.use_amp:
            self.scaler = GradScaler(
                "cuda",
                init_scale=config.grad_scaler_init_scale,
                growth_factor=config.grad_scaler_growth_factor,
                backoff_factor=config.grad_scaler_backoff_factor,
                growth_interval=config.grad_scaler_growth_interval,
            )

    def forward(
        self,
        rays_o: torch.Tensor,
        rays_d: torch.Tensor,
        near: float | None = None,
        far: float | None = None,
        num_samples: int | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass with modern optimizations.

        Args:
            rays_o: [N, 3] ray origins
            rays_d: [N, 3] ray directions
            near: Near plane distance (optional)
            far: Far plane distance (optional)
            num_samples: Number of samples per ray (optional)

        Returns:
            Tuple containing:
                rgb: [N, 3] rendered RGB values
                density: [N] rendered density values
        """
        # Move inputs to device efficiently
        rays_o = rays_o.to(self.config.device, non_blocking=True)
        rays_d = rays_d.to(self.config.device, non_blocking=True)

        # Set default values
        num_samples = num_samples or self.config.num_samples

        # Render with automatic mixed precision
        with autocast(device_type="cuda" if torch.cuda.is_available() else "cpu"):
            # Sample points along rays
            pts, t_vals = self.renderer.sample_rays(
                rays_o,
                rays_d,
                near or self.config.near_plane,
                far or self.config.far_plane,
                num_samples,
            )

            # Get point features
            point_features = self.encoding(pts)

            # Forward through MLP
            h = point_features
            for layer in self.sigma_net[:-1]:
                h = layer(h)
            outputs = self.sigma_net[-1](h)

            # Split outputs
            density = F.relu(outputs[:, 0])
            geo_features = outputs[:, 1:]

            # Encode view directions
            viewdirs = rays_d / torch.norm(rays_d, dim=-1, keepdim=True)
            viewdirs = viewdirs[:, None].expand(-1, num_samples, -1).reshape(-1, 3)
            direction_features = self.direction_encoder(viewdirs)

            # Concatenate features
            features = torch.cat([geo_features, direction_features], dim=-1)

            # Forward through color network
            color = self.color_net(features)

            # Volume rendering
            render_outputs = self.renderer.render_rays(color, density, t_vals, rays_d)

        return render_outputs["rgb"], density

    def train_step(
        self,
        batch: dict[str, torch.Tensor],
        optimizer: torch.optim.Optimizer,
    ) -> dict[str, torch.Tensor]:
        """Perform a single training step with modern optimizations.

        Args:
            batch: Dictionary containing training data
            optimizer: PyTorch optimizer

        Returns:
            Dictionary containing loss values
        """
        # Move batch to device efficiently
        batch = {k: v.to(self.config.device, non_blocking=True) for k, v in batch.items()}

        # Forward pass with automatic mixed precision
        with autocast(device_type="cuda" if torch.cuda.is_available() else "cpu"):
            outputs = self.forward(
                batch["rays_o"],
                batch["rays_d"],
                batch.get("near"),
                batch.get("far"),
            )

            # Compute loss
            loss = F.mse_loss(outputs[0], batch["rgb"])

        # Optimization step with gradient scaling
        optimizer.zero_grad(set_to_none=True)
        self.scaler.scale(loss).backward()
        self.scaler.step(optimizer)
        self.scaler.update()

        return {"loss": loss.item()}

    @torch.inference_mode()
    def evaluate(
        self,
        batch: dict[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        """Evaluate model with modern optimizations.

        Args:
            batch: Dictionary containing evaluation data

        Returns:
            Dictionary containing evaluation metrics
        """
        # Move batch to device efficiently
        batch = {k: v.to(self.config.device, non_blocking=True) for k, v in batch.items()}

        # Forward pass with automatic mixed precision
        with autocast(device_type="cuda" if torch.cuda.is_available() else "cpu"):
            outputs = self.forward(
                batch["rays_o"],
                batch["rays_d"],
                batch.get("near"),
                batch.get("far"),
            )

            # Compute metrics
            mse = F.mse_loss(outputs[0], batch["rgb"])
            psnr = -10.0 * torch.log10(mse + 1e-8)

            metrics = {
                "mse": mse.item(),
                "psnr": psnr.item(),
            }

        return metrics


class InstantNGPLoss(nn.Module):
    """Loss function for Instant NGP training."""

    def __init__(self, config: InstantNGPConfig):
        super().__init__()
        self.config = config

    def forward(self, pred_rgb, target_rgb, pred_density=None, positions=None):
        """
        Compute Instant NGP losses.

        Args:
            pred_rgb: [N, 3] predicted RGB
            target_rgb: [N, 3] target RGB
            pred_density: [N, 1] predicted density (optional)
            positions: [N, 3] 3D positions (optional)

        Returns:
            Dictionary of losses
        """
        losses = {}

        # RGB reconstruction loss
        rgb_loss = F.mse_loss(pred_rgb, target_rgb)
        losses["rgb_loss"] = rgb_loss

        # Entropy regularization on density
        if pred_density is not None and self.config.lambda_entropy > 0:
            # Encourage sparsity in density
            entropy_loss = -torch.mean(pred_density * torch.log(pred_density + 1e-10))
            losses["entropy_loss"] = self.config.lambda_entropy * entropy_loss

        # Total variation loss on hash grid (simplified)
        if self.config.lambda_tv > 0:
            tv_loss = 0.0
            # In practice, this would compute TV loss on the hash grid
            # For simplicity, we use a placeholder
            losses["tv_loss"] = self.config.lambda_tv * tv_loss

        # Total loss
        total_loss = sum(losses.values())
        losses["total_loss"] = total_loss

        # PSNR for monitoring
        with torch.no_grad():
            mse = F.mse_loss(pred_rgb, target_rgb)
            psnr = -10.0 * torch.log10(mse + 1e-8)
            losses["psnr"] = psnr

        return losses


class InstantNGPRenderer:
    """Renderer for Instant NGP."""

    def __init__(self, config: InstantNGPConfig):
        self.config = config

    def sample_rays(self, rays_o, rays_d, near, far, num_samples):
        """Sample points along rays.

        Args:
            rays_o: [N, 3] ray origins
            rays_d: [N, 3] ray directions
            near: float or [N] near plane distances
            far: float or [N] far plane distances
            num_samples: int number of samples per ray

        Returns:
            pts: [N * num_samples, 3] sampled points
            z_vals: [N, num_samples] depths of sampled points
        """
        device = rays_o.device
        num_rays = rays_o.shape[0]

        # Convert scalar near/far to tensors if needed
        if isinstance(near, (int, float)):
            near = torch.full((num_rays,), near, device=device)
        if isinstance(far, (int, float)):
            far = torch.full((num_rays,), far, device=device)

        # Linear sampling in depth
        t_vals = torch.linspace(0.0, 1.0, steps=num_samples, device=device)
        z_vals = near[:, None] * (1.0 - t_vals) + far[:, None] * t_vals  # [N_rays, N_samples]

        # Add noise for stochastic sampling
        if hasattr(self, "training") and self.training:
            mids = 0.5 * (z_vals[..., 1:] + z_vals[..., :-1])
            upper = torch.cat([mids, z_vals[..., -1:]], -1)
            lower = torch.cat([z_vals[..., :1], mids], -1)
            t_rand = torch.rand(z_vals.shape, device=device)
            z_vals = lower + (upper - lower) * t_rand

        # Get sample points
        pts = (
            rays_o[..., None, :] + rays_d[..., None, :] * z_vals[..., :, None]
        )  # [N_rays, N_samples, 3]

        return pts.reshape(-1, 3), z_vals

    def volume_render(self, rgb, density, z_vals, rays_d):
        """Volume rendering with alpha compositing.

        Args:
            rgb: [N_rays, N_samples, 3] RGB values at sampled points
            density: [N_rays, N_samples, 1] Density values at sampled points
            z_vals: [N_rays, N_samples] Depths of sampled points
            rays_d: [N_rays, 3] Ray directions

        Returns:
            Dictionary containing rendered outputs
        """
        device = rgb.device

        # Compute distances between adjacent samples
        dists = z_vals[..., 1:] - z_vals[..., :-1]
        dists = torch.cat(
            [dists, torch.tensor([1e10], device=device).expand(dists[..., :1].shape)], -1
        )

        # Multiply by ray direction norm
        dists = dists * torch.norm(rays_d[..., None, :], dim=-1)

        # Compute alpha values
        alpha = 1.0 - torch.exp(-density.squeeze(-1) * dists)  # [N_rays, N_samples]

        # Compute transmittance
        transmittance = torch.cumprod(
            torch.cat([torch.ones_like(alpha[..., :1]), 1.0 - alpha + 1e-10], -1), -1
        )[..., :-1]

        # Compute weights
        weights = alpha * transmittance  # [N_rays, N_samples]

        # Composite RGB
        rgb_map = torch.sum(weights[..., None] * rgb, -2)  # [N_rays, 3]

        # Composite depth
        depth_map = torch.sum(weights * z_vals, -1)  # [N_rays]

        # Accumulated opacity
        acc_map = torch.sum(weights, -1)  # [N_rays]

        return {
            "rgb": rgb_map,
            "depth": depth_map,
            "acc": acc_map,
            "weights": weights,
            "z_vals": z_vals,
        }

    def render_rays(self, rgb, density, z_vals, rays_d):
        """Render rays through the model.

        Args:
            rgb: [N_rays * N_samples, 3] RGB values at sampled points
            density: [N_rays * N_samples] Density values at sampled points
            z_vals: [N_rays, N_samples] Depths of sampled points
            rays_d: [N_rays, 3] Ray directions

        Returns:
            Dictionary containing rendered outputs
        """
        num_rays = rays_d.shape[0]
        num_samples = z_vals.shape[1]

        # Reshape inputs
        rgb = rgb.reshape(num_rays, num_samples, 3)
        density = density.reshape(num_rays, num_samples, 1)

        # Volume rendering
        return self.volume_render(rgb, density, z_vals, rays_d)


# Add alias for backward compatibility after class definition
InstantNGP = InstantNGPModel
