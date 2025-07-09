from __future__ import annotations

from typing import Any

"""
Grid-guided Neural Radiance Fields for Large Urban Scenes.

This module implements the core components of Grid-NeRF as described in:
"Grid-guided neural radiance fields for large urban scenes"

Key features:
- Hierarchical grid structure for large-scale scenes
- Multi-resolution grid representation
- Grid-guided neural networks
- Efficient rendering for urban environments
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
import numpy as np
import math
import logging
from pathlib import Path
from collections.abc import Mapping
from torch.amp.autocast_mode import autocast
from torch.amp.grad_scaler import GradScaler

logger = logging.getLogger(__name__)

# Type aliases for modern Python 3.10
Tensor = torch.Tensor
Device = torch.device | str
DType = torch.dtype
TensorDict = dict[str, Tensor]


@dataclass
class GridNeRFConfig:
    """Configuration for Grid-NeRF model with modern features."""

    # Grid settings
    grid_size: int = 128  # Resolution of feature grid
    feature_dim: int = 32  # Dimension of grid features
    num_levels: int = 16  # Number of feature grid levels
    base_resolution: int = 16  # Base resolution for hash encoding
    finest_resolution: int = 512  # Finest resolution for hash encoding
    log2_hashmap_size: int = 19  # Log2 of hash table size

    # Network architecture
    mlp_hidden_dim: int = 64
    mlp_num_layers: int = 3
    skip_connections: list[int] = (2,)
    activation: str = "relu"
    output_activation: str = "sigmoid"

    # Rendering settings
    num_samples: int = 128  # Samples per ray
    num_importance_samples: int = 64  # Additional importance samples
    background_color: str = "white"
    near_plane: float = 0.1
    far_plane: float = 100.0

    # Training settings
    learning_rate: float = 1e-3
    weight_decay: float = 1e-6
    warmup_steps: int = 500
    max_steps: int = 50000

    # Loss weights
    rgb_loss_weight: float = 1.0
    alpha_loss_weight: float = 0.01
    depth_loss_weight: float = 0.1
    tv_loss_weight: float = 0.01

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

    # Evaluation
    eval_every_n_steps: int = 100
    num_eval_images: int = 8

    def __post_init__(self):
        """Post-initialization validation and initialization."""
        # Validate grid settings
        assert self.grid_size > 0, "Grid size must be positive"
        assert self.feature_dim > 0, "Feature dimension must be positive"
        assert (
            self.base_resolution < self.finest_resolution
        ), "Base resolution must be less than finest resolution"

        # Initialize device
        self.device = torch.device(self.default_device if torch.cuda.is_available() else "cpu")

        # Initialize grad scaler for AMP
        if self.use_amp:
            self.grad_scaler = GradScaler(
                init_scale=self.grad_scaler_init_scale,
                growth_factor=self.grad_scaler_growth_factor,
                backoff_factor=self.grad_scaler_backoff_factor,
                growth_interval=self.grad_scaler_growth_interval,
            )

        # Create checkpoint directory
        self.checkpoint_dir = Path(self.checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # Initialize activation functions
        self.activation_fn = getattr(F, self.activation)
        self.output_activation_fn = getattr(F, self.output_activation)


class HierarchicalGrid(nn.Module):
    """Hierarchical grid structure for multi-resolution scene representation."""

    def __init__(self, config: GridNeRFConfig):
        super().__init__()
        self.config = config
        self.scene_bounds = torch.tensor(config.scene_bounds)
        self.scene_size = self.scene_bounds[3:] - self.scene_bounds[:3]

        # Create multi-level grids
        self.grids = nn.ParameterList()
        self.grid_resolutions = []

        for level in range(config.num_grid_levels):
            # Exponentially increase resolution
            resolution = config.base_grid_resolution * (2**level)
            resolution = min(resolution, config.max_grid_resolution)
            self.grid_resolutions.append(resolution)

            # Create grid features for this level
            grid_features = nn.Parameter(
                torch.randn(resolution, resolution, resolution, config.grid_feature_dim) * 0.1
            )
            self.grids.append(grid_features)

    def world_to_grid_coords(self, world_coords: torch.Tensor, level: int) -> torch.Tensor:
        """Convert world coordinates to grid coordinates for a specific level."""
        resolution = self.grid_resolutions[level]

        # Normalize to [0, 1]
        normalized = (world_coords - self.scene_bounds[:3]) / self.scene_size

        # Scale to grid resolution
        grid_coords = normalized * (resolution - 1)

        return grid_coords

    def sample_grid_features(self, world_coords: torch.Tensor, level: int) -> torch.Tensor:
        """Sample grid features at world coordinates using trilinear interpolation."""
        grid_coords = self.world_to_grid_coords(world_coords, level)

        # Trilinear interpolation
        grid_features = self.grids[level]
        features = F.grid_sample(
            grid_features.permute(3, 0, 1, 2).unsqueeze(0),  # [1, C, D, H, W]
            grid_coords.unsqueeze(0).unsqueeze(0).unsqueeze(0),  # [1, 1, 1, N, 3]
            mode="bilinear",
            padding_mode="border",
            align_corners=True,
        )

        # Remove extra dimensions and transpose
        features = features.squeeze(0).squeeze(1).squeeze(1).transpose(0, 1)  # [N, C]

        return features

    def get_multi_level_features(self, world_coords: torch.Tensor) -> torch.Tensor:
        """Get features from all grid levels and concatenate them."""
        all_features = []

        for level in range(self.config.num_grid_levels):
            features = self.sample_grid_features(world_coords, level)
            all_features.append(features)

        # Concatenate features from all levels
        multi_level_features = torch.cat(all_features, dim=-1)

        return multi_level_features

    def get_occupied_cells(self, level: int, threshold: float = 0.01) -> torch.Tensor:
        """Get mask of occupied cells based on feature magnitude."""
        grid_features = self.grids[level]
        feature_magnitude = torch.norm(grid_features, dim=-1)
        occupied_mask = feature_magnitude > threshold

        return occupied_mask


class GridGuidedMLP(nn.Module):
    """MLP network guided by hierarchical grid features."""

    def __init__(self, config: GridNeRFConfig):
        super().__init__()
        self.config = config

        # Input dimension: grid features from all levels + positional encoding
        input_dim = config.grid_feature_dim * config.num_grid_levels

        # Add positional encoding dimension
        pos_encode_dim = 3 * 2 * 10  # 3 coords * 2 (sin/cos) * 10 frequencies
        input_dim += pos_encode_dim

        # Density network
        density_layers = []
        for i in range(config.mlp_num_layers):
            if i == 0:
                density_layers.append(nn.Linear(input_dim, config.mlp_hidden_dim))
            else:
                density_layers.append(nn.Linear(config.mlp_hidden_dim, config.mlp_hidden_dim))

            if i < config.mlp_num_layers - 1:
                density_layers.append(nn.ReLU(inplace=True))

        # Final density layer
        density_layers.append(nn.Linear(config.mlp_hidden_dim, 1))
        self.density_net = nn.Sequential(*density_layers)

        # Color network
        if config.view_dependent:
            view_embed_dim = 3 * 2 * 4  # 3 coords * 2 (sin/cos) * 4 frequencies
            color_input_dim = config.mlp_hidden_dim + view_embed_dim
        else:
            color_input_dim = config.mlp_hidden_dim

        self.color_net = nn.Sequential(
            nn.Linear(
                color_input_dim,
                config.mlp_hidden_dim // 2,
            )
        )

        # Feature extraction layer for color network
        self.feature_layer = nn.Linear(config.mlp_hidden_dim, config.mlp_hidden_dim)

    def positional_encoding(self, coords: torch.Tensor, num_freqs: int = 10) -> torch.Tensor:
        """Apply positional encoding to coordinates."""
        encoded = []

        for freq in range(num_freqs):
            for func in [torch.sin, torch.cos]:
                encoded.append(func(coords * (2.0**freq) * math.pi))

        return torch.cat(encoded, dim=-1)

    def forward(
        self, grid_features: torch.Tensor, view_dirs: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass through the MLP network.

        Args:
            grid_features: Multi-level grid features [N, D]
            view_dirs: Optional view directions [N, 3]

        Returns:
            tuple[torch.Tensor, torch.Tensor]: (rgb_colors, densities)
                - rgb_colors: RGB colors [N, 3]
                - densities: Volume densities [N, 1]
        """
        # Process through density network
        density_features = self.density_net(grid_features)

        # Extract features for color network
        color_features = self.feature_layer(density_features)

        if self.config.view_dependent and view_dirs is not None:
            # Encode view directions
            view_embed = self.positional_encoding(view_dirs, num_freqs=4)
            color_input = torch.cat([color_features, view_embed], dim=-1)
        else:
            color_input = color_features

        # Process through color network
        rgb = torch.sigmoid(self.color_net(color_input))

        return rgb, density_features


class GridNeRFRenderer(nn.Module):
    """Neural renderer for Grid-NeRF."""

    def __init__(self, config: GridNeRFConfig) -> None:
        """Initialize the renderer.

        Args:
            config: Configuration object for Grid-NeRF
        """
        super().__init__()
        self.config = config

    def sample_points_along_rays(
        self,
        ray_origins: torch.Tensor,
        ray_directions: torch.Tensor,
        near: float,
        far: float,
        num_samples: int,
        stratified: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Sample points along rays using stratified sampling.

        Args:
            ray_origins: Ray origin points [N, 3]
            ray_directions: Ray direction vectors [N, 3]
            near: Near plane distance
            far: Far plane distance
            num_samples: Number of samples per ray
            stratified: Whether to use stratified sampling

        Returns:
            tuple[torch.Tensor, torch.Tensor]: (sample_points, t_vals)
                - sample_points: Sampled 3D points [N, num_samples, 3]
                - t_vals: Distance values along rays [N, num_samples]
        """
        # Generate sampling distances
        t_vals = torch.linspace(0.0, 1.0, num_samples, device=ray_origins.device)
        t_vals = near + t_vals * (far - near)

        if stratified and self.training:
            # Add random offsets for stratified sampling
            noise = torch.rand_like(t_vals) * (far - near) / num_samples
            t_vals = t_vals + noise

        # Expand t_vals for batch dimension
        t_vals = t_vals.expand(ray_origins.shape[0], num_samples)

        # Compute sample points
        sample_points = ray_origins.unsqueeze(-2) + ray_directions.unsqueeze(  # [N, 1, 3]
            -2
        ) * t_vals.unsqueeze(
            -1
        )  # [N, S, 3]

        return sample_points, t_vals

    def hierarchical_sampling(
        self,
        ray_origins: torch.Tensor,
        ray_directions: torch.Tensor,
        coarse_weights: torch.Tensor,
        coarse_t_vals: torch.Tensor,
        num_fine_samples: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Sample points hierarchically based on coarse network predictions.

        Args:
            ray_origins: Ray origin points [N, 3]
            ray_directions: Ray direction vectors [N, 3]
            coarse_weights: Weights from coarse network [N, num_coarse_samples]
            coarse_t_vals: Distance values from coarse sampling [N, num_coarse_samples]
            num_fine_samples: Number of fine samples to generate

        Returns:
            tuple[torch.Tensor, torch.Tensor]: (fine_points, fine_t_vals)
                - fine_points: Sampled 3D points [N, num_fine_samples, 3]
                - fine_t_vals: Distance values along rays [N, num_fine_samples]
        """
        # Add small epsilon to weights to prevent division by zero
        weights = coarse_weights + 1e-5

        # Get PDF and CDF for importance sampling
        pdf = weights / torch.sum(weights, dim=-1, keepdim=True)
        cdf = torch.cumsum(pdf, dim=-1)
        cdf = torch.cat([torch.zeros_like(cdf[..., :1]), cdf], dim=-1)

        # Draw uniform samples
        if self.training:
            u = torch.rand(ray_origins.shape[0], num_fine_samples, device=ray_origins.device)
        else:
            u = torch.linspace(0.0, 1.0, num_fine_samples, device=ray_origins.device)
            u = u.expand(ray_origins.shape[0], num_fine_samples)

        # Invert CDF to get samples
        u = u.contiguous()
        inds = torch.searchsorted(cdf, u, right=True)
        below = torch.clamp(inds - 1, 0, cdf.shape[-1] - 1)
        above = torch.clamp(inds, 0, cdf.shape[-1] - 1)

        cdf_below = torch.gather(cdf, -1, below)
        cdf_above = torch.gather(cdf, -1, above)
        t_below = torch.gather(coarse_t_vals, -1, below)
        t_above = torch.gather(coarse_t_vals, -1, above)

        denom = cdf_above - cdf_below
        denom = torch.where(denom < 1e-5, torch.ones_like(denom), denom)
        t = (u - cdf_below) / denom
        fine_t_vals = t_below + t * (t_above - t_below)

        # Compute fine sample points
        fine_points = ray_origins.unsqueeze(-2) + ray_directions.unsqueeze(
            -2
        ) * fine_t_vals.unsqueeze(-1)

        return fine_points, fine_t_vals

    def volume_rendering(
        self,
        colors: torch.Tensor,
        densities: torch.Tensor,
        t_vals: torch.Tensor,
        ray_directions: torch.Tensor,
        background_color: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """Render colors and depths using volume rendering equation.

        Args:
            colors: RGB colors for each sample [N, num_samples, 3]
            densities: Volume density values [N, num_samples, 1]
            t_vals: Distance values along rays [N, num_samples]
            ray_directions: Ray direction vectors [N, 3]
            background_color: Optional background color [3]

        Returns:
            dict[str, torch.Tensor]: Rendering outputs containing:
                - 'color': Rendered RGB colors [N, 3]
                - 'depth': Rendered depth values [N, 1]
                - 'weights': Sample weights [N, num_samples]
                - 'transmittance': Ray transmittance values [N, num_samples]
        """
        # Convert densities to alpha values
        deltas = t_vals[..., 1:] - t_vals[..., :-1]
        delta_inf = 1e10 * torch.ones_like(deltas[..., :1])
        deltas = torch.cat([deltas, delta_inf], dim=-1)

        # Account for viewing direction
        deltas = deltas * torch.norm(ray_directions.unsqueeze(-1), dim=-1)

        # Compute alpha values
        alpha = 1.0 - torch.exp(-densities * deltas)

        # Compute weights and transmittance
        transmittance = torch.cumprod(
            torch.cat([torch.ones_like(alpha[:, :1]), 1.0 - alpha + 1e-10], dim=-1), dim=-1
        )[..., :-1]
        weights = alpha * transmittance

        # Compute color and depth
        color = torch.sum(weights.unsqueeze(-1) * colors, dim=-2)
        depth = torch.sum(weights * t_vals, dim=-1, keepdim=True)

        # Add background color if provided
        if background_color is not None:
            background_color = background_color.to(color.device)
            color = color + (1.0 - weights.sum(dim=-1, keepdim=True)) * background_color

        outputs = {
            "color": color,
            "depth": depth,
            "weights": weights,
            "transmittance": transmittance,
        }

        return outputs


class GridNeRF(nn.Module):
    """Grid-NeRF model with modern optimizations."""

    def __init__(self, config: GridNeRFConfig):
        super().__init__()
        self.config = config

        # Initialize feature grid
        self.feature_grid = HierarchicalGrid(
            grid_size=config.grid_size,
            feature_dim=config.feature_dim,
            num_levels=config.num_levels,
            base_resolution=config.base_resolution,
            finest_resolution=config.finest_resolution,
            log2_hashmap_size=config.log2_hashmap_size,
        )

        # Initialize MLP
        self.mlp = nn.ModuleList(
            [
                nn.Linear(config.feature_dim, config.mlp_hidden_dim),
                *[
                    nn.Linear(config.mlp_hidden_dim, config.mlp_hidden_dim)
                    for _ in range(config.mlp_num_layers - 1)
                ],
                nn.Linear(config.mlp_hidden_dim, 4),  # RGB + density
            ]
        )

        # Initialize AMP grad scaler
        self.grad_scaler = config.grad_scaler if config.use_amp else None

        # Initialize cache for grid samples
        self.grid_samples_cache = {} if config.cache_grid_samples else None

        # Move model to device
        self.to(config.device)

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize network weights."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity=self.config.activation)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, coords: Tensor, view_dirs: Tensor | None = None) -> TensorDict:
        """Forward pass with automatic mixed precision support."""
        # Move inputs to device efficiently
        coords = coords.to(self.config.device, non_blocking=self.config.use_non_blocking)
        if view_dirs is not None:
            view_dirs = view_dirs.to(self.config.device, non_blocking=self.config.use_non_blocking)

        # Process points in chunks for memory efficiency
        outputs_list = []
        for i in range(0, coords.shape[0], self.config.chunk_size):
            chunk_coords = coords[i : i + self.config.chunk_size]
            chunk_view_dirs = (
                view_dirs[i : i + self.config.chunk_size] if view_dirs is not None else None
            )

            # Use AMP for forward pass
            with autocast(enabled=self.config.use_amp):
                # Query feature grid
                if self.config.cache_grid_samples:
                    cache_key = chunk_coords.shape
                    if cache_key not in self.grid_samples_cache:
                        self.grid_samples_cache[cache_key] = self.feature_grid(chunk_coords)
                    features = self.grid_samples_cache[cache_key]
                else:
                    features = self.feature_grid(chunk_coords)

                # Process through MLP
                x = features
                for i, layer in enumerate(self.mlp):
                    x = layer(x)
                    if i < len(self.mlp) - 1:  # No activation on final layer
                        x = self.config.activation_fn(x)
                        # Add skip connections
                        if i in self.config.skip_connections:
                            x = torch.cat([x, features], dim=-1)

                # Split output into RGB and density
                rgb = self.config.output_activation_fn(x[..., :3])
                density = F.relu(x[..., 3])

                chunk_outputs = {"rgb": rgb, "density": density, "features": features}

            outputs_list.append(chunk_outputs)

        # Combine chunk outputs
        outputs = {
            k: torch.cat([out[k] for out in outputs_list], dim=0) for k in outputs_list[0].keys()
        }

        return outputs

    def training_step(
        self, batch: TensorDict, optimizer: torch.optim.Optimizer
    ) -> tuple[Tensor, TensorDict]:
        """Optimized training step with AMP support."""
        # Zero gradients efficiently
        optimizer.zero_grad(set_to_none=self.config.set_grad_to_none)

        # Move batch to device efficiently
        batch = {
            k: v.to(self.config.device, non_blocking=self.config.use_non_blocking)
            for k, v in batch.items()
        }

        # Forward pass with AMP
        with autocast(enabled=self.config.use_amp):
            outputs = self(batch["coords"], batch.get("view_dirs"))
            loss = self.compute_loss(outputs, batch)

        # Backward pass with grad scaling
        if self.config.use_amp:
            self.grad_scaler.scale(loss["total_loss"]).backward()
            self.grad_scaler.step(optimizer)
            self.grad_scaler.update()
        else:
            loss["total_loss"].backward()
            optimizer.step()

        return loss["total_loss"], loss

    @torch.inference_mode()
    def validation_step(self, batch: TensorDict) -> TensorDict:
        """Efficient validation step."""
        batch = {
            k: v.to(self.config.device, non_blocking=self.config.use_non_blocking)
            for k, v in batch.items()
        }

        outputs = self(batch["coords"], batch.get("view_dirs"))
        loss = self.compute_loss(outputs, batch)

        return loss

    def compute_loss(self, predictions: TensorDict, targets: TensorDict) -> TensorDict:
        """Compute training losses."""
        losses = {}

        # RGB loss
        if "rgb" in targets:
            losses["rgb"] = F.mse_loss(predictions["rgb"], targets["rgb"])
            losses["rgb"] *= self.config.rgb_loss_weight

        # Alpha/density loss
        if "density" in targets:
            losses["alpha"] = F.mse_loss(predictions["density"], targets["density"])
            losses["alpha"] *= self.config.alpha_loss_weight

        # Depth loss
        if "depth" in targets:
            losses["depth"] = F.mse_loss(predictions["depth"], targets["depth"])
            losses["depth"] *= self.config.depth_loss_weight

        # Total variation loss for feature grid
        if self.config.tv_loss_weight > 0:
            tv_loss = self.feature_grid.total_variation()
            losses["tv"] = tv_loss * self.config.tv_loss_weight

        # Total loss
        losses["total_loss"] = sum(losses.values())

        return losses

    def update_learning_rate(self, optimizer: torch.optim.Optimizer, step: int):
        """Update learning rate with warmup and decay."""
        if step < self.config.warmup_steps:
            # Linear warmup
            lr = self.config.learning_rate * (step / self.config.warmup_steps)
        else:
            # Cosine decay
            progress = (step - self.config.warmup_steps) / (
                self.config.max_steps - self.config.warmup_steps
            )
            lr = self.config.learning_rate * 0.5 * (1 + math.cos(math.pi * progress))

        # Update optimizer learning rate
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr
