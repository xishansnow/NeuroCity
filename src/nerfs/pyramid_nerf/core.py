from __future__ import annotations

from typing import Any, Optional, Union

"""
PyNeRF Core Module
Implements the main PyNeRF model with pyramidal representation
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
import numpy as np
import logging
from torch.amp.autocast_mode import autocast
from torch.amp.grad_scaler import GradScaler
from pathlib import Path

from .pyramid_encoder import PyramidEncoder, HashEncoder
from .pyramid_renderer import PyramidRenderer
from .utils import compute_sample_area, get_pyramid_level, interpolate_pyramid_outputs

logger = logging.getLogger(__name__)

# Type aliases for modern Python 3.10
Tensor = torch.Tensor
Device = torch.device | str
DType = torch.dtype
TensorDict = Dict[str, Tensor]


@dataclass
class PyramidNeRFConfig:
    """Configuration for Pyramid NeRF model with modern features."""

    # Network architecture
    hidden_dim: int = 256
    num_layers: int = 8
    skip_connections: List[int] = (4,)
    activation: str = "relu"
    output_activation: str = "sigmoid"

    # Pyramid settings
    num_levels: int = 4
    base_resolution: Tuple[int, int, int] = (32, 32, 32)
    feature_dim: int = 32
    use_trilinear: bool = True

    # Positional encoding
    pos_encoding_levels: int = 10
    dir_encoding_levels: int = 4

    # Rendering settings
    num_samples: int = 192
    num_importance_samples: int = 96
    background_color: str = "white"
    near_plane: float = 0.1
    far_plane: float = 1000.0

    # Training settings
    learning_rate: float = 5e-4
    learning_rate_decay_steps: int = 50000
    learning_rate_decay_mult: float = 0.1
    weight_decay: float = 1e-6

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
        assert len(self.base_resolution) == 3, "Base resolution must be a 3-tuple"
        assert all(s > 0 for s in self.base_resolution), "Resolution must be positive"
        assert self.num_levels > 0, "Number of levels must be positive"

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


class PyramidNeRF(nn.Module):
    """Pyramid NeRF model with modern optimizations."""

    def __init__(self, config: PyramidNeRFConfig):
        super().__init__()
        self.config = config

        # Initialize pyramid levels
        self.pyramid_levels = nn.ModuleList(
            [
                PyramidLevel(
                    resolution=[s * (2**i) for s in config.base_resolution],
                    feature_dim=config.feature_dim,
                    use_trilinear=config.use_trilinear,
                )
                for i in range(config.num_levels)
            ]
        )

        # Initialize network
        self.network = PyramidNeRFNetwork(
            feature_dim=config.feature_dim * config.num_levels,
            hidden_dim=config.hidden_dim,
            num_layers=config.num_layers,
            skip_connections=config.skip_connections,
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
                # Get features from each pyramid level
                pyramid_features = []
                for level in self.pyramid_levels:
                    level_features = level(chunk_coords)
                    pyramid_features.append(level_features)

                # Concatenate features from all levels
                combined_features = torch.cat(pyramid_features, dim=-1)

                # Process through network
                chunk_outputs = self.network(combined_features, chunk_view_dirs)

            outputs_list.append(chunk_outputs)

        # Combine chunk outputs
        outputs = {
            k: torch.cat([out[k] for out in outputs_list], dim=0) for k in outputs_list[0].keys()
        }

        return outputs

    def training_step(
        self, batch: TensorDict, optimizer: torch.optim.Optimizer
    ) -> Tuple[Tensor, TensorDict]:
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

        # Density regularization
        if "density" in predictions:
            losses["density_reg"] = torch.mean(predictions["density"])

        # Feature regularization for each pyramid level
        for i, level in enumerate(self.pyramid_levels):
            level_features = level.features
            feature_norm = torch.norm(level_features, dim=-1).mean()
            losses[f"level_{i}_reg"] = feature_norm

        # Total loss
        losses["total_loss"] = sum(losses.values())

        return losses

    def update_learning_rate(self, optimizer: torch.optim.Optimizer, step: int):
        """Update learning rate with decay."""
        decay_steps = step // self.config.learning_rate_decay_steps
        decay_factor = self.config.learning_rate_decay_mult**decay_steps

        # Update optimizer learning rate
        for param_group in optimizer.param_groups:
            param_group["lr"] = self.config.learning_rate * decay_factor


class PyramidLevel(nn.Module):
    """Single level of the pyramid representation."""

    def __init__(self, resolution: List[int], feature_dim: int, use_trilinear: bool = True):
        super().__init__()
        self.resolution = resolution
        self.feature_dim = feature_dim
        self.use_trilinear = use_trilinear

        # Initialize feature grid
        self.features = nn.Parameter(torch.randn(1, feature_dim, *resolution) * 0.01)

    def forward(self, coords: Tensor) -> Tensor:
        """Sample features at given coordinates."""
        # Normalize coordinates to [-1, 1]
        normalized_coords = coords * 2.0 - 1.0

        # Reshape coordinates for grid sampling
        coords_for_sampling = normalized_coords.view(-1, 1, 1, 3).permute(0, 3, 1, 2)

        # Sample features
        if self.use_trilinear:
            sampled_features = F.grid_sample(
                self.features.expand(coords.shape[0], -1, -1, -1, -1),
                coords_for_sampling,
                mode="bilinear",
                padding_mode="border",
                align_corners=True,
            )
        else:
            sampled_features = F.grid_sample(
                self.features.expand(coords.shape[0], -1, -1, -1, -1),
                coords_for_sampling,
                mode="nearest",
                padding_mode="border",
                align_corners=True,
            )

        return sampled_features.squeeze(-1).squeeze(-1).permute(0, 2, 1)


class PyramidNeRFNetwork(nn.Module):
    """Network module for Pyramid NeRF."""

    def __init__(
        self, feature_dim: int, hidden_dim: int, num_layers: int, skip_connections: List[int]
    ):
        super().__init__()
        self.feature_dim = feature_dim
        self.hidden_dim = hidden_dim
        self.skip_connections = skip_connections

        # Initialize network layers
        self.layers = nn.ModuleList()

        # First layer
        self.layers.append(nn.Linear(feature_dim, hidden_dim))

        # Hidden layers
        for i in range(num_layers - 1):
            input_dim = hidden_dim + feature_dim if i in skip_connections else hidden_dim
            self.layers.append(nn.Linear(input_dim, hidden_dim))

        # Output layers
        self.density_head = nn.Linear(hidden_dim, 1)
        self.feature_head = nn.Linear(hidden_dim, hidden_dim)

        # View-dependent layers
        self.view_layers = nn.Sequential(
            nn.Linear(hidden_dim + 3, hidden_dim // 2), nn.ReLU(), nn.Linear(hidden_dim // 2, 3)
        )

    def forward(self, features: Tensor, view_dirs: Tensor | None = None) -> TensorDict:
        """Forward pass through the network."""
        # Initial layer
        x = F.relu(self.layers[0](features))

        # Hidden layers with skip connections
        for i, layer in enumerate(self.layers[1:], 1):
            if i in self.skip_connections:
                x = torch.cat([x, features], dim=-1)
            x = F.relu(layer(x))

        # Density prediction
        density = self.density_head(x)

        # Feature extraction
        latent_features = self.feature_head(x)

        # View-dependent RGB prediction
        if view_dirs is not None:
            view_input = torch.cat([latent_features, view_dirs], dim=-1)
            rgb = torch.sigmoid(self.view_layers(view_input))
        else:
            rgb = torch.zeros_like(density.expand(-1, 3))

        return {"rgb": rgb, "density": density, "features": latent_features}


class PyNeRFMLP(nn.Module):
    """MLP head for PyNeRF pyramid level"""

    def __init__(
        self, input_dim: int, hidden_dim: int = 64, num_layers: int = 2, use_viewdirs: bool = True
    ):
        super().__init__()
        self.use_viewdirs = use_viewdirs

        # Density network
        density_layers = []
        for i in range(num_layers):
            if i == 0:
                density_layers.append(nn.Linear(input_dim, hidden_dim))
            else:
                density_layers.append(nn.Linear(hidden_dim, hidden_dim))
            density_layers.append(nn.ReLU(inplace=True))

        density_layers.append(nn.Linear(hidden_dim, 1))
        self.density_net = nn.Sequential(*density_layers)

        # Color network
        color_input_dim = hidden_dim
        if use_viewdirs:
            color_input_dim += 3  # Add viewing direction

        color_layers = []
        for i in range(num_layers):
            if i == 0:
                color_layers.append(nn.Linear(color_input_dim, hidden_dim))
            else:
                color_layers.append(nn.Linear(hidden_dim, hidden_dim))
            color_layers.append(nn.ReLU(inplace=True))

        color_layers.append(nn.Linear(hidden_dim, 3))
        color_layers.append(nn.Sigmoid())
        self.color_net = nn.Sequential(*color_layers)

        # Feature extraction for color network
        self.feature_linear = nn.Linear(input_dim, hidden_dim)

    def forward(
        self, encoded_pts: torch.Tensor, viewdirs: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through MLP

        Args:
            encoded_pts: Encoded position features [N, input_dim]
            viewdirs: Viewing directions [N, 3]

        Returns:
            tuple of (rgb, sigma)
        """
        # Density prediction
        sigma = self.density_net(encoded_pts)
        sigma = F.relu(sigma)

        # Color prediction
        features = F.relu(self.feature_linear(encoded_pts))

        if self.use_viewdirs and viewdirs is not None:
            # Normalize viewing directions
            viewdirs = F.normalize(viewdirs, dim=-1)
            color_input = torch.cat([features, viewdirs], dim=-1)
        else:
            color_input = features

        rgb = self.color_net(color_input)

        return rgb, sigma.squeeze(-1)


class PyNeRFLoss(nn.Module):
    """Loss function for PyNeRF training"""

    def __init__(self, config: PyramidNeRFConfig):
        super().__init__()
        self.config = config
        self.mse_loss = nn.MSELoss()

    def forward(
        self, predictions: Dict[str, torch.Tensor], targets: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """
        Compute PyNeRF loss

        Args:
            predictions: Model predictions
            targets: Ground truth targets

        Returns:
            Dictionary of loss components
        """
        losses = {}

        # Main color loss
        if "rgb" in predictions and "rgb" in targets:
            losses["color_loss"] = self.mse_loss(predictions["rgb"], targets["rgb"])

        # Coarse color loss (if using hierarchical sampling)
        if "rgb_coarse" in predictions and "rgb" in targets:
            losses["color_loss_coarse"] = self.mse_loss(predictions["rgb_coarse"], targets["rgb"])

        # Total loss
        total_loss = sum(losses.values())
        losses["total_loss"] = total_loss

        return losses
