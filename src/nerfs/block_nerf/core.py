"""
Block-NeRF Core Components

This module provides the core components for Block-NeRF implementation,
following the refactoring pattern used in SVRaster.

Components:
- BlockNeRFConfig: Configuration for Block-NeRF
- BlockNeRFModel: Main Block-NeRF model
- BlockNeRFLoss: Loss functions for training
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# Type aliases
Tensor = torch.Tensor
Device = Union[torch.device, str]
TensorDict = dict[str, Tensor]


@dataclass
class BlockNeRFConfig:
    """Configuration for Block-NeRF model.

    This configuration class follows the pattern used in SVRaster for
    centralized configuration management.
    """

    # Scene configuration
    scene_bounds: tuple[float, float, float, float, float, float] = (
        -100.0,
        -100.0,
        -10.0,
        100.0,
        100.0,
        10.0,
    )
    block_size: float = 75.0
    overlap_ratio: float = 0.1
    max_blocks: int = 64

    # Network architecture
    pos_encoding_levels: int = 16
    dir_encoding_levels: int = 4
    appearance_dim: int = 32
    exposure_dim: int = 8
    hidden_dim: int = 256
    num_layers: int = 8
    skip_connections: list[int] = field(default_factory=lambda: [4])
    use_integrated_encoding: bool = True

    # Appearance and exposure
    num_appearance_embeddings: int = 1000
    appearance_embedding_dim: int = 32
    exposure_embedding_dim: int = 8

    # Block-specific settings
    block_radius: float = 50.0
    visibility_threshold: float = 0.1
    use_visibility_network: bool = True
    use_pose_refinement: bool = False

    # Loss configuration
    density_activation: str = "exp"  # "exp" or "relu"
    color_activation: str = "sigmoid"
    rgb_loss_weight: float = 1.0
    depth_loss_weight: float = 0.1
    eikonal_loss_weight: float = 0.1

    # Device and precision
    device: str = "cuda"
    dtype: str = "float32"
    use_amp: bool = True


def positional_encoding(x: Tensor, L: int) -> Tensor:
    """Positional encoding for input coordinates.

    Args:
        x: Input tensor of shape (..., D)
        L: Number of encoding levels

    Returns:
        Encoded tensor of shape (..., D * 2 * L)
    """
    encoding = []
    for i in range(L):
        for fn in [torch.sin, torch.cos]:
            encoding.append(fn(2**i * math.pi * x))
    return torch.cat(encoding, dim=-1)


def integrated_positional_encoding(means: Tensor, covs: Tensor, L: int) -> Tensor:
    """Integrated positional encoding for mip-NeRF style inputs.

    Args:
        means: Mean coordinates (..., 3)
        covs: Covariance matrices (..., 3, 3) or diagonal (..., 3)
        L: Number of encoding levels

    Returns:
        Encoded tensor
    """
    if covs.dim() == means.dim():  # Diagonal covariance
        diag_covs = covs
    else:  # Full covariance matrix
        diag_covs = torch.diagonal(covs, dim1=-2, dim2=-1)

    encoding = []
    for i in range(L):
        freq = 2**i * math.pi
        # Compute expected value of sin/cos under Gaussian
        for fn_idx, fn in enumerate([torch.sin, torch.cos]):
            variance_term = -0.5 * (freq**2) * diag_covs
            if fn_idx == 0:  # sin
                encoding.append(torch.exp(variance_term) * torch.sin(freq * means))
            else:  # cos
                encoding.append(torch.exp(variance_term) * torch.cos(freq * means))

    return torch.cat(encoding, dim=-1)


class BlockNeRFNetwork(nn.Module):
    """Block-NeRF Neural Network based on mip-NeRF architecture."""

    def __init__(self, config: BlockNeRFConfig):
        super().__init__()
        self.config = config

        # Input dimensions
        if config.use_integrated_encoding:
            pos_input_dim = 3 + 3 * 2 * config.pos_encoding_levels
        else:
            pos_input_dim = 3 * 2 * config.pos_encoding_levels

        dir_input_dim = 3 * 2 * config.dir_encoding_levels

        # Density network (f_sigma)
        self.density_layers = nn.ModuleList()
        self.density_layers.append(nn.Linear(pos_input_dim, config.hidden_dim))

        for i in range(1, config.num_layers):
            if i in config.skip_connections:
                self.density_layers.append(
                    nn.Linear(config.hidden_dim + pos_input_dim, config.hidden_dim)
                )
            else:
                self.density_layers.append(nn.Linear(config.hidden_dim, config.hidden_dim))

        # Density output
        self.density_output = nn.Linear(config.hidden_dim, 1)

        # Feature vector for color network
        self.feature_layer = nn.Linear(config.hidden_dim, config.hidden_dim)

        # Color network - includes appearance and exposure conditioning
        color_input_dim = (
            config.hidden_dim + dir_input_dim + config.appearance_dim + config.exposure_dim
        )

        self.color_layers = nn.ModuleList(
            [
                nn.Linear(color_input_dim, config.hidden_dim // 2),
                nn.Linear(config.hidden_dim // 2, config.hidden_dim // 2),
            ]
        )

        # RGB output
        self.color_output = nn.Linear(config.hidden_dim // 2, 3)

        # Initialize weights
        self.apply(self._init_weights)

    def _init_weights(self, m: nn.Module) -> None:
        """Initialize network weights."""
        if isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    def encode_position(self, means: Tensor, covs: Tensor | None = None) -> Tensor:
        """Encode 3D positions with integrated encoding."""
        if self.config.use_integrated_encoding and covs is not None:
            encoded = integrated_positional_encoding(means, covs, self.config.pos_encoding_levels)
            # Include raw coordinates
            encoded = torch.cat([means, encoded], dim=-1)
        else:
            encoded = positional_encoding(means, self.config.pos_encoding_levels)
        return encoded

    def encode_direction(self, directions: Tensor) -> Tensor:
        """Encode viewing directions."""
        return positional_encoding(directions, self.config.dir_encoding_levels)

    def forward(
        self,
        positions: Tensor,
        directions: Tensor,
        appearance_embedding: Tensor,
        exposure: Tensor,
        position_covs: Tensor | None = None,
    ) -> TensorDict:
        """Forward pass through Block-NeRF network."""
        # Encode inputs
        encoded_pos = self.encode_position(positions, position_covs)
        encoded_dir = self.encode_direction(directions)

        # Density network forward pass
        x = encoded_pos
        for i, layer in enumerate(self.density_layers):
            if i in self.config.skip_connections and i > 0:
                x = torch.cat([x, encoded_pos], dim=-1)
            x = F.relu(layer(x))

        # Get density
        raw_density = self.density_output(x)
        if self.config.density_activation == "exp":
            density = torch.exp(raw_density)
        else:
            density = F.relu(raw_density)

        # Get feature vector for color
        feature = self.feature_layer(x)

        # Color network forward pass
        color_input = torch.cat([feature, encoded_dir, appearance_embedding, exposure], dim=-1)

        color_x = color_input
        for layer in self.color_layers:
            color_x = F.relu(layer(color_x))

        raw_color = self.color_output(color_x)
        if self.config.color_activation == "sigmoid":
            color = torch.sigmoid(raw_color)
        else:
            color = raw_color

        return {
            "density": density,
            "color": color,
            "raw_density": raw_density,
            "raw_color": raw_color,
        }


class BlockNeRFModel(nn.Module):
    """Main Block-NeRF model that manages individual blocks."""

    def __init__(self, config: BlockNeRFConfig):
        super().__init__()
        self.config = config
        self.device = torch.device(config.device)

        # Individual block neural network
        self.network = BlockNeRFNetwork(config)

        # Appearance embeddings
        self.appearance_embeddings = nn.Embedding(
            config.num_appearance_embeddings, config.appearance_embedding_dim
        )

        # Exposure encoding
        self.exposure_layers = nn.Sequential(
            nn.Linear(1, config.exposure_embedding_dim),
            nn.ReLU(),
            nn.Linear(config.exposure_embedding_dim, config.exposure_embedding_dim),
        )

        # Block center and radius
        self.register_buffer("block_center", torch.zeros(3))
        self.register_buffer("block_radius", torch.tensor(config.block_radius))

        # Initialize weights
        self.apply(self._init_weights)

    def _init_weights(self, m: nn.Module) -> None:
        """Initialize model weights."""
        if isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, std=0.01)

    def set_block_center(self, center: Tensor) -> None:
        """Set the center of this block."""
        self.block_center.copy_(center)

    def get_appearance_embedding(self, appearance_ids: Tensor) -> Tensor:
        """Get appearance embeddings for given IDs."""
        return self.appearance_embeddings(appearance_ids)

    def encode_exposure(self, exposure_values: Tensor) -> Tensor:
        """Encode exposure values."""
        return self.exposure_layers(exposure_values)

    def is_in_block(self, positions: Tensor) -> Tensor:
        """Check if positions are within this block's radius."""
        distances = torch.norm(positions - self.block_center, dim=-1)
        return distances <= self.block_radius

    def forward(
        self,
        positions: Tensor,
        directions: Tensor,
        appearance_ids: Tensor,
        exposure_values: Tensor,
        position_covs: Tensor | None = None,
    ) -> TensorDict:
        """Forward pass through Block-NeRF model."""
        # Get embeddings
        appearance_emb = self.get_appearance_embedding(appearance_ids)
        exposure_emb = self.encode_exposure(exposure_values)

        # Forward through network
        outputs = self.network(positions, directions, appearance_emb, exposure_emb, position_covs)

        # Add block information
        outputs["block_center"] = self.block_center.unsqueeze(0).expand(positions.shape[0], -1)
        outputs["in_block"] = self.is_in_block(positions)

        return outputs

    def get_block_info(self) -> dict[str, Tensor]:
        """Get information about this block."""
        return {
            "center": self.block_center,
            "radius": self.block_radius,
        }


class BlockNeRFLoss(nn.Module):
    """Loss functions for Block-NeRF training."""

    def __init__(self, config: BlockNeRFConfig):
        super().__init__()
        self.config = config

    def rgb_loss(self, pred_rgb: Tensor, target_rgb: Tensor) -> Tensor:
        """RGB reconstruction loss."""
        return F.mse_loss(pred_rgb, target_rgb)

    def depth_loss(self, pred_depth: Tensor, target_depth: Tensor) -> Tensor:
        """Depth loss (if depth supervision is available)."""
        valid_mask = target_depth > 0
        if valid_mask.sum() == 0:
            return torch.tensor(0.0, device=pred_depth.device)
        return F.mse_loss(pred_depth[valid_mask], target_depth[valid_mask])

    def eikonal_loss(self, gradients: Tensor) -> Tensor:
        """Eikonal loss for surface regularization."""
        gradient_norm = torch.norm(gradients, dim=-1)
        return F.mse_loss(gradient_norm, torch.ones_like(gradient_norm))

    def forward(
        self,
        predictions: TensorDict,
        targets: TensorDict,
        gradients: Tensor | None = None,
    ) -> TensorDict:
        """Compute total loss."""
        losses = {}

        # RGB loss
        rgb_loss = self.rgb_loss(predictions["rgb"], targets["rgb"])
        losses["rgb_loss"] = rgb_loss

        # Depth loss (optional)
        if "depth" in targets:
            depth_loss = self.depth_loss(predictions["depth"], targets["depth"])
            losses["depth_loss"] = depth_loss
        else:
            losses["depth_loss"] = torch.tensor(0.0, device=rgb_loss.device)

        # Eikonal loss (optional)
        if gradients is not None:
            eikonal_loss = self.eikonal_loss(gradients)
            losses["eikonal_loss"] = eikonal_loss
        else:
            losses["eikonal_loss"] = torch.tensor(0.0, device=rgb_loss.device)

        # Total loss
        total_loss = (
            self.config.rgb_loss_weight * losses["rgb_loss"]
            + self.config.depth_loss_weight * losses["depth_loss"]
            + self.config.eikonal_loss_weight * losses["eikonal_loss"]
        )

        losses["total_loss"] = total_loss
        return losses


# Utility functions for device management
def get_device_info() -> dict[str, Union[bool, int, str]]:
    """Get device information."""
    return {
        "cuda_available": torch.cuda.is_available(),
        "device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
        "current_device": torch.cuda.current_device() if torch.cuda.is_available() else -1,
        "device_name": torch.cuda.get_device_name() if torch.cuda.is_available() else "CPU",
    }


def check_compatibility() -> None:
    """Check system compatibility for Block-NeRF."""
    device_info = get_device_info()
    print("Block-NeRF System Compatibility Check")
    print("=" * 40)
    print(f"CUDA Available: {device_info['cuda_available']}")
    print(f"GPU Devices: {device_info['device_count']}")
    if device_info["cuda_available"]:
        print(f"Current Device: {device_info['current_device']}")
        print(f"Device Name: {device_info['device_name']}")
    print(f"PyTorch Version: {torch.__version__}")
    print("=" * 40)


# Package constants
__version__ = "1.0.0"
CUDA_AVAILABLE = torch.cuda.is_available()
