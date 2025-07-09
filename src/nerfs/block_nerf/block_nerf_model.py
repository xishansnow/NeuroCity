from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

"""
Block-NeRF Model Implementation

This module contains the core Block-NeRF neural network architecture
based on mip-NeRF with extensions for appearance embeddings, pose refinement, and exposure conditioning.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math

# Type aliases
Tensor = torch.Tensor
Device = Union[torch.device, str]
DType = torch.dtype
TensorDict = dict[str, Tensor]

# Generic types
T = TypeVar("T")
Module = TypeVar("Module", bound=nn.Module)


def positional_encoding(x: Tensor, L: int) -> Tensor:
    """
    Positional encoding for input coordinates

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
    """
    Integrated positional encoding for mip-NeRF

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
    """
    Block-NeRF Neural Network based on mip-NeRF architecture
    """

    def __init__(
        self,
        pos_encoding_levels: int = 16,
        dir_encoding_levels: int = 4,
        appearance_dim: int = 32,
        exposure_dim: int = 8,
        hidden_dim: int = 256,
        num_layers: int = 8,
        skip_connections: list[int] | None = None,
        use_integrated_encoding: bool = True,
    ):
        super().__init__()

        self.pos_encoding_levels = pos_encoding_levels
        self.dir_encoding_levels = dir_encoding_levels
        self.appearance_dim = appearance_dim
        self.exposure_dim = exposure_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.skip_connections = skip_connections or [4]
        self.use_integrated_encoding = use_integrated_encoding

        # Input dimensions
        if use_integrated_encoding:
            pos_input_dim = 3 + 3 * 2 * pos_encoding_levels  # position + encoding
        else:
            pos_input_dim = 3 * 2 * pos_encoding_levels

        dir_input_dim = 3 * 2 * dir_encoding_levels

        # Density network (f_sigma)
        self.density_layers = nn.ModuleList()
        self.density_layers.append(nn.Linear(pos_input_dim, hidden_dim))

        for i in range(1, num_layers):
            if i in self.skip_connections:
                self.density_layers.append(nn.Linear(hidden_dim + pos_input_dim, hidden_dim))
            else:
                self.density_layers.append(nn.Linear(hidden_dim, hidden_dim))

        # Density output
        self.density_output = nn.Linear(hidden_dim, 1)

        # Feature vector for color network
        self.feature_layer = nn.Linear(hidden_dim, hidden_dim)

        # Color network (f_c) - includes appearance and exposure conditioning
        color_input_dim = hidden_dim + dir_input_dim + appearance_dim + exposure_dim

        self.color_layers = nn.ModuleList(
            [
                nn.Linear(color_input_dim, hidden_dim // 2),
                nn.Linear(hidden_dim // 2, hidden_dim // 2),
            ]
        )

        # RGB output
        self.color_output = nn.Linear(hidden_dim // 2, 3)

        # Initialize weights
        self.apply(self._init_weights)

    def _init_weights(self, m: nn.Module) -> None:
        """Initialize network weights.

        Args:
            m: Module to initialize
        """
        if isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    def encode_position(
        self,
        means: Tensor,
        covs: Tensor | None = None,
    ) -> Tensor:
        """Encode 3D positions with integrated encoding.

        Args:
            means: Mean positions (..., 3)
            covs: Optional position covariances (..., 3, 3)

        Returns:
            Encoded positions
        """
        if self.use_integrated_encoding and covs is not None:
            encoded = integrated_positional_encoding(means, covs, self.pos_encoding_levels)
            if self.use_integrated_encoding:
                # Include raw coordinates
                encoded = torch.cat([means, encoded], dim=-1)
        else:
            encoded = positional_encoding(means, self.pos_encoding_levels)
        return encoded

    def encode_direction(self, directions: Tensor) -> Tensor:
        """Encode viewing directions.

        Args:
            directions: Viewing directions (..., 3)

        Returns:
            Encoded directions
        """
        return positional_encoding(directions, self.dir_encoding_levels)

    def forward(
        self,
        positions: Tensor,
        directions: Tensor,
        appearance_embedding: Tensor,
        exposure: Tensor,
        position_covs: Tensor | None = None,
    ) -> TensorDict:
        """Forward pass through Block-NeRF network.

        Args:
            positions: 3D positions (..., 3)
            directions: Viewing directions (..., 3)
            appearance_embedding: Appearance codes (..., appearance_dim)
            exposure: Exposure values (..., exposure_dim)
            position_covs: Optional position covariances (..., 3, 3)

        Returns:
            Dictionary with 'density', 'color', and 'features' outputs
        """
        # Move inputs to device efficiently
        device = positions.device
        positions = positions.to(device, non_blocking=True)
        directions = directions.to(device, non_blocking=True)
        appearance_embedding = appearance_embedding.to(device, non_blocking=True)
        exposure = exposure.to(device, non_blocking=True)
        if position_covs is not None:
            position_covs = position_covs.to(device, non_blocking=True)

        # Encode inputs
        encoded_pos = self.encode_position(positions, position_covs)
        encoded_dir = self.encode_direction(directions)

        # Density network forward pass with skip connections
        x = encoded_pos
        for i, layer in enumerate(self.density_layers):
            if i in self.skip_connections and i > 0:
                x = torch.cat([x, encoded_pos], dim=-1)
            x = F.relu(layer(x))

        # Density output with noise for regularization
        density = F.relu(self.density_output(x))
        if self.training:
            noise = torch.randn_like(density) * 0.1
            density = density + noise

        # Feature vector for color network
        features = self.feature_layer(x)

        # Color network input
        color_input = torch.cat([features, encoded_dir, appearance_embedding, exposure], dim=-1)

        # Color network forward pass
        x = color_input
        for layer in self.color_layers:
            x = F.relu(layer(x))

        # Color output
        color = torch.sigmoid(self.color_output(x))

        return {"density": density, "color": color, "features": features}


class BlockNeRF(nn.Module):
    """
    Complete Block-NeRF model with all components
    """

    def __init__(
        self,
        network_config: dict,
        block_center: Tensor,
        block_radius: float,
        num_appearance_embeddings: int = 1000,
        appearance_dim: int = 32,
        exposure_dim: int = 8,
        device: Device | None = None,
    ):
        super().__init__()

        self.block_center = nn.Parameter(block_center, requires_grad=False)
        self.block_radius = block_radius
        self.appearance_dim = appearance_dim
        self.exposure_dim = exposure_dim

        # Main NeRF network
        self.network = BlockNeRFNetwork(
            appearance_dim=appearance_dim, exposure_dim=exposure_dim, **network_config
        )

        # Appearance embeddings
        self.appearance_embeddings = nn.Embedding(num_appearance_embeddings, appearance_dim)

        # Exposure encoding
        self.exposure_encoding = nn.Linear(1, exposure_dim)

        # Initialize embeddings
        nn.init.normal_(self.appearance_embeddings.weight, 0, 0.01)
        nn.init.xavier_uniform_(self.exposure_encoding.weight)

        # Initialize AMP scaler
        self.scaler = GradScaler()

        # Move model to device
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(self.device)

    def encode_exposure(self, exposure_values: Tensor) -> Tensor:
        """
        Encode exposure values

        Args:
            exposure_values: Raw exposure values (..., 1)

        Returns:
            Encoded exposure (..., exposure_dim)
        """
        # Move to device efficiently
        exposure_values = exposure_values.to(self.device, non_blocking=True)

        # Normalize exposure values
        mean = 0.5
        std = 2.0
        normalized = (exposure_values - mean) / std

        # Create exposure encoding
        encoding = []
        for i in range(self.exposure_dim):
            encoding.append(torch.sin(2**i * math.pi * normalized))
            encoding.append(torch.cos(2**i * math.pi * normalized))

        return torch.cat(encoding, dim=-1)

    def get_appearance_embedding(self, appearance_ids: Tensor) -> Tensor:
        """Get appearance embeddings for given IDs"""
        return self.appearance_embeddings(appearance_ids.to(self.device))

    def is_in_block(self, positions: Tensor) -> Tensor:
        """Check if positions are within this block's radius"""
        distances = torch.norm(positions - self.block_center, dim=-1)
        return distances <= self.block_radius

    def forward(
        self,
        positions: Tensor,
        directions: Tensor,
        appearance_ids: Tensor,
        exposure_values: Tensor,
        position_covs: Tensor | None = None,
    ):
        """
        Forward pass through complete Block-NeRF

        Args:
            positions: 3D positions (..., 3)
            directions: Viewing directions (..., 3)
            appearance_ids: Appearance embedding IDs (..., )
            exposure_values: Exposure values (..., 1)
            position_covs: Position covariances (..., 3, 3)

        Returns:
            Dictionary with network outputs
        """
        # Get embeddings
        appearance_emb = self.get_appearance_embedding(appearance_ids)
        exposure_emb = self.encode_exposure(exposure_values)

        # Expand embeddings to match position dimensions
        if appearance_emb.dim() == 2 and positions.dim() > 2:
            # Expand for batch processing
            shape = list(positions.shape[:-1]) + [self.appearance_dim]
            appearance_emb = appearance_emb.view(-1, self.appearance_dim).expand(shape)

        if exposure_emb.dim() == 2 and positions.dim() > 2:
            shape = list(positions.shape[:-1]) + [self.exposure_dim]
            exposure_emb = exposure_emb.view(-1, self.exposure_dim).expand(shape)

        # Forward through network
        outputs = self.network(
            positions=positions,
            directions=directions,
            appearance_embedding=appearance_emb,
            exposure=exposure_emb,
            position_covs=position_covs,
        )

        # Add block information
        outputs["block_center"] = self.block_center
        outputs["block_radius"] = self.block_radius
        outputs["in_block"] = self.is_in_block(positions)

        return outputs

    def get_block_info(self) -> dict[str, Union[Tensor, float]]:
        """Get block metadata"""
        return {
            "center": self.block_center,
            "radius": self.block_radius,
            "appearance_dim": self.appearance_dim,
            "exposure_dim": self.exposure_dim,
        }

    def render_rays(
        self,
        rays_o: Tensor,
        rays_d: Tensor,
        near: float,
        far: float,
        appearance_embedding: Optional[Tensor] = None,
        **kwargs,
    ) -> dict[str, Tensor]:
        """Render rays"""
        # Implementation of render_rays method
        pass

    def forward(
        self,
        rays_o: Tensor,
        rays_d: Tensor,
        appearance_embedding: Optional[Tensor] = None,
        **kwargs,
    ) -> dict[str, Tensor]:
        """Forward pass for rays"""
        # Implementation of forward method for rays
        pass
