from __future__ import annotations

from typing import Any, Optional, Union

"""
BungeeNeRF Core Module
Implements the main BungeeNeRF model with progressive training blocks
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
import numpy as np
import logging

from .progressive_encoder import ProgressivePositionalEncoder
from .multiscale_renderer import MultiScaleRenderer

logger = logging.getLogger(__name__)


@dataclass
class BungeeNeRFConfig:
    """Configuration for BungeeNeRF model"""

    # Progressive structure
    num_stages: int = 4
    base_resolution: int = 16
    max_resolution: int = 2048
    scale_factor: float = 4.0

    # Positional encoding
    num_freqs_base: int = 4
    num_freqs_max: int = 10
    include_input: bool = True

    # MLP architecture
    hidden_dim: int = 256
    num_layers: int = 8
    skip_layers: List[int] = None

    # Progressive blocks
    block_hidden_dim: int = 128
    block_num_layers: int = 4

    # Training parameters
    batch_size: int = 4096
    learning_rate: float = 5e-4
    max_steps: int = 200000

    # Sampling
    num_samples: int = 64
    num_importance: int = 128
    perturb: bool = True

    # Multi-scale parameters
    scale_weights: List[float] = None
    distance_thresholds: List[float] = None

    # Loss weights
    color_loss_weight: float = 1.0
    depth_loss_weight: float = 0.1
    progressive_loss_weight: float = 0.05

    def __post_init__(self):
        if self.skip_layers is None:
            self.skip_layers = [4]
        if self.scale_weights is None:
            self.scale_weights = [1.0, 0.8, 0.6, 0.4]
        if self.distance_thresholds is None:
            self.distance_thresholds = [100.0, 50.0, 25.0, 10.0]


class ProgressiveBlock(nn.Module):
    """
    Progressive block that gets added during training
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 128,
        num_layers: int = 4,
        output_dim: int = 3,
        activation: str = "relu",
    ):
        super().__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.output_dim = output_dim

        # Activation function
        if activation == "relu":
            self.activation = nn.ReLU(inplace=True)
        elif activation == "leaky_relu":
            self.activation = nn.LeakyReLU(0.2, inplace=True)
        else:
            self.activation = nn.ReLU(inplace=True)

        # Build layers
        layers = []
        for i in range(num_layers):
            if i == 0:
                layers.append(nn.Linear(input_dim, hidden_dim))
            elif i == num_layers - 1:
                layers.append(nn.Linear(hidden_dim, output_dim))
            else:
                layers.append(nn.Linear(hidden_dim, hidden_dim))

            if i < num_layers - 1:
                layers.append(self.activation)

        self.layers = nn.Sequential(*layers)

        # Initialize weights
        self._initialize_weights()

    def _initialize_weights(self):
        """Initialize network weights"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through progressive block

        Args:
            x: Input features [N, input_dim]

        Returns:
            Output features [N, output_dim]
        """
        return self.layers(x)


class BungeeNeRF(nn.Module):
    """
    BungeeNeRF model with progressive training
    """

    def __init__(self, config: BungeeNeRFConfig):
        super().__init__()

        self.config = config
        self.current_stage = 0

        # Progressive positional encoder
        self.pos_encoder = ProgressivePositionalEncoder(
            num_freqs_base=config.num_freqs_base,
            num_freqs_max=config.num_freqs_max,
            include_input=config.include_input,
        )

        # Base MLP network
        pos_dim = self.pos_encoder.get_output_dim()
        self.base_mlp = self._build_base_mlp(pos_dim, config)

        # Progressive blocks (added during training)
        self.progressive_blocks = nn.ModuleList()

        # Multi-scale renderer
        self.renderer = MultiScaleRenderer(config)

        logger.info(f"BungeeNeRF initialized with {self.count_parameters():, } parameters")

    def _build_base_mlp(self, input_dim: int, config: BungeeNeRFConfig) -> nn.Module:
        """Build base MLP network"""

        layers = []

        # Input layer
        layers.append(nn.Linear(input_dim, config.hidden_dim))
        layers.append(nn.ReLU(inplace=True))

        # Hidden layers
        for i in range(1, config.num_layers):
            if i in config.skip_layers:
                # Skip connection
                layers.append(nn.Linear(config.hidden_dim + input_dim, config.hidden_dim))
            else:
                layers.append(nn.Linear(config.hidden_dim, config.hidden_dim))

            if i < config.num_layers - 1:
                layers.append(nn.ReLU(inplace=True))

        # Output layers
        # RGB output
        rgb_layer = nn.Linear(config.hidden_dim, 3)
        # Density output
        sigma_layer = nn.Linear(config.hidden_dim, 1)

        return nn.ModuleDict(
            {"backbone": nn.Sequential(*layers), "rgb_head": rgb_layer, "sigma_head": sigma_layer}
        )

    def add_progressive_block(self, stage: int):
        """Add a new progressive block for the given stage"""

        if stage >= len(self.progressive_blocks):
            # Calculate input dimension for this stage
            base_features = self.config.hidden_dim

            # Create new progressive block
            block = ProgressiveBlock(
                input_dim=base_features,
                hidden_dim=self.config.block_hidden_dim,
                num_layers=self.config.block_num_layers,
                output_dim=3,  # RGB output
            )

            self.progressive_blocks.append(block)
            logger.info(f"Added progressive block for stage {stage}")

    def set_current_stage(self, stage: int):
        """Set current training stage"""
        self.current_stage = stage

        # Add progressive blocks up to current stage
        for s in range(stage + 1):
            if s >= len(self.progressive_blocks):
                self.add_progressive_block(s)

        # Update positional encoder
        self.pos_encoder.set_current_stage(stage)

        logger.info(f"Set current stage to {stage}")

    def forward(
        self,
        rays_o: torch.Tensor,
        rays_d: torch.Tensor,
        bounds: torch.Tensor,
        distances: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass through BungeeNeRF

        Args:
            rays_o: Ray origins [N, 3]
            rays_d: Ray directions [N, 3]
            bounds: Near/far bounds [N, 2]
            distances: Distance to camera [N] (optional)

        Returns:
            Dictionary containing rendered outputs
        """
        batch_size = rays_o.shape[0]
        device = rays_o.device

        # Sample points along rays
        z_vals = self._sample_along_rays(rays_o, rays_d, bounds)

        # Get sample points
        pts = rays_o[..., None, :] + rays_d[..., None, :] * z_vals[..., :, None]
        pts_flat = pts.reshape(-1, 3)

        # Progressive positional encoding
        pts_encoded = self.pos_encoder(pts_flat)

        # Forward through base MLP
        features = self._forward_base_mlp(pts_encoded)

        # Get base RGB and density
        rgb_base = torch.sigmoid(features["rgb"])
        sigma_base = F.relu(features["sigma"])

        # Progressive refinement
        rgb_final = rgb_base

        # Apply progressive blocks based on distance
        if distances is not None and len(self.progressive_blocks) > 0:
            # Expand distances to match the number of samples per ray
            num_samples = rgb_base.shape[0] // batch_size
            distances_expanded = distances.repeat_interleave(num_samples)

            # Determine which blocks to use based on distance
            for stage, block in enumerate(self.progressive_blocks):
                if stage <= self.current_stage:
                    # Apply block based on distance threshold
                    threshold = self.config.distance_thresholds[
                        min(stage, len(self.config.distance_thresholds))
                    ]

                    mask = distances_expanded < threshold

                    if mask.any():
                        # Apply progressive block
                        block_features = features["backbone"][mask]

                        if block_features.shape[0] > 0:
                            block_rgb = torch.sigmoid(block(block_features))

                            # Blend with base RGB
                            weight = self.config.scale_weights[
                                min(stage, len(self.config.scale_weights))
                            ]
                            rgb_final[mask] = rgb_final[mask] * (1 - weight) + block_rgb * weight

        # Reshape back to ray format
        rgb_final = rgb_final.reshape(batch_size, -1, 3)
        sigma_final = sigma_base.reshape(batch_size, -1)

        # Volume rendering
        outputs = self.renderer.render(rgb_final, sigma_final, z_vals, rays_d)

        return outputs

    def _forward_base_mlp(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Forward pass through base MLP"""

        # Store input for skip connections
        input_x = x

        # Forward through backbone
        features = x
        for i, layer in enumerate(self.base_mlp["backbone"]):
            if isinstance(layer, nn.Linear) and layer.in_features > self.config.hidden_dim:
                # Skip connection layer
                features = torch.cat([features, input_x], dim=-1)

            features = layer(features)

        # Get outputs
        rgb = self.base_mlp["rgb_head"](features)
        sigma = self.base_mlp["sigma_head"](features)

        return {"backbone": features, "rgb": rgb, "sigma": sigma}

    def _sample_along_rays(
        self, rays_o: torch.Tensor, rays_d: torch.Tensor, bounds: torch.Tensor
    ) -> torch.Tensor:
        """Sample points along rays"""

        batch_size = rays_o.shape[0]
        device = rays_o.device

        # Get near and far bounds
        near = bounds[..., 0]
        far = bounds[..., 1]

        # Sample depths
        t_vals = torch.linspace(0.0, 1.0, self.config.num_samples, device=device)
        z_vals = near[..., None] * (1.0 - t_vals) + far[..., None] * t_vals

        # Add perturbation during training
        if self.training and self.config.perturb:
            # Get intervals between samples
            mids = 0.5 * (z_vals[..., 1:] + z_vals[..., :-1])
            upper = torch.cat([mids, z_vals[..., -1:]], dim=-1)
            lower = torch.cat([z_vals[..., :1], mids], dim=-1)

            # Stratified sampling
            t_rand = torch.rand_like(z_vals)
            z_vals = lower + (upper - lower) * t_rand

        return z_vals

    def count_parameters(self) -> int:
        """Count total number of parameters"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def get_progressive_info(self) -> Dict[str, any]:
        """Get information about progressive structure"""
        return {
            "current_stage": self.current_stage,
            "num_stages": self.config.num_stages,
            "num_progressive_blocks": len(self.progressive_blocks),
        }


class BungeeNeRFLoss(nn.Module):
    """
    Loss function for BungeeNeRF training
    """

    def __init__(self, config: BungeeNeRFConfig):
        super().__init__()
        self.config = config

    def forward(
        self, outputs: Dict[str, torch.Tensor], targets: Dict[str, torch.Tensor], stage: int = 0
    ) -> Dict[str, torch.Tensor]:
        """
        Compute BungeeNeRF loss

        Args:
            outputs: Model outputs
            targets: Ground truth targets
            stage: Current training stage

        Returns:
            Dictionary of loss values
        """
        losses = {}

        # Color loss
        color_loss = F.mse_loss(outputs["rgb"], targets["rgb"])
        losses["color_loss"] = color_loss

        # Depth loss (if available)
        if "depth" in targets and "depth" in outputs:
            depth_loss = F.smooth_l1_loss(outputs["depth"], targets["depth"])
            losses["depth_loss"] = depth_loss
        else:
            losses["depth_loss"] = torch.tensor(0.0, device=outputs["rgb"].device)

        # Progressive loss (encourage smooth transitions)
        if stage > 0:
            # Regularization on progressive blocks
            progressive_loss = torch.tensor(0.0, device=outputs["rgb"].device)

            # Add smoothness regularization if needed
            if "weights" in outputs:
                weights = outputs["weights"]
                # Encourage smooth weight distributions
                weight_var = torch.var(weights, dim=-1).mean()
                progressive_loss = progressive_loss + weight_var

            losses["progressive_loss"] = progressive_loss
        else:
            losses["progressive_loss"] = torch.tensor(0.0, device=outputs["rgb"].device)

        # Total loss
        total_loss = (
            self.config.color_loss_weight * losses["color_loss"]
            + self.config.depth_loss_weight * losses["depth_loss"]
            + self.config.progressive_loss_weight * losses["progressive_loss"]
        )

        losses["total_loss"] = total_loss

        return losses
