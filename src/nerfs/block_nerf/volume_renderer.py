"""
Volume Renderer for Block-NeRF Training

This module provides volume rendering functionality for Block-NeRF training,
following the pattern established in SVRaster where volume rendering is used
during training for stability.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .core import BlockNeRFConfig, BlockNeRFModel

# Type aliases
Tensor = torch.Tensor
TensorDict = dict[str, Tensor]


@dataclass
class VolumeRendererConfig:
    """Configuration for Block-NeRF volume renderer."""

    # Sampling configuration
    num_samples: int = 64
    num_importance_samples: int = 64
    perturb: bool = True
    raw_noise_std: float = 1.0

    # Ray configuration
    near_plane: float = 0.1
    far_plane: float = 1000.0
    use_viewdirs: bool = True

    # Hierarchical sampling
    use_hierarchical_sampling: bool = True
    sample_at_infinity: bool = False

    # Performance
    chunk_size: int = 1024 * 32
    use_gradient_checkpointing: bool = False

    # White background
    white_background: bool = True


class VolumeRenderer(nn.Module):
    """
    Volume Renderer for Block-NeRF training.

    This component is tightly coupled with training and provides stable
    volume rendering for the training phase.
    """

    def __init__(self, config: VolumeRendererConfig):
        super().__init__()
        self.config = config

    def sample_rays(
        self,
        ray_origins: Tensor,
        ray_directions: Tensor,
        near: Tensor | None = None,
        far: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        """Sample points along rays."""
        batch_size = ray_origins.shape[0]
        device = ray_origins.device

        # Use provided near/far or defaults
        if near is None:
            near = torch.full((batch_size,), self.config.near_plane, device=device)
        if far is None:
            far = torch.full((batch_size,), self.config.far_plane, device=device)

        # Sample t values
        t_vals = torch.linspace(0.0, 1.0, self.config.num_samples, device=device)
        t_vals = near.unsqueeze(-1) + (far - near).unsqueeze(-1) * t_vals

        # Perturb samples
        if self.config.perturb and self.training:
            mids = 0.5 * (t_vals[..., 1:] + t_vals[..., :-1])
            upper = torch.cat([mids, t_vals[..., -1:]], dim=-1)
            lower = torch.cat([t_vals[..., :1], mids], dim=-1)
            t_rand = torch.rand_like(t_vals)
            t_vals = lower + (upper - lower) * t_rand

        # Compute sample positions
        positions = ray_origins.unsqueeze(-2) + ray_directions.unsqueeze(-2) * t_vals.unsqueeze(-1)

        return positions, t_vals

    def hierarchical_sample(
        self,
        ray_origins: Tensor,
        ray_directions: Tensor,
        t_vals: Tensor,
        weights: Tensor,
    ) -> tuple[Tensor, Tensor]:
        """Hierarchical sampling based on coarse weights."""
        # Get pdf from weights
        weights = weights + 1e-5  # Prevent division by zero
        pdf = weights / torch.sum(weights, dim=-1, keepdim=True)
        cdf = torch.cumsum(pdf, dim=-1)
        cdf = torch.cat([torch.zeros_like(cdf[..., :1]), cdf], dim=-1)

        # Sample from CDF
        if self.config.perturb and self.training:
            u = torch.rand(
                list(cdf.shape[:-1]) + [self.config.num_importance_samples], device=cdf.device
            )
        else:
            u = torch.linspace(0.0, 1.0, self.config.num_importance_samples, device=cdf.device)
            u = u.expand(list(cdf.shape[:-1]) + [self.config.num_importance_samples])

        # Invert CDF
        u = u.contiguous()
        indices = torch.searchsorted(cdf, u, right=True)
        below = torch.clamp(indices - 1, 0, cdf.shape[-1] - 1)
        above = torch.clamp(indices, 0, cdf.shape[-1] - 1)

        indices_g = torch.stack([below, above], dim=-1)
        matched_shape = list(indices_g.shape[:-1]) + [cdf.shape[-1]]
        cdf_g = torch.gather(cdf.unsqueeze(-2).expand(matched_shape), dim=-1, index=indices_g)
        bins_g = torch.gather(t_vals.unsqueeze(-2).expand(matched_shape), dim=-1, index=indices_g)

        denom = cdf_g[..., 1] - cdf_g[..., 0]
        denom = torch.where(denom < 1e-5, torch.ones_like(denom), denom)
        t = (u - cdf_g[..., 0]) / denom

        t_samples = bins_g[..., 0] + t * (bins_g[..., 1] - bins_g[..., 0])

        # Combine with original samples
        t_vals_combined = torch.cat([t_vals, t_samples], dim=-1)
        t_vals_combined, _ = torch.sort(t_vals_combined, dim=-1)

        # Compute new positions
        positions = ray_origins.unsqueeze(-2) + ray_directions.unsqueeze(
            -2
        ) * t_vals_combined.unsqueeze(-1)

        return positions, t_vals_combined

    def compute_weights(
        self,
        densities: Tensor,
        t_vals: Tensor,
    ) -> tuple[Tensor, Tensor]:
        """Compute weights from densities and t values."""
        # Compute deltas
        dists = t_vals[..., 1:] - t_vals[..., :-1]
        dists = torch.cat(
            [dists, torch.full_like(dists[..., :1], 1e10)], dim=-1  # Last distance is infinity
        )

        # Add noise during training
        if self.training and self.config.raw_noise_std > 0:
            noise = torch.randn_like(densities) * self.config.raw_noise_std
            densities = densities + noise

        # Compute alpha values
        alpha = 1.0 - torch.exp(-densities * dists)

        # Compute transmittance
        transmittance = torch.cumprod(
            torch.cat([torch.ones_like(alpha[..., :1]), 1.0 - alpha[..., :-1] + 1e-10], dim=-1),
            dim=-1,
        )

        # Compute weights
        weights = alpha * transmittance

        return weights, alpha

    def render_rays(
        self,
        model: BlockNeRFModel,
        ray_origins: Tensor,
        ray_directions: Tensor,
        appearance_ids: Tensor,
        exposure_values: Tensor,
        near: Tensor | None = None,
        far: Tensor | None = None,
        position_covs: Tensor | None = None,
    ) -> TensorDict:
        """Render rays using volume rendering."""
        # Sample points along rays
        positions, t_vals = self.sample_rays(ray_origins, ray_directions, near, far)

        # Prepare inputs for model
        batch_size, num_samples = positions.shape[:2]
        positions_flat = positions.reshape(-1, 3)
        directions_flat = ray_directions.unsqueeze(-2).expand(-1, num_samples, -1).reshape(-1, 3)
        appearance_ids_flat = appearance_ids.unsqueeze(-1).expand(-1, num_samples).reshape(-1)
        exposure_values_flat = (
            exposure_values.unsqueeze(-2).expand(-1, num_samples, -1).reshape(-1, 1)
        )

        if position_covs is not None:
            position_covs_flat = position_covs.reshape(-1, position_covs.shape[-1])
        else:
            position_covs_flat = None

        # Forward pass through model
        outputs = model(
            positions_flat,
            directions_flat,
            appearance_ids_flat,
            exposure_values_flat,
            position_covs_flat,
        )

        # Reshape outputs
        densities = outputs["density"].reshape(batch_size, num_samples)
        colors = outputs["color"].reshape(batch_size, num_samples, 3)

        # Compute weights
        weights, alpha = self.compute_weights(densities, t_vals)

        # Render RGB
        rgb = torch.sum(weights.unsqueeze(-1) * colors, dim=-2)

        # Render depth
        depth = torch.sum(weights * t_vals, dim=-1)

        # Render accumulation
        acc = torch.sum(weights, dim=-1)

        # Add white background if enabled
        if self.config.white_background:
            rgb = rgb + (1.0 - acc.unsqueeze(-1))

        result = {
            "rgb": rgb,
            "depth": depth,
            "acc": acc,
            "weights": weights,
            "alpha": alpha,
            "t_vals": t_vals,
        }

        # Hierarchical sampling
        if self.config.use_hierarchical_sampling and self.config.num_importance_samples > 0:
            # Sample fine points
            fine_positions, fine_t_vals = self.hierarchical_sample(
                ray_origins, ray_directions, t_vals, weights
            )

            # Forward pass for fine samples
            fine_batch_size, fine_num_samples = fine_positions.shape[:2]
            fine_positions_flat = fine_positions.reshape(-1, 3)
            fine_directions_flat = (
                ray_directions.unsqueeze(-2).expand(-1, fine_num_samples, -1).reshape(-1, 3)
            )
            fine_appearance_ids_flat = (
                appearance_ids.unsqueeze(-1).expand(-1, fine_num_samples).reshape(-1)
            )
            fine_exposure_values_flat = (
                exposure_values.unsqueeze(-2).expand(-1, fine_num_samples, -1).reshape(-1, 1)
            )

            if position_covs is not None:
                fine_position_covs_flat = position_covs.reshape(-1, position_covs.shape[-1])
            else:
                fine_position_covs_flat = None

            fine_outputs = model(
                fine_positions_flat,
                fine_directions_flat,
                fine_appearance_ids_flat,
                fine_exposure_values_flat,
                fine_position_covs_flat,
            )

            # Reshape fine outputs
            fine_densities = fine_outputs["density"].reshape(fine_batch_size, fine_num_samples)
            fine_colors = fine_outputs["color"].reshape(fine_batch_size, fine_num_samples, 3)

            # Compute fine weights
            fine_weights, fine_alpha = self.compute_weights(fine_densities, fine_t_vals)

            # Render fine RGB
            fine_rgb = torch.sum(fine_weights.unsqueeze(-1) * fine_colors, dim=-2)
            fine_depth = torch.sum(fine_weights * fine_t_vals, dim=-1)
            fine_acc = torch.sum(fine_weights, dim=-1)

            # Add white background if enabled
            if self.config.white_background:
                fine_rgb = fine_rgb + (1.0 - fine_acc.unsqueeze(-1))

            # Update result with fine samples
            result.update(
                {
                    "fine_rgb": fine_rgb,
                    "fine_depth": fine_depth,
                    "fine_acc": fine_acc,
                    "fine_weights": fine_weights,
                    "fine_alpha": fine_alpha,
                    "fine_t_vals": fine_t_vals,
                }
            )

            # Use fine results as main output
            result["rgb"] = fine_rgb
            result["depth"] = fine_depth
            result["acc"] = fine_acc

        return result

    def forward(
        self,
        model: BlockNeRFModel,
        ray_origins: Tensor,
        ray_directions: Tensor,
        appearance_ids: Tensor,
        exposure_values: Tensor,
        **kwargs,
    ) -> TensorDict:
        """Forward pass through volume renderer."""
        return self.render_rays(
            model, ray_origins, ray_directions, appearance_ids, exposure_values, **kwargs
        )


def create_volume_renderer(config: VolumeRendererConfig | None = None) -> VolumeRenderer:
    """Create a volume renderer with default configuration."""
    if config is None:
        config = VolumeRendererConfig()
    return VolumeRenderer(config)
