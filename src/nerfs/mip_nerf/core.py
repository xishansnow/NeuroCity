from __future__ import annotations
from typing import Any, Optional, TypeVar, TypedDict, cast

"""
Core Mip-NeRF implementation

This module implements the core components of Mip-NeRF including:
- Integrated Positional Encoding (IPE)
- Conical frustum representation
- Multi-scale MLP architecture
- Volumetric rendering with anti-aliasing
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
import numpy as np
import math
from torch.amp.autocast_mode import autocast
from torch.amp.grad_scaler import GradScaler
from pathlib import Path
from typing import TypeAlias, Any

Tensor: TypeAlias = torch.Tensor
Device: TypeAlias = torch.device | str
DType: TypeAlias = torch.dtype
TensorDict: TypeAlias = Dict[str, Tensor]

TensorType = TypeVar("TensorType", bound=torch.Tensor)


class RenderOutput(TypedDict):
    """Type definition for render output dictionary"""

    rgb: torch.Tensor
    depth: torch.Tensor
    acc_alpha: torch.Tensor
    weights: torch.Tensor


class NetworkOutput(TypedDict):
    """Type definition for network output dictionary"""

    density: torch.Tensor
    rgb: torch.Tensor


class PredictionOutput(TypedDict):
    """Type definition for prediction output dictionary"""

    coarse: RenderOutput
    fine: Optional[RenderOutput]


class LossOutput(TypedDict):
    """Type definition for loss output dictionary"""

    coarse_loss: torch.Tensor
    fine_loss: Optional[torch.Tensor]
    total_loss: torch.Tensor
    psnr: torch.Tensor


@dataclass
class MipNeRFConfig:
    """Configuration for Mip-NeRF model with modern features."""

    # Network architecture
    hidden_dim: int = 256
    num_layers: int = 8
    skip_connections: List[int] = (4,)
    activation: str = "relu"
    output_activation: str = "sigmoid"

    # Integrated positional encoding
    min_deg_point: int = 0
    max_deg_point: int = 16
    min_deg_view: int = 0
    max_deg_view: int = 4

    # Rendering settings
    num_coarse_samples: int = 128
    num_fine_samples: int = 128
    disparity_space_sampling: bool = True
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
        assert self.num_coarse_samples > 0, "Number of coarse samples must be positive"
        assert self.num_fine_samples > 0, "Number of fine samples must be positive"
        assert self.near_plane < self.far_plane, "Near plane must be less than far plane"

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


class IntegratedPositionalEncoder(nn.Module):
    """
    Integrated Positional Encoding (IPE) for Mip-NeRF

    Instead of encoding individual points, IPE encodes entire conical frustums
    by integrating the positional encoding over the volume of the frustum.
    """

    def __init__(self, min_deg: int, max_deg: int):
        super().__init__()
        self.min_deg = min_deg
        self.max_deg = max_deg
        self.num_levels = max_deg - min_deg

    def expected_sin(self, means: torch.Tensor, vars: torch.Tensor) -> torch.Tensor:
        """Expected value of sin(x) where x ~ N(means, vars)"""
        return cast(torch.Tensor, torch.exp(-0.5 * vars) * torch.sin(means))

    def expected_cos(self, means: torch.Tensor, vars: torch.Tensor) -> torch.Tensor:
        """Expected value of cos(x) where x ~ N(means, vars)"""
        return cast(torch.Tensor, torch.exp(-0.5 * vars) * torch.cos(means))

    def integrated_pos_enc(self, means: torch.Tensor, vars: torch.Tensor) -> torch.Tensor:
        """
        Compute integrated positional encoding for multivariate Gaussians

        Args:
            means: [..., 3] tensor of means
            vars: [..., 3] tensor of variances

        Returns:
            [..., 2*3*num_levels] tensor of encoded features
        """
        # Create frequency scales
        scales = torch.pow(
            2.0,
            torch.arange,
        )

        # Scale means and variances
        scaled_means = means[..., None, :] * scales[None, :, None]  # [..., num_levels, 3]
        scaled_vars = vars[..., None, :] * scales[None, :, None] ** 2  # [..., num_levels, 3]

        # Compute expected sin and cos
        expected_sin_vals = self.expected_sin(scaled_means, scaled_vars)
        expected_cos_vals = self.expected_cos(scaled_means, scaled_vars)

        # Concatenate and flatten
        encoding = torch.cat([expected_sin_vals, expected_cos_vals], dim=-1)  # [..., num_levels, 6]
        return cast(
            torch.Tensor,
            encoding.reshape,
        )

    def forward(self, means: torch.Tensor, vars: torch.Tensor) -> torch.Tensor:
        """Forward pass of IPE"""
        return self.integrated_pos_enc(means, vars)


class ConicalFrustum:
    """
    Represents a conical frustum for Mip-NeRF

    A conical frustum is the 3D shape traced out by a square pixel as it is projected
    through 3D space. Each frustum is characterized by its mean and covariance.
    """

    def __init__(self, means: torch.Tensor, covs: torch.Tensor):
        """
        Args:
            means: [..., 3] tensor of frustum centers
            covs: [..., 3, 3] tensor of frustum covariances
        """
        self.means = means
        self.covs = covs

    @classmethod
    def from_rays(
        cls,
        origins: torch.Tensor,
        directions: torch.Tensor,
        t_vals: torch.Tensor,
        pixel_radius: float = 1.0,
    ) -> ConicalFrustum:
        """
        Create conical frustums from rays and t values

        Args:
            origins: [..., 3] ray origins
            directions: [..., 3] ray directions
            t_vals: [..., num_samples] t values along rays
            pixel_radius: Radius of pixel in camera coordinates
        """
        # Compute frustum centers
        means = origins[..., None, :] + directions[..., None, :] * t_vals[..., :, None]

        # Compute frustum covariances
        # This is a simplified version - full implementation would consider pixel footprint
        dt = torch.diff(t_vals, dim=-1)
        dt = torch.cat([dt, dt[..., -1:]], dim=-1)

        # Radial variance grows with distance
        t_centered = t_vals
        radial_var = pixel_radius**2 * t_centered**2
        axial_var = (dt / 3.0) ** 2

        # Create diagonal covariance matrices
        batch_shape = means.shape[:-1]
        covs = torch.zeros(*batch_shape, 3, 3, device=means.device, dtype=means.dtype)

        # Radial components (perpendicular to ray direction)
        dir_norm = F.normalize(directions, dim=-1)

        # Create perpendicular vectors for each direction
        up = torch.tensor([0.0, 0.0, 1.0], device=dir_norm.device).expand_as(dir_norm)
        # If direction is parallel to up, use different reference
        parallel_mask = torch.abs(torch.sum(dir_norm * up, dim=-1)) > 0.9
        up = up.clone()  # Make it writable
        up[parallel_mask] = torch.tensor([1.0, 0.0, 0.0], device=dir_norm.device)

        u = torch.cross(dir_norm, up, dim=-1)
        u = F.normalize(u, dim=-1)
        v = torch.cross(dir_norm, u, dim=-1)
        v = F.normalize(v, dim=-1)

        # Expand dimensions to match covariance tensor
        u_expanded = u[..., None, :].expand(*batch_shape, 3)  # [..., num_samples, 3]
        v_expanded = v[..., None, :].expand(*batch_shape, 3)  # [..., num_samples, 3]
        dir_expanded = dir_norm[..., None, :].expand(*batch_shape, 3)  # [..., num_samples, 3]

        for i in range(3):
            for j in range(3):
                covs[..., i, j] = (
                    radial_var
                    * (
                        u_expanded[..., i] * u_expanded[..., j]
                        + v_expanded[..., i] * v_expanded[..., j]
                    )
                    + axial_var * dir_expanded[..., i] * dir_expanded[..., j]
                )

        return cls(means, covs)

    def to_gaussian(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Convert to Gaussian representation (means and variances)"""
        # Extract diagonal variances from covariance matrices
        vars = torch.diagonal(self.covs, dim1=-2, dim2=-1)
        return self.means, vars


class MipNeRFMLP(nn.Module):
    """
    Multi-Layer Perceptron for Mip-NeRF with integrated positional encoding
    """

    def __init__(self, config: MipNeRFConfig):
        super().__init__()
        self.config = config

        # Positional encoders
        self.pos_encoder = IntegratedPositionalEncoder(config.min_deg_point, config.max_deg_point)
        self.view_encoder = IntegratedPositionalEncoder(0, config.deg_view)

        # Calculate input dimensions
        pos_enc_dim = 2 * 3 * (config.max_deg_point - config.min_deg_point)
        view_enc_dim = 2 * 3 * config.deg_view if config.use_viewdirs else 0

        # Main network layers
        self.pts_linears = nn.ModuleList(
            [nn.Linear(pos_enc_dim, config.netwidth)]
            + [
                (
                    nn.Linear(config.netwidth, config.netwidth)
                    if i not in [4]
                    else nn.Linear(config.netwidth + pos_enc_dim, config.netwidth)
                )
                for i in range(config.netdepth - 1)
            ]
        )

        # Output layers
        self.density_linear = nn.Linear(config.netwidth, 1)
        self.feature_linear = nn.Linear(config.netwidth, config.netwidth)

        if config.use_viewdirs:
            self.views_linears = nn.ModuleList(
                [nn.Linear(config.netwidth + view_enc_dim, config.netwidth // 2)]
            )
            self.rgb_linear = nn.Linear(config.netwidth // 2, 3)
        else:
            self.rgb_linear = nn.Linear(config.netwidth, 3)

    def forward(
        self,
        frustums: ConicalFrustum,
        viewdirs: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass of MLP

        Args:
            frustums: Conical frustums for sampling
            viewdirs: Optional viewing directions for view-dependent effects

        Returns:
            Dictionary containing network outputs
        """
        # Convert frustums to Gaussian representation
        means, vars = frustums.to_gaussian()

        # Encode position and direction
        pos_enc = self.pos_encoder(means, vars)
        if viewdirs is not None and self.config.use_viewdirs:
            # Normalize viewing directions
            viewdirs = F.normalize(viewdirs, dim=-1)
            # Encode viewing direction (no uncertainty)
            view_enc = self.view_encoder(viewdirs, torch.zeros_like(viewdirs))
            # Concatenate position and direction encodings
            x = torch.cat([pos_enc, view_enc], dim=-1)
        else:
            x = pos_enc

        # Forward through MLP
        h = x
        for i, linear in enumerate(self.pts_linears):
            h = F.relu(linear(h))
            if i == 4:
                h = torch.cat([h, x], dim=-1)

        # Split output into density and color
        density = self.density_linear(h)
        features = self.feature_linear(h)

        if self.config.use_viewdirs and viewdirs is not None:
            # Combine features with view encoding
            h = torch.cat([features, view_enc], dim=-1)
            for linear in self.views_linears:
                h = F.relu(linear(h))
            rgb = torch.sigmoid(self.rgb_linear(h))
        else:
            rgb = torch.sigmoid(self.rgb_linear(features))

        return {"density": density, "rgb": rgb}


class MipNeRFRenderer(nn.Module):
    """
    Volumetric renderer for Mip-NeRF with anti-aliasing
    """

    def __init__(self, config: MipNeRFConfig):
        super().__init__()
        self.config = config

    def sample_along_rays(
        self,
        origins: torch.Tensor,
        directions: torch.Tensor,
        near: float,
        far: float,
        num_samples: int,
        perturb: bool = True,
    ) -> torch.Tensor:
        """Sample points along rays"""
        # Linear sampling in disparity space
        t_vals = torch.linspace(0.0, 1.0, num_samples, device=origins.device)
        if not perturb:
            t_vals = 1.0 / (1.0 / near * (1.0 - t_vals) + 1.0 / far * t_vals)
        else:
            # Stratified sampling
            mids = 0.5 * (t_vals[:-1] + t_vals[1:])
            upper = torch.cat([mids, t_vals[-1:]])
            lower = torch.cat([t_vals[:1], mids])
            t_rand = torch.rand_like(t_vals)
            t_vals = lower + (upper - lower) * t_rand
            t_vals = 1.0 / (1.0 / near * (1.0 - t_vals) + 1.0 / far * t_vals)

        return t_vals.expand(*origins.shape[:-1], num_samples)

    def hierarchical_sample(
        self,
        origins: torch.Tensor,
        directions: torch.Tensor,
        t_vals: torch.Tensor,
        weights: torch.Tensor,
        num_importance: int,
        perturb: bool = True,
    ) -> torch.Tensor:
        """Hierarchical sampling based on coarse network weights"""
        # Convert weights to PDF
        weights = weights[..., 1:-1]  # Remove first and last (no contribution)
        weights = weights + 1e-5  # Prevent zero weights
        pdf = weights / torch.sum(weights, dim=-1, keepdim=True)
        cdf = torch.cumsum(pdf, dim=-1)
        cdf = torch.cat([torch.zeros_like(cdf[..., :1]), cdf], dim=-1)

        # Sample from CDF
        if not perturb:
            u = torch.linspace(0.0, 1.0, num_importance, device=origins.device)
            u = u.expand(*cdf.shape[:-1], num_importance)
        else:
            u = torch.rand(*cdf.shape[:-1], num_importance, device=origins.device)

        # Invert CDF
        inds = torch.searchsorted(cdf, u, right=True)
        below = torch.clamp(inds - 1, 0, cdf.shape[-1] - 1)
        above = torch.clamp(inds, 0, cdf.shape[-1] - 1)

        inds_g = torch.stack([below, above], dim=-1)
        cdf_g = torch.gather(cdf[..., None], -2, inds_g)
        bins_g = torch.gather(t_vals[..., None], -2, inds_g)

        denom = cdf_g[..., 1] - cdf_g[..., 0]
        denom = torch.where(denom < 1e-5, torch.ones_like(denom), denom)
        t = (u - cdf_g[..., 0]) / denom
        samples = bins_g[..., 0] + t * (bins_g[..., 1] - bins_g[..., 0])

        return samples

    def volumetric_rendering(
        self,
        densities: torch.Tensor,
        colors: torch.Tensor,
        t_vals: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """Volumetric rendering equation"""
        # Compute distances between samples
        dists = torch.diff(t_vals, dim=-1)
        dists = torch.cat([dists, torch.full_like(dists[..., :1], 1e10)], dim=-1)

        # Compute alpha values
        alpha = 1.0 - torch.exp(-F.relu(densities[..., 0]) * dists)

        # Compute transmittance
        transmittance = torch.cumprod(1.0 - alpha + 1e-10, dim=-1)
        transmittance = torch.cat([torch.ones_like(transmittance[..., :1]), transmittance], dim=-1)

        # Compute weights
        weights = alpha * transmittance

        # Render color
        rgb = torch.sum(weights[..., None] * colors, dim=-2)

        # Compute depth
        depth = torch.sum(weights * t_vals, dim=-1)

        # Compute accumulated alpha (opacity)
        acc_alpha = torch.sum(weights, dim=-1)

        return {"rgb": rgb, "depth": depth, "acc_alpha": acc_alpha, "weights": weights}


class MipNeRF(nn.Module):
    """MIP-NeRF model with modern optimizations."""

    def __init__(self, config: MipNeRFConfig):
        super().__init__()
        self.config = config

        # Initialize coarse network
        self.coarse_network = MipNeRFMLP(config)

        # Initialize fine network
        self.fine_network = MipNeRFMLP(config)

        # Initialize AMP grad scaler
        self.grad_scaler = config.grad_scaler if config.use_amp else None

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

    def forward(
        self,
        rays_o: Tensor,
        rays_d: Tensor,
        view_dirs: Tensor | None = None,
        near: Tensor | None = None,
        far: Tensor | None = None,
    ) -> TensorDict:
        """Forward pass with automatic mixed precision support."""
        # Move inputs to device efficiently
        rays_o = rays_o.to(self.config.device, non_blocking=self.config.use_non_blocking)
        rays_d = rays_d.to(self.config.device, non_blocking=self.config.use_non_blocking)
        if view_dirs is not None:
            view_dirs = view_dirs.to(self.config.device, non_blocking=self.config.use_non_blocking)
        if near is not None:
            near = near.to(self.config.device, non_blocking=self.config.use_non_blocking)
        if far is not None:
            far = far.to(self.config.device, non_blocking=self.config.use_non_blocking)

        # Set default near and far planes
        if near is None:
            near = torch.full_like(rays_o[..., 0], self.config.near_plane)
        if far is None:
            far = torch.full_like(rays_o[..., 0], self.config.far_plane)

        # Process rays in chunks for memory efficiency
        outputs_list = []
        for i in range(0, rays_o.shape[0], self.config.chunk_size):
            chunk_rays_o = rays_o[i : i + self.config.chunk_size]
            chunk_rays_d = rays_d[i : i + self.config.chunk_size]
            chunk_near = near[i : i + self.config.chunk_size]
            chunk_far = far[i : i + self.config.chunk_size]
            chunk_view_dirs = (
                view_dirs[i : i + self.config.chunk_size] if view_dirs is not None else None
            )

            # Use AMP for forward pass
            with autocast(enabled=self.config.use_amp):
                # Sample points along rays
                if self.config.disparity_space_sampling:
                    t_vals = torch.linspace(0.0, 1.0, self.config.num_coarse_samples)
                    t_vals = t_vals.expand(chunk_rays_o.shape[0], -1)
                    z_vals = 1.0 / (
                        1.0 / chunk_near.unsqueeze(-1) * (1.0 - t_vals)
                        + 1.0 / chunk_far.unsqueeze(-1) * t_vals
                    )
                else:
                    t_vals = torch.linspace(0.0, 1.0, self.config.num_coarse_samples)
                    z_vals = (
                        chunk_near.unsqueeze(-1) * (1.0 - t_vals) + chunk_far.unsqueeze(-1) * t_vals
                    )

                # Add noise to sampling
                if self.training:
                    mids = 0.5 * (z_vals[..., 1:] + z_vals[..., :-1])
                    upper = torch.cat([mids, z_vals[..., -1:]], -1)
                    lower = torch.cat([z_vals[..., :1], mids], -1)
                    t_rand = torch.rand_like(z_vals)
                    z_vals = lower + (upper - lower) * t_rand

                # Get sample points and intervals
                pts = chunk_rays_o.unsqueeze(-2) + chunk_rays_d.unsqueeze(-2) * z_vals.unsqueeze(-1)
                intervals = z_vals[..., 1:] - z_vals[..., :-1]

                # Run coarse network
                coarse_output = self.coarse_network(pts, intervals, chunk_view_dirs)
                coarse_rgb = coarse_output["rgb"]
                coarse_density = coarse_output["density"]

                # Compute weights for importance sampling
                weights = self._compute_weights(coarse_density, z_vals, chunk_rays_d)

                # Importance sampling
                z_vals_mid = 0.5 * (z_vals[..., 1:] + z_vals[..., :-1])
                z_samples = self._sample_pdf(
                    z_vals_mid, weights[..., 1:-1], self.config.num_fine_samples
                )
                z_vals_combined, _ = torch.sort(torch.cat([z_vals, z_samples], -1), -1)

                # Get fine sample points and intervals
                pts = chunk_rays_o.unsqueeze(-2) + chunk_rays_d.unsqueeze(
                    -2
                ) * z_vals_combined.unsqueeze(-1)
                intervals = z_vals_combined[..., 1:] - z_vals_combined[..., :-1]

                # Run fine network
                fine_output = self.fine_network(pts, intervals, chunk_view_dirs)
                fine_rgb = fine_output["rgb"]
                fine_density = fine_output["density"]

                chunk_outputs = {
                    "coarse": {
                        "rgb": coarse_rgb,
                        "density": coarse_density,
                        "weights": weights,
                        "z_vals": z_vals,
                    },
                    "fine": {"rgb": fine_rgb, "density": fine_density, "z_vals": z_vals_combined},
                }

            outputs_list.append(chunk_outputs)

        # Combine chunk outputs
        outputs = {
            level: {
                k: torch.cat([out[level][k] for out in outputs_list], dim=0)
                for k in outputs_list[0][level].keys()
            }
            for level in ["coarse", "fine"]
        }

        return outputs

    def _compute_weights(self, density: Tensor, z_vals: Tensor, rays_d: Tensor) -> Tensor:
        """Compute weights for volume rendering."""
        dists = z_vals[..., 1:] - z_vals[..., :-1]
        dists = torch.cat([dists, torch.tensor([1e10]).expand(dists[..., :1].shape)], -1)
        dists = dists * torch.norm(rays_d[..., None, :], dim=-1)

        alpha = 1.0 - torch.exp(-F.relu(density) * dists)
        weights = (
            alpha
            * torch.cumprod(
                torch.cat([torch.ones_like(alpha[..., :1]), 1.0 - alpha + 1e-10], -1), -1
            )[..., :-1]
        )

        return weights

    def _sample_pdf(self, bins: Tensor, weights: Tensor, num_samples: int) -> Tensor:
        """Sample from probability density function for importance sampling."""
        weights = weights + 1e-5
        pdf = weights / torch.sum(weights, -1, keepdim=True)
        cdf = torch.cumsum(pdf, -1)
        cdf = torch.cat([torch.zeros_like(cdf[..., :1]), cdf], -1)

        u = torch.rand(list(cdf.shape[:-1]) + [num_samples], device=cdf.device)
        inds = torch.searchsorted(cdf, u, right=True)
        below = torch.clamp(inds - 1, 0)
        above = torch.clamp(inds, max=cdf.shape[-1] - 1)

        inds_g = torch.stack([below, above], -1)
        cdf_g = torch.gather(
            cdf[..., None, :].expand(*inds_g.shape[:-1], cdf.shape[-1]), -1, inds_g
        )
        bins_g = torch.gather(
            bins[..., None, :].expand(*inds_g.shape[:-1], bins.shape[-1]), -1, inds_g
        )

        denom = cdf_g[..., 1] - cdf_g[..., 0]
        denom = torch.where(denom < 1e-5, torch.ones_like(denom), denom)
        t = (u - cdf_g[..., 0]) / denom
        samples = bins_g[..., 0] + t * (bins_g[..., 1] - bins_g[..., 0])

        return samples

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
            outputs = self(
                batch["rays_o"],
                batch["rays_d"],
                batch.get("view_dirs"),
                batch.get("near"),
                batch.get("far"),
            )
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

        outputs = self(
            batch["rays_o"],
            batch["rays_d"],
            batch.get("view_dirs"),
            batch.get("near"),
            batch.get("far"),
        )
        loss = self.compute_loss(outputs, batch)

        return loss

    def compute_loss(self, predictions: TensorDict, targets: TensorDict) -> TensorDict:
        """Compute training losses."""
        losses = {}

        # Coarse RGB loss
        if "rgb" in targets:
            losses["coarse_rgb"] = F.mse_loss(predictions["coarse"]["rgb"], targets["rgb"])

        # Fine RGB loss
        if "rgb" in targets:
            losses["fine_rgb"] = F.mse_loss(predictions["fine"]["rgb"], targets["rgb"])

        # Total loss
        losses["total_loss"] = losses["coarse_rgb"] + losses["fine_rgb"]

        return losses

    def update_learning_rate(self, optimizer: torch.optim.Optimizer, step: int):
        """Update learning rate with decay."""
        decay_steps = step // self.config.learning_rate_decay_steps
        decay_factor = self.config.learning_rate_decay_mult**decay_steps

        # Update optimizer learning rate
        for param_group in optimizer.param_groups:
            param_group["lr"] = self.config.learning_rate * decay_factor
