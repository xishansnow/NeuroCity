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

TensorType = TypeVar('TensorType', bound=torch.Tensor)

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
    """Configuration for Mip-NeRF model"""
    
    # Network architecture
    netdepth: int = 8  # Number of layers in network
    netwidth: int = 256  # Number of channels per layer
    netdepth_fine: int = 8  # Number of layers in fine network
    netwidth_fine: int = 256  # Number of channels per layer in fine network
    
    # Positional encoding
    multires: int = 10  # Log2 of max freq for positional encoding (3D location)
    multires_views: int = 4  # Log2 of max freq for positional encoding (2D direction)
    
    # Sampling
    num_samples: int = 64  # Number of coarse samples per ray
    num_importance: int = 128  # Number of additional fine samples per ray
    
    # Rendering
    perturb: float = 1.0  # Set to 0 for no jitter, 1 for jitter
    use_viewdirs: bool = True  # Use full 5D input instead of 3D
    
    # Loss weights
    lambda_coarse: float = 1.0  # Weight for coarse network loss
    lambda_fine: float = 1.0  # Weight for fine network loss
    
    # Training
    lr_init: float = 5e-4  # Initial learning rate
    lr_final: float = 5e-6  # Final learning rate
    lr_decay: int = 250  # Exponential learning rate decay (in 1000s)
    grad_max_norm: float = 0.0  # Gradient clipping norm
    grad_max_val: float = 0.0  # Gradient clipping value
    
    # Mip-NeRF specific
    min_deg_point: int = 0  # Min degree of positional encoding for 3D points
    max_deg_point: int = 12  # Max degree of positional encoding for 3D points
    deg_view: int = 4  # Degree of positional encoding for viewdirs
    resample_padding: float = 0.01  # Dirichlet/alpha "padding" on the histogram
    stop_level_grad: bool = True  # If True, don't backprop gradients through levels
    use_multiscale_loss: bool = True  # If True, use multiscale loss

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
        axial_var = (dt / 3.0)**2
        
        # Create diagonal covariance matrices
        batch_shape = means.shape[:-1]
        covs = torch.zeros(*batch_shape, 3, 3, device=means.device, dtype=means.dtype)
        
        # Radial components (perpendicular to ray direction)
        dir_norm = F.normalize(directions, dim=-1)
        
        # Create perpendicular vectors for each direction
        up = torch.tensor([0., 0., 1.], device=dir_norm.device).expand_as(dir_norm)
        # If direction is parallel to up, use different reference
        parallel_mask = torch.abs(torch.sum(dir_norm * up, dim=-1)) > 0.9
        up = up.clone()  # Make it writable
        up[parallel_mask] = torch.tensor([1., 0., 0.], device=dir_norm.device)
        
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
                covs[..., i, j] = (radial_var * 
                                 (u_expanded[..., i] * u_expanded[..., j] + 
                                  v_expanded[..., i] * v_expanded[..., j]) +
                                 axial_var * 
                                 dir_expanded[..., i] * dir_expanded[..., j])
        
        return cls(means, covs)
    
    def to_gaussian(self) -> tuple[torch.Tensor, torch.Tensor]:
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
        self.pos_encoder = IntegratedPositionalEncoder(
            config.min_deg_point, config.max_deg_point
        )
        self.view_encoder = IntegratedPositionalEncoder(0, config.deg_view)
        
        # Calculate input dimensions
        pos_enc_dim = 2 * 3 * (config.max_deg_point - config.min_deg_point)
        view_enc_dim = 2 * 3 * config.deg_view if config.use_viewdirs else 0
        
        # Main network layers
        self.pts_linears = nn.ModuleList([
            nn.Linear(pos_enc_dim, config.netwidth)
        ] + [
            nn.Linear(config.netwidth, config.netwidth) if i not in [4] 
            else nn.Linear(config.netwidth + pos_enc_dim, config.netwidth)
            for i in range(config.netdepth - 1)
        ])
        
        # Output layers
        self.density_linear = nn.Linear(config.netwidth, 1)
        self.feature_linear = nn.Linear(config.netwidth, config.netwidth)
        
        if config.use_viewdirs:
            self.views_linears = nn.ModuleList([
                nn.Linear(config.netwidth + view_enc_dim, config.netwidth // 2)
            ])
            self.rgb_linear = nn.Linear(config.netwidth // 2, 3)
        else:
            self.rgb_linear = nn.Linear(config.netwidth, 3)
    
    def forward(
        self,
        frustums: ConicalFrustum,
        viewdirs: Optional[torch.Tensor] = None,
    ) -> dict[str, torch.Tensor]:
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
        
        return {
            'density': density, 'rgb': rgb
        }

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
        t_vals = torch.linspace(0., 1., num_samples, device=origins.device)
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
            u = torch.linspace(0., 1., num_importance, device=origins.device)
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
    ) -> dict[str, torch.Tensor]:
        """Volumetric rendering equation"""
        # Compute distances between samples
        dists = torch.diff(t_vals, dim=-1)
        dists = torch.cat([dists, torch.full_like(dists[..., :1], 1e10)], dim=-1)
        
        # Compute alpha values
        alpha = 1.0 - torch.exp(-F.relu(densities[..., 0]) * dists)
        
        # Compute transmittance
        transmittance = torch.cumprod(1.0 - alpha + 1e-10, dim=-1)
        transmittance = torch.cat(
            [torch.ones_like(transmittance[..., :1]), transmittance], dim=-1
        )
        
        # Compute weights
        weights = alpha * transmittance
        
        # Render color
        rgb = torch.sum(weights[..., None] * colors, dim=-2)
        
        # Compute depth
        depth = torch.sum(weights * t_vals, dim=-1)
        
        # Compute accumulated alpha (opacity)
        acc_alpha = torch.sum(weights, dim=-1)
        
        return {
            'rgb': rgb, 'depth': depth, 'acc_alpha': acc_alpha, 'weights': weights
        }

class MipNeRF(nn.Module):
    """
    Complete Mip-NeRF model with coarse and fine networks
    """
    
    def __init__(self, config: MipNeRFConfig):
        super().__init__()
        self.config = config
        
        # Networks
        self.coarse_mlp = MipNeRFMLP(config)
        if config.num_importance > 0:
            # Fine network with same architecture but potentially different parameters
            fine_config = MipNeRFConfig(**{
                **config.__dict__, 'netdepth': config.netdepth_fine, 'netwidth': config.netwidth_fine
            })
            self.fine_mlp = MipNeRFMLP(fine_config)
        
        # Renderer
        self.renderer = MipNeRFRenderer(config)
    
    def forward(
        self,
        origins: torch.Tensor,
        directions: torch.Tensor,
        viewdirs: torch.Tensor,
        near: float,
        far: float,
        pixel_radius: float = 1.0,
    ) -> PredictionOutput:
        """
        Forward pass of Mip-NeRF
        
        Args:
            origins: [..., 3] ray origins
            directions: [..., 3] ray directions
            viewdirs: [..., 3] viewing directions
            near: Near plane distance
            far: Far plane distance
            pixel_radius: Radius of pixel in camera coordinates
            
        Returns:
            Dictionary containing rendered results
        """
        empty_render: RenderOutput = {
            'rgb': torch.zeros_like(
                origins,
            )
        }
        
        results: PredictionOutput = {
            'coarse': empty_render, 'fine': None
        }
        
        # Coarse sampling
        t_vals_coarse = self.renderer.sample_along_rays(
            origins, directions, near, far, self.config.num_samples, perturb=self.config.perturb > 0
        )
        
        # Create conical frustums for coarse samples
        frustums_coarse = ConicalFrustum.from_rays(
            origins, directions, t_vals_coarse, pixel_radius
        )
        
        # Coarse network prediction
        coarse_pred = self.coarse_mlp(frustums_coarse, viewdirs)
        
        # Coarse rendering
        coarse_render_dict = self.renderer.volumetric_rendering(
            coarse_pred['density'], coarse_pred['rgb'], t_vals_coarse
        )
        coarse_render: RenderOutput = {
            'rgb': coarse_render_dict['rgb'], 'depth': coarse_render_dict['depth'], 'acc_alpha': coarse_render_dict['acc_alpha'], 'weights': coarse_render_dict['weights']
        }
        results['coarse'] = coarse_render
        
        # Fine sampling and rendering
        if self.config.num_importance > 0:
            # Hierarchical sampling
            t_vals_fine = self.renderer.hierarchical_sample(
                origins, directions, t_vals_coarse, coarse_render['weights'], self.config.num_importance, perturb=self.config.perturb > 0
            )
            
            # Combine coarse and fine samples
            t_vals_combined = torch.sort(torch.cat([t_vals_coarse, t_vals_fine], dim=-1), dim=-1)[0]
            
            # Create conical frustums for combined samples
            frustums_fine = ConicalFrustum.from_rays(
                origins, directions, t_vals_combined, pixel_radius
            )
            
            # Fine network prediction
            fine_pred = self.fine_mlp(frustums_fine, viewdirs)
            
            # Fine rendering
            fine_render_dict = self.renderer.volumetric_rendering(
                fine_pred['density'], fine_pred['rgb'], t_vals_combined
            )
            fine_render: RenderOutput = {
                'rgb': fine_render_dict['rgb'], 'depth': fine_render_dict['depth'], 'acc_alpha': fine_render_dict['acc_alpha'], 'weights': fine_render_dict['weights']
            }
            results['fine'] = fine_render
        
        return results

class MipNeRFLoss(nn.Module):
    """Loss function for Mip-NeRF"""
    
    def __init__(self, config: MipNeRFConfig):
        super().__init__()
        self.config = config
    
    def forward(self, pred: PredictionOutput, target: torch.Tensor) -> dict[str, torch.Tensor]:
        """
        Compute Mip-NeRF loss
        
        Args:
            pred: Dictionary containing 'coarse' and optionally 'fine' predictions
            target: [..., 3] target RGB values
            
        Returns:
            Dictionary containing loss components
        """
        losses: LossOutput = {
            'coarse_loss': torch.tensor(0.0)
        }
        
        # Coarse loss
        coarse_render = cast(RenderOutput, pred['coarse'])
        coarse_rgb = coarse_render['rgb']
        coarse_loss = F.mse_loss(coarse_rgb, target)
        losses['coarse_loss'] = coarse_loss
        
        # Fine loss (if available)
        if 'fine' in pred:
            fine_render = cast(RenderOutput, pred['fine'])
            fine_rgb = fine_render['rgb']
            fine_loss = F.mse_loss(fine_rgb, target)
            losses['fine_loss'] = fine_loss
            
            # Total loss is weighted combination
            total_loss = (self.config.lambda_coarse * coarse_loss + 
                         self.config.lambda_fine * fine_loss)
        else:
            total_loss = coarse_loss
        
        losses['total_loss'] = total_loss
        
        # Compute PSNR using the best available prediction
        final_render = cast(RenderOutput, pred.get('fine', pred['coarse']))
        losses['psnr'] = -10.0 * torch.log10(F.mse_loss(final_render['rgb'], target))
        
        return losses 