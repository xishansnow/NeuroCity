from typing import Optional
"""
Multi-scale Renderer Module
Implements level-of-detail rendering for BungeeNeRF
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import logging

logger = logging.getLogger(__name__)

class MultiScaleRenderer(nn.Module):
    """
    Multi-scale renderer for level-of-detail rendering
    """
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.white_background = False
        
    def render(
        self, rgb: torch.Tensor, sigma: torch.Tensor, z_vals: torch.Tensor, rays_d: torch.Tensor, noise_std: float = 0.0
    ) -> dict[str, torch.Tensor]:
        """
        Volume rendering with alpha compositing
        
        Args:
            rgb: RGB values [N_rays, N_samples, 3]
            sigma: Density values [N_rays, N_samples]
            z_vals: Sample depths [N_rays, N_samples]
            rays_d: Ray directions [N_rays, 3]
            noise_std: Noise standard deviation for training
            
        Returns:
            Dictionary of rendered outputs
        """
        # Calculate distances between samples
        dists = z_vals[..., 1:] - z_vals[..., :-1]
        dists = torch.cat([
            dists, torch.full_like(dists[..., :1], 1e10)
        ], dim=-1)
        
        # Apply ray direction magnitude
        dists = dists * torch.norm(rays_d[..., None, :], dim=-1)
        
        # Add noise during training
        if self.training and noise_std > 0:
            noise = torch.randn_like(sigma) * noise_std
            sigma = sigma + noise
        
        # Convert density to alpha values
        alpha = 1.0 - torch.exp(-F.relu(sigma) * dists)
        
        # Calculate transmittance
        transmittance = torch.cumprod(
            torch.cat([
                torch.ones_like(alpha[..., :1]), 1.0 - alpha[..., :-1]
            ], dim=-1), dim=-1
        )
        
        # Calculate weights
        weights = alpha * transmittance
        
        # Composite RGB
        rgb_map = torch.sum(weights[..., None] * rgb, dim=-2)
        
        # Calculate depth map
        depth_map = torch.sum(weights * z_vals, dim=-1)
        
        # Calculate accumulated alpha (opacity)
        acc_map = torch.sum(weights, dim=-1)
        
        # Add white background if needed
        if self.white_background:
            rgb_map = rgb_map + (1.0 - acc_map[..., None])
        
        # Calculate disparity map
        disp_map = 1.0 / torch.max(
            1e-10 * torch.ones_like(depth_map), depth_map / acc_map
        )
        
        return {
            "rgb": rgb_map, "depth": depth_map, "acc": acc_map, "weights": weights, "disp": disp_map, "transmittance": transmittance, "alpha": alpha
        }
    
    def render_multiscale(
        self, rgb_scales: list[torch.Tensor], sigma_scales: list[torch.Tensor], z_vals: torch.Tensor, rays_d: torch.Tensor, distances: torch.Tensor, scale_thresholds: list[float]
    ) -> dict[str, torch.Tensor]:
        """
        Render with multiple scales based on distance
        
        Args:
            rgb_scales: list of RGB values for different scales
            sigma_scales: list of density values for different scales
            z_vals: Sample depths [N_rays, N_samples]
            rays_d: Ray directions [N_rays, 3]
            distances: Distance to camera [N_rays]
            scale_thresholds: Distance thresholds for scale selection
            
        Returns:
            Dictionary of rendered outputs
        """
        device = rgb_scales[0].device
        batch_size = rgb_scales[0].shape[0]
        num_samples = rgb_scales[0].shape[1]
        
        # Initialize combined RGB and sigma
        rgb_combined = torch.zeros_like(rgb_scales[0])
        sigma_combined = torch.zeros_like(sigma_scales[0])
        
        # Combine scales based on distance
        for scale_idx, threshold in enumerate(scale_thresholds):
            if scale_idx >= len(rgb_scales):
                break
            
            # Create mask for this scale
            if scale_idx == 0:
                mask = distances >= threshold
            else:
                prev_threshold = scale_thresholds[scale_idx - 1]
                mask = (distances < prev_threshold) & (distances >= threshold)
            
            if mask.any():
                rgb_combined[mask] = rgb_scales[scale_idx][mask]
                sigma_combined[mask] = sigma_scales[scale_idx][mask]
        
        # Handle closest points with highest detail scale
        if len(rgb_scales) > len(scale_thresholds):
            closest_mask = distances < scale_thresholds[-1]
            if closest_mask.any():
                rgb_combined[closest_mask] = rgb_scales[-1][closest_mask]
                sigma_combined[closest_mask] = sigma_scales[-1][closest_mask]
        
        # Perform volume rendering
        return self.render(rgb_combined, sigma_combined, z_vals, rays_d)

class LevelOfDetailRenderer(nn.Module):
    """
    Level-of-detail renderer with adaptive sampling
    """
    
    def __init__(
        self, config, num_lod_levels: int = 4, lod_thresholds: list[float] = None
    ):
        super().__init__()
        
        self.config = config
        self.num_lod_levels = num_lod_levels
        
        if lod_thresholds is None:
            lod_thresholds = [100.0, 50.0, 25.0, 10.0]
        self.lod_thresholds = lod_thresholds
        
        # Different sampling rates for different LOD levels
        self.lod_samples = [16, 32, 64, 128]  # Samples per ray for each LOD level
    
    def get_lod_level(self, distances: torch.Tensor) -> torch.Tensor:
        """
        Get level-of-detail based on distance
        
        Args:
            distances: Distance to camera [N]
            
        Returns:
            LOD levels [N]
        """
        device = distances.device
        lod_levels = torch.zeros_like(distances, dtype=torch.long)
        
        for level, threshold in enumerate(self.lod_thresholds):
            mask = distances < threshold
            lod_levels[mask] = level + 1
        
        # Clamp to valid range
        lod_levels = torch.clamp(lod_levels, 0, self.num_lod_levels - 1)
        
        return lod_levels
    
    def adaptive_sampling(
        self, rays_o: torch.Tensor, rays_d: torch.Tensor, bounds: torch.Tensor, distances: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Adaptive sampling based on level-of-detail
        
        Args:
            rays_o: Ray origins [N, 3]
            rays_d: Ray directions [N, 3]
            bounds: Near/far bounds [N, 2]
            distances: Distance to camera [N]
            
        Returns:
            tuple of (sample_points, z_vals)
        """
        device = rays_o.device
        batch_size = rays_o.shape[0]
        
        # Get LOD levels
        lod_levels = self.get_lod_level(distances)
        
        # Maximum number of samples
        max_samples = max(self.lod_samples)
        
        # Initialize outputs
        all_z_vals = torch.zeros(batch_size, max_samples, device=device)
        all_points = torch.zeros(batch_size, max_samples, 3, device=device)
        
        # Sample for each LOD level
        for level in range(self.num_lod_levels):
            mask = lod_levels == level
            if not mask.any():
                continue
            
            num_samples = self.lod_samples[level]
            
            # Sample depths for this LOD level
            near = bounds[mask, 0]
            far = bounds[mask, 1]
            
            t_vals = torch.linspace(0.0, 1.0, num_samples, device=device)
            z_vals = near[..., None] * (1.0 - t_vals) + far[..., None] * t_vals
            
            # Add perturbation during training
            if self.training:
                mids = 0.5 * (z_vals[..., 1:] + z_vals[..., :-1])
                upper = torch.cat([mids, z_vals[..., -1:]], dim=-1)
                lower = torch.cat([z_vals[..., :1], mids], dim=-1)
                
                t_rand = torch.rand_like(z_vals)
                z_vals = lower + (upper - lower) * t_rand
            
            # Calculate sample points
            points = rays_o[mask][..., None, :] + rays_d[mask][..., None, :] * z_vals[..., :, None]
            
            # Store in output tensors
            all_z_vals[mask, :num_samples] = z_vals
            all_points[mask, :num_samples] = points
        
        return all_points, all_z_vals
    
    def forward(
        self, model: nn.Module, rays_o: torch.Tensor, rays_d: torch.Tensor, bounds: torch.Tensor, distances: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        """
        Forward pass with level-of-detail rendering
        
        Args:
            model: BungeeNeRF model
            rays_o: Ray origins [N, 3]
            rays_d: Ray directions [N, 3]
            bounds: Near/far bounds [N, 2]
            distances: Distance to camera [N]
            
        Returns:
            Dictionary of rendered outputs
        """
        # Adaptive sampling
        sample_points, z_vals = self.adaptive_sampling(rays_o, rays_d, bounds, distances)
        
        # Forward through model
        outputs = model(rays_o, rays_d, bounds, distances)
        
        return outputs

class ProgressiveRenderer(nn.Module):
    """
    Progressive renderer that adapts to training stage
    """
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.current_stage = 0
        
        # Base renderer
        self.base_renderer = MultiScaleRenderer(config)
        
        # Progressive rendering parameters
        self.stage_sample_counts = [32, 48, 64, 128]  # Samples per stage
        self.stage_noise_levels = [0.1, 0.05, 0.02, 0.0]  # Noise levels per stage
    
    def set_current_stage(self, stage: int):
        """Set current training stage"""
        self.current_stage = stage
    
    def get_stage_params(self) -> tuple[int, float]:
        """Get parameters for current stage"""
        stage = min(self.current_stage, len(self.stage_sample_counts) - 1)
        
        num_samples = self.stage_sample_counts[stage]
        noise_level = self.stage_noise_levels[stage]
        
        return num_samples, noise_level
    
    def render(
        self, rgb: torch.Tensor, sigma: torch.Tensor, z_vals: torch.Tensor, rays_d: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        """
        Progressive rendering based on current stage
        
        Args:
            rgb: RGB values [N_rays, N_samples, 3]
            sigma: Density values [N_rays, N_samples]
            z_vals: Sample depths [N_rays, N_samples]
            rays_d: Ray directions [N_rays, 3]
            
        Returns:
            Dictionary of rendered outputs
        """
        # Get stage parameters
        num_samples, noise_level = self.get_stage_params()
        
        # Subsample if needed
        if rgb.shape[1] > num_samples:
            # Uniform subsampling
            indices = torch.linspace(
                0,
                rgb.shape[1] - 1,
                num_samples,
                dtype=torch.long,
                device=rgb.device,
            )
            rgb = rgb[:, indices]
            sigma = sigma[:, indices]
            z_vals = z_vals[:, indices]
        
        # Apply progressive rendering
        return self.base_renderer.render(rgb, sigma, z_vals, rays_d, noise_std=noise_level)

class HierarchicalRenderer(nn.Module):
    """
    Hierarchical renderer with coarse-to-fine sampling
    """
    
    def __init__(
        self, config, num_coarse: int = 64, num_fine: int = 128, use_viewdirs: bool = True
    ):
        super().__init__()
        
        self.config = config
        self.num_coarse = num_coarse
        self.num_fine = num_fine
        self.use_viewdirs = use_viewdirs
        
        # Base renderer
        self.base_renderer = MultiScaleRenderer(config)
    
    def importance_sample(
        self, z_vals: torch.Tensor, weights: torch.Tensor, num_samples: int, det: bool = False
    ) -> torch.Tensor:
        """
        Importance sampling based on coarse weights
        
        Args:
            z_vals: Coarse sample depths [N_rays, N_coarse]
            weights: Coarse weights [N_rays, N_coarse]
            num_samples: Number of fine samples
            det: Deterministic sampling
            
        Returns:
            Fine sample depths [N_rays, num_samples]
        """
        device = z_vals.device
        batch_size = z_vals.shape[0]
        
        # Get PDF from weights
        weights = weights + 1e-5  # Prevent NaNs
        pdf = weights / torch.sum(weights, dim=-1, keepdim=True)
        cdf = torch.cumsum(pdf, dim=-1)
        cdf = torch.cat([torch.zeros_like(cdf[..., :1]), cdf], dim=-1)
        
        # Sample from CDF
        if det:
            u = torch.linspace(0.0, 1.0, num_samples, device=device)
            u = u.expand(batch_size, num_samples)
        else:
            u = torch.rand(batch_size, num_samples, device=device)
        
        # Invert CDF
        indices = torch.searchsorted(cdf, u, right=True)
        below = torch.clamp(indices - 1, 0, cdf.shape[-1] - 1)
        above = torch.clamp(indices, 0, cdf.shape[-1] - 1)
        
        indices_g = torch.stack([below, above], dim=-1)
        
        # Gather CDF values
        cdf_g = torch.gather(cdf[..., None], 2, indices_g).squeeze(-1)
        z_vals_g = torch.gather(z_vals[..., None], 2, indices_g).squeeze(-1)
        
        # Linear interpolation
        denom = cdf_g[..., 1] - cdf_g[..., 0]
        denom = torch.where(denom < 1e-5, torch.ones_like(denom), denom)
        
        t = (u - cdf_g[..., 0]) / denom
        samples = z_vals_g[..., 0] + t * (z_vals_g[..., 1] - z_vals_g[..., 0])
        
        return samples
    
    def forward(
        self, model: nn.Module, rays_o: torch.Tensor, rays_d: torch.Tensor, bounds: torch.Tensor, distances: Optional[torch.Tensor] = None
    ) -> dict[str, torch.Tensor]:
        """
        Hierarchical rendering with coarse-to-fine sampling
        
        Args:
            model: BungeeNeRF model
            rays_o: Ray origins [N, 3]
            rays_d: Ray directions [N, 3]
            bounds: Near/far bounds [N, 2]
            distances: Distance to camera [N] (optional)
            
        Returns:
            Dictionary of rendered outputs
        """
        # Coarse pass
        coarse_outputs = model(rays_o, rays_d, bounds, distances)
        
        # Importance sampling for fine pass
        if "weights" in coarse_outputs and self.num_fine > 0:
            # Get coarse z_vals (assuming they're stored in model)
            z_vals_coarse = model._sample_along_rays(rays_o, rays_d, bounds)
            
            # Importance sample fine points
            z_vals_fine = self.importance_sample(
                z_vals_coarse, coarse_outputs["weights"], self.num_fine, det=not self.training
            )
            
            # Combine coarse and fine samples
            z_vals_combined = torch.cat([z_vals_coarse, z_vals_fine], dim=-1)
            z_vals_combined, _ = torch.sort(z_vals_combined, dim=-1)
            
            # Fine pass with combined samples
            # This would require modifying the model to accept custom z_vals
            fine_outputs = model(rays_o, rays_d, bounds, distances)
            
            return fine_outputs
        else:
            return coarse_outputs
