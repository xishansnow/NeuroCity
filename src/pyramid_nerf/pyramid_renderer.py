"""
Pyramid Renderer Module
Implements volume rendering for PyNeRF with multi-scale sampling
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple
import numpy as np
import logging

logger = logging.getLogger(__name__)


class PyramidRenderer(nn.Module):
    """
    Pyramid renderer for multi-scale volume rendering
    """
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.white_background = False
        
    def render(
        self,
        rgb: torch.Tensor,
        sigma: torch.Tensor,
        z_vals: torch.Tensor,
        rays_d: torch.Tensor,
        noise_std: float = 0.0
    ) -> Dict[str, torch.Tensor]:
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
            dists,
            torch.full_like(dists[..., :1], 1e10)
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
                torch.ones_like(alpha[..., :1]),
                1.0 - alpha[..., :-1]
            ], dim=-1),
            dim=-1
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
            1e-10 * torch.ones_like(depth_map),
            depth_map / acc_map
        )
        
        return {
            "rgb": rgb_map,
            "depth": depth_map,
            "acc": acc_map,
            "weights": weights,
            "disp": disp_map,
            "transmittance": transmittance,
            "alpha": alpha
        }
    
    def render_path(
        self,
        model: nn.Module,
        render_poses: torch.Tensor,
        hwf: Tuple[int, int, float],
        chunk: int = 1024,
        **kwargs
    ) -> List[Dict[str, torch.Tensor]]:
        """
        Render a path of poses
        
        Args:
            model: PyNeRF model
            render_poses: Camera poses [N, 4, 4]
            hwf: Height, width, focal length
            chunk: Chunk size for rendering
            
        Returns:
            List of rendered images
        """
        H, W, focal = hwf
        results = []
        
        model.eval()
        with torch.no_grad():
            for pose in render_poses:
                rays_o, rays_d = self.get_rays(H, W, focal, pose)
                
                # Flatten rays
                rays_o = rays_o.reshape(-1, 3)
                rays_d = rays_d.reshape(-1, 3)
                
                # Render in chunks
                all_outputs = []
                for i in range(0, rays_o.shape[0], chunk):
                    chunk_rays_o = rays_o[i:i+chunk]
                    chunk_rays_d = rays_d[i:i+chunk]
                    
                    # Set bounds (assuming bounded scene)
                    bounds = torch.tensor([2.0, 6.0], device=rays_o.device)
                    bounds = bounds.expand(chunk_rays_o.shape[0], 2)
                    
                    outputs = model(chunk_rays_o, chunk_rays_d, bounds)
                    all_outputs.append(outputs)
                
                # Concatenate chunk outputs
                final_outputs = {}
                for key in all_outputs[0].keys():
                    final_outputs[key] = torch.cat([
                        chunk_out[key] for chunk_out in all_outputs
                    ], dim=0)
                
                # Reshape to image
                for key in final_outputs:
                    if final_outputs[key].ndim == 1:
                        final_outputs[key] = final_outputs[key].reshape(H, W)
                    elif final_outputs[key].ndim == 2:
                        final_outputs[key] = final_outputs[key].reshape(H, W, -1)
                
                results.append(final_outputs)
        
        return results
    
    @staticmethod
    def get_rays(
        H: int,
        W: int,
        focal: float,
        c2w: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Generate rays for a camera
        
        Args:
            H: Image height
            W: Image width
            focal: Focal length
            c2w: Camera-to-world transformation [4, 4]
            
        Returns:
            Tuple of (ray_origins, ray_directions)
        """
        device = c2w.device
        
        # Create pixel coordinates
        i, j = torch.meshgrid(
            torch.linspace(0, W-1, W, device=device),
            torch.linspace(0, H-1, H, device=device),
            indexing='ij'
        )
        i = i.t()
        j = j.t()
        
        # Convert to normalized device coordinates
        dirs = torch.stack([
            (i - W * 0.5) / focal,
            -(j - H * 0.5) / focal,
            -torch.ones_like(i)
        ], dim=-1)
        
        # Transform ray directions to world coordinates
        rays_d = torch.sum(
            dirs[..., None, :] * c2w[:3, :3],
            dim=-1
        )
        
        # Ray origins are camera position
        rays_o = c2w[:3, -1].expand(rays_d.shape)
        
        return rays_o, rays_d


class VolumetricRenderer(nn.Module):
    """
    Advanced volumetric renderer with importance sampling
    """
    
    def __init__(
        self,
        num_samples: int = 64,
        num_importance_samples: int = 128,
        use_hierarchical_sampling: bool = True,
        perturb: bool = True,
        raw_noise_std: float = 0.0
    ):
        super().__init__()
        self.num_samples = num_samples
        self.num_importance_samples = num_importance_samples
        self.use_hierarchical_sampling = use_hierarchical_sampling
        self.perturb = perturb
        self.raw_noise_std = raw_noise_std
        
    def sample_along_rays(
        self,
        rays_o: torch.Tensor,
        rays_d: torch.Tensor,
        near: torch.Tensor,
        far: torch.Tensor,
        num_samples: int,
        perturb: bool = True
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Sample points along rays
        
        Args:
            rays_o: Ray origins [N, 3]
            rays_d: Ray directions [N, 3]
            near: Near bounds [N]
            far: Far bounds [N]
            num_samples: Number of samples per ray
            perturb: Whether to add noise to samples
            
        Returns:
            Tuple of (sample_points, z_values)
        """
        N_rays = rays_o.shape[0]
        device = rays_o.device
        
        # Create evenly spaced samples
        t_vals = torch.linspace(0.0, 1.0, num_samples, device=device)
        z_vals = near[..., None] * (1.0 - t_vals) + far[..., None] * t_vals
        
        # Add noise to samples during training
        if perturb and self.training:
            # Get intervals between samples
            mids = 0.5 * (z_vals[..., 1:] + z_vals[..., :-1])
            upper = torch.cat([mids, z_vals[..., -1:]], dim=-1)
            lower = torch.cat([z_vals[..., :1], mids], dim=-1)
            
            # Add uniform noise
            t_rand = torch.rand(z_vals.shape, device=device)
            z_vals = lower + (upper - lower) * t_rand
        
        # Calculate sample points
        pts = rays_o[..., None, :] + rays_d[..., None, :] * z_vals[..., :, None]
        
        return pts, z_vals
    
    def importance_sample(
        self,
        z_vals: torch.Tensor,
        weights: torch.Tensor,
        num_samples: int,
        det: bool = False
    ) -> torch.Tensor:
        """
        Importance sampling based on weights
        
        Args:
            z_vals: Original z values [N_rays, N_samples]
            weights: Sample weights [N_rays, N_samples]
            num_samples: Number of importance samples
            det: Deterministic sampling
            
        Returns:
            New z values [N_rays, N_samples + num_samples]
        """
        device = z_vals.device
        N_rays, N_samples = z_vals.shape
        
        # Get PDF
        weights = weights + 1e-5  # Prevent division by zero
        pdf = weights / torch.sum(weights, dim=-1, keepdim=True)
        cdf = torch.cumsum(pdf, dim=-1)
        cdf = torch.cat([torch.zeros_like(cdf[..., :1]), cdf], dim=-1)
        
        # Take uniform samples
        if det:
            u = torch.linspace(0.0, 1.0, num_samples, device=device)
            u = u.expand(N_rays, num_samples)
        else:
            u = torch.rand(N_rays, num_samples, device=device)
        
        # Invert CDF
        u = u.contiguous()
        indices = torch.searchsorted(cdf, u, right=True)
        below = torch.clamp_min(indices - 1, 0)
        above = torch.clamp_max(indices, N_samples)
        
        indices_g = torch.stack([below, above], dim=-1)
        matched_shape = list(indices_g.shape[:-1]) + [cdf.shape[-1]]
        cdf_g = torch.gather(cdf.unsqueeze(-2).expand(matched_shape), -1, indices_g)
        bins_g = torch.gather(z_vals.unsqueeze(-2).expand(matched_shape), -1, indices_g)
        
        denom = cdf_g[..., 1] - cdf_g[..., 0]
        denom = torch.where(denom < 1e-5, torch.ones_like(denom), denom)
        t = (u - cdf_g[..., 0]) / denom
        samples = bins_g[..., 0] + t * (bins_g[..., 1] - bins_g[..., 0])
        
        # Combine with original samples
        z_vals_combined, _ = torch.sort(
            torch.cat([z_vals, samples], dim=-1),
            dim=-1
        )
        
        return z_vals_combined
    
    def forward(
        self,
        model: nn.Module,
        rays_o: torch.Tensor,
        rays_d: torch.Tensor,
        bounds: torch.Tensor,
        **kwargs
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass through volumetric renderer
        
        Args:
            model: PyNeRF model
            rays_o: Ray origins [N, 3]
            rays_d: Ray directions [N, 3]
            bounds: Near/far bounds [N, 2]
            
        Returns:
            Dictionary of rendered outputs
        """
        near, far = bounds[..., 0], bounds[..., 1]
        
        # Coarse sampling
        pts_coarse, z_vals_coarse = self.sample_along_rays(
            rays_o, rays_d, near, far, self.num_samples, self.perturb
        )
        
        # Query coarse model
        outputs_coarse = model(
            rays_o, rays_d, 
            torch.stack([near, far], dim=-1),
            sample_points=pts_coarse,
            z_vals=z_vals_coarse
        )
        
        results = {"rgb_coarse": outputs_coarse["rgb"]}
        
        # Fine sampling with importance sampling
        if self.use_hierarchical_sampling and self.num_importance_samples > 0:
            z_vals_fine = self.importance_sample(
                z_vals_coarse,
                outputs_coarse["weights"][..., 1:-1],  # Exclude endpoints
                self.num_importance_samples,
                det=not self.training
            )
            
            # Query fine model
            pts_fine = rays_o[..., None, :] + rays_d[..., None, :] * z_vals_fine[..., :, None]
            outputs_fine = model(
                rays_o, rays_d,
                torch.stack([near, far], dim=-1),
                sample_points=pts_fine,
                z_vals=z_vals_fine
            )
            
            results.update({
                "rgb": outputs_fine["rgb"],
                "depth": outputs_fine["depth"],
                "acc": outputs_fine["acc"],
                "weights": outputs_fine["weights"]
            })
        else:
            results.update({
                "rgb": outputs_coarse["rgb"],
                "depth": outputs_coarse["depth"],
                "acc": outputs_coarse["acc"],
                "weights": outputs_coarse["weights"]
            })
        
        return results


class AntiAliasingRenderer(nn.Module):
    """
    Anti-aliasing renderer with cone tracing (inspired by Mip-NeRF)
    """
    
    def __init__(self, cone_angle: float = 0.00628):
        super().__init__()
        self.cone_angle = cone_angle
    
    def compute_cone_radius(
        self,
        z_vals: torch.Tensor,
        pixel_radius: float = 1.0
    ) -> torch.Tensor:
        """
        Compute cone radius at each sample point
        
        Args:
            z_vals: Sample depths [N_rays, N_samples]
            pixel_radius: Pixel radius in world coordinates
            
        Returns:
            Cone radii [N_rays, N_samples]
        """
        return pixel_radius * (z_vals * self.cone_angle + 1.0)
    
    def compute_sample_areas(
        self,
        rays_o: torch.Tensor,
        rays_d: torch.Tensor,
        z_vals: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute sample areas for pyramid level selection
        
        Args:
            rays_o: Ray origins [N_rays, 3]
            rays_d: Ray directions [N_rays, 3]
            z_vals: Sample depths [N_rays, N_samples]
            
        Returns:
            Sample areas [N_rays, N_samples]
        """
        # Compute cone radius at each sample
        cone_radii = self.compute_cone_radius(z_vals)
        
        # Sample area is proportional to cone radius squared
        sample_areas = np.pi * cone_radii ** 2
        
        return sample_areas
    
    def forward(
        self,
        model: nn.Module,
        rays_o: torch.Tensor,
        rays_d: torch.Tensor,
        bounds: torch.Tensor,
        **kwargs
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass with anti-aliasing
        
        Args:
            model: PyNeRF model
            rays_o: Ray origins [N, 3]
            rays_d: Ray directions [N, 3]
            bounds: Near/far bounds [N, 2]
            
        Returns:
            Dictionary of rendered outputs
        """
        # Standard volume rendering with cone-based sampling
        return model(rays_o, rays_d, bounds, **kwargs) 