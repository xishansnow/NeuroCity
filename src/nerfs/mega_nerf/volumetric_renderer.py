"""
Volumetric Renderer for Mega-NeRF

This module implements volumetric rendering for Mega-NeRF models, including hierarchical sampling and efficient batch processing.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Union 
import logging
from tqdm import tqdm

logger = logging.getLogger(__name__)


class VolumetricRenderer:
    """Standard volumetric renderer for NeRF-style models"""
    
    def __init__(
        self,
        num_coarse_samples: int = 64,
        num_fine_samples: int = 128,
        near: float = 0.1,
        far: float = 1000.0,
        use_hierarchical_sampling: bool = True,
        white_background: bool = False,
        training: bool = False
    ):
        """
        Initialize volumetric renderer
        
        Args:
            num_coarse_samples: Number of coarse samples per ray
            num_fine_samples: Number of fine samples per ray
            near: Near plane distance
            far: Far plane distance
            use_hierarchical_sampling: Whether to use hierarchical sampling
            white_background: Whether to use white background
            training: Whether to use training mode
        """
        self.num_coarse_samples = num_coarse_samples
        self.num_fine_samples = num_fine_samples
        self.near = near
        self.far = far
        self.use_hierarchical_sampling = use_hierarchical_sampling
        self.white_background = white_background
        self.training = training
    def render_rays(
        self,
        model,
        ray_origins: torch.Tensor,
        ray_directions: torch.Tensor,
        appearance_ids: Optional[torch.Tensor] = None,
        return_weights: bool = False,
        return_depth: bool = False
    ) -> dict[str, torch.Tensor]:
        """
        Render a batch of rays
        
        Args:
            model: Mega-NeRF model
            ray_origins: Ray origins (N, 3)
            ray_directions: Ray directions (N, 3)
            appearance_ids: Appearance embedding IDs (N, )
            return_weights: Whether to return sampling weights
            return_depth: Whether to return depth maps
            
        Returns:
            Dictionary containing rendered outputs
        """
        device = ray_origins.device
        N_rays = ray_origins.shape[0]
        
        # Normalize ray directions
        ray_directions = ray_directions / torch.norm(ray_directions, dim=-1, keepdim=True)
        
        # Sample points along rays (coarse sampling)
        t_vals_coarse = self._sample_along_rays(N_rays, self.num_coarse_samples, device)
        
        # Get sample points
        points_coarse = ray_origins.unsqueeze(1) + ray_directions.unsqueeze(1) * t_vals_coarse.unsqueeze(2)
        
        # Flatten for model forward pass
        points_flat = points_coarse.reshape(-1, 3)
        dirs_flat = ray_directions.unsqueeze(1)        
        
        # Appearance IDs
        if appearance_ids is not None:
            app_ids_flat = appearance_ids.unsqueeze(1)
        else:
            app_ids_flat = None
        
        # Forward pass (coarse)
        density_coarse, color_coarse = model(points_flat, dirs_flat, app_ids_flat)
        
        # Reshape outputs
        density_coarse = density_coarse.reshape(N_rays, self.num_coarse_samples, 1)
        color_coarse = color_coarse.reshape(N_rays, self.num_coarse_samples, 3)
        
        # Volume rendering (coarse)
        rgb_coarse, weights_coarse, depth_coarse = self._volume_render(
            density_coarse, color_coarse, t_vals_coarse
        )
        
        outputs: dict[str, torch.Tensor] = {}
        
        outputs['rgb_coarse'] = rgb_coarse
        outputs['depth_coarse'] = depth_coarse
        outputs['weights_coarse'] = weights_coarse
        
        # Fine sampling (hierarchical)
        if self.use_hierarchical_sampling and self.num_fine_samples > 0:
            # Sample additional points based on coarse weights
            t_vals_fine = self._hierarchical_sample(
                t_vals_coarse,
                weights_coarse,
                self.num_fine_samples
                )
            # Combine coarse and fine samples
            t_vals_combined = torch.cat([t_vals_coarse, t_vals_fine], dim=-1)
            t_vals_combined, sort_indices = torch.sort(t_vals_combined, dim=-1)
            
            # Get combined sample points
            points_combined = ray_origins.unsqueeze(1) + ray_directions.unsqueeze(1) * t_vals_combined.unsqueeze(2)
            
            # Flatten for model forward pass
            points_flat = points_combined.reshape(-1, 3)
            dirs_flat = ray_directions.unsqueeze(1)
            
            # Appearance IDs
            if appearance_ids is not None:
                app_ids_flat = appearance_ids.unsqueeze(1)
            else:
                app_ids_flat = None
            
            # Forward pass (fine)
            density_fine, color_fine = model(points_flat, dirs_flat, app_ids_flat)
            
            # Reshape outputs
            density_fine = density_fine.reshape(N_rays, t_vals_combined.shape[1], 1)
            color_fine = color_fine.reshape(N_rays, t_vals_combined.shape[1], 3)
            
            # Volume rendering (fine)
            rgb_fine, weights_fine, depth_fine = self._volume_render(
                density_fine, color_fine, t_vals_combined
            )
            
            outputs.update({
                'rgb_fine': rgb_fine,
                'depth_fine': depth_fine,
                'weights_fine': weights_fine,
                't_vals_fine': t_vals_combined
            })
        else:
            outputs.update({
                'rgb_coarse': rgb_coarse,
                'depth_coarse': depth_coarse,
                'weights_coarse': weights_coarse,
                't_vals_coarse': t_vals_coarse
            })
        
        # Add optional outputs
        if return_weights:
            if self.use_hierarchical_sampling:
                outputs['weights_fine'] = weights_fine
            else:
                outputs['weights_coarse'] = weights_coarse
        
        if return_depth:
            if self.use_hierarchical_sampling:
                outputs['depth_fine'] = depth_fine
            else:
                outputs['depth_coarse'] = depth_coarse
        
        return outputs
    
    def _sample_along_rays(self, N_rays: int, N_samples: int, device: torch.device) -> torch.Tensor:
        """Sample points along rays"""
        # Uniform sampling in depth
        t_vals = torch.linspace(self.near, self.far, N_samples, device=device)
        t_vals = t_vals.expand(N_rays, N_samples)
        
        # Add noise during training
        if self.training:
            # Get intervals between samples
            mids = 0.5 * (t_vals[:, 1:] + t_vals[:, :-1])
            upper = torch.cat([mids, t_vals[:, -1:]], dim=-1)
            lower = torch.cat([t_vals[:, :1], mids], dim=-1)
            
            # Uniform samples in those intervals
            t_rand = torch.rand_like(t_vals)
            t_vals = lower + (upper - lower) * t_rand
        
        return t_vals
    
    def _hierarchical_sample(
        self,
        t_vals: torch.Tensor,
        weights: torch.Tensor,
        N_samples: int
    ) -> torch.Tensor:
        """Hierarchical sampling based on coarse weights"""
        device = t_vals.device
        N_rays = t_vals.shape[0]
        
        # Get PDF from weights
        weights = weights + 1e-5  # Prevent division by zero
        pdf = weights / torch.sum(weights, dim=-1, keepdim=True)
        cdf = torch.cumsum(pdf, dim=-1)
        cdf = torch.cat([torch.zeros_like(cdf[:, :1]), cdf], dim=-1)
        
        # Sample from CDF
        u = torch.rand(N_rays, N_samples, device=device)
        
        # Invert CDF
        indices = torch.searchsorted(cdf, u, right=True)
        below = torch.clamp(indices - 1, 0, cdf.shape[-1] - 1)
        above = torch.clamp(indices, 0, cdf.shape[-1] - 1)
        
        # Linear interpolation
        indices_g = torch.stack([below, above], dim=-1)
        cdf_g = torch.gather(cdf.unsqueeze(-1).expand(-1, -1, 2), 1, indices_g)
        t_vals_g = torch.gather(t_vals.unsqueeze(-1).expand(-1, -1, 2), 1, indices_g)
        
        denom = cdf_g[:, :, 1] - cdf_g[:, :, 0]
        denom = torch.where(denom < 1e-5, torch.ones_like(denom), denom)
        
        t = (u - cdf_g[:, :, 0]) / denom
        t_vals_fine = t_vals_g[:, :, 0] + t * (t_vals_g[:, :, 1] - t_vals_g[:, :, 0])
        
        return t_vals_fine
    
    def _volume_render(
        self,
        density: torch.Tensor,
        color: torch.Tensor,
        t_vals: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Perform volume rendering"""
        # Compute deltas
        deltas = t_vals[:, 1:] - t_vals[:, :-1]
        deltas = torch.cat([deltas, torch.full_like(deltas[:, :1], 1e10)], dim=-1)
        
        # Compute alpha values
        alpha = 1.0 - torch.exp(-density.squeeze(-1) * deltas)
        
        # Compute transmittance
        transmittance = torch.cumprod(
            torch.cat([torch.ones_like(alpha[:, :1]), 1.0 - alpha[:, :-1] + 1e-10], dim=-1), dim=-1
        )[:, :-1]
        
        # Compute weights
        weights = alpha * transmittance
        
        # Render RGB
        rgb = torch.sum(weights.unsqueeze(-1) * color, dim=1)
        
        # Add background
        if self.white_background:
            acc_weights = torch.sum(weights, dim=-1)
            rgb = rgb + (1.0 - acc_weights.unsqueeze(-1))
        
        # Render depth
        depth = torch.sum(weights * t_vals, dim=-1)
        
        return rgb, weights, depth
    
    def render_image(
        self,
        model,
        camera_info,
        chunk_size: int = 1024,
        appearance_id: Optional[int] = None
    ) -> dict[str, np.ndarray]:
        """
        Render a full image
        
        Args:
            model: Mega-NeRF model
            camera_info: Camera information
            chunk_size: Chunk size for ray processing
            appearance_id: Appearance embedding ID
            
        Returns:
            Dictionary containing rendered image and depth
        """
        device = next(model.parameters()).device
        H, W = camera_info.height, camera_info.width
        
        # Generate rays
        ray_origins, ray_directions = self._generate_camera_rays(camera_info, device)
        ray_origins = ray_origins.reshape(-1, 3)
        ray_directions = ray_directions.reshape(-1, 3)
        
        # Appearance IDs
        if appearance_id is not None:
            appearance_ids = torch.full(
            )
        else:
            appearance_ids = None
        
        # Render in chunks
        rgb_chunks = []
        depth_chunks = []
        
        model.eval()
        with torch.no_grad():
            for i in tqdm(range(0, ray_origins.shape[0], chunk_size), desc="Rendering"):
                end_i = min(i + chunk_size, ray_origins.shape[0])
                
                chunk_origins = ray_origins[i:end_i]
                chunk_directions = ray_directions[i:end_i]
                chunk_appearance_ids = appearance_ids[i:end_i] if appearance_ids is not None else None
                
                outputs = self.render_rays(
                    model, chunk_origins, chunk_directions, chunk_appearance_ids
                )
                
                rgb_chunks.append(outputs['rgb'].cpu())
                depth_chunks.append(outputs['depth'].cpu())
        
        # Combine chunks
        rgb = torch.cat(rgb_chunks, dim=0).reshape(H, W, 3).numpy()
        depth = torch.cat(depth_chunks, dim=0).reshape(H, W).numpy()
        
        return {
            'rgb': rgb, 'depth': depth
        }
    
    def _generate_camera_rays(
        self,
        camera_info,
        device: torch.device
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Generate rays for a camera"""
        H, W = camera_info.height, camera_info.width
        
        # Pixel coordinates
        i, j = torch.meshgrid(
            torch.arange(W, device=device), torch.arange(H, device=device), indexing='xy'
        )
        
        # Camera coordinates
        intrinsics = torch.from_numpy(camera_info.intrinsics).to(device)
        dirs = torch.stack([
            (
                i - intrinsics[0,
                2]
	)        ], dim=-1).float()
        
        # Transform to world coordinates
        transform_matrix = torch.from_numpy(camera_info.transform_matrix).to(device)
        ray_directions = torch.sum(dirs[..., None, :] * transform_matrix[:3, :3], dim=-1)
        ray_origins = transform_matrix[:3, 3].expand(ray_directions.shape)
        
        return ray_origins, ray_directions


class BatchRenderer:
    """Efficient batch renderer for multiple rays and cameras"""
    
    def __init__(self, base_renderer: VolumetricRenderer):
        """
        Initialize batch renderer
        
        Args:
            base_renderer: Base volumetric renderer
        """
        self.base_renderer = base_renderer
    
    def render_rays_batch(
        self,
        model,
        ray_batch: dict[str,
        torch.Tensor],
        chunk_size: int = 1024
    ) -> dict[str, torch.Tensor]:
        """
        Render a batch of rays efficiently
        
        Args:
            model: Mega-NeRF model
            ray_batch: Batch of rays containing 'ray_origins', 'ray_directions', etc.
            chunk_size: Chunk size for processing
            
        Returns:
            Rendered outputs
        """
        ray_origins = ray_batch['ray_origins']
        ray_directions = ray_batch['ray_directions']
        appearance_ids = ray_batch.get('appearance_ids')
        
        N_rays = ray_origins.shape[0]
        
        # Process in chunks
        outputs = {}
        
        for i in range(0, N_rays, chunk_size):
            end_i = min(i + chunk_size, N_rays)
            
            chunk_origins = ray_origins[i:end_i]
            chunk_directions = ray_directions[i:end_i]
            chunk_appearance_ids = appearance_ids[i:end_i] if appearance_ids is not None else None
            
            chunk_outputs = self.base_renderer.render_rays(
                model, chunk_origins, chunk_directions, chunk_appearance_ids
            )
            
            # Accumulate outputs
            for key, value in chunk_outputs.items():
                if key not in outputs:
                    outputs[key] = []
                outputs[key].append(value)
        
        # Concatenate chunks
        for key in outputs:
            outputs[key] = torch.cat(outputs[key], dim=0)
        
        return outputs
    
    def render_multiple_views(
        self,
        model,
        camera_list: List,
        chunk_size: int = 1024,
        appearance_ids: Optional[list[int]] = None
    ) -> list[dict[str, np.ndarray]]:
        """
        Render multiple camera views
        
        Args:
            model: Mega-NeRF model
            camera_list: List of camera information
            chunk_size: Chunk size for ray processing
            appearance_ids: List of appearance IDs for each camera
            
        Returns:
            List of rendered outputs for each camera
        """
        results = []
        
        for i, camera_info in enumerate(camera_list):
            appearance_id = appearance_ids[i] if appearance_ids else None
            
            result = self.base_renderer.render_image(
                model, camera_info, chunk_size, appearance_id
            )
            
            results.append(result)
        
        return results


class InteractiveRenderer:
    """Interactive renderer with caching for real-time visualization"""
    
    def __init__(self, model, base_renderer: VolumetricRenderer, cache_size: int = 100):
        """
        Initialize interactive renderer
        
        Args:
            model: Mega-NeRF model
            base_renderer: Base volumetric renderer
            cache_size: Maximum number of cached renders
        """
        self.model = model
        self.base_renderer = base_renderer
        self.cache_size = cache_size
        self.render_cache = {}
        self.cache_order = []
    
    def render_view(
        self,
        camera_info,
        use_cache: bool = True,
        appearance_id: Optional[int] = None
    ) -> dict[str, np.ndarray]:
        """
        Render a view with caching
        
        Args:
            camera_info: Camera information
            use_cache: Whether to use cached results
            appearance_id: Appearance embedding ID
            
        Returns:
            Rendered outputs
        """
        # Create cache key
        cache_key = self._create_cache_key(camera_info, appearance_id)
        
        # Check cache
        if use_cache and cache_key in self.render_cache:
            return self.render_cache[cache_key]
        
        # Render new view
        result = self.base_renderer.render_image(
            self.model, camera_info, appearance_id=appearance_id
        )
        
        # Update cache
        if use_cache:
            self._update_cache(cache_key, result)
        
        return result
    
    def _create_cache_key(self, camera_info, appearance_id: Optional[int]) -> str:
        """Create a cache key for a camera view"""
        # Use camera position and orientation for key
        pos = camera_info.get_position()
        rot = camera_info.get_rotation()
        
        # Simple hash of camera parameters
        key_data = f"{pos[0]:.2f}_{pos[1]:.2f}_{pos[2]:.2f}_"
        key_data += f"{rot[0, 0]:.3f}_{rot[0, 1]:.3f}_{rot[0, 2]:.3f}_"
        key_data += f"{appearance_id if appearance_id is not None else 0}"
        
        return key_data
    
    def _update_cache(self, key: str, result: dict[str, np.ndarray]):
        """Update render cache"""
        # Remove oldest entry if cache is full
        if len(self.render_cache) >= self.cache_size:
            oldest_key = self.cache_order.pop(0)
            del self.render_cache[oldest_key]
        
        # Add new entry
        self.render_cache[key] = result
        self.cache_order.append(key)
    
    def clear_cache(self):
        """Clear render cache"""
        self.render_cache.clear()
        self.cache_order.clear()
    
    
    
    def get_cache_hit_rate(self) -> float:
        """Get cache hit rate"""
        return self.cache_hits / (self.cache_hits + self.cache_misses)
    
    def get_cache_stats(self) -> dict[str, Any]:
        """Get cache statistics"""
        return {
            'cache_size': len(self.render_cache),
            'cache_hits': self.cache_hits,
            'cache_misses': self.cache_misses
        }