from __future__ import annotations

from typing import Any, Optional, Union
"""
Multi-resolution rendering components for Mega-NeRF++

This module implements adaptive level-of-detail rendering optimized for 
high-resolution photogrammetric images and large-scale scenes.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math

class MultiResolutionRenderer(nn.Module):
    """
    Multi-resolution renderer that adapts detail level based on 
    viewing distance and image resolution
    """
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.num_lods = config.num_lods
        self.lod_threshold = config.lod_threshold
        
        # Sampling strategies for different LODs
        self.coarse_samples = [
            config.num_samples // (2**i) if i > 0 else config.num_samples
            for i in range(self.num_lods)
        ]
        
        self.fine_samples = [
            config.num_importance // (2**i) if i > 0 else config.num_importance
            for i in range(self.num_lods)
        ]
    
    def determine_lod(
        self,
        rays_o: torch.Tensor,
        rays_d: torch.Tensor,
        scene_bounds: torch.Tensor,
        target_resolution: float,
    ) -> torch.Tensor:
        """
        Determine level of detail for each ray
        
        Args:
            rays_o: [..., 3] ray origins
            rays_d: [..., 3] ray directions
            scene_bounds: [2, 3] scene bounding box
            target_resolution: Target pixel resolution
            
        Returns:
            [...] LOD level for each ray (0 = highest detail)
        """
        # Calculate distance to scene center
        scene_center = (scene_bounds[0] + scene_bounds[1]) / 2
        scene_size = torch.norm(scene_bounds[1] - scene_bounds[0])
        
        # Distance from ray origin to scene center
        distances = torch.norm(rays_o - scene_center, dim=-1)
        
        # Normalize distances
        normalized_distances = distances / scene_size
        
        # Calculate ray angles (how perpendicular to viewing direction)
        ray_angles = torch.abs(
            torch.sum,
        )
        
        # Combine distance and angle to determine LOD
        # Closer distances and direct angles get higher detail
        detail_factor = (1.0 - normalized_distances) * ray_angles
        
        # Map to LOD levels
        lod_levels = torch.zeros_like(detail_factor, dtype=torch.long)
        
        for lod in range(self.num_lods):
            threshold = 1.0 - (lod + 1) / self.num_lods
            lod_levels[detail_factor < threshold] = lod
        
        return lod_levels
    
    def adaptive_sampling(
        self,
        rays_o: torch.Tensor,
        rays_d: torch.Tensor,
        near: float,
        far: float,
        lod_levels: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Adaptive sampling based on LOD levels
        
        Args:
            rays_o: [..., 3] ray origins
            rays_d: [..., 3] ray directions
            near: Near plane distance
            far: Far plane distance
            lod_levels: [...] LOD level for each ray
            
        Returns:
            Dictionary with sampling results for different LODs
        """
        batch_shape = rays_o.shape[:-1]
        device = rays_o.device
        
        # Group rays by LOD level
        lod_groups = {}
        for lod in range(self.num_lods):
            mask = (lod_levels == lod)
            if mask.any():
                lod_groups[lod] = {
                    'mask': mask, 'rays_o': rays_o[mask], 'rays_d': rays_d[mask], 'indices': torch.where(
                        mask,
                    )
                }
        
        # Sample for each LOD group
        all_t_vals = torch.zeros(*batch_shape, self.coarse_samples[0], device=device)
        all_weights = torch.zeros(*batch_shape, self.coarse_samples[0], device=device)
        
        for lod, group in lod_groups.items():
            num_samples = self.coarse_samples[lod]
            
            # Sample t values for this LOD
            t_vals = self._sample_along_rays(
                group['rays_o'], group['rays_d'], near, far, num_samples
            )
            
            # Store results back to full tensor
            indices = group['indices']
            if t_vals.shape[-1] < all_t_vals.shape[-1]:
                # Pad with far values if fewer samples
                padding = all_t_vals.shape[-1] - t_vals.shape[-1]
                t_vals = torch.cat([
                    t_vals, torch.full((*t_vals.shape[:-1], padding), far, device=device)
                ], dim=-1)
            
            all_t_vals[indices] = t_vals[:, :all_t_vals.shape[-1]]
        
        return {
            't_vals': all_t_vals, 'lod_groups': lod_groups
        }
    
    def _sample_along_rays(
        self,
        rays_o: torch.Tensor,
        rays_d: torch.Tensor,
        near: float,
        far: float,
        num_samples: int,
    ) -> torch.Tensor:
        """Sample points along rays"""
        t_vals = torch.linspace(0., 1., num_samples, device=rays_o.device)
        
        # Linear sampling in disparity space for better near-field resolution
        t_vals = 1. / (1. / near * (1. - t_vals) + 1. / far * t_vals)
        
        # Add stratified sampling
        mids = 0.5 * (t_vals[:-1] + t_vals[1:])
        upper = torch.cat([mids, t_vals[-1:]])
        lower = torch.cat([t_vals[:1], mids])
        t_rand = torch.rand_like(t_vals)
        t_vals = lower + (upper - lower) * t_rand
        
        return t_vals.expand(*rays_o.shape[:-1], num_samples)

class AdaptiveLODRenderer(nn.Module):
    """
    Adaptive Level-of-Detail renderer with progressive refinement
    """
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.multires_renderer = MultiResolutionRenderer(config)
        
        # Progressive refinement parameters
        self.refinement_levels = 3
        self.refinement_threshold = 0.05
        
    def render_with_lod(
        self,
        model: nn.Module,
        rays_o: torch.Tensor,
        rays_d: torch.Tensor,
        scene_bounds: torch.Tensor,
        near: float,
        far: float,
        target_resolution: float = 1.0,
    ) -> Dict[str, torch.Tensor]:
        """
        Render with adaptive level of detail
        
        Args:
            model: NeRF model to use for rendering
            rays_o: [..., 3] ray origins
            rays_d: [..., 3] ray directions
            scene_bounds: [2, 3] scene bounding box
            near: Near plane distance
            far: Far plane distance
            target_resolution: Target pixel resolution
            
        Returns:
            Rendered results with LOD information
        """
        # Determine LOD for each ray
        lod_levels = self.multires_renderer.determine_lod(
            rays_o, rays_d, scene_bounds, target_resolution
        )
        
        # Adaptive sampling
        sampling_results = self.multires_renderer.adaptive_sampling(
            rays_o, rays_d, near, far, lod_levels
        )
        
        # Render each LOD group
        results = self._render_lod_groups(
            model, sampling_results, rays_o, rays_d, lod_levels
        )
        
        # Progressive refinement for high-detail areas
        if target_resolution > 0.5:  # Only for high-resolution rendering
            results = self._progressive_refinement(
                model, results, rays_o, rays_d, near, far, lod_levels
            )
        
        return results
    
    def _render_lod_groups(
        self,
        model: nn.Module,
        sampling_results: dict,
        rays_o: torch.Tensor,
        rays_d: torch.Tensor,
        lod_levels: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """Render each LOD group separately"""
        
        batch_shape = rays_o.shape[:-1]
        device = rays_o.device
        
        # Initialize output tensors
        rgb_output = torch.zeros(*batch_shape, 3, device=device)
        depth_output = torch.zeros(*batch_shape, device=device)
        alpha_output = torch.zeros(*batch_shape, device=device)
        
        t_vals = sampling_results['t_vals']
        
        # Render each LOD group
        for lod, group in sampling_results['lod_groups'].items():
            if not group['mask'].any():
                continue
                
            indices = group['indices']
            group_rays_o = group['rays_o']
            group_rays_d = group['rays_d']
            group_t_vals = t_vals[indices]
            
            # Sample positions along rays
            pts = group_rays_o[..., None, :] + group_rays_d[..., None, :] * group_t_vals[..., :, None]
            
            # Forward through model at appropriate LOD
            output = model(pts, group_rays_d[..., None, :].expand_as(pts), lod=lod)
            
            # Volume rendering
            rendered = self._volume_render(
                output['density'], output['color'], group_t_vals, group_rays_d
            )
            
            # Store results
            rgb_output[indices] = rendered['rgb']
            depth_output[indices] = rendered['depth']
            alpha_output[indices] = rendered['acc_alpha']
        
        return {
            'rgb': rgb_output, 'depth': depth_output, 'acc_alpha': alpha_output, 'lod_levels': lod_levels
        }
    
    def _progressive_refinement(
        self,
        model: nn.Module,
        initial_results: dict,
        rays_o: torch.Tensor,
        rays_d: torch.Tensor,
        near: float,
        far: float,
        lod_levels: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """Apply progressive refinement to high-detail areas"""
        
        results = initial_results.copy()
        
        # Identify rays that need refinement
        high_detail_mask = (lod_levels == 0)  # Highest detail level
        
        if not high_detail_mask.any():
            return results
        
        # Extract high-detail rays
        hd_rays_o = rays_o[high_detail_mask]
        hd_rays_d = rays_d[high_detail_mask]
        hd_indices = torch.where(high_detail_mask)
        
        # Progressive refinement iterations
        for refinement_level in range(self.refinement_levels):
            # Increase sampling density
            num_samples = self.config.num_samples * (2 ** refinement_level)
            
            # Sample with higher density
            t_vals = self._dense_sampling(hd_rays_o, hd_rays_d, near, far, num_samples)
            
            # Sample positions
            pts = hd_rays_o[..., None, :] + hd_rays_d[..., None, :] * t_vals[..., :, None]
            
            # Forward through model
            output = model(pts, hd_rays_d[..., None, :].expand_as(pts), lod=0)
            
            # Volume rendering
            refined = self._volume_render(
                output['density'], output['color'], t_vals, hd_rays_d
            )
            
            # Update results for high-detail rays
            results['rgb'][hd_indices] = refined['rgb']
            results['depth'][hd_indices] = refined['depth']
            results['acc_alpha'][hd_indices] = refined['acc_alpha']
            
            # Check convergence (simplified)
            if refinement_level > 0:
                # In practice, would check for convergence criteria
                break
        
        return results
    
    def _dense_sampling(
        self,
        rays_o: torch.Tensor,
        rays_d: torch.Tensor,
        near: float,
        far: float,
        num_samples: int,
    ) -> torch.Tensor:
        """Dense sampling for progressive refinement"""
        
        # Use importance sampling based on previous results
        # For now, use stratified sampling with higher density
        t_vals = torch.linspace(0., 1., num_samples, device=rays_o.device)
        
        # Inverse depth sampling
        t_vals = 1. / (1. / near * (1. - t_vals) + 1. / far * t_vals)
        
        # Fine stratified sampling
        t_vals = t_vals.expand(*rays_o.shape[:-1], num_samples)
        noise = torch.rand_like(t_vals) * (far - near) / num_samples
        t_vals = t_vals + noise
        
        return t_vals
    
    def _volume_render(
        self,
        densities: torch.Tensor,
        colors: torch.Tensor,
        t_vals: torch.Tensor,
        rays_d: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """Volume rendering computation"""
        
        # Compute distances between samples
        dists = torch.diff(t_vals, dim=-1)
        dists = torch.cat([dists, torch.full_like(dists[..., :1], 1e10)], dim=-1)
        
        # Multiply by ray direction norm
        dists = dists * torch.norm(rays_d[..., None, :], dim=-1)
        
        # Compute alpha values
        alpha = 1.0 - torch.exp(-F.relu(densities[..., 0]) * dists)
        
        # Compute transmittance
        transmittance = torch.cumprod(1.0 - alpha + 1e-10, dim=-1)
        transmittance = torch.cat([
            torch.ones_like(transmittance[..., :1]), transmittance[..., :-1]
        ], dim=-1)
        
        # Compute weights
        weights = alpha * transmittance
        
        # Accumulate colors and depth
        rgb = torch.sum(weights[..., None] * colors, dim=-2)
        depth = torch.sum(weights * t_vals, dim=-1)
        acc_alpha = torch.sum(weights, dim=-1)
        
        return {
            'rgb': rgb, 'depth': depth, 'acc_alpha': acc_alpha, 'weights': weights
        }

class PhotogrammetricVolumetricRenderer(nn.Module):
    """
    Specialized volumetric renderer optimized for photogrammetric data
    """
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.adaptive_renderer = AdaptiveLODRenderer(config)
        
        # Photogrammetric-specific parameters
        self.depth_regularization = True
        self.semantic_rendering = hasattr(config, 'lambda_semantic') and config.lambda_semantic > 0
        
    def render_photogrammetric(
        self,
        model: nn.Module,
        rays_o: torch.Tensor,
        rays_d: torch.Tensor,
        scene_bounds: torch.Tensor,
        camera_intrinsics: torch.Tensor,
        image_resolution: Tuple[int,
        int],
        near: float,
        far: float,
    ) -> Dict[str, torch.Tensor]:
        """
        Render with photogrammetric optimizations
        
        Args:
            model: NeRF model
            rays_o: [..., 3] ray origins
            rays_d: [..., 3] ray directions
            scene_bounds: [2, 3] scene bounding box
            camera_intrinsics: [3, 3] camera intrinsic matrix
            image_resolution: (width, height) of target image
            near: Near plane distance
            far: Far plane distance
            
        Returns:
            Rendered results with photogrammetric information
        """
        # Calculate target resolution based on image size and scene scale
        scene_size = torch.norm(scene_bounds[1] - scene_bounds[0])
        pixel_size = scene_size / max(image_resolution)
        target_resolution = 1.0 / pixel_size
        
        # Adaptive LOD rendering
        results = self.adaptive_renderer.render_with_lod(
            model, rays_o, rays_d, scene_bounds, near, far, target_resolution
        )
        
        # Add photogrammetric-specific computations
        if self.depth_regularization:
            results = self._add_depth_regularization(results, rays_o, rays_d)
        
        if self.semantic_rendering:
            results = self._add_semantic_rendering(model, results, rays_o, rays_d)
        
        # Add photogrammetric quality metrics
        results = self._compute_photogrammetric_metrics(results, camera_intrinsics)
        
        return results
    
    def _add_depth_regularization(
        self,
        results: dict,
        rays_o: torch.Tensor,
        rays_d: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:   
        """Add depth regularization for better geometric consistency"""
        
        # Compute depth gradients for smoothness
        depth = results['depth']
        
        if len(depth.shape) >= 2:  # Spatial dimensions available
            # Compute spatial gradients
            depth_dx = torch.diff(depth, dim=-1, prepend=depth[..., :1])
            depth_dy = torch.diff(depth, dim=-2, prepend=depth[..., :1, :])
            
            # Depth smoothness loss
            depth_smoothness = torch.mean(torch.abs(depth_dx)) + torch.mean(torch.abs(depth_dy))
            results['depth_smoothness'] = depth_smoothness
        
        # Depth variance for confidence estimation
        if 'weights' in results:
            depth_variance = torch.sum(
                results['weights'] * (results['depth'] - results['depth_mean']) ** 2,
            )
            results['depth_variance'] = depth_variance
            results['depth_confidence'] = 1.0 / (1.0 + depth_variance)
        
        return results
    
    def _add_semantic_rendering(
        self,
        model: nn.Module,
        results: dict,
        rays_o: torch.Tensor,
        rays_d: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """Add semantic rendering if supported by the model"""
        
        # Check if model supports semantic output
        if hasattr(model, 'semantic_head'):
            # Re-render with semantic output
            # This is a simplified implementation
            semantic_output = torch.zeros(*results['rgb'].shape[:-1], device=results['rgb'].device)
            results['semantics'] = semantic_output
        
        return results
    
    def _compute_photogrammetric_metrics(
        self,
        results: dict,
        camera_intrinsics: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """Compute photogrammetric quality metrics"""
        # cannot import name 'AdaptiveOctree' from 'nerfs.mega_nerf_plus.spatial_partitioner'
        from nerfs.mega_nerf_plus.spatial_partitioner import AdaptiveOctree

        # Get the octree
        octree = AdaptiveOctree(self.config)

        # Get the partitions
        partitions = octree.get_partitions(self.config)
        # Reprojection accuracy (simplified)
        if 'depth' in results:
            focal_length = camera_intrinsics[0, 0]  # Assuming fx
            
            # Estimate reprojection error based on depth uncertainty
            depth_uncertainty = results.get('depth_variance', torch.zeros_like(results['depth']))
            reprojection_error = depth_uncertainty / focal_length
            results['reprojection_error'] = reprojection_error
        
        # Coverage confidence based on accumulated alpha
        if 'acc_alpha' in results:
            coverage_confidence = results['acc_alpha']
            results['coverage_confidence'] = coverage_confidence
        
        # Multi-view consistency (would require multiple views)
        # This is a placeholder - real implementation would compare across views
        consistency_score = torch.ones_like(results['depth'])
        results['multiview_consistency'] = consistency_score
        
        return results
    
    def batch_render(
        self,
        model: nn.Module,
        rays_batch: Dict[str,
        torch.Tensor],
        scene_bounds: torch.Tensor,
        chunk_size: int = 1024,
    ) -> Dict[str, torch.Tensor]:
        """
        Batch rendering for large numbers of rays
        
        Args:
            model: NeRF model
            rays_batch: Dictionary with 'origins', 'directions', etc.
            scene_bounds: Scene bounding box
            chunk_size: Number of rays to process at once
            
        Returns:
            Batched rendering results
        """
        rays_o = rays_batch['origins']
        rays_d = rays_batch['directions']
        total_rays = rays_o.shape[0]
        
        # Initialize output tensors
        all_results = {}
        
        # Process in chunks
        for i in range(0, total_rays, chunk_size):
            end_i = min(i + chunk_size, total_rays)
            
            chunk_rays_o = rays_o[i:end_i]
            chunk_rays_d = rays_d[i:end_i]
            
            # Render chunk
            chunk_results = self.render_photogrammetric(
                model, chunk_rays_o, chunk_rays_d, scene_bounds, rays_batch.get(
                    'intrinsics',
                    torch.eye,
                )
            )
            
            # Accumulate results
            if i == 0:
                # Initialize with first chunk
                for key, value in chunk_results.items():
                    if isinstance(value, torch.Tensor):
                        all_results[key] = torch.zeros(
                            total_rays,
                            *value.shape[1:],
                            device=value.device,
                            dtype=value.dtype,
                        )
                        all_results[key][i:end_i] = value
                    else:
                        all_results[key] = value
            else:
                # Accumulate subsequent chunks
                for key, value in chunk_results.items():
                    if isinstance(value, torch.Tensor) and key in all_results:
                        all_results[key][i:end_i] = value
        
        return all_results 