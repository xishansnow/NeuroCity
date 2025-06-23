"""
Rendering utilities for InfNeRF.

This module provides tools for:
- Distributed rendering across multiple devices
- Memory-efficient rendering for large scenes
- Batch ray sampling and processing
- Performance optimization utilities
"""

import torch
import torch.nn.functional as F
import torch.distributed as dist
from torch.utils.data import DataLoader
import numpy as np
import math
from typing import Dict, List, Optional, Tuple, Any, Union, Callable
import time
from contextlib import contextmanager

from ..core import InfNeRF, InfNeRFConfig, OctreeNode


class DistributedRenderer:
    """
    Distributed renderer for InfNeRF across multiple devices.
    """
    
    def __init__(self, 
                 model: InfNeRF,
                 device_ids: List[int],
                 master_device: int = 0):
        """
        Initialize distributed renderer.
        
        Args:
            model: InfNeRF model
            device_ids: List of GPU device IDs
            master_device: Master device ID
        """
        self.model = model
        self.device_ids = device_ids
        self.master_device = master_device
        self.num_devices = len(device_ids)
        
        # Distribute octree nodes across devices
        self._distribute_nodes()
    
    def _distribute_nodes(self):
        """Distribute octree nodes across available devices."""
        all_nodes = self.model.all_nodes
        
        # Group nodes by spatial regions for better locality
        self.device_nodes = {device_id: [] for device_id in self.device_ids}
        
        # Simple round-robin distribution (could be improved with spatial partitioning)
        for i, node in enumerate(all_nodes):
            device_id = self.device_ids[i % self.num_devices]
            self.device_nodes[device_id].append(node)
            
            # Move node to appropriate device
            node.nerf.to(f'cuda:{device_id}')
    
    def render_distributed(self,
                          rays_o: torch.Tensor,
                          rays_d: torch.Tensor,
                          near: float,
                          far: float,
                          focal_length: float,
                          pixel_width: float,
                          chunk_size: int = 1024) -> Dict[str, torch.Tensor]:
        """
        Render rays using distributed processing.
        
        Args:
            rays_o: [N, 3] ray origins
            rays_d: [N, 3] ray directions
            near: Near plane distance
            far: Far plane distance
            focal_length: Camera focal length
            pixel_width: Pixel width
            chunk_size: Chunk size for processing
            
        Returns:
            Rendered outputs
        """
        num_rays = rays_o.shape[0]
        
        # Initialize output tensors
        device = f'cuda:{self.master_device}'
        colors = torch.zeros(num_rays, 3, device=device)
        depths = torch.zeros(num_rays, device=device)
        accs = torch.zeros(num_rays, device=device)
        
        # Process rays in chunks
        for i in range(0, num_rays, chunk_size):
            end_i = min(i + chunk_size, num_rays)
            
            chunk_rays_o = rays_o[i:end_i]
            chunk_rays_d = rays_d[i:end_i]
            
            # Render chunk
            chunk_result = self._render_chunk_distributed(
                chunk_rays_o, chunk_rays_d, near, far, focal_length, pixel_width
            )
            
            # Store results
            colors[i:end_i] = chunk_result['rgb']
            depths[i:end_i] = chunk_result['depth']
            accs[i:end_i] = chunk_result['acc']
        
        return {
            'rgb': colors,
            'depth': depths, 
            'acc': accs
        }
    
    def _render_chunk_distributed(self,
                                 rays_o: torch.Tensor,
                                 rays_d: torch.Tensor,
                                 near: float,
                                 far: float,
                                 focal_length: float,
                                 pixel_width: float) -> Dict[str, torch.Tensor]:
        """Render a chunk of rays using distributed nodes."""
        # Sample points along rays
        batch_size = rays_o.shape[0]
        num_samples = self.model.config.num_samples
        
        # Generate samples
        z_vals, pts = self._sample_rays(rays_o, rays_d, near, far, num_samples)
        radii = self._compute_radii(z_vals, focal_length, pixel_width)
        
        # Distribute rendering across devices
        colors = torch.zeros(batch_size, num_samples, 3, device=rays_o.device)
        densities = torch.zeros(batch_size, num_samples, 1, device=rays_o.device)
        
        # Process each device's nodes
        for device_id in self.device_ids:
            device_colors, device_densities = self._render_on_device(
                device_id, pts, rays_d, radii
            )
            
            # Aggregate results (simple addition - could be more sophisticated)
            colors += device_colors.to(rays_o.device)
            densities += device_densities.to(rays_o.device)
        
        # Volume rendering
        rendered = self._volume_render(colors, densities, z_vals, rays_d)
        
        return rendered
    
    def _render_on_device(self,
                         device_id: int,
                         pts: torch.Tensor,
                         rays_d: torch.Tensor,
                         radii: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Render points using nodes on a specific device."""
        device = f'cuda:{device_id}'
        nodes = self.device_nodes[device_id]
        
        batch_size, num_samples, _ = pts.shape
        colors = torch.zeros(batch_size, num_samples, 3, device=device)
        densities = torch.zeros(batch_size, num_samples, 1, device=device)
        
        # Move data to device
        pts_device = pts.to(device)
        rays_d_device = rays_d.to(device)
        radii_device = radii.to(device)
        
        # Process each sample
        for i in range(batch_size):
            for j in range(num_samples):
                sample_pos = pts_device[i, j]
                sample_radius = radii_device[i, j].item()
                
                # Find appropriate node on this device
                selected_node = self._find_node_on_device(
                    nodes, sample_pos, sample_radius
                )
                
                if selected_node is not None:
                    with torch.cuda.device(device):
                        result = selected_node.nerf(
                            sample_pos.unsqueeze(0),
                            rays_d_device[i:i+1]
                        )
                    
                    colors[i, j] = result['color'].squeeze(0)
                    densities[i, j] = result['density'].squeeze(0)
        
        return colors, densities
    
    def _find_node_on_device(self,
                            nodes: List[OctreeNode],
                            position: torch.Tensor,
                            radius: float) -> Optional[OctreeNode]:
        """Find appropriate node on device for given position and radius."""
        pos_np = position.detach().cpu().numpy()
        
        # Find nodes that contain this position
        for node in nodes:
            if not node.is_pruned and node.contains_point(pos_np):
                # Check if GSD is appropriate
                if node.gsd <= radius * 2:
                    return node
        
        return None
    
    def _sample_rays(self, rays_o, rays_d, near, far, num_samples):
        """Sample points along rays."""
        batch_size = rays_o.shape[0]
        device = rays_o.device
        
        t_vals = torch.linspace(0.0, 1.0, num_samples, device=device)
        z_vals = near + (far - near) * t_vals
        z_vals = z_vals.expand(batch_size, num_samples)
        
        pts = rays_o[..., None, :] + rays_d[..., None, :] * z_vals[..., :, None]
        
        return z_vals, pts
    
    def _compute_radii(self, z_vals, focal_length, pixel_width):
        """Compute sampling radii."""
        return z_vals * pixel_width / (2 * focal_length)
    
    def _volume_render(self, colors, densities, z_vals, rays_d):
        """Perform volume rendering."""
        dists = z_vals[..., 1:] - z_vals[..., :-1]
        dists = torch.cat([dists, torch.full_like(dists[..., :1], 1e10)], dim=-1)
        
        alpha = 1.0 - torch.exp(-densities[..., 0] * dists)
        T = torch.cumprod(1.0 - alpha + 1e-10, dim=-1)
        T = torch.cat([torch.ones_like(T[..., :1]), T[..., :-1]], dim=-1)
        
        weights = alpha * T
        rgb = torch.sum(weights[..., None] * colors, dim=-2)
        depth = torch.sum(weights * z_vals, dim=-1)
        acc = torch.sum(weights, dim=-1)
        
        return {'rgb': rgb, 'depth': depth, 'acc': acc}


class MemoryEfficientRenderer:
    """
    Memory-efficient renderer for large-scale scenes.
    """
    
    def __init__(self, 
                 model: InfNeRF,
                 max_memory_gb: float = 8.0,
                 adaptive_chunking: bool = True):
        """
        Initialize memory-efficient renderer.
        
        Args:
            model: InfNeRF model
            max_memory_gb: Maximum GPU memory to use
            adaptive_chunking: Use adaptive chunk sizing
        """
        self.model = model
        self.max_memory_gb = max_memory_gb
        self.adaptive_chunking = adaptive_chunking
        
        # Estimate memory usage
        self._estimate_memory_usage()
    
    def _estimate_memory_usage(self):
        """Estimate memory usage for rendering."""
        # Get model memory usage
        model_memory = self.model.get_memory_usage()
        self.model_memory_mb = model_memory['total_mb']
        
        # Calculate available memory for rendering
        self.available_memory_mb = (self.max_memory_gb * 1024) - self.model_memory_mb
        
        # Estimate chunk size based on available memory
        bytes_per_sample = 4 * (3 + 1 + 3)  # RGB + density + position (float32)
        samples_per_mb = (1024 * 1024) // bytes_per_sample
        
        self.base_chunk_size = int(self.available_memory_mb * samples_per_mb * 0.5)  # 50% margin
        self.base_chunk_size = max(64, min(self.base_chunk_size, 4096))  # Clamp
    
    @contextmanager
    def memory_monitor(self):
        """Context manager for monitoring memory usage."""
        if torch.cuda.is_available():
            initial_memory = torch.cuda.memory_allocated()
            yield
            final_memory = torch.cuda.memory_allocated()
            memory_used = (final_memory - initial_memory) / (1024 * 1024)  # MB
            
            if memory_used > self.available_memory_mb:
                print(f"Warning: Memory usage ({memory_used:.1f} MB) exceeds limit "
                      f"({self.available_memory_mb:.1f} MB)")
        else:
            yield
    
    def render_memory_efficient(self,
                               rays_o: torch.Tensor,
                               rays_d: torch.Tensor,
                               near: float,
                               far: float,
                               focal_length: float,
                               pixel_width: float,
                               progress_callback: Optional[Callable] = None) -> Dict[str, torch.Tensor]:
        """
        Render rays with memory-efficient chunking.
        
        Args:
            rays_o: [N, 3] ray origins
            rays_d: [N, 3] ray directions
            near: Near plane distance
            far: Far plane distance
            focal_length: Camera focal length
            pixel_width: Pixel width
            progress_callback: Optional progress callback function
            
        Returns:
            Rendered outputs
        """
        num_rays = rays_o.shape[0]
        device = rays_o.device
        
        # Initialize output tensors
        colors = torch.zeros(num_rays, 3, device=device)
        depths = torch.zeros(num_rays, device=device)
        accs = torch.zeros(num_rays, device=device)
        
        # Determine chunk size
        if self.adaptive_chunking:
            chunk_size = self._adaptive_chunk_size(num_rays)
        else:
            chunk_size = self.base_chunk_size
        
        # Process in chunks
        num_chunks = math.ceil(num_rays / chunk_size)
        
        for chunk_idx in range(num_chunks):
            start_idx = chunk_idx * chunk_size
            end_idx = min(start_idx + chunk_size, num_rays)
            
            with self.memory_monitor():
                # Extract chunk
                chunk_rays_o = rays_o[start_idx:end_idx]
                chunk_rays_d = rays_d[start_idx:end_idx]
                
                # Render chunk
                chunk_result = self._render_chunk_efficient(
                    chunk_rays_o, chunk_rays_d, near, far, focal_length, pixel_width
                )
                
                # Store results
                colors[start_idx:end_idx] = chunk_result['rgb']
                depths[start_idx:end_idx] = chunk_result['depth']
                accs[start_idx:end_idx] = chunk_result['acc']
            
            # Clear cache periodically
            if chunk_idx % 10 == 0:
                torch.cuda.empty_cache()
            
            # Progress callback
            if progress_callback:
                progress = (chunk_idx + 1) / num_chunks
                progress_callback(progress)
        
        return {
            'rgb': colors,
            'depth': depths,
            'acc': accs
        }
    
    def _adaptive_chunk_size(self, num_rays: int) -> int:
        """Determine adaptive chunk size based on scene complexity."""
        # Get number of active nodes
        active_nodes = sum(1 for node in self.model.all_nodes if not node.is_pruned)
        
        # Adjust chunk size based on scene complexity
        complexity_factor = max(0.1, min(1.0, 100 / active_nodes))
        adaptive_size = int(self.base_chunk_size * complexity_factor)
        
        # Ensure reasonable bounds
        return max(32, min(adaptive_size, num_rays))
    
    def _render_chunk_efficient(self, *args, **kwargs):
        """Render chunk with memory efficiency optimizations."""
        # Use the standard forward pass but with gradient checkpointing if training
        if self.model.training:
            return torch.utils.checkpoint.checkpoint(
                self.model.forward, *args, **kwargs
            )
        else:
            return self.model(*args, **kwargs)


def distributed_rendering(model: InfNeRF,
                         rays_o: torch.Tensor,
                         rays_d: torch.Tensor,
                         device_ids: List[int],
                         **kwargs) -> Dict[str, torch.Tensor]:
    """
    Perform distributed rendering across multiple devices.
    
    Args:
        model: InfNeRF model
        rays_o: [N, 3] ray origins
        rays_d: [N, 3] ray directions
        device_ids: List of GPU device IDs
        **kwargs: Additional rendering parameters
        
    Returns:
        Rendered outputs
    """
    renderer = DistributedRenderer(model, device_ids)
    return renderer.render_distributed(rays_o, rays_d, **kwargs)


def memory_efficient_rendering(model: InfNeRF,
                              rays_o: torch.Tensor,
                              rays_d: torch.Tensor,
                              max_memory_gb: float = 8.0,
                              **kwargs) -> Dict[str, torch.Tensor]:
    """
    Perform memory-efficient rendering.
    
    Args:
        model: InfNeRF model
        rays_o: [N, 3] ray origins
        rays_d: [N, 3] ray directions
        max_memory_gb: Maximum GPU memory to use
        **kwargs: Additional rendering parameters
        
    Returns:
        Rendered outputs
    """
    renderer = MemoryEfficientRenderer(model, max_memory_gb)
    return renderer.render_memory_efficient(rays_o, rays_d, **kwargs)


def batch_ray_sampling(images: List[torch.Tensor],
                      cameras: List[Dict[str, torch.Tensor]],
                      batch_size: int,
                      sampling_strategy: str = 'random') -> Dict[str, torch.Tensor]:
    """
    Sample rays from multiple images in batches.
    
    Args:
        images: List of images [H, W, 3]
        cameras: List of camera parameters
        batch_size: Number of rays per batch
        sampling_strategy: 'random', 'uniform', or 'importance'
        
    Returns:
        Batch of rays and target colors
    """
    all_rays_o = []
    all_rays_d = []
    all_colors = []
    
    # Generate rays for all images
    for img, cam in zip(images, cameras):
        rays_o, rays_d, colors = _generate_rays_from_image(img, cam)
        all_rays_o.append(rays_o)
        all_rays_d.append(rays_d)
        all_colors.append(colors)
    
    # Concatenate all rays
    all_rays_o = torch.cat(all_rays_o, dim=0)
    all_rays_d = torch.cat(all_rays_d, dim=0)
    all_colors = torch.cat(all_colors, dim=0)
    
    # Sample based on strategy
    if sampling_strategy == 'random':
        indices = torch.randperm(len(all_rays_o))[:batch_size]
    elif sampling_strategy == 'uniform':
        indices = torch.arange(0, len(all_rays_o), len(all_rays_o) // batch_size)[:batch_size]
    else:  # importance sampling
        indices = _importance_sampling(all_colors, batch_size)
    
    return {
        'rays_o': all_rays_o[indices],
        'rays_d': all_rays_d[indices],
        'target_rgb': all_colors[indices]
    }


def _generate_rays_from_image(image: torch.Tensor,
                             camera: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Generate rays from a single image."""
    H, W = image.shape[:2]
    device = image.device
    
    # Create pixel coordinates
    i, j = torch.meshgrid(
        torch.arange(W, device=device),
        torch.arange(H, device=device),
        indexing='xy'
    )
    
    # Camera parameters
    focal = camera['focal']
    principal = camera.get('principal', torch.tensor([W/2, H/2], device=device))
    extrinsic = camera['extrinsic']  # [4, 4]
    
    # Convert to camera coordinates
    dirs = torch.stack([
        (i - principal[0]) / focal[0],
        -(j - principal[1]) / focal[1],
        -torch.ones_like(i)
    ], dim=-1)  # [H, W, 3]
    
    # Transform to world coordinates
    rotation = extrinsic[:3, :3]
    translation = extrinsic[:3, 3]
    
    rays_d = dirs @ rotation.T  # [H, W, 3]
    rays_d = F.normalize(rays_d, dim=-1)
    
    # Ray origins (camera center)
    rays_o = -rotation.T @ translation  # [3]
    rays_o = rays_o.expand_as(rays_d)  # [H, W, 3]
    
    # Flatten
    rays_o = rays_o.reshape(-1, 3)
    rays_d = rays_d.reshape(-1, 3)
    colors = image.reshape(-1, 3)
    
    return rays_o, rays_d, colors


def _importance_sampling(colors: torch.Tensor, batch_size: int) -> torch.Tensor:
    """Perform importance sampling based on color variance."""
    # Compute importance weights based on color variance
    color_variance = torch.var(colors, dim=-1)
    weights = F.softmax(color_variance, dim=0)
    
    # Sample based on weights
    indices = torch.multinomial(weights, batch_size, replacement=True)
    
    return indices


class RenderingProfiler:
    """
    Profiler for analyzing rendering performance.
    """
    
    def __init__(self):
        """Initialize rendering profiler."""
        self.timings = {}
        self.memory_usage = {}
        self.ray_counts = {}
    
    @contextmanager
    def profile(self, name: str):
        """Profile a rendering operation."""
        # Start timing
        start_time = time.time()
        
        if torch.cuda.is_available():
            start_memory = torch.cuda.memory_allocated()
        
        yield
        
        # End timing
        end_time = time.time()
        elapsed = end_time - start_time
        
        if torch.cuda.is_available():
            end_memory = torch.cuda.memory_allocated()
            memory_used = (end_memory - start_memory) / (1024 * 1024)  # MB
        else:
            memory_used = 0
        
        # Store results
        if name not in self.timings:
            self.timings[name] = []
            self.memory_usage[name] = []
        
        self.timings[name].append(elapsed)
        self.memory_usage[name].append(memory_used)
    
    def get_stats(self) -> Dict[str, Dict[str, float]]:
        """Get profiling statistics."""
        stats = {}
        
        for name in self.timings:
            timings = self.timings[name]
            memory = self.memory_usage[name]
            
            stats[name] = {
                'avg_time': np.mean(timings),
                'std_time': np.std(timings),
                'min_time': np.min(timings),
                'max_time': np.max(timings),
                'avg_memory_mb': np.mean(memory),
                'max_memory_mb': np.max(memory),
                'num_calls': len(timings)
            }
        
        return stats
    
    def print_summary(self):
        """Print profiling summary."""
        stats = self.get_stats()
        
        print("\n=== Rendering Performance Summary ===")
        for name, stat in stats.items():
            print(f"\n{name}:")
            print(f"  Time: {stat['avg_time']:.3f}±{stat['std_time']:.3f}s "
                  f"(min: {stat['min_time']:.3f}s, max: {stat['max_time']:.3f}s)")
            print(f"  Memory: {stat['avg_memory_mb']:.1f} MB average, "
                  f"{stat['max_memory_mb']:.1f} MB peak")
            print(f"  Calls: {stat['num_calls']}")


# Global profiler instance
rendering_profiler = RenderingProfiler() 