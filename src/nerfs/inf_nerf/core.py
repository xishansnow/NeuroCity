"""
Core implementation of InfNeRF: Infinite Scale NeRF with O(log n) Space Complexity

This module implements the main components of InfNeRF including:
- Octree-based Level of Detail (LoD) structure
- Multi-scale neural radiance fields
- Anti-aliasing rendering
- Memory-efficient large-scale scene handling
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from typing import Dict, List, Optional, Tuple, Union, Any
from dataclasses import dataclass
from abc import ABC, abstractmethod


@dataclass
class InfNeRFConfig:
    """Configuration for InfNeRF model."""
    
    # Octree structure parameters
    max_depth: int = 8                   # Maximum octree depth
    min_depth: int = 2                   # Minimum octree depth
    adaptive_depth: bool = True          # Use adaptive depth based on scene complexity
    
    # Level of Detail parameters  
    base_resolution: int = 32            # Base grid resolution at root
    grid_size: int = 2048               # Grid size for each node
    max_gsd: float = 1.0                # Maximum Ground Sampling Distance (meters)
    min_gsd: float = 0.01               # Minimum Ground Sampling Distance (meters)
    
    # Neural network parameters
    hidden_dim: int = 64                # Hidden layer dimension
    geo_feat_dim: int = 15              # Geometry feature dimension  
    num_layers: int = 2                 # Number of MLP layers
    num_layers_color: int = 3           # Color network layers
    
    # Hash encoding parameters (per node)
    num_levels: int = 16                # Hash encoding levels per node
    level_dim: int = 2                  # Features per level
    per_level_scale: float = 2.0        # Scale factor between levels
    log2_hashmap_size: int = 19         # Hash table size (2^19)
    
    # Sampling and rendering
    num_samples: int = 64               # Number of coarse samples
    num_importance: int = 128           # Number of fine samples  
    perturb_radius: bool = True         # Perturb sampling radius for anti-aliasing
    radius_perturbation_range: float = 0.5  # Perturbation range
    
    # Training parameters
    learning_rate: float = 1e-2         # Learning rate
    weight_decay: float = 1e-6          # Weight decay
    batch_size: int = 4096              # Ray batch size
    
    # Loss weights
    lambda_rgb: float = 1.0             # RGB loss weight
    lambda_depth: float = 0.1           # Depth loss weight
    lambda_distortion: float = 0.01     # Distortion loss weight
    lambda_transparency: float = 1e-3   # Transparency loss weight
    lambda_regularization: float = 1e-4  # Level consistency regularization
    
    # Memory and performance
    use_mixed_precision: bool = True    # Use mixed precision training
    max_memory_gb: float = 16.0         # Maximum GPU memory usage
    distributed_training: bool = False   # Enable distributed training
    
    # Scene bounds
    scene_bound: float = 1.0            # Scene bounding box size
    
    # Pruning parameters
    use_pruning: bool = True            # Enable octree pruning
    pruning_threshold: float = 1e-4     # Density threshold for pruning
    sparse_point_threshold: int = 10    # Minimum sparse points for node


class OctreeNode:
    """
    Individual node in the InfNeRF octree structure.
    Each node represents a cubic space with its own NeRF.
    """
    
    def __init__(self, 
                 center: np.ndarray,
                 size: float,
                 level: int,
                 config: InfNeRFConfig,
                 parent: Optional['OctreeNode'] = None):
        """
        Initialize octree node.
        
        Args:
            center: Center coordinates of the node [3]
            size: Size of the cubic space
            level: Depth level in octree (0 = root)
            config: InfNeRF configuration
            parent: Parent node (None for root)
        """
        self.center = center
        self.size = size
        self.level = level
        self.config = config  
        self.parent = parent
        self.children: List[Optional['OctreeNode']] = [None] * 8
        self.is_leaf = True
        
        # Calculate Ground Sampling Distance (GSD)
        self.gsd = size / config.grid_size
        
        # Initialize NeRF for this node
        self.nerf = LoDAwareNeRF(config, level)
        
        # AABB bounds
        half_size = size / 2
        self.bounds_min = center - half_size
        self.bounds_max = center + half_size
        
        # Sparse points for pruning (will be populated during training)
        self.sparse_points: List[np.ndarray] = []
        self.is_pruned = False
    
    def get_aabb(self) -> Tuple[np.ndarray, np.ndarray]:
        """Get axis-aligned bounding box."""
        return self.bounds_min.copy(), self.bounds_max.copy()
    
    def contains_point(self, point: np.ndarray) -> bool:
        """Check if point is within this node's AABB."""
        return np.all(point >= self.bounds_min) and np.all(point < self.bounds_max)
    
    def subdivide(self) -> List['OctreeNode']:
        """
        Subdivide this node into 8 children.
        
        Returns:
            List of 8 child nodes
        """
        if not self.is_leaf:
            return [child for child in self.children if child is not None]
        
        self.is_leaf = False
        child_size = self.size / 2
        child_level = self.level + 1
        
        # Create 8 child nodes
        for i in range(8):
            # Calculate child center offset
            offset = np.array([
                (i & 1) * child_size - child_size/2,
                ((i >> 1) & 1) * child_size - child_size/2, 
                ((i >> 2) & 1) * child_size - child_size/2
            ])
            
            child_center = self.center + offset
            self.children[i] = OctreeNode(
                center=child_center,
                size=child_size,
                level=child_level,
                config=self.config,
                parent=self
            )
        
        return [child for child in self.children if child is not None]
    
    def find_containing_child(self, point: np.ndarray) -> Optional['OctreeNode']:
        """Find which child node contains the given point."""
        if self.is_leaf:
            return None
            
        for child in self.children:
            if child is not None and child.contains_point(point):
                return child
        return None
    
    def get_memory_size(self) -> int:
        """Estimate memory size of this node in bytes."""
        # Estimate based on NeRF parameters
        param_count = sum(p.numel() for p in self.nerf.parameters())
        return param_count * 4  # 4 bytes per float32 parameter


class LoDAwareNeRF(nn.Module):
    """
    Level-of-Detail aware Neural Radiance Field.
    Each octree node has its own LoDAwareNeRF with appropriate complexity.
    """
    
    def __init__(self, config: InfNeRFConfig, level: int):
        super().__init__()
        self.config = config
        self.level = level
        
        # Adjust network complexity based on level
        # Higher levels (finer detail) get more complex networks
        complexity_factor = min(1.0, (level + 1) / config.max_depth)
        
        self.hidden_dim = max(32, int(config.hidden_dim * complexity_factor))
        self.num_layers = max(1, int(config.num_layers * complexity_factor))
        
        # Hash encoding for spatial features
        self.spatial_encoder = HashEncoder(config)
        
        # Direction encoding (shared across levels)
        self.dir_encoder = SphericalHarmonicsEncoder(degree=4)
        
        # Geometry network
        geo_layers = []
        input_dim = self.spatial_encoder.output_dim
        
        geo_layers.append(nn.Linear(input_dim, self.hidden_dim))
        geo_layers.append(nn.ReLU(inplace=True))
        
        for _ in range(self.num_layers - 1):
            geo_layers.append(nn.Linear(self.hidden_dim, self.hidden_dim))
            geo_layers.append(nn.ReLU(inplace=True))
        
        self.geo_net = nn.Sequential(*geo_layers)
        
        # Density head
        self.density_head = nn.Linear(self.hidden_dim, 1 + config.geo_feat_dim)
        
        # Color network
        color_input_dim = config.geo_feat_dim + self.dir_encoder.output_dim
        color_layers = []
        
        color_layers.append(nn.Linear(color_input_dim, config.hidden_dim))
        color_layers.append(nn.ReLU(inplace=True))
        
        for _ in range(config.num_layers_color - 1):
            color_layers.append(nn.Linear(config.hidden_dim, config.hidden_dim))
            color_layers.append(nn.ReLU(inplace=True))
        
        self.color_net = nn.Sequential(*color_layers)
        self.color_head = nn.Linear(config.hidden_dim, 3)
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, m):
        """Initialize network weights."""
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                torch.nn.init.zeros_(m.bias)
    
    def forward(self, positions: torch.Tensor, directions: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Forward pass through LoD-aware NeRF.
        
        Args:
            positions: [N, 3] 3D positions  
            directions: [N, 3] viewing directions
            
        Returns:
            Dictionary with 'density' and 'color' keys
        """
        # Encode spatial positions
        spatial_features = self.spatial_encoder(positions)
        
        # Geometry forward pass
        geo_features = self.geo_net(spatial_features)
        geo_out = self.density_head(geo_features)
        
        # Split density and geometry features
        density = geo_out[..., :1]  # [N, 1]
        geo_feat = geo_out[..., 1:]  # [N, geo_feat_dim]
        
        # Encode viewing directions
        dir_features = self.dir_encoder(directions)
        
        # Color forward pass
        color_input = torch.cat([geo_feat, dir_features], dim=-1)
        color_features = self.color_net(color_input)
        color = self.color_head(color_features)
        
        # Apply activations
        density = F.softplus(density - 1.0)  # Density activation
        color = torch.sigmoid(color)         # RGB activation
        
        return {
            'density': density,
            'color': color
        }


class HashEncoder(nn.Module):
    """Multi-resolution hash encoding for spatial positions."""
    
    def __init__(self, config: InfNeRFConfig):
        super().__init__()
        self.config = config
        
        # Initialize hash tables for each level
        self.embeddings = nn.ModuleList()
        
        for i in range(config.num_levels):
            resolution = int(config.base_resolution * (config.per_level_scale ** i))
            params_in_level = min(resolution ** 3, 2 ** config.log2_hashmap_size)
            
            embedding = nn.Embedding(params_in_level, config.level_dim)
            nn.init.uniform_(embedding.weight, -1e-4, 1e-4)
            self.embeddings.append(embedding)
        
        self.output_dim = config.num_levels * config.level_dim
    
    def forward(self, positions: torch.Tensor) -> torch.Tensor:
        """Hash encode positions."""
        # Simplified hash encoding - in practice would use optimized CUDA implementation
        encoded_features = []
        
        for i, embedding in enumerate(self.embeddings):
            resolution = int(self.config.base_resolution * (self.config.per_level_scale ** i))
            
            # Scale positions to grid resolution
            scaled_pos = (positions + 1.0) / 2.0 * (resolution - 1)
            scaled_pos = torch.clamp(scaled_pos, 0, resolution - 1)
            
            # Simple hash function (would be optimized in practice)
            indices = (scaled_pos[..., 0] * resolution**2 + 
                      scaled_pos[..., 1] * resolution + 
                      scaled_pos[..., 2]).long()
            indices = torch.clamp(indices, 0, embedding.num_embeddings - 1)
            
            features = embedding(indices)
            encoded_features.append(features)
        
        return torch.cat(encoded_features, dim=-1)


class SphericalHarmonicsEncoder(nn.Module):
    """Spherical harmonics encoding for viewing directions."""
    
    def __init__(self, degree: int = 4):
        super().__init__()
        self.degree = degree
        self.output_dim = (degree + 1) ** 2
    
    def forward(self, directions: torch.Tensor) -> torch.Tensor:
        """Encode directions using spherical harmonics."""
        # Simplified SH encoding - would use optimized implementation
        x, y, z = directions.unbind(-1)
        
        # Degree 0
        features = [torch.ones_like(x) * 0.5]
        
        # Degree 1  
        if self.degree >= 1:
            features.extend([x, y, z])
        
        # Degree 2
        if self.degree >= 2:
            features.extend([
                x * y, x * z, y * z,
                x * x - y * y, 3 * z * z - 1
            ])
        
        # Higher degrees would be added here
        
        return torch.stack(features[:self.output_dim], dim=-1)


class InfNeRFRenderer(nn.Module):
    """Renderer for InfNeRF with octree-based sampling."""
    
    def __init__(self, config: InfNeRFConfig):
        super().__init__()
        self.config = config
    
    def sample_rays_lod(self, 
                       rays_o: torch.Tensor, 
                       rays_d: torch.Tensor,
                       near: float, 
                       far: float,
                       num_samples: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Sample points along rays with LoD-aware sampling.
        
        Args:
            rays_o: [N, 3] ray origins
            rays_d: [N, 3] ray directions  
            near: near plane distance
            far: far plane distance
            num_samples: number of samples per ray
            
        Returns:
            z_vals: [N, num_samples] sample distances
            pts: [N, num_samples, 3] sample points
        """
        batch_size = rays_o.shape[0]
        device = rays_o.device
        
        # Stratified sampling
        t_vals = torch.linspace(0.0, 1.0, num_samples, device=device)
        z_vals = near + (far - near) * t_vals
        z_vals = z_vals.expand(batch_size, num_samples)
        
        # Add noise for stratified sampling
        if self.training:
            mids = 0.5 * (z_vals[..., 1:] + z_vals[..., :-1])
            upper = torch.cat([mids, z_vals[..., -1:]], dim=-1)
            lower = torch.cat([z_vals[..., :1], mids], dim=-1)
            t_rand = torch.rand_like(z_vals)
            z_vals = lower + (upper - lower) * t_rand
        
        # Get sample points
        pts = rays_o[..., None, :] + rays_d[..., None, :] * z_vals[..., :, None]
        
        return z_vals, pts
    
    def calculate_sample_radius(self, 
                              z_vals: torch.Tensor,
                              focal_length: float,
                              pixel_width: float) -> torch.Tensor:
        """
        Calculate sampling sphere radius based on pixel footprint.
        
        Args:
            z_vals: [N, num_samples] sample distances
            focal_length: camera focal length
            pixel_width: pixel width in image coordinates
            
        Returns:
            radii: [N, num_samples] sampling radii
        """
        # Radius proportional to depth (Eq. 3 in paper)
        radii = z_vals * pixel_width / (2 * focal_length)
        
        # Add perturbation for anti-aliasing (Eq. 4 in paper)
        if self.config.perturb_radius and self.training:
            perturbation = torch.rand_like(radii) - 0.5  # [-0.5, 0.5] 
            perturbation *= self.config.radius_perturbation_range
            radii = radii * (2.0 ** perturbation)
        
        return radii
    
    def volume_render(self,
                     colors: torch.Tensor,
                     densities: torch.Tensor, 
                     z_vals: torch.Tensor,
                     rays_d: torch.Tensor,
                     white_bkgd: bool = False) -> Dict[str, torch.Tensor]:
        """
        Volume render colors and densities.
        
        Args:
            colors: [N, num_samples, 3] RGB colors
            densities: [N, num_samples, 1] volume densities
            z_vals: [N, num_samples] sample distances
            rays_d: [N, 3] ray directions
            white_bkgd: whether to use white background
            
        Returns:
            Dictionary with rendered outputs
        """
        # Calculate distances between samples
        dists = z_vals[..., 1:] - z_vals[..., :-1]
        dists = torch.cat([dists, torch.full_like(dists[..., :1], 1e10)], dim=-1)
        
        # Account for ray direction magnitude
        dists = dists * torch.norm(rays_d[..., None, :], dim=-1)
        
        # Calculate alpha values
        alpha = 1.0 - torch.exp(-densities[..., 0] * dists)
        
        # Calculate transmittance
        T = torch.cumprod(1.0 - alpha + 1e-10, dim=-1)
        T = torch.cat([torch.ones_like(T[..., :1]), T[..., :-1]], dim=-1)
        
        # Calculate weights
        weights = alpha * T
        
        # Render RGB
        rgb = torch.sum(weights[..., None] * colors, dim=-2)
        
        # Render depth  
        depth = torch.sum(weights * z_vals, dim=-1)
        
        # Render weights (for importance sampling)
        acc = torch.sum(weights, dim=-1)
        
        # Add background
        if white_bkgd:
            rgb = rgb + (1.0 - acc[..., None])
        
        return {
            'rgb': rgb,
            'depth': depth, 
            'acc': acc,
            'weights': weights
        }


class InfNeRF(nn.Module):
    """
    Main InfNeRF model with octree-based Level of Detail structure.
    """
    
    def __init__(self, config: InfNeRFConfig):
        super().__init__()
        self.config = config
        
        # Initialize octree structure
        self.root_node = OctreeNode(
            center=np.zeros(3),
            size=config.scene_bound * 2,  # Full scene size
            level=0,
            config=config
        )
        
        # Renderer
        self.renderer = InfNeRFRenderer(config)
        
        # Keep track of all nodes for training
        self.all_nodes: List[OctreeNode] = [self.root_node]
        
        # Level selection threshold
        self.level_threshold = config.max_gsd / config.min_gsd
    
    def build_octree(self, sparse_points: np.ndarray, max_depth: Optional[int] = None):
        """
        Build octree structure based on sparse points from SfM.
        
        Args:
            sparse_points: [N, 3] sparse points from structure from motion
            max_depth: maximum octree depth (uses config if None)
        """
        if max_depth is None:
            max_depth = self.config.max_depth
        
        # Recursive octree construction
        self._build_octree_recursive(self.root_node, sparse_points, max_depth)
        
        # Update node list
        self._update_node_list()
    
    def _build_octree_recursive(self, 
                               node: OctreeNode, 
                               sparse_points: np.ndarray,
                               max_depth: int):
        """Recursively build octree based on sparse points."""
        # Add sparse points to current node
        points_in_node = []
        for point in sparse_points:
            if node.contains_point(point):
                points_in_node.append(point)
        
        node.sparse_points = points_in_node
        
        # Stop if max depth reached or insufficient points
        if (node.level >= max_depth or 
            len(points_in_node) < self.config.sparse_point_threshold):
            return
        
        # Subdivide and recurse
        children = node.subdivide()
        for child in children:
            self._build_octree_recursive(child, sparse_points, max_depth)
    
    def _update_node_list(self):
        """Update list of all nodes in octree."""
        self.all_nodes = []
        self._collect_nodes_recursive(self.root_node)
    
    def _collect_nodes_recursive(self, node: OctreeNode):
        """Recursively collect all nodes."""
        self.all_nodes.append(node)
        if not node.is_leaf:
            for child in node.children:
                if child is not None:
                    self._collect_nodes_recursive(child)
    
    def find_node_for_sample(self, 
                           position: torch.Tensor,
                           radius: float) -> OctreeNode:
        """
        Find appropriate octree node for sampling based on position and radius.
        
        Args:
            position: [3] 3D position
            radius: sampling radius
            
        Returns:
            Appropriate octree node
        """
        # Convert to numpy for octree traversal
        pos_np = position.detach().cpu().numpy()
        
        # Start from root
        current_node = self.root_node
        
        # Traverse down octree until appropriate level
        while not current_node.is_leaf:
            # Check if we should go deeper based on GSD vs radius
            if current_node.gsd <= radius:
                break
                
            # Find child containing position
            child = current_node.find_containing_child(pos_np)
            if child is None or child.is_pruned:
                break
                
            current_node = child
        
        return current_node
    
    def forward(self,
               rays_o: torch.Tensor,
               rays_d: torch.Tensor, 
               near: float,
               far: float,
               focal_length: float,
               pixel_width: float,
               **kwargs) -> Dict[str, torch.Tensor]:
        """
        Forward pass through InfNeRF.
        
        Args:
            rays_o: [N, 3] ray origins
            rays_d: [N, 3] ray directions
            near: near plane distance
            far: far plane distance  
            focal_length: camera focal length
            pixel_width: pixel width
            
        Returns:
            Rendered outputs
        """
        batch_size = rays_o.shape[0]
        
        # Sample points along rays
        z_vals, pts = self.renderer.sample_rays_lod(
            rays_o, rays_d, near, far, self.config.num_samples
        )
        
        # Calculate sampling radii
        radii = self.renderer.calculate_sample_radius(
            z_vals, focal_length, pixel_width
        )
        
        # Query octree nodes for each sample
        colors = []
        densities = []
        
        for i in range(batch_size):
            ray_colors = []
            ray_densities = []
            
            for j in range(self.config.num_samples):
                sample_pos = pts[i, j]  # [3]
                sample_radius = radii[i, j].item()
                
                # Find appropriate node
                node = self.find_node_for_sample(sample_pos, sample_radius)
                
                # Query NeRF
                with torch.no_grad() if not self.training else torch.enable_grad():
                    result = node.nerf(
                        sample_pos.unsqueeze(0),  # [1, 3]
                        rays_d[i:i+1]            # [1, 3]
                    )
                
                ray_colors.append(result['color'])
                ray_densities.append(result['density'])
            
            colors.append(torch.cat(ray_colors, dim=0))
            densities.append(torch.cat(ray_densities, dim=0))
        
        colors = torch.stack(colors, dim=0)      # [N, num_samples, 3]
        densities = torch.stack(densities, dim=0) # [N, num_samples, 1]
        
        # Volume render
        rendered = self.renderer.volume_render(
            colors, densities, z_vals, rays_d, 
            white_bkgd=kwargs.get('white_bkgd', False)
        )
        
        return rendered
    
    def get_memory_usage(self) -> Dict[str, float]:
        """Get memory usage statistics."""
        total_memory = 0
        memory_by_level = {}
        
        for node in self.all_nodes:
            level = node.level
            node_memory = node.get_memory_size()
            total_memory += node_memory
            
            if level not in memory_by_level:
                memory_by_level[level] = 0
            memory_by_level[level] += node_memory
        
        return {
            'total_mb': total_memory / (1024 * 1024),
            'by_level_mb': {k: v / (1024 * 1024) for k, v in memory_by_level.items()},
            'num_nodes': len(self.all_nodes),
            'max_level': max(node.level for node in self.all_nodes)
        } 