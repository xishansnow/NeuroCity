"""
from __future__ import annotations

Instant NGP CUDA-accelerated implementation for GTX 1080 Ti
Compatible with PyTorch without tiny-cuda-nn dependency
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from typing import Optional, Tuple, Dict, Any
import warnings

class HashEncoder(nn.Module):
    """
    Multiresolution hash encoding optimized for GTX 1080 Ti
    """
    
    def __init__(
        self,
        num_levels: int = 16,
        base_resolution: int = 16,
        finest_resolution: int = 512,
        log2_hashmap_size: int = 19,
        feature_dim: int = 2,
        use_cuda: bool = True,
        aabb_min: Optional[torch.Tensor] = None,
        aabb_max: Optional[torch.Tensor] = None
    ):
        super().__init__()
        
        self.num_levels = num_levels
        self.base_resolution = base_resolution
        self.finest_resolution = finest_resolution
        self.log2_hashmap_size = log2_hashmap_size
        self.feature_dim = feature_dim
        self.hashmap_size = 2 ** log2_hashmap_size
        self.use_cuda = use_cuda and torch.cuda.is_available()
        
        # Set up bounding box
        if aabb_min is None:
            aabb_min = torch.tensor([-1.0, -1.0, -1.0])
        if aabb_max is None:
            aabb_max = torch.tensor([1.0, 1.0, 1.0])
        
        self.register_buffer('aabb_min', aabb_min)
        self.register_buffer('aabb_max', aabb_max)
        
        # Calculate level parameters
        self.per_level_scale = 2.0
        self.resolutions = []
        self.offsets = []
        
        total_params = 0
        for level in range(num_levels):
            resolution = int(base_resolution * (self.per_level_scale ** level))
            resolution = min(resolution, finest_resolution)
            self.resolutions.append(resolution)
            
            params_in_level = min(resolution ** 3, self.hashmap_size)
            self.offsets.append(total_params)
            total_params += params_in_level
        
        # Store as tensors
        self.register_buffer('resolutions_tensor', torch.tensor(self.resolutions, dtype=torch.int32))
        self.register_buffer('offsets_tensor', torch.tensor(self.offsets, dtype=torch.int32))
        
        # Initialize hash table parameters
        self.embeddings = nn.Parameter(
            torch.randn(total_params, feature_dim) * 0.0001
        )
        
        # Try to load CUDA extension
        self.cuda_backend = None
        if self.use_cuda:
            try:
                # Add CUDA extension path
                import sys
                import os
                cuda_path = os.path.join(os.path.dirname(__file__), 'cuda')
                if cuda_path not in sys.path:
                    sys.path.insert(0, cuda_path)
                
                import instant_ngp_cuda
                self.cuda_backend = instant_ngp_cuda
                print("✅ Loaded Instant NGP CUDA backend")
            except ImportError:
                print("⚠️  CUDA backend not available, using PyTorch fallback")
                self.use_cuda = False
    
    def forward(self, positions: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through hash encoding
        
        Args:
            positions: [N, 3] tensor of 3D positions
            
        Returns:
            encoded: [N, num_levels * feature_dim] encoded features
        """
        if self.use_cuda and self.cuda_backend is not None:
            return self._forward_cuda(positions)
        else:
            return self._forward_torch(positions)
    
    def _forward_cuda(self, positions: torch.Tensor) -> torch.Tensor:
        """CUDA accelerated forward pass"""
        return self.cuda_backend.hash_encode_forward(
            positions,
            self.embeddings,
            self.resolutions_tensor,
            self.offsets_tensor.to(torch.uint32),
            self.num_levels,
            self.feature_dim,
            self.hashmap_size,
            1.0,  # scale
            self.aabb_min,
            self.aabb_max
        )
    
    def _forward_torch(self, positions: torch.Tensor) -> torch.Tensor:
        """PyTorch fallback implementation"""
        N = positions.shape[0]
        device = positions.device
        
        # Normalize positions to [0, 1]
        pos_normalized = (positions - self.aabb_min) / (self.aabb_max - self.aabb_min)
        pos_normalized = torch.clamp(pos_normalized, 0.0, 1.0)
        
        encoded_features = []
        
        for level in range(self.num_levels):
            resolution = self.resolutions[level]
            offset = self.offsets[level]
            
            # Scale positions to grid resolution
            pos_scaled = pos_normalized * (resolution - 1)
            
            # Get integer coordinates
            pos_grid = torch.floor(pos_scaled).long()
            pos_grid = torch.clamp(pos_grid, 0, resolution - 1)
            
            # Get interpolation weights
            weights = pos_scaled - pos_grid.float()
            
            # Trilinear interpolation
            level_features = torch.zeros(N, self.feature_dim, device=device)
            
            for dx in range(2):
                for dy in range(2):
                    for dz in range(2):
                        # Corner coordinates
                        corner = pos_grid + torch.tensor([dx, dy, dz], device=device)
                        corner = torch.clamp(corner, 0, resolution - 1)
                        
                        # Hash corner coordinates
                        hash_indices = self._spatial_hash(corner, resolution, offset)
                        
                        # Get interpolation weight
                        w = ((1 - dx) * (1 - weights[:, 0:1]) + dx * weights[:, 0:1]) * \
                            ((1 - dy) * (1 - weights[:, 1:2]) + dy * weights[:, 1:2]) * \
                            ((1 - dz) * (1 - weights[:, 2:3]) + dz * weights[:, 2:3])
                        
                        # Add weighted features
                        level_features += w * self.embeddings[hash_indices]
            
            encoded_features.append(level_features)
        
        return torch.cat(encoded_features, dim=1)
    
    def _spatial_hash(self, coords: torch.Tensor, resolution: int, offset: int) -> torch.Tensor:
        """Spatial hash function for coordinate hashing"""
        # Simple hash function
        hash_val = coords[:, 0] * 73856093 + coords[:, 1] * 19349663 + coords[:, 2] * 83492791
        hash_val = hash_val % min(resolution ** 3, self.hashmap_size)
        return offset + hash_val


class SHEncoder(nn.Module):
    """
    Spherical harmonics encoding for view directions
    """
    
    def __init__(self, degree: int = 4, use_cuda: bool = True):
        super().__init__()
        self.degree = degree
        self.output_dim = (degree + 1) ** 2
        self.use_cuda = use_cuda and torch.cuda.is_available()
        
        # Try to load CUDA extension
        self.cuda_backend = None
        if self.use_cuda:
            try:
                import instant_ngp_cuda
                self.cuda_backend = instant_ngp_cuda
            except ImportError:
                self.use_cuda = False
    
    def forward(self, directions: torch.Tensor) -> torch.Tensor:
        """
        Encode directions using spherical harmonics
        
        Args:
            directions: [N, 3] unit direction vectors
            
        Returns:
            encoded: [N, (degree+1)^2] encoded directions
        """
        if self.use_cuda and self.cuda_backend is not None:
            return self.cuda_backend.sh_encode(directions, self.degree)
        else:
            return self._forward_torch(directions)
    
    def _forward_torch(self, directions: torch.Tensor) -> torch.Tensor:
        """PyTorch fallback implementation"""
        x, y, z = directions[:, 0], directions[:, 1], directions[:, 2]
        
        # Normalize directions
        norm = torch.sqrt(x**2 + y**2 + z**2)
        x, y, z = x / norm, y / norm, z / norm
        
        result = []
        
        # l=0
        result.append(torch.ones_like(x) * 0.28209479177387814)  # 1/(2*sqrt(pi))
        
        if self.degree >= 1:
            # l=1
            result.append(-0.48860251190291987 * y)  # -sqrt(3)*y/(2*sqrt(pi))
            result.append(0.48860251190291987 * z)   # sqrt(3)*z/(2*sqrt(pi))
            result.append(-0.48860251190291987 * x)  # -sqrt(3)*x/(2*sqrt(pi))
        
        if self.degree >= 2:
            # l=2
            result.append(1.0925484305920792 * x * y)  # sqrt(15)*x*y/(2*sqrt(pi))
            result.append(-1.0925484305920792 * y * z) # -sqrt(15)*y*z/(2*sqrt(pi))
            result.append(0.31539156525252005 * (2 * z**2 - x**2 - y**2))  # sqrt(5)*(3*z^2-1)/(4*sqrt(pi))
            result.append(-1.0925484305920792 * x * z) # -sqrt(15)*x*z/(2*sqrt(pi))
            result.append(0.5462742152960396 * (x**2 - y**2))  # sqrt(15)*(x^2-y^2)/(4*sqrt(pi))
        
        if self.degree >= 3:
            # l=3 (can be extended)
            result.append(-0.5900435899266435 * y * (3 * x**2 - y**2))
            result.append(2.890611442640554 * x * y * z)
            result.append(-0.4570457994644658 * y * (4 * z**2 - x**2 - y**2))
            result.append(0.3731763325901154 * z * (2 * z**2 - 3 * x**2 - 3 * y**2))
            result.append(-0.4570457994644658 * x * (4 * z**2 - x**2 - y**2))
            result.append(1.445305721320277 * z * (x**2 - y**2))
            result.append(-0.5900435899266435 * x * (x**2 - 3 * y**2))
        
        return torch.stack(result, dim=1)


class InstantNGPModel(nn.Module):
    """
    Complete Instant NGP model optimized for GTX 1080 Ti
    """
    
    def __init__(
        self,
        # Hash encoding parameters
        num_levels: int = 16,
        base_resolution: int = 16,
        finest_resolution: int = 512,
        log2_hashmap_size: int = 19,
        feature_dim: int = 2,
        # Network parameters
        hidden_dim: int = 64,
        num_layers: int = 2,
        geo_feature_dim: int = 15,
        num_layers_color: int = 3,
        hidden_dim_color: int = 64,
        # Spherical harmonics
        sh_degree: int = 4,
        # Other parameters
        use_cuda: bool = True,
        aabb_min: Optional[torch.Tensor] = None,
        aabb_max: Optional[torch.Tensor] = None
    ):
        super().__init__()
        
        # Hash encoder for positions
        self.hash_encoder = HashEncoder(
            num_levels=num_levels,
            base_resolution=base_resolution,
            finest_resolution=finest_resolution,
            log2_hashmap_size=log2_hashmap_size,
            feature_dim=feature_dim,
            use_cuda=use_cuda,
            aabb_min=aabb_min,
            aabb_max=aabb_max
        )
        
        # Spherical harmonics encoder for directions
        self.sh_encoder = SHEncoder(degree=sh_degree, use_cuda=use_cuda)
        
        # Density network
        hash_dim = num_levels * feature_dim
        self.density_net = nn.Sequential()
        
        # Input layer
        self.density_net.add_module('input', nn.Linear(hash_dim, hidden_dim))
        self.density_net.add_module('input_relu', nn.ReLU(inplace=True))
        
        # Hidden layers
        for i in range(num_layers - 1):
            self.density_net.add_module(f'hidden_{i}', nn.Linear(hidden_dim, hidden_dim))
            self.density_net.add_module(f'hidden_{i}_relu', nn.ReLU(inplace=True))
        
        # Output layer
        self.density_net.add_module('output', nn.Linear(hidden_dim, 1 + geo_feature_dim))
        
        # Color network
        sh_dim = (sh_degree + 1) ** 2
        color_input_dim = geo_feature_dim + sh_dim
        
        self.color_net = nn.Sequential()
        
        # Input layer
        self.color_net.add_module('input', nn.Linear(color_input_dim, hidden_dim_color))
        self.color_net.add_module('input_relu', nn.ReLU(inplace=True))
        
        # Hidden layers
        for i in range(num_layers_color - 1):
            self.color_net.add_module(f'hidden_{i}', nn.Linear(hidden_dim_color, hidden_dim_color))
            self.color_net.add_module(f'hidden_{i}_relu', nn.ReLU(inplace=True))
        
        # Output layer
        self.color_net.add_module('output', nn.Linear(hidden_dim_color, 3))
        self.color_net.add_module('output_sigmoid', nn.Sigmoid())
    
    def forward(self, positions: torch.Tensor, directions: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through Instant NGP
        
        Args:
            positions: [N, 3] 3D positions
            directions: [N, 3] view directions
            
        Returns:
            density: [N, 1] density values
            color: [N, 3] RGB colors
        """
        # Encode positions
        pos_encoded = self.hash_encoder(positions)
        
        # Get density and geometry features
        density_output = self.density_net(pos_encoded)
        density = F.softplus(density_output[:, 0:1])
        geo_features = density_output[:, 1:]
        
        # Encode directions
        dir_encoded = self.sh_encoder(directions)
        
        # Get color
        color_input = torch.cat([geo_features, dir_encoded], dim=1)
        color = self.color_net(color_input)
        
        return density, color
    
    def density(self, positions: torch.Tensor) -> torch.Tensor:
        """Get density only (faster for some applications)"""
        pos_encoded = self.hash_encoder(positions)
        density_output = self.density_net(pos_encoded)
        return F.softplus(density_output[:, 0:1])


# Compatibility layer for the existing codebase
class InstantNGP(InstantNGPModel):
    """Alias for backward compatibility"""
    pass


def create_instant_ngp_model(**kwargs) -> InstantNGPModel:
    """Factory function to create Instant NGP model"""
    return InstantNGPModel(**kwargs)
