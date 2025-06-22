"""
Pyramid Encoder Module
Implements multi-resolution hash encoding for PyNeRF
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, List, Tuple
import numpy as np
import logging

logger = logging.getLogger(__name__)


class HashEncoder(nn.Module):
    """
    Multi-resolution hash encoding based on Instant-NGP
    """
    
    def __init__(
        self,
        resolution: int = 64,
        hash_table_size: int = 2**20,
        features_per_level: int = 4,
        num_levels: int = 1,
        finest_resolution: Optional[int] = None,
        log2_hashmap_size: int = 20
    ):
        super().__init__()
        
        self.resolution = resolution
        self.hash_table_size = hash_table_size
        self.features_per_level = features_per_level
        self.num_levels = num_levels
        self.finest_resolution = finest_resolution or resolution
        self.log2_hashmap_size = log2_hashmap_size
        
        # Create hash tables for each level
        self.hash_tables = nn.ParameterList()
        self.resolutions = []
        
        if num_levels == 1:
            # Single level
            self.resolutions = [resolution]
            hash_table = nn.Parameter(
                torch.randn(hash_table_size, features_per_level) * 0.0001
            )
            self.hash_tables.append(hash_table)
        else:
            # Multi-level
            b = np.exp(np.log(self.finest_resolution / resolution) / (num_levels - 1))
            for level in range(num_levels):
                level_resolution = int(resolution * (b ** level))
                self.resolutions.append(level_resolution)
                
                hash_table = nn.Parameter(
                    torch.randn(hash_table_size, features_per_level) * 0.0001
                )
                self.hash_tables.append(hash_table)
        
        logger.debug(f"HashEncoder initialized with resolutions: {self.resolutions}")
    
    def hash_function(self, coords: torch.Tensor) -> torch.Tensor:
        """
        Hash function for spatial coordinates
        
        Args:
            coords: Integer coordinates [N, 3]
            
        Returns:
            Hash indices [N]
        """
        # Simple hash function using prime numbers
        primes = torch.tensor([1, 2654435761, 805459861], device=coords.device)
        hash_val = torch.sum(coords * primes, dim=-1)
        return hash_val % self.hash_table_size
    
    def interpolate_features(
        self,
        positions: torch.Tensor,
        resolution: int,
        hash_table: torch.Tensor
    ) -> torch.Tensor:
        """
        Trilinear interpolation of hash table features
        
        Args:
            positions: 3D positions [N, 3]
            resolution: Grid resolution
            hash_table: Hash table [hash_size, features]
            
        Returns:
            Interpolated features [N, features]
        """
        # Scale positions to grid resolution
        scaled_pos = positions * resolution
        
        # Get integer coordinates
        coords_floor = torch.floor(scaled_pos).long()
        coords_ceil = coords_floor + 1
        
        # Get fractional parts
        t = scaled_pos - coords_floor.float()
        
        # Clamp coordinates to valid range
        coords_floor = torch.clamp(coords_floor, 0, resolution - 1)
        coords_ceil = torch.clamp(coords_ceil, 0, resolution - 1)
        
        # Get all 8 corner coordinates for trilinear interpolation
        corners = []
        for i in [0, 1]:
            for j in [0, 1]:
                for k in [0, 1]:
                    corner_coords = torch.stack([
                        coords_floor[:, 0] + i,
                        coords_floor[:, 1] + j,
                        coords_floor[:, 2] + k
                    ], dim=-1)
                    corners.append(corner_coords)
        
        # Hash corner coordinates and lookup features
        corner_features = []
        for corner_coords in corners:
            hash_indices = self.hash_function(corner_coords)
            features = hash_table[hash_indices]
            corner_features.append(features)
        
        # Trilinear interpolation
        # Interpolate along x
        c00 = corner_features[0] * (1 - t[:, 0:1]) + corner_features[1] * t[:, 0:1]
        c01 = corner_features[2] * (1 - t[:, 0:1]) + corner_features[3] * t[:, 0:1]
        c10 = corner_features[4] * (1 - t[:, 0:1]) + corner_features[5] * t[:, 0:1]
        c11 = corner_features[6] * (1 - t[:, 0:1]) + corner_features[7] * t[:, 0:1]
        
        # Interpolate along y
        c0 = c00 * (1 - t[:, 1:2]) + c10 * t[:, 1:2]
        c1 = c01 * (1 - t[:, 1:2]) + c11 * t[:, 1:2]
        
        # Interpolate along z
        result = c0 * (1 - t[:, 2:3]) + c1 * t[:, 2:3]
        
        return result
    
    def forward(self, positions: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through hash encoder
        
        Args:
            positions: 3D positions [N, 3] in [0, 1]
            
        Returns:
            Encoded features [N, features_per_level * num_levels]
        """
        # Normalize positions to [0, 1]
        positions = torch.clamp(positions, 0.0, 1.0)
        
        features = []
        for level in range(self.num_levels):
            level_features = self.interpolate_features(
                positions,
                self.resolutions[level],
                self.hash_tables[level]
            )
            features.append(level_features)
        
        # Concatenate features from all levels
        return torch.cat(features, dim=-1)


class PyramidEncoder(nn.Module):
    """
    Pyramid encoder that manages multiple hash encoders for different scales
    """
    
    def __init__(
        self,
        num_levels: int = 8,
        base_resolution: int = 16,
        scale_factor: float = 2.0,
        max_resolution: int = 2048,
        hash_table_size: int = 2**20,
        features_per_level: int = 4
    ):
        super().__init__()
        
        self.num_levels = num_levels
        self.base_resolution = base_resolution
        self.scale_factor = scale_factor
        self.max_resolution = max_resolution
        
        # Create encoders for each pyramid level
        self.encoders = nn.ModuleDict()
        self.resolutions = []
        
        for level in range(num_levels):
            resolution = min(
                base_resolution * (scale_factor ** level),
                max_resolution
            )
            self.resolutions.append(int(resolution))
            
            encoder = HashEncoder(
                resolution=int(resolution),
                hash_table_size=hash_table_size,
                features_per_level=features_per_level,
                num_levels=1
            )
            self.encoders[f"level_{level}"] = encoder
        
        logger.info(f"PyramidEncoder initialized with {num_levels} levels")
        logger.info(f"Resolutions: {self.resolutions}")
    
    def forward(
        self,
        positions: torch.Tensor,
        level: Optional[int] = None
    ) -> torch.Tensor:
        """
        Forward pass through pyramid encoder
        
        Args:
            positions: 3D positions [N, 3]
            level: Specific level to encode (if None, encode all levels)
            
        Returns:
            Encoded features
        """
        if level is not None:
            # Encode specific level
            encoder = self.encoders[f"level_{level}"]
            return encoder(positions)
        else:
            # Encode all levels and concatenate
            features = []
            for level in range(self.num_levels):
                encoder = self.encoders[f"level_{level}"]
                level_features = encoder(positions)
                features.append(level_features)
            return torch.cat(features, dim=-1)
    
    def get_level_features(
        self,
        positions: torch.Tensor,
        level: int
    ) -> torch.Tensor:
        """Get features for a specific pyramid level"""
        encoder = self.encoders[f"level_{level}"]
        return encoder(positions)
    
    def get_output_dim(self, level: Optional[int] = None) -> int:
        """Get output feature dimension"""
        if level is not None:
            return self.encoders[f"level_{level}"].features_per_level
        else:
            return sum(
                encoder.features_per_level 
                for encoder in self.encoders.values()
            )


class PositionalEncoding(nn.Module):
    """
    Positional encoding for high-frequency details
    """
    
    def __init__(
        self,
        num_freqs: int = 10,
        max_freq_log2: int = 9,
        include_input: bool = True,
        log_sampling: bool = True
    ):
        super().__init__()
        
        self.num_freqs = num_freqs
        self.max_freq_log2 = max_freq_log2
        self.include_input = include_input
        self.log_sampling = log_sampling
        
        # Create frequency bands
        if log_sampling:
            freq_bands = 2.0 ** torch.linspace(0.0, max_freq_log2, num_freqs)
        else:
            freq_bands = torch.linspace(2.0**0.0, 2.0**max_freq_log2, num_freqs)
        
        self.register_buffer('freq_bands', freq_bands)
        
        # Calculate output dimension
        self.output_dim = 0
        if include_input:
            self.output_dim += 3
        self.output_dim += 3 * num_freqs * 2  # sin and cos for each frequency
    
    def forward(self, positions: torch.Tensor) -> torch.Tensor:
        """
        Apply positional encoding
        
        Args:
            positions: Input positions [N, 3]
            
        Returns:
            Encoded positions [N, output_dim]
        """
        encoded = []
        
        if self.include_input:
            encoded.append(positions)
        
        for freq in self.freq_bands:
            for func in [torch.sin, torch.cos]:
                encoded.append(func(positions * freq))
        
        return torch.cat(encoded, dim=-1)


class IntegratedPositionalEncoding(nn.Module):
    """
    Integrated positional encoding for anti-aliasing (from Mip-NeRF)
    """
    
    def __init__(
        self,
        num_freqs: int = 10,
        max_freq_log2: int = 9
    ):
        super().__init__()
        
        self.num_freqs = num_freqs
        self.max_freq_log2 = max_freq_log2
        
        # Create frequency bands
        freq_bands = 2.0 ** torch.linspace(0.0, max_freq_log2, num_freqs)
        self.register_buffer('freq_bands', freq_bands)
        
        self.output_dim = 3 * num_freqs * 2  # sin and cos for each frequency
    
    def forward(
        self,
        positions: torch.Tensor,
        covariances: torch.Tensor
    ) -> torch.Tensor:
        """
        Apply integrated positional encoding
        
        Args:
            positions: Mean positions [N, 3]
            covariances: Covariance matrices [N, 3, 3]
            
        Returns:
            Integrated encoded positions [N, output_dim]
        """
        encoded = []
        
        for freq in self.freq_bands:
            # Compute variance for this frequency
            variance = torch.sum(
                covariances * (freq ** 2), 
                dim=(-2, -1)
            )
            
            # Integrated encoding with Gaussian weighting
            for func in [torch.sin, torch.cos]:
                # Apply function to mean position
                encoded_mean = func(positions * freq)
                
                # Apply Gaussian weighting
                gaussian_weight = torch.exp(-0.5 * variance.unsqueeze(-1))
                encoded.append(encoded_mean * gaussian_weight)
        
        return torch.cat(encoded, dim=-1) 