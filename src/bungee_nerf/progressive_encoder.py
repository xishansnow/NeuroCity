"""
Progressive Positional Encoder Module
Implements progressive positional encoding for BungeeNeRF
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, List, Tuple
import numpy as np
import logging

logger = logging.getLogger(__name__)


class ProgressivePositionalEncoder(nn.Module):
    """
    Progressive positional encoder that gradually increases frequency bands
    """
    
    def __init__(
        self,
        num_freqs_base: int = 4,
        num_freqs_max: int = 10,
        include_input: bool = True,
        log_sampling: bool = True,
        input_dim: int = 3
    ):
        super().__init__()
        
        self.num_freqs_base = num_freqs_base
        self.num_freqs_max = num_freqs_max
        self.include_input = include_input
        self.log_sampling = log_sampling
        self.input_dim = input_dim
        self.current_stage = 0
        
        # Generate frequency bands
        if log_sampling:
            freq_bands = 2.0 ** torch.linspace(0.0, num_freqs_max - 1, num_freqs_max)
        else:
            freq_bands = torch.linspace(1.0, 2.0 ** (num_freqs_max - 1), num_freqs_max)
        
        self.register_buffer('freq_bands', freq_bands)
        
        # Calculate output dimensions
        self.output_dim = self._calculate_output_dim()
        
    def _calculate_output_dim(self) -> int:
        """Calculate output dimension"""
        dim = 0
        if self.include_input:
            dim += self.input_dim
        
        # Each frequency band contributes 2 * input_dim (sin and cos)
        dim += 2 * self.num_freqs_max * self.input_dim
        
        return dim
    
    def get_output_dim(self) -> int:
        """Get output dimension"""
        return self.output_dim
    
    def set_current_stage(self, stage: int):
        """Set current training stage"""
        self.current_stage = stage
    
    def get_current_freqs(self) -> int:
        """Get current number of frequency bands"""
        # Progressive increase in frequency bands
        progress = self.current_stage / max(1, 4 - 1)  # Assuming 4 stages
        current_freqs = self.num_freqs_base + int(progress * (self.num_freqs_max - self.num_freqs_base))
        return min(current_freqs, self.num_freqs_max)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply progressive positional encoding
        
        Args:
            x: Input coordinates [N, input_dim]
            
        Returns:
            Encoded coordinates [N, output_dim]
        """
        device = x.device
        batch_size = x.shape[0]
        
        # Get current number of frequency bands
        current_freqs = self.get_current_freqs()
        
        # Start with input coordinates if requested
        if self.include_input:
            encoded = [x]
        else:
            encoded = []
        
        # Apply sinusoidal encoding for current frequency bands
        for i in range(current_freqs):
            freq = self.freq_bands[i]
            
            # Apply frequency to input
            freq_input = x * freq
            
            # Add sin and cos components
            encoded.append(torch.sin(freq_input))
            encoded.append(torch.cos(freq_input))
        
        # Pad with zeros for unused frequency bands
        for i in range(current_freqs, self.num_freqs_max):
            # Add zero padding for unused frequencies
            encoded.append(torch.zeros_like(x))  # sin component
            encoded.append(torch.zeros_like(x))  # cos component
        
        # Concatenate all components
        result = torch.cat(encoded, dim=-1)
        
        return result


class MultiScaleEncoder(nn.Module):
    """
    Multi-scale encoder that handles different levels of detail
    """
    
    def __init__(
        self,
        num_scales: int = 4,
        base_freqs: int = 4,
        max_freqs: int = 10,
        include_input: bool = True
    ):
        super().__init__()
        
        self.num_scales = num_scales
        self.base_freqs = base_freqs
        self.max_freqs = max_freqs
        self.include_input = include_input
        
        # Create encoders for different scales
        self.encoders = nn.ModuleList()
        
        for scale in range(num_scales):
            # Calculate frequency range for this scale
            freqs_for_scale = base_freqs + scale * ((max_freqs - base_freqs) // (num_scales - 1))
            freqs_for_scale = min(freqs_for_scale, max_freqs)
            
            encoder = ProgressivePositionalEncoder(
                num_freqs_base=base_freqs,
                num_freqs_max=freqs_for_scale,
                include_input=include_input
            )
            
            self.encoders.append(encoder)
    
    def get_output_dim(self) -> int:
        """Get output dimension (maximum across all encoders)"""
        if len(self.encoders) == 0:
            return 0
        return max(encoder.get_output_dim() for encoder in self.encoders)
    
    def forward(
        self,
        x: torch.Tensor,
        scale: int = 0,
        distances: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Apply multi-scale encoding
        
        Args:
            x: Input coordinates [N, 3]
            scale: Scale level to use
            distances: Distance to camera for adaptive encoding [N]
            
        Returns:
            Encoded coordinates
        """
        if distances is not None:
            # Adaptive encoding based on distance
            return self._adaptive_encoding(x, distances)
        else:
            # Fixed scale encoding
            scale = min(scale, len(self.encoders) - 1)
            return self.encoders[scale](x)
    
    def _adaptive_encoding(
        self,
        x: torch.Tensor,
        distances: torch.Tensor
    ) -> torch.Tensor:
        """
        Adaptive encoding based on distance to camera
        
        Args:
            x: Input coordinates [N, 3]
            distances: Distance to camera [N]
            
        Returns:
            Adaptively encoded coordinates
        """
        device = x.device
        batch_size = x.shape[0]
        
        # Define distance thresholds for different scales
        # Closer objects get higher frequency encoding
        thresholds = [100.0, 50.0, 25.0, 10.0]
        
        # Use the encoder with maximum output dimension
        max_output_dim = max(encoder.get_output_dim() for encoder in self.encoders)
        encoded = torch.zeros(batch_size, max_output_dim, device=device)
        
        # Apply different encoders based on distance
        for scale, threshold in enumerate(thresholds):
            if scale >= len(self.encoders):
                break
            
            # Find points within this distance threshold
            if scale == 0:
                mask = distances >= threshold
            else:
                prev_threshold = thresholds[scale - 1]
                mask = (distances < prev_threshold) & (distances >= threshold)
            
            if mask.any():
                # Apply encoder for this scale
                scale_encoded = self.encoders[scale](x[mask])
                # Pad if necessary
                if scale_encoded.shape[1] < max_output_dim:
                    padding = torch.zeros(scale_encoded.shape[0], max_output_dim - scale_encoded.shape[1], device=device)
                    scale_encoded = torch.cat([scale_encoded, padding], dim=1)
                encoded[mask] = scale_encoded
        
        # Handle very close objects with highest frequency encoder
        closest_mask = distances < thresholds[-1]
        if closest_mask.any():
            highest_scale = len(self.encoders) - 1
            closest_encoded = self.encoders[highest_scale](x[closest_mask])
            # Pad if necessary
            if closest_encoded.shape[1] < max_output_dim:
                padding = torch.zeros(closest_encoded.shape[0], max_output_dim - closest_encoded.shape[1], device=device)
                closest_encoded = torch.cat([closest_encoded, padding], dim=1)
            encoded[closest_mask] = closest_encoded
        
        return encoded


class FrequencyScheduler:
    """
    Scheduler for progressive frequency activation
    """
    
    def __init__(
        self,
        num_freqs_base: int = 4,
        num_freqs_max: int = 10,
        num_stages: int = 4,
        schedule_type: str = "linear"
    ):
        self.num_freqs_base = num_freqs_base
        self.num_freqs_max = num_freqs_max
        self.num_stages = num_stages
        self.schedule_type = schedule_type
        
        # Create frequency schedule
        self.schedule = self._create_schedule()
    
    def _create_schedule(self) -> List[int]:
        """Create frequency activation schedule"""
        
        if self.schedule_type == "linear":
            # Linear increase in frequency bands
            schedule = []
            for stage in range(self.num_stages):
                progress = stage / max(1, self.num_stages - 1)
                freqs = self.num_freqs_base + int(progress * (self.num_freqs_max - self.num_freqs_base))
                schedule.append(min(freqs, self.num_freqs_max))
        
        elif self.schedule_type == "exponential":
            # Exponential increase in frequency bands
            schedule = []
            for stage in range(self.num_stages):
                progress = stage / max(1, self.num_stages - 1)
                exp_progress = progress ** 2  # Quadratic growth
                freqs = self.num_freqs_base + int(exp_progress * (self.num_freqs_max - self.num_freqs_base))
                schedule.append(min(freqs, self.num_freqs_max))
        
        elif self.schedule_type == "step":
            # Step-wise increase
            step_size = (self.num_freqs_max - self.num_freqs_base) // self.num_stages
            schedule = []
            for stage in range(self.num_stages):
                freqs = self.num_freqs_base + stage * step_size
                schedule.append(min(freqs, self.num_freqs_max))
        
        else:
            # Default to linear
            schedule = self._create_schedule_linear()
        
        return schedule
    
    def get_freqs_for_stage(self, stage: int) -> int:
        """Get number of frequency bands for given stage"""
        stage = min(stage, len(self.schedule) - 1)
        return self.schedule[stage]


class AnisotropicEncoder(nn.Module):
    """
    Anisotropic positional encoder for handling different scales in different directions
    """
    
    def __init__(
        self,
        num_freqs: int = 10,
        anisotropy_factors: List[float] = None,
        include_input: bool = True
    ):
        super().__init__()
        
        self.num_freqs = num_freqs
        self.include_input = include_input
        
        # Default anisotropy factors (x, y, z)
        if anisotropy_factors is None:
            anisotropy_factors = [1.0, 1.0, 0.5]  # Reduce z-axis frequency for aerial views
        
        self.anisotropy_factors = torch.tensor(anisotropy_factors, dtype=torch.float32)
        self.register_buffer('aniso_factors', self.anisotropy_factors)
        
        # Generate frequency bands
        freq_bands = 2.0 ** torch.linspace(0.0, num_freqs - 1, num_freqs)
        self.register_buffer('freq_bands', freq_bands)
        
        # Calculate output dimension
        self.output_dim = (2 * num_freqs + 1) * 3 if include_input else 2 * num_freqs * 3
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply anisotropic positional encoding
        
        Args:
            x: Input coordinates [N, 3]
            
        Returns:
            Encoded coordinates [N, output_dim]
        """
        device = x.device
        
        # Apply anisotropy factors
        x_aniso = x * self.aniso_factors.to(device)
        
        # Start with input if requested
        if self.include_input:
            encoded = [x_aniso]
        else:
            encoded = []
        
        # Apply sinusoidal encoding
        for freq in self.freq_bands:
            freq_input = x_aniso * freq
            encoded.append(torch.sin(freq_input))
            encoded.append(torch.cos(freq_input))
        
        return torch.cat(encoded, dim=-1)


class HierarchicalEncoder(nn.Module):
    """
    Hierarchical encoder that combines multiple scales
    """
    
    def __init__(
        self,
        scales: List[float] = None,
        num_freqs_per_scale: int = 6,
        include_input: bool = True
    ):
        super().__init__()
        
        if scales is None:
            scales = [1.0, 2.0, 4.0, 8.0]
        
        self.scales = scales
        self.num_freqs_per_scale = num_freqs_per_scale
        self.include_input = include_input
        
        # Create encoders for each scale
        self.scale_encoders = nn.ModuleList()
        
        for scale in scales:
            # Adjust frequency bands for this scale
            freq_bands = scale * 2.0 ** torch.linspace(0.0, num_freqs_per_scale - 1, num_freqs_per_scale)
            
            encoder = nn.Module()
            encoder.register_buffer('freq_bands', freq_bands)
            self.scale_encoders.append(encoder)
        
        # Calculate output dimension
        total_freqs = len(scales) * num_freqs_per_scale
        self.output_dim = (2 * total_freqs + 1) * 3 if include_input else 2 * total_freqs * 3
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply hierarchical encoding
        
        Args:
            x: Input coordinates [N, 3]
            
        Returns:
            Encoded coordinates [N, output_dim]
        """
        # Start with input if requested
        if self.include_input:
            encoded = [x]
        else:
            encoded = []
        
        # Apply encoding for each scale
        for encoder in self.scale_encoders:
            for freq in encoder.freq_bands:
                freq_input = x * freq
                encoded.append(torch.sin(freq_input))
                encoded.append(torch.cos(freq_input))
        
        return torch.cat(encoded, dim=-1)
