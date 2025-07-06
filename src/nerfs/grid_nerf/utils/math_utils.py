"""
Grid-NeRF Math Utilities

This module provides mathematical utilities for Grid-NeRF:
- Positional encoding
- Safe tensor normalization
"""

import torch
import torch.nn.functional as F
import numpy as np

def positional_encoding(x: torch.Tensor, L: int = 10) -> torch.Tensor:
    """
    Apply positional encoding to input coordinates.
    
    Args:
        x: Input coordinates [..., D]
        L: Number of frequency bands
        
    Returns:
        Encoded coordinates [..., D * (2 * L + 1)]
    """
    shape = x.shape
    x = x.view(-1, shape[-1])  # [..., D] -> [N, D]
    
    # Original coordinates
    encoded = [x]
    
    # Sinusoidal encoding
    for i in range(L):
        freq = 2.0 ** i
        encoded.append(torch.sin(freq * np.pi * x))
        encoded.append(torch.cos(freq * np.pi * x))
    
    encoded = torch.cat(encoded, dim=-1)  # [N, D * (2 * L + 1)]
    
    # Reshape back to original shape
    new_shape = shape[:-1] + (encoded.shape[-1], )
    return encoded.view(new_shape)

def safe_normalize(x: torch.Tensor, dim: int = -1, eps: float = 1e-8) -> torch.Tensor:
    """
    Safely normalize a tensor along a dimension.
    
    Args:
        x: Input tensor
        dim: Dimension to normalize along
        eps: Small constant for numerical stability
        
    Returns:
        Normalized tensor
    """
    return F.normalize(x, dim=dim, eps=eps) 