"""
Regularization utilities for Instant NGP.

This module provides regularization functions including total variation loss,
entropy loss, and other regularization techniques used in neural rendering.
"""

import torch
import torch.nn.functional as F
from typing import Optional


def compute_tv_loss(grid: torch.Tensor, loss_type: str = 'l2') -> torch.Tensor:
    """
    Compute total variation loss for a 3D grid.
    
    Args:
        grid: [B, C, D, H, W] grid tensor
        loss_type: Type of loss ('l1' or 'l2')
        
    Returns:
        Total variation loss
    """
    if grid.dim() != 5:
        raise ValueError(f"Expected 5D tensor, got {grid.dim()}D")
    
    # Compute differences along each axis
    diff_x = grid[:, :, :, :, 1:] - grid[:, :, :, :, :-1]
    diff_y = grid[:, :, :, 1:, :] - grid[:, :, :, :-1, :]
    diff_z = grid[:, :, 1:, :, :] - grid[:, :, :-1, :, :]
    
    if loss_type == 'l1':
        tv_loss = (diff_x.abs().mean() + 
                   diff_y.abs().mean() + 
                   diff_z.abs().mean())
    elif loss_type == 'l2':
        tv_loss = (diff_x.pow(2).mean() + 
                   diff_y.pow(2).mean() + 
                   diff_z.pow(2).mean())
    else:
        raise ValueError(f"Unknown loss type: {loss_type}")
    
    return tv_loss


def compute_entropy_loss(weights: torch.Tensor, 
                        eps: float = 1e-8) -> torch.Tensor:
    """
    Compute entropy loss to encourage weight sparsity.
    
    Args:
        weights: [N, K] weight values (e.g., ray sampling weights)
        eps: Small epsilon for numerical stability
        
    Returns:
        Entropy loss
    """
    # Normalize weights to probabilities
    probs = weights / (weights.sum(dim=-1, keepdim=True) + eps)
    
    # Compute entropy
    log_probs = torch.log(probs + eps)
    entropy = -(probs * log_probs).sum(dim=-1)
    
    # Return negative entropy (to minimize entropy)
    return -entropy.mean()


def compute_distortion_loss(weights: torch.Tensor,
                           z_vals: torch.Tensor) -> torch.Tensor:
    """
    Compute distortion loss to encourage compact weight distributions.
    
    Args:
        weights: [N, K] ray sampling weights  
        z_vals: [N, K] z-values along rays
        
    Returns:
        Distortion loss
    """
    # Normalize weights
    weights_norm = weights / (weights.sum(dim=-1, keepdim=True) + 1e-8)
    
    # Compute intervals
    dists = z_vals[:, 1:] - z_vals[:, :-1]
    dists = torch.cat([dists, dists[:, -1:]], dim=-1)  # Extend last interval
    
    # Compute expected distance
    expected_dist = (weights_norm * z_vals).sum(dim=-1, keepdim=True)
    
    # Compute variance
    variance = (weights_norm * (z_vals - expected_dist).pow(2)).sum(dim=-1)
    
    return variance.mean()


def compute_sparsity_loss(features: torch.Tensor,
                         loss_type: str = 'l1') -> torch.Tensor:
    """
    Compute sparsity loss to encourage sparse feature representations.
    
    Args:
        features: [N, D] feature tensor
        loss_type: Type of sparsity loss ('l1' or 'l2')
        
    Returns:
        Sparsity loss
    """
    if loss_type == 'l1':
        return features.abs().mean()
    elif loss_type == 'l2':
        return features.pow(2).mean()
    else:
        raise ValueError(f"Unknown loss type: {loss_type}")


def compute_smoothness_loss(positions: torch.Tensor,
                           features: torch.Tensor,
                           num_neighbors: int = 8) -> torch.Tensor:
    """
    Compute smoothness loss based on feature similarity of nearby positions.
    
    Args:
        positions: [N, 3] 3D positions
        features: [N, D] feature vectors
        num_neighbors: Number of neighbors to consider
        
    Returns:
        Smoothness loss
    """
    N = positions.shape[0]
    
    if N < num_neighbors:
        return torch.tensor(0.0, device=positions.device)
    
    # Compute pairwise distances
    dists = torch.cdist(positions, positions)  # [N, N]
    
    # Get nearest neighbors (excluding self)
    _, neighbor_indices = torch.topk(dists, num_neighbors + 1, 
                                   dim=-1, largest=False)
    neighbor_indices = neighbor_indices[:, 1:]  # Exclude self
    
    # Get neighbor features
    neighbor_features = features[neighbor_indices]  # [N, K, D]
    
    # Compute feature differences
    feature_diffs = neighbor_features - features.unsqueeze(1)  # [N, K, D]
    
    # Smoothness loss
    smoothness = feature_diffs.pow(2).mean()
    
    return smoothness


def compute_elastic_loss(positions: torch.Tensor,
                        reference_positions: torch.Tensor) -> torch.Tensor:
    """
    Compute elastic regularization loss to maintain spatial relationships.
    
    Args:
        positions: [N, 3] current positions
        reference_positions: [N, 3] reference positions
        
    Returns:
        Elastic loss
    """
    displacement = positions - reference_positions
    elastic_energy = displacement.pow(2).sum(dim=-1).mean()
    
    return elastic_energy


def compute_occupancy_loss(density: torch.Tensor,
                          positions: torch.Tensor,
                          scene_bounds: torch.Tensor) -> torch.Tensor:
    """
    Compute occupancy loss to penalize density outside scene bounds.
    
    Args:
        density: [N, 1] density values
        positions: [N, 3] 3D positions
        scene_bounds: [2, 3] scene bounds [min, max]
        
    Returns:
        Occupancy loss
    """
    bbox_min, bbox_max = scene_bounds[0], scene_bounds[1]
    
    # Check if positions are outside bounds
    outside_mask = ((positions < bbox_min) | (positions > bbox_max)).any(dim=-1)
    
    # Penalize density outside bounds
    outside_density = density[outside_mask]
    
    if len(outside_density) > 0:
        return outside_density.pow(2).mean()
    else:
        return torch.tensor(0.0, device=density.device)


def compute_color_smoothness_loss(colors: torch.Tensor,
                                 positions: torch.Tensor,
                                 num_neighbors: int = 4) -> torch.Tensor:
    """
    Compute color smoothness loss for spatially nearby points.
    
    Args:
        colors: [N, 3] RGB colors
        positions: [N, 3] 3D positions
        num_neighbors: Number of neighbors to consider
        
    Returns:
        Color smoothness loss
    """
    return compute_smoothness_loss(positions, colors, num_neighbors)


def compute_density_smoothness_loss(density: torch.Tensor,
                                   positions: torch.Tensor,
                                   num_neighbors: int = 4) -> torch.Tensor:
    """
    Compute density smoothness loss for spatially nearby points.
    
    Args:
        density: [N, 1] density values
        positions: [N, 3] 3D positions
        num_neighbors: Number of neighbors to consider
        
    Returns:
        Density smoothness loss
    """
    return compute_smoothness_loss(positions, density, num_neighbors)


def apply_gradient_clipping(model: torch.nn.Module,
                           max_norm: float = 1.0) -> float:
    """
    Apply gradient clipping to model parameters.
    
    Args:
        model: PyTorch model
        max_norm: Maximum gradient norm
        
    Returns:
        Total norm of gradients before clipping
    """
    return torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)


def compute_hash_collision_loss(hash_indices: torch.Tensor,
                               table_size: int) -> torch.Tensor:
    """
    Compute hash collision regularization loss.
    
    Args:
        hash_indices: [N] hash table indices
        table_size: Size of hash table
        
    Returns:
        Hash collision loss
    """
    # Count frequency of each hash index
    hist = torch.histc(hash_indices.float(), bins=table_size, 
                      min=0, max=table_size-1)
    
    # Compute variance of histogram (high variance = many collisions)
    collision_loss = hist.var()
    
    return collision_loss


class RegularizationScheduler:
    """Scheduler for regularization weights during training."""
    
    def __init__(self, 
                 initial_weights: dict,
                 schedule_type: str = 'exponential',
                 decay_rate: float = 0.95,
                 decay_steps: int = 1000):
        """
        Initialize regularization scheduler.
        
        Args:
            initial_weights: Dictionary of initial regularization weights
            schedule_type: Type of scheduling ('exponential', 'linear', 'cosine')
            decay_rate: Decay rate for exponential schedule
            decay_steps: Steps between decay
        """
        self.initial_weights = initial_weights
        self.current_weights = initial_weights.copy()
        self.schedule_type = schedule_type
        self.decay_rate = decay_rate
        self.decay_steps = decay_steps
        self.step_count = 0
    
    def step(self):
        """Step the scheduler."""
        self.step_count += 1
        
        if self.step_count % self.decay_steps == 0:
            if self.schedule_type == 'exponential':
                for key in self.current_weights:
                    self.current_weights[key] *= self.decay_rate
            elif self.schedule_type == 'linear':
                decay_factor = 1.0 - (self.step_count / (10 * self.decay_steps))
                decay_factor = max(0.1, decay_factor)  # Minimum weight
                for key in self.current_weights:
                    self.current_weights[key] = self.initial_weights[key] * decay_factor
    
    def get_weights(self) -> dict:
        """Get current regularization weights."""
        return self.current_weights


def test_regularization_functions():
    """Test regularization function implementations."""
    print("Testing regularization functions...")
    
    # Test TV loss
    grid = torch.randn(1, 8, 4, 4, 4)
    tv_loss = compute_tv_loss(grid)
    print(f"TV loss test: {tv_loss.item():.6f}")
    
    # Test entropy loss
    weights = torch.rand(100, 64)
    entropy_loss = compute_entropy_loss(weights)
    print(f"Entropy loss test: {entropy_loss.item():.6f}")
    
    # Test distortion loss
    z_vals = torch.linspace(0, 1, 64).expand(100, 64)
    distortion_loss = compute_distortion_loss(weights, z_vals)
    print(f"Distortion loss test: {distortion_loss.item():.6f}")
    
    # Test sparsity loss
    features = torch.randn(100, 32)
    sparsity_loss = compute_sparsity_loss(features)
    print(f"Sparsity loss test: {sparsity_loss.item():.6f}")
    
    print("Regularization function tests completed!")


if __name__ == "__main__":
    test_regularization_functions() 