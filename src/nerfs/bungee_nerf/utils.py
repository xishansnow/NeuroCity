from typing import Any, Optional, Union
"""
Utility functions for BungeeNeRF
"""

import torch
import torch.nn.functional as F
import numpy as np
import logging
import os
import json

logger = logging.getLogger(__name__)

def compute_scale_factor(
    distances: torch.Tensor, scale_thresholds: list[float]
) -> torch.Tensor:
    """
    Compute scale factors based on distance
    
    Args:
        distances: Distance to camera [N]
        scale_thresholds: Distance thresholds for different scales
        
    Returns:
        Scale factors [N]
    """
    device = distances.device
    scale_factors = torch.ones_like(distances)
    
    for i, threshold in enumerate(scale_thresholds):
        mask = distances < threshold
        # Higher scale factor for closer objects
        scale_factors[mask] = 2.0 ** (len(scale_thresholds) - i)
    
    return scale_factors

def get_level_of_detail(
    distances: torch.Tensor, lod_thresholds: list[float]
) -> torch.Tensor:
    """
    Get level of detail based on distance
    
    Args:
        distances: Distance to camera [N]
        lod_thresholds: Distance thresholds for LOD levels
        
    Returns:
        LOD levels [N]
    """
    device = distances.device
    lod_levels = torch.zeros_like(distances, dtype=torch.long)
    
    for level, threshold in enumerate(lod_thresholds):
        mask = distances < threshold
        lod_levels[mask] = level + 1
    
    # Clamp to valid range
    lod_levels = torch.clamp(lod_levels, 0, len(lod_thresholds))
    
    return lod_levels

def progressive_positional_encoding(
    x: torch.Tensor, num_freqs: int, stage: int, max_stages: int, include_input: bool = True
) -> torch.Tensor:
    """
    Apply progressive positional encoding
    
    Args:
        x: Input coordinates [N, 3]
        num_freqs: Number of frequency bands
        stage: Current training stage
        max_stages: Maximum number of stages
        include_input: Whether to include input coordinates
        
    Returns:
        Encoded coordinates [N, encoded_dim]
    """
    device = x.device
    
    # Calculate current number of active frequencies
    progress = stage / max(1, max_stages - 1)
    active_freqs = int(progress * num_freqs)
    active_freqs = min(active_freqs, num_freqs)
    
    # Start with input if requested
    if include_input:
        encoded = [x]
    else:
        encoded = []
    
    # Apply sinusoidal encoding
    for i in range(num_freqs):
        freq = 2.0 ** i
        
        if i < active_freqs:
            # Active frequency
            encoded.append(torch.sin(x * freq))
            encoded.append(torch.cos(x * freq))
        else:
            # Inactive frequency (zero padding)
            encoded.append(torch.zeros_like(x))
            encoded.append(torch.zeros_like(x))
    
    return torch.cat(encoded, dim=-1)

def multiscale_sampling(
    rays_o: torch.Tensor, rays_d: torch.Tensor, bounds: torch.Tensor, distances: torch.Tensor, num_samples_base: int = 64, scale_factors: list[float] = None
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Multi-scale sampling along rays
    
    Args:
        rays_o: Ray origins [N, 3]
        rays_d: Ray directions [N, 3]
        bounds: Near/far bounds [N, 2]
        distances: Distance to camera [N]
        num_samples_base: Base number of samples
        scale_factors: Scale factors for different distances
        
    Returns:
        tuple of (sample_points, z_vals)
    """
    device = rays_o.device
    batch_size = rays_o.shape[0]
    
    if scale_factors is None:
        scale_factors = [1.0, 1.5, 2.0, 3.0]
    
    # Determine sampling density based on distance
    lod_levels = get_level_of_detail(distances, [100.0, 50.0, 25.0, 10.0])
    
    # Calculate number of samples for each ray
    num_samples_per_ray = torch.full(
    )
    
    for level, scale_factor in enumerate(scale_factors):
        mask = lod_levels == level
        if mask.any():
            num_samples_per_ray[mask] = int(num_samples_base * scale_factor)
    
    # Maximum number of samples
    max_samples = num_samples_per_ray.max().item()
    
    # Initialize output tensors
    all_z_vals = torch.zeros(batch_size, max_samples, device=device)
    all_points = torch.zeros(batch_size, max_samples, 3, device=device)
    
    # Sample for each ray
    for i in range(batch_size):
        num_samples = num_samples_per_ray[i].item()
        
        # Sample depths
        near = bounds[i, 0]
        far = bounds[i, 1]
        
        t_vals = torch.linspace(0.0, 1.0, num_samples, device=device)
        z_vals = near * (1.0 - t_vals) + far * t_vals
        
        # Calculate sample points
        points = rays_o[i:i+1, None, :] + rays_d[i:i+1, None, :] * z_vals[:, None]
        
        # Store in output tensors
        all_z_vals[i, :num_samples] = z_vals
        all_points[i, :num_samples] = points.squeeze(0)
    
    return all_points, all_z_vals

def compute_multiscale_loss(
    outputs: dict[str, torch.Tensor], targets: dict[str, torch.Tensor], distances: torch.Tensor, stage: int, config
) -> dict[str, torch.Tensor]:
    """
    Compute multi-scale loss for BungeeNeRF
    
    Args:
        outputs: Model outputs
        targets: Ground truth targets
        distances: Distance to camera
        stage: Current training stage
        config: Model configuration
        
    Returns:
        Dictionary of loss values
    """
    losses = {}
    device = outputs["rgb"].device
    
    # Color loss
    color_loss = F.mse_loss(outputs["rgb"], targets["rgb"])
    losses["color_loss"] = color_loss
    
    # Distance-weighted loss
    # Give more weight to closer objects
    distance_weights = 1.0 / (distances + 1.0)
    distance_weights = distance_weights / distance_weights.mean()
    
    weighted_color_loss = (
        distance_weights * F.mse_loss,
    )
    losses["weighted_color_loss"] = weighted_color_loss
    
    # Progressive loss (encourage smooth transitions between stages)
    if stage > 0:
        progressive_loss = torch.tensor(0.0, device=device)
        
        # Regularization on weights
        if "weights" in outputs:
            weights = outputs["weights"]
            # Encourage smooth weight distributions
            weight_smoothness = torch.var(weights, dim=-1).mean()
            progressive_loss = progressive_loss + weight_smoothness
        
        losses["progressive_loss"] = progressive_loss
    else:
        losses["progressive_loss"] = torch.tensor(0.0, device=device)
    
    # Depth loss (if available)
    if "depth" in targets and "depth" in outputs:
        depth_loss = F.smooth_l1_loss(outputs["depth"], targets["depth"])
        losses["depth_loss"] = depth_loss
    else:
        losses["depth_loss"] = torch.tensor(0.0, device=device)
    
    # Total loss
    total_loss = (
        config.color_loss_weight * losses["color_loss"] +
        0.5 * losses["weighted_color_loss"] +
        config.depth_loss_weight * losses["depth_loss"] +
        config.progressive_loss_weight * losses["progressive_loss"]
    )
    
    losses["total_loss"] = total_loss
    
    return losses

def save_bungee_model(
    model: torch.nn.Module, config: dict, save_path: str, stage: Optional[int] = None, epoch: Optional[int] = None, optimizer_state: Optional[dict] = None
) -> None:
    """
    Save BungeeNeRF model and configuration
    
    Args:
        model: BungeeNeRF model
        config: Model configuration
        save_path: Path to save model
        stage: Current training stage
        epoch: Current epoch
        optimizer_state: Optimizer state dict
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    checkpoint = {
        'model_state_dict': model.state_dict(
        )
    }
    
    if stage is not None:
        checkpoint['stage'] = stage
    
    if epoch is not None:
        checkpoint['epoch'] = epoch
    
    if optimizer_state is not None:
        checkpoint['optimizer_state_dict'] = optimizer_state
    
    torch.save(checkpoint, save_path)
    logger.info(f"Model saved to {save_path}")

def load_bungee_model(
    model: torch.nn.Module, load_path: str, device: str = "cuda"
) -> tuple[torch.nn.Module, dict]:
    """
    Load BungeeNeRF model and configuration
    
    Args:
        model: BungeeNeRF model instance
        load_path: Path to load model from
        device: Device to load model on
        
    Returns:
        tuple of (loaded_model, config)
    """
    checkpoint = torch.load(load_path, map_location=device)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    config = checkpoint.get('config', {})
    
    # Restore progressive state if available
    if hasattr(model, 'set_current_stage') and 'stage' in checkpoint:
        model.set_current_stage(checkpoint['stage'])
    
    logger.info(f"Model loaded from {load_path}")
    
    return model, config

def compute_psnr(
    img_pred: torch.Tensor, img_gt: torch.Tensor, mask: Optional[torch.Tensor] = None
) -> float:
    """
    Compute Peak Signal-to-Noise Ratio (PSNR)
    
    Args:
        img_pred: Predicted image [H, W, 3]
        img_gt: Ground truth image [H, W, 3]
        mask: Optional mask [H, W]
        
    Returns:
        PSNR value
    """
    if mask is not None:
        img_pred = img_pred[mask]
        img_gt = img_gt[mask]
    
    mse = torch.mean((img_pred - img_gt) ** 2)
    
    if mse == 0:
        return float('inf')
    
    psnr = 20 * torch.log10(1.0 / torch.sqrt(mse))
    return psnr.item()

def compute_ssim(
    img_pred: torch.Tensor, img_gt: torch.Tensor, window_size: int = 11, sigma: float = 1.5
) -> float:
    """
    Compute Structural Similarity Index (SSIM)
    
    Args:
        img_pred: Predicted image [H, W, 3]
        img_gt: Ground truth image [H, W, 3]
        window_size: Window size for SSIM computation
        sigma: Gaussian sigma for window
        
    Returns:
        SSIM value
    """
    # Convert to grayscale if needed
    if img_pred.shape[-1] == 3:
        img_pred = torch.mean(img_pred, dim=-1, keepdim=True)
        img_gt = torch.mean(img_gt, dim=-1, keepdim=True)
    
    # Add batch and channel dimensions
    img_pred = img_pred.permute(2, 0, 1).unsqueeze(0)
    img_gt = img_gt.permute(2, 0, 1).unsqueeze(0)
    
    # Create Gaussian window
    coords = torch.arange(window_size, dtype=torch.float32, device=img_pred.device)
    coords -= window_size // 2
    g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    g /= g.sum()
    
    window = g.outer(g).unsqueeze(0).unsqueeze(0)
    
    # Constants for SSIM
    C1 = (0.01) ** 2
    C2 = (0.03) ** 2
    
    # Compute means
    mu1 = F.conv2d(img_pred, window, padding=window_size//2)
    mu2 = F.conv2d(img_gt, window, padding=window_size//2)
    
    mu1_sq = mu1 ** 2
    mu2_sq = mu2 ** 2
    mu1_mu2 = mu1 * mu2
    
    # Compute variances and covariance
    sigma1_sq = F.conv2d(img_pred ** 2, window, padding=window_size//2) - mu1_sq
    sigma2_sq = F.conv2d(img_gt ** 2, window, padding=window_size//2) - mu2_sq
    sigma12 = F.conv2d(img_pred * img_gt, window, padding=window_size//2) - mu1_mu2
    
    # Compute SSIM
    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / \
               ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
    
    return ssim_map.mean().item()

def create_progressive_schedule(
    num_stages: int = 4, steps_per_stage: int = 50000, warmup_steps: int = 1000
) -> dict[str, any]:
    """
    Create progressive training schedule
    
    Args:
        num_stages: Number of progressive stages
        steps_per_stage: Training steps per stage
        warmup_steps: Warmup steps for each stage
        
    Returns:
        Progressive training schedule
    """
    schedule = {
        "num_stages": num_stages,
        "steps_per_stage": steps_per_stage,
        "warmup_steps": warmup_steps,
        "total_steps": num_stages * steps_per_stage,
        "stage_transitions": [i * steps_per_stage for i in range(1, num_stages)]
    }
    
    return schedule

def apply_progressive_schedule(
    step: int, schedule: dict[str, any]
) -> int:
    """
    Get current stage based on training step
    
    Args:
        step: Current training step
        schedule: Progressive training schedule
        
    Returns:
        Current stage
    """
    stage = 0
    for transition_step in schedule["stage_transitions"]:
        if step >= transition_step:
            stage += 1
        else:
            break
    
    return min(stage, schedule["num_stages"] - 1)

def convert_ges_to_nerf_poses(
    ges_file: str, output_file: str, coordinate_system: str = "ENU"
) -> None:
    """
    Convert Google Earth Studio poses to NeRF format
    
    Args:
        ges_file: Path to GES JSON file
        output_file: Path to output NeRF poses
        coordinate_system: Target coordinate system
    """
    with open(ges_file, 'r') as f:
        ges_data = json.load(f)
    
    camera_frames = ges_data["cameraFrames"]
    poses = []
    
    for frame in camera_frames:
        # Convert GES pose to NeRF format
        position = frame["position"]
        rotation = frame["rotation"]
        
        # Create transformation matrix
        pose = np.eye(4, dtype=np.float32)
        
        # Convert rotation (apply coordinate system conversion)
        euler_angles = np.array([
            -rotation["x"], 180 - rotation["y"], 180 + rotation["z"]
        ], dtype=np.float32)
        
        # Convert to radians and create rotation matrix
        from scipy.spatial.transform import Rotation
        r = Rotation.from_euler('xyz', np.radians(euler_angles))
        pose[:3, :3] = r.as_matrix()
        
        # Set position
        pose[:3, 3] = [position["x"], position["y"], position["z"]]
        
        poses.append(pose.tolist())
    
    # Save poses
    output_data = {
        "camera_angle_x": np.radians(ges_data.get("fovVertical", 30.0)),
        "frames": [
            {
                "transform_matrix": pose,
            }
        ]
    }
    
    with open(output_file, 'w') as f:
        json.dump(output_data, f, indent=2)
    
    logger.info(f"Converted {len(poses)} poses from GES to NeRF format")

def compute_scene_bounds(
    poses: np.ndarray, percentile: float = 95.0
) -> tuple[float, float]:
    """
    Compute scene bounds from camera poses
    
    Args:
        poses: Camera poses [N, 4, 4]
        percentile: Percentile for robust bound estimation
        
    Returns:
        tuple of (near, far) bounds
    """
    # Get camera positions
    positions = poses[:, :3, 3]
    
    # Compute distances from scene center
    scene_center = positions.mean(axis=0)
    distances = np.linalg.norm(positions - scene_center, axis=1)
    
    # Robust bound estimation
    near = np.percentile(distances, 100 - percentile) * 0.1
    far = np.percentile(distances, percentile) * 2.0
    
    # Ensure valid bounds
    near = max(near, 0.01)
    far = max(far, near * 2.0)
    
    return near, far

def visualize_multiscale_data(
    dataset, save_path: str, num_samples: int = 100
) -> None:
    """
    Visualize multi-scale data distribution
    
    Args:
        dataset: BungeeNeRF dataset
        save_path: Path to save visualization
        num_samples: Number of samples to visualize
    """
    import matplotlib.pyplot as plt
    
    # Sample data
    indices = np.random.choice(len(dataset), min(num_samples, len(dataset)), replace=False)
    distances = [dataset.get_distance(i) for i in indices]
    
    # Create histogram
    plt.figure(figsize=(10, 6))
    plt.hist(distances, bins=20, alpha=0.7, edgecolor='black')
    plt.xlabel('Distance to Camera')
    plt.ylabel('Number of Images')
    plt.title('Multi-scale Data Distribution')
    plt.grid(True, alpha=0.3)
    
    # Add scale threshold lines
    if hasattr(dataset, 'scale_thresholds'):
        for i, threshold in enumerate(dataset.scale_thresholds):
            plt.axvline(
                threshold,
                color='red',
                linestyle='--',
                alpha=0.7,
                label=f'Scale {i} threshold',
            )
    
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Multi-scale data visualization saved to {save_path}")
