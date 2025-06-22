"""
Grid-NeRF Utilities Module

This module provides utility functions for Grid-NeRF including:
- Image metrics (PSNR, SSIM, LPIPS)
- Visualization and rendering utilities  
- I/O operations
- Logging setup
- Learning rate scheduling
- Mathematical utilities
"""

import os
import cv2
import torch
import torch.nn.functional as F
import numpy as np
import logging
from pathlib import Path
from typing import Optional, Union, Tuple, List, Dict, Any
import matplotlib.pyplot as plt
from PIL import Image
import json
import yaml
from torchvision.utils import save_image as torch_save_image
from torch.optim.lr_scheduler import _LRScheduler


# Image Metrics
def compute_psnr(pred: torch.Tensor, target: torch.Tensor, max_val: float = 1.0) -> float:
    """
    Compute Peak Signal-to-Noise Ratio (PSNR) between predicted and target images.
    
    Args:
        pred: Predicted image tensor [H, W, 3] or [B, H, W, 3]
        target: Target image tensor [H, W, 3] or [B, H, W, 3]  
        max_val: Maximum pixel value (1.0 for normalized images)
        
    Returns:
        PSNR value in dB
    """
    mse = F.mse_loss(pred, target)
    if mse == 0:
        return float('inf')
    return 20 * torch.log10(max_val / torch.sqrt(mse)).item()


def compute_ssim(
    pred: torch.Tensor, 
    target: torch.Tensor,
    window_size: int = 11,
    data_range: float = 1.0
) -> float:
    """
    Compute Structural Similarity Index (SSIM) between predicted and target images.
    
    Args:
        pred: Predicted image tensor [H, W, 3] or [B, H, W, 3]
        target: Target image tensor [H, W, 3] or [B, H, W, 3]
        window_size: Size of the sliding window
        data_range: Dynamic range of the images
        
    Returns:
        SSIM value
    """
    # Ensure images are in [B, C, H, W] format
    if pred.dim() == 3:
        pred = pred.permute(2, 0, 1).unsqueeze(0)  # [H, W, C] -> [1, C, H, W]
        target = target.permute(2, 0, 1).unsqueeze(0)
    elif pred.dim() == 4 and pred.shape[-1] == 3:
        pred = pred.permute(0, 3, 1, 2)  # [B, H, W, C] -> [B, C, H, W]
        target = target.permute(0, 3, 1, 2)
    
    # Convert to grayscale for SSIM computation
    pred_gray = 0.299 * pred[:, 0] + 0.587 * pred[:, 1] + 0.114 * pred[:, 2]
    target_gray = 0.299 * target[:, 0] + 0.587 * target[:, 1] + 0.114 * target[:, 2]
    
    pred_gray = pred_gray.unsqueeze(1)  # [B, 1, H, W]
    target_gray = target_gray.unsqueeze(1)
    
    # SSIM constants
    C1 = (0.01 * data_range) ** 2
    C2 = (0.03 * data_range) ** 2
    
    # Create Gaussian window
    sigma = 1.5
    window = create_gaussian_window(window_size, sigma).to(pred.device)
    window = window.expand(1, 1, window_size, window_size)
    
    mu1 = F.conv2d(pred_gray, window, padding=window_size//2)
    mu2 = F.conv2d(target_gray, window, padding=window_size//2)
    
    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2
    
    sigma1_sq = F.conv2d(pred_gray * pred_gray, window, padding=window_size//2) - mu1_sq
    sigma2_sq = F.conv2d(target_gray * target_gray, window, padding=window_size//2) - mu2_sq
    sigma12 = F.conv2d(pred_gray * target_gray, window, padding=window_size//2) - mu1_mu2
    
    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / \
               ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
    
    return ssim_map.mean().item()


def create_gaussian_window(window_size: int, sigma: float) -> torch.Tensor:
    """Create a 2D Gaussian window for SSIM computation."""
    gauss = torch.Tensor([
        np.exp(-(x - window_size//2)**2 / float(2*sigma**2)) 
        for x in range(window_size)
    ])
    gauss = gauss / gauss.sum()
    window_2d = gauss.unsqueeze(1) @ gauss.unsqueeze(0)
    return window_2d


def compute_lpips(pred: torch.Tensor, target: torch.Tensor, net: str = 'alex') -> float:
    """
    Compute LPIPS (Learned Perceptual Image Patch Similarity) metric.
    Note: This requires the lpips package to be installed.
    
    Args:
        pred: Predicted image tensor [H, W, 3] or [B, H, W, 3]
        target: Target image tensor [H, W, 3] or [B, H, W, 3]
        net: Network to use ('alex', 'vgg', 'squeeze')
        
    Returns:
        LPIPS value
    """
    try:
        import lpips
    except ImportError:
        print("Warning: lpips package not installed. Returning 0.0")
        return 0.0
    
    # Initialize LPIPS model (cache it to avoid reloading)
    if not hasattr(compute_lpips, '_lpips_model'):
        compute_lpips._lpips_model = lpips.LPIPS(net=net)
        if pred.is_cuda:
            compute_lpips._lpips_model = compute_lpips._lpips_model.cuda()
    
    lpips_model = compute_lpips._lpips_model
    
    # Ensure images are in [B, C, H, W] format and in range [-1, 1]
    if pred.dim() == 3:
        pred = pred.permute(2, 0, 1).unsqueeze(0)  # [H, W, C] -> [1, C, H, W]
        target = target.permute(2, 0, 1).unsqueeze(0)
    elif pred.dim() == 4 and pred.shape[-1] == 3:
        pred = pred.permute(0, 3, 1, 2)  # [B, H, W, C] -> [B, C, H, W]
        target = target.permute(0, 3, 1, 2)
    
    # Convert from [0, 1] to [-1, 1]
    pred = pred * 2.0 - 1.0
    target = target * 2.0 - 1.0
    
    with torch.no_grad():
        lpips_val = lpips_model(pred, target)
    
    return lpips_val.mean().item()


# Visualization and I/O
def save_image(
    image: torch.Tensor, 
    path: Union[str, Path], 
    normalize: bool = True,
    quality: int = 95
) -> None:
    """
    Save a tensor image to file.
    
    Args:
        image: Image tensor [H, W, C] or [C, H, W]
        path: Output file path
        normalize: Whether to normalize to [0, 1] range
        quality: JPEG quality (if saving as JPEG)
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    
    # Convert to numpy
    if isinstance(image, torch.Tensor):
        if image.dim() == 3 and image.shape[0] == 3:  # [C, H, W]
            image = image.permute(1, 2, 0)  # [H, W, C]
        image = image.detach().cpu().numpy()
    
    # Normalize if requested
    if normalize:
        image = np.clip(image, 0, 1)
    
    # Convert to 8-bit
    image = (image * 255).astype(np.uint8)
    
    # Save image
    if path.suffix.lower() in ['.jpg', '.jpeg']:
        cv2.imwrite(str(path), cv2.cvtColor(image, cv2.COLOR_RGB2BGR), 
                   [cv2.IMWRITE_JPEG_QUALITY, quality])
    else:
        cv2.imwrite(str(path), cv2.cvtColor(image, cv2.COLOR_RGB2BGR))


def load_image(path: Union[str, Path], target_size: Optional[Tuple[int, int]] = None) -> np.ndarray:
    """
    Load an image from file.
    
    Args:
        path: Image file path
        target_size: Optional target size (H, W)
        
    Returns:
        Image array [H, W, C] in range [0, 1]
    """
    image = cv2.imread(str(path))
    if image is None:
        raise ValueError(f"Could not load image: {path}")
    
    # Convert BGR to RGB
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Resize if requested
    if target_size is not None:
        image = cv2.resize(image, (target_size[1], target_size[0]))
    
    # Normalize to [0, 1]
    return image.astype(np.float32) / 255.0


def create_video_from_images(
    image_dir: Union[str, Path],
    output_path: Union[str, Path],
    fps: int = 30,
    pattern: str = "*.png"
) -> None:
    """
    Create a video from a directory of images.
    
    Args:
        image_dir: Directory containing images
        output_path: Output video file path
        fps: Frames per second
        pattern: Glob pattern for image files
    """
    image_dir = Path(image_dir)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Get sorted list of images
    image_paths = sorted(image_dir.glob(pattern))
    if not image_paths:
        raise ValueError(f"No images found in {image_dir} with pattern {pattern}")
    
    # Read first image to get dimensions
    first_image = cv2.imread(str(image_paths[0]))
    if first_image is None:
        raise ValueError(f"Could not read first image: {image_paths[0]}")
    
    height, width = first_image.shape[:2]
    
    # Create video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
    
    try:
        for image_path in image_paths:
            image = cv2.imread(str(image_path))
            if image is not None:
                writer.write(image)
    finally:
        writer.release()
    
    print(f"Created video: {output_path}")


def create_comparison_grid(
    images: List[torch.Tensor],
    titles: Optional[List[str]] = None,
    output_path: Optional[Union[str, Path]] = None,
    figsize: Tuple[int, int] = (15, 5)
) -> Optional[plt.Figure]:
    """
    Create a comparison grid of images.
    
    Args:
        images: List of image tensors [H, W, C]
        titles: Optional list of titles for each image
        output_path: Optional path to save the figure
        figsize: Figure size
        
    Returns:
        Matplotlib figure if output_path is None
    """
    n_images = len(images)
    fig, axes = plt.subplots(1, n_images, figsize=figsize)
    
    if n_images == 1:
        axes = [axes]
    
    for i, (ax, image) in enumerate(zip(axes, images)):
        # Convert tensor to numpy if needed
        if isinstance(image, torch.Tensor):
            image = image.detach().cpu().numpy()
        
        # Ensure correct shape and range
        if image.shape[-1] != 3:
            image = image.transpose(1, 2, 0)  # [C, H, W] -> [H, W, C]
        image = np.clip(image, 0, 1)
        
        ax.imshow(image)
        ax.axis('off')
        
        if titles and i < len(titles):
            ax.set_title(titles[i])
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        return None
    else:
        return fig


# Logging and Configuration
def setup_logging(log_file: Optional[Union[str, Path]] = None, level: int = logging.INFO) -> None:
    """Setup logging configuration."""
    handlers = [logging.StreamHandler()]
    
    if log_file:
        log_file = Path(log_file)
        log_file.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_file))
    
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=handlers
    )


def load_config(config_path: Union[str, Path]) -> Dict[str, Any]:
    """Load configuration from YAML or JSON file."""
    config_path = Path(config_path)
    
    with open(config_path, 'r') as f:
        if config_path.suffix.lower() in ['.yml', '.yaml']:
            return yaml.safe_load(f)
        elif config_path.suffix.lower() == '.json':
            return json.load(f)
        else:
            raise ValueError(f"Unsupported config file format: {config_path.suffix}")


def save_config(config: Dict[str, Any], output_path: Union[str, Path]) -> None:
    """Save configuration to YAML or JSON file."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        if output_path.suffix.lower() in ['.yml', '.yaml']:
            yaml.dump(config, f, default_flow_style=False)
        elif output_path.suffix.lower() == '.json':
            json.dump(config, f, indent=2)
        else:
            raise ValueError(f"Unsupported config file format: {output_path.suffix}")


# Learning Rate Scheduling
class CosineAnnealingWarmRestarts(_LRScheduler):
    """Cosine Annealing with Warm Restarts scheduler."""
    
    def __init__(
        self,
        optimizer,
        T_0: int,
        T_mult: int = 1,
        eta_min: float = 0,
        last_epoch: int = -1
    ):
        self.T_0 = T_0
        self.T_i = T_0
        self.T_mult = T_mult
        self.eta_min = eta_min
        self.T_cur = last_epoch
        super().__init__(optimizer, last_epoch)
    
    def get_lr(self):
        return [
            self.eta_min + (base_lr - self.eta_min) * 
            (1 + np.cos(np.pi * self.T_cur / self.T_i)) / 2
            for base_lr in self.base_lrs
        ]
    
    def step(self, epoch=None):
        if epoch is None:
            epoch = self.last_epoch + 1
            self.T_cur = self.T_cur + 1
            if self.T_cur >= self.T_i:
                self.T_cur = self.T_cur - self.T_i
                self.T_i = self.T_i * self.T_mult
        else:
            if epoch >= self.T_0:
                if self.T_mult == 1:
                    self.T_cur = epoch % self.T_0
                else:
                    n = int(np.log((epoch / self.T_0 * (self.T_mult - 1) + 1), self.T_mult))
                    self.T_cur = epoch - self.T_0 * (self.T_mult ** n - 1) / (self.T_mult - 1)
                    self.T_i = self.T_0 * self.T_mult ** (n)
            else:
                self.T_i = self.T_0
                self.T_cur = epoch
        
        self.last_epoch = epoch
        for param_group, lr in zip(self.optimizer.param_groups, self.get_lr()):
            param_group['lr'] = lr


def get_learning_rate_scheduler(
    optimizer,
    scheduler_type: str = 'cosine',
    max_steps: int = 100000,
    warmup_steps: int = 1000,
    **kwargs
):
    """Get learning rate scheduler."""
    if scheduler_type == 'cosine':
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=max_steps, **kwargs
        )
    elif scheduler_type == 'cosine_restarts':
        return CosineAnnealingWarmRestarts(
            optimizer, T_0=max_steps // 4, **kwargs
        )
    elif scheduler_type == 'exponential':
        gamma = kwargs.get('gamma', 0.1 ** (1.0 / max_steps))
        return torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=gamma)
    elif scheduler_type == 'step':
        step_size = kwargs.get('step_size', max_steps // 4)
        gamma = kwargs.get('gamma', 0.5)
        return torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)
    elif scheduler_type == 'linear':
        return torch.optim.lr_scheduler.LinearLR(
            optimizer, start_factor=1.0, end_factor=0.01, total_iters=max_steps
        )
    else:
        raise ValueError(f"Unknown scheduler type: {scheduler_type}")


# Mathematical Utilities
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
    new_shape = shape[:-1] + (encoded.shape[-1],)
    return encoded.view(new_shape)


def safe_normalize(x: torch.Tensor, dim: int = -1, eps: float = 1e-8) -> torch.Tensor:
    """Safely normalize a tensor along a dimension."""
    return F.normalize(x, dim=dim, eps=eps)


def get_ray_directions(H: int, W: int, focal: float, center: Optional[Tuple[float, float]] = None) -> torch.Tensor:
    """
    Get ray directions for a pinhole camera.
    
    Args:
        H: Image height
        W: Image width  
        focal: Focal length
        center: Optical center (cx, cy), defaults to image center
        
    Returns:
        Ray directions [H, W, 3]
    """
    if center is None:
        center = (W / 2, H / 2)
    
    cx, cy = center
    
    i, j = torch.meshgrid(
        torch.linspace(0, W-1, W),
        torch.linspace(0, H-1, H),
        indexing='ij'
    )
    i = i.t()  # [H, W]
    j = j.t()  # [H, W]
    
    directions = torch.stack([
        (i - cx) / focal,
        -(j - cy) / focal,  # Negative for right-handed coordinate system
        -torch.ones_like(i)
    ], dim=-1)  # [H, W, 3]
    
    return directions


def sample_along_rays(
    rays_o: torch.Tensor,
    rays_d: torch.Tensor,
    near: float,
    far: float,
    n_samples: int,
    perturb: bool = True
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Sample points along rays.
    
    Args:
        rays_o: Ray origins [N, 3]
        rays_d: Ray directions [N, 3]
        near: Near bound
        far: Far bound
        n_samples: Number of samples per ray
        perturb: Whether to add random perturbation
        
    Returns:
        Sampled points [N, n_samples, 3] and distances [N, n_samples]
    """
    N = rays_o.shape[0]
    device = rays_o.device
    
    # Create sample distances
    t_vals = torch.linspace(0.0, 1.0, n_samples, device=device)
    z_vals = near * (1.0 - t_vals) + far * t_vals  # [n_samples]
    z_vals = z_vals.expand(N, n_samples)  # [N, n_samples]
    
    # Add random perturbation
    if perturb:
        # Get intervals between samples
        mids = 0.5 * (z_vals[..., 1:] + z_vals[..., :-1])
        upper = torch.cat([mids, z_vals[..., -1:]], dim=-1)
        lower = torch.cat([z_vals[..., :1], mids], dim=-1)
        
        # Stratified samples in those intervals
        t_rand = torch.rand(z_vals.shape, device=device)
        z_vals = lower + (upper - lower) * t_rand
    
    # Compute sample points
    points = rays_o[..., None, :] + rays_d[..., None, :] * z_vals[..., :, None]
    
    return points, z_vals


def volume_rendering(
    rgb: torch.Tensor,
    density: torch.Tensor,
    z_vals: torch.Tensor,
    rays_d: torch.Tensor,
    white_background: bool = False
) -> Dict[str, torch.Tensor]:
    """
    Perform volume rendering.
    
    Args:
        rgb: RGB values [N, n_samples, 3]
        density: Density values [N, n_samples]  
        z_vals: Sample distances [N, n_samples]
        rays_d: Ray directions [N, 3]
        white_background: Whether to use white background
        
    Returns:
        Dictionary with rendered RGB, depth, weights, etc.
    """
    # Compute distances between adjacent samples
    dists = z_vals[..., 1:] - z_vals[..., :-1]  # [N, n_samples-1]
    dists = torch.cat([
        dists,
        torch.full_like(dists[..., :1], 1e10)  # Last sample has infinite distance
    ], dim=-1)  # [N, n_samples]
    
    # Multiply by ray direction norm to get real distances
    dists = dists * torch.norm(rays_d[..., None, :], dim=-1)
    
    # Compute alpha values
    alpha = 1.0 - torch.exp(-density * dists)  # [N, n_samples]
    
    # Compute transmittance
    T = torch.cumprod(1.0 - alpha + 1e-10, dim=-1)  # [N, n_samples]
    T = torch.cat([torch.ones_like(T[..., :1]), T[..., :-1]], dim=-1)  # Shift
    
    # Compute weights
    weights = alpha * T  # [N, n_samples]
    
    # Render RGB
    rgb_rendered = torch.sum(weights[..., None] * rgb, dim=-2)  # [N, 3]
    
    # Add background
    if white_background:
        acc_map = torch.sum(weights, dim=-1)  # [N]
        rgb_rendered = rgb_rendered + (1.0 - acc_map[..., None])
    
    # Render depth
    depth = torch.sum(weights * z_vals, dim=-1)  # [N]
    
    # Compute disparity
    disp = 1.0 / torch.max(1e-10 * torch.ones_like(depth), depth / torch.sum(weights, dim=-1))
    
    return {
        'rgb': rgb_rendered,
        'depth': depth,
        'disp': disp,
        'weights': weights,
        'alpha': alpha,
        'transmittance': T
    } 