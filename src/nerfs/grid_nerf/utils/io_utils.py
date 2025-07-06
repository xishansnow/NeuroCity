"""
from __future__ import annotations

Grid-NeRF I/O Utilities

This module provides I/O utilities for Grid-NeRF:
- Image saving and loading
- Video creation from image sequences
- Visualization utilities
"""

import cv2
import numpy as np
import torch
import matplotlib.pyplot as plt
from pathlib import Path
from collections.abc import Sequence
from typing import Tuple


def save_image(
    image: torch.Tensor, path: str | Path, normalize: bool = True, quality: int = 95
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
        cv2.imwrite(str(path), cv2.cvtColor(image, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, quality])
    else:
        cv2.imwrite(str(path), cv2.cvtColor(image, cv2.COLOR_RGB2BGR))

def load_image(path: str | Path, target_size: Tuple[int, int] | None = None) -> np.ndarray:
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
    image_dir: str | Path, output_path: str | Path, fps: int = 30, pattern: str = "*.png"
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
    images: Sequence[torch.Tensor], titles: Sequence[str] | None = None, output_path: str | Path | None = None, figsize: Tuple[int, int] = (15, 5)
) -> plt.Figure | None:
    """
    Create a comparison grid of images.
    
    Args:
        images: list of image tensors [H, W, C]
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