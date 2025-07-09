"""
from __future__ import annotations

Visualization utilities for Plenoxels.

This module provides functions for visualizing voxel grids, density fields,
and rendering novel view videos.
"""

from typing import Any, Optional, Union, Tuple
import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import imageio
import logging
from tqdm import tqdm

logger = logging.getLogger(__name__)


def visualize_voxel_grid(
    grid: torch.Tensor,
    threshold: float = 0.01,
    output_path: Optional[Union[str, Path]] = None,
    title: str = "Voxel Grid Visualization",
) -> None:
    """Visualize a voxel grid using matplotlib.

    Args:
        grid: Voxel grid tensor of shape [D, H, W] or [D, H, W, C]
        threshold: Threshold for binary visualization
        output_path: Path to save the visualization
        title: Plot title
    """
    # Handle channel dimension
    if grid.dim() == 4:
        # If multi-channel, take mean across channels
        grid = grid.mean(dim=-1)

    # Convert to binary occupancy
    occupancy = (grid > threshold).float()

    # Create 3D plot
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection="3d")

    # Get occupied voxel coordinates
    occupied = torch.nonzero(occupancy)
    if len(occupied) > 0:
        ax.scatter(
            occupied[:, 0].cpu(),
            occupied[:, 1].cpu(),
            occupied[:, 2].cpu(),
            c="b",
            marker="s",
            alpha=0.5,
        )

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title(title)

    if output_path is not None:
        plt.savefig(output_path)
    else:
        plt.show()
    plt.close()


def visualize_density_field(
    density: torch.Tensor,
    scene_bounds: torch.Tensor,
    resolution: tuple[int, int, int] = (64, 64, 64),
    output_path: Optional[Union[str, Path]] = None,
    title: str = "Density Field Visualization",
) -> None:
    """Visualize a density field using volume rendering.

    Args:
        density: Density tensor of shape [D, H, W]
        scene_bounds: Scene bounds tensor [min_x, min_y, min_z, max_x, max_y, max_z]
        resolution: Visualization resolution
        output_path: Path to save the visualization
        title: Plot title
    """
    # Create figure
    fig = plt.figure(figsize=(15, 5))

    # Plot XY, YZ, and XZ slices
    for i, (dim1, dim2, dim3) in enumerate([(0, 1, 2), (1, 2, 0), (0, 2, 1)]):
        ax = fig.add_subplot(1, 3, i + 1)

        # Take middle slice
        slice_idx = density.shape[dim3] // 2
        slice_indices = [slice(None)] * 3
        slice_indices[dim3] = slice_idx
        density_slice = density[slice_indices].cpu()

        # Plot slice
        im = ax.imshow(
            density_slice,
            extent=[
                scene_bounds[dim1].item(),
                scene_bounds[dim1 + 3].item(),
                scene_bounds[dim2].item(),
                scene_bounds[dim2 + 3].item(),
            ],
            cmap="viridis",
        )
        plt.colorbar(im, ax=ax)
        ax.set_title(f"{'XYZ'[dim1]}{'XYZ'[dim2]} Slice")
        ax.set_xlabel(f"{'XYZ'[dim1]}")
        ax.set_ylabel(f"{'XYZ'[dim2]}")

    plt.suptitle(title)
    if output_path is not None:
        plt.savefig(output_path)
    else:
        plt.show()
    plt.close()


def create_occupancy_visualization(
    grid: torch.Tensor,
    threshold: float = 0.01,
    output_path: Optional[Union[str, Path]] = None,
) -> np.ndarray:
    """Create a visualization of grid occupancy.

    Args:
        grid: Voxel grid tensor
        threshold: Threshold for binary visualization
        output_path: Path to save the visualization

    Returns:
        Visualization image as numpy array
    """
    # Convert to binary occupancy
    occupancy = (grid > threshold).float()

    # Create RGB visualization
    vis = torch.zeros(*grid.shape[:3], 3, device=grid.device)
    vis[occupancy > 0] = torch.tensor([0.0, 0.0, 1.0], device=grid.device)  # Blue for occupied
    vis = (vis * 255).byte().cpu().numpy()

    if output_path is not None:
        imageio.imwrite(output_path, vis)

    return vis


def render_novel_view_video(
    model: torch.nn.Module,
    poses: torch.Tensor,
    H: int,
    W: int,
    focal: float,
    output_path: Union[str, Path],
    fps: int = 30,
    chunk_size: int = 4096,
) -> None:
    """Render a video of novel views.

    Args:
        model: Trained Plenoxel model
        poses: Camera poses of shape [N, 3, 4]
        H: Image height
        W: Image width
        focal: Focal length
        output_path: Path to save the video
        fps: Video frames per second
        chunk_size: Maximum rays to process at once
    """
    output_path = Path(output_path)
    if not output_path.parent.exists():
        output_path.parent.mkdir(parents=True)

    frames = []
    model.eval()

    with torch.no_grad():
        for pose in tqdm(poses, desc="Rendering frames"):
            # Generate rays
            i, j = torch.meshgrid(
                torch.linspace(0, W - 1, W), torch.linspace(0, H - 1, H), indexing="xy"
            )
            i = i.to(model.device)
            j = j.to(model.device)

            # Convert to camera coordinates
            dirs = torch.stack(
                [(i - W / 2) / focal, -(j - H / 2) / focal, -torch.ones_like(i)], dim=-1
            )

            # Transform to world coordinates
            rays_d = torch.sum(dirs[..., None, :] * pose[:3, :3], dim=-1)
            rays_o = pose[:3, -1].expand(rays_d.shape)

            # Render in chunks
            rgb_chunks = []
            for i in range(0, rays_o.shape[0], chunk_size):
                chunk_rays_o = rays_o[i : i + chunk_size]
                chunk_rays_d = rays_d[i : i + chunk_size]
                outputs = model(chunk_rays_o, chunk_rays_d)
                rgb_chunks.append(outputs["rgb"].cpu())

            # Combine chunks
            rgb = torch.cat(rgb_chunks, dim=0)
            rgb = rgb.reshape(H, W, 3)

            # Convert to image
            img = (rgb * 255).byte().numpy()
            frames.append(img)

    # Save video
    imageio.mimsave(output_path, frames, fps=fps)
