"""
Block-NeRF Inference Renderer

This module provides inference rendering functionality for Block-NeRF,
tightly coupled with block rasterization for efficient inference.
"""

from __future__ import annotations

import math
from pathlib import Path
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from .core import BlockNeRFConfig, BlockNeRFModel
from .block_rasterizer import BlockRasterizer, BlockRasterizerConfig
from .block_manager import BlockManager

# Type aliases
Tensor = torch.Tensor
TensorDict = dict[str, Tensor]


@dataclass
class BlockNeRFRendererConfig:
    """Configuration for Block-NeRF inference renderer."""

    # Rendering quality
    image_width: int = 800
    image_height: int = 600
    antialiasing: bool = True
    samples_per_pixel: int = 1

    # Performance optimization
    chunk_size: int = 1024
    use_cached_blocks: bool = True
    max_cached_blocks: int = 8

    # Block selection
    visibility_culling: bool = True
    distance_culling: bool = True
    max_render_distance: float = 1000.0

    # Output format
    output_format: str = "rgb"  # "rgb", "depth", "alpha", "all"
    depth_range: tuple[float, float] = (0.1, 1000.0)

    # Background
    background_color: tuple[float, float, float] = (1.0, 1.0, 1.0)
    use_white_background: bool = True

    # Device
    device: str = "cuda"


class BlockNeRFRenderer:
    """
    Block-NeRF Inference Renderer with block rasterization integration.

    This renderer is tightly coupled with the BlockRasterizer for efficient inference.
    """

    def __init__(
        self,
        model_config: BlockNeRFConfig,
        renderer_config: BlockNeRFRendererConfig,
        rasterizer: BlockRasterizer,
        device: Optional[str] = None,
    ):
        self.model_config = model_config
        self.config = renderer_config
        self.device = torch.device(device or renderer_config.device)

        # Block rasterizer (tightly coupled)
        self.rasterizer = rasterizer.to(self.device)

        # Block manager
        self.block_manager = BlockManager(
            scene_bounds=model_config.scene_bounds,
            block_size=model_config.block_size,
            overlap_ratio=model_config.overlap_ratio,
            device=self.device,
        )

        # Block cache for performance
        if renderer_config.use_cached_blocks:
            self.block_cache = {}
            self.cache_access_order = []

    def load_blocks(self, checkpoint_dir: str) -> None:
        """Load Block-NeRF models from checkpoint directory."""
        checkpoint_dir = Path(checkpoint_dir)

        if not checkpoint_dir.exists():
            raise FileNotFoundError(f"Checkpoint directory not found: {checkpoint_dir}")

        # Load block layout
        layout_file = checkpoint_dir / "block_layout.json"
        if layout_file.exists():
            self.block_manager.load_block_layout(str(layout_file))

        # Load individual blocks
        for block_file in checkpoint_dir.glob("block_*.pth"):
            block_name = block_file.stem

            # Create block model
            block = BlockNeRFModel(self.model_config).to(self.device)

            # Load weights
            checkpoint = torch.load(block_file, map_location=self.device)
            if "model_state_dict" in checkpoint:
                block.load_state_dict(checkpoint["model_state_dict"])
            else:
                block.load_state_dict(checkpoint)

            # Set to evaluation mode
            block.eval()

            # Add to block manager
            self.block_manager.blocks[block_name] = block

            print(f"Loaded block: {block_name}")

        print(f"Loaded {len(self.block_manager.blocks)} blocks total")

    def get_camera_rays(
        self,
        camera_pose: Tensor,
        intrinsics: Tensor,
        width: int,
        height: int,
    ) -> tuple[Tensor, Tensor]:
        """Generate camera rays for rendering."""
        device = camera_pose.device

        # Create pixel coordinates
        i, j = torch.meshgrid(
            torch.linspace(0, width - 1, width, device=device),
            torch.linspace(0, height - 1, height, device=device),
            indexing="ij",
        )
        i = i.t()  # Transpose to get correct orientation
        j = j.t()

        # Convert to normalized device coordinates
        dirs = torch.stack(
            [
                (i - intrinsics[0, 2]) / intrinsics[0, 0],
                -(j - intrinsics[1, 2]) / intrinsics[1, 1],  # Negative for correct orientation
                -torch.ones_like(i),
            ],
            dim=-1,
        )

        # Transform ray directions to world coordinates
        ray_directions = torch.sum(dirs[..., None, :] * camera_pose[:3, :3], dim=-1)

        # Normalize directions
        ray_directions = ray_directions / torch.norm(ray_directions, dim=-1, keepdim=True)

        # Ray origins are the camera position
        ray_origins = camera_pose[:3, 3].expand(ray_directions.shape)

        return ray_origins, ray_directions

    def select_visible_blocks(
        self,
        camera_pose: Tensor,
        frustum_bounds: Optional[Tensor] = None,
    ) -> list[str]:
        """Select blocks visible from the camera."""
        camera_position = camera_pose[:3, 3]
        camera_direction = -camera_pose[:3, 2]  # Camera looks in -Z direction

        visible_blocks = self.block_manager.get_blocks_for_camera(
            camera_position,
            camera_direction,
            max_distance=self.config.max_render_distance,
            use_visibility=self.config.visibility_culling,
        )

        return visible_blocks

    def manage_block_cache(self, block_names: list[str]) -> list[BlockNeRFModel]:
        """Manage block cache for efficient memory usage."""
        if not self.config.use_cached_blocks:
            return [self.block_manager.blocks[name] for name in block_names]

        cached_blocks = []

        for block_name in block_names:
            if block_name in self.block_cache:
                # Move to end of access order
                self.cache_access_order.remove(block_name)
                self.cache_access_order.append(block_name)
                cached_blocks.append(self.block_cache[block_name])
            else:
                # Load block into cache
                if len(self.block_cache) >= self.config.max_cached_blocks:
                    # Remove least recently used block
                    lru_block = self.cache_access_order.pop(0)
                    del self.block_cache[lru_block]

                # Add new block to cache
                block = self.block_manager.blocks[block_name]
                self.block_cache[block_name] = block
                self.cache_access_order.append(block_name)
                cached_blocks.append(block)

        return cached_blocks

    def render_image(
        self,
        camera_pose: Tensor,
        intrinsics: Tensor,
        width: Optional[int] = None,
        height: Optional[int] = None,
        appearance_id: int = 0,
        exposure_value: float = 1.0,
    ) -> TensorDict:
        """Render a full image from the given camera pose."""
        # Use config dimensions if not provided
        if width is None:
            width = self.config.image_width
        if height is None:
            height = self.config.image_height

        # Generate camera rays
        ray_origins, ray_directions = self.get_camera_rays(camera_pose, intrinsics, width, height)

        # Flatten rays for processing
        ray_origins_flat = ray_origins.reshape(-1, 3)
        ray_directions_flat = ray_directions.reshape(-1, 3)

        # Select visible blocks
        visible_block_names = self.select_visible_blocks(camera_pose)

        if not visible_block_names:
            # No visible blocks, return background
            rgb = torch.full(
                (height, width, 3),
                self.config.background_color,
                device=self.device,
                dtype=torch.float32,
            )
            depth = torch.zeros((height, width), device=self.device, dtype=torch.float32)
            alpha = torch.zeros((height, width), device=self.device, dtype=torch.float32)

            return {"rgb": rgb, "depth": depth, "alpha": alpha}

        # Load blocks into cache
        visible_blocks = self.manage_block_cache(visible_block_names)

        # Get block centers and radii
        block_centers = [self.block_manager.block_centers[name] for name in visible_block_names]
        block_radii = [self.model_config.block_radius] * len(visible_block_names)

        # Prepare appearance and exposure
        num_rays = ray_origins_flat.shape[0]
        appearance_ids = torch.full(
            (num_rays,), appearance_id, device=self.device, dtype=torch.long
        )
        exposure_values = torch.full(
            (num_rays, 1), exposure_value, device=self.device, dtype=torch.float32
        )

        # Render in chunks
        rgb_chunks = []
        depth_chunks = []
        alpha_chunks = []

        chunk_size = self.config.chunk_size
        for i in range(0, num_rays, chunk_size):
            end_idx = min(i + chunk_size, num_rays)

            chunk_ray_origins = ray_origins_flat[i:end_idx]
            chunk_ray_directions = ray_directions_flat[i:end_idx]
            chunk_appearance_ids = appearance_ids[i:end_idx]
            chunk_exposure_values = exposure_values[i:end_idx]

            # Render chunk using rasterizer
            with torch.no_grad():
                chunk_outputs = self.rasterizer(
                    visible_blocks,
                    chunk_ray_origins,
                    chunk_ray_directions,
                    block_centers,
                    block_radii,
                    chunk_appearance_ids,
                    chunk_exposure_values,
                )

            rgb_chunks.append(chunk_outputs["rgb"])
            depth_chunks.append(chunk_outputs["depth"])
            alpha_chunks.append(chunk_outputs["alpha"])

        # Concatenate chunks
        rgb_flat = torch.cat(rgb_chunks, dim=0)
        depth_flat = torch.cat(depth_chunks, dim=0)
        alpha_flat = torch.cat(alpha_chunks, dim=0)

        # Reshape to image dimensions
        rgb = rgb_flat.reshape(height, width, 3)
        depth = depth_flat.reshape(height, width)
        alpha = alpha_flat.reshape(height, width)

        # Apply background
        if self.config.use_white_background:
            rgb = rgb + (1.0 - alpha.unsqueeze(-1)) * torch.tensor(
                self.config.background_color, device=self.device
            )

        return {
            "rgb": rgb,
            "depth": depth,
            "alpha": alpha,
        }

    def render_video(
        self,
        camera_poses: list[Tensor],
        intrinsics: Tensor,
        output_path: str,
        width: Optional[int] = None,
        height: Optional[int] = None,
        fps: int = 30,
        appearance_id: int = 0,
        exposure_value: float = 1.0,
    ) -> None:
        """Render a video from a sequence of camera poses."""
        try:
            import imageio
        except ImportError:
            raise ImportError(
                "imageio is required for video rendering. Install with: pip install imageio"
            )

        frames = []

        print(f"Rendering {len(camera_poses)} frames...")

        for i, camera_pose in enumerate(camera_poses):
            print(f"Rendering frame {i+1}/{len(camera_poses)}")

            # Render frame
            outputs = self.render_image(
                camera_pose, intrinsics, width, height, appearance_id, exposure_value
            )

            # Convert to numpy and scale to [0, 255]
            rgb_np = outputs["rgb"].cpu().numpy()
            rgb_np = np.clip(rgb_np * 255, 0, 255).astype(np.uint8)

            frames.append(rgb_np)

        # Save video
        imageio.mimsave(output_path, frames, fps=fps)
        print(f"Video saved to {output_path}")

    def set_eval_mode(self) -> None:
        """Set all components to evaluation mode."""
        for block in self.block_manager.blocks.values():
            block.eval()
        self.rasterizer.eval()


def create_block_nerf_renderer(
    model_config: BlockNeRFConfig,
    renderer_config: Optional[BlockNeRFRendererConfig] = None,
    rasterizer_config: Optional[BlockRasterizerConfig] = None,
    device: Optional[str] = None,
) -> BlockNeRFRenderer:
    """Create a Block-NeRF renderer with default configurations."""
    if renderer_config is None:
        renderer_config = BlockNeRFRendererConfig()

    if rasterizer_config is None:
        rasterizer_config = BlockRasterizerConfig()

    # Configure rasterizer with white background from renderer config
    if hasattr(rasterizer_config, "white_background"):
        rasterizer_config.white_background = renderer_config.use_white_background

    rasterizer = BlockRasterizer(rasterizer_config)

    return BlockNeRFRenderer(model_config, renderer_config, rasterizer, device)
