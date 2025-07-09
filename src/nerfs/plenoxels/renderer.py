"""
Plenoxels Renderer - Inference-Only Implementation

This module implements a dedicated renderer class for Plenoxels that focuses
exclusively on inference functionality, following the same pattern as SVRaster.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import numpy as np
import logging
from pathlib import Path
import time

from .config import PlenoxelInferenceConfig
from .core import VoxelGrid, VolumetricRenderer

logger = logging.getLogger(__name__)


class PlenoxelRenderer:
    """Dedicated renderer class for Plenoxels inference.

    This class handles all inference-related functionality including:
    - High-quality ray rendering
    - Batch processing for efficiency
    - Multiple output formats (RGB, depth, normals)
    - Memory-efficient tiled rendering
    - Model loading and saving

    Optimized for inference performance and memory usage.
    """

    def __init__(self, config: PlenoxelInferenceConfig):
        """Initialize the Plenoxel renderer.

        Args:
            config: Inference configuration
        """
        self.config = config
        self.device = config.device

        # Initialize components (will be loaded from checkpoint)
        self.voxel_grid: VoxelGrid | None = None
        self.volumetric_renderer = VolumetricRenderer(config)

        # Performance optimization
        self._setup_inference_optimizations()

        logger.info(f"PlenoxelRenderer initialized with device: {self.device}")

    def _setup_inference_optimizations(self):
        """Setup optimizations for inference."""
        if self.config.optimize_for_inference:
            # Enable inference mode
            torch.set_grad_enabled(False)

            # Setup half precision if requested
            if self.config.use_half_precision and self.device.type == "cuda":
                self.dtype = torch.float16
            else:
                self.dtype = torch.float32

    def load_voxel_grid(self, voxel_grid: VoxelGrid):
        """Load a pre-trained voxel grid.

        Args:
            voxel_grid: Trained voxel grid
        """
        self.voxel_grid = voxel_grid.to(self.device)
        self.voxel_grid.eval()

        # Apply inference optimizations
        if self.config.use_half_precision and self.device.type == "cuda":
            self.voxel_grid = self.voxel_grid.half()

        logger.info(f"Loaded voxel grid with resolution: {self.voxel_grid.resolution}")

    @classmethod
    def from_checkpoint(
        cls, checkpoint_path: str, config: Optional[PlenoxelInferenceConfig] = None
    ) -> PlenoxelRenderer:
        """Load renderer from a training checkpoint.

        Args:
            checkpoint_path: Path to the checkpoint file
            config: Optional inference config (will use default if not provided)

        Returns:
            PlenoxelRenderer: Loaded renderer ready for inference
        """
        checkpoint = torch.load(checkpoint_path, map_location="cpu")

        # Create config if not provided
        if config is None:
            train_config = checkpoint["config"]
            config = PlenoxelInferenceConfig(
                grid_resolution=train_config.grid_resolution,
                scene_bounds=train_config.scene_bounds,
                sh_degree=train_config.sh_degree,
                near_plane=train_config.near_plane,
                far_plane=train_config.far_plane,
                step_size=train_config.step_size,
                sigma_thresh=train_config.sigma_thresh,
                stop_thresh=train_config.stop_thresh,
                num_samples=train_config.num_samples,
                device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
            )

        # Create renderer
        renderer = cls(config)

        # Load voxel grid
        voxel_grid = VoxelGrid(
            resolution=config.grid_resolution,
            scene_bounds=config.scene_bounds,
            num_sh_coeffs=(config.sh_degree + 1) ** 2,
            device=config.device,
        )
        voxel_grid.load_state_dict(checkpoint["voxel_grid_state"])
        renderer.load_voxel_grid(voxel_grid)

        logger.info(f"Renderer loaded from checkpoint: {checkpoint_path}")
        return renderer

    def render_rays(
        self,
        rays_o: torch.Tensor,
        rays_d: torch.Tensor,
        chunk_size: int | None = None,
        return_depth: bool = None,
        return_normals: bool = None,
        return_weights: bool = None,
    ) -> dict[str, torch.Tensor]:
        """Render rays to produce RGB colors and optional additional outputs.

        Args:
            rays_o: Ray origins [..., 3]
            rays_d: Ray directions [..., 3]
            chunk_size: Optional chunk size for batch processing
            return_depth: Whether to return depth (default from config)
            return_normals: Whether to return normals (default from config)
            return_weights: Whether to return weights (default from config)

        Returns:
            Dictionary containing:
                - rgb: Rendered colors [..., 3]
                - depth: Rendered depths [...] (if requested)
                - normals: Rendered normals [..., 3] (if requested)
                - weights: Ray weights [..., num_samples] (if requested)
        """
        if self.voxel_grid is None:
            raise ValueError("No voxel grid loaded. Use load_voxel_grid() or from_checkpoint()")

        # Use config defaults if not specified
        return_depth = return_depth if return_depth is not None else self.config.render_depth
        return_normals = (
            return_normals if return_normals is not None else self.config.render_normals
        )
        return_weights = (
            return_weights if return_weights is not None else self.config.render_weights
        )

        # Default chunk size
        chunk_size = chunk_size or self.config.chunk_size

        # Move to device and convert dtype
        rays_o = rays_o.to(self.device, dtype=self.dtype)
        rays_d = rays_d.to(self.device, dtype=self.dtype)

        # Store original shape
        original_shape = rays_o.shape[:-1]
        rays_o = rays_o.view(-1, 3)
        rays_d = rays_d.view(-1, 3)

        # Render in chunks for memory efficiency
        all_outputs = []

        with torch.no_grad():
            for i in range(0, rays_o.shape[0], chunk_size):
                chunk_rays_o = rays_o[i : i + chunk_size]
                chunk_rays_d = rays_d[i : i + chunk_size]

                # Render chunk
                chunk_outputs = self._render_chunk(
                    chunk_rays_o,
                    chunk_rays_d,
                    return_depth=return_depth,
                    return_normals=return_normals,
                    return_weights=return_weights,
                )

                all_outputs.append(chunk_outputs)

        # Concatenate results
        outputs = {}
        for key in all_outputs[0].keys():
            outputs[key] = torch.cat([chunk[key] for chunk in all_outputs], dim=0)

        # Reshape back to original shape
        for key, value in outputs.items():
            if value.dim() > 1:
                outputs[key] = value.view(*original_shape, *value.shape[1:])
            else:
                outputs[key] = value.view(*original_shape)

        return outputs

    def _render_chunk(
        self,
        rays_o: torch.Tensor,
        rays_d: torch.Tensor,
        return_depth: bool = True,
        return_normals: bool = False,
        return_weights: bool = False,
    ) -> dict[str, torch.Tensor]:
        """Render a chunk of rays."""
        # Determine number of samples
        num_samples = (
            self.config.max_samples_per_ray if self.config.high_quality else self.config.num_samples
        )

        # Use volumetric renderer
        outputs = self.volumetric_renderer.render_rays(
            self.voxel_grid,
            rays_o,
            rays_d,
            num_samples=num_samples,
            near=self.config.near_plane,
            far=self.config.far_plane,
            use_adaptive_sampling=self.config.adaptive_sampling,
        )

        # Filter outputs based on requirements
        result = {"rgb": outputs["rgb"]}

        if return_depth:
            result["depth"] = outputs["depth"]

        if return_normals:
            # Compute normals from gradients if needed
            result["normals"] = self._compute_normals(outputs)

        if return_weights:
            result["weights"] = outputs["weights"]

        return result

    def _compute_normals(self, outputs: dict[str, torch.Tensor]) -> torch.Tensor:
        """Compute surface normals from density gradients."""
        # This is a placeholder - in practice, you'd compute gradients
        # of the density field to get surface normals
        batch_size = outputs["rgb"].shape[0]
        return torch.zeros(batch_size, 3, device=self.device, dtype=self.dtype)

    def render_image(
        self,
        height: int,
        width: int,
        camera_matrix: torch.Tensor,
        camera_pose: torch.Tensor,
        return_depth: bool = None,
        return_normals: bool = None,
        use_tiled_rendering: bool = None,
    ) -> dict[str, np.ndarray]:
        """Render a complete image from a camera viewpoint.

        Args:
            height: Image height in pixels
            width: Image width in pixels
            camera_matrix: Camera intrinsic matrix [3, 3]
            camera_pose: Camera pose matrix [4, 4] (world-to-camera)
            return_depth: Whether to return depth map
            return_normals: Whether to return normal map
            use_tiled_rendering: Whether to use tiled rendering for large images

        Returns:
            Dictionary containing:
                - rgb: RGB image [H, W, 3] as numpy array
                - depth: Depth map [H, W] (if requested)
                - normals: Normal map [H, W, 3] (if requested)
        """
        if self.voxel_grid is None:
            raise ValueError("No voxel grid loaded")

        # Use config defaults
        return_depth = return_depth if return_depth is not None else self.config.render_depth
        return_normals = (
            return_normals if return_normals is not None else self.config.render_normals
        )
        use_tiled_rendering = (
            use_tiled_rendering if use_tiled_rendering is not None else (height * width > 512 * 512)
        )

        if use_tiled_rendering:
            return self._render_image_tiled(
                height, width, camera_matrix, camera_pose, return_depth, return_normals
            )
        else:
            return self._render_image_full(
                height, width, camera_matrix, camera_pose, return_depth, return_normals
            )

    def _render_image_full(
        self,
        height: int,
        width: int,
        camera_matrix: torch.Tensor,
        camera_pose: torch.Tensor,
        return_depth: bool,
        return_normals: bool,
    ) -> dict[str, np.ndarray]:
        """Render complete image at once."""
        # Generate rays
        rays_o, rays_d = self._generate_rays(height, width, camera_matrix, camera_pose)

        # Render rays
        outputs = self.render_rays(
            rays_o,
            rays_d,
            return_depth=return_depth,
            return_normals=return_normals,
        )

        # Convert to numpy and reshape
        result = {}
        result["rgb"] = outputs["rgb"].cpu().numpy().reshape(height, width, 3)

        if return_depth:
            result["depth"] = outputs["depth"].cpu().numpy().reshape(height, width)

        if return_normals:
            result["normals"] = outputs["normals"].cpu().numpy().reshape(height, width, 3)

        return result

    def _render_image_tiled(
        self,
        height: int,
        width: int,
        camera_matrix: torch.Tensor,
        camera_pose: torch.Tensor,
        return_depth: bool,
        return_normals: bool,
    ) -> dict[str, np.ndarray]:
        """Render image using tiles for memory efficiency."""
        tile_size = self.config.tile_size
        overlap = self.config.overlap

        # Initialize output arrays
        rgb_image = np.zeros((height, width, 3), dtype=np.float32)
        depth_image = np.zeros((height, width), dtype=np.float32) if return_depth else None
        normal_image = np.zeros((height, width, 3), dtype=np.float32) if return_normals else None

        # Render tiles
        for y in range(0, height, tile_size - overlap):
            for x in range(0, width, tile_size - overlap):
                # Tile boundaries
                y_end = min(y + tile_size, height)
                x_end = min(x + tile_size, width)

                # Generate rays for tile
                tile_rays_o, tile_rays_d = self._generate_rays_tile(
                    y, x, y_end, x_end, camera_matrix, camera_pose
                )

                # Render tile
                tile_outputs = self.render_rays(
                    tile_rays_o,
                    tile_rays_d,
                    return_depth=return_depth,
                    return_normals=return_normals,
                )

                # Copy to output (handle overlaps by averaging)
                tile_h, tile_w = y_end - y, x_end - x

                # RGB
                tile_rgb = tile_outputs["rgb"].cpu().numpy().reshape(tile_h, tile_w, 3)
                rgb_image[y:y_end, x:x_end] = tile_rgb

                # Depth
                if return_depth:
                    tile_depth = tile_outputs["depth"].cpu().numpy().reshape(tile_h, tile_w)
                    depth_image[y:y_end, x:x_end] = tile_depth

                # Normals
                if return_normals:
                    tile_normals = tile_outputs["normals"].cpu().numpy().reshape(tile_h, tile_w, 3)
                    normal_image[y:y_end, x:x_end] = tile_normals

        # Prepare result
        result = {"rgb": rgb_image}
        if return_depth:
            result["depth"] = depth_image
        if return_normals:
            result["normals"] = normal_image

        return result

    def _generate_rays(
        self,
        height: int,
        width: int,
        camera_matrix: torch.Tensor,
        camera_pose: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Generate rays for the entire image."""
        # Create pixel coordinates
        i, j = torch.meshgrid(
            torch.arange(width, dtype=torch.float32),
            torch.arange(height, dtype=torch.float32),
            indexing="xy",
        )

        return self._generate_rays_from_pixels(i, j, camera_matrix, camera_pose)

    def _generate_rays_tile(
        self,
        y_start: int,
        x_start: int,
        y_end: int,
        x_end: int,
        camera_matrix: torch.Tensor,
        camera_pose: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Generate rays for a tile."""
        # Create pixel coordinates for tile
        i, j = torch.meshgrid(
            torch.arange(x_start, x_end, dtype=torch.float32),
            torch.arange(y_start, y_end, dtype=torch.float32),
            indexing="xy",
        )

        return self._generate_rays_from_pixels(i, j, camera_matrix, camera_pose)

    def _generate_rays_from_pixels(
        self,
        i: torch.Tensor,
        j: torch.Tensor,
        camera_matrix: torch.Tensor,
        camera_pose: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Generate rays from pixel coordinates."""
        device = self.device

        # Move inputs to device
        i = i.to(device)
        j = j.to(device)
        camera_matrix = camera_matrix.to(device)
        camera_pose = camera_pose.to(device)

        # Camera intrinsics
        fx, fy = camera_matrix[0, 0], camera_matrix[1, 1]
        cx, cy = camera_matrix[0, 2], camera_matrix[1, 2]

        # Convert pixels to camera coordinates
        x = (i - cx) / fx
        y = (j - cy) / fy
        z = torch.ones_like(x)

        # Ray directions in camera space
        dirs = torch.stack([x, y, z], dim=-1)  # [H, W, 3]

        # Transform to world space
        # Assuming camera_pose is world-to-camera, we need camera-to-world
        camera_to_world = torch.inverse(camera_pose)

        # Ray origins (camera position in world space)
        rays_o = camera_to_world[:3, 3].expand_as(dirs)

        # Ray directions in world space
        rays_d = torch.sum(dirs[..., None, :] * camera_to_world[:3, :3], dim=-1)
        rays_d = torch.nn.functional.normalize(rays_d, dim=-1)

        return rays_o, rays_d

    def render_video(
        self,
        camera_trajectory: list[tuple[torch.Tensor, torch.Tensor]],
        height: int,
        width: int,
        output_path: str | None = None,
        fps: int = 30,
    ) -> list[np.ndarray]:
        """Render a video from a camera trajectory.

        Args:
            camera_trajectory: List of (camera_matrix, camera_pose) tuples
            height: Video height
            width: Video width
            output_path: Optional path to save video
            fps: Frames per second

        Returns:
            List of rendered frames as numpy arrays
        """
        frames = []

        logger.info(f"Rendering video with {len(camera_trajectory)} frames...")

        for i, (camera_matrix, camera_pose) in enumerate(camera_trajectory):
            logger.info(f"Rendering frame {i+1}/{len(camera_trajectory)}")

            # Render frame
            outputs = self.render_image(height, width, camera_matrix, camera_pose)
            frame = (outputs["rgb"] * 255).astype(np.uint8)
            frames.append(frame)

        # Save video if path provided
        if output_path:
            self._save_video(frames, output_path, fps)

        return frames

    def _save_video(self, frames: list[np.ndarray], output_path: str, fps: int):
        """Save frames as video."""
        try:
            import cv2

            height, width = frames[0].shape[:2]
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

            for frame in frames:
                # Convert RGB to BGR for OpenCV
                frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                out.write(frame_bgr)

            out.release()
            logger.info(f"Video saved to: {output_path}")

        except ImportError:
            logger.warning("OpenCV not available, cannot save video")

    def save_model(self, output_path: str):
        """Save the renderer model for later use.

        Args:
            output_path: Path to save the model
        """
        if self.voxel_grid is None:
            raise ValueError("No voxel grid loaded")

        model_data = {
            "config": self.config,
            "voxel_grid_state": self.voxel_grid.state_dict(),
        }

        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        torch.save(model_data, output_path)
        logger.info(f"Model saved to: {output_path}")

    @classmethod
    def load_model(cls, model_path: str) -> PlenoxelRenderer:
        """Load a saved renderer model.

        Args:
            model_path: Path to the saved model

        Returns:
            PlenoxelRenderer: Loaded renderer
        """
        model_data = torch.load(model_path, map_location="cpu")

        # Create renderer
        renderer = cls(model_data["config"])

        # Load voxel grid
        voxel_grid = VoxelGrid(
            resolution=model_data["config"].grid_resolution,
            scene_bounds=model_data["config"].scene_bounds,
            num_sh_coeffs=(model_data["config"].sh_degree + 1) ** 2,
            device=model_data["config"].device,
        )
        voxel_grid.load_state_dict(model_data["voxel_grid_state"])
        renderer.load_voxel_grid(voxel_grid)

        logger.info(f"Model loaded from: {model_path}")
        return renderer

    def get_model_info(self) -> dict[str, any]:
        """Get information about the loaded model.

        Returns:
            Dictionary containing model information
        """
        if self.voxel_grid is None:
            return {"status": "No model loaded"}

        stats = self.voxel_grid.get_occupancy_stats()

        return {
            "status": "Model loaded",
            "resolution": self.voxel_grid.resolution,
            "sh_degree": self.config.sh_degree,
            "device": str(self.device),
            "dtype": str(self.dtype),
            "occupancy_stats": stats,
        }


def create_plenoxel_renderer(config: PlenoxelInferenceConfig) -> PlenoxelRenderer:
    """Factory function to create a Plenoxel renderer.

    Args:
        config: Inference configuration

    Returns:
        PlenoxelRenderer: Configured renderer instance
    """
    return PlenoxelRenderer(config)
