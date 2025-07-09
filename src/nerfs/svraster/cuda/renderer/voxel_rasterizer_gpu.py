"""
CUDA-accelerated VoxelRasterizer implementation

This module provides a GPU-optimized version of the VoxelRasterizer
using CUDA kernels for high-performance voxel-based rendering.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
import numpy as np
from typing import Optional, Dict, Any, Tuple
import logging
import time

logger = logging.getLogger(__name__)

# Try to import CUDA extension
try:
    import voxel_rasterizer_cuda

    # Check if required functions are available
    if hasattr(voxel_rasterizer_cuda, "voxel_rasterization"):
        CUDA_AVAILABLE = True
        logger.info("VoxelRasterizer CUDA extension loaded successfully")
    else:
        raise AttributeError("CUDA extension missing required functions")
except (ImportError, AttributeError):
    CUDA_AVAILABLE = False
    voxel_rasterizer_cuda = None
    logger.warning(
        "VoxelRasterizer CUDA extension not available, falling back to CPU implementation"
    )


# Export the functions for external use
def get_voxel_rasterization_function():
    """Get the voxel_rasterization function if available"""
    if CUDA_AVAILABLE and voxel_rasterizer_cuda is not None:
        return voxel_rasterizer_cuda.voxel_rasterization
    return None


def get_create_camera_matrix_function():
    """Get the create_camera_matrix function if available"""
    if CUDA_AVAILABLE and voxel_rasterizer_cuda is not None:
        return voxel_rasterizer_cuda.create_camera_matrix
    return None


def get_rays_to_camera_matrix_function():
    """Get the rays_to_camera_matrix function if available"""
    if CUDA_AVAILABLE and voxel_rasterizer_cuda is not None:
        return voxel_rasterizer_cuda.rays_to_camera_matrix
    return None


def get_benchmark_function():
    """Get the benchmark function if available"""
    if CUDA_AVAILABLE and voxel_rasterizer_cuda is not None:
        return voxel_rasterizer_cuda.benchmark
    return None


class VoxelRasterizerGPU:
    """
    CUDA-accelerated voxel rasterizer for SVRaster

    This class provides high-performance voxel-based rendering using CUDA kernels.
    It implements the same interface as the CPU version but with significant
    performance improvements for large voxel grids.
    """

    def __init__(self, config):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Performance tracking
        self.performance_stats = {
            "projection_time": 0.0,
            "culling_time": 0.0,
            "sorting_time": 0.0,
            "rasterization_time": 0.0,
            "total_time": 0.0,
        }

        # Background color
        self.background_color = torch.tensor(
            config.background_color, dtype=torch.float32, device=self.device
        )

        logger.info(f"VoxelRasterizerGPU initialized on device: {self.device}")

    def __call__(
        self,
        voxels: dict[str, torch.Tensor],
        camera_matrix: torch.Tensor,
        intrinsics: torch.Tensor,
        viewport_size: tuple[int, int],
    ) -> dict[str, torch.Tensor]:
        """
        Main rasterization entry point

        Args:
            voxels: Dictionary containing voxel data
                - positions: [N, 3] voxel center positions
                - sizes: [N] or [N, 3] voxel sizes
                - densities: [N] density values
                - colors: [N, C] color coefficients
            camera_matrix: [4, 4] camera transformation matrix
            intrinsics: [3, 3] camera intrinsics matrix
            viewport_size: (width, height) viewport dimensions

        Returns:
            Dictionary containing rendered results
                - rgb: [H, W, 3] RGB image
                - depth: [H, W] depth map
        """
        if not CUDA_AVAILABLE:
            return self._fallback_cpu_rendering(voxels, camera_matrix, intrinsics, viewport_size)

        start_time = time.time()

        # Ensure all tensors are on the correct device
        voxels = self._ensure_tensors_on_device(voxels)
        camera_matrix = camera_matrix.to(self.device)
        intrinsics = intrinsics.to(self.device)

        # Convert viewport size to tensor
        viewport_tensor = torch.tensor(viewport_size, dtype=torch.int32, device=self.device)

        # Perform CUDA rasterization
        rgb, depth = voxel_rasterizer_cuda.voxel_rasterization(
            voxels["positions"],
            voxels["sizes"],
            voxels["densities"],
            voxels["colors"],
            camera_matrix,
            intrinsics,
            viewport_tensor,
            self.config.near_plane,
            self.config.far_plane,
            self.background_color,
            self.config.density_activation,
            self.config.color_activation,
            getattr(self.config, "sh_degree", 2),
        )

        self.performance_stats["total_time"] = time.time() - start_time

        return {"rgb": rgb, "depth": depth}

    def _ensure_tensors_on_device(self, voxels: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """Ensure all voxel tensors are on the correct device"""
        return {
            key: tensor.to(self.device) if isinstance(tensor, torch.Tensor) else tensor
            for key, tensor in voxels.items()
        }

    def _fallback_cpu_rendering(
        self,
        voxels: dict[str, torch.Tensor],
        camera_matrix: torch.Tensor,
        intrinsics: torch.Tensor,
        viewport_size: tuple[int, int],
    ) -> dict[str, torch.Tensor]:
        """Fallback to CPU rendering when CUDA is not available"""
        logger.warning("CUDA not available, using CPU fallback")

        # Import CPU version
        from nerfs.svraster.voxel_rasterizer import VoxelRasterizer

        cpu_rasterizer = VoxelRasterizer(self.config)
        return cpu_rasterizer(voxels, camera_matrix, intrinsics, viewport_size)

    def benchmark_performance(
        self,
        num_voxels: int = 10000,
        viewport_size: tuple[int, int] = (800, 600),
        num_iterations: int = 100,
    ) -> dict[str, float]:
        """
        Benchmark the rasterizer performance

        Args:
            num_voxels: Number of voxels to test with
            viewport_size: Viewport dimensions
            num_iterations: Number of iterations for benchmarking

        Returns:
            Performance statistics
        """
        if not CUDA_AVAILABLE:
            logger.warning("CUDA not available, cannot benchmark")
            return {}

        # Generate test data
        voxels = self._generate_test_voxels(num_voxels)
        camera_matrix = self._generate_test_camera_matrix()
        intrinsics = self._generate_test_intrinsics(viewport_size)
        viewport_tensor = torch.tensor(viewport_size, dtype=torch.int32, device=self.device)

        # Run benchmark
        timings = voxel_rasterizer_cuda.benchmark(
            voxels["positions"],
            voxels["sizes"],
            voxels["densities"],
            voxels["colors"],
            camera_matrix,
            intrinsics,
            viewport_tensor,
            num_iterations,
        )

        logger.info(f"Benchmark results: {timings}")
        return timings

    def _generate_test_voxels(self, num_voxels: int) -> dict[str, torch.Tensor]:
        """Generate test voxel data"""
        device = self.device

        # Random positions in a unit cube
        positions = torch.rand(num_voxels, 3, device=device) * 2.0 - 1.0

        # Random sizes
        sizes = torch.rand(num_voxels, device=device) * 0.1 + 0.01

        # Random densities
        densities = torch.randn(num_voxels, device=device) * 0.5

        # Random colors (RGB)
        colors = torch.rand(num_voxels, 3, device=device)

        return {"positions": positions, "sizes": sizes, "densities": densities, "colors": colors}

    def _generate_test_camera_matrix(self) -> torch.Tensor:
        """Generate a test camera matrix"""
        device = self.device

        # Simple camera looking at origin from (0, 0, 2)
        camera_matrix = torch.eye(4, device=device)
        camera_matrix[2, 3] = 2.0  # Move camera back

        return camera_matrix

    def _generate_test_intrinsics(self, viewport_size: tuple[int, int]) -> torch.Tensor:
        """Generate test camera intrinsics"""
        device = self.device
        width, height = viewport_size

        # Simple pinhole camera model
        focal_length = min(width, height) * 0.8
        cx, cy = width / 2, height / 2

        intrinsics = torch.tensor(
            [[focal_length, 0.0, cx], [0.0, focal_length, cy], [0.0, 0.0, 1.0]],
            dtype=torch.float32,
            device=device,
        )

        return intrinsics

    def get_performance_stats(self) -> dict[str, float]:
        """Get current performance statistics"""
        return self.performance_stats.copy()

    def reset_performance_stats(self) -> None:
        """Reset performance statistics"""
        for key in self.performance_stats:
            self.performance_stats[key] = 0.0

    def print_performance_stats(self) -> None:
        """Print current performance statistics"""
        logger.info("VoxelRasterizerGPU Performance Statistics:")
        for key, value in self.performance_stats.items():
            logger.info(f"  {key}: {value:.4f}s")


class VoxelRasterizerGPUTrainer:
    """
    Trainer for GPU-accelerated voxel rasterizer

    This class provides training utilities for the GPU rasterizer,
    including loss computation and optimization.
    """

    def __init__(self, rasterizer: VoxelRasterizerGPU, config, device: Optional[str] = None):
        self.rasterizer = rasterizer
        self.config = config
        self.device = torch.device(device) if device else rasterizer.device

        # Loss functions
        self.rgb_loss_fn = torch.nn.MSELoss()
        self.depth_loss_fn = torch.nn.MSELoss()

        logger.info(f"VoxelRasterizerGPUTrainer initialized on device: {self.device}")

    def compute_loss(
        self, predicted: dict[str, torch.Tensor], target: dict[str, torch.Tensor]
    ) -> dict[str, torch.Tensor]:
        """
        Compute training loss

        Args:
            predicted: Predicted rendering results
            target: Target rendering results

        Returns:
            Dictionary containing loss components
        """
        losses = {}

        # RGB loss
        if "rgb" in predicted and "rgb" in target:
            losses["rgb_loss"] = self.rgb_loss_fn(predicted["rgb"], target["rgb"])

        # Depth loss
        if "depth" in predicted and "depth" in target:
            losses["depth_loss"] = self.depth_loss_fn(predicted["depth"], target["depth"])

        # Total loss
        total_loss = sum(losses.values())
        losses["total_loss"] = total_loss

        return losses

    def train_step(
        self,
        voxels: dict[str, torch.Tensor],
        camera_matrix: torch.Tensor,
        intrinsics: torch.Tensor,
        viewport_size: tuple[int, int],
        target: dict[str, torch.Tensor],
    ) -> dict[str, float]:
        """
        Perform a single training step

        Args:
            voxels: Voxel data
            camera_matrix: Camera transformation matrix
            intrinsics: Camera intrinsics
            viewport_size: Viewport dimensions
            target: Target rendering results

        Returns:
            Loss values
        """
        # Forward pass
        predicted = self.rasterizer(voxels, camera_matrix, intrinsics, viewport_size)

        # Compute loss
        losses = self.compute_loss(predicted, target)

        # Convert to float for logging
        loss_values = {key: value.item() for key, value in losses.items()}

        return loss_values


# Utility functions
def create_camera_matrix_from_pose(camera_pose: torch.Tensor) -> torch.Tensor:
    """Create camera matrix from pose matrix"""
    if CUDA_AVAILABLE:
        return voxel_rasterizer_cuda.create_camera_matrix(camera_pose)
    else:
        return camera_pose


def estimate_camera_from_rays(
    ray_origins: torch.Tensor, ray_directions: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Estimate camera matrix and intrinsics from ray data"""
    if CUDA_AVAILABLE:
        return voxel_rasterizer_cuda.rays_to_camera_matrix(ray_origins, ray_directions)
    else:
        # CPU fallback
        device = ray_origins.device

        # Estimate camera center
        camera_center = torch.mean(ray_origins, dim=0)

        # Estimate camera direction
        mean_direction = torch.mean(ray_directions, dim=0)
        mean_direction = mean_direction / torch.norm(mean_direction)

        # Build camera coordinate system
        forward = -mean_direction
        up_vec = torch.tensor([0.0, 1.0, 0.0], device=device)
        right = torch.cross(forward, up_vec)
        right = right / torch.norm(right)
        up = torch.cross(right, forward)

        # Build camera matrix
        rotation = torch.stack([right, up, forward], dim=1)
        translation = -torch.matmul(rotation.t(), camera_center.unsqueeze(1)).squeeze(1)

        camera_matrix = torch.eye(4, device=device)
        camera_matrix[:3, :3] = rotation.t()
        camera_matrix[:3, 3] = translation

        # Simplified intrinsics
        intrinsics = torch.tensor(
            [[800.0, 0.0, 400.0], [0.0, 800.0, 300.0], [0.0, 0.0, 1.0]], device=device
        )

        return camera_matrix, intrinsics


def benchmark_rasterizer(
    num_voxels: int = 10000, viewport_size: tuple[int, int] = (800, 600), num_iterations: int = 100
) -> dict[str, float]:
    """Benchmark the rasterizer performance"""
    if not CUDA_AVAILABLE:
        logger.warning("CUDA not available, cannot benchmark")
        return {}

    # Create a simple config
    class SimpleConfig:
        def __init__(self):
            self.near_plane = 0.1
            self.far_plane = 100.0
            self.background_color = [0.0, 0.0, 0.0]
            self.density_activation = "exp"
            self.color_activation = "sigmoid"

    config = SimpleConfig()
    rasterizer = VoxelRasterizerGPU(config)

    return rasterizer.benchmark_performance(num_voxels, viewport_size, num_iterations)
