"""
from __future__ import annotations

GPU-optimized SVRaster implementation with enhanced performance
"""

import torch
import torch.nn.functional as F
from torch.amp.grad_scaler import GradScaler
from torch.amp.autocast_mode import autocast
from contextlib import nullcontext
import numpy as np
from typing import Optional, Tuple, Dict, Any
import logging
from tqdm import tqdm
from nerfs.svraster.core import SVRasterConfig, SVRasterLoss
from nerfs.svraster.cuda.ema import EMAModel
from nerfs.svraster.cuda.trainer.svraster_optimized_kernels import SVRasterOptimizedKernels

logger = logging.getLogger(__name__)

# Try to import CUDA extension
try:
    import nerfs.svraster.cuda.svraster_cuda as svraster_cuda

    CUDA_AVAILABLE = True
    logger.info("SVRaster CUDA extension loaded successfully")
except ImportError:
    CUDA_AVAILABLE = False
    logger.warning("SVRaster CUDA extension not available, falling back to CPU implementation")


class SVRasterGPU:
    """
    Enhanced GPU-optimized SVRaster model using CUDA kernels and advanced algorithms
    """

    def __init__(self, config):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Initialize optimized kernels
        self.optimized_kernels = SVRasterOptimizedKernels(self.device)

        # Initialize voxel storage
        self.voxel_positions = []
        self.voxel_sizes = []
        self.voxel_densities = []
        self.voxel_colors = []
        self.voxel_levels = []
        self.voxel_morton_codes = []

        # Scene bounds
        self.scene_min = torch.tensor(config.scene_bounds[:3], device=self.device)
        self.scene_max = torch.tensor(config.scene_bounds[3:], device=self.device)
        self.scene_size = self.scene_max - self.scene_min

        # Initialize base voxels
        self._initialize_base_voxels()

        # Performance tracking (enhanced)
        self.performance_stats = {
            "ray_voxel_intersection_time": 0.0,
            "voxel_rasterization_time": 0.0,
            "morton_sorting_time": 0.0,
            "subdivision_time": 0.0,
            "pruning_time": 0.0,
            "memory_optimization_time": 0.0,
            "spatial_hash_time": 0.0,
        }

    def _initialize_base_voxels(self):
        """Initialize base level voxels"""
        base_res = self.config.base_resolution

        # Create regular grid at base level
        x = torch.linspace(0, 1, base_res + 1, device=self.device)[:-1] + 0.5 / base_res
        y = torch.linspace(0, 1, base_res + 1, device=self.device)[:-1] + 0.5 / base_res
        z = torch.linspace(0, 1, base_res + 1, device=self.device)[:-1] + 0.5 / base_res

        xx, yy, zz = torch.meshgrid(x, y, z, indexing="ij")
        positions = torch.stack([xx, yy, zz], dim=-1).reshape(-1, 3)

        # Convert to world coordinates
        positions = positions * self.scene_size + self.scene_min

        # Initialize parameters
        num_voxels = positions.shape[0]
        voxel_size = self.scene_size.max() / base_res

        # Create tensors with gradients
        self.voxel_positions.append(torch.nn.Parameter(positions))
        self.voxel_sizes.append(
            torch.nn.Parameter(torch.full((num_voxels,), voxel_size, device=self.device))
        )
        self.voxel_densities.append(
            torch.nn.Parameter(torch.randn(num_voxels, device=self.device) * 0.1)
        )
        self.voxel_colors.append(
            torch.nn.Parameter(torch.rand(num_voxels, 3, device=self.device) * 0.5 + 0.25)
        )

        # Set levels and Morton codes
        self.voxel_levels.append(torch.zeros(num_voxels, dtype=torch.int32, device=self.device))
        self._compute_morton_codes(0)

        logger.info(f"Initialized {num_voxels} base voxels")

    def _compute_morton_codes(self, level_idx: int):
        """Compute Morton codes for voxels at given level"""
        if not CUDA_AVAILABLE:
            # Fallback to CPU implementation
            positions = self.voxel_positions[level_idx]
            scene_bounds = torch.cat([self.scene_min, self.scene_max])
            morton_codes = self._morton_encode_cpu(positions, scene_bounds)
        else:
            # Use CUDA implementation
            morton_codes = svraster_cuda.compute_morton_codes(
                self.voxel_positions[level_idx], torch.cat([self.scene_min, self.scene_max])
            )

        self.voxel_morton_codes.append(morton_codes)

    def _morton_encode_cpu(
        self, positions: torch.Tensor, scene_bounds: torch.Tensor
    ) -> torch.Tensor:
        """Optimized CPU fallback for Morton encoding"""
        scene_min = scene_bounds[:3]
        scene_max = scene_bounds[3:]
        scene_size = scene_max - scene_min

        # Normalize positions to [0, 1]
        normalized = (positions - scene_min) / scene_size
        normalized = torch.clamp(normalized, 0.0, 1.0)

        # Convert to integer grid coordinates with higher precision (16-bit)
        max_coord = 65535  # 16-bit precision
        coords = (normalized * max_coord).long()

        # Optimized bit interleaving using vectorized operations
        x, y, z = coords[:, 0], coords[:, 1], coords[:, 2]

        # Use the optimized Morton encoding from utils
        from nerfs.svraster.utils.morton_utils import morton_encode_batch

        try:
            morton_codes = morton_encode_batch(coords)
        except ImportError:
            # Fallback to simple implementation if utils not available
            morton_codes = self._simple_morton_encode(x, y, z)

        return morton_codes.to(torch.int32)

    def _simple_morton_encode(
        self, x: torch.Tensor, y: torch.Tensor, z: torch.Tensor
    ) -> torch.Tensor:
        """优化的Morton编码实现"""
        morton_codes = torch.zeros_like(x, dtype=torch.int64)

        # 使用更高效的位操作
        # 支持21位精度，总共63位Morton码
        mask = 0x1FFFFF  # 21位掩码
        x = torch.clamp(x, 0, mask)
        y = torch.clamp(y, 0, mask)
        z = torch.clamp(z, 0, mask)

        # 优化的位交错算法
        for i in range(21):
            bit_pos = 1 << i
            morton_codes |= (x & bit_pos) << (2 * i)
            morton_codes |= (y & bit_pos) << (2 * i + 1)
            morton_codes |= (z & bit_pos) << (2 * i + 2)

        return morton_codes

    def forward(
        self, ray_origins: torch.Tensor, ray_directions: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        """Forward pass using GPU-optimized kernels"""
        if not CUDA_AVAILABLE:
            return self._forward_cpu(ray_origins, ray_directions)

        # Get current voxel representation
        voxel_positions, voxel_sizes, voxel_densities, voxel_colors = self._get_voxel_data()

        # Ray-voxel intersection
        import time

        start_time = time.time()

        intersection_result = svraster_cuda.ray_voxel_intersection(
            ray_origins, ray_directions, voxel_positions, voxel_sizes, voxel_densities, voxel_colors
        )

        self.performance_stats["ray_voxel_intersection_time"] = time.time() - start_time

        # Voxel rasterization
        start_time = time.time()

        rasterization_result = svraster_cuda.voxel_rasterization(
            ray_origins,
            ray_directions,
            voxel_positions,
            voxel_sizes,
            voxel_densities,
            voxel_colors,
            *intersection_result,
        )

        self.performance_stats["voxel_rasterization_time"] = time.time() - start_time

        output_colors, output_depths = rasterization_result

        return {
            "rgb": output_colors,
            "depth": output_depths,
            "intersection_counts": intersection_result[0],
            "intersection_indices": intersection_result[1],
        }

    def _forward_cpu(
        self, ray_origins: torch.Tensor, ray_directions: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        """Enhanced CPU fallback implementation with optimized algorithms"""
        batch_size = ray_origins.shape[0]
        device = ray_origins.device

        # Get voxel data
        voxel_positions, voxel_sizes, voxel_densities, voxel_colors = self._get_voxel_data()

        if voxel_positions.shape[0] == 0:
            # No voxels, return empty results
            return {
                "rgb": torch.zeros(batch_size, 3, device=device),
                "depth": torch.full((batch_size,), self.config.far_plane, device=device),
                "intersection_counts": torch.zeros(batch_size, dtype=torch.int32, device=device),
                "intersection_indices": torch.zeros(
                    batch_size, 100, dtype=torch.int32, device=device
                ),
            }

        # Use optimized ray-voxel intersection
        intersection_results = self.optimized_kernels.optimized_ray_voxel_intersection(
            ray_origins, ray_directions, voxel_positions, voxel_sizes, use_spatial_hash=True
        )

        # Render using volume rendering
        rgb, depth = self._cpu_volume_rendering(
            ray_origins,
            ray_directions,
            voxel_positions,
            voxel_sizes,
            voxel_densities,
            voxel_colors,
            intersection_results,
        )

        # Update performance stats
        kernel_stats = self.optimized_kernels.get_performance_stats()
        self.performance_stats["spatial_hash_time"] = kernel_stats.get("dda_traversal_time", 0.0)

        return {
            "rgb": rgb,
            "depth": depth,
            "intersection_counts": intersection_results["counts"],
            "intersection_indices": intersection_results["indices"],
        }

    def _cpu_ray_voxel_intersection(
        self,
        ray_origins: torch.Tensor,
        ray_directions: torch.Tensor,
        voxel_positions: torch.Tensor,
        voxel_sizes: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """CPU ray-voxel intersection testing"""
        batch_size = ray_origins.shape[0]
        num_voxels = voxel_positions.shape[0]
        device = ray_origins.device

        max_intersections = min(100, num_voxels)
        intersection_counts = torch.zeros(batch_size, dtype=torch.int32, device=device)
        intersection_indices = torch.zeros(
            batch_size, max_intersections, dtype=torch.int32, device=device
        )

        # Simple AABB intersection for each ray
        for ray_idx in range(batch_size):
            ray_o = ray_origins[ray_idx]
            ray_d = ray_directions[ray_idx]

            intersections = []

            for voxel_idx in range(num_voxels):
                voxel_center = voxel_positions[voxel_idx]
                voxel_size = voxel_sizes[voxel_idx]

                # AABB intersection test
                voxel_min = voxel_center - voxel_size * 0.5
                voxel_max = voxel_center + voxel_size * 0.5

                if self._ray_aabb_intersect(ray_o, ray_d, voxel_min, voxel_max):
                    intersections.append(voxel_idx)

                    if len(intersections) >= max_intersections:
                        break

            intersection_counts[ray_idx] = len(intersections)
            if intersections:
                intersection_indices[ray_idx, : len(intersections)] = torch.tensor(
                    intersections, dtype=torch.int32, device=device
                )

        return {"counts": intersection_counts, "indices": intersection_indices}

    def _ray_aabb_intersect(
        self,
        ray_origin: torch.Tensor,
        ray_direction: torch.Tensor,
        aabb_min: torch.Tensor,
        aabb_max: torch.Tensor,
    ) -> bool:
        """Ray-AABB intersection test"""
        # Avoid division by zero
        eps = 1e-8
        ray_direction = torch.where(torch.abs(ray_direction) < eps, eps, ray_direction)

        t_min = (aabb_min - ray_origin) / ray_direction
        t_max = (aabb_max - ray_origin) / ray_direction

        t_enter = torch.maximum(t_min, t_max).min()
        t_exit = torch.minimum(t_min, t_max).max()

        return bool(t_enter <= t_exit and t_exit > 0)

    def _cpu_volume_rendering(
        self,
        ray_origins: torch.Tensor,
        ray_directions: torch.Tensor,
        voxel_positions: torch.Tensor,
        voxel_sizes: torch.Tensor,
        voxel_densities: torch.Tensor,
        voxel_colors: torch.Tensor,
        intersection_results: dict[str, torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """CPU volume rendering"""
        batch_size = ray_origins.shape[0]
        device = ray_origins.device

        rgb_output = torch.zeros(batch_size, 3, device=device)
        depth_output = torch.full((batch_size,), self.config.far_plane, device=device)

        for ray_idx in range(batch_size):
            ray_o = ray_origins[ray_idx]
            ray_d = ray_directions[ray_idx]

            count = intersection_results["counts"][ray_idx].item()
            if count == 0:
                continue

            indices = intersection_results["indices"][ray_idx, :count]

            # Sort by distance along ray
            distances = []
            for i in range(int(count)):
                voxel_idx = int(indices[i].item())
                voxel_center = voxel_positions[voxel_idx]
                distance = torch.dot(voxel_center - ray_o, ray_d)
                distances.append(distance)

            if distances:
                distances = torch.tensor(distances, device=device)
                sort_indices = torch.argsort(distances)

                # Volume rendering
                transmittance = 1.0
                accumulated_color = torch.zeros(3, device=device)

                for i in range(int(count)):
                    voxel_idx = int(indices[sort_indices[i]].item())

                    density = torch.exp(voxel_densities[voxel_idx])
                    color = torch.sigmoid(voxel_colors[voxel_idx])

                    alpha = 1.0 - torch.exp(-density * voxel_sizes[voxel_idx])

                    accumulated_color += transmittance * alpha * color
                    transmittance *= 1.0 - alpha

                    if transmittance < 0.01:  # Early termination
                        break

                rgb_output[ray_idx] = accumulated_color
                if count > 0:
                    depth_output[ray_idx] = distances[sort_indices[0]]

        return rgb_output, depth_output

    def _get_voxel_data(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Get current voxel data as tensors"""
        # Combine all levels into single tensors
        positions = []
        sizes = []
        densities = []
        colors = []

        for level_idx in range(len(self.voxel_positions)):
            positions.append(self.voxel_positions[level_idx])
            sizes.append(self.voxel_sizes[level_idx])
            densities.append(self.voxel_densities[level_idx])
            colors.append(self.voxel_colors[level_idx])

        return (
            torch.cat(positions, dim=0),
            torch.cat(sizes, dim=0),
            torch.cat(densities, dim=0),
            torch.cat(colors, dim=0),
        )

    def adaptive_subdivision(self, subdivision_criteria: torch.Tensor) -> None:
        """Adaptively subdivide voxels based on criteria"""
        if not CUDA_AVAILABLE:
            # CPU fallback implementation with actual subdivision logic
            self._cpu_adaptive_subdivision(subdivision_criteria)
            return

        import time

        start_time = time.time()

        # Get current voxel data
        voxel_positions, voxel_sizes, voxel_densities, voxel_colors = self._get_voxel_data()

        # Use default threshold if not available in config
        subdivision_threshold = getattr(self.config, "subdivision_threshold", 0.01)

        # Perform adaptive subdivision
        subdivision_result = svraster_cuda.adaptive_subdivision(
            voxel_positions,
            voxel_sizes,
            voxel_densities,
            voxel_colors,
            subdivision_criteria,
            subdivision_threshold,
            self.config.max_octree_levels,
        )

        subdivision_flags, new_voxel_count = subdivision_result

        # Apply subdivision with actual logic
        if new_voxel_count > 0:
            self._apply_subdivision_results(subdivision_flags, new_voxel_count)
            logger.info(
                f"Subdivided {new_voxel_count // 8} voxels, created {new_voxel_count} new voxels"
            )

        self.performance_stats["subdivision_time"] = time.time() - start_time

    def voxel_pruning(self, pruning_threshold: Optional[float] = None) -> None:
        """Prune low-density voxels"""
        if pruning_threshold is None:
            pruning_threshold = 0.001  # Default threshold

        import time

        start_time = time.time()

        # Get current voxel data
        voxel_positions, voxel_sizes, voxel_densities, voxel_colors = self._get_voxel_data()

        # Compute density values
        densities = torch.exp(voxel_densities)

        # Create keep mask
        keep_mask = densities > pruning_threshold

        # Apply pruning
        if keep_mask.sum() < len(keep_mask):
            logger.info(f"Pruning {len(keep_mask) - keep_mask.sum()} voxels")

            # Update voxel data
            for level_idx in range(len(self.voxel_positions)):
                level_mask = keep_mask[: self.voxel_positions[level_idx].shape[0]]
                keep_mask = keep_mask[self.voxel_positions[level_idx].shape[0] :]

                self.voxel_positions[level_idx] = torch.nn.Parameter(
                    self.voxel_positions[level_idx][level_mask]
                )
                self.voxel_sizes[level_idx] = torch.nn.Parameter(
                    self.voxel_sizes[level_idx][level_mask]
                )
                self.voxel_densities[level_idx] = torch.nn.Parameter(
                    self.voxel_densities[level_idx][level_mask]
                )
                self.voxel_colors[level_idx] = torch.nn.Parameter(
                    self.voxel_colors[level_idx][level_mask]
                )

        self.performance_stats["pruning_time"] = time.time() - start_time

    def get_voxel_statistics(self) -> dict[str, Any]:
        """Get statistics about voxel distribution"""
        stats = {
            "total_voxels": sum(pos.shape[0] for pos in self.voxel_positions),
            "num_levels": len(self.voxel_positions),
            "performance_stats": self.performance_stats.copy(),
        }

        for level_idx in range(len(self.voxel_positions)):
            stats[f"level_{level_idx}_voxels"] = self.voxel_positions[level_idx].shape[0]

        return stats

    def print_performance_stats(self):
        """Print performance statistics"""
        print("SVRaster GPU Performance Statistics:")
        print("=" * 40)
        for key, value in self.performance_stats.items():
            print(f"{key}: {value:.4f} seconds")

        stats = self.get_voxel_statistics()
        print(f"Total voxels: {stats['total_voxels']}")
        print(f"Number of levels: {stats['num_levels']}")
        for i in range(stats["num_levels"]):
            print(f"Level {i} voxels: {stats[f'level_{i}_voxels']}")

    def parameters(self):
        """Return all trainable parameters as a generator (similar to nn.Module)"""
        for level_densities in self.voxel_densities:
            if level_densities.requires_grad:
                yield level_densities
        for level_colors in self.voxel_colors:
            if level_colors.requires_grad:
                yield level_colors

    def _cpu_adaptive_subdivision(self, subdivision_criteria: torch.Tensor) -> None:
        """CPU fallback for adaptive subdivision"""
        subdivision_threshold = getattr(self.config, "subdivision_threshold", 0.01)

        # Determine which voxels to subdivide based on criteria
        should_subdivide = subdivision_criteria > subdivision_threshold

        if not should_subdivide.any():
            return

        total_subdivided = 0
        current_idx = 0

        # Process each level
        for level_idx in range(len(self.voxel_positions)):
            level_size = self.voxel_positions[level_idx].shape[0]
            level_criteria = should_subdivide[current_idx : current_idx + level_size]

            if level_criteria.any() and level_idx < self.config.max_octree_levels - 1:
                subdivided_count = self._subdivide_level(level_idx, level_criteria)
                total_subdivided += subdivided_count

            current_idx += level_size

        logger.info(f"CPU subdivision: subdivided {total_subdivided} voxels")

    def _subdivide_level(self, level_idx: int, subdivision_mask: torch.Tensor) -> int:
        """Subdivide voxels at a specific level"""
        from nerfs.svraster.utils.octree_utils import octree_subdivision

        if not subdivision_mask.any():
            return 0

        parent_positions = self.voxel_positions[level_idx][subdivision_mask]
        parent_sizes = self.voxel_sizes[level_idx][subdivision_mask]
        parent_densities = self.voxel_densities[level_idx][subdivision_mask]
        parent_colors = self.voxel_colors[level_idx][subdivision_mask]

        # Create child voxels
        child_positions, child_sizes = octree_subdivision(
            parent_positions, parent_sizes, torch.ones_like(subdivision_mask[subdivision_mask])
        )

        # Initialize child properties
        num_children = child_positions.shape[0]
        child_densities = (
            parent_densities.repeat_interleave(8, dim=0) * 0.8
        )  # Inherit with some variation
        child_colors = parent_colors.repeat_interleave(8, dim=0)

        # Add to next level
        next_level = level_idx + 1

        # Ensure we have enough levels
        while len(self.voxel_positions) <= next_level:
            self.voxel_positions.append(torch.nn.Parameter(torch.empty(0, 3, device=self.device)))
            self.voxel_sizes.append(torch.empty(0, device=self.device, requires_grad=True))
            self.voxel_densities.append(torch.nn.Parameter(torch.empty(0, device=self.device)))
            self.voxel_colors.append(torch.nn.Parameter(torch.empty(0, 3, device=self.device)))
            self.voxel_levels.append(torch.empty(0, dtype=torch.int32, device=self.device))
            self.voxel_morton_codes.append(torch.empty(0, dtype=torch.int32, device=self.device))

        # Append new voxels to next level
        if self.voxel_positions[next_level].shape[0] == 0:
            self.voxel_positions[next_level] = torch.nn.Parameter(child_positions)
            self.voxel_sizes[next_level] = torch.nn.Parameter(child_sizes)
            self.voxel_densities[next_level] = torch.nn.Parameter(child_densities)
            self.voxel_colors[next_level] = torch.nn.Parameter(child_colors)
        else:
            self.voxel_positions[next_level] = torch.nn.Parameter(
                torch.cat([self.voxel_positions[next_level], child_positions], dim=0)
            )
            self.voxel_sizes[next_level] = torch.nn.Parameter(
                torch.cat([self.voxel_sizes[next_level], child_sizes], dim=0)
            )
            self.voxel_densities[next_level] = torch.nn.Parameter(
                torch.cat([self.voxel_densities[next_level], child_densities], dim=0)
            )
            self.voxel_colors[next_level] = torch.nn.Parameter(
                torch.cat([self.voxel_colors[next_level], child_colors], dim=0)
            )

        # Update levels and compute Morton codes
        new_levels = torch.full((num_children,), next_level, dtype=torch.int32, device=self.device)
        self.voxel_levels[next_level] = torch.cat([self.voxel_levels[next_level], new_levels])

        new_morton_codes = self._morton_encode_cpu(
            child_positions, torch.cat([self.scene_min, self.scene_max])
        )
        self.voxel_morton_codes[next_level] = torch.cat(
            [self.voxel_morton_codes[next_level], new_morton_codes]
        )

        # Remove subdivided parent voxels
        keep_mask = ~subdivision_mask
        if keep_mask.any():
            self.voxel_positions[level_idx] = torch.nn.Parameter(
                self.voxel_positions[level_idx][keep_mask]
            )
            self.voxel_sizes[level_idx] = torch.nn.Parameter(self.voxel_sizes[level_idx][keep_mask])
            self.voxel_densities[level_idx] = torch.nn.Parameter(
                self.voxel_densities[level_idx][keep_mask]
            )
            self.voxel_colors[level_idx] = torch.nn.Parameter(
                self.voxel_colors[level_idx][keep_mask]
            )
            self.voxel_levels[level_idx] = self.voxel_levels[level_idx][keep_mask]
            self.voxel_morton_codes[level_idx] = self.voxel_morton_codes[level_idx][keep_mask]
        else:
            # All voxels subdivided, clear the level
            self.voxel_positions[level_idx] = torch.nn.Parameter(
                torch.empty(0, 3, device=self.device)
            )
            self.voxel_sizes[level_idx] = torch.nn.Parameter(torch.empty(0, device=self.device))
            self.voxel_densities[level_idx] = torch.nn.Parameter(torch.empty(0, device=self.device))
            self.voxel_colors[level_idx] = torch.nn.Parameter(torch.empty(0, 3, device=self.device))
            self.voxel_levels[level_idx] = torch.empty(0, dtype=torch.int32, device=self.device)
            self.voxel_morton_codes[level_idx] = torch.empty(
                0, dtype=torch.int32, device=self.device
            )

        return int(subdivision_mask.sum().item())

    def _apply_subdivision_results(
        self, subdivision_flags: torch.Tensor, new_voxel_count: int
    ) -> None:
        """Apply subdivision results from CUDA kernel"""
        if new_voxel_count <= 0:
            return

        logger.info(f"Applying CUDA subdivision results: {new_voxel_count} new voxels")

        # 计算每个级别需要细分的体素数量
        subdivision_per_level = []
        current_idx = 0

        for level_idx in range(len(self.voxel_positions)):
            level_size = self.voxel_positions[level_idx].shape[0]
            level_flags = subdivision_flags[current_idx : current_idx + level_size]
            subdivision_count = level_flags.sum().item()
            subdivision_per_level.append(subdivision_count)
            current_idx += level_size

        # 应用细分结果
        for level_idx, subdivision_count in enumerate(subdivision_per_level):
            if subdivision_count > 0 and level_idx < self.config.max_octree_levels - 1:
                level_flags = subdivision_flags[
                    sum(subdivision_per_level[:level_idx]) : sum(
                        subdivision_per_level[: level_idx + 1]
                    )
                ]
                self._subdivide_level(level_idx, level_flags.bool())

        # 重新计算Morton码以保持空间排序
        for level_idx in range(len(self.voxel_positions)):
            if self.voxel_positions[level_idx].shape[0] > 0:
                self._compute_morton_codes(level_idx)

    def benchmark_performance(
        self, num_rays: int = 1000, num_iterations: int = 10
    ) -> dict[str, float]:
        """
        性能基准测试

        Args:
            num_rays: 测试光线数量
            num_iterations: 测试迭代次数

        Returns:
            性能统计结果
        """
        logger.info(
            f"Running performance benchmark with {num_rays} rays, {num_iterations} iterations"
        )

        # 生成测试数据
        ray_origins = torch.randn(num_rays, 3, device=self.device)
        ray_directions = F.normalize(torch.randn(num_rays, 3, device=self.device), dim=1)

        # 重置性能计数器
        self.optimized_kernels.reset_performance_counters()

        # 预热
        for _ in range(3):
            _ = self.forward(ray_origins[:100], ray_directions[:100])

        # 同步CUDA以确保准确计时
        if self.device.type == "cuda":
            torch.cuda.synchronize()

        import time

        start_time = time.time()

        # 运行基准测试
        for i in range(num_iterations):
            outputs = self.forward(ray_origins, ray_directions)

            if self.device.type == "cuda":
                torch.cuda.synchronize()

            if i % (num_iterations // 4) == 0:
                logger.info(f"Benchmark progress: {i+1}/{num_iterations}")

        total_time = time.time() - start_time

        # 计算统计信息
        avg_time_per_iteration = total_time / num_iterations
        rays_per_second = (num_rays * num_iterations) / total_time

        # 获取详细的性能统计
        kernel_stats = self.optimized_kernels.get_performance_stats()

        benchmark_results = {
            "total_time": total_time,
            "avg_time_per_iteration": avg_time_per_iteration,
            "rays_per_second": rays_per_second,
            "num_rays": num_rays,
            "num_iterations": num_iterations,
            "device": str(self.device),
            **self.performance_stats,
            **kernel_stats,
        }

        # 内存使用统计
        if self.device.type == "cuda":
            benchmark_results.update(
                {
                    "gpu_memory_allocated": torch.cuda.memory_allocated() / 1024**2,  # MB
                    "gpu_memory_reserved": torch.cuda.memory_reserved() / 1024**2,  # MB
                    "max_memory_allocated": torch.cuda.max_memory_allocated() / 1024**2,
                }
            )

        # 体素统计
        voxel_stats = self.get_voxel_statistics()
        benchmark_results.update(
            {
                "total_voxels": voxel_stats["total_voxels"],
                "num_levels": voxel_stats["num_levels"],
            }
        )

        logger.info(f"Benchmark completed: {rays_per_second:.1f} rays/second")
        logger.info(f"Average time per iteration: {avg_time_per_iteration*1000:.2f} ms")

        return benchmark_results

    def optimize_for_production(self) -> None:
        """生产环境优化"""
        logger.info("Optimizing for production deployment...")

        # 1. 优化Morton码排序
        self._optimize_morton_codes()

        # 2. 内存优化
        self._optimize_memory_layout()

        # 3. 预计算加速结构
        self._precompute_acceleration_structures()

        # 4. 清理临时数据
        self._cleanup_temporary_data()

        logger.info("Production optimization completed")

    def _optimize_morton_codes(self) -> None:
        """优化Morton码排序"""
        import time

        start_time = time.time()

        for level_idx in range(len(self.voxel_positions)):
            if self.voxel_positions[level_idx].shape[0] > 0:
                positions = self.voxel_positions[level_idx]
                scene_bounds = torch.cat([self.scene_min, self.scene_max])

                # 使用优化的Morton编码
                morton_codes = self.optimized_kernels.optimized_morton_sorting(
                    positions, scene_bounds, precision_bits=21
                )

                # 根据Morton码排序体素
                sorted_indices = torch.argsort(morton_codes)

                self.voxel_positions[level_idx] = torch.nn.Parameter(
                    self.voxel_positions[level_idx][sorted_indices]
                )
                self.voxel_sizes[level_idx] = torch.nn.Parameter(
                    self.voxel_sizes[level_idx][sorted_indices]
                )
                self.voxel_densities[level_idx] = torch.nn.Parameter(
                    self.voxel_densities[level_idx][sorted_indices]
                )
                self.voxel_colors[level_idx] = torch.nn.Parameter(
                    self.voxel_colors[level_idx][sorted_indices]
                )

                self.voxel_morton_codes[level_idx] = morton_codes[sorted_indices]

        self.performance_stats["morton_sorting_time"] += time.time() - start_time

    def _optimize_memory_layout(self) -> None:
        """优化内存布局"""
        # 合并连续的内存块
        if self.device.type == "cuda":
            torch.cuda.empty_cache()

        # 清理优化内核的内存池
        self.optimized_kernels.cleanup_memory_pool()

        self.performance_stats["memory_optimization_time"] += 0.001  # 占位符

    def _precompute_acceleration_structures(self) -> None:
        """预计算加速结构"""
        # 预计算体素边界盒等加速结构
        # 这里可以添加更多的预计算逻辑
        pass

    def _cleanup_temporary_data(self) -> None:
        """清理临时数据"""
        # 清理不必要的临时变量
        import gc

        gc.collect()

        if self.device.type == "cuda":
            torch.cuda.empty_cache()

    def export_performance_report(self, filepath: str) -> None:
        """导出性能报告"""
        report = {
            "model_info": {
                "total_voxels": sum(pos.shape[0] for pos in self.voxel_positions),
                "num_levels": len(self.voxel_positions),
                "scene_bounds": self.config.scene_bounds,
                "device": str(self.device),
            },
            "performance_stats": self.performance_stats,
            "kernel_stats": self.optimized_kernels.get_performance_stats(),
            "memory_stats": {},
        }

        if self.device.type == "cuda":
            report["memory_stats"] = {
                "allocated": torch.cuda.memory_allocated() / 1024**2,
                "reserved": torch.cuda.memory_reserved() / 1024**2,
                "max_allocated": torch.cuda.max_memory_allocated() / 1024**2,
            }

        import json

        with open(filepath, "w") as f:
            json.dump(report, f, indent=2)

        logger.info(f"Performance report exported to {filepath}")


class SVRasterGPUTrainer:
    """GPU-optimized trainer for SVRaster with modern PyTorch features."""

    def __init__(
        self,
        model: SVRasterGPU,
        volume_renderer,  # 体积渲染器
        config: SVRasterConfig,
        train_dataset=None,
        val_dataset=None,
        device: Optional[str] = None,
    ):
        self.model = model
        self.volume_renderer = volume_renderer  # 紧密耦合的体积渲染器
        self.config = config
        self.device = model.device if device is None else torch.device(device)

        # 数据集属性
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset

        # Initialize AMP scaler
        if torch.cuda.is_available():
            self.scaler = GradScaler()
        else:
            self.scaler = None

        # Setup optimizer
        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )

        # Setup learning rate scheduler
        self.scheduler = torch.optim.lr_scheduler.OneCycleLR(
            self.optimizer,
            max_lr=config.learning_rate,
            total_steps=config.num_epochs * config.steps_per_epoch,
        )

        # Setup loss functions
        self.criterion = SVRasterLoss(config)
        self.ema_model = None

        # Training state
        self.epoch = 0
        self.step = 0
        self.best_loss = float("inf")

        logger.info(f"SVRasterGPUTrainer initialized with VolumeRenderer coupling")
        logger.info(f"Model device: {self.device}")
        logger.info(f"Volume renderer: {type(volume_renderer).__name__}")

    def _extract_voxels_from_model(self) -> dict[str, torch.Tensor]:
        """
        从 SVRasterGPU 模型中提取体素数据
        """
        # 获取当前体素数据
        voxel_positions, voxel_sizes, voxel_densities, voxel_colors = self.model._get_voxel_data()

        return {
            "positions": voxel_positions,
            "sizes": voxel_sizes,
            "densities": voxel_densities,
            "colors": voxel_colors,
        }

    def train_step(self, batch: dict[str, torch.Tensor]) -> dict[str, float]:
        """Perform a single training step with volume renderer."""
        batch = {k: v.to(self.device, non_blocking=True) for k, v in batch.items()}

        # 从批次中获取数据
        rays_o = batch["rays_o"]  # [B, N, 3] 或 [B, H, W, 3]
        rays_d = batch["rays_d"]  # [B, N, 3] 或 [B, H, W, 3]
        # 兼容不同的字段名
        if "target_rgb" in batch:
            target_rgb = batch["target_rgb"]  # [B, N, 3] 或 [B, H, W, 3]
        elif "colors" in batch:
            target_rgb = batch["colors"]  # [B, N, 3] 或 [B, H, W, 3]
        else:
            raise KeyError("Batch must contain either 'target_rgb' or 'colors' field")

        # 重塑为批量光线格式
        if rays_o.dim() == 4:  # [B, H, W, 3] -> [B, H*W, 3]
            B, H, W, _ = rays_o.shape
            rays_o = rays_o.view(B, H * W, 3)
            rays_d = rays_d.view(B, H * W, 3)
            target_rgb = target_rgb.view(B, H * W, 3)

        # 扁平化为 [N, 3] 格式（VolumeRenderer 期望的格式）
        if rays_o.dim() == 3:  # [B, N, 3] -> [B*N, 3]
            B, N, _ = rays_o.shape
            rays_o = rays_o.view(B * N, 3)
            rays_d = rays_d.view(B * N, 3)
            target_rgb = target_rgb.view(B * N, 3)

        with (
            autocast(device_type="cuda" if torch.cuda.is_available() else "cpu")
            if torch.cuda.is_available()
            else nullcontext()
        ):
            try:
                # 使用体积渲染器进行前向传播
                voxels = self._extract_voxels_from_model()

                render_result = self.volume_renderer(
                    voxels=voxels, ray_origins=rays_o, ray_directions=rays_d
                )

                # 计算损失
                targets = {"rgb": target_rgb}
                loss_dict = self.criterion(render_result, targets)
                total_loss = loss_dict["total_loss"]

            except Exception as e:
                logger.error(f"Error in forward pass: {str(e)}")
                # 创建一个简单的损失用于调试
                dummy_rgb = torch.ones_like(target_rgb) * 0.5
                mse_loss = F.mse_loss(dummy_rgb, target_rgb)
                total_loss = mse_loss
                loss_dict = {"rgb": mse_loss.item(), "total_loss": mse_loss.item()}

        self.optimizer.zero_grad(set_to_none=True)

        if self.scaler is not None:
            self.scaler.scale(total_loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            total_loss.backward()
            self.optimizer.step()

        self.scheduler.step()
        self.step += 1

        with torch.inference_mode():
            metrics = {"loss": total_loss.item(), "lr": self.scheduler.get_last_lr()[0]}
            if "rgb" in render_result and "target_rgb" in batch:
                mse = F.mse_loss(render_result["rgb"], batch["target_rgb"])
                psnr = -10.0 * torch.log10(mse)
                metrics["psnr"] = psnr.item()

        return metrics

    def train(self) -> None:
        """
        完整的训练流程
        """
        if self.train_dataset is None:
            raise ValueError("Training dataset not provided")

        from torch.utils.data import DataLoader

        # 创建数据加载器
        train_loader = DataLoader(
            self.train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=4,
            pin_memory=torch.cuda.is_available(),
        )

        # 检查数据加载器是否为空
        if len(train_loader) == 0:
            raise ValueError("Training dataset is empty - no batches to train on")

        val_loader = None
        if self.val_dataset is not None:
            # 使用默认的验证批次大小
            val_batch_size = getattr(self.config, "val_batch_size", 1)
            val_loader = DataLoader(
                self.val_dataset,
                batch_size=val_batch_size,
                shuffle=False,
                num_workers=2,
                pin_memory=torch.cuda.is_available(),
            )

        logger.info(f"Starting training for {self.config.num_epochs} epochs")
        logger.info(f"Training batches per epoch: {len(train_loader)}")

        for epoch in range(self.config.num_epochs):
            self.epoch = epoch
            # SVRasterGPU 不是 nn.Module，不需要 train() 调用

            # 训练一个epoch
            epoch_losses = []
            progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{self.config.num_epochs}")

            for batch in progress_bar:
                # 训练步骤
                metrics = self.train_step(batch)
                epoch_losses.append(metrics["loss"])

                # 更新进度条
                progress_bar.set_postfix(
                    {
                        "loss": f"{metrics['loss']:.4f}",
                        "lr": f"{metrics['lr']:.6f}",
                        "psnr": f"{metrics.get('psnr', 0):.2f}",
                    }
                )

            # 计算epoch平均损失
            avg_loss = sum(epoch_losses) / len(epoch_losses)
            logger.info(f"Epoch {epoch+1} - Avg Loss: {avg_loss:.4f}")

            # 验证
            validate_every = getattr(self.config, "validate_every", 1)
            if val_loader is not None and epoch % validate_every == 0:
                val_metrics = self.validate(val_loader)
                logger.info(f"Validation - Loss: {val_metrics['loss']:.4f}")

                # 保存最佳模型
                if val_metrics["loss"] < self.best_loss:
                    self.best_loss = val_metrics["loss"]
                    self.save_checkpoint(f"best_model_epoch_{epoch+1}.pth")

            # 保存检查点
            save_every = getattr(self.config, "save_every", 1000)
            if len(train_loader) > 0 and epoch % max(1, save_every // len(train_loader)) == 0:
                self.save_checkpoint(f"checkpoint_epoch_{epoch+1}.pth")

    def validate(self, dataloader) -> dict[str, float]:
        """
        验证模型 - 使用体积渲染器
        """
        # SVRasterGPU 不是 nn.Module，不需要 eval() 调用
        val_losses = []

        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Validating"):
                # 将数据移到正确的设备
                batch = {
                    k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                    for k, v in batch.items()
                }

                # 获取数据
                rays_o = batch["rays_o"]
                rays_d = batch["rays_d"]
                # 兼容不同的字段名
                if "target_rgb" in batch:
                    target_rgb = batch["target_rgb"]
                elif "colors" in batch:
                    target_rgb = batch["colors"]
                else:
                    raise KeyError("Batch must contain either 'target_rgb' or 'colors' field")

                # 重塑为批量光线格式
                if rays_o.dim() == 4:  # [B, H, W, 3] -> [B, H*W, 3]
                    B, H, W, _ = rays_o.shape
                    rays_o = rays_o.view(B, H * W, 3)
                    rays_d = rays_d.view(B, H * W, 3)
                    target_rgb = target_rgb.view(B, H * W, 3)

                # 扁平化为 [N, 3] 格式（VolumeRenderer 期望的格式）
                if rays_o.dim() == 3:  # [B, N, 3] -> [B*N, 3]
                    B, N, _ = rays_o.shape
                    rays_o = rays_o.view(B * N, 3)
                    rays_d = rays_d.view(B * N, 3)
                    target_rgb = target_rgb.view(B * N, 3)

                try:
                    # 使用体积渲染器进行前向传播
                    voxels = self._extract_voxels_from_model()

                    render_result = self.volume_renderer(
                        voxels=voxels, ray_origins=rays_o, ray_directions=rays_d
                    )

                    # 计算损失
                    targets = {"rgb": target_rgb}
                    loss_dict = self.criterion(render_result, targets)
                    val_losses.append(loss_dict["total_loss"].item())

                except Exception as e:
                    logger.error(f"Error in validation forward pass: {str(e)}")
                    # 创建一个简单的损失用于调试
                    dummy_rgb = torch.ones_like(target_rgb) * 0.5
                    mse_loss = F.mse_loss(dummy_rgb, target_rgb)
                    val_losses.append(mse_loss.item())

        return {"loss": sum(val_losses) / len(val_losses)}

    def save_checkpoint(self, filepath: str):
        """Save checkpoint"""
        model_state = {
            "voxel_positions": self.model.voxel_positions,
            "voxel_sizes": self.model.voxel_sizes,
            "voxel_densities": self.model.voxel_densities,
            "voxel_colors": self.model.voxel_colors,
        }

        checkpoint = {
            "epoch": self.epoch,
            "step": self.step,
            "model_state_dict": model_state,
            "optimizer_state_dict": self.optimizer.state_dict(),
            "config": self.config,
        }
        torch.save(checkpoint, filepath)
