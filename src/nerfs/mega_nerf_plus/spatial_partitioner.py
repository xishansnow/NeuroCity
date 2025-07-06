"""
from __future__ import annotations

Spatial partitioning components for Mega-NeRF++

This module implements various spatial partitioning strategies for large-scale scenes:
- Adaptive octree partitioning
- Hierarchical grid partitioning
- Photogrammetric-aware partitioning
- Memory-efficient partitioning
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Optional, Any, TypeVar, TypedDict, Sequence, cast
from dataclasses import dataclass
import math


@dataclass
class PartitionConfig:
    """Configuration for spatial partitioning"""

    max_partition_size: int = 1024
    min_partition_size: int = 64
    overlap_ratio: float = 0.1
    max_depth: int = 8
    min_samples_per_partition: int = 1000
    adaptive_threshold: float = 0.01


# Type aliases for improved readability
T = TypeVar("T")


class OctreePartitionDict(TypedDict):
    bounds: torch.Tensor
    depth: int
    parent_idx: int | None
    children: List[int]


class PartitionDict(TypedDict):
    bounds: Tuple[float, ...]
    bounds_with_overlap: Tuple[float, ...]
    centroid: np.ndarray
    size: Tuple[int, int, int]
    data: Dict[str, Any]
    children: Optional[List["PartitionDict"]]  # For hierarchical partitioning
    coverage_score: Optional[float]  # For photogrammetric partitioning
    has_overlap: Optional[bool]  # For overlap tracking
    photogrammetric: Optional[bool]  # For photogrammetric metadata
    optimal_lod: Optional[int]  # For LOD selection
    rendering_priority: Optional[float]  # For rendering order
    memory_requirement: Optional[float]  # For memory tracking


class SpatialPartitioner:
    """
    Base class for spatial partitioning strategies
    """

    def __init__(self, scene_bounds: Tuple[float, ...], config: PartitionConfig):
        self.scene_bounds = scene_bounds
        self.config = config
        self.partitions: List[PartitionDict] = []
        self._partition_bounds: List[torch.Tensor] = []  # Use protected name

    @property
    def partition_bounds(self) -> List[torch.Tensor]:
        return self._partition_bounds

    @partition_bounds.setter
    def partition_bounds(self, value: List[torch.Tensor]) -> None:
        self._partition_bounds = value

    def partition_scene(
        self,
        scene_bounds: torch.Tensor,
        camera_positions: torch.Tensor,
    ) -> List[PartitionDict]:
        """
        Partition the scene into manageable chunks

        Args:
            scene_bounds: [2, 3] tensor with min/max bounds
            camera_positions: [N, 3] camera positions

        Returns:
            List of partition dictionaries
        """
        raise NotImplementedError

    def get_partition_for_point(self, point: torch.Tensor) -> int | None:
        """Get partition index for a given point"""
        for i, bounds in enumerate(self.partition_bounds):
            if self._point_in_bounds(point, bounds):
                return i
        return None

    def _point_in_bounds(self, point: torch.Tensor, bounds: torch.Tensor) -> bool:
        """Check if point is within bounds"""
        in_bounds = torch.all(point >= bounds[0]) and torch.all(point <= bounds[1])
        return bool(in_bounds.item())


class AdaptiveOctreePartitioner(SpatialPartitioner):
    """
    Adaptive octree partitioning based on scene complexity
    """

    def __init__(self, scene_bounds: Tuple[float, ...], config: PartitionConfig):
        super().__init__(scene_bounds, config)
        self.density_grid: torch.Tensor | None = None
        self.octree_partitions: List[OctreePartitionDict] = []

    def partition_scene(
        self,
        scene_bounds: torch.Tensor,
        camera_positions: torch.Tensor,
    ) -> List[PartitionDict]:
        """
        Create adaptive octree partitioning

        Args:
            scene_bounds: [2, 3] tensor with min/max bounds
            camera_positions: [N, 3] camera positions

        Returns:
            List of partition dictionaries
        """
        self.octree_partitions = []
        self._partition_bounds = []  # Use protected attribute

        # Initialize root node
        root_partition: OctreePartitionDict = {
            "bounds": scene_bounds,
            "depth": 0,
            "parent_idx": None,
            "children": [],
        }
        self.octree_partitions.append(root_partition)
        self._partition_bounds.append(scene_bounds)

        # Recursively subdivide
        self._subdivide_recursive(0, scene_bounds, camera_positions)

        # Convert octree partitions to standard partitions
        self.partitions = self._convert_to_standard_partitions()
        return self.partitions

    def _convert_to_standard_partitions(self) -> List[PartitionDict]:
        """Convert octree partitions to standard format"""
        standard_partitions: List[PartitionDict] = []

        for octree_part in self.octree_partitions:
            bounds = octree_part["bounds"]
            min_bounds = [float(x) for x in bounds[0].cpu().tolist()]
            max_bounds = [float(x) for x in bounds[1].cpu().tolist()]
            center = ((bounds[0] + bounds[1]) / 2).cpu().numpy()

            partition: PartitionDict = {
                "bounds": tuple(min_bounds + max_bounds),
                "bounds_with_overlap": tuple(min_bounds + max_bounds),
                "centroid": center,
                "size": (64, 64, 64),
                "data": {},
            }
            standard_partitions.append(partition)

        return standard_partitions

    def _subdivide_recursive(
        self,
        partition_idx: int,
        bounds: torch.Tensor,
        camera_positions: torch.Tensor,
        depth: int = 0,
    ) -> None:
        """Recursively subdivide partition if needed"""
        if depth >= self.config.max_depth:
            return

        # Check if subdivision is needed
        if self._should_subdivide(bounds, camera_positions):
            # Create child partitions
            child_bounds = self._create_child_bounds(bounds)

            for child_bound in child_bounds:
                child_idx = len(self.octree_partitions)

                # Create child partition
                child_partition: OctreePartitionDict = {
                    "bounds": child_bound,
                    "depth": depth + 1,
                    "parent_idx": partition_idx,
                    "children": [],
                }

                # Update parent's children list
                self.octree_partitions[partition_idx]["children"].append(child_idx)

                # Add child partition
                self.octree_partitions.append(child_partition)
                self._partition_bounds.append(child_bound)

                # Recursively subdivide child
                self._subdivide_recursive(
                    child_idx, child_bound, camera_positions, depth + 1
                )

    def _should_subdivide(
        self, bounds: torch.Tensor, camera_positions: torch.Tensor
    ) -> bool:
        """Check if partition should be subdivided"""
        # Simple heuristic based on camera density
        center = (bounds[0] + bounds[1]) / 2
        size = bounds[1] - bounds[0]
        max_size = float(torch.max(size).item())

        if max_size < self.config.min_partition_size:
            return False

        # Count cameras within bounds
        in_bounds = torch.all(
            (camera_positions >= bounds[0]) & (camera_positions <= bounds[1]), dim=1
        )
        camera_count = int(torch.sum(in_bounds).item())

        return camera_count > self.config.min_samples_per_partition

    def _create_child_bounds(self, bounds: torch.Tensor) -> List[torch.Tensor]:
        """Create bounds for child partitions"""
        center = (bounds[0] + bounds[1]) / 2
        child_bounds = []

        # Create 8 octants
        for x in [0, 1]:
            for y in [0, 1]:
                for z in [0, 1]:
                    min_bound = torch.where(
                        torch.tensor([x, y, z], device=bounds.device) == 0,
                        bounds[0],
                        center,
                    )
                    max_bound = torch.where(
                        torch.tensor([x, y, z], device=bounds.device) == 0,
                        center,
                        bounds[1],
                    )
                    child_bounds.append(torch.stack([min_bound, max_bound]))

        return child_bounds


class HierarchicalGridPartitioner(SpatialPartitioner):
    """
    Hierarchical grid partitioning with adaptive resolution
    """

    def __init__(self, scene_bounds: Tuple[float, ...], config: PartitionConfig):
        super().__init__(scene_bounds, config)
        self.grid_hierarchy: List[List[PartitionDict]] = []

    def partition_scene(
        self,
        scene_bounds: torch.Tensor,
        camera_positions: torch.Tensor,
        target_partition_count: Optional[int] = None,
    ) -> List[PartitionDict]:
        """
        Create hierarchical grid partitioning

        Args:
            scene_bounds: [2, 3] scene bounding box
            camera_positions: [N, 3] camera positions
            target_partition_count: Target number of partitions

        Returns:
            List of partition dictionaries
        """
        if target_partition_count is None:
            target_partition_count = max(8, len(camera_positions) // 100)

        # Determine grid resolution for target partition count
        grid_res = math.ceil(target_partition_count ** (1 / 3))

        # Create hierarchical levels
        levels = []
        current_res = 2  # Start with 2x2x2

        while current_res <= grid_res:
            level = self._create_grid_level(scene_bounds, current_res, camera_positions)
            levels.append(level)
            current_res *= 2

        self.grid_hierarchy = levels

        # Use finest level as partitions
        partitions = levels[-1] if levels else []

        # Add overlap between adjacent partitions
        partitions = self._add_partition_overlap(partitions, scene_bounds)

        self.partitions = partitions
        self.partition_bounds = [p["bounds"] for p in partitions]

        return partitions

    def _create_grid_level(
        self,
        scene_bounds: torch.Tensor,
        resolution: int,
        camera_positions: torch.Tensor,
    ) -> List[PartitionDict]:
        """Create grid level with given resolution"""
        partitions = []

        # Calculate grid cell size
        scene_size = scene_bounds[1] - scene_bounds[0]
        cell_size = scene_size / resolution

        for i in range(resolution):
            for j in range(resolution):
                for k in range(resolution):
                    # Calculate cell bounds
                    mins = scene_bounds[0] + torch.tensor(
                        [i, j, k],
                        dtype=torch.float32,
                    )
                    maxs = mins + cell_size

                    cell_bounds = torch.stack([mins, maxs])

                    # Find cameras for this cell
                    cell_cameras = self._get_cameras_for_cell(
                        cell_bounds, camera_positions
                    )

                    # Only create partition if it has cameras
                    if len(cell_cameras) > 0:
                        partition = {
                            "bounds": cell_bounds,
                            "cameras": cell_cameras,
                            "grid_idx": (
                                i,
                                j,
                                k,
                            ),
                        }
                        partitions.append(partition)

        return partitions

    def _get_cameras_for_cell(
        self,
        cell_bounds: torch.Tensor,
        camera_positions: torch.Tensor,
    ) -> torch.Tensor:
        """Get cameras that observe a grid cell"""
        # Expand cell bounds slightly for camera inclusion
        expansion = 0.1 * (cell_bounds[1] - cell_bounds[0])
        expanded_bounds = torch.stack(
            [cell_bounds[0] - expansion, cell_bounds[1] + expansion]
        )

        # Check which cameras are within or near the cell
        within_x = (camera_positions[:, 0] >= expanded_bounds[0, 0]) & (
            camera_positions[:, 0] <= expanded_bounds[1, 0]
        )
        within_y = (camera_positions[:, 1] >= expanded_bounds[0, 1]) & (
            camera_positions[:, 1] <= expanded_bounds[1, 1]
        )
        within_z = (camera_positions[:, 2] >= expanded_bounds[0, 2]) & (
            camera_positions[:, 2] <= expanded_bounds[1, 2]
        )

        # Also include cameras that can see the cell (simplified)
        center = (cell_bounds[0] + cell_bounds[1]) / 2
        max_distance = torch.norm(cell_bounds[1] - cell_bounds[0]) * 2
        distances = torch.norm(camera_positions - center.unsqueeze(0), dim=1)
        within_range = distances < max_distance

        valid_mask = (within_x & within_y & within_z) | within_range
        return camera_positions[valid_mask]

    def _add_partition_overlap(
        self,
        partitions: List[PartitionDict],
        scene_bounds: torch.Tensor,
    ) -> List[PartitionDict]:
        """Add overlap between adjacent partitions"""
        if not partitions:
            return partitions

        # Calculate overlap size
        scene_size = scene_bounds[1] - scene_bounds[0]
        min_size = torch.min(scene_size)
        overlap_size = min_size * self.config.overlap_ratio

        # Expand each partition by overlap amount
        for partition in partitions:
            bounds = partition["bounds"]
            overlap_vector = torch.full((3,), overlap_size / 2)

            # Expand bounds
            new_bounds = torch.stack(
                [
                    torch.max(
                        bounds[0] - overlap_vector,
                        scene_bounds[0],
                    )
                ]
            )

            partition["bounds"] = new_bounds
            partition["has_overlap"] = True

        return partitions


class PhotogrammetricPartitioner(SpatialPartitioner):
    """
    Photogrammetric-aware spatial partitioning

    Takes into account image coverage, resolution, and viewing angles
    """

    def __init__(self, config: PartitionConfig):
        super().__init__(config)

    def partition_scene(
        self,
        scene_bounds: torch.Tensor,
        camera_positions: torch.Tensor,
        camera_orientations: torch.Tensor,
        image_resolutions: torch.Tensor,
        intrinsics: torch.Tensor,
    ) -> List[PartitionDict]:
        """
        Create photogrammetric-aware partitioning

        Args:
            scene_bounds: [2, 3] scene bounding box
            camera_positions: [N, 3] camera positions
            camera_orientations: [N, 3, 3] camera rotation matrices
            image_resolutions: [N, 2] image resolutions (width, height)
            intrinsics: [N, 3, 3] camera intrinsic matrices

        Returns:
            List of partition dictionaries with photogrammetric info
        """
        # Calculate viewing coverage for each region
        coverage_map = self._compute_coverage_map(
            scene_bounds,
            camera_positions,
            camera_orientations,
            image_resolutions,
            intrinsics,
        )

        # Adaptive partitioning based on coverage
        partitions = self._adaptive_partition_by_coverage(
            scene_bounds,
            coverage_map,
            camera_positions,
            camera_orientations,
            image_resolutions,
            intrinsics,
        )

        # Add photogrammetric metadata
        partitions = self._add_photogrammetric_metadata(partitions)

        self.partitions = partitions
        self.partition_bounds = [p["bounds"] for p in partitions]

        return partitions

    def _compute_coverage_map(
        self,
        scene_bounds: torch.Tensor,
        camera_positions: torch.Tensor,
        camera_orientations: torch.Tensor,
        image_resolutions: torch.Tensor,
        intrinsics: torch.Tensor,
    ) -> torch.Tensor:
        """Compute coverage map for the scene"""

        # Create coarse 3D grid for coverage analysis
        grid_res = 32
        coverage_grid = torch.zeros(grid_res, grid_res, grid_res)

        # Sample points in the scene
        scene_size = scene_bounds[1] - scene_bounds[0]
        for i in range(grid_res):
            for j in range(grid_res):
                for k in range(grid_res):
                    # Grid point position
                    grid_pos = (
                        scene_bounds[0]
                        + torch.tensor([i, j, k], dtype=torch.float32)
                        * scene_size
                        / grid_res
                    )

                    # Count cameras that can see this point
                    visibility_count = 0
                    total_resolution = 0

                    for cam_idx in range(len(camera_positions)):
                        if self._point_visible_from_camera(
                            grid_pos,
                            camera_positions[cam_idx],
                            camera_orientations[cam_idx],
                            intrinsics[cam_idx],
                        ):
                            visibility_count += 1
                            total_resolution += torch.prod(image_resolutions[cam_idx])

                    # Coverage score combines visibility and resolution
                    coverage_grid[i, j, k] = visibility_count * math.sqrt(
                        total_resolution
                    )

        return coverage_grid

    def _point_visible_from_camera(
        self,
        point: torch.Tensor,
        camera_pos: torch.Tensor,
        camera_rot: torch.Tensor,
        intrinsic: torch.Tensor,
    ) -> bool:
        """Check if point is visible from camera (simplified)"""

        # Transform point to camera coordinates
        point_cam = camera_rot @ (point - camera_pos)

        # Check if point is in front of camera
        if point_cam[2] <= 0:
            return False

        # Project to image plane
        point_proj = intrinsic @ point_cam
        pixel = point_proj[:2] / point_proj[2]

        # Check if projection is within reasonable image bounds
        # (using simplified bounds check)
        return (-100 <= pixel[0] <= 8192) and (-100 <= pixel[1] <= 8192)

    def _adaptive_partition_by_coverage(
        self,
        scene_bounds: torch.Tensor,
        coverage_map: torch.Tensor,
        camera_positions: torch.Tensor,
        camera_orientations: torch.Tensor,
        image_resolutions: torch.Tensor,
        intrinsics: torch.Tensor,
    ) -> List[PartitionDict]:
        """Partition scene based on coverage analysis"""

        partitions = []

        # Start with coarse grid and refine high-coverage areas
        grid_res = coverage_map.shape[0]
        scene_size = scene_bounds[1] - scene_bounds[0]
        cell_size = scene_size / grid_res

        for i in range(grid_res):
            for j in range(grid_res):
                for k in range(grid_res):
                    coverage_score = coverage_map[i, j, k]

                    # Create partition for cells with sufficient coverage
                    if coverage_score > 0:
                        # Calculate cell bounds
                        mins = scene_bounds[0] + torch.tensor(
                            [i, j, k],
                            dtype=torch.float32,
                        )
                        maxs = mins + cell_size
                        cell_bounds = torch.stack([mins, maxs])

                        # Refine high-coverage cells
                        if coverage_score > coverage_map.mean() * 2:
                            # Subdivide high-coverage cell
                            sub_partitions = self._subdivide_cell(
                                cell_bounds,
                                camera_positions,
                                camera_orientations,
                                image_resolutions,
                                intrinsics,
                                depth=1,
                            )
                            partitions.extend(sub_partitions)
                        else:
                            # Use cell as single partition
                            partition = {
                                "bounds": cell_bounds,
                                "coverage_score": coverage_score.item(),
                            }
                            partitions.append(partition)

        return partitions

    def _subdivide_cell(
        self,
        bounds: torch.Tensor,
        camera_positions: torch.Tensor,
        camera_orientations: torch.Tensor,
        image_resolutions: torch.Tensor,
        intrinsics: torch.Tensor,
        depth: int,
    ) -> List[PartitionDict]:
        """Recursively subdivide high-coverage cells"""

        if depth > 2:  # Limit subdivision depth
            return [{"bounds": bounds, "subdivided": False}]

        # Subdivide into 8 octants
        center = (bounds[0] + bounds[1]) / 2
        sub_partitions = []

        for octant in range(8):
            # Calculate octant bounds
            octant_bounds = self._get_octant_bounds(bounds, center, octant)

            # Check coverage for this octant
            coverage = self._estimate_octant_coverage(
                octant_bounds,
                camera_positions,
                camera_orientations,
                image_resolutions,
                intrinsics,
            )

            if coverage > 0:
                # Further subdivide if coverage is very high
                if coverage > 10 and depth < 2:
                    deeper_partitions = self._subdivide_cell(
                        octant_bounds,
                        camera_positions,
                        camera_orientations,
                        image_resolutions,
                        intrinsics,
                        depth + 1,
                    )
                    sub_partitions.extend(deeper_partitions)
                else:
                    partition = {
                        "bounds": octant_bounds,
                        "coverage_score": coverage,
                        "subdivision_depth": depth,
                    }
                    sub_partitions.append(partition)

        return sub_partitions

    def _get_octant_bounds(
        self,
        parent_bounds: torch.Tensor,
        center: torch.Tensor,
        octant: int,
    ) -> torch.Tensor:
        """Get bounds for octant (same as AdaptiveOctree)"""
        mins = parent_bounds[0].clone()
        maxs = parent_bounds[1].clone()

        if octant & 1:
            mins[0] = center[0]
        else:
            maxs[0] = center[0]
        if octant & 2:
            mins[1] = center[1]
        else:
            maxs[1] = center[1]
        if octant & 4:
            mins[2] = center[2]
        else:
            maxs[2] = center[2]

        return torch.stack([mins, maxs])

    def _estimate_octant_coverage(
        self,
        bounds: torch.Tensor,
        camera_positions: torch.Tensor,
        camera_orientations: torch.Tensor,
        image_resolutions: torch.Tensor,
        intrinsics: torch.Tensor,
    ) -> float:
        """Estimate coverage for an octant"""

        # Sample a few points in the octant
        center = (bounds[0] + bounds[1]) / 2
        size = bounds[1] - bounds[0]

        sample_points = [
            center,  # Center
            bounds[0],
            bounds[1],  # Corners
            bounds[0] + 0.5 * size,  # Mid-point
        ]

        total_coverage = 0
        for point in sample_points:
            visibility_count = 0
            for cam_idx in range(len(camera_positions)):
                if self._point_visible_from_camera(
                    point,
                    camera_positions[cam_idx],
                    camera_orientations[cam_idx],
                    intrinsics[cam_idx],
                ):
                    visibility_count += 1

            total_coverage += visibility_count

        return total_coverage / len(sample_points)

    def _add_photogrammetric_metadata(
        self, partitions: List[PartitionDict]
    ) -> List[PartitionDict]:
        """Add photogrammetric metadata to partitions"""

        for partition in partitions:
            # Add metadata fields
            partition.update(
                {
                    "photogrammetric": True,
                    "optimal_lod": self._estimate_optimal_lod(
                        partition,
                    ),
                }
            )

        return partitions

    def _estimate_optimal_lod(self, partition: PartitionDict) -> int:
        """Estimate optimal level of detail for partition"""
        coverage = partition.get("coverage_score", 1.0)

        if coverage > 5:
            return 0  # Highest detail
        elif coverage > 2:
            return 1  # Medium detail
        else:
            return 2  # Lower detail

    def _estimate_memory_requirement(self, partition: PartitionDict) -> float:
        """Estimate memory requirement for partition (in MB)"""
        bounds = partition["bounds"]
        volume = torch.prod(bounds[1] - bounds[0]).item()
        coverage = partition.get("coverage_score", 1.0)

        # Heuristic: memory scales with volume and coverage
        base_memory = volume * 0.1  # MB per unit volume
        coverage_multiplier = 1 + coverage * 0.5

        return base_memory * coverage_multiplier

    def _create_adaptive_octree(
        self,
        bounds: torch.Tensor,
        camera_positions: torch.Tensor,
        depth: int = 0,
        max_depth: int = 4,
    ) -> List[torch.Tensor]:
        """
        Create adaptive octree structure based on scene complexity.

        Args:
            bounds: [2, 3] tensor with min/max bounds
            camera_positions: [N, 3] camera positions
            depth: Current recursion depth
            max_depth: Maximum recursion depth

        Returns:
            List of partition bounds tensors
        """
        # Check termination conditions
        if depth >= max_depth:
            return [bounds]

        # Estimate partition complexity
        center = (bounds[0] + bounds[1]) / 2
        size = bounds[1] - bounds[0]

        # Sample points in partition
        num_samples = 1000
        sample_points = torch.rand(num_samples, 3, device=bounds.device)
        sample_points = sample_points * size + bounds[0]

        # Estimate complexity based on visibility
        complexity = self._estimate_partition_complexity(
            sample_points, camera_positions
        )

        # Subdivide if complex enough
        if complexity > 0.5 and depth < max_depth:
            # Create 8 child partitions
            child_bounds = []
            for i in range(2):
                for j in range(2):
                    for k in range(2):
                        min_bound = torch.clone(bounds[0])
                        max_bound = torch.clone(center)

                        if i == 1:
                            min_bound[0] = center[0]
                            max_bound[0] = bounds[1][0]
                        if j == 1:
                            min_bound[1] = center[1]
                            max_bound[1] = bounds[1][1]
                        if k == 1:
                            min_bound[2] = center[2]
                            max_bound[2] = bounds[1][2]

                        child_bounds.append(torch.stack([min_bound, max_bound]))

            # Recursively subdivide children
            all_bounds = []
            for child in child_bounds:
                all_bounds.extend(
                    self._create_adaptive_octree(
                        child, camera_positions, depth + 1, max_depth
                    )
                )
            return all_bounds

        return [bounds]

    def _estimate_partition_complexity(
        self, points: torch.Tensor, camera_positions: torch.Tensor
    ) -> float:
        """
        Estimate partition complexity based on visibility.

        Args:
            points: [N, 3] sampled points
            camera_positions: [M, 3] camera positions

        Returns:
            Complexity score between 0 and 1
        """
        # Simple visibility-based complexity
        visible_points = 0
        for point in points:
            for cam_pos in camera_positions:
                if torch.norm(point - cam_pos) < 5.0:  # Arbitrary distance threshold
                    visible_points += 1
                    break

        return visible_points / len(points)
    
    
