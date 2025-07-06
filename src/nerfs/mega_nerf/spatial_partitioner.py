from __future__ import annotations

"""
Spatial Partitioner for Mega-NeRF

This module handles the spatial decomposition of large-scale scenes
into manageable subregions for parallel training and rendering.
"""

from typing import Any, Optional, Union


import numpy as np
import torch
from dataclasses import dataclass
import logging
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)

class SpatialPartitioner(ABC):
    """Base class for spatial partitioning strategies"""
    
    def __init__(self, scene_bounds: Tuple[float, ...], config: Dict[str, Any]):
        self.scene_bounds = scene_bounds
        self.config = config
        
    @abstractmethod
    def create_partitions(self) -> List[Dict[str, Any]]:
        """Create spatial partitions"""
        pass
    
    @abstractmethod
    def assign_points_to_partitions(self, points: np.ndarray) -> np.ndarray:
        """Assign 3D points to partitions"""
        pass
    
    @abstractmethod
    def get_partition_bounds(self, partition_idx: int) -> Tuple[float, ...]:
        """Get bounds for a specific partition"""
        pass

class GridPartitioner(SpatialPartitioner):
    """Grid-based spatial partitioner"""
    
    def __init__(
        self,
        scene_bounds: Tuple[float,
        float,
        float,
        float,
        float,
        float],
        grid_size: Tuple[int,int] = (16, 16),
        overlap_factor: float = 0.15
    ):
        """
        Initialize grid partitioner
        
        Args:
            scene_bounds: (x_min, y_min, z_min, x_max, y_max, z_max)
            grid_size: (grid_x, grid_y) - number of partitions in each dimension
            overlap_factor: Overlap ratio between adjacent partitions
        """
        config = {
            'grid_size': grid_size, 'overlap_factor': overlap_factor
        }
        super().__init__(scene_bounds, config)
        
        self.grid_size = grid_size
        self.overlap_factor = overlap_factor
        self.partitions = self.create_partitions()
        
    def create_partitions(self) -> List[Dict[str, Any]]:
        """Create grid-based partitions"""
        x_min, y_min, z_min, x_max, y_max, z_max = self.scene_bounds
        grid_x, grid_y = self.grid_size
        
        # Calculate partition dimensions
        x_step = (x_max - x_min) / grid_x
        y_step = (y_max - y_min) / grid_y
        
        # Calculate overlap distances
        x_overlap = x_step * self.overlap_factor
        y_overlap = y_step * self.overlap_factor
        
        partitions = []
        partition_idx = 0
        
        for i in range(grid_x):
            for j in range(grid_y):
                # Base partition bounds
                part_x_min = x_min + i * x_step
                part_x_max = x_min + (i + 1) * x_step
                part_y_min = y_min + j * y_step
                part_y_max = y_min + (j + 1) * y_step
                
                # Add overlap
                part_x_min_overlap = max(x_min, part_x_min - x_overlap)
                part_x_max_overlap = min(x_max, part_x_max + x_overlap)
                part_y_min_overlap = max(y_min, part_y_min - y_overlap)
                part_y_max_overlap = min(y_max, part_y_max + y_overlap)
                
                # Centroid
                centroid = np.array([
                    (
                        part_x_min + part_x_max,
                    )
                ])
                
                partition = {
                    'idx': partition_idx, 'grid_coords': (
                        i,
                        j,
                    )
                }
                
                partitions.append(partition)
                partition_idx += 1
        
        logger.info(f"Created {len(partitions)} grid partitions ({grid_x}x{grid_y})")
        return partitions
    
    def assign_points_to_partitions(self, points: np.ndarray) -> np.ndarray:
        """Assign points to the nearest partition centroid"""
        centroids = np.array([p['centroid'] for p in self.partitions])
        
        # Compute distances to all centroids
        distances = np.linalg.norm(points[:, None, :] - centroids[None, :, :], axis=2)
        
        # Assign to nearest centroid
        assignments = np.argmin(distances, axis=1)
        
        return assignments
    
    def get_partition_bounds(self, partition_idx: int) -> Tuple[float, ...]:
        """Get bounds for a specific partition"""
        if partition_idx >= len(self.partitions):
            raise ValueError(f"Partition index {partition_idx} out of range")
        
        return self.partitions[partition_idx]['bounds']
    
    def get_partition_bounds_with_overlap(self, partition_idx: int) -> Tuple[float, ...]:
        """Get bounds with overlap for a specific partition"""
        if partition_idx >= len(self.partitions):
            raise ValueError(f"Partition index {partition_idx} out of range")
        
        return self.partitions[partition_idx]['bounds_with_overlap']
    
    def get_partition_centroid(self, partition_idx: int) -> np.ndarray:
        """Get centroid for a specific partition"""
        if partition_idx >= len(self.partitions):
            raise ValueError(f"Partition index {partition_idx} out of range")
        
        return self.partitions[partition_idx]['centroid'].copy()
    
    def get_adjacent_partitions(self, partition_idx: int) -> List[int]:
        """Get indices of adjacent partitions"""
        if partition_idx >= len(self.partitions):
            raise ValueError(f"Partition index {partition_idx} out of range")
        
        current_partition = self.partitions[partition_idx]
        i, j = current_partition['grid_coords']
        grid_x, grid_y = self.grid_size
        
        adjacent = []
        
        # Check all 8 neighbors
        for di in [-1, 0, 1]:
            for dj in [-1, 0, 1]:
                if di == 0 and dj == 0:
                    continue
                
                ni, nj = i + di, j + dj
                if 0 <= ni < grid_x and 0 <= nj < grid_y:
                    neighbor_idx = ni * grid_y + nj
                    adjacent.append(neighbor_idx)
        
        return adjacent
    
    def visualize_partitions(self, save_path: Optional[str] = None):
        """Visualize the partition layout"""
        import matplotlib.pyplot as plt
        import matplotlib.patches as patches
        
        fig, ax = plt.subplots(1, 1, figsize=(12, 8))
        
        # Plot partition boundaries
        for partition in self.partitions:
            bounds = partition['bounds']
            x_min, y_min, _, x_max, y_max, _ = bounds
            
            # Draw partition rectangle
            rect = patches.Rectangle(
                (
                    x_min,
                    y_min,
                )
            )
            ax.add_patch(rect)
            
            # Draw centroid
            centroid = partition['centroid']
            ax.plot(centroid[0], centroid[1], 'ro', markersize=8)
            
            # Add partition index
            ax.text(
                centroid[0],
                centroid[1],
                str,
            )
        
        # Set axis properties
        x_min, y_min, _, x_max, y_max, _ = self.scene_bounds
        ax.set_xlim(x_min - 10, x_max + 10)
        ax.set_ylim(y_min - 10, y_max + 10)
        ax.set_aspect('equal')
        ax.set_xlabel('X (meters)')
        ax.set_ylabel('Y (meters)')
        ax.set_title('Mega-NeRF Spatial Partitioning (Top View)')
        ax.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            logger.info(f"Partition visualization saved to {save_path}")
        
        return fig, ax

class GeometryAwarePartitioner(SpatialPartitioner):
    """Geometry-aware spatial partitioner that considers scene structure"""
    
    def __init__(
        self,
        scene_bounds: Tuple[float,
        float,
        float,
        float,
        float,
        float],
        camera_positions: np.ndarray,
        num_partitions: int = 8,
        overlap_factor: float = 0.15,
        use_kmeans: bool = True
    ):
        """
        Initialize geometry-aware partitioner
        
        Args:
            scene_bounds: Scene boundaries
            camera_positions: Camera positions for geometry estimation (N, 3)
            num_partitions: Number of partitions to create
            overlap_factor: Overlap ratio between partitions
            use_kmeans: Whether to use k-means clustering for partitioning
        """
        config = {
            'num_partitions': num_partitions, 'overlap_factor': overlap_factor, 'use_kmeans': use_kmeans
        }
        super().__init__(scene_bounds, config)
        
        self.camera_positions = camera_positions
        self.num_partitions = num_partitions
        self.overlap_factor = overlap_factor
        self.use_kmeans = use_kmeans
        
        self.partitions = self.create_partitions()
    
    def create_partitions(self) -> List[Dict[str, Any]]:
        """Create geometry-aware partitions"""
        if self.use_kmeans:
            return self._create_kmeans_partitions()
        else:
            return self._create_density_based_partitions()
    
    def _create_kmeans_partitions(self) -> List[Dict[str, Any]]:
        """Create partitions using k-means clustering on camera positions"""
        from sklearn.cluster import KMeans
        
        # Use only X, Y coordinates for 2D clustering
        camera_positions_2d = self.camera_positions[:, :2]
        
        # Perform k-means clustering
        kmeans = KMeans(n_clusters=self.num_partitions, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(camera_positions_2d)
        centroids_2d = kmeans.cluster_centers_
        
        # Create partitions
        partitions = []
        x_min, y_min, z_min, x_max, y_max, z_max = self.scene_bounds
        
        for i, centroid_2d in enumerate(centroids_2d):
            # Get points assigned to this cluster
            cluster_points = camera_positions_2d[cluster_labels == i]
            
            if len(cluster_points) == 0:
                continue
            
            # Compute cluster bounds
            cluster_x_min, cluster_y_min = cluster_points.min(axis=0)
            cluster_x_max, cluster_y_max = cluster_points.max(axis=0)
            
            # Add margin based on cluster size
            x_margin = (cluster_x_max - cluster_x_min) * self.overlap_factor
            y_margin = (cluster_y_max - cluster_y_min) * self.overlap_factor
            
            # Expand bounds with margin
            part_x_min = max(x_min, cluster_x_min - x_margin)
            part_x_max = min(x_max, cluster_x_max + x_margin)
            part_y_min = max(y_min, cluster_y_min - y_margin)
            part_y_max = min(y_max, cluster_y_max + y_margin)
            
            # 3D centroid
            centroid = np.array([centroid_2d[0], centroid_2d[1], (z_min + z_max) / 2])
            
            partition = {
                'idx': i, 'bounds': (
                    part_x_min,
                    part_y_min,
                    z_min,
                    part_x_max,
                    part_y_max,
                    z_max,
                )
            }
            
            partitions.append(partition)
        
        logger.info(f"Created {len(partitions)} k-means based partitions")
        return partitions
    
    def _create_density_based_partitions(self) -> List[Dict[str, Any]]:
        """Create partitions based on camera density"""
        # Create a 2D histogram of camera positions
        x_min, y_min, z_min, x_max, y_max, z_max = self.scene_bounds
        
        # Create density grid
        grid_resolution = 50
        x_edges = np.linspace(x_min, x_max, grid_resolution + 1)
        y_edges = np.linspace(y_min, y_max, grid_resolution + 1)
        
        density, _, _ = np.histogram2d(
            self.camera_positions[:, 0], self.camera_positions[:, 1], bins=[x_edges, y_edges]
        )
        
        # Find high-density regions
        from scipy import ndimage
        from sklearn.cluster import DBSCAN
        
        # Smooth density
        density_smooth = ndimage.gaussian_filter(density, sigma=2)
        
        # Find peaks
        peaks = []
        threshold = np.percentile(density_smooth, 80)
        
        peak_coords = np.where(density_smooth > threshold)
        if len(peak_coords[0]) > 0:
            # Convert grid coordinates to world coordinates
            for i, j in zip(peak_coords[0], peak_coords[1]):
                x = x_edges[i] + (x_edges[1] - x_edges[0]) / 2
                y = y_edges[j] + (y_edges[1] - y_edges[0]) / 2
                peaks.append([x, y])
        
        peaks = np.array(peaks)
        
        # Cluster peaks to get partition centers
        if len(peaks) > self.num_partitions:
            from sklearn.cluster import KMeans
            kmeans = KMeans(n_clusters=self.num_partitions, random_state=42, n_init=10)
            kmeans.fit(peaks)
            centroids_2d = kmeans.cluster_centers_
        else:
            centroids_2d = peaks
        
        # Create partitions around centroids
        partitions = []
        avg_spacing = np.sqrt((x_max - x_min) * (y_max - y_min) / self.num_partitions)
        
        for i, centroid_2d in enumerate(centroids_2d):
            # Define partition bounds
            half_size = avg_spacing / 2 * (1 + self.overlap_factor)
            
            part_x_min = max(x_min, centroid_2d[0] - half_size)
            part_x_max = min(x_max, centroid_2d[0] + half_size)
            part_y_min = max(y_min, centroid_2d[1] - half_size)
            part_y_max = min(y_max, centroid_2d[1] + half_size)
            
            centroid = np.array([centroid_2d[0], centroid_2d[1], (z_min + z_max) / 2])
            
            partition = {
                'idx': i, 
                'bounds': (
                    part_x_min,
                    part_y_min,
                    z_min,
                    part_x_max,
                    part_y_max,
                    z_max
                )
            }
            
            partitions.append(partition)
        
        logger.info(f"Created {len(partitions)} density-based partitions")
        return partitions
    
    def assign_points_to_partitions(self, points: np.ndarray) -> np.ndarray:
        """Assign points to partitions based on distance to centroids"""
        centroids = np.array([p['centroid'] for p in self.partitions])
        
        # Compute distances
        distances = np.linalg.norm(points[:, None, :] - centroids[None, :, :], axis=2)
        
        # Assign to nearest centroid
        assignments = np.argmin(distances, axis=1)
        
        return assignments
    
    def get_partition_bounds(self, partition_idx: int) -> Tuple[float, ...]:
        """Get bounds for a specific partition"""
        if partition_idx >= len(self.partitions):
            raise ValueError(f"Partition index {partition_idx} out of range")
        
        return self.partitions[partition_idx]['bounds']
    
    def get_camera_coverage_stats(self) -> Dict[str, Any]:
        """Get statistics about camera coverage for each partition"""
        stats = {}
        
        # Assign cameras to partitions
        assignments = self.assign_points_to_partitions(self.camera_positions)
        
        for i, partition in enumerate(self.partitions):
            mask = assignments == i
            num_cameras = mask.sum()
            
            if num_cameras > 0:
                partition_cameras = self.camera_positions[mask]
                camera_density = num_cameras / ((partition['bounds'][3] - partition['bounds'][0]) * 
                                              (partition['bounds'][4] - partition['bounds'][1]))
            else:
                camera_density = 0
            
            stats[f'partition_{i}'] = {
                'num_cameras': int(
                    num_cameras,
                )
            }
        
        return stats
    
    def visualize_partitions_with_cameras(self, save_path: Optional[str] = None):
        """Visualize partitions with camera positions"""
        import matplotlib.pyplot as plt
        import matplotlib.patches as patches
        
        fig, ax = plt.subplots(1, 1, figsize=(12, 8))
        
        # Plot camera positions
        ax.scatter(
            self.camera_positions[:,
            0],
            self.camera_positions[:,
            1],
            c='red',
            s=10,
            alpha=0.6,
            label='Cameras',
        )
        
        # Plot partitions
        colors = plt.cm.Set3(np.linspace(0, 1, len(self.partitions)))
        
        for i, (partition, color) in enumerate(zip(self.partitions, colors)):
            bounds = partition['bounds']
            x_min, y_min, _, x_max, y_max, _ = bounds
            
            # Draw partition rectangle
            rect = patches.Rectangle(
                (
                    x_min,
                    y_min,
                )
            )
            ax.add_patch(rect)
            
            # Draw centroid
            centroid = partition['centroid']
            ax.plot(centroid[0], centroid[1], 'ko', markersize=10)
            ax.text(
                centroid[0],
                centroid[1],
                str,
            )
        
        # Set axis properties
        x_min, y_min, _, x_max, y_max, _ = self.scene_bounds
        ax.set_xlim(x_min - 10, x_max + 10)
        ax.set_ylim(y_min - 10, y_max + 10)
        ax.set_aspect('equal')
        ax.set_xlabel('X (meters)')
        ax.set_ylabel('Y (meters)')
        ax.set_title('Geometry-Aware Partitioning with Camera Positions')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            logger.info(f"Partition visualization saved to {save_path}")
        
        return fig, ax 