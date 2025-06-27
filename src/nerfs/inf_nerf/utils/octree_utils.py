from typing import Any, Optional
"""
Octree utilities for InfNeRF.

This module provides tools for:
- Octree construction from sparse points
- Intelligent octree pruning
- Ground Sampling Distance (GSD) calculation
- Octree visualization and analysis
"""

import numpy as np
import torch
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import json
from pathlib import Path

from ..core import OctreeNode, InfNeRFConfig

class OctreeBuilder:
    """
    Builder for constructing octrees from sparse points.
    """
    
    def __init__(self, config: InfNeRFConfig):
        """
        Initialize octree builder.
        
        Args:
            config: InfNeRF configuration
        """
        self.config = config
    
    def build_from_sparse_points(
        self,
        sparse_points: np.ndarray,
        scene_bounds: Optional[tuple[np.ndarray,
        np.ndarray]] = None,
    ) -> OctreeNode:
        """
        Build octree from sparse points.
        
        Args:
            sparse_points: [N, 3] sparse points from SfM
            scene_bounds: Optional (min_bounds, max_bounds) for scene
            
        Returns:
            Root node of the constructed octree
        """
        # Calculate scene bounds if not provided
        if scene_bounds is None:
            min_bounds = np.min(sparse_points, axis=0)
            max_bounds = np.max(sparse_points, axis=0)
            
            # Add padding
            padding = np.max(max_bounds - min_bounds) * 0.1
            min_bounds -= padding
            max_bounds += padding
        else:
            min_bounds, max_bounds = scene_bounds
        
        # Calculate root node parameters
        scene_size = np.max(max_bounds - min_bounds)
        scene_center = (min_bounds + max_bounds) / 2
        
        # Create root node
        root_node = OctreeNode(
            center=scene_center, size=scene_size, level=0, config=self.config
        )
        
        # Recursively build octree
        self._build_recursive(root_node, sparse_points, 0)
        
        return root_node
    
    def _build_recursive(self, node: OctreeNode, sparse_points: np.ndarray, current_depth: int):
        """
        Recursively build octree nodes.
        
        Args:
            node: Current node to process
            sparse_points: Sparse points in the scene
            current_depth: Current depth in the octree
        """
        # Find points within this node
        points_in_node = self._find_points_in_node(node, sparse_points)
        node.sparse_points = points_in_node
        
        # Stop conditions
        if (current_depth >= self.config.max_depth or
            len(points_in_node) < self.config.sparse_point_threshold):
            return
        
        # Check if node should be subdivided based on point density
        if self._should_subdivide(node, points_in_node):
            # Subdivide node
            children = node.subdivide()
            
            # Recursively build children
            for child in children:
                self._build_recursive(child, sparse_points, current_depth + 1)
    
    def _find_points_in_node(self, node: OctreeNode, points: np.ndarray) -> list[np.ndarray]:
        """Find points within a node's bounding box."""
        if len(points) == 0:
            return []
        
        # Check which points are within the node's AABB
        points_in_node = []
        for point in points:
            if node.contains_point(point):
                points_in_node.append(point)
        
        return points_in_node
    
    def _should_subdivide(self, node: OctreeNode, points_in_node: list[np.ndarray]) -> bool:
        """
        Determine if a node should be subdivided.
        
        Args:
            node: Node to check
            points_in_node: Points within the node
            
        Returns:
            True if node should be subdivided
        """
        # Don't subdivide if too few points
        if len(points_in_node) < self.config.sparse_point_threshold:
            return False
        
        # Check if GSD is still too large
        if node.gsd > self.config.min_gsd:
            return True
        
        # Check point distribution (subdivide if points are spread out)
        if len(points_in_node) > 2:
            points_array = np.array(points_in_node)
            point_spread = np.std(points_array, axis=0)
            avg_spread = np.mean(point_spread)
            
            # Subdivide if points are spread across more than 50% of node size
            if avg_spread > node.size * 0.25:
                return True
        
        return False

class OctreePruner:
    """
    Pruner for removing unnecessary octree nodes.
    """
    
    def __init__(self, config: InfNeRFConfig):
        """
        Initialize octree pruner.
        
        Args:
            config: InfNeRF configuration
        """
        self.config = config
    
    def prune_octree(self, root_node: OctreeNode) -> dict[str, Any]:
        """
        Prune octree by removing nodes with insufficient data.
        
        Args:
            root_node: Root node of the octree
            
        Returns:
            Statistics about the pruning process
        """
        # Count nodes before pruning
        nodes_before = self._count_nodes(root_node)
        
        # Perform pruning
        pruned_count = self._prune_recursive(root_node)
        
        # Count nodes after pruning
        nodes_after = self._count_nodes(root_node)
        
        stats = {
            'nodes_before': nodes_before, 'nodes_after': nodes_after, 'nodes_pruned': pruned_count, 'compression_ratio': nodes_after / nodes_before if nodes_before > 0 else 1.0
        }
        
        return stats
    
    def _prune_recursive(self, node: OctreeNode) -> int:
        """
        Recursively prune octree nodes.
        
        Args:
            node: Current node to process
            
        Returns:
            Number of nodes pruned
        """
        pruned_count = 0
        
        if node.is_leaf:
            # Check if leaf node should be pruned
            if self._should_prune_leaf(node):
                node.is_pruned = True
                pruned_count += 1
        else:
            # Recursively process children
            for i, child in enumerate(node.children):
                if child is not None:
                    pruned_count += self._prune_recursive(child)
                    
                    # Remove child if it's pruned
                    if child.is_pruned:
                        node.children[i] = None
            
            # Check if all children are pruned/None
            if all(child is None or child.is_pruned for child in node.children):
                # Convert to leaf if all children are gone
                node.is_leaf = True
                node.children = [None] * 8
                
                # Check if this node should also be pruned
                if self._should_prune_leaf(node):
                    node.is_pruned = True
                    pruned_count += 1
        
        return pruned_count
    
    def _should_prune_leaf(self, node: OctreeNode) -> bool:
        """
        Determine if a leaf node should be pruned.
        
        Args:
            node: Leaf node to check
            
        Returns:
            True if node should be pruned
        """
        # Don't prune root node
        if node.level == 0:
            return False
        
        # Prune if no sparse points
        if len(node.sparse_points) == 0:
            return True
        
        # Prune if too few sparse points
        if len(node.sparse_points) < self.config.sparse_point_threshold // 2:
            return True
        
        return False
    
    def _count_nodes(self, node: OctreeNode) -> int:
        """Count total number of nodes in octree."""
        if node.is_pruned:
            return 0
        
        count = 1  # Current node
        
        if not node.is_leaf:
            for child in node.children:
                if child is not None:
                    count += self._count_nodes(child)
        
        return count

def calculate_gsd(node_size: float, grid_size: int) -> float:
    """
    Calculate Ground Sampling Distance for a node.
    
    Args:
        node_size: Size of the node (cube edge length)
        grid_size: Grid resolution for the node
        
    Returns:
        Ground Sampling Distance in meters
    """
    return node_size / grid_size

def visualize_octree(
    root_node: OctreeNode,
    max_depth: Optional[int] = None,
    show_sparse_points: bool = True,
    save_path: Optional[str] = None,
) -> None:
    """
    Visualize octree structure.
    
    Args:
        root_node: Root node of the octree
        max_depth: Maximum depth to visualize (None for all)
        show_sparse_points: Whether to show sparse points
        save_path: Path to save the figure (optional)
    """
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Collect nodes to visualize
    nodes_to_plot = []
    _collect_nodes_recursive(root_node, nodes_to_plot, max_depth)
    
    # Plot nodes as wireframe cubes
    colors = plt.cm.viridis(np.linspace(0, 1, len(nodes_to_plot)))
    
    for i, node in enumerate(nodes_to_plot):
        if node.is_pruned:
            continue
            
        # Get node bounds
        min_bounds, max_bounds = node.get_aabb()
        
        # Create wireframe cube
        _plot_cube_wireframe(
            ax,
            min_bounds,
            max_bounds,
            color=colors[i],
            alpha=0.3,
            linewidth=2 if node.is_leaf else 1,
        )
        
        # Plot sparse points if requested
        if show_sparse_points and len(node.sparse_points) > 0:
            points = np.array(node.sparse_points)
            ax.scatter(points[:, 0], points[:, 1], points[:, 2], c='red', s=1, alpha=0.6)
    
    # Set labels and title
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title(f'InfNeRF Octree Visualization\n'
                f'Nodes: {len(nodes_to_plot)}, Max Depth: {max_depth or "All"}')
    
    # Equal aspect ratio
    ax.set_box_aspect([1, 1, 1])
    
    # Save or show
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Octree visualization saved to: {save_path}")
    else:
        plt.show()
    
    plt.close()

def _collect_nodes_recursive(
    node: OctreeNode,
    nodes_list: list[OctreeNode],
    max_depth: Optional[int],
) -> None:
    """Recursively collect nodes for visualization."""
    if max_depth is not None and node.level > max_depth:
        return
    
    nodes_list.append(node)
    
    if not node.is_leaf:
        for child in node.children:
            if child is not None:
                _collect_nodes_recursive(child, nodes_list, max_depth)

def _plot_cube_wireframe(
    ax,
    min_bounds: np.ndarray,
    max_bounds: np.ndarray,
    color='blue',
    alpha=0.3,
    linewidth=1,
) -> None:
    """Plot a wireframe cube."""
    # Define cube vertices
    vertices = [
        [min_bounds[0], min_bounds[1], min_bounds[2]], [max_bounds[0], min_bounds[1], min_bounds[2]], [max_bounds[0], max_bounds[1], min_bounds[2]], [min_bounds[0], max_bounds[1], min_bounds[2]], [min_bounds[0], min_bounds[1], max_bounds[2]], [max_bounds[0], min_bounds[1], max_bounds[2]], [max_bounds[0], max_bounds[1], max_bounds[2]], [min_bounds[0], max_bounds[1], max_bounds[2]]
    ]
    
    # Define cube edges
    edges = [
        [0, 1], [1, 2], [2, 3], [3, 0], # Bottom face
        [4, 5], [5, 6], [6, 7], [7, 4], # Top face
        [0, 4], [1, 5], [2, 6], [3, 7]   # Vertical edges
    ]
    
    # Plot edges
    for edge in edges:
        points = np.array([vertices[edge[0]], vertices[edge[1]]])
        ax.plot3D(
            points[:,
            0],
            points[:,
            1],
            points[:,
            2],
            color=color,
            alpha=alpha,
            linewidth=linewidth,
        )

def analyze_octree_memory(root_node: OctreeNode) -> dict[str, Any]:
    """
    Analyze memory usage of octree structure.
    
    Args:
        root_node: Root node of the octree
        
    Returns:
        Dictionary with memory analysis
    """
    analysis = {
        'total_nodes': 0, 'leaf_nodes': 0, 'internal_nodes': 0, 'pruned_nodes': 0, 'nodes_by_level': {
        }
    }
    
    def analyze_recursive(node: OctreeNode):
        # Count nodes
        analysis['total_nodes'] += 1
        
        if node.is_pruned:
            analysis['pruned_nodes'] += 1
            return
        
        if node.is_leaf:
            analysis['leaf_nodes'] += 1
        else:
            analysis['internal_nodes'] += 1
        
        # Count by level
        level = node.level
        if level not in analysis['nodes_by_level']:
            analysis['nodes_by_level'][level] = 0
            analysis['memory_by_level'][level] = 0.0
        
        analysis['nodes_by_level'][level] += 1
        
        # Memory usage
        node_memory = node.get_memory_size() / (1024 * 1024)  # MB
        analysis['memory_by_level'][level] += node_memory
        analysis['total_memory_mb'] += node_memory
        
        # Recurse to children
        if not node.is_leaf:
            for child in node.children:
                if child is not None:
                    analyze_recursive(child)
    
    analyze_recursive(root_node)
    
    # Calculate averages
    if analysis['total_nodes'] > 0:
        analysis['average_memory_per_node_mb'] = (
            analysis['total_memory_mb'] / analysis['total_nodes']
        )
    
    return analysis

def export_octree_structure(root_node: OctreeNode, export_path: str):
    """
    Export octree structure to JSON file.
    
    Args:
        root_node: Root node of the octree
        export_path: Path to save the JSON file
    """
    def node_to_dict(node: OctreeNode) -> dict[str, Any]:
        node_dict = {
            'center': node.center.tolist(
            )
        }
        
        if not node.is_leaf:
            node_dict['children'] = []
            for child in node.children:
                if child is not None:
                    node_dict['children'].append(node_to_dict(child))
                else:
                    node_dict['children'].append(None)
        
        return node_dict
    
    octree_data = {
        'octree_structure': node_to_dict(root_node), 'analysis': analyze_octree_memory(root_node)
    }
    
    with open(export_path, 'w') as f:
        json.dump(octree_data, f, indent=2)
    
    print(f"Octree structure exported to: {export_path}") 