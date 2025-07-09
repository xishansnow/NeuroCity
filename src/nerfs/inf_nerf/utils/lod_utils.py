from __future__ import annotations

from typing import Optional, Any

"""
Level of Detail (LoD) utilities for InfNeRF.

This module provides tools for:
- Level of Detail management
- Anti-aliasing sampling
- LoD level determination
- Pyramid supervision for multi-scale training
"""

import torch
import torch.nn.functional as F
import numpy as np
import math

from ..core import InfNeRFConfig, OctreeNode


class LoDManager:
    """
    Manager for Level of Detail operations in InfNeRF.
    """

    def __init__(self, config: InfNeRFConfig):
        """
        Initialize LoD manager.

        Args:
            config: InfNeRF configuration
        """
        self.config = config
        self.level_thresholds = self._compute_level_thresholds()

    def _compute_level_thresholds(self) -> list[float]:
        """Compute GSD thresholds for each octree level."""
        thresholds = []

        for level in range(self.config.max_depth + 1):
            # GSD decreases exponentially with depth
            base_gsd = self.config.max_gsd
            level_gsd = base_gsd / (2**level)
            thresholds.append(max(level_gsd, self.config.min_gsd))

        return thresholds

    def determine_lod_level(self, sample_radius: float, max_level: int) -> int:
        """
        Determine appropriate LoD level based on sample radius.

        Args:
            sample_radius: Radius of the sampling sphere
            max_level: Maximum available level in octree

        Returns:
            Appropriate LoD level
        """
        # Find the level where GSD matches the sample radius
        for level in range(min(max_level + 1, len(self.level_thresholds))):
            if self.level_thresholds[level] <= sample_radius:
                return level

        # Return deepest available level if no match
        return min(max_level, len(self.level_thresholds) - 1)

    def get_lod_weights(self, distances: torch.Tensor, focal_length: float) -> torch.Tensor:
        """
        Compute LoD weights based on distance from camera.

        Args:
            distances: [N] distances from camera
            focal_length: Camera focal length

        Returns:
            [N] LoD weights (0=coarse, 1=fine)
        """
        # Closer objects get finer detail
        max_distance = torch.max(distances)
        min_distance = torch.min(distances)

        # Normalize distances
        if max_distance > min_distance:
            normalized_distances = (distances - min_distance) / (max_distance - min_distance)
        else:
            normalized_distances = torch.zeros_like(distances)

        # Invert so closer objects get higher weights (finer detail)
        lod_weights = 1.0 - normalized_distances

        return lod_weights

    def adaptive_sampling_density(self, node: OctreeNode, distance_to_camera: float) -> int:
        """
        Determine adaptive sampling density based on LoD.

        Args:
            node: Octree node
            distance_to_camera: Distance from node to camera

        Returns:
            Number of samples to use for this node
        """
        base_samples = self.config.num_samples

        # Adjust based on node level (finer levels get more samples)
        level_factor = 1.0 + (node.level * 0.2)

        # Adjust based on distance (closer gets more samples)
        distance_factor = 1.0 / (1.0 + distance_to_camera * 0.1)

        adaptive_samples = int(base_samples * level_factor * distance_factor)

        # Clamp to reasonable range
        return max(16, min(adaptive_samples, base_samples * 4))


def anti_aliasing_sampling(
    positions: torch.Tensor,
    radii: torch.Tensor,
    perturbation_strength: float = 0.5,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Apply anti-aliasing sampling with radius perturbation.

    Args:
        positions: [N, 3] sample positions
        radii: [N] sampling radii
        perturbation_strength: Strength of radius perturbation

    Returns:
        perturbed_positions: [N, 3] perturbed positions
        perturbed_radii: [N] perturbed radii
    """
    device = positions.device
    batch_size = positions.shape[0]

    # Perturb radii (Equation 4 in paper)
    if perturbation_strength > 0:
        # Random perturbation in [-0.5, 0.5]
        perturbation = (torch.rand(batch_size, device=device) - 0.5) * perturbation_strength
        perturbed_radii = radii * (2.0**perturbation)
    else:
        perturbed_radii = radii

    # Add small spatial perturbation to positions
    spatial_perturbation = torch.randn_like(positions) * perturbed_radii.unsqueeze(-1) * 0.1
    perturbed_positions = positions + spatial_perturbation

    return perturbed_positions, perturbed_radii


def determine_lod_level(sample_radius: float, root_gsd: float, max_level: int) -> int:
    """
    Determine LoD level based on sample radius (Equation 5 in paper).

    Args:
        sample_radius: Radius of the sampling sphere
        root_gsd: Ground Sampling Distance of root node
        max_level: Maximum available level

    Returns:
        Appropriate LoD level
    """
    if sample_radius <= 0:
        return max_level

    # Calculate level based on GSD ratio (Equation 5)
    level = math.floor(math.log2(root_gsd / sample_radius))

    # Clamp to valid range
    return max(0, min(level, max_level))


def pyramid_supervision(
    images: list[torch.Tensor],
    num_levels: int = 4,
    scale_factor: float = 2.0,
) -> list[list[torch.Tensor]]:
    """
    Create image pyramids for multi-scale supervision.

    Args:
        images: list of input images [H, W, 3]
        num_levels: Number of pyramid levels
        scale_factor: Scale factor between levels

    Returns:
        list of image pyramids, one per input image
    """
    pyramids = []

    for img in images:
        pyramid = [img]  # Original resolution
        current_img = img

        # Build pyramid by successive downsampling
        for level in range(1, num_levels):
            # Calculate new size
            current_h, current_w = current_img.shape[-2:]
            new_h = max(1, int(current_h / scale_factor))
            new_w = max(1, int(current_w / scale_factor))

            # Downsample using bilinear interpolation
            if current_img.dim() == 3:  # [H, W, C]
                current_img = current_img.permute(2, 0, 1).unsqueeze(0)  # [1, C, H, W]
                current_img = F.interpolate(
                    current_img,
                    size=(new_h, new_w),
                )
                current_img = current_img.squeeze(0).permute(1, 2, 0)  # [H, W, C]
            else:  # Already in [C, H, W] format
                current_img = current_img.unsqueeze(0)  # [1, C, H, W]
                current_img = F.interpolate(
                    current_img,
                    size=(new_h, new_w),
                )
                current_img = current_img.squeeze(0)  # [C, H, W]

            pyramid.append(current_img)

        pyramids.append(pyramid)

    return pyramids


def compute_level_consistency_loss(
    parent_node: OctreeNode,
    child_nodes: list[OctreeNode],
    sample_positions: torch.Tensor,
    sample_directions: torch.Tensor,
) -> torch.Tensor:
    """
    Compute level consistency regularization loss.

    Args:
        parent_node: Parent octree node
        child_nodes: list of child nodes
        sample_positions: [N, 3] sample positions
        sample_directions: [N, 3] sample directions

    Returns:
        Consistency loss
    """
    device = sample_positions.device
    consistency_loss = torch.tensor(0.0, device=device)

    try:
        # Get parent predictions
        with torch.no_grad():
            parent_result = parent_node.nerf(sample_positions, sample_directions)
            parent_density = parent_result["density"]
            parent_color = parent_result["color"]

        # Get child predictions and compare
        for child in child_nodes:
            if child is None or child.is_pruned:
                continue

            # Find samples within child's bounds
            child_mask = torch.zeros(len(sample_positions), dtype=torch.bool, device=device)

            for i, pos in enumerate(sample_positions):
                pos_np = pos.detach().cpu().numpy()
                if child.contains_point(pos_np):
                    child_mask[i] = True

            if not child_mask.any():
                continue

            # Get predictions for samples in this child
            child_positions = sample_positions[child_mask]
            child_directions = sample_directions[child_mask]

            child_result = child.nerf(child_positions, child_directions)
            child_density = child_result["density"]
            child_color = child_result["color"]

            # Compare with parent predictions
            parent_density_subset = parent_density[child_mask]
            parent_color_subset = parent_color[child_mask]

            # Density consistency
            density_loss = F.mse_loss(child_density, parent_density_subset.detach())
            consistency_loss += density_loss

            # Color consistency (weighted by density)
            color_weights = torch.sigmoid(child_density.detach())
            color_loss = F.mse_loss(
                child_color * color_weights, parent_color_subset.detach() * color_weights
            )
            consistency_loss += color_loss

    except Exception as e:
        # Skip if there's an error
        pass

    return consistency_loss


def compute_sample_radius(
    depth: torch.Tensor,
    focal_length: float,
    pixel_width: float = 1.0,
) -> torch.Tensor:
    """
    Compute sampling sphere radius based on pixel footprint (Equation 3 in paper).

    Args:
        depth: [N] depth values
        focal_length: Camera focal length
        pixel_width: Pixel width in image coordinates

    Returns:
        [N] sampling radii
    """
    # Radius proportional to depth (Equation 3)
    radii = depth * pixel_width / (2 * focal_length)
    return radii


def frustum_culling(
    octree_nodes: list[OctreeNode],
    camera_origin: np.ndarray,
    view_direction: np.ndarray,
    field_of_view: float,
    near_plane: float,
    far_plane: float,
) -> list[OctreeNode]:
    """
    Perform frustum culling to select visible octree nodes.

    Args:
        octree_nodes: list of octree nodes
        camera_origin: [3] camera position
        view_direction: [3] camera view direction (normalized)
        field_of_view: Field of view in radians
        near_plane: Near clipping plane distance
        far_plane: Far clipping plane distance

    Returns:
        list of visible nodes
    """
    visible_nodes = []

    # Compute frustum parameters
    cos_half_fov = math.cos(field_of_view / 2)

    for node in octree_nodes:
        if node.is_pruned:
            continue

        # Get node center and bounds
        node_center = node.center
        node_radius = node.size * math.sqrt(3) / 2  # Half diagonal of cube

        # Vector from camera to node center
        to_node = node_center - camera_origin
        distance = np.linalg.norm(to_node)

        # Skip if too far or too close
        if distance - node_radius > far_plane or distance + node_radius < near_plane:
            continue

        # Check if within field of view
        if distance > 0:
            to_node_normalized = to_node / distance
            dot_product = np.dot(view_direction, to_node_normalized)

            # Add some tolerance based on node size
            angular_radius = math.atan(node_radius / distance)
            effective_cos_threshold = math.cos(field_of_view / 2 + angular_radius)

            if dot_product >= effective_cos_threshold:
                visible_nodes.append(node)

    return visible_nodes


def adaptive_lod_selection(
    visible_nodes: list[OctreeNode],
    camera_origin: np.ndarray,
    pixel_size_at_distance: callable,
) -> list[OctreeNode]:
    """
    Select appropriate LoD level for each visible node.

    Args:
        visible_nodes: list of visible octree nodes
        camera_origin: [3] camera position
        pixel_size_at_distance: Function to compute pixel size at given distance

    Returns:
        list of nodes with appropriate LoD
    """
    selected_nodes = []

    for node in visible_nodes:
        # Calculate distance to camera
        distance = np.linalg.norm(node.center - camera_origin)

        # Get pixel size at this distance
        pixel_size = pixel_size_at_distance(distance)

        # Check if node's GSD is appropriate for this pixel size
        if node.gsd <= pixel_size * 2:  # Use node if GSD is fine enough
            selected_nodes.append(node)
        elif node.parent is not None and node.parent.gsd > pixel_size * 0.5:
            # Use parent if it's not too coarse
            if node.parent not in selected_nodes:
                selected_nodes.append(node.parent)

    return list(set(selected_nodes))  # Remove duplicates


class MultiScaleRenderer:
    """
    Multi-scale renderer for efficient LoD rendering.
    """

    def __init__(self, config: InfNeRFConfig):
        """
        Initialize multi-scale renderer.

        Args:
            config: InfNeRF configuration
        """
        self.config = config
        self.lod_manager = LoDManager(config)

    def render_with_lod(
        self,
        nodes: list[OctreeNode],
        rays_o: torch.Tensor,
        rays_d: torch.Tensor,
        near: float,
        far: float,
        focal_length: float,
    ) -> dict[str, torch.Tensor]:
        """
        Render rays using adaptive LoD selection.

        Args:
            nodes: Available octree nodes
            rays_o: [N, 3] ray origins
            rays_d: [N, 3] ray directions
            near: Near plane distance
            far: Far plane distance
            focal_length: Camera focal length

        Returns:
            Rendered outputs
        """
        num_rays = rays_o.shape[0]
        device = rays_o.device

        # Sample points along rays
        t_vals = torch.linspace(0.0, 1.0, self.config.num_samples, device=device)
        z_vals = near + (far - near) * t_vals
        z_vals = z_vals.expand(num_rays, self.config.num_samples)

        # Get sample points
        pts = rays_o[..., None, :] + rays_d[..., None, :] * z_vals[..., :, None]

        # Compute sampling radii
        radii = compute_sample_radius(z_vals, focal_length)

        # Collect colors and densities
        colors = torch.zeros(*pts.shape, device=device)
        densities = torch.zeros(pts.shape[:-1] + (1,), device=device)

        for i in range(num_rays):
            for j in range(self.config.num_samples):
                sample_pos = pts[i, j]
                sample_radius = radii[i, j].item()

                # Find appropriate node
                selected_node = self._select_node_for_sample(nodes, sample_pos, sample_radius)

                if selected_node is not None:
                    # Query NeRF
                    result = selected_node.nerf(sample_pos.unsqueeze(0), rays_d[i : i + 1])

                    colors[i, j] = result["color"].squeeze(0)
                    densities[i, j] = result["density"].squeeze(0)

        # Volume rendering
        dists = z_vals[..., 1:] - z_vals[..., :-1]
        dists = torch.cat([dists, torch.full_like(dists[..., :1], 1e10)], dim=-1)

        alpha = 1.0 - torch.exp(-densities[..., 0] * dists)
        T = torch.cumprod(1.0 - alpha + 1e-10, dim=-1)
        T = torch.cat([torch.ones_like(T[..., :1]), T[..., :-1]], dim=-1)

        weights = alpha * T
        rgb = torch.sum(weights[..., None] * colors, dim=-2)
        depth = torch.sum(weights * z_vals, dim=-1)
        acc = torch.sum(weights, dim=-1)

        return {"rgb": rgb, "depth": depth, "acc": acc, "weights": weights}

    def _select_node_for_sample(
        self,
        nodes: list[OctreeNode],
        position: torch.Tensor,
        radius: float,
    ) -> OctreeNode:
        """Select appropriate node for a sample."""
        pos_np = position.detach().cpu().numpy()

        # Find nodes that contain this position
        containing_nodes = []
        for node in nodes:
            if not node.is_pruned and node.contains_point(pos_np):
                containing_nodes.append(node)

        if not containing_nodes:
            return None

        # Select node with appropriate GSD
        best_node = None
        best_gsd_ratio = float("inf")

        for node in containing_nodes:
            gsd_ratio = abs(math.log2(node.gsd / radius)) if radius > 0 else float("inf")
            if gsd_ratio < best_gsd_ratio:
                best_gsd_ratio = gsd_ratio
                best_node = node

        return best_node
