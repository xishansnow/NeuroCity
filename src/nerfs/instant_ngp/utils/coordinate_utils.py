"""
from __future__ import annotations

Coordinate transformation utilities for Instant NGP.

This module provides coordinate transformation functions including
unisphere contraction, spherical coordinates, and other spatial operations.
"""

import torch
import numpy as np


def contract_to_unisphere(positions: torch.Tensor, scene_radius: float = 1.0) -> torch.Tensor:
    """
    Contract infinite coordinates to unit sphere.

    This function maps infinite 3D space to a bounded region, allowing
    the hash encoding to handle unbounded scenes efficiently.

    Args:
        positions: [N, 3] 3D positions
        scene_radius: Radius of the inner sphere (default: 1.0)

    Returns:
        [N, 3] contracted positions
    """
    # Compute distance from origin
    distances = torch.norm(positions, dim=-1, keepdim=True)

    # Points inside the scene radius are unchanged
    mask_inside = distances <= scene_radius

    # Points outside are contracted using the formula:
    # r' = 2 - scene_radius / r
    # This maps [scene_radius, inf) to [scene_radius, 2]
    distances_contracted = torch.where(mask_inside, distances, 2 - scene_radius / distances)

    # Normalize directions and scale by contracted distance
    directions = positions / (distances + 1e-8)  # Avoid division by zero
    contracted_positions = directions * distances_contracted

    return contracted_positions


def uncontract_from_unisphere(
    contracted_positions: torch.Tensor,
    scene_radius: float = 1.0,
) -> torch.Tensor:
    """
    Reverse unisphere contraction to get original coordinates.

    Args:
        contracted_positions: [N, 3] contracted positions
        scene_radius: Radius of the inner sphere

    Returns:
        [N, 3] original positions
    """
    # Compute contracted distances
    distances_contracted = torch.norm(contracted_positions, dim=-1, keepdim=True)

    # Points inside scene radius are unchanged
    mask_inside = distances_contracted <= scene_radius

    # Reverse contraction: r = scene_radius / (2 - r')
    distances_original = torch.where(
        mask_inside, distances_contracted, scene_radius / (2 - distances_contracted)
    )

    # Get directions and scale by original distance
    directions = contracted_positions / (distances_contracted + 1e-8)
    original_positions = directions * distances_original

    return original_positions


def spherical_to_cartesian(spherical: torch.Tensor) -> torch.Tensor:
    """
    Convert spherical coordinates to Cartesian.

    Args:
        spherical: [N, 3] spherical coordinates (r, theta, phi)
                  where theta is azimuthal angle [0, 2π]
                  and phi is polar angle [0, π]

    Returns:
        [N, 3] Cartesian coordinates (x, y, z)
    """
    r = spherical[..., 0]
    theta = spherical[..., 1]  # Azimuthal angle
    phi = spherical[..., 2]  # Polar angle

    # Convert to Cartesian
    x = r * torch.sin(phi) * torch.cos(theta)
    y = r * torch.sin(phi) * torch.sin(theta)
    z = r * torch.cos(phi)

    return torch.stack([x, y, z], dim=-1)


def cartesian_to_spherical(cartesian: torch.Tensor) -> torch.Tensor:
    """
    Convert Cartesian coordinates to spherical.

    Args:
        cartesian: [N, 3] Cartesian coordinates (x, y, z)

    Returns:
        [N, 3] spherical coordinates (r, theta, phi)
    """
    x, y, z = cartesian[..., 0], cartesian[..., 1], cartesian[..., 2]

    # Radius
    r = torch.sqrt(x**2 + y**2 + z**2)

    # Azimuthal angle (0 to 2π)
    theta = torch.atan2(y, x)
    theta = torch.where(theta < 0, theta + 2 * np.pi, theta)

    # Polar angle (0 to π)
    phi = torch.acos(torch.clamp(z / (r + 1e-8), -1 + 1e-8, 1 - 1e-8))

    return torch.stack([r, theta, phi], dim=-1)


def normalize_coordinates(
    positions: torch.Tensor,
    bbox_min: torch.Tensor,
    bbox_max: torch.Tensor,
) -> torch.Tensor:
    """
    Normalize coordinates to [0, 1] range.

    Args:
        positions: [N, 3] positions
        bbox_min: [3] minimum bounds
        bbox_max: [3] maximum bounds

    Returns:
        [N, 3] normalized positions
    """
    bbox_size = bbox_max - bbox_min
    normalized = (positions - bbox_min) / (bbox_size + 1e-8)

    return torch.clamp(normalized, 0, 1)


def denormalize_coordinates(
    normalized_positions: torch.Tensor,
    bbox_min: torch.Tensor,
    bbox_max: torch.Tensor,
) -> torch.Tensor:
    """
    Denormalize coordinates from [0, 1] to original range.

    Args:
        normalized_positions: [N, 3] normalized positions
        bbox_min: [3] minimum bounds
        bbox_max: [3] maximum bounds

    Returns:
        [N, 3] denormalized positions
    """
    bbox_size = bbox_max - bbox_min
    positions = normalized_positions * bbox_size + bbox_min

    return positions


def apply_pose_transform(positions: torch.Tensor, pose: torch.Tensor) -> torch.Tensor:
    """
    Apply pose transformation to positions.

    Args:
        positions: [N, 3] positions
        pose: [4, 4] transformation matrix

    Returns:
        [N, 3] transformed positions
    """
    # Convert to homogeneous coordinates
    ones = torch.ones(positions.shape[0], 1, device=positions.device)
    positions_homo = torch.cat([positions, ones], dim=1)

    # Apply transformation
    transformed_homo = (pose @ positions_homo.T).T

    # Convert back to 3D
    transformed = transformed_homo[:, :3]

    return transformed


def rotate_points(points: torch.Tensor, rotation_matrix: torch.Tensor) -> torch.Tensor:
    """
    Rotate points using rotation matrix.

    Args:
        points: [N, 3] points
        rotation_matrix: [3, 3] rotation matrix

    Returns:
        [N, 3] rotated points
    """
    return (rotation_matrix @ points.T).T


def compute_scene_bounds(
    positions: torch.Tensor,
    percentile: float = 95.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Compute scene bounds from positions.

    Args:
        positions: [N, 3] positions
        percentile: Percentile for robust bound estimation

    Returns:
        tuple of (bbox_min, bbox_max)
    """
    # Use percentiles for robust bound estimation
    bbox_min = torch.quantile(positions, (100 - percentile) / 100, dim=0)
    bbox_max = torch.quantile(positions, percentile / 100, dim=0)

    return bbox_min, bbox_max


def generate_random_directions(num_directions: int, device: torch.device = None) -> torch.Tensor:
    """
    Generate uniformly distributed random directions on unit sphere.

    Args:
        num_directions: Number of directions to generate
        device: Device to place tensor on

    Returns:
        [num_directions, 3] unit direction vectors
    """
    if device is None:
        device = torch.device("cpu")

    # Generate random points on sphere using normal distribution
    directions = torch.randn(num_directions, 3, device=device)

    # Normalize to unit sphere
    directions = directions / torch.norm(directions, dim=-1, keepdim=True)

    return directions


def fibonacci_sphere(num_points: int, device: torch.device = None) -> torch.Tensor:
    """
    Generate evenly distributed points on sphere using Fibonacci spiral.

    Args:
        num_points: Number of points to generate
        device: Device to place tensor on

    Returns:
        [num_points, 3] points on unit sphere
    """
    if device is None:
        device = torch.device("cpu")

    indices = torch.arange(0, num_points, dtype=torch.float32, device=device)

    # Golden angle
    golden_angle = np.pi * (3 - np.sqrt(5))

    # Spherical coordinates
    theta = golden_angle * indices  # Azimuthal angle
    y = 1 - (indices / (num_points - 1)) * 2  # y goes from 1 to -1
    radius = torch.sqrt(1 - y * y)

    # Convert to Cartesian
    x = torch.cos(theta) * radius
    z = torch.sin(theta) * radius

    return torch.stack([x, y, z], dim=-1)


def compute_viewing_direction(camera_pos: torch.Tensor, world_pos: torch.Tensor) -> torch.Tensor:
    """
    Compute viewing direction from camera to world position.

    Args:
        camera_pos: [3] or [N, 3] camera position(s)
        world_pos: [N, 3] world positions

    Returns:
        [N, 3] normalized viewing directions
    """
    # Compute direction vectors
    if camera_pos.dim() == 1:
        camera_pos = camera_pos.unsqueeze(0)

    directions = world_pos - camera_pos

    # Normalize
    directions = directions / torch.norm(directions, dim=-1, keepdim=True)

    return directions


def sample_hemisphere(
    num_samples: int,
    normal: torch.Tensor,
    device: torch.device = None,
) -> torch.Tensor:
    """
    Sample directions from hemisphere around given normal.

    Args:
        num_samples: Number of samples
        normal: [3] normal vector
        device: Device to place tensor on

    Returns:
        [num_samples, 3] hemisphere directions
    """
    if device is None:
        device = normal.device

    # Generate random directions on sphere
    directions = generate_random_directions(num_samples, device)

    # Flip directions that point away from normal
    dot_products = torch.sum(directions * normal, dim=-1, keepdim=True)
    directions = torch.where(dot_products < 0, -directions, directions)

    return directions


def coordinate_grid_3d(
    resolution: int,
    bbox_min: torch.Tensor = None,
    bbox_max: torch.Tensor = None,
    device: torch.device = None,
) -> torch.Tensor:
    """
    Generate 3D coordinate grid.

    Args:
        resolution: Grid resolution
        bbox_min: [3] minimum bounds (default: [-1, -1, -1])
        bbox_max: [3] maximum bounds (default: [1, 1, 1])
        device: Device to place tensor on

    Returns:
        [resolution^3, 3] grid coordinates
    """
    if device is None:
        device = torch.device("cpu")

    if bbox_min is None:
        bbox_min = torch.tensor([-1, -1, -1], device=device)
    if bbox_max is None:
        bbox_max = torch.tensor([1, 1, 1], device=device)

    # Create 1D coordinate arrays
    x = torch.linspace(bbox_min[0], bbox_max[0], resolution, device=device)
    y = torch.linspace(bbox_min[1], bbox_max[1], resolution, device=device)
    z = torch.linspace(bbox_min[2], bbox_max[2], resolution, device=device)

    # Create 3D meshgrid
    X, Y, Z = torch.meshgrid(x, y, z, indexing="ij")

    # Flatten and stack
    coords = torch.stack([X.flatten(), Y.flatten(), Z.flatten()], dim=-1)

    return coords


def test_coordinate_transforms() -> None:
    """Test coordinate transformation functions."""
    print("Testing coordinate transformations...")

    # Test unisphere contraction
    positions = torch.tensor(
        [
            [0.5, 0.5, 0.5],  # Inside sphere
            [2.0, 0.0, 0.0],  # Outside sphere
            [0.0, 3.0, 4.0],  # Far outside
        ]
    )

    contracted = contract_to_unisphere(positions)
    uncontracted = uncontract_from_unisphere(contracted)

    # Check if contraction is approximately reversible
    contraction_error = torch.mean(torch.abs(positions - uncontracted))
    print(f"Contraction reversibility error: {contraction_error:.6f}")

    # Test spherical coordinates
    cartesian = torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])

    spherical = cartesian_to_spherical(cartesian)
    cartesian_back = spherical_to_cartesian(spherical)

    spherical_error = torch.mean(torch.abs(cartesian - cartesian_back))
    print(f"Spherical coordinate error: {spherical_error:.6f}")

    # Test direction generation
    directions = generate_random_directions(100)
    norms = torch.norm(directions, dim=-1)

    # Check if directions are unit vectors
    unit_vector_error = torch.mean(torch.abs(norms - 1.0))
    print(f"Unit vector error: {unit_vector_error:.6f}")

    print("Coordinate transformation tests completed!")


if __name__ == "__main__":
    test_coordinate_transforms()
