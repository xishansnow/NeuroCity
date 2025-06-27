from typing import Optional, Union
"""
Geometry processing utilities for DNMP.

This module provides functions for geometric computations, transformations, and spatial operations.
"""

import torch
import numpy as np

def rodrigues_rotation_matrix(axis: torch.Tensor, angle: torch.Tensor) -> torch.Tensor:
    """
    Compute rotation matrix using Rodrigues' formula.
    
    Args:
        axis: Rotation axis [3] (normalized)
        angle: Rotation angle in radians
        
    Returns:
        rotation_matrix: 3x3 rotation matrix
    """
    device = axis.device
    
    # Ensure axis is normalized
    axis = axis / torch.norm(axis)
    
    # Rodrigues' formula
    K = torch.tensor([
        [0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]
    ], device=device)
    
    I = torch.eye(3, device=device)
    R = I + torch.sin(angle) * K + (1 - torch.cos(angle)) * torch.mm(K, K)
    
    return R

def look_at_matrix(eye: torch.Tensor, target: torch.Tensor, up: torch.Tensor) -> torch.Tensor:
    """
    Compute look-at view matrix.
    
    Args:
        eye: Camera position [3]
        target: Target position [3]
        up: Up vector [3]
        
    Returns:
        view_matrix: 4x4 view transformation matrix
    """
    device = eye.device
    
    # Compute camera coordinate system
    forward = F.normalize(target - eye, dim=0)
    right = F.normalize(torch.cross(forward, up), dim=0)
    up_new = torch.cross(right, forward)
    
    # Build view matrix
    view_matrix = torch.zeros(4, 4, device=device)
    view_matrix[0, :3] = right
    view_matrix[1, :3] = up_new
    view_matrix[2, :3] = -forward
    view_matrix[:3, 3] = -torch.stack([
        torch.dot(right, eye), torch.dot(up_new, eye), torch.dot(-forward, eye)
    ])
    view_matrix[3, 3] = 1.0
    
    return view_matrix

def perspective_projection_matrix(
    fov: float,
    aspect: float,
    near: float,
    far: float
):
    """
    Compute perspective projection matrix.
    
    Args:
        fov: Field of view in radians
        aspect: Aspect ratio (width/height)
        near: Near clipping plane
        far: Far clipping plane
        
    Returns:
        projection_matrix: 4x4 projection matrix
    """
    f = 1.0 / torch.tan(fov / 2.0)
    
    proj_matrix = torch.zeros(4, 4)
    proj_matrix[0, 0] = f / aspect
    proj_matrix[1, 1] = f
    proj_matrix[2, 2] = (far + near) / (near - far)
    proj_matrix[2, 3] = (2 * far * near) / (near - far)
    proj_matrix[3, 2] = -1.0
    
    return proj_matrix

def transform_points(points: torch.Tensor, transformation: torch.Tensor) -> torch.Tensor:
    """
    Transform points using 4x4 transformation matrix.
    
    Args:
        points: Points [N, 3]
        transformation: 4x4 transformation matrix
        
    Returns:
        transformed_points: Transformed points [N, 3]
    """
    # Convert to homogeneous coordinates
    points_homo = torch.cat([points, torch.ones_like(points[:, 0:1])], dim=1)
    
    # Apply transformation
    transformed_homo = points_homo @ transformation.T
    
    # Convert back to 3D coordinates
    transformed_points = transformed_homo[:, :3] / transformed_homo[:, 3:4]
    
    return transformed_points

def compute_bounding_box(points: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Compute axis-aligned bounding box.
    
    Args:
        points: Points [N, 3]
        
    Returns:
        min_bounds: Minimum coordinates [3]
        max_bounds: Maximum coordinates [3]
    """
    min_bounds = points.min(dim=0)[0]
    max_bounds = points.max(dim=0)[0]
    
    return min_bounds, max_bounds

def compute_oriented_bounding_box(points: torch.Tensor) -> dict[str, torch.Tensor]:
    """
    Compute oriented bounding box using PCA.
    
    Args:
        points: Points [N, 3]
        
    Returns:
        Dictionary containing OBB information
    """
    device = points.device
    
    # Center points
    center = points.mean(dim=0)
    centered_points = points - center
    
    # Compute covariance matrix
    cov_matrix = torch.mm(centered_points.T, centered_points) / (len(points) - 1)
    
    # Eigendecomposition
    eigenvalues, eigenvectors = torch.linalg.eigh(cov_matrix)
    
    # Sort by eigenvalues (descending)
    sorted_indices = torch.argsort(eigenvalues, descending=True)
    eigenvalues = eigenvalues[sorted_indices]
    eigenvectors = eigenvectors[:, sorted_indices]
    
    # Project points onto principal axes
    projected_points = torch.mm(centered_points, eigenvectors)
    
    # Compute extents
    min_proj = projected_points.min(dim=0)[0]
    max_proj = projected_points.max(dim=0)[0]
    extents = max_proj - min_proj
    
    return {
        'center': center, 'axes': eigenvectors, 'extents': extents, 'eigenvalues': eigenvalues
    }

def point_in_triangle(point: torch.Tensor, triangle: torch.Tensor) -> torch.Tensor:
    """
    Check if 2D point is inside triangle using barycentric coordinates.
    
    Args:
        point: Point coordinates [2] or [N, 2]
        triangle: Triangle vertices [3, 2]
        
    Returns:
        inside: Boolean indicating if point is inside triangle
    """
    if point.dim() == 1:
        point = point.unsqueeze(0)
        squeeze_output = True
    else:
        squeeze_output = False
    
    v0, v1, v2 = triangle[0], triangle[1], triangle[2]
    
    # Compute barycentric coordinates
    v0v1 = v1 - v0
    v0v2 = v2 - v0
    v0p = point - v0.unsqueeze(0)
    
    dot00 = torch.sum(v0v2 * v0v2)
    dot01 = torch.sum(v0v2 * v0v1)
    dot02 = torch.sum(v0v2.unsqueeze(0) * v0p, dim=1)
    dot11 = torch.sum(v0v1 * v0v1)
    dot12 = torch.sum(v0v1.unsqueeze(0) * v0p, dim=1)
    
    # Compute barycentric coordinates
    inv_denom = 1.0 / (dot00 * dot11 - dot01 * dot01)
    u = (dot11 * dot02 - dot01 * dot12) * inv_denom
    v = (dot00 * dot12 - dot01 * dot02) * inv_denom
    
    # Check if point is inside triangle
    inside = (u >= 0) & (v >= 0) & (u + v <= 1)
    
    if squeeze_output:
        return inside.squeeze(0)
    else:
        return inside

def ray_triangle_intersection(
    ray_origin: torch.Tensor,
    ray_direction: torch.Tensor,
    triangle: torch.Tensor,
    epsilon: float = 1e-8
):
    """
    Compute ray-triangle intersection using Möller-Trumbore algorithm.
    
    Args:
        ray_origin: Ray origin [3]
        ray_direction: Ray direction [3] (normalized)
        triangle: Triangle vertices [3, 3]
        epsilon: Small value for numerical stability
        
    Returns:
        Dictionary containing intersection information
    """
    device = ray_origin.device
    
    v0, v1, v2 = triangle[0], triangle[1], triangle[2]
    
    # Compute triangle edges
    edge1 = v1 - v0
    edge2 = v2 - v0
    
    # Compute determinant
    h = torch.cross(ray_direction, edge2)
    det = torch.dot(edge1, h)
    
    # Check if ray is parallel to triangle
    if torch.abs(det) < epsilon:
        return {'intersects': False, 'distance': torch.inf, 'barycentric': torch.zeros(3)}
    
    inv_det = 1.0 / det
    s = ray_origin - v0
    u = inv_det * torch.dot(s, h)
    
    if u < 0.0 or u > 1.0:
        return {'intersects': False, 'distance': torch.inf, 'barycentric': torch.zeros(3)}
    
    q = torch.cross(s, edge1)
    v = inv_det * torch.dot(ray_direction, q)
    
    if v < 0.0 or u + v > 1.0:
        return {'intersects': False, 'distance': torch.inf, 'barycentric': torch.zeros(3)}
    
    # Compute intersection distance
    t = inv_det * torch.dot(edge2, q)
    
    if t > epsilon:
        # Ray intersects triangle
        w = 1.0 - u - v
        barycentric = torch.tensor([w, u, v], device=device)
        return {'intersects': True, 'distance': t, 'barycentric': barycentric}
    else:
        # Intersection is behind ray origin
        return {'intersects': False, 'distance': torch.inf, 'barycentric': torch.zeros(3)}

def sphere_ray_intersection(
    ray_origin: torch.Tensor,
    ray_direction: torch.Tensor,
    sphere_center: torch.Tensor,
    sphere_radius: float
):
    """
    Compute ray-sphere intersection.
    
    Args:
        ray_origin: Ray origin [3]
        ray_direction: Ray direction [3] (normalized)
        sphere_center: Sphere center [3]
        sphere_radius: Sphere radius
        
    Returns:
        Dictionary containing intersection information
    """
    device = ray_origin.device
    
    # Vector from ray origin to sphere center
    oc = ray_origin - sphere_center
    
    # Quadratic equation coefficients
    a = torch.dot(ray_direction, ray_direction)
    b = 2.0 * torch.dot(oc, ray_direction)
    c = torch.dot(oc, oc) - sphere_radius * sphere_radius
    
    # Discriminant
    discriminant = b * b - 4 * a * c
    
    if discriminant < 0:
        # No intersection
        return {'intersects': False, 'distances': torch.tensor([torch.inf, torch.inf])}
    
    # Compute intersection distances
    sqrt_discriminant = torch.sqrt(discriminant)
    t1 = (-b - sqrt_discriminant) / (2.0 * a)
    t2 = (-b + sqrt_discriminant) / (2.0 * a)
    
    return {'intersects': True, 'distances': torch.tensor([t1, t2])}

def compute_face_normals(vertices: torch.Tensor, faces: torch.Tensor) -> torch.Tensor:
    """
    Compute face normals for triangular mesh.
    
    Args:
        vertices: Mesh vertices [V, 3]
        faces: Mesh faces [F, 3]
        
    Returns:
        face_normals: Face normals [F, 3]
    """
    v0 = vertices[faces[:, 0]]
    v1 = vertices[faces[:, 1]]
    v2 = vertices[faces[:, 2]]
    
    # Compute cross product
    normals = torch.cross(v1 - v0, v2 - v0, dim=1)
    
    # Normalize
    normals = normals / (torch.norm(normals, dim=1, keepdim=True) + 1e-8)
    
    return normals

def chamfer_distance(points1: torch.Tensor, points2: torch.Tensor) -> torch.Tensor:
    """
    Compute Chamfer distance between two point sets.
    
    Args:
        points1: First point set [N, 3]
        points2: Second point set [M, 3]
        
    Returns:
        chamfer_dist: Chamfer distance
    """
    # Compute pairwise distances
    dist_matrix = torch.cdist(points1, points2)
    
    # Find nearest neighbors
    dist1_to_2 = dist_matrix.min(dim=1)[0]  # [N]
    dist2_to_1 = dist_matrix.min(dim=0)[0]  # [M]
    
    # Chamfer distance is sum of both directions
    chamfer_dist = dist1_to_2.mean() + dist2_to_1.mean()
    
    return chamfer_dist 