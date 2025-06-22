"""
Camera utilities for Nerfacto.

This module provides utility functions for camera operations, ray generation,
and camera pose transformations.
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import Tuple, List, Optional, Dict, Union
import math


def generate_rays(
    camera_poses: torch.Tensor,
    camera_intrinsics: torch.Tensor,
    image_height: int,
    image_width: int,
    device: str = 'cuda'
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Generate rays from camera parameters.
    
    Args:
        camera_poses: Camera poses [N, 4, 4] (camera to world transform)
        camera_intrinsics: Camera intrinsics [N, 4] (fx, fy, cx, cy)
        image_height: Image height
        image_width: Image width
        device: Device to use
        
    Returns:
        Tuple of (ray_origins, ray_directions) each [N, H, W, 3]
    """
    N = camera_poses.shape[0]
    
    # Create pixel coordinates
    i, j = torch.meshgrid(
        torch.arange(image_width, device=device),
        torch.arange(image_height, device=device),
        indexing='ij'
    )
    i = i.t().float()  # [H, W]
    j = j.t().float()  # [H, W]
    
    ray_origins_list = []
    ray_directions_list = []
    
    for n in range(N):
        # Get intrinsics
        fx, fy, cx, cy = camera_intrinsics[n]
        
        # Convert pixel coordinates to camera coordinates
        x = (i - cx) / fx
        y = (j - cy) / fy
        z = torch.ones_like(x)
        
        # Ray directions in camera coordinates
        directions = torch.stack([x, y, z], dim=-1)  # [H, W, 3]
        
        # Normalize directions
        directions = directions / torch.norm(directions, dim=-1, keepdim=True)
        
        # Transform to world coordinates
        c2w = camera_poses[n]  # [4, 4]
        R = c2w[:3, :3]  # [3, 3]
        T = c2w[:3, 3]   # [3]
        
        # Ray directions in world coordinates
        world_directions = torch.sum(directions[..., None, :] * R, dim=-1)
        
        # Ray origins (camera center in world coordinates)
        world_origins = T.expand_as(world_directions)
        
        ray_origins_list.append(world_origins)
        ray_directions_list.append(world_directions)
    
    ray_origins = torch.stack(ray_origins_list, dim=0)
    ray_directions = torch.stack(ray_directions_list, dim=0)
    
    return ray_origins, ray_directions


def sample_rays_uniform(
    ray_origins: torch.Tensor,
    ray_directions: torch.Tensor,
    num_rays: int,
    device: str = 'cuda'
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Uniformly sample rays from the given ray batch.
    
    Args:
        ray_origins: Ray origins [N, H, W, 3]
        ray_directions: Ray directions [N, H, W, 3]
        num_rays: Number of rays to sample
        device: Device to use
        
    Returns:
        Tuple of sampled (ray_origins, ray_directions)
    """
    N, H, W = ray_origins.shape[:3]
    
    # Flatten rays
    origins_flat = ray_origins.reshape(-1, 3)  # [N*H*W, 3]
    directions_flat = ray_directions.reshape(-1, 3)  # [N*H*W, 3]
    
    # Sample random indices
    total_rays = N * H * W
    indices = torch.randperm(total_rays, device=device)[:num_rays]
    
    # Sample rays
    sampled_origins = origins_flat[indices]
    sampled_directions = directions_flat[indices]
    
    return sampled_origins, sampled_directions


def get_camera_frustum(
    pose: torch.Tensor,
    intrinsics: torch.Tensor,
    image_height: int,
    image_width: int,
    depth: float = 1.0
) -> torch.Tensor:
    """
    Get camera frustum corners in world coordinates.
    
    Args:
        pose: Camera pose [4, 4]
        intrinsics: Camera intrinsics [4] (fx, fy, cx, cy)
        image_height: Image height
        image_width: Image width
        depth: Frustum depth
        
    Returns:
        Frustum corners [8, 3]
    """
    fx, fy, cx, cy = intrinsics
    
    # Image corners in pixel coordinates
    corners_2d = torch.tensor([
        [0, 0],
        [image_width, 0],
        [image_width, image_height],
        [0, image_height]
    ], dtype=torch.float32, device=pose.device)
    
    # Convert to camera coordinates
    corners_3d_near = []
    corners_3d_far = []
    
    for corner in corners_2d:
        x = (corner[0] - cx) / fx
        y = (corner[1] - cy) / fy
        
        # Near and far points
        near_point = torch.tensor([x * 0.1, y * 0.1, 0.1], device=pose.device)
        far_point = torch.tensor([x * depth, y * depth, depth], device=pose.device)
        
        corners_3d_near.append(near_point)
        corners_3d_far.append(far_point)
    
    # Stack all corners
    corners_3d = torch.stack(corners_3d_near + corners_3d_far, dim=0)  # [8, 3]
    
    # Transform to world coordinates
    R = pose[:3, :3]
    T = pose[:3, 3]
    
    world_corners = torch.matmul(corners_3d, R.t()) + T
    
    return world_corners


def convert_poses_to_nerfstudio(poses: torch.Tensor) -> torch.Tensor:
    """
    Convert poses from OpenCV/COLMAP convention to Nerfstudio convention.
    
    Args:
        poses: Camera poses [N, 4, 4] in OpenCV convention
        
    Returns:
        Poses in Nerfstudio convention [N, 4, 4]
    """
    # Nerfstudio uses: x right, y up, z back
    # OpenCV uses: x right, y down, z forward
    
    # Create conversion matrix
    conversion = torch.tensor([
        [1, 0, 0, 0],
        [0, -1, 0, 0],
        [0, 0, -1, 0],
        [0, 0, 0, 1]
    ], dtype=poses.dtype, device=poses.device)
    
    # Apply conversion
    converted_poses = torch.matmul(poses, conversion)
    
    return converted_poses


def compute_camera_distances(poses: torch.Tensor) -> torch.Tensor:
    """
    Compute distances between all camera pairs.
    
    Args:
        poses: Camera poses [N, 4, 4]
        
    Returns:
        Distance matrix [N, N]
    """
    # Extract camera positions
    positions = poses[:, :3, 3]  # [N, 3]
    
    # Compute pairwise distances
    distances = torch.cdist(positions, positions, p=2)
    
    return distances


def get_nearest_cameras(
    query_pose: torch.Tensor,
    reference_poses: torch.Tensor,
    k: int = 5
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Find k nearest cameras to a query camera.
    
    Args:
        query_pose: Query camera pose [4, 4]
        reference_poses: Reference camera poses [N, 4, 4]
        k: Number of nearest cameras to find
        
    Returns:
        Tuple of (nearest_indices, distances)
    """
    query_pos = query_pose[:3, 3]  # [3]
    ref_positions = reference_poses[:, :3, 3]  # [N, 3]
    
    # Compute distances
    distances = torch.norm(ref_positions - query_pos, dim=1)
    
    # Find k nearest
    nearest_indices = torch.topk(distances, k, largest=False).indices
    nearest_distances = distances[nearest_indices]
    
    return nearest_indices, nearest_distances


def interpolate_poses(
    pose1: torch.Tensor,
    pose2: torch.Tensor,
    t: float
) -> torch.Tensor:
    """
    Interpolate between two camera poses.
    
    Args:
        pose1: First pose [4, 4]
        pose2: Second pose [4, 4]
        t: Interpolation parameter [0, 1]
        
    Returns:
        Interpolated pose [4, 4]
    """
    # Linear interpolation for translation
    trans1 = pose1[:3, 3]
    trans2 = pose2[:3, 3]
    interp_trans = (1 - t) * trans1 + t * trans2
    
    # SLERP for rotation
    R1 = pose1[:3, :3]
    R2 = pose2[:3, :3]
    
    # Convert to quaternions (simplified)
    # In practice, use proper SLERP implementation
    interp_R = (1 - t) * R1 + t * R2
    
    # Orthogonalize rotation matrix
    U, _, V = torch.svd(interp_R)
    interp_R = torch.matmul(U, V.transpose(-2, -1))
    
    # Construct interpolated pose
    interp_pose = torch.eye(4, device=pose1.device, dtype=pose1.dtype)
    interp_pose[:3, :3] = interp_R
    interp_pose[:3, 3] = interp_trans
    
    return interp_pose


def create_spiral_path(
    center: torch.Tensor,
    radius: float,
    height: float,
    num_views: int,
    device: str = 'cuda'
) -> torch.Tensor:
    """
    Create spiral camera path for rendering.
    
    Args:
        center: Scene center [3]
        radius: Spiral radius
        height: Height variation
        num_views: Number of views
        device: Device to use
        
    Returns:
        Camera poses [num_views, 4, 4]
    """
    poses = []
    
    for i in range(num_views):
        theta = 2 * math.pi * i / num_views
        
        # Spiral position
        x = center[0] + radius * math.cos(theta)
        y = center[1] + height * math.sin(4 * theta) / 4
        z = center[2] + radius * math.sin(theta)
        
        pos = torch.tensor([x, y, z], device=device)
        
        # Look at center
        look_dir = center - pos
        look_dir = look_dir / torch.norm(look_dir)
        
        # Up vector
        up = torch.tensor([0, 1, 0], device=device, dtype=torch.float32)
        
        # Right vector
        right = torch.cross(look_dir, up)
        right = right / torch.norm(right)
        
        # Recalculate up
        up = torch.cross(right, look_dir)
        
        # Create pose matrix
        pose = torch.eye(4, device=device)
        pose[:3, 0] = right
        pose[:3, 1] = up
        pose[:3, 2] = -look_dir  # Forward is -z
        pose[:3, 3] = pos
        
        poses.append(pose)
    
    return torch.stack(poses, dim=0)


def compute_camera_rays_directions(
    camera_intrinsics: torch.Tensor,
    image_height: int,
    image_width: int,
    device: str = 'cuda'
) -> torch.Tensor:
    """
    Compute ray directions for all pixels in camera coordinate system.
    
    Args:
        camera_intrinsics: Camera intrinsics [4] (fx, fy, cx, cy)
        image_height: Image height
        image_width: Image width
        device: Device to use
        
    Returns:
        Ray directions [H, W, 3] in camera coordinates
    """
    fx, fy, cx, cy = camera_intrinsics
    
    # Create pixel coordinates
    i, j = torch.meshgrid(
        torch.arange(image_width, device=device),
        torch.arange(image_height, device=device),
        indexing='ij'
    )
    i = i.t().float()
    j = j.t().float()
    
    # Convert to normalized device coordinates
    x = (i - cx) / fx
    y = (j - cy) / fy
    z = torch.ones_like(x)
    
    # Ray directions
    directions = torch.stack([x, y, z], dim=-1)
    
    # Normalize
    directions = directions / torch.norm(directions, dim=-1, keepdim=True)
    
    return directions


def transform_rays(
    ray_origins: torch.Tensor,
    ray_directions: torch.Tensor,
    transform: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Transform rays by a transformation matrix.
    
    Args:
        ray_origins: Ray origins [..., 3]
        ray_directions: Ray directions [..., 3]
        transform: Transformation matrix [4, 4]
        
    Returns:
        Transformed (ray_origins, ray_directions)
    """
    # Transform origins
    origins_homogeneous = torch.cat([
        ray_origins,
        torch.ones(*ray_origins.shape[:-1], 1, device=ray_origins.device)
    ], dim=-1)
    
    transformed_origins = torch.matmul(origins_homogeneous, transform.t())
    transformed_origins = transformed_origins[..., :3]
    
    # Transform directions (only rotation)
    R = transform[:3, :3]
    transformed_directions = torch.matmul(ray_directions, R.t())
    
    return transformed_origins, transformed_directions 