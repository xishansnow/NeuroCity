from typing import Any, Optional, Union
"""
Dataset module for BungeeNeRF
Handles multi-scale data loading and Google Earth Studio data
"""

import torch
import torch.utils.data as data
from torch.utils.data import DataLoader
import numpy as np
import os
import json
from PIL import Image
import cv2
import logging
from scipy.spatial.transform import Rotation

logger = logging.getLogger(__name__)

class BungeeNeRFDataset(data.Dataset):
    """
    Dataset class for BungeeNeRF training with multi-scale support
    """
    
    def __init__(
        self, data_dir: str, split: str = "train", img_downscale: int = 1, use_cache: bool = True, white_background: bool = False, near_far: Optional[tuple[float, float]] = None, scale_factor: float = 4.0, num_scales: int = 4, **kwargs
    ):
        super().__init__()
        
        self.data_dir = data_dir
        self.split = split
        self.img_downscale = img_downscale
        self.use_cache = use_cache
        self.white_background = white_background
        self.near_far = near_far
        self.scale_factor = scale_factor
        self.num_scales = num_scales
        
        # Load dataset
        self._load_dataset()
        
        logger.info(f"Loaded {len(self.images)} images for {split} split")
    
    def _load_dataset(self):
        """Load dataset based on format detection"""
        
        # Try to detect dataset format
        if os.path.exists(os.path.join(self.data_dir, "transforms_train.json")):
            self._load_nerf_synthetic()
        elif os.path.exists(os.path.join(self.data_dir, "poses_bounds.npy")):
            self._load_llff()
        elif os.path.exists(os.path.join(self.data_dir, "metadata.json")):
            self._load_google_earth()
        else:
            raise ValueError(f"Unknown dataset format in {self.data_dir}")
    
    def _load_nerf_synthetic(self):
        """Load NeRF synthetic dataset (Blender format)"""
        
        # Load transforms
        transforms_file = os.path.join(self.data_dir, f"transforms_{self.split}.json")
        
        if not os.path.exists(transforms_file):
            # Fallback to train transforms for test/val
            transforms_file = os.path.join(self.data_dir, "transforms_train.json")
        
        with open(transforms_file, 'r') as f:
            transforms = json.load(f)
        
        # Get camera parameters
        self.camera_angle_x = transforms.get("camera_angle_x", 0.6911112070083618)
        
        # Load images and poses
        self.images = []
        self.poses = []
        self.image_paths = []
        self.distances = []
        
        frames = transforms["frames"]
        if self.split == "train":
            frames = frames[::1]  # Use all training frames
        elif self.split == "val":
            frames = frames[::8]  # Use every 8th frame for validation
        elif self.split == "test":
            frames = frames  # Use all test frames
        
        for frame in frames:
            # Load image
            img_path = os.path.join(self.data_dir, frame["file_path"] + ".png")
            if not os.path.exists(img_path):
                img_path = os.path.join(self.data_dir, frame["file_path"])
            
            image = Image.open(img_path).convert("RGBA")
            
            # Downscale if needed
            if self.img_downscale > 1:
                w, h = image.size
                image = image.resize((w // self.img_downscale, h // self.img_downscale))
            
            image = np.array(image) / 255.0
            
            # Handle alpha channel
            if image.shape[-1] == 4:
                if self.white_background:
                    # Blend with white background
                    image = image[..., :3] * image[..., 3:] + (1 - image[..., 3:])
                else:
                    # Keep alpha channel
                    pass
            
            self.images.append(image[..., :3])
            self.image_paths.append(img_path)
            
            # Load pose
            pose = np.array(frame["transform_matrix"], dtype=np.float32)
            self.poses.append(pose)
            
            # Calculate distance from origin (for multi-scale)
            distance = np.linalg.norm(pose[:3, 3])
            self.distances.append(distance)
        
        self.images = np.stack(self.images, axis=0)
        self.poses = np.stack(self.poses, axis=0)
        self.distances = np.array(self.distances)
        
        # Get image dimensions
        self.H, self.W = self.images.shape[1:3]
        
        # Calculate focal length
        self.focal = 0.5 * self.W / np.tan(0.5 * self.camera_angle_x)
        
        # Set near/far bounds
        if self.near_far is None:
            self.near_far = (2.0, 6.0)  # Default for synthetic scenes
    
    def _load_llff(self):
        """Load LLFF dataset format"""
        
        # Load poses and bounds
        poses_bounds = np.load(os.path.join(self.data_dir, "poses_bounds.npy"))
        
        # Split poses and bounds
        poses = poses_bounds[:, :-2].reshape([-1, 3, 5])
        bounds = poses_bounds[:, -2:]
        
        # Get image paths
        img_dir = os.path.join(self.data_dir, "images")
        if not os.path.exists(img_dir):
            img_dir = self.data_dir
        
        img_files = sorted([f for f in os.listdir(img_dir) 
                           if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
        
        # Load images
        self.images = []
        self.image_paths = []
        self.distances = []
        
        for img_file in img_files:
            img_path = os.path.join(img_dir, img_file)
            image = Image.open(img_path).convert("RGB")
            
            # Downscale if needed
            if self.img_downscale > 1:
                w, h = image.size
                image = image.resize((w // self.img_downscale, h // self.img_downscale))
            
            image = np.array(image) / 255.0
            self.images.append(image)
            self.image_paths.append(img_path)
        
        self.images = np.stack(self.images, axis=0)
        
        # Process poses
        self.poses = []
        for pose in poses:
            # Convert LLFF pose to NeRF pose
            pose_nerf = np.eye(4)
            pose_nerf[:3, :4] = pose[:3, :4]
            self.poses.append(pose_nerf)
            
            # Calculate distance
            distance = np.linalg.norm(pose[:3, 3])
            self.distances.append(distance)
        
        self.poses = np.stack(self.poses, axis=0)
        self.distances = np.array(self.distances)
        
        # Get image dimensions
        self.H, self.W = self.images.shape[1:3]
        
        # Get focal length from poses
        hwf = poses[0, :3, -1]
        self.focal = hwf[2]
        
        # Set near/far bounds
        if self.near_far is None:
            self.near_far = (bounds.min(), bounds.max())
        
        # Split dataset
        if self.split == "train":
            indices = list(range(0, len(self.images), 8))[:-1]  # Skip every 8th for test
        elif self.split == "val":
            indices = list(range(0, len(self.images), 8))[-1:]  # Last test image for val
        elif self.split == "test":
            indices = list(range(0, len(self.images), 8))  # Every 8th image
        
        self.images = self.images[indices]
        self.poses = self.poses[indices]
        self.distances = self.distances[indices]
        self.image_paths = [self.image_paths[i] for i in indices]
    
    def _load_google_earth(self):
        """Load Google Earth Studio dataset"""
        
        # Load metadata
        metadata_file = os.path.join(self.data_dir, "metadata.json")
        with open(metadata_file, 'r') as f:
            metadata = json.load(f)
        
        # Get camera parameters
        self.W = metadata.get("width", 1920)
        self.H = metadata.get("height", 1080)
        fov_vertical = metadata.get("fovVertical", 30.0)
        
        # Calculate focal length
        self.focal = 0.5 * self.H / np.tan(0.5 * np.radians(fov_vertical))
        
        # Load camera frames
        camera_frames = metadata["cameraFrames"]
        
        # Filter frames based on split
        if self.split == "train":
            frames = camera_frames[::2]  # Every other frame for training
        elif self.split == "val":
            frames = camera_frames[1::8]  # Sparse validation set
        elif self.split == "test":
            frames = camera_frames[::4]  # Test set
        else:
            frames = camera_frames
        
        self.images = []
        self.poses = []
        self.image_paths = []
        self.distances = []
        
        for i, frame in enumerate(frames):
            # Load image
            img_path = os.path.join(self.data_dir, "images", f"frame_{i:06d}.jpg")
            if not os.path.exists(img_path):
                img_path = os.path.join(self.data_dir, "images", f"{i:06d}.jpg")
            
            if os.path.exists(img_path):
                image = Image.open(img_path).convert("RGB")
                
                # Downscale if needed
                if self.img_downscale > 1:
                    w, h = image.size
                    image = image.resize((w // self.img_downscale, h // self.img_downscale))
                
                image = np.array(image) / 255.0
                self.images.append(image)
                self.image_paths.append(img_path)
                
                # Convert GES pose to NeRF format
                pose = self._ges_to_nerf_pose(frame)
                self.poses.append(pose)
                
                # Calculate distance
                distance = np.linalg.norm(pose[:3, 3])
                self.distances.append(distance)
        
        if len(self.images) == 0:
            raise ValueError(f"No images found in {self.data_dir}")
        
        self.images = np.stack(self.images, axis=0)
        self.poses = np.stack(self.poses, axis=0)
        self.distances = np.array(self.distances)
        
        # Update image dimensions after loading
        self.H, self.W = self.images.shape[1:3]
        
        # Set near/far bounds for aerial scenes
        if self.near_far is None:
            min_dist = self.distances.min()
            max_dist = self.distances.max()
            self.near_far = (min_dist * 0.1, max_dist * 2.0)
    
    def _ges_to_nerf_pose(self, frame: dict) -> np.ndarray:
        """Convert Google Earth Studio pose to NeRF format"""
        
        # Get position and rotation from GES
        position = frame["position"]
        rotation = frame["rotation"]
        
        # Convert position to numpy array
        pos = np.array([position["x"], position["y"], position["z"]], dtype=np.float32)
        
        # Convert rotation to rotation matrix
        # GES uses XYZ Euler angles in degrees
        euler_angles = np.array([rotation["x"], rotation["y"], rotation["z"]], dtype=np.float32)
        
        # Apply coordinate system conversion
        # x' = -x, y' = 180 - y, z' = 180 + z (as mentioned in the paper)
        euler_angles[0] = -euler_angles[0]
        euler_angles[1] = 180 - euler_angles[1]
        euler_angles[2] = 180 + euler_angles[2]
        
        # Convert to radians
        euler_angles = np.radians(euler_angles)
        
        # Create rotation matrix
        r = Rotation.from_euler('xyz', euler_angles)
        rot_matrix = r.as_matrix()
        
        # Create 4x4 pose matrix
        pose = np.eye(4, dtype=np.float32)
        pose[:3, :3] = rot_matrix
        pose[:3, 3] = pos
        
        return pose
    
    def get_rays(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Get rays for a specific image
        
        Args:
            idx: Image index
            
        Returns:
            tuple of (ray_origins, ray_directions)
        """
        pose = self.poses[idx]
        
        # Create pixel coordinates
        i, j = np.meshgrid(
            np.arange(self.W, dtype=np.float32), np.arange(self.H, dtype=np.float32), indexing='xy'
        )
        
        # Convert to camera coordinates
        dirs = np.stack([
            (i - self.W * 0.5) / self.focal, -(j - self.H * 0.5) / self.focal, -np.ones_like(i)
        ], axis=-1)
        
        # Transform to world coordinates
        rays_d = np.sum(dirs[..., None, :] * pose[:3, :3], axis=-1)
        rays_o = np.broadcast_to(pose[:3, -1], rays_d.shape)
        
        return torch.from_numpy(rays_o), torch.from_numpy(rays_d)
    
    def get_distance(self, idx: int) -> float:
        """Get distance to camera for given index"""
        return self.distances[idx]
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        """
        Get a single data sample
        
        Args:
            idx: Sample index
            
        Returns:
            Dictionary containing image, rays, and metadata
        """
        # Get image
        image = torch.from_numpy(self.images[idx]).float()
        
        # Get rays
        rays_o, rays_d = self.get_rays(idx)
        
        # Get pose
        pose = torch.from_numpy(self.poses[idx]).float()
        
        # Get distance
        distance = torch.tensor(self.distances[idx], dtype=torch.float32)
        
        # Create bounds
        near, far = self.near_far
        bounds = torch.tensor([near, far], dtype=torch.float32)
        bounds = bounds.expand(rays_o.shape[0], rays_o.shape[1], 2)
        
        return {
            "image": image, "rays_o": rays_o, "rays_d": rays_d, "pose": pose, "bounds": bounds, "distance": distance, "idx": idx
        }

class MultiScaleDataset(BungeeNeRFDataset):
    """
    Multi-scale dataset for progressive training
    Organizes data by distance/scale for BungeeNeRF
    """
    
    def __init__(
        self, data_dir: str, split: str = "train", scale_thresholds: list[float] = None, **kwargs
    ):
        if scale_thresholds is None:
            scale_thresholds = [100.0, 50.0, 25.0, 10.0]
        
        self.scale_thresholds = scale_thresholds
        
        super().__init__(data_dir, split, **kwargs)
        
        # Organize data by scale
        self._organize_by_scale()
    
    def _organize_by_scale(self):
        """Organize data by distance/scale"""
        
        self.scale_indices = [[] for _ in range(len(self.scale_thresholds) + 1)]
        
        for idx, distance in enumerate(self.distances):
            # Determine scale level
            scale_level = len(self.scale_thresholds)  # Default to finest scale
            
            for level, threshold in enumerate(self.scale_thresholds):
                if distance >= threshold:
                    scale_level = level
                    break
            
            self.scale_indices[scale_level].append(idx)
        
        logger.info(f"Scale distribution: {[len(indices) for indices in self.scale_indices]}")
    
    def get_scale_data(self, scale: int) -> list[int]:
        """Get indices for a specific scale"""
        if scale < len(self.scale_indices):
            return self.scale_indices[scale]
        else:
            return []
    
    def get_progressive_data(self, max_scale: int) -> list[int]:
        """Get indices for progressive training up to max_scale"""
        indices = []
        for scale in range(max_scale + 1):
            indices.extend(self.get_scale_data(scale))
        return indices

class GoogleEarthDataset(BungeeNeRFDataset):
    """
    Specialized dataset for Google Earth Studio data
    """
    
    def __init__(
        self, data_dir: str, split: str = "train", coordinate_system: str = "ENU", scale_scene: bool = True, **kwargs
    ):
        self.coordinate_system = coordinate_system
        self.scale_scene = scale_scene
        
        super().__init__(data_dir, split, **kwargs)
        
        if self.scale_scene:
            self._scale_scene_to_unit_sphere()
    
    def _scale_scene_to_unit_sphere(self):
        """Scale scene to fit within unit sphere for better positional encoding"""
        
        # Find scene bounds
        positions = self.poses[:, :3, 3]
        scene_center = positions.mean(axis=0)
        scene_radius = np.linalg.norm(positions - scene_center, axis=1).max()
        
        # Scale factor to fit in [-pi, pi]
        scale_factor = np.pi / scene_radius
        
        # Apply scaling
        for i in range(len(self.poses)):
            self.poses[i][:3, 3] = (self.poses[i][:3, 3] - scene_center) * scale_factor
        
        # Update distances
        self.distances = np.linalg.norm(self.poses[:, :3, 3], axis=1)
        
        # Update near/far bounds
        if self.near_far is not None:
            near, far = self.near_far
            self.near_far = (near * scale_factor, far * scale_factor)
        
        logger.info(f"Scaled scene by factor {scale_factor:.6f}")

def create_bungee_dataloader(
    dataset: BungeeNeRFDataset, batch_size: int = 1, shuffle: bool = True, num_workers: int = 4, **kwargs
) -> DataLoader:
    """
    Create a DataLoader for BungeeNeRF dataset
    
    Args:
        dataset: BungeeNeRF dataset
        batch_size: Batch size
        shuffle: Whether to shuffle data
        num_workers: Number of worker processes
        
    Returns:
        PyTorch DataLoader
    """
    return DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, pin_memory=True, **kwargs
    )
