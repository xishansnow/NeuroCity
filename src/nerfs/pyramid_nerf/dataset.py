from typing import Any, Optional, Union
"""
Dataset module for PyNeRF
Handles data loading and preprocessing for multi-scale training
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

logger = logging.getLogger(__name__)

class PyNeRFDataset(data.Dataset):
    """
    Dataset class for PyNeRF training
    Supports various data formats including NeRF synthetic, LLFF, and custom formats
    """
    
    def __init__(
        self, data_dir: str, split: str = "train", img_downscale: int = 1, use_cache: bool = True, white_background: bool = False, near_far: Optional[tuple[float, float]] = None, **kwargs
    ):
        super().__init__()
        
        self.data_dir = data_dir
        self.split = split
        self.img_downscale = img_downscale
        self.use_cache = use_cache
        self.white_background = white_background
        self.near_far = near_far
        
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
        
        self.images = np.stack(self.images, axis=0)
        self.poses = np.stack(self.poses, axis=0)
        
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
        
        self.poses = np.stack(self.poses, axis=0)
        
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
        self.image_paths = [self.image_paths[i] for i in indices]
    
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
        
        # Create bounds
        near, far = self.near_far
        bounds = torch.tensor([near, far], dtype=torch.float32)
        bounds = bounds.expand(rays_o.shape[0], rays_o.shape[1], 2)
        
        return {
            "image": image, "rays_o": rays_o, "rays_d": rays_d, "pose": pose, "bounds": bounds, "idx": idx
        }

class MultiScaleDataset(PyNeRFDataset):
    """
    Multi-scale dataset for pyramid training
    Generates images at different resolutions for coarse-to-fine training
    """
    
    def __init__(
        self, data_dir: str, split: str = "train", scales: list[int] = [1, 2, 4, 8], **kwargs
    ):
        self.scales = scales
        super().__init__(data_dir, split, **kwargs)
        
        # Generate multi-scale images
        self._generate_multiscale_images()
    
    def _generate_multiscale_images(self):
        """Generate images at multiple scales"""
        
        self.multiscale_images = {}
        self.multiscale_focals = {}
        
        for scale in self.scales:
            if scale == 1:
                # Original scale
                self.multiscale_images[scale] = self.images
                self.multiscale_focals[scale] = self.focal
            else:
                # Downscaled images
                scaled_images = []
                for img in self.images:
                    h, w = img.shape[:2]
                    new_h, new_w = h // scale, w // scale
                    
                    # Use PIL for high-quality resizing
                    img_pil = Image.fromarray((img * 255).astype(np.uint8))
                    img_scaled = img_pil.resize((new_w, new_h), Image.LANCZOS)
                    img_scaled = np.array(img_scaled) / 255.0
                    
                    scaled_images.append(img_scaled)
                
                self.multiscale_images[scale] = np.stack(scaled_images, axis=0)
                self.multiscale_focals[scale] = self.focal / scale
    
    def get_rays_multiscale(
        self, idx: int, scale: int = 1
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Get rays for a specific image at a given scale
        
        Args:
            idx: Image index
            scale: Scale factor
            
        Returns:
            tuple of (ray_origins, ray_directions)
        """
        pose = self.poses[idx]
        focal = self.multiscale_focals[scale]
        h, w = self.multiscale_images[scale].shape[1:3]
        
        # Create pixel coordinates
        i, j = np.meshgrid(
            np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32), indexing='xy'
        )
        
        # Convert to camera coordinates
        dirs = np.stack([
            (i - w * 0.5) / focal, -(j - h * 0.5) / focal, -np.ones_like(i)
        ], axis=-1)
        
        # Transform to world coordinates
        rays_d = np.sum(dirs[..., None, :] * pose[:3, :3], axis=-1)
        rays_o = np.broadcast_to(pose[:3, -1], rays_d.shape)
        
        return torch.from_numpy(rays_o), torch.from_numpy(rays_d)
    
    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        """
        Get a multi-scale data sample
        
        Args:
            idx: Sample index
            
        Returns:
            Dictionary containing multi-scale data
        """
        # Get base sample
        sample = super().__getitem__(idx)
        
        # Add multi-scale data
        sample["multiscale_images"] = {}
        sample["multiscale_rays_o"] = {}
        sample["multiscale_rays_d"] = {}
        sample["multiscale_bounds"] = {}
        
        for scale in self.scales:
            # Get scaled image
            image = torch.from_numpy(self.multiscale_images[scale][idx]).float()
            
            # Get scaled rays
            rays_o, rays_d = self.get_rays_multiscale(idx, scale)
            
            # Create bounds
            near, far = self.near_far
            bounds = torch.tensor([near, far], dtype=torch.float32)
            bounds = bounds.expand(rays_o.shape[0], rays_o.shape[1], 2)
            
            sample["multiscale_images"][scale] = image
            sample["multiscale_rays_o"][scale] = rays_o
            sample["multiscale_rays_d"][scale] = rays_d
            sample["multiscale_bounds"][scale] = bounds
        
        return sample

def create_dataloader(
    dataset: PyNeRFDataset, batch_size: int = 1, shuffle: bool = True, num_workers: int = 4, **kwargs
) -> DataLoader:
    """
    Create a DataLoader for PyNeRF dataset
    
    Args:
        dataset: PyNeRF dataset
        batch_size: Batch size
        shuffle: Whether to shuffle data
        num_workers: Number of worker processes
        
    Returns:
        PyTorch DataLoader
    """
    return DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, pin_memory=True, **kwargs
    )

def collate_fn(batch: list[dict]) -> dict[str, torch.Tensor]:
    """
    Custom collate function for PyNeRF data
    
    Args:
        batch: list of data samples
        
    Returns:
        Batched data dictionary
    """
    # Simple batching for now
    # Can be extended for more complex batching strategies
    return torch.utils.data.default_collate(batch)
