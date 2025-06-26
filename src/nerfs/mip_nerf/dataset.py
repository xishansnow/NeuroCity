"""
Dataset module for Mip-NeRF

This module implements dataset classes for loading and preprocessing data for Mip-NeRF training, including support for Blender synthetic scenes and real image datasets.
"""

import torch
import torch.utils.data as data
import numpy as np
import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import cv2
from PIL import Image
import imageio


class MipNeRFDataset(data.Dataset):
    """
    Base dataset class for Mip-NeRF
    
    This class handles the common functionality for loading images, poses, and camera parameters.
    """
    
    def __init__(
        self,
        data_dir: str,
        split: str = 'train',
        img_wh: Optional[tuple[int,int]] = None,
        white_bkgd: bool = False,
        half_res: bool = False,
        testskip: int = 1,
    ) -> None:
        """
        Args:
            data_dir: Path to dataset directory
            split: Dataset split ('train', 'val', 'test')
            img_wh: Tuple of (width, height) to resize images
            white_bkgd: Whether to use white background
            half_res: Whether to use half resolution
            testskip: Skip every N frames for test set
        """
        self.data_dir = Path(data_dir)
        self.split = split
        self.img_wh = img_wh
        self.white_bkgd = white_bkgd
        self.half_res = half_res
        self.testskip = testskip
        
        self.images = []
        self.poses = []
        self.rays = []
        self.rgbs = []
        
        self.focal = None
        self.near = None
        self.far = None
        
        self._load_data()
        
    def _load_data(self):
        """Load dataset - to be implemented by subclasses"""
        raise NotImplementedError
        
    def __len__(self):
        return len(self.rays)
    
    def __getitem__(self, idx):
        return {
            'rays': self.rays[idx], 'rgbs': self.rgbs[idx]
        }
    
    def get_rays(self, pose: torch.Tensor, H: int, W: int, focal: float) -> dict[str, torch.Tensor]:
        """
        Generate rays from camera pose and intrinsics
        
        Args:
            pose: [4, 4] camera pose matrix
            H: Image height
            W: Image width
            focal: Camera focal length
            
        Returns:
            Dictionary containing ray origins, directions, and radii
        """
        # Create pixel coordinates
        i, j = torch.meshgrid(
            torch.arange(W, dtype=torch.float32),
            torch.arange(H, dtype=torch.float32),
            indexing='ij'
        )
        
        # Convert to camera coordinates
        dirs = torch.stack([
            (i - W * 0.5) / focal, -(j - H * 0.5) / focal, -torch.ones_like(i)
        ], dim=-1)
        
        # Transform to world coordinates
        rays_d = torch.sum(dirs[..., None, :] * pose[:3, :3], dim=-1)
        rays_o = pose[:3, 3].expand(rays_d.shape)
        
        # Compute pixel radii for anti-aliasing
        pixel_radius = 1.0 / focal * np.sqrt(2) / 2
        radii = torch.full(rays_d.shape[:-1], pixel_radius)
        
        return {
            'rays_o': rays_o, 'rays_d': rays_d, 'radii': radii
        }


class BlenderMipNeRFDataset(MipNeRFDataset):
    """
    Dataset for Blender synthetic scenes (NeRF synthetic dataset)
    """
    
    def _load_data(self):
        """Load Blender synthetic data"""
        # Load transforms file
        transforms_file = self.data_dir / f'transforms_{self.split}.json'
        
        with open(transforms_file, 'r') as f:
            meta = json.load(f)
        
        # Extract camera parameters
        camera_angle_x = meta['camera_angle_x']
        
        # Load images and poses
        frames = meta['frames']
        if self.split == 'test':
            frames = frames[::self.testskip]
        
        self.poses = []
        self.images = []
        
        for frame in frames:
            # Load pose
            pose = np.array(frame['transform_matrix'], dtype=np.float32)
            self.poses.append(torch.from_numpy(pose))
            
            # Load image
            img_path = self.data_dir / f"{frame['file_path']}.png"
            img = imageio.imread(img_path)
            
            # Handle different image formats
            if img.dtype == np.uint8:
                img = img.astype(np.float32) / 255.0
            
            # Handle alpha channel
            if img.shape[-1] == 4:
                if self.white_bkgd:
                    img = img[..., :3] * img[..., -1:] + (1.0 - img[..., -1:])
                else:
                    img = img[..., :3]
            
            self.images.append(torch.from_numpy(img))
        
        # Determine image dimensions
        H, W = self.images[0].shape[:2]
        
        if self.half_res:
            H = H // 2
            W = W // 2
            # Resize images
            self.images = [
                torch.from_numpy(cv2.resize(img.numpy(), (W, H), interpolation=cv2.INTER_AREA))
                for img in self.images
            ]
        
        if self.img_wh is not None:
            W, H = self.img_wh
            # Resize images
            self.images = [
                torch.from_numpy(cv2.resize(img.numpy(), (W, H), interpolation=cv2.INTER_AREA))
                for img in self.images
            ]
        
        # Compute focal length
        self.focal = 0.5 * W / np.tan(0.5 * camera_angle_x)
        
        # Set near and far planes
        self.near = 2.0
        self.far = 6.0
        
        # Generate rays for all images
        self._generate_rays(H, W)
    
    def _generate_rays(self, H: int, W: int):
        """Generate all rays for the dataset"""
        all_rays = []
        all_rgbs = []
        
        for img, pose in zip(self.images, self.poses):
            rays = self.get_rays(pose, H, W, self.focal)
            
            # Flatten spatial dimensions
            rays_o = rays['rays_o'].reshape(-1, 3)
            rays_d = rays['rays_d'].reshape(-1, 3)
            radii = rays['radii'].reshape(-1)
            rgbs = img.reshape(-1, 3)
            
            # Store rays
            ray_batch = torch.cat([rays_o, rays_d, radii.unsqueeze(-1)], dim=-1)
            all_rays.append(ray_batch)
            all_rgbs.append(rgbs)
        
        self.rays = torch.cat(all_rays, dim=0)
        self.rgbs = torch.cat(all_rgbs, dim=0)


class LLFFMipNeRFDataset(MipNeRFDataset):
    """
    Dataset for LLFF (Local Light Field Fusion) real scenes
    """
    
    def _load_data(self):
        """Load LLFF data"""
        # Load poses and bounds
        poses_bounds = np.load(self.data_dir / 'poses_bounds.npy')
        poses = poses_bounds[:, :-2].reshape([-1, 3, 5])  # [N, 3, 5]
        bounds = poses_bounds[:, -2:]  # [N, 2]
        
        # Extract intrinsics and extrinsics
        H, W, focal = poses[0, :, -1]
        H, W = int(H), int(W)
        
        # Convert poses to 4x4 matrices
        poses = np.concatenate(
            [poses[:, :, 1:5], np.tile(np.eye(3)[None], [poses.shape[0], 1, 1])], dim=-2
        )
        
        # Load images
        imgdir = self.data_dir / 'images'
        imgfiles = sorted(list(imgdir.glob('*.JPG')) + list(imgdir.glob('*.jpg')) + list(imgdir.glob('*.png')))
        
        if len(imgfiles) != poses.shape[0]:
            raise ValueError(f"Number of images ({len(imgfiles)}) doesn't match number of poses ({poses.shape[0]})")
        
        # Split data
        i_test = np.arange(0, poses.shape[0], 8)  # Every 8th image for test
        i_val = i_test
        i_train = np.array([i for i in np.arange(poses.shape[0]) if i not in i_test])
        
        if self.split == 'train':
            indices = i_train
        elif self.split == 'val':
            indices = i_val
        else:  # test
            indices = i_test[::self.testskip]
        
        # Load selected images and poses
        self.images = []
        self.poses = []
        
        for i in indices:
            # Load image
            img = imageio.imread(imgfiles[i])
            if img.dtype == np.uint8:
                img = img.astype(np.float32) / 255.0
            
            self.images.append(torch.from_numpy(img))
            self.poses.append(torch.from_numpy(poses[i].astype(np.float32)))
        
        # Handle resizing
        if self.half_res:
            H = H // 2
            W = W // 2
            focal = focal / 2.0
            self.images = [
                torch.from_numpy(cv2.resize(img.numpy(), (W, H), interpolation=cv2.INTER_AREA))
                for img in self.images
            ]
        
        if self.img_wh is not None:
            W_new, H_new = self.img_wh
            focal = focal * W_new / W
            W, H = W_new, H_new
            self.images = [
                torch.from_numpy(cv2.resize(img.numpy(), (W, H), interpolation=cv2.INTER_AREA))
                for img in self.images
            ]
        
        self.focal = focal
        
        # Set near and far from bounds
        self.near = bounds.min() * 0.9
        self.far = bounds.max() * 1.1
        
        # Generate rays
        self._generate_rays(H, W)
    
    def _generate_rays(self, H: int, W: int):
        """Generate all rays for the dataset"""
        all_rays = []
        all_rgbs = []
        
        for img, pose in zip(self.images, self.poses):
            rays = self.get_rays(pose, H, W, self.focal)
            
            # Flatten spatial dimensions
            rays_o = rays['rays_o'].reshape(-1, 3)
            rays_d = rays['rays_d'].reshape(-1, 3)
            radii = rays['radii'].reshape(-1)
            rgbs = img.reshape(-1, 3)
            
            # Store rays
            ray_batch = torch.cat([rays_o, rays_d, radii.unsqueeze(-1)], dim=-1)
            all_rays.append(ray_batch)
            all_rgbs.append(rgbs)
        
        self.rays = torch.cat(all_rays, dim=0)
        self.rgbs = torch.cat(all_rgbs, dim=0)


class MipNeRFRayDataset(data.Dataset):
    """
    Dataset that yields individual rays for training
    """
    
    def __init__(self, dataset: MipNeRFDataset, batch_size: int = 1024):
        """
        Args:
            dataset: Base MipNeRF dataset
            batch_size: Number of rays per batch
        """
        self.rays = dataset.rays
        self.rgbs = dataset.rgbs
        self.batch_size = batch_size
        
        # Shuffle rays
        indices = torch.randperm(len(self.rays))
        self.rays = self.rays[indices]
        self.rgbs = self.rgbs[indices]
    
    def __len__(self):
        return len(self.rays) // self.batch_size
    
    def __getitem__(self, idx):
        start_idx = idx * self.batch_size
        end_idx = min((idx + 1) * self.batch_size, len(self.rays))
        
        batch_rays = self.rays[start_idx:end_idx]
        batch_rgbs = self.rgbs[start_idx:end_idx]
        
        # Parse ray data
        rays_o = batch_rays[:, :3]
        rays_d = batch_rays[:, 3:6]
        radii = batch_rays[:, 6]
        
        return {
            'rays_o': rays_o, 'rays_d': rays_d, 'radii': radii, 'rgbs': batch_rgbs
        }


def create_mip_nerf_dataset(
    data_dir: str,
    dataset_type: str = 'blender',
    split: str = 'train',
    **kwargs: Any,
) -> MipNeRFDataset:
    """
    Factory function to create MipNeRF datasets
    
    Args:
        data_dir: Path to dataset director
        dataset_type: Type of dataset ('blender', 'llff', 'custom')
        split: Dataset split ('train', 'val', 'test')
        img_wh: Tuple of (width, height) to resize images
        white_bkgd: Whether to use white background
        half_res: Whether to use half resolution
        testskip: Skip every N frames for test set
        **kwargs: Additional dataset-specific arguments
        
    Returns:
        MipNeRFDataset instance
    """
    if dataset_type.lower() == 'blender':
        return BlenderMipNeRFDataset(data_dir, split, **kwargs)
    elif dataset_type.lower() == 'llff':
        return LLFFMipNeRFDataset(data_dir, split, **kwargs)
    else:
        raise ValueError(f"Unknown dataset type: {dataset_type}")


def create_mip_nerf_dataloader(
    dataset: MipNeRFDataset | MipNeRFRayDataset, batch_size: int = 4096, shuffle: bool = True, num_workers: int = 4, pin_memory: bool = True
) -> data.DataLoader:
    """
    Create DataLoader for MipNeRF dataset
    
    Args:
        dataset: MipNeRF dataset
        batch_size: Batch size (if None, use dataset's natural batching)
        shuffle: Whether to shuffle data
        num_workers: Number of worker processes
        
    Returns:
        PyTorch DataLoader
    """
    if isinstance(dataset, MipNeRFRayDataset):
        # Ray dataset already handles batching
        return data.DataLoader(
            dataset, batch_size=1, # Each item is already a batch
            shuffle=shuffle, num_workers=num_workers, collate_fn=lambda x: x[0]  # Remove extra batch dimension
        )
    else:
        # Regular dataset
        batch_size = batch_size or len(dataset)
        return data.DataLoader(
            dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, pin_memory=pin_memory
        )


def collate_fn(batch: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    """
    Custom collate function for MipNeRF data
    
    Args:
        batch: List of data samples
        
    Returns:
        Batched data
    """
    keys = batch[0].keys()
    batched = {}
    
    for key in keys:
        tensors = [item[key] for item in batch]
        batched[key] = torch.stack(tensors, dim=0)
    
    return batched


class MipNeRFImageDataset(data.Dataset):
    """
    Dataset that yields full images for validation/testing
    """
    
    def __init__(self, dataset: MipNeRFDataset):
        """
        Args:
            dataset: Base MipNeRF dataset
        """
        self.base_dataset = dataset
        self.images = dataset.images
        self.poses = dataset.poses
        self.focal = dataset.focal
        
        # Get image dimensions
        if len(self.images) > 0:
            self.H, self.W = self.images[0].shape[:2]
        else:
            self.H = self.W = 0
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        img = self.images[idx]
        pose = self.poses[idx]
        
        # Generate rays for the entire image
        rays = self.base_dataset.get_rays(pose, self.H, self.W, self.focal)
        
        return  {
            'image': img, 'pose': pose, 'rays_o': rays['rays_o'], 'rays_d': rays['rays_d'], 'radii': rays['radii']
        } 
        
        