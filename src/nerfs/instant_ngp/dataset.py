from __future__ import annotations

from typing import Any, Optional, Union
"""
Dataset module for Instant NGP.

This module provides data loading and preprocessing utilities for training 
Instant NGP models on NeRF-style datasets.
"""

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
import json
import os
from pathlib import Path
import cv2
from PIL import Image
import math

class InstantNGPDataset(Dataset):
    """Dataset for Instant NGP training."""
    
    def __init__(
        self, data_root: str, split: str = 'train', img_wh: Optional[Tuple[int, int]] = None, spherify: bool = False, val_num: int = 1, use_cache: bool = True, contract_coords: bool = True, **kwargs
    ):
        """
        Initialize dataset.
        
        Args:
            data_root: Root directory containing the dataset
            split: Dataset split ('train', 'val', 'test')
            img_wh: Target image size (width, height)
            spherify: Whether to spherify poses for 360° scenes
            val_num: Number of validation images
            use_cache: Whether to cache loaded data
            contract_coords: Whether to contract coordinates to unit sphere
        """
        self.data_root = Path(data_root)
        self.split = split
        self.img_wh = img_wh
        self.spherify = spherify
        self.val_num = val_num
        self.use_cache = use_cache
        self.contract_coords = contract_coords
        
        # Load camera parameters and poses
        self.load_meta()
        
        # Load images
        self.load_images()
        
        # Precompute rays if training
        if split == 'train':
            self.precompute_rays()
    
    def load_meta(self) -> None:
        """Load camera metadata from transforms.json."""
        transforms_file = self.data_root / f"transforms_{self.split}.json"
        
        if not transforms_file.exists():
            # Try generic transforms.json
            transforms_file = self.data_root / "transforms.json"
        
        with open(transforms_file, 'r') as f:
            self.meta = json.load(f)
        
        # Extract camera parameters
        if 'camera_angle_x' in self.meta:
            self.focal = 0.5 * self.img_wh[0] / np.tan(0.5 * self.meta['camera_angle_x'])
        elif 'fl_x' in self.meta:
            self.focal = self.meta['fl_x']
        else:
            raise ValueError("No focal length information found in transforms.json")
        
        # Extract poses
        self.poses = []
        self.image_paths = []
        
        frames = self.meta['frames']
        
        # Split data
        if self.split == 'train':
            frames = frames[self.val_num:]
        elif self.split == 'val':
            frames = frames[:self.val_num]
        
        for frame in frames:
            pose = np.array(frame['transform_matrix'])
            self.poses.append(pose)
            
            # Handle different image naming conventions
            if 'file_path' in frame:
                img_path = frame['file_path']
            else:
                img_path = f"r_{frame['rotation']:03d}.png"
            
            if not img_path.startswith('/'):
                img_path = self.data_root / img_path
            
            self.image_paths.append(img_path)
        
        self.poses = np.array(self.poses, dtype=np.float32)
        
        # Spherify poses if requested
        if self.spherify:
            self.poses = self.spherify_poses()
        
        # Set bounds
        if 'near' in self.meta:
            self.near = self.meta['near']
        else:
            self.near = 0.1
            
        if 'far' in self.meta:
            self.far = self.meta['far']
        else:
            self.far = 10.0
    
    def spherify_poses(self) -> np.ndarray:
        """Convert poses to spherical coordinate system."""
        poses = self.poses.copy()
        
        # Center poses
        poses[:, :3, 3] -= np.mean(poses[:, :3, 3], axis=0)
        
        # Spherify
        radius = np.linalg.norm(poses[:, :3, 3], axis=1).mean()
        
        # Set all poses to same radius
        for i in range(len(poses)):
            direction = poses[i, :3, 3] / np.linalg.norm(poses[i, :3, 3])
            poses[i, :3, 3] = direction * radius
        
        return poses
    
    def load_images(self) -> None:
        """Load and preprocess images."""
        self.images = []
        
        for img_path in self.image_paths:
            img = Image.open(img_path).convert('RGB')
            
            # Resize if specified
            if self.img_wh is not None:
                img = img.resize(self.img_wh, Image.LANCZOS)
            else:
                self.img_wh = img.size
            
            # Convert to tensor
            img = np.array(img) / 255.0
            self.images.append(img)
        
        self.images = np.array(self.images, dtype=np.float32)
    
    def precompute_rays(self) -> None:
        """Precompute ray directions for training."""
        W, H = self.img_wh
        
        # Create pixel coordinates
        i, j = np.meshgrid(
            np.arange(W, dtype=np.float32), np.arange(H, dtype=np.float32), indexing='xy'
        )
        
        # Camera coordinates
        directions = np.stack([
            (i - W * 0.5) / self.focal, -(j - H * 0.5) / self.focal, -np.ones_like(i)
        ], axis=-1)
        
        # Normalize directions
        directions = directions / np.linalg.norm(directions, axis=-1, keepdims=True)
        
        # Transform directions to world coordinates
        self.all_rays_o = []
        self.all_rays_d = []
        self.all_rgbs = []
        
        for i, pose in enumerate(self.poses):
            # Ray origins (camera positions)
            rays_o = np.broadcast_to(pose[:3, 3], (H, W, 3))
            
            # Ray directions
            rays_d = directions @ pose[:3, :3].T
            
            # Flatten
            rays_o = rays_o.reshape(-1, 3)
            rays_d = rays_d.reshape(-1, 3)
            rgbs = self.images[i].reshape(-1, 3)
            
            self.all_rays_o.append(rays_o)
            self.all_rays_d.append(rays_d)
            self.all_rgbs.append(rgbs)
        
        # Concatenate all rays
        self.all_rays_o = np.concatenate(self.all_rays_o, axis=0)
        self.all_rays_d = np.concatenate(self.all_rays_d, axis=0)
        self.all_rgbs = np.concatenate(self.all_rgbs, axis=0)
        
        # Contract coordinates if requested
        if self.contract_coords:
            self.all_rays_o = self.contract_to_unisphere(self.all_rays_o)
    
    def contract_to_unisphere(self, positions: np.ndarray) -> np.ndarray:
        """Contract infinite coordinates to unit sphere."""
        mag = np.linalg.norm(positions, axis=-1, keepdims=True)
        mask = mag > 1
        
        contracted = positions.copy()
        contracted[mask] = (2 - 1/mag[mask]) * (positions[mask] / mag[mask])
        
        return contracted
    
    def get_rays(self, pose_idx: int) -> Tuple[np.ndarray, np.ndarray]:
        """Get rays for a specific pose."""
        W, H = self.img_wh
        
        # Create pixel coordinates  
        i, j = np.meshgrid(
            np.arange(W, dtype=np.float32), np.arange(H, dtype=np.float32), indexing='xy'
        )
        
        # Camera coordinates
        directions = np.stack([
            (i - W * 0.5) / self.focal, -(j - H * 0.5) / self.focal, -np.ones_like(i)
        ], axis=-1)
        
        # Normalize directions
        directions = directions / np.linalg.norm(directions, axis=-1, keepdims=True)
        
        pose = self.poses[pose_idx]
        
        # Ray origins
        rays_o = np.broadcast_to(pose[:3, 3], (H, W, 3))
        
        # Ray directions
        rays_d = directions @ pose[:3, :3].T
        
        return rays_o, rays_d
    
    def __len__(self) -> int:
        if self.split == 'train':
            return len(self.all_rays_o)
        else:
            return len(self.poses)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        if self.split == 'train':
            # Return individual ray
            return {
                'rays_o': torch.FloatTensor(
                    self.all_rays_o[idx],
                )
            }
        else:
            # Return full image
            rays_o, rays_d = self.get_rays(idx)
            
            return {
                'rays_o': torch.FloatTensor(
                    rays_o,
                )
            }

def create_instant_ngp_dataloader(
    data_root: str, 
    split: str = 'train', 
    batch_size: int = 8192, 
    img_wh: Optional[Tuple[int, int]] = None, 
    num_workers: int = 4, 
    shuffle: Optional[bool] = None, 
    **kwargs: Any
) -> DataLoader:
    """
    Create DataLoader for Instant NGP dataset.
    
    Args:
        data_root: Root directory containing the dataset
        split: Dataset split ('train', 'val', 'test')
        batch_size: Batch size for training
        img_wh: Target image size
        num_workers: Number of worker processes
        shuffle: Whether to shuffle data (default: True for train, False for val/test)
        **kwargs: Additional arguments for dataset
    
    Returns:
        DataLoader instance
    """
    # Set default shuffle behavior
    if shuffle is None:
        shuffle = (split == 'train')
    
    # Create dataset
    dataset = InstantNGPDataset(
        data_root=data_root, split=split, img_wh=img_wh, **kwargs
    )
    
    # Create dataloader
    dataloader = DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, pin_memory=True, persistent_workers=num_workers > 0
    )
    
    return dataloader

def load_blender_data(basedir: str, half_res: bool = False, testskip: int = 1) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[int], List[int]]:
    """
    Load Blender synthetic dataset.
    
    Args:
        basedir: Base directory containing transforms_*.json files
        half_res: Whether to load half resolution images
        testskip: Skip every N test images
        
    Returns:
        tuple of (images, poses, render_poses, hwf, i_split)
    """
    splits = ['train', 'val', 'test']
    metas = {}
    
    for split in splits:
        with open(os.path.join(basedir, f'transforms_{split}.json'), 'r') as fp:
            metas[split] = json.load(fp)
    
    all_imgs = []
    all_poses = []
    counts = [0]
    
    for split in splits:
        meta = metas[split]
        imgs = []
        poses = []
        
        if split == 'train' or testskip == 0:
            skip = 1
        else:
            skip = testskip
        
        for frame in meta['frames'][::skip]:
            fname = os.path.join(basedir, frame['file_path'] + '.png')
            img = Image.open(fname)
            
            if half_res:
                img = img.resize((img.width // 2, img.height // 2), Image.LANCZOS)
            
            img = np.array(img) / 255.0
            imgs.append(img)
            poses.append(np.array(frame['transform_matrix']))
        
        imgs = np.array(imgs).astype(np.float32)
        poses = np.array(poses).astype(np.float32)
        counts.append(counts[-1] + imgs.shape[0])
        
        all_imgs.append(imgs)
        all_poses.append(poses)
    
    i_split = [np.arange(counts[i], counts[i+1]) for i in range(3)]
    
    imgs = np.concatenate(all_imgs, 0)
    poses = np.concatenate(all_poses, 0)
    
    H, W = imgs[0].shape[:2]
    camera_angle_x = float(metas['train']['camera_angle_x'])
    focal = 0.5 * W / np.tan(0.5 * camera_angle_x)
    
    render_poses = torch.stack([
        pose_spherical(angle, -30.0, 4.0) 
        for angle in np.linspace(-180, 180, 40+1)[:-1]
    ], 0)
    
    return imgs, poses, render_poses, [H, W, focal], i_split

def pose_spherical(theta: float, phi: float, radius: float) -> torch.Tensor:
    """Generate spherical pose."""
    c2w = trans_t(radius)
    c2w = rot_phi(phi/180.*np.pi) @ c2w
    c2w = rot_theta(theta/180.*np.pi) @ c2w
    c2w = torch.Tensor(np.array([[-1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 1]])) @ c2w
    return c2w

def trans_t(t: float) -> torch.Tensor:
    return torch.Tensor([
        [1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, t], [0, 0, 0, 1]
    ]).float()

def rot_phi(phi: float) -> torch.Tensor:
    return torch.Tensor([
        [1, 0, 0, 0], 
        [0, np.cos(phi), 0, -np.sin(phi)], 
        [0, np.sin(phi), np.cos(phi), 0], 
        [0, 0, 0, 1]
    ]).float()

def rot_theta(th: float) -> torch.Tensor:
    return torch.Tensor([
        [np.cos(th), 0, np.sin(th), 0], 
        [0, 1, 0, 0],
        [-np.sin(th), 0, np.cos(th), 0],
        [0, 0, 0, 1]
    ]).float()

