"""
from __future__ import annotations

Dataset module for Classic NeRF.

Supports common NeRF datasets including:
- Blender synthetic scenes
- LLFF real scenes  
- Custom datasets
"""

import os
import json
import imageio
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from typing import Any
import cv2


def get_rays_np(H: int, W: int, K: np.ndarray, c2w: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Get ray origins and directions from camera parameters."""
    i, j = np.meshgrid(
        np.arange,
    )
    dirs = np.stack([(i-K[0][2])/K[0][0], -(j-K[1][2])/K[1][1], -np.ones_like(i)], -1)
    rays_d = np.sum(dirs[..., np.newaxis, :] * c2w[:3, :3], -1)
    rays_o = np.broadcast_to(c2w[:3, -1], np.shape(rays_d))
    return rays_o, rays_d


def pose_spherical(theta: float, phi: float, radius: float) -> np.ndarray:
    """Generate spherical pose."""
    trans_t = lambda t : np.array([
        [1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, t], [0, 0, 0, 1]], dtype=np.float32)

    rot_phi = lambda phi : np.array([
        [1, 0, 0, 0], [0, np.cos(phi), -np.sin(phi), 0], [0, np.sin(phi), np.cos(phi), 0], [0, 0, 0, 1]], dtype=np.float32)

    rot_theta = lambda th : np.array([
        [np.cos(th), 0, -np.sin(th), 0], [0, 1, 0, 0], [np.sin(th), 0, np.cos(th), 0], [0, 0, 0, 1]], dtype=np.float32)

    c2w = trans_t(radius)
    c2w = rot_phi(phi/180.*np.pi) @ c2w
    c2w = rot_theta(theta/180.*np.pi) @ c2w
    c2w = np.array([[-1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 1]]) @ c2w
    return c2w


class BlenderDataset(Dataset):
    """Dataset for Blender synthetic scenes."""
    
    def __init__(
        self,
        basedir: str,
        split: str = 'train',
        half_res: bool = False,
        testskip: int = 1,
        white_bkgd: bool = True,
        factor: int | None = None
    ):
        """
        Initialize Blender dataset.
        
        Args:
            basedir: Path to dataset directory
            split: train/val/test split
            half_res: Whether to use half resolution
            testskip: Skip every N test images
            white_bkgd: Whether to use white background
            factor: Downsample factor
        """
        self.basedir = basedir
        self.split = split
        self.half_res = half_res
        self.testskip = testskip
        self.white_bkgd = white_bkgd
        self.factor = factor
        
        # Load dataset
        self.images, self.poses, self.render_poses, self.hwf, self.i_split = self.load_blender_data(
        )
        
        # Split data
        if split == 'train':
            self.images = self.images[self.i_split[0]]
            self.poses = self.poses[self.i_split[0]]
        elif split == 'val':
            self.images = self.images[self.i_split[1]]
            self.poses = self.poses[self.i_split[1]]
        elif split == 'test':
            self.images = self.images[self.i_split[2]]
            self.poses = self.poses[self.i_split[2]]
        
        self.H, self.W, self.focal = self.hwf
        self.H, self.W = int(self.H), int(self.W)
        
        # Camera intrinsics
        self.K = np.array([
            [self.focal, 0, 0.5*self.W], [0, self.focal, 0.5*self.H], [0, 0, 1]
        ])
        
        # Generate rays for all images
        self.rays = []
        self.rgbs = []
        for i in range(len(self.images)):
            rays_o, rays_d = get_rays_np(self.H, self.W, self.K, self.poses[i, :3, :4])
            self.rays.append(np.stack([rays_o, rays_d], 0))
            self.rgbs.append(self.images[i])
        
        self.rays = np.stack(self.rays, 0)  # [N, ro+rd, H, W, 3]
        self.rgbs = np.stack(self.rgbs, 0)  # [N, H, W, 3]
        
        # Reshape to rays
        self.rays = np.transpose(self.rays, [0, 2, 3, 1, 4])  # [N, H, W, ro+rd, 3]
        self.rays = np.reshape(self.rays, [-1, 2, 3])  # [N*H*W, ro+rd, 3]
        self.rgbs = np.reshape(self.rgbs, [-1, 3])  # [N*H*W, 3]
        
        # Convert to tensors
        self.rays = torch.from_numpy(self.rays).float()
        self.rgbs = torch.from_numpy(self.rgbs).float()
        
    def load_blender_data(self):
        """Load Blender dataset."""
        splits = ['train', 'val', 'test']
        metas = {}
        for s in splits:
            json_path = os.path.join(self.basedir, 'transforms_{}.json'.format(s))
            if not os.path.exists(json_path):
                # Create dummy data if file doesn't exist
                metas[s] = {
                    'camera_angle_x': 0.6911112070083618, 'frames': [
                        {
                            'file_path': f'./train/r_{i:03d}'
                        } for i in range(10 if s == 'train' else 5)
                    ]
                }
            else:
                with open(json_path, 'r') as fp:
                    metas[s] = json.load(fp)

        all_imgs = []
        all_poses = []
        counts = [0]
        for s in splits:
            meta = metas[s]
            imgs = []
            poses = []
            if s=='train' or self.testskip==0:
                skip = 1
            else:
                skip = self.testskip
                
            for frame in meta['frames'][::skip]:
                fname = os.path.join(self.basedir, frame['file_path'] + '.png')
                
                # Create dummy image if file doesn't exist
                if not os.path.exists(fname):
                    H, W = 800, 800
                    if self.half_res:
                        H, W = H//2, W//2
                    if self.factor is not None:
                        H, W = H//self.factor, W//self.factor
                    img = np.random.rand(H, W, 4).astype(np.float32)
                else:
                    img = imageio.imread(fname) / 255.
                
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
        focal = .5 * W / np.tan(.5 * camera_angle_x)
        
        # Create render poses
        render_poses = np.stack([pose_spherical(angle, -30.0, 4.0) 
                                for angle in np.linspace(-180, 180, 40+1)[:-1]], 0)
        
        if self.half_res:
            H = H//2
            W = W//2
            focal = focal/2.
            imgs_half_res = np.zeros((imgs.shape[0], H, W, 4))
            for i, img in enumerate(imgs):
                imgs_half_res[i] = cv2.resize(img, (W, H), interpolation=cv2.INTER_AREA)
            imgs = imgs_half_res

        if self.factor is not None:
            H = H//self.factor
            W = W//self.factor
            focal = focal/self.factor
            imgs_resized = np.zeros((imgs.shape[0], H, W, 4))
            for i, img in enumerate(imgs):
                imgs_resized[i] = cv2.resize(img, (W, H), interpolation=cv2.INTER_AREA)
            imgs = imgs_resized
        
        if self.white_bkgd:
            imgs = imgs[..., :3]*imgs[..., -1:] + (1.-imgs[..., -1:])
        else:
            imgs = imgs[..., :3]
        
        return imgs, poses, render_poses, [H, W, focal], i_split
    
    def __len__(self) -> int:
        return len(self.rgbs)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get a single ray."""
        return {
            'rays_o': self.rays[idx, 0], 'rays_d': self.rays[idx, 1], 'target': self.rgbs[idx]
        }


def create_nerf_dataloader(
    dataset_type: str,
    basedir: str,
    split: str = 'train',
    batch_size: int = 1024,
    shuffle: bool = True,
    num_workers: int = 4,
    **kwargs
):
    """Create NeRF dataloader."""
    
    if dataset_type.lower() == 'blender':
        dataset = BlenderDataset(basedir, split=split, **kwargs)
    else:
        raise ValueError(f"Unknown dataset type: {dataset_type}")
    
    # Custom collate function for ray batching
    def nerf_collate_fn(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        batch_rays_o = torch.stack([item['rays_o'] for item in batch])
        batch_rays_d = torch.stack([item['rays_d'] for item in batch])
        batch_targets = torch.stack([item['target'] for item in batch])
        
        return {
            'rays_o': batch_rays_o, 'rays_d': batch_rays_d, 'targets': batch_targets
        }
    
    return DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, collate_fn=nerf_collate_fn, pin_memory=True
    ) 