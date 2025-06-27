from __future__ import annotations

"""
Plenoxels Dataset Module

This module provides dataset classes for loading and processing data for Plenoxels training.
Supports multiple data formats including COLMAP, Blender, and LLFF.

Key features:
- Efficient ray sampling and batching
- Multiple data format support
- Image preprocessing and augmentation
- Camera pose handling and normalization
"""

from typing import Any, Optional


import os
import json
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from dataclasses import dataclass
import cv2
from pathlib import Path
import imageio
import logging

logger = logging.getLogger(__name__)

@dataclass
class PlenoxelDatasetConfig:
    """Configuration for Plenoxel dataset."""
    
    # Data paths
    data_dir: str = ""
    dataset_type: str = "blender"  # blender, colmap, llff, instant_ngp
    
    # Image settings
    downsample_factor: int = 1
    white_background: bool = False
    load_every: int = 1  # Load every nth image
    
    # Train/test split
    train_skip: int = 1  # Use every nth image for training
    test_skip: int = 8   # Use every nth image for testing
    
    # Camera settings
    ndc: bool = False  # Use NDC coordinates
    spherify: bool = False  # Spherify poses
    
    # Ray sampling
    batch_size: int = 4096
    num_rays_train: int = 1024  # Rays per training iteration
    num_rays_eval: int = None   # All rays for evaluation
    precrop_fraction: float = 1.0  # Fraction of image to crop during early training
    precrop_iterations: int = 500
    
    # Data augmentation
    use_random_background: bool = False
    color_jitter: bool = False
    
    # Bounds
    near: float = 0.1
    far: float = 10.0
    scene_scale: float = 1.0

def load_blender_data(
    basedir: str,
    half_res: bool = False,
    testskip: int = 1,
    white_bkgd: bool = False,
) -> dict[str, np.ndarray]:
    """Load Blender synthetic dataset."""
    splits = ['train', 'val', 'test']
    metas = {}
    
    for s in splits:
        with open(os.path.join(basedir, f'transforms_{s}.json'), 'r') as fp:
            metas[s] = json.load(fp)
    
    all_imgs = []
    all_poses = []
    counts = [0]
    
    for s in splits:
        meta = metas[s]
        imgs = []
        poses = []
        
        if s == 'train' or testskip == 0:
            skip = 1
        else:
            skip = testskip
        
        for frame in meta['frames'][::skip]:
            fname = os.path.join(basedir, frame['file_path'] + '.png')
            img = imageio.imread(fname)
            
            # Convert to float and normalize
            img = (img / 255.0).astype(np.float32)
            
            # Handle alpha channel
            if img.shape[-1] == 4:
                if white_bkgd:
                    img = img[..., :3] * img[..., -1:] + (1. - img[..., -1:])
                else:
                    img = img[..., :3] * img[..., -1:]
            
            imgs.append(img)
            poses.append(np.array(frame['transform_matrix']))
        
        imgs = np.stack(imgs, axis=0)
        poses = np.stack(poses, axis=0)
        
        all_imgs.append(imgs)
        all_poses.append(poses)
        counts.append(counts[-1] + imgs.shape[0])
    
    i_split = [np.arange(counts[i], counts[i+1]) for i in range(3)]
    
    imgs = np.concatenate(all_imgs, axis=0)
    poses = np.concatenate(all_poses, axis=0)
    
    # Get camera parameters
    H, W = imgs.shape[1:3]
    camera_angle_x = float(metas['train']['camera_angle_x'])
    focal = 0.5 * W / np.tan(0.5 * camera_angle_x)
    
    if half_res:
        H = H // 2
        W = W // 2
        focal = focal / 2.0
        imgs_half_res = np.zeros((imgs.shape[0], H, W, imgs.shape[-1]))
        for i, img in enumerate(imgs):
            imgs_half_res[i] = cv2.resize(img, (W, H), interpolation=cv2.INTER_AREA)
        imgs = imgs_half_res
    
    return {
        'images': imgs, 'poses': poses, 'i_split': i_split, 'H': H, 'W': W, 'focal': focal, 'near': 2.0, 'far': 6.0
    }

def load_colmap_data(
    basedir: str,
    factor: int = 8,
    width: Optional[int] = None,
    height: Optional[int] = None,
    load_imgs: bool = True,
) -> dict[str, np.ndarray]:
    """Load COLMAP dataset."""
    try:
        from .colmap_utils import read_cameras_binary, read_images_binary, read_points3d_binary
    except ImportError:
        logger.warning("COLMAP utils not available. Please install colmap_utils.")
        return {}
    
    camerasfile = os.path.join(basedir, 'sparse/0/cameras.bin')
    camdata = read_cameras_binary(camerasfile)
    
    # Get camera parameters
    list_of_keys = list(camdata.keys())
    cam = camdata[list_of_keys[0]]
    
    if cam.model == "SIMPLE_PINHOLE":
        H, W, f = cam.height, cam.width, cam.params[0]
        hwf = np.array([H, W, f]).reshape([3, 1])
        K = np.array([
            [f, 0, W/2], [0, f, H/2], [0, 0, 1]
        ])
    elif cam.model == "PINHOLE":
        H, W, fx, fy, cx, cy = cam.height, cam.width, *cam.params
        hwf = np.array([H, W, (fx + fy) / 2]).reshape([3, 1])
        K = np.array([
            [fx, 0, cx], [0, fy, cy], [0, 0, 1]
        ])
    else:
        raise ValueError(f"Unsupported camera model: {cam.model}")
    
    # Read images
    imagesfile = os.path.join(basedir, 'sparse/0/images.bin')
    imdata = read_images_binary(imagesfile)
    
    w2c_mats = []
    bottom = np.array([0, 0, 0, 1]).reshape([1, 4])
    
    for k in imdata:
        im = imdata[k]
        R = im.qvec2rotmat()
        t = im.tvec.reshape([3, 1])
        m = np.concatenate([np.concatenate([R, t], 1), bottom], 0)
        w2c_mats.append(m)
    
    w2c_mats = np.stack(w2c_mats, 0)
    c2w_mats = np.linalg.inv(w2c_mats)
    
    poses = c2w_mats[:, :3, :4]  # [N, 3, 4]
    
    if load_imgs:
        imgdir = os.path.join(basedir, 'images')
        imgfiles = [os.path.join(imgdir, im.name) for im in imdata.values()]
        
        def imread(f):
            if f.endswith('png'):
                return imageio.imread(f, ignoregamma=True)
            else:
                return imageio.imread(f)
        
        imgs = [imread(f)[..., :3] / 255. for f in imgfiles]
        imgs = np.stack(imgs, axis=0)
        
        # Downsample if needed
        if factor > 1:
            sh = imgs.shape
            imgs = imgs.reshape([sh[0], sh[1] // factor, factor, sh[2] // factor, factor, sh[3]])
            imgs = imgs.mean(4).mean(2)
            H, W = H // factor, W // factor
            hwf = [H, W, hwf[2] / factor]
    else:
        imgs = None
    
    return {
        'images': imgs, 'poses': poses, 'hwf': hwf, 'K': K, 'near': 0.1, 'far': 100.0
    }

class PlenoxelDataset(Dataset):
    """Main dataset class for Plenoxels."""
    
    def __init__(self, config: PlenoxelDatasetConfig, split: str = 'train') -> None:
        """
        Initialize Plenoxel dataset.
        
        Args:
            config: Dataset configuration
            split: Dataset split ('train', 'val', 'test')
        """
        self.config = config
        self.split = split
        
        # Load data based on dataset type
        if config.dataset_type == 'blender':
            self.data = load_blender_data(
                config.data_dir, half_res=(
                    config.downsample_factor > 1,
                )
            )
        elif config.dataset_type == 'colmap':
            self.data = load_colmap_data(
                config.data_dir, factor=config.downsample_factor
            )
        else:
            raise ValueError(f"Unsupported dataset type: {config.dataset_type}")
        
        # Extract split-specific data
        self._setup_split_data()
        
        # Setup ray sampling
        self._setup_ray_sampling()
    
    def _setup_split_data(self) -> None:
        """Setup data for the current split."""
        if self.config.dataset_type == 'blender':
            i_split = self.data['i_split']
            if self.split == 'train':
                self.indices = i_split[0]
            elif self.split == 'val':
                self.indices = i_split[1]
            else:  # test
                self.indices = i_split[2]
        else:
            # For other datasets, manually split
            n_imgs = len(self.data['images'])
            if self.split == 'train':
                self.indices = np.arange(0, n_imgs, self.config.train_skip)
            elif self.split == 'val':
                self.indices = np.arange(1, n_imgs, self.config.test_skip)
            else:  # test
                self.indices = np.arange(2, n_imgs, self.config.test_skip)
        
        # Extract images and poses for this split
        self.images = self.data['images'][self.indices]
        self.poses = self.data['poses'][self.indices]
        
        # Get image dimensions
        if 'H' in self.data and 'W' in self.data:
            self.H, self.W = self.data['H'], self.data['W']
            self.focal = self.data['focal']
        elif 'hwf' in self.data:
            self.H, self.W, self.focal = self.data['hwf']
        
        # Get near/far bounds
        self.near = self.data.get('near', self.config.near)
        self.far = self.data.get('far', self.config.far)
        
        logger.info(f"Loaded {len(self.images)} {self.split} images ({self.H}x{self.W})")
    
    def _setup_ray_sampling(self) -> None:
        """Setup ray sampling for efficient training."""
        # Generate all rays for this split
        self.rays_o, self.rays_d = self._get_rays_np()
        
        if self.split == 'train':
            # Flatten for random sampling
            self.rays_o = self.rays_o.reshape(-1, 3)
            self.rays_d = self.rays_d.reshape(-1, 3)
            self.images_flat = self.images.reshape(-1, 3)
            
            # Pre-crop for early training
            if self.config.precrop_fraction < 1.0:
                dH = int(self.H // 2 * self.config.precrop_fraction)
                dW = int(self.W // 2 * self.config.precrop_fraction)
                coords = torch.stack(
                    torch.meshgrid(
                        torch.linspace(
                            self.H // 2 - dH,
                            self.H // 2 + dH - 1,
                            2 * dH,
                        )
                    ), -1)
                coords = torch.reshape(coords, [-1, 2]).long()
                
                # Create precrop indices
                self.precrop_indices = coords[:, 0] * self.W + coords[:, 1]
                self.precrop_indices = self.precrop_indices.numpy()
    
    def _get_rays_np(self) -> tuple[np.ndarray, np.ndarray]:
        """Generate ray origins and directions for all images."""
        rays_o = []
        rays_d = []
        
        for pose in self.poses:
            # Generate rays for this image
            i, j = np.meshgrid(np.arange(self.W, dtype=np.float32), np.arange(self.H, dtype=np.float32), indexing='ij')
            
            # Convert to camera coordinates
            dirs = np.stack([(i - self.W * 0.5) / self.focal, -(j - self.H * 0.5) / self.focal, -np.ones_like(i)], dim=-1)
            
            # Transform ray directions to world coordinates
            rays_d_cam = dirs @ pose[:3, :3].T
            rays_o_cam = np.broadcast_to(pose[:3, -1], rays_d_cam.shape)
            
            rays_o.append(rays_o_cam)
            rays_d.append(rays_d_cam)
        
        rays_o = np.stack(rays_o, axis=0)  # [N, H, W, 3]
        rays_d = np.stack(rays_d, axis=0)  # [N, H, W, 3]
        
        return rays_o, rays_d
    
    def __len__(self) -> int:
        """Get dataset length."""
        if self.split == 'train':
            return len(self.rays_o) // self.config.num_rays_train
        else:
            return len(self.images)
    
    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        """Get a batch of data."""
        if self.split == 'train':
            return self._get_train_batch(idx)
        else:
            return self._get_eval_batch(idx)
    
    def _get_train_batch(self, idx: int) -> dict[str, torch.Tensor]:
        """Get training batch with random ray sampling."""
        # Random ray sampling
        if hasattr(self, 'precrop_indices') and idx < self.config.precrop_iterations:
            # Use pre-cropped region for early training
            ray_indices = np.random.choice(
                self.precrop_indices, size=self.config.num_rays_train, replace=False
            )
        else:
            # Random sampling from all rays
            ray_indices = np.random.choice(
                len(self.rays_o), size=self.config.num_rays_train, replace=False
            )
        
        # Get sampled rays and colors
        rays_o = torch.from_numpy(self.rays_o[ray_indices]).float()
        rays_d = torch.from_numpy(self.rays_d[ray_indices]).float()
        colors = torch.from_numpy(self.images_flat[ray_indices]).float()
        
        return {
            'rays_o': rays_o, 'rays_d': rays_d, 'colors': colors, 'near': torch.tensor(
                self.near,
            )
        }
    
    def _get_eval_batch(self, idx: int) -> dict[str, torch.Tensor]:
        """Get evaluation batch (full image)."""
        # Return full image data
        rays_o = torch.from_numpy(self.rays_o[idx]).float()  # [H, W, 3]
        rays_d = torch.from_numpy(self.rays_d[idx]).float()  # [H, W, 3]
        colors = torch.from_numpy(self.images[idx]).float()  # [H, W, 3]
        
        return {
            'rays_o': rays_o, 'rays_d': rays_d, 'colors': colors, 'near': torch.tensor(
                self.near,
            )
        }
    
    def get_test_poses(self, n_poses: int = 40) -> np.ndarray:
        """Generate test poses for novel view synthesis."""
        # Generate spiral poses
        poses = self.poses
        c2w = poses.mean(axis=0)
        
        # Compute spiral path
        up = normalize(poses[:, :3, 1].sum(0))
        close_depth, inf_depth = self.near, self.far
        
        # Create spiral path
        render_poses = []
        for theta in np.linspace(0., 2. * np.pi, n_poses, endpoint=False):
            # Rotate around up axis
            rotate = np.array([[np.cos(theta)], [np.sin(theta)], [0]])
            
            pose = c2w @ rotate
            render_poses.append(pose[:3, :4])
        
        return np.stack(render_poses, axis=0)

def normalize(x: np.ndarray) -> np.ndarray:
    """Normalize vector."""
    return x / np.linalg.norm(x)

def create_plenoxel_dataloader(
    config: PlenoxelDatasetConfig,
    split: str = 'train',
    shuffle: bool = None,
) -> DataLoader:
    """Create a DataLoader for Plenoxel dataset."""
    dataset = PlenoxelDataset(config, split)
    
    if shuffle is None:
        shuffle = (split == 'train')
    
    return DataLoader(
        dataset, batch_size=1, # Handle batching internally
        shuffle=shuffle, num_workers=0, # Single threaded for ray sampling
        pin_memory=True
    )

def create_plenoxel_dataset(
    data_dir: str,
    dataset_type: str = 'blender',
    downsample_factor: int = 1,
    **kwargs: Any,
) -> PlenoxelDatasetConfig:
    """Create a Plenoxel dataset configuration."""
    config = PlenoxelDatasetConfig(
        data_dir=data_dir, dataset_type=dataset_type, downsample_factor=downsample_factor, **kwargs
    )
    
    return config 