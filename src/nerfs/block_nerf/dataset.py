from __future__ import annotations

"""
Dataset for Block-NeRF

This module handles data loading and preprocessing for Block-NeRF training, including multi-view images, camera poses, and metadata.
"""

from typing import Any, Optional


import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import cv2
import json
import os
from pathlib import Path
import random
from PIL import Image
import h5py

class BlockNeRFDataset(Dataset):
    """
    Dataset for Block-NeRF training
    
    Supports various data formats including COLMAP, LLFF, and custom formats.
    """
    
    def __init__(
        self,
        data_root: str | Path,
        split: str = 'train',
        img_scale: float = 1.0,
        ray_batch_size: int = 1024,
        use_cache: bool = True,
        cache_dir: Optional[str] = None,
        load_appearance_ids: bool = True,
        load_exposure: bool = True,
        background_color: Tuple[float,
        float,
        float] =,
    )
        """
        Initialize Block-NeRF dataset
        
        Args:
            data_root: Root directory containing the dataset
            split: Data split ('train', 'val', 'test')
            img_scale: Scale factor for images
            ray_batch_size: Number of rays per batch
            use_cache: Whether to use cached data
            cache_dir: Directory for cached data
            load_appearance_ids: Whether to load appearance embeddings
            load_exposure: Whether to load exposure values
            background_color: Background color for alpha compositing
        """
        self.data_root = Path(data_root)
        self.split = split
        self.img_scale = img_scale
        self.ray_batch_size = ray_batch_size
        self.use_cache = use_cache
        self.cache_dir = Path(cache_dir) if cache_dir else self.data_root / 'cache'
        self.load_appearance_ids = load_appearance_ids
        self.load_exposure = load_exposure
        self.background_color = torch.tensor(background_color, dtype=torch.float32)
        
        # Create cache directory
        if self.use_cache:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Load dataset metadata
        self.metadata = self._load_metadata()
        
        # Load images and poses
        self.images, self.poses, self.intrinsics = self._load_images_and_poses()
        
        # Load additional data
        self.appearance_ids = self._load_appearance_ids() if load_appearance_ids else None
        self.exposure_values = self._load_exposure_values() if load_exposure else None
        
        # Precompute rays if using cache
        self.rays = None
        self.colors = None
        if self.use_cache:
            self._precompute_rays()
        
        self.num_images = len(self.images)
        print(f"Loaded {self.num_images} images for {split} split")
    
    def _load_metadata(self) -> Dict[str, Any]:
        """Load dataset metadata"""
        metadata_path = self.data_root / 'metadata.json'
        
        if metadata_path.exists():
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
        else:
            # Default metadata
            metadata = {
                'scene_bounds': [[-100, 100], [-100, 100], [-10, 10]], 'near': 0.1, 'far': 100.0, 'format': 'colmap'
            }
        
        return metadata
    
    def _load_images_and_poses(self) -> Tuple[List[np.ndarray], np.ndarray, np.ndarray]:
        """Load images, camera poses, and intrinsics"""
        cache_path = self.cache_dir / f'{self.split}_images_poses.npz'
        
        if self.use_cache and cache_path.exists():
            print(f"Loading cached images and poses from {cache_path}")
            data = np.load(cache_path)
            images = [data[f'image_{i}'] for i in range(len([k for k in data.keys() if k.startswith('image_')]))]
            poses = data['poses']
            intrinsics = data['intrinsics']
            return images, poses, intrinsics
        
        # Detect data format and load accordingly
        if (self.data_root / 'sparse').exists():
            images, poses, intrinsics = self._load_colmap_data()
        elif (self.data_root / 'poses_bounds.npy').exists():
            images, poses, intrinsics = self._load_llff_data()
        else:
            images, poses, intrinsics = self._load_custom_data()
        
        # Cache the data
        if self.use_cache:
            cache_data = {'poses': poses, 'intrinsics': intrinsics}
            for i, img in enumerate(images):
                cache_data[f'image_{i}'] = img
            np.savez_compressed(cache_path, **cache_data)
            print(f"Cached images and poses to {cache_path}")
        
        return images, poses, intrinsics
    
    def _load_colmap_data(self) -> Tuple[List[np.ndarray], np.ndarray, np.ndarray]:
        """Load COLMAP format data"""
        from .utils.colmap_utils import read_cameras_binary, read_images_binary, read_points3D_binary
        
        sparse_dir = self.data_root / 'sparse' / '0'
        images_dir = self.data_root / 'images'
        
        # Read COLMAP data
        cameras = read_cameras_binary(sparse_dir / 'cameras.bin')
        images_data = read_images_binary(sparse_dir / 'images.bin')
        
        # Get split indices
        split_file = self.data_root / f'{self.split}.txt'
        if split_file.exists():
            with open(split_file, 'r') as f:
                split_names = [line.strip() for line in f.readlines()]
        else:
            # Use all images for training split
            split_names = [img.name for img in images_data.values()]
        
        # Load images and poses
        images = []
        poses = []
        intrinsics_list = []
        
        for img_data in images_data.values():
            if img_data.name not in split_names:
                continue
            
            # Load image
            img_path = images_dir / img_data.name
            if not img_path.exists():
                continue
            
            image = cv2.imread(str(img_path))
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Scale image if needed
            if self.img_scale != 1.0:
                h, w = image.shape[:2]
                new_h, new_w = int(h * self.img_scale), int(w * self.img_scale)
                image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
            
            image = image.astype(np.float32) / 255.0
            images.append(image)
            
            # Get camera intrinsics
            camera = cameras[img_data.camera_id]
            fx, fy, cx, cy = camera.params[:4]
            
            # Scale intrinsics
            fx *= self.img_scale
            fy *= self.img_scale
            cx *= self.img_scale
            cy *= self.img_scale
            
            intrinsics = np.array([
                [fx, 0, cx], [0, fy, cy], [0, 0, 1]
            ])
            intrinsics_list.append(intrinsics)
            
            # Get camera pose (world to camera)
            R = img_data.qvec2rotmat()
            t = img_data.tvec
            
            # Convert to camera to world
            pose = np.eye(4)
            pose[:3, :3] = R.T
            pose[:3, 3] = -R.T @ t
            poses.append(pose)
        
        poses = np.array(poses)
        intrinsics = np.array(intrinsics_list)
        
        return images, poses, intrinsics
    
    def _load_llff_data(self) -> Tuple[List[np.ndarray], np.ndarray, np.ndarray]:
        """Load LLFF format data"""
        poses_bounds = np.load(self.data_root / 'poses_bounds.npy')
        
        # Split poses and bounds
        poses = poses_bounds[:, :-2].reshape([-1, 3, 5])
        bounds = poses_bounds[:, -2:]
        
        # Get images
        images_dir = self.data_root / 'images'
        image_files = sorted(list(images_dir.glob('*.jpg')) + list(images_dir.glob('*.png')))
        
        images = []
        for img_path in image_files:
            image = cv2.imread(str(img_path))
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            if self.img_scale != 1.0:
                h, w = image.shape[:2]
                new_h, new_w = int(h * self.img_scale), int(w * self.img_scale)
                image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
            
            image = image.astype(np.float32) / 255.0
            images.append(image)
        
        # Extract intrinsics and poses
        H, W, focal = poses[0, :, -1]
        H, W = int(H * self.img_scale), int(W * self.img_scale)
        focal = focal * self.img_scale
        
        intrinsics = np.array([
            [focal, 0, W/2], [0, focal, H/2], [0, 0, 1]
        ])
        intrinsics = np.tile(intrinsics[None], (len(poses), 1, 1))
        
        # Convert poses to 4x4 matrices
        poses_4x4 = np.zeros((len(poses), 4, 4))
        poses_4x4[:, :3, :4] = poses[:, :, :4]
        poses_4x4[:, 3, 3] = 1.0
        
        return images, poses_4x4, intrinsics
    
    def _load_custom_data(self) -> Tuple[List[np.ndarray], np.ndarray, np.ndarray]:
        """Load custom format data"""
        # Implement custom data loading logic here
        raise NotImplementedError("Custom data format not implemented")
    
    def _load_appearance_ids(self) -> Optional[np.ndarray]:
        """Load appearance embedding IDs"""
        appearance_file = self.data_root / 'appearance_ids.npy'
        
        if appearance_file.exists():
            return np.load(appearance_file)
        else:
            # Generate sequential appearance IDs
            return np.arange(len(self.images))
    
    def _load_exposure_values(self) -> Optional[np.ndarray]:
        """Load exposure values"""
        exposure_file = self.data_root / 'exposure_values.npy'
        
        if exposure_file.exists():
            return np.load(exposure_file)
        else:
            # Default exposure values
            return np.ones(len(self.images))
    
    def _precompute_rays(self):
        """Precompute all rays and colors"""
        cache_path = self.cache_dir / f'{self.split}_rays.h5'
        
        if cache_path.exists():
            print(f"Loading cached rays from {cache_path}")
            with h5py.File(cache_path, 'r') as f:
                self.rays = {
                    'origins': f['ray_origins'][:], 'directions': f['ray_directions'][:], 'image_ids': f['image_ids'][:]
                }
                self.colors = f['colors'][:]
            return
        
        print("Precomputing rays...")
        all_ray_origins = []
        all_ray_directions = []
        all_colors = []
        all_image_ids = []
        
        for i, (image, pose, intrinsic) in enumerate(zip(self.images, self.poses, self.intrinsics)):
            H, W = image.shape[:2]
            
            # Generate ray origins and directions
            ray_origins, ray_directions = self._get_rays(H, W, intrinsic, pose)
            
            # Flatten
            ray_origins = ray_origins.reshape(-1, 3)
            ray_directions = ray_directions.reshape(-1, 3)
            colors = image.reshape(-1, 3)
            image_ids = np.full(ray_origins.shape[0], i)
            
            all_ray_origins.append(ray_origins)
            all_ray_directions.append(ray_directions)
            all_colors.append(colors)
            all_image_ids.append(image_ids)
        
        # Concatenate all rays
        self.rays = {
            'origins': np.concatenate(
                all_ray_origins,
                axis=0,
            )
        }
        self.colors = np.concatenate(all_colors, axis=0)
        
        # Cache the rays
        with h5py.File(cache_path, 'w') as f:
            f.create_dataset('ray_origins', data=self.rays['origins'])
            f.create_dataset('ray_directions', data=self.rays['directions'])
            f.create_dataset('image_ids', data=self.rays['image_ids'])
            f.create_dataset('colors', data=self.colors)
        
        print(f"Cached {len(self.colors)} rays to {cache_path}")
    
    def _get_rays(
        self,
        H: int,
        W: int,
        intrinsic: np.ndarray,
        pose: np.ndarray,
    )
        """Generate rays for a camera"""
        i, j = np.meshgrid(np.arange(W), np.arange(H), indexing='xy')
        
        # Pixel coordinates to camera coordinates
        dirs = np.stack([
            (
                i - intrinsic[0,
                2],
            )
        ], axis=-1)
        
        # Transform ray directions to world coordinates
        ray_directions = np.sum(dirs[..., None, :] * pose[:3, :3], axis=-1)
        
        # Ray origins (camera center in world coordinates)
        ray_origins = np.broadcast_to(pose[:3, 3], ray_directions.shape)
        
        return ray_origins, ray_directions
    
    def __len__(self) -> int:
        """Return dataset length"""
        if self.rays is not None:
            return len(self.colors) // self.ray_batch_size
        else:
            return len(self.images)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get a batch of rays"""
        if self.rays is not None:
            # Return batch of rays
            start_idx = idx * self.ray_batch_size
            end_idx = min(start_idx + self.ray_batch_size, len(self.colors))
            
            # Get ray batch
            ray_origins = torch.from_numpy(self.rays['origins'][start_idx:end_idx]).float()
            ray_directions = torch.from_numpy(self.rays['directions'][start_idx:end_idx]).float()
            colors = torch.from_numpy(self.colors[start_idx:end_idx]).float()
            image_ids = torch.from_numpy(self.rays['image_ids'][start_idx:end_idx]).long()
            
            # Get camera positions (from poses)
            unique_image_ids = torch.unique(image_ids)
            if len(unique_image_ids) == 1:
                camera_position = torch.from_numpy(
                    self.poses[unique_image_ids[0].item,
                )
            else:
                # Use mean position if multiple images
                positions = [self.poses[img_id.item()][:3, 3] for img_id in unique_image_ids]
                camera_position = torch.from_numpy(np.mean(positions, axis=0)).float()
            
            batch = {
                'ray_origins': ray_origins, 'ray_directions': ray_directions, 'rgb': colors, 'image_ids': image_ids, 'camera_positions': camera_position.unsqueeze(
                    0,
                )
            }
            
            # Add appearance IDs
            if self.appearance_ids is not None:
                appearance_ids = torch.from_numpy(self.appearance_ids[image_ids]).long()
                batch['appearance_ids'] = appearance_ids
            else:
                batch['appearance_ids'] = image_ids
            
            # Add exposure values
            if self.exposure_values is not None:
                exposure_values = torch.from_numpy(self.exposure_values[image_ids]).float().unsqueeze(-1)
                batch['exposure_values'] = exposure_values
            else:
                batch['exposure_values'] = torch.ones(len(image_ids), 1)
            
            return batch
        
        else:
            # Return full image
            image = torch.from_numpy(self.images[idx]).float()
            pose = torch.from_numpy(self.poses[idx]).float()
            intrinsic = torch.from_numpy(self.intrinsics[idx]).float()
            
            batch = {
                'image': image, 'pose': pose, 'intrinsic': intrinsic, 'image_id': torch.tensor(
                    idx,
                    dtype=torch.long,
                )
            }
            
            return batch
    
    def get_camera_positions(self) -> np.ndarray:
        """Get all camera positions"""
        return self.poses[:, :3, 3]
    
    def get_scene_bounds(self) -> Tuple[np.ndarray, np.ndarray]:
        """Get scene bounding box"""
        camera_positions = self.get_camera_positions()
        
        # Compute bounds from camera positions with some margin
        min_bounds = camera_positions.min(axis=0) - 10.0
        max_bounds = camera_positions.max(axis=0) + 10.0
        
        return min_bounds, max_bounds
    
    def create_dataloader(
        self,
        batch_size: int = 1,
        shuffle: bool = True,
        num_workers: int = 4,
    )
        """Create a DataLoader for this dataset"""
        return DataLoader(
            self, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, pin_memory=True
        )

    def get_metadata(self) -> Dict[str, Any]:
        """Get dataset metadata"""
        return self.metadata 