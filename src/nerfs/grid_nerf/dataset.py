"""
Dataset module for Grid-NeRF.

This module provides dataset classes for loading and preprocessing large-scale
urban scene data including KITTI-360, Waymo, and other autonomous driving datasets.
"""

import torch
import torch.utils.data as data
import numpy as np
import os
import json
from typing import Dict, List, Tuple, Optional, Union
from pathlib import Path
import cv2
from PIL import Image
import pickle


class GridNeRFDataset(data.Dataset):
    """Base dataset class for Grid-NeRF training on large urban scenes."""
    
    def __init__(self,
                 data_root: str,
                 split: str = 'train',
                 image_size: Tuple[int, int] = (512, 512),
                 scene_bounds: Tuple[float, ...] = (-100, -100, -10, 100, 100, 50),
                 subsample_factor: int = 1,
                 max_images: Optional[int] = None):
        
        self.data_root = Path(data_root)
        self.split = split
        self.image_size = image_size
        self.scene_bounds = scene_bounds
        self.subsample_factor = subsample_factor
        self.max_images = max_images
        
        # Will be populated by subclasses
        self.samples = []
        self.cameras = {}
        self.scene_scale = 1.0
        self.scene_center = np.array([0, 0, 0])
        
        self.load_data()
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get a training sample."""
        sample = self.samples[idx]
        
        # Load image
        image = self.load_image(sample['image_path'])
        
        # Load camera parameters
        camera = self.load_camera(sample['camera_id'])
        
        # Generate rays
        rays_o, rays_d = self.generate_rays(camera, image.shape[:2])
        
        # Load depth if available
        depth = None
        if 'depth_path' in sample and sample['depth_path'] is not None:
            depth = self.load_depth(sample['depth_path'])
        
        # Apply scene normalization
        rays_o = self.normalize_positions(rays_o)
        
        return {
            'image': image,
            'rays_o': rays_o,
            'rays_d': rays_d,
            'camera': camera,
            'depth': depth,
            'sample_idx': idx,
            'scene_bounds': torch.tensor(self.scene_bounds, dtype=torch.float32)
        }
    
    def load_data(self):
        """Load dataset metadata. To be implemented by subclasses."""
        raise NotImplementedError
    
    def load_image(self, image_path: str) -> torch.Tensor:
        """Load and preprocess image."""
        image = Image.open(image_path).convert('RGB')
        image = image.resize(self.image_size, Image.LANCZOS)
        image = np.array(image) / 255.0
        return torch.from_numpy(image).float()
    
    def load_depth(self, depth_path: str) -> torch.Tensor:
        """Load depth map."""
        if depth_path.endswith('.npy'):
            depth = np.load(depth_path)
        else:
            depth = cv2.imread(depth_path, cv2.IMREAD_ANYDEPTH)
            depth = depth.astype(np.float32) / 1000.0  # Convert to meters
        
        depth = cv2.resize(depth, self.image_size, interpolation=cv2.INTER_NEAREST)
        return torch.from_numpy(depth).float()
    
    def load_camera(self, camera_id: str) -> Dict[str, torch.Tensor]:
        """Load camera parameters."""
        if camera_id in self.cameras:
            return self.cameras[camera_id]
        else:
            raise ValueError(f"Camera {camera_id} not found")
    
    def generate_rays(self, camera: Dict[str, torch.Tensor], 
                     image_shape: Tuple[int, int]) -> Tuple[torch.Tensor, torch.Tensor]:
        """Generate camera rays."""
        height, width = image_shape
        
        # Get camera parameters
        K = camera['K']  # Intrinsic matrix [3, 3]
        R = camera['R']  # Rotation matrix [3, 3]
        T = camera['T']  # Translation vector [3]
        
        # Generate pixel coordinates
        i, j = torch.meshgrid(
            torch.arange(width, dtype=torch.float32),
            torch.arange(height, dtype=torch.float32),
            indexing='xy'
        )
        
        # Convert to normalized device coordinates
        dirs = torch.stack([
            (i - K[0, 2]) / K[0, 0],
            (j - K[1, 2]) / K[1, 1],
            torch.ones_like(i)
        ], dim=-1)
        
        # Transform ray directions to world coordinates
        rays_d = torch.sum(dirs[..., None, :] * R.T, dim=-1)
        rays_d = rays_d / torch.norm(rays_d, dim=-1, keepdim=True)
        
        # Ray origins (camera center in world coordinates)
        rays_o = -torch.sum(R.T * T, dim=-1).expand_as(rays_d)
        
        return rays_o, rays_d
    
    def normalize_positions(self, positions: torch.Tensor) -> torch.Tensor:
        """Normalize positions to scene bounds."""
        # Apply scene scaling and centering
        normalized = (positions - torch.tensor(self.scene_center)) * self.scene_scale
        return normalized
    
    def compute_scene_normalization(self, all_positions: np.ndarray):
        """Compute scene normalization parameters."""
        # Compute scene center and scale
        min_pos = np.min(all_positions, axis=0)
        max_pos = np.max(all_positions, axis=0)
        
        self.scene_center = (min_pos + max_pos) / 2
        scene_extent = np.max(max_pos - min_pos)
        self.scene_scale = 1.0 / (scene_extent * 0.5)  # Normalize to [-1, 1]


class KITTI360GridDataset(GridNeRFDataset):
    """KITTI-360 dataset for Grid-NeRF training."""
    
    def __init__(self, 
                 data_root: str,
                 sequence: str = '2013_05_28_drive_0000_sync',
                 camera_id: int = 0,
                 *args, **kwargs):
        
        self.sequence = sequence
        self.camera_id = camera_id
        super().__init__(data_root, *args, **kwargs)
    
    def load_data(self):
        """Load KITTI-360 dataset."""
        sequence_dir = self.data_root / 'data_2d_raw' / self.sequence
        
        if not sequence_dir.exists():
            # Create dummy data if not exists
            self.samples = []
            self.cameras = {}
            return
        
        # Load camera calibration
        calib_file = self.data_root / 'calibration' / 'calib_cam_to_pose.txt'
        if calib_file.exists():
            self.load_calibration(calib_file)
        else:
            # Default calibration
            self.K = np.array([[552.554, 0, 512], [0, 552.554, 384], [0, 0, 1]])
        
        # Load poses
        pose_file = self.data_root / 'data_poses' / self.sequence / 'poses.txt'
        if pose_file.exists():
            poses = self.load_poses(pose_file)
        else:
            poses = {}
        
        # Load image list
        image_dir = sequence_dir / f'image_0{self.camera_id}' / 'data_rect'
        if image_dir.exists():
            image_files = sorted(list(image_dir.glob('*.png')))
        else:
            image_files = []
        
        if self.max_images is not None:
            image_files = image_files[:self.max_images]
        
        # Subsample if needed
        if self.subsample_factor > 1:
            image_files = image_files[::self.subsample_factor]
        
        # Create samples
        all_positions = []
        
        for i, image_file in enumerate(image_files):
            frame_id = int(image_file.stem)
            
            if frame_id in poses:
                pose = poses[frame_id]
            else:
                # Default pose
                pose = np.eye(4)
                pose[:3, 3] = [i * 2.0, 0, 1.5]  # Simple trajectory
            
            camera_center = pose[:3, 3]
            all_positions.append(camera_center)
            
            sample = {
                'image_path': str(image_file),
                'camera_id': f'cam_{self.camera_id}_{frame_id}',
                'pose': pose,
                'frame_id': frame_id
            }
            
            # Check for depth
            depth_file = image_file.parent.parent / 'depth' / image_file.name
            if depth_file.exists():
                sample['depth_path'] = str(depth_file)
            
            self.samples.append(sample)
        
        # Compute scene normalization
        if all_positions:
            self.compute_scene_normalization(np.array(all_positions))
        
        # Store camera parameters
        for sample in self.samples:
            self.cameras[sample['camera_id']] = self._create_camera_params(sample)
    
    def load_calibration(self, calib_file: Path):
        """Load camera calibration from KITTI-360."""
        with open(calib_file, 'r') as f:
            lines = f.readlines()
        
        for line in lines:
            if line.startswith(f'image_0{self.camera_id}:'):
                # Parse camera matrix
                values = list(map(float, line.split()[1:]))
                self.K = np.array(values).reshape(3, 4)[:, :3]
                break
        else:
            # Default calibration
            self.K = np.array([[552.554, 0, 512], [0, 552.554, 384], [0, 0, 1]])
    
    def load_poses(self, pose_file: Path) -> Dict[int, np.ndarray]:
        """Load camera poses from KITTI-360."""
        poses = {}
        
        with open(pose_file, 'r') as f:
            for line in f:
                values = list(map(float, line.split()))
                frame_id = int(values[0])
                pose_matrix = np.array(values[1:]).reshape(3, 4)
                
                # Convert to 4x4 matrix
                pose = np.eye(4)
                pose[:3, :4] = pose_matrix
                poses[frame_id] = pose
        
        return poses
    
    def _create_camera_params(self, sample: Dict) -> Dict[str, torch.Tensor]:
        """Create camera parameters for a sample."""
        pose = sample['pose']
        
        # Extract rotation and translation
        R = pose[:3, :3]
        T = pose[:3, 3]
        
        return {
            'K': torch.from_numpy(self.K).float(),
            'R': torch.from_numpy(R).float(),
            'T': torch.from_numpy(T).float(),
            'pose': torch.from_numpy(pose).float()
        }


class WaymoGridDataset(GridNeRFDataset):
    """Waymo dataset for Grid-NeRF training."""
    
    def __init__(self, 
                 data_root: str,
                 segment_name: str,
                 camera_name: str = 'FRONT',
                 *args, **kwargs):
        
        self.segment_name = segment_name
        self.camera_name = camera_name
        super().__init__(data_root, *args, **kwargs)
    
    def load_data(self):
        """Load Waymo dataset."""
        segment_dir = self.data_root / self.segment_name
        
        if not segment_dir.exists():
            raise FileNotFoundError(f"Waymo segment not found: {segment_dir}")
        
        # Load metadata
        metadata_file = segment_dir / 'metadata.json'
        with open(metadata_file, 'r') as f:
            metadata = json.load(f)
        
        # Load camera calibration
        cameras = metadata['cameras']
        if self.camera_name not in cameras:
            raise ValueError(f"Camera {self.camera_name} not found in segment")
        
        self.camera_calib = cameras[self.camera_name]
        
        # Load poses
        poses_file = segment_dir / 'poses.json'
        with open(poses_file, 'r') as f:
            poses = json.load(f)
        
        # Load image list
        image_dir = segment_dir / 'images' / self.camera_name
        image_files = sorted(list(image_dir.glob('*.jpg')))
        
        if self.max_images is not None:
            image_files = image_files[:self.max_images]
        
        # Subsample if needed
        if self.subsample_factor > 1:
            image_files = image_files[::self.subsample_factor]
        
        # Create samples
        all_positions = []
        
        for image_file in image_files:
            timestamp = image_file.stem
            
            if timestamp in poses:
                pose_data = poses[timestamp]
                pose = np.array(pose_data['pose']).reshape(4, 4)
                camera_center = pose[:3, 3]
                all_positions.append(camera_center)
                
                sample = {
                    'image_path': str(image_file),
                    'camera_id': f'{self.camera_name}_{timestamp}',
                    'pose': pose,
                    'timestamp': timestamp
                }
                
                # Check for depth
                depth_file = segment_dir / 'depth' / self.camera_name / f'{timestamp}.png'
                if depth_file.exists():
                    sample['depth_path'] = str(depth_file)
                
                self.samples.append(sample)
        
        # Compute scene normalization
        if all_positions:
            self.compute_scene_normalization(np.array(all_positions))
        
        # Store camera parameters
        for sample in self.samples:
            self.cameras[sample['camera_id']] = self._create_camera_params(sample)
    
    def _create_camera_params(self, sample: Dict) -> Dict[str, torch.Tensor]:
        """Create camera parameters for a sample."""
        pose = sample['pose']
        calib = self.camera_calib
        
        # Camera intrinsics
        K = np.array([
            [calib['fx'], 0, calib['cx']],
            [0, calib['fy'], calib['cy']],
            [0, 0, 1]
        ])
        
        # Extract rotation and translation
        R = pose[:3, :3]
        T = pose[:3, 3]
        
        return {
            'K': torch.from_numpy(K).float(),
            'R': torch.from_numpy(R).float(),
            'T': torch.from_numpy(T).float(),
            'pose': torch.from_numpy(pose).float()
        }


class UrbanSceneGridDataset(GridNeRFDataset):
    """Generic urban scene dataset for Grid-NeRF."""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    
    def load_data(self):
        """Load generic urban scene dataset."""
        metadata_file = self.data_root / f"{self.split}_metadata.json"
        
        if not metadata_file.exists():
            raise FileNotFoundError(f"Metadata file not found: {metadata_file}")
        
        with open(metadata_file, 'r') as f:
            metadata = json.load(f)
        
        # Load samples
        self.samples = metadata['samples']
        
        # Load camera parameters
        cameras_data = metadata['cameras']
        self.cameras = {}
        
        for camera_data in cameras_data:
            camera_id = camera_data['id']
            self.cameras[camera_id] = {
                'K': torch.tensor(camera_data['K'], dtype=torch.float32),
                'R': torch.tensor(camera_data['R'], dtype=torch.float32),
                'T': torch.tensor(camera_data['T'], dtype=torch.float32),
                'pose': torch.tensor(camera_data['pose'], dtype=torch.float32)
            }
        
        # Compute scene normalization if positions are available
        if 'scene_info' in metadata:
            scene_info = metadata['scene_info']
            self.scene_center = np.array(scene_info['center'])
            self.scene_scale = scene_info['scale']


def create_dataloader(dataset: GridNeRFDataset,
                     batch_size: int = 1,
                     shuffle: bool = True,
                     num_workers: int = 4,
                     ray_batch_size: int = 4096) -> data.DataLoader:
    """
    Create dataloader for Grid-NeRF training.
    
    Args:
        dataset: Grid-NeRF dataset
        batch_size: Number of images per batch
        shuffle: Whether to shuffle data
        num_workers: Number of worker processes
        ray_batch_size: Number of rays to sample per image
        
    Returns:
        DataLoader instance
    """
    
    def collate_fn(batch):
        """Custom collate function for ray sampling."""
        if len(batch) == 1:
            # Single image case - sample rays
            sample = batch[0]
            
            # Get image and rays
            image = sample['image']  # [H, W, 3]
            rays_o = sample['rays_o']  # [H, W, 3]
            rays_d = sample['rays_d']  # [H, W, 3]
            
            height, width = image.shape[:2]
            
            # Sample random rays
            num_rays = min(ray_batch_size, height * width)
            ray_indices = torch.randperm(height * width)[:num_rays]
            
            # Convert to 2D coordinates
            ray_y = ray_indices // width
            ray_x = ray_indices % width
            
            # Sample rays and colors
            sampled_rays_o = rays_o[ray_y, ray_x]  # [num_rays, 3]
            sampled_rays_d = rays_d[ray_y, ray_x]  # [num_rays, 3]
            sampled_colors = image[ray_y, ray_x]  # [num_rays, 3]
            
            batch_data = {
                'rays_o': sampled_rays_o,
                'rays_d': sampled_rays_d,
                'target_colors': sampled_colors,
                'scene_bounds': sample['scene_bounds']
            }
            
            # Add depth if available
            if sample['depth'] is not None:
                sampled_depth = sample['depth'][ray_y, ray_x]
                batch_data['target_depth'] = sampled_depth
            
            return batch_data
        else:
            # Multiple images - return as is
            return torch.utils.data.dataloader.default_collate(batch)
    
    return data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True
    )


def create_dataset(dataset_type: str, *args, **kwargs) -> GridNeRFDataset:
    """
    Factory function to create datasets.
    
    Args:
        dataset_type: Type of dataset ('kitti360', 'waymo', 'urban')
        *args, **kwargs: Arguments for dataset constructor
        
    Returns:
        Dataset instance
    """
    if dataset_type.lower() == 'kitti360':
        return KITTI360GridDataset(*args, **kwargs)
    elif dataset_type.lower() == 'waymo':
        return WaymoGridDataset(*args, **kwargs)
    elif dataset_type.lower() == 'urban':
        return UrbanSceneGridDataset(*args, **kwargs)
    else:
        raise ValueError(f"Unknown dataset type: {dataset_type}")


class RayBatchSampler:
    """Utility class for efficient ray batch sampling."""
    
    def __init__(self, dataset: GridNeRFDataset, rays_per_batch: int = 4096):
        self.dataset = dataset
        self.rays_per_batch = rays_per_batch
        
        # Pre-compute ray information for all images
        self.ray_info = []
        self._precompute_rays()
    
    def _precompute_rays(self):
        """Pre-compute ray origins and directions for all images."""
        print("Pre-computing rays for dataset...")
        
        for i in range(len(self.dataset)):
            sample = self.dataset[i]
            
            rays_o = sample['rays_o']  # [H, W, 3]
            rays_d = sample['rays_d']  # [H, W, 3]
            image = sample['image']    # [H, W, 3]
            
            height, width = image.shape[:2]
            
            # Flatten rays
            rays_o_flat = rays_o.view(-1, 3)
            rays_d_flat = rays_d.view(-1, 3)
            colors_flat = image.view(-1, 3)
            
            ray_info = {
                'rays_o': rays_o_flat,
                'rays_d': rays_d_flat,
                'colors': colors_flat,
                'sample_idx': i,
                'scene_bounds': sample['scene_bounds']
            }
            
            if sample['depth'] is not None:
                depth_flat = sample['depth'].view(-1)
                ray_info['depth'] = depth_flat
            
            self.ray_info.append(ray_info)
    
    def sample_batch(self) -> Dict[str, torch.Tensor]:
        """Sample a batch of rays from random images."""
        # Sample random image
        img_idx = torch.randint(0, len(self.ray_info), (1,)).item()
        ray_data = self.ray_info[img_idx]
        
        # Sample random rays from this image
        num_rays = len(ray_data['rays_o'])
        ray_indices = torch.randperm(num_rays)[:self.rays_per_batch]
        
        batch = {
            'rays_o': ray_data['rays_o'][ray_indices],
            'rays_d': ray_data['rays_d'][ray_indices],
            'target_colors': ray_data['colors'][ray_indices],
            'scene_bounds': ray_data['scene_bounds']
        }
        
        if 'depth' in ray_data:
            batch['target_depth'] = ray_data['depth'][ray_indices]
        
        return batch 