from typing import Optional, Union
"""
Dataset module for DNMP-NeRF.

This module provides dataset classes for loading and preprocessing urban scene data
including KITTI-360, Waymo, and other point cloud + image datasets.
"""

import torch
import torch.utils.data as data
import numpy as np
import os
import json
from pathlib import Path
import cv2
from PIL import Image
import open3d as o3d

class DNMPDataset(data.Dataset):
    """Base dataset class for DNMP training."""
    
    def __init__(
        self,
        data_root: str,
        split: str = 'train',
        image_size: tuple[int,
        int] = (512, 512),
        max_points: int = 100000,
        voxel_size: float = 0.1,
        scene_bounds: tuple[float, float, float, float, float, float] = (-50, 50, -50, 50, -5, 5)
    ):
        
        self.data_root = Path(data_root)
        self.split = split
        self.image_size = image_size
        self.max_points = max_points
        self.voxel_size = voxel_size
        self.scene_bounds = scene_bounds
        
        # Will be populated by subclasses
        self.samples = []
        self.cameras = {}
        self.point_cloud = None
        
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
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
        if 'depth_path' in sample:
            depth = self.load_depth(sample['depth_path'])
        
        return {
            'image': image, 'rays_o': rays_o, 'rays_d': rays_d, 'camera': camera, 'depth': depth, 'sample_idx': idx
        }
    
    def load_image(self, image_path: str) -> torch.Tensor:
        """Load and preprocess image."""
        image = Image.open(image_path).convert('RGB')
        image = image.resize(self.image_size, Image.LANCZOS)
        image = np.array(image) / 255.0
        return torch.from_numpy(image).float()
    
    def load_depth(self, depth_path: str) -> torch.Tensor:
        """Load depth map."""
        depth = cv2.imread(depth_path, cv2.IMREAD_ANYDEPTH)
        depth = cv2.resize(depth, self.image_size, interpolation=cv2.INTER_NEAREST)
        depth = depth.astype(np.float32) / 1000.0  # Convert to meters
        return torch.from_numpy(depth).float()
    
    def load_camera(self, camera_id: str) -> dict[str, torch.Tensor]:
        """Load camera parameters."""
        if camera_id in self.cameras:
            return self.cameras[camera_id]
        else:
            raise ValueError(f"Camera {camera_id} not found")
    
    def generate_rays(
        self,
        camera: dict[str,
        torch.Tensor],
        image_shape: tuple[int,
        int]
    ):
        """Generate camera rays."""
        height, width = image_shape
        
        # Get camera parameters
        K = camera['K']  # Intrinsic matrix [3, 3]
        R = camera['R']  # Rotation matrix [3, 3]
        T = camera['T']  # Translation vector [3]
        
        # Generate pixel coordinates
        i, j = torch.meshgrid(
            torch.arange(
                width,
                dtype=torch.float32,
            )
        )
        
        # Convert to normalized device coordinates
        dirs = torch.stack([
            (i - K[0, 2]) / K[0, 0], (j - K[1, 2]) / K[1, 1], torch.ones_like(i)
        ], dim=-1)
        
        # Transform ray directions to world coordinates
        rays_d = torch.sum(dirs[..., None, :] * R.T, dim=-1)
        rays_d = rays_d / torch.norm(rays_d, dim=-1, keepdim=True)
        
        # Ray origins (camera center in world coordinates)
        rays_o = -torch.sum(R.T * T, dim=-1).expand_as(rays_d)
        
        return rays_o, rays_d
    
    def load_point_cloud(self, point_cloud_path: str) -> torch.Tensor:
        """Load and preprocess point cloud."""
        if point_cloud_path.endswith('.ply'):
            pcd = o3d.io.read_point_cloud(point_cloud_path)
            points = np.asarray(pcd.points)
        elif point_cloud_path.endswith('.npy'):
            points = np.load(point_cloud_path)
        else:
            raise ValueError(f"Unsupported point cloud format: {point_cloud_path}")
        
        # Subsample if too many points
        if len(points) > self.max_points:
            indices = np.random.choice(len(points), self.max_points, replace=False)
            points = points[indices]
        
        # Apply scene bounds if specified
        if self.scene_bounds is not None:
            min_bound = np.array(self.scene_bounds[:3])
            max_bound = np.array(self.scene_bounds[3:])
            
            mask = np.all(points >= min_bound, axis=1) & np.all(points <= max_bound, axis=1)
            points = points[mask]
        
        return torch.from_numpy(points).float()
    
    def voxelize_point_cloud(self, points: torch.Tensor) -> dict[str, torch.Tensor]:
        """Voxelize point cloud for DNMP initialization."""
        # Convert to numpy for processing
        points_np = points.numpy()
        
        # Compute voxel grid
        min_coords = points_np.min(axis=0)
        max_coords = points_np.max(axis=0)
        
        voxel_coords = np.floor((points_np - min_coords) / self.voxel_size).astype(int)
        
        # Find unique voxels
        unique_voxels, inverse_indices = np.unique(voxel_coords, axis=0, return_inverse=True)
        
        # Compute voxel centers
        voxel_centers = unique_voxels * self.voxel_size + min_coords + self.voxel_size / 2
        
        return {
            'voxel_centers': torch.from_numpy(
                voxel_centers,
            )
        }

class UrbanSceneDataset(DNMPDataset):
    """Generic urban scene dataset."""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.load_data()
    
    def load_data(self):
        """Load dataset metadata."""
        metadata_path = self.data_root / f"{self.split}_metadata.json"
        
        if metadata_path.exists():
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
            
            self.samples = metadata['samples']
            self.cameras = {cam['id']: self._parse_camera(cam) 
                           for cam in metadata['cameras']}
            
            # Load point cloud
            if 'point_cloud_path' in metadata:
                pcd_path = self.data_root / metadata['point_cloud_path']
                self.point_cloud = self.load_point_cloud(str(pcd_path))
        else:
            raise FileNotFoundError(f"Metadata file not found: {metadata_path}")
    
    def _parse_camera(self, camera_data: dict) -> dict[str, torch.Tensor]:
        """Parse camera parameters from metadata."""
        return {
            'K': torch.tensor(
                camera_data['intrinsic'],
                dtype=torch.float32,
            )
        }

class KITTI360Dataset(DNMPDataset):
    """KITTI-360 dataset for urban scene reconstruction."""
    
    def __init__(
        self,
        data_root: str,
        sequence: str = '2013_05_28_drive_0000_sync',
        *args,
        **kwargs
    ):
        
        self.sequence = sequence
        super().__init__(data_root, *args, **kwargs)
        self.load_kitti360_data()
    
    def load_kitti360_data(self):
        """Load KITTI-360 specific data."""
        sequence_path = self.data_root / self.sequence
        
        # Load calibration
        calib_path = self.data_root / 'calibration'
        self.load_calibration(calib_path)
        
        # Load poses
        poses_path = self.data_root / 'data_poses' / self.sequence / 'poses.txt'
        self.poses = self.load_poses(poses_path)
        
        # Load image paths
        image_dir = sequence_path / 'image_00' / 'data_rect'
        self.image_paths = sorted(list(image_dir.glob('*.png')))
        
        # Create samples
        self.samples = []
        for i, image_path in enumerate(self.image_paths):
            if i < len(self.poses):
                self.samples.append({
                    'image_path': str(
                        image_path,
                    )
                })
        
        # Load point cloud (if available)
        pcd_path = sequence_path / 'velodyne_points' / 'data'
        if pcd_path.exists():
            self.load_velodyne_data(pcd_path)
    
    def load_calibration(self, calib_path: Path):
        """Load KITTI-360 calibration data."""
        # Load camera intrinsics
        calib_cam_path = calib_path / 'perspective.txt'
        
        self.cameras = {}
        if calib_cam_path.exists():
            with open(calib_cam_path, 'r') as f:
                lines = f.readlines()
            
            for line in lines:
                if line.startswith('P_rect_00'):
                    P = np.array([float(x) for x in line.split()[1:]]).reshape(3, 4)
                    K = P[:3, :3]
                    
                    self.cameras['cam_00'] = {
                        'K': torch.from_numpy(K).float(), 'P': torch.from_numpy(P).float()
                    }
    
    def load_poses(self, poses_path: Path) -> list[torch.Tensor]:
        """Load camera poses."""
        poses = []
        
        if poses_path.exists():
            with open(poses_path, 'r') as f:
                lines = f.readlines()
            
            for line in lines:
                pose_data = [float(x) for x in line.split()]
                pose_matrix = np.array(pose_data).reshape(3, 4)
                
                # Convert to 4x4 homogeneous matrix
                pose_4x4 = np.eye(4)
                pose_4x4[:3, :4] = pose_matrix
                
                poses.append(torch.from_numpy(pose_4x4).float())
        
        return poses
    
    def load_velodyne_data(self, velodyne_dir: Path):
        """Load Velodyne point cloud data."""
        velodyne_files = sorted(list(velodyne_dir.glob('*.bin')))
        
        # Load first few point clouds and combine
        all_points = []
        for i, velo_file in enumerate(velodyne_files[:10]):  # Use first 10 scans
            points = np.fromfile(velo_file, dtype=np.float32).reshape(-1, 4)
            points = points[:, :3]  # Remove intensity
            
            # Transform to world coordinates using pose
            if i < len(self.poses):
                pose = self.poses[i].numpy()
                points_homo = np.concatenate([points, np.ones((len(points), 1))], axis=1)
                points_world = (pose @ points_homo.T).T[:, :3]
                all_points.append(points_world)
        
        if all_points:
            self.point_cloud = torch.from_numpy(np.concatenate(all_points)).float()
    
    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        """Get KITTI-360 sample with pose information."""
        sample = super().__getitem__(idx)
        
        # Add pose information
        if idx < len(self.poses):
            pose = self.poses[idx]
            
            # Update camera parameters with pose
            R = pose[:3, :3]
            T = pose[:3, 3]
            
            sample['camera']['R'] = R
            sample['camera']['T'] = T
        
        return sample

class WaymoDataset(DNMPDataset):
    """Waymo Open Dataset for urban scene reconstruction."""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.load_waymo_data()
    
    def load_waymo_data(self):
        """Load Waymo dataset."""
        # This would require the Waymo Open Dataset tools
        # For now, we provide a placeholder implementation
        
        waymo_dir = self.data_root / 'waymo_data'
        
        # Load processed Waymo data (images, poses, point clouds)
        if waymo_dir.exists():
            self.load_processed_waymo_data(waymo_dir)
        else:
            raise NotImplementedError("Waymo dataset loading not fully implemented")
    
    def load_processed_waymo_data(self, waymo_dir: Path):
        """Load preprocessed Waymo data."""
        # Load metadata
        metadata_path = waymo_dir / f"{self.split}_metadata.json"
        if metadata_path.exists():
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
            
            self.samples = metadata['samples']
            self.cameras = {cam['id']: self._parse_camera(cam) 
                           for cam in metadata['cameras']}
            
            # Load aggregated point cloud
            pcd_path = waymo_dir / 'aggregated_pointcloud.ply'
            if pcd_path.exists():
                self.point_cloud = self.load_point_cloud(str(pcd_path))

def create_dataset(dataset_type: str, *args, **kwargs) -> DNMPDataset:
    """
    Factory function to create datasets.
    
    Args:
        dataset_type: Type of dataset ('urban', 'kitti360', 'waymo')
        *args, **kwargs: Dataset-specific arguments
        
    Returns:
        Dataset instance
    """
    if dataset_type.lower() == 'urban':
        return UrbanSceneDataset(*args, **kwargs)
    elif dataset_type.lower() == 'kitti360':
        return KITTI360Dataset(*args, **kwargs)
    elif dataset_type.lower() == 'waymo':
        return WaymoDataset(*args, **kwargs)
    else:
        raise ValueError(f"Unknown dataset type: {dataset_type}")

def create_dataloader(
    dataset: DNMPDataset,
    batch_size: int = 1,
    shuffle: bool = True,
    num_workers: int = 4
):
    """Create DataLoader for DNMP dataset."""
    return data.DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, pin_memory=True, collate_fn=dnmp_collate_fn
    )

def dnmp_collate_fn(batch: list[dict]) -> dict[str, torch.Tensor]:
    """Custom collate function for DNMP datasets."""
    # Handle variable-sized data
    collated = {}
    
    for key in batch[0].keys():
        if key in ['image', 'rays_o', 'rays_d']:
            # Stack image and ray data
            collated[key] = torch.stack([item[key] for item in batch])
        elif key == 'camera':
            # Handle camera parameters
            collated[key] = {}
            for cam_key in batch[0][key].keys():
                if isinstance(batch[0][key][cam_key], torch.Tensor):
                    collated[key][cam_key] = torch.stack([item[key][cam_key] for item in batch])
                else:
                    collated[key][cam_key] = [item[key][cam_key] for item in batch]
        elif key == 'depth':
            # Handle depth (may be None for some samples)
            depths = [item[key] for item in batch if item[key] is not None]
            if depths:
                collated[key] = torch.stack(depths)
            else:
                collated[key] = None
        else:
            # Default handling
            collated[key] = [item[key] for item in batch]
    
    return collated 