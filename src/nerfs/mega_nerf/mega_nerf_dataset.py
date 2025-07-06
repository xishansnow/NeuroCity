from __future__ import annotations

"""
Mega-NeRF Dataset Module

This module handles data loading, preprocessing, and partitioning
for Mega-NeRF training and evaluation.
"""

from typing import Any, Optional, Union


import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import cv2
import json
import os
from pathlib import Path
import logging
from dataclasses import dataclass
import pickle
import h5py
from PIL import Image

logger = logging.getLogger(__name__)

@dataclass
class CameraInfo:
    """Camera information structure"""
    transform_matrix: np.ndarray  # 4x4 camera-to-world transform
    intrinsics: np.ndarray       # 3x3 intrinsic matrix
    image_path: str              # Path to image file
    image_id: int                # Unique image identifier
    width: int                   # Image width
    height: int                  # Image height
    
    def get_position(self) -> np.ndarray:
        """Get camera position in world coordinates"""
        return self.transform_matrix[:3, 3]
    
    def get_rotation(self) -> np.ndarray:
        """Get camera rotation matrix"""
        return self.transform_matrix[:3, :3]
    
    def get_focal_length(self) -> Tuple[float, float]:
        """Get focal lengths (fx, fy)"""
        return self.intrinsics[0, 0], self.intrinsics[1, 1]
    
    def get_principal_point(self) -> Tuple[float, float]:
        """Get principal point (cx, cy)"""
        return self.intrinsics[0, 2], self.intrinsics[1, 2]

class CameraDataset:
    """Dataset for managing camera information and images"""
    
    def __init__(
        self,
        data_root: str,
        split: str = 'train',
        image_scale: float = 1.0,
        load_images: bool = True,
    ):
        """
        Initialize camera dataset
        
        Args:
            data_root: Root directory containing the dataset
            split: Data split ('train', 'val', 'test')
            image_scale: Scale factor for images
            load_images: Whether to load image data
        """
        self.data_root = Path(data_root)
        self.split = split
        self.image_scale = image_scale
        self.load_images = load_images
        
        # Load camera data
        self.cameras = self._load_cameras()
        
        # Load images if requested
        self.images = {}
        if load_images:
            self._load_images()
        
        logger.info(f"Loaded {len(self.cameras)} cameras for {split} split")
    
    def _load_cameras(self) -> List[CameraInfo]:
        """Load camera information from various formats"""
        # Try different data formats
        if (self.data_root / 'transforms.json').exists():
            return self._load_nerf_format()
        elif (self.data_root / 'sparse').exists():
            return self._load_colmap_format()
        elif (self.data_root / 'poses_bounds.npy').exists():
            return self._load_llff_format()
        else:
            # Create synthetic data for demo
            return self._create_synthetic_cameras()
    
    def _load_nerf_format(self) -> List[CameraInfo]:
        """Load NeRF-style transforms.json format"""
        transforms_file = self.data_root / 'transforms.json'
        
        with open(transforms_file, 'r') as f:
            transforms = json.load(f)
        
        cameras = []
        
        # Get camera intrinsics
        if 'fl_x' in transforms and 'fl_y' in transforms:
            fx, fy = transforms['fl_x'], transforms['fl_y']
        elif 'camera_angle_x' in transforms:
            # Compute focal length from field of view
            w = transforms.get('w', 800)
            fx = fy = w / (2 * np.tan(transforms['camera_angle_x'] / 2))
        else:
            fx = fy = 800  # Default focal length
        
        cx = transforms.get('cx', transforms.get('w', 800) / 2)
        cy = transforms.get('cy', transforms.get('h', 600) / 2)
        
        intrinsics = np.array([
            [fx, 0, cx], [0, fy, cy], [0, 0, 1]
        ])
        
        # Scale intrinsics
        intrinsics[:2] *= self.image_scale
        
        # Load frame information
        for i, frame in enumerate(transforms['frames']):
            # Skip if not in the correct split
            if self.split in frame.get('split', self.split):
                continue
            
            # Transform matrix
            transform_matrix = np.array(frame['transform_matrix'])
            
            # Image path
            image_path = self.data_root / frame['file_path']
            if not image_path.suffix:
                # Try common extensions
                for ext in ['.png', '.jpg', '.jpeg']:
                    if (self.data_root / (frame['file_path'] + ext)).exists():
                        image_path = self.data_root / (frame['file_path'] + ext)
                        break
            
            # Get image dimensions
            if image_path.exists():
                with Image.open(image_path) as img:
                    width, height = img.size
                    width = int(width * self.image_scale)
                    height = int(height * self.image_scale)
            else:
                width, height = 800, 600
            
            camera = CameraInfo(
                transform_matrix=transform_matrix,
                intrinsics=intrinsics,
                image_path=str(image_path)
            )
            
            cameras.append(camera)
        
        return cameras
    
    def _load_colmap_format(self) -> List[CameraInfo]:
        """Load COLMAP format data"""
        try:
            from .utils.colmap_utils import read_cameras_binary, read_images_binary
        except ImportError:
            logger.error("COLMAP utils not available")
            return self._create_synthetic_cameras()
        
        sparse_dir = self.data_root / 'sparse' / '0'
        images_dir = self.data_root / 'images'
        
        # Read COLMAP data
        cameras_data = read_cameras_binary(sparse_dir / 'cameras.bin')
        images_data = read_images_binary(sparse_dir / 'images.bin')
        
        # Get split file
        split_file = self.data_root / f'{self.split}.txt'
        if split_file.exists():
            with open(split_file, 'r') as f:
                split_names = set(line.strip() for line in f.readlines())
        else:
            split_names = None
        
        cameras = []
        
        for img_id, img_data in images_data.items():
            # Check split
            if split_names and img_data.name not in split_names:
                continue
            
            # Get camera parameters
            camera_data = cameras_data[img_data.camera_id]
            
            # Build intrinsics matrix
            if camera_data.model == 'PINHOLE':
                fx, fy, cx, cy = camera_data.params
            elif camera_data.model == 'SIMPLE_PINHOLE':
                f, cx, cy = camera_data.params
                fx = fy = f
            else:
                logger.warning(f"Unsupported camera model: {camera_data.model}")
                continue
            
            intrinsics = np.array([
                [fx, 0, cx], [0, fy, cy], [0, 0, 1]
            ]) * self.image_scale
            
            # Build transform matrix (COLMAP to NeRF convention)
            R = img_data.qvec2rotmat()
            t = img_data.tvec
            
            # Convert from COLMAP (camera-to-world) to NeRF convention
            transform_matrix = np.eye(4)
            transform_matrix[:3, :3] = R.T
            transform_matrix[:3, 3] = -R.T @ t
            
            # Image path
            image_path = images_dir / img_data.name
            
            # Get image dimensions
            if image_path.exists():
                with Image.open(image_path) as img:
                    width, height = img.size
                    width = int(width * self.image_scale)
                    height = int(height * self.image_scale)
            else:
                width = int(camera_data.width * self.image_scale)
                height = int(camera_data.height * self.image_scale)
            
            camera = CameraInfo(
                transform_matrix=transform_matrix,
                intrinsics=intrinsics,
                image_path=str(image_path)
            )
            
            cameras.append(camera)
        
        return cameras
    
    def _load_llff_format(self) -> List[CameraInfo]:
        """Load LLFF format data"""
        poses_bounds = np.load(self.data_root / 'poses_bounds.npy')
        
        # Split poses and bounds
        poses = poses_bounds[:, :-2].reshape([-1, 3, 5])
        bounds = poses_bounds[:, -2:]
        
        # Get image files
        images_dir = self.data_root / 'images'
        image_files = sorted(list(images_dir.glob('*.jpg')) + list(images_dir.glob('*.png')))
        
        cameras = []
        
        for i, (pose, image_file) in enumerate(zip(poses, image_files)):
            # Extract camera parameters
            H, W, focal = pose[:, -1]
            H, W = int(H * self.image_scale), int(W * self.image_scale)
            focal = focal * self.image_scale
            
            # Build intrinsics
            intrinsics = np.array([
                [focal, 0, W/2], [0, focal, H/2], [0, 0, 1]
            ])
            
            # Build transform matrix
            transform_matrix = np.eye(4)
            transform_matrix[:3, :4] = pose[:, :4]
            
            camera = CameraInfo(
                transform_matrix=transform_matrix,
                intrinsics=intrinsics,
                image_path=str(image_file)
            )
            
            cameras.append(camera)
        
        return cameras
    
    def _create_synthetic_cameras(self) -> List[CameraInfo]:
        """Create synthetic camera data for demo purposes"""
        cameras = []
        
        # Create circular camera path
        num_cameras = 100 if self.split == 'train' else 20
        radius = 50
        height = 20
        
        for i in range(num_cameras):
            angle = i * 2 * np.pi / num_cameras
            
            # Camera position
            pos = np.array([
                radius * np.cos(angle), radius * np.sin(angle), height
            ])
            
            # Look at center
            target = np.array([0, 0, 0])
            up = np.array([0, 0, 1])
            
            # Build rotation matrix
            forward = target - pos
            forward = forward / np.linalg.norm(forward)
            right = np.cross(forward, up)
            right = right / np.linalg.norm(right)
            up = np.cross(right, forward)
            
            # Transform matrix
            transform_matrix = np.eye(4)
            transform_matrix[:3, 0] = right
            transform_matrix[:3, 1] = up
            transform_matrix[:3, 2] = -forward
            transform_matrix[:3, 3] = pos
            
            # Intrinsics
            focal = 800 * self.image_scale
            width, height = int(800 * self.image_scale), int(600 * self.image_scale)
            intrinsics = np.array([
                [focal, 0, width/2], [0, focal, height/2], [0, 0, 1]
            ])
            
            camera = CameraInfo(
                transform_matrix=transform_matrix,
                intrinsics=intrinsics,
                image_path=f"synthetic_image_{i:03d}"
            )
            
            cameras.append(camera)
        
        logger.info(f"Created {len(cameras)} synthetic cameras")
        return cameras
    
    def _load_images(self) -> None:
        """Load image data"""
        for camera in self.cameras:
            if Path(camera.image_path).exists():
                image = cv2.imread(camera.image_path)
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                
                # Scale image
                if self.image_scale != 1.0:
                    image = cv2.resize(image, None, fx=self.image_scale, fy=self.image_scale)
                
                image = image.astype(np.float32) / 255.0
                self.images[camera.image_id] = image
            else:
                # Create synthetic image
                image = np.random.rand(camera.height, camera.width, 3).astype(np.float32)
                self.images[camera.image_id] = image
    
    def get_camera_positions(self) -> np.ndarray:
        """Get all camera positions"""
        positions = []
        for camera in self.cameras:
            positions.append(camera.get_position())
        return np.array(positions)
    
    def get_scene_bounds(self) -> Tuple[np.ndarray, np.ndarray]:
        """Estimate scene bounds from camera positions"""
        positions = self.get_camera_positions()
        
        # Compute bounds with margin
        margin = 10.0
        min_bounds = positions.min(axis=0) - margin
        max_bounds = positions.max(axis=0) + margin
        
        return min_bounds, max_bounds
    
    def get_camera(self, camera_id: int) -> CameraInfo:
        """Get camera by ID"""
        for camera in self.cameras:
            if camera.image_id == camera_id:
                return camera
        raise ValueError(f"Camera {camera_id} not found")
    
    def get_image(self, camera_id: int) -> Optional[np.ndarray]:
        """Get image by camera ID"""
        return self.images.get(camera_id)

class MegaNeRFDataset(Dataset):
    """Dataset for Mega-NeRF training with spatial partitioning"""
    
    def __init__(
        self,
        data_root: str,
        partitioner,
        split: str = 'train',
        ray_batch_size: int = 1024,
        image_scale: float = 1.0,
        use_cache: bool = True,
        cache_dir: Optional[str] = None,
    ):
        """
        Initialize Mega-NeRF dataset
        
        Args:
            data_root: Root directory containing the dataset
            partitioner: Spatial partitioner instance
            split: Data split ('train', 'val', 'test')
            ray_batch_size: Number of rays per batch
            image_scale: Scale factor for images
            use_cache: Whether to use cached data
            cache_dir: Directory for cached data
        """
        self.data_root = Path(data_root)
        self.partitioner = partitioner
        self.split = split
        self.ray_batch_size = ray_batch_size
        self.image_scale = image_scale
        self.use_cache = use_cache
        self.cache_dir = Path(cache_dir) if cache_dir else self.data_root / 'cache'
        
        # Create cache directory
        if self.use_cache:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Load camera dataset
        self.camera_dataset = CameraDataset(
            data_root=data_root, split=split, image_scale=image_scale, load_images=True
        )
        
        # Create data partitions
        self.data_partitions = self._create_data_partitions()
        
        # Precompute rays if using cache
        self.rays_cache = {}
        if self.use_cache:
            self._precompute_rays()
        
        logger.info(f"Created {len(self.data_partitions)} data partitions")
    
    def _create_data_partitions(self) -> List[Dict[str, Any]]:
        """Create data partitions based on spatial partitioner"""
        # Get camera positions
        camera_positions = self.camera_dataset.get_camera_positions()
        
        # Assign cameras to partitions
        assignments = self.partitioner.assign_points_to_partitions(camera_positions)
        
        # Group cameras by partition
        partitions = []
        num_partitions = len(self.partitioner.partitions)
        
        for partition_idx in range(num_partitions):
            mask = assignments == partition_idx
            partition_cameras = [self.camera_dataset.cameras[i] for i in range(len(self.camera_dataset.cameras)) if mask[i]]
            
            partition_data = {
                'partition_idx': partition_idx,
                'cameras': partition_cameras,
                'camera_indices': np.where(mask)
            }
            
            partitions.append(partition_data)
        
        return partitions
    
    def _precompute_rays(self) -> None:
        """Precompute rays for all partitions"""
        for partition_idx, partition in enumerate(self.data_partitions):
            cache_file = self.cache_dir / f'{self.split}_partition_{partition_idx}_rays.h5'
            
            if cache_file.exists():
                logger.info(f"Loading cached rays for partition {partition_idx}")
                with h5py.File(cache_file, 'r') as f:
                    self.rays_cache[partition_idx] = {
                        'ray_origins': f['ray_origins'][:], 'ray_directions': f['ray_directions'][:], 'colors': f['colors'][:], 'camera_ids': f['camera_ids'][:]
                    }
            else:
                logger.info(f"Precomputing rays for partition {partition_idx}")
                rays_data = self._compute_partition_rays(partition)
                
                # Cache the rays
                with h5py.File(cache_file, 'w') as f:
                    f.create_dataset('ray_origins', data=rays_data['ray_origins'])
                    f.create_dataset('ray_directions', data=rays_data['ray_directions'])
                    f.create_dataset('colors', data=rays_data['colors'])
                    f.create_dataset('camera_ids', data=rays_data['camera_ids'])
                
                self.rays_cache[partition_idx] = rays_data
    
    def _compute_partition_rays(self, partition: Dict[str, Any]) -> Dict[str, np.ndarray]:
        """Compute rays for a specific partition"""
        all_ray_origins = []
        all_ray_directions = []
        all_colors = []
        all_camera_ids = []
        
        for camera in partition['cameras']:
            # Generate rays for this camera
            ray_origins, ray_directions = self._generate_camera_rays(camera)
            
            # Get image colors
            image = self.camera_dataset.get_image(camera.image_id)
            if image is not None:
                colors = image.reshape(-1, 3)
            else:
                colors = np.ones((ray_origins.shape[0], 3), dtype=np.float32)
            
            # Camera IDs
            camera_ids = np.full(ray_origins.shape[0], camera.image_id, dtype=np.int32)
            
            all_ray_origins.append(ray_origins)
            all_ray_directions.append(ray_directions)
            all_colors.append(colors)
            all_camera_ids.append(camera_ids)
        
        return {
            'ray_origins': np.concatenate(all_ray_origins, axis=0)
        }
    
    def _generate_camera_rays(self, camera: CameraInfo) -> Tuple[np.ndarray, np.ndarray]:
        """Generate rays for a camera"""
        H, W = camera.height, camera.width
        
        # Pixel coordinates
        i, j = np.meshgrid(np.arange(W), np.arange(H), indexing='xy')
        
        # Camera coordinates
        dirs = np.stack([(i - camera.intrinsics[0, 2], j - camera.intrinsics[1, 2])], axis=-1).astype(np.float32)
        
        # Transform to world coordinates
        ray_directions = np.sum(dirs[..., None, :] * camera.transform_matrix[:3, :3], axis=-1)
        ray_origins = np.broadcast_to(camera.transform_matrix[:3, 3], ray_directions.shape)
        
        # Flatten
        ray_origins = ray_origins.reshape(-1, 3)
        ray_directions = ray_directions.reshape(-1, 3)
        
        return ray_origins, ray_directions
    
    def get_partition_data(self, partition_idx: int) -> Dict[str, Any]:
        """Get data for a specific partition"""
        if partition_idx >= len(self.data_partitions):
            raise ValueError(f"Partition {partition_idx} out of range")
        
        partition = self.data_partitions[partition_idx]
        rays_data = self.rays_cache.get(partition_idx, {})
        
        return {
            'partition': partition, 'rays': rays_data
        }
    
    def __len__(self) -> int:
        """Return dataset length"""
        total_rays = sum(len(rays_data.get('colors', [])) for rays_data in self.rays_cache.values())
        return max(1, total_rays // self.ray_batch_size)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get a batch of rays"""
        # Cycle through partitions
        partition_idx = idx % len(self.data_partitions)
        rays_data = self.rays_cache.get(partition_idx, {})
        
        if not rays_data:
            # Return empty batch
            return {
                'ray_origins': torch.zeros(0, 3)
            }
        
        # Sample rays from partition
        num_rays = len(rays_data['colors'])
        if num_rays == 0:
            return self.__getitem__((idx + 1) % len(self))
        
        # Random sampling
        ray_indices = np.random.choice(num_rays, min(self.ray_batch_size, num_rays), replace=False)
        
        batch = {
            'ray_origins': torch.from_numpy(rays_data['ray_origins'][ray_indices])
        }
        
        return batch
    
    def create_dataloader(
        self,
        batch_size: int = 1,
        shuffle: bool = True,
        num_workers: int = 4,
    ) -> DataLoader:
        """Create a DataLoader for this dataset"""
        return DataLoader(
            self, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, pin_memory=True
        ) 