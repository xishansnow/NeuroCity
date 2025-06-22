"""
Dataset components for Mega-NeRF++

This module implements dataset classes optimized for large-scale photogrammetric data:
- High-resolution image handling
- Memory-efficient data loading
- Photogrammetric metadata processing
- Multi-scale data preparation
"""

import torch
import torch.utils.data as data
import numpy as np
import json
import os
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union, Any
import cv2
from PIL import Image, ImageFile
import imageio
import h5py
from concurrent.futures import ThreadPoolExecutor
import pickle
import tifffile

# Allow loading of truncated images
ImageFile.LOAD_TRUNCATED_IMAGES = True


class MegaNeRFPlusDataset(data.Dataset):
    """
    Base dataset class for Mega-NeRF++ with optimizations for large-scale scenes
    """
    
    def __init__(self, data_dir: str, split: str = 'train', 
                 max_image_resolution: int = 8192,
                 downsample_factor: int = 4,
                 use_cached_rays: bool = True,
                 cache_dir: Optional[str] = None,
                 num_workers: int = 8):
        """
        Args:
            data_dir: Path to dataset directory
            split: Dataset split ('train', 'val', 'test')
            max_image_resolution: Maximum supported image resolution
            downsample_factor: Initial downsampling factor
            use_cached_rays: Whether to use cached ray data
            cache_dir: Directory for caching processed data
            num_workers: Number of worker processes for data loading
        """
        self.data_dir = Path(data_dir)
        self.split = split
        self.max_image_resolution = max_image_resolution
        self.downsample_factor = downsample_factor
        self.use_cached_rays = use_cached_rays
        self.cache_dir = Path(cache_dir) if cache_dir else self.data_dir / 'cache'
        self.num_workers = num_workers
        
        # Create cache directory
        self.cache_dir.mkdir(exist_ok=True)
        
        # Data containers
        self.images = []
        self.poses = []
        self.intrinsics = []
        self.image_paths = []
        self.rays = []
        self.rgbs = []
        
        # Scene metadata
        self.scene_bounds = None
        self.near = None
        self.far = None
        
        # Load and process data
        self._load_data()
        
    def _load_data(self):
        """Load dataset - to be implemented by subclasses"""
        raise NotImplementedError
        
    def __len__(self):
        return len(self.rays) if self.rays else len(self.images)
    
    def __getitem__(self, idx):
        if self.rays:
            return {
                'rays': self.rays[idx],
                'rgbs': self.rgbs[idx],
                'metadata': self._get_metadata(idx)
            }
        else:
            return self._get_image_data(idx)
    
    def _get_metadata(self, idx: int) -> Dict[str, Any]:
        """Get metadata for a data sample"""
        return {
            'index': idx,
            'split': self.split
        }
    
    def _get_image_data(self, idx: int) -> Dict[str, Any]:
        """Get full image data for validation/testing"""
        return {
            'image': self.images[idx],
            'pose': self.poses[idx], 
            'intrinsics': self.intrinsics[idx],
            'image_path': str(self.image_paths[idx])
        }


class PhotogrammetricDataset(MegaNeRFPlusDataset):
    """
    Dataset for photogrammetric data with high-resolution image support
    """
    
    def __init__(self, data_dir: str, split: str = 'train', **kwargs):
        self.photogrammetric_metadata = {}
        super().__init__(data_dir, split, **kwargs)
    
    def _load_data(self):
        """Load photogrammetric dataset"""
        
        # Load poses and intrinsics
        poses_file = self.data_dir / 'poses.txt'
        intrinsics_file = self.data_dir / 'intrinsics.txt'
        
        if poses_file.exists() and intrinsics_file.exists():
            self._load_poses_intrinsics(poses_file, intrinsics_file)
        else:
            # Try COLMAP format
            self._load_colmap_data()
        
        # Load images with high-resolution support
        self._load_high_resolution_images()
        
        # Process photogrammetric metadata
        self._process_photogrammetric_metadata()
        
        # Generate rays if needed
        if self.use_cached_rays:
            self._load_or_generate_rays()
    
    def _load_poses_intrinsics(self, poses_file: Path, intrinsics_file: Path):
        """Load poses and intrinsics from text files"""
        
        # Load poses (4x4 matrices)
        poses_data = np.loadtxt(poses_file)
        num_images = poses_data.shape[0] // 4
        
        self.poses = []
        for i in range(num_images):
            pose_matrix = poses_data[i*4:(i+1)*4, :]
            self.poses.append(torch.from_numpy(pose_matrix.astype(np.float32)))
        
        # Load intrinsics (3x3 matrices)
        intrinsics_data = np.loadtxt(intrinsics_file)
        if intrinsics_data.ndim == 1:
            # Single intrinsic matrix for all images
            intrinsic_matrix = intrinsics_data.reshape(3, 3)
            self.intrinsics = [torch.from_numpy(intrinsic_matrix.astype(np.float32))] * num_images
        else:
            # Individual intrinsics for each image
            num_intrinsics = intrinsics_data.shape[0] // 3
            self.intrinsics = []
            for i in range(num_intrinsics):
                intrinsic_matrix = intrinsics_data[i*3:(i+1)*3, :]
                self.intrinsics.append(torch.from_numpy(intrinsic_matrix.astype(np.float32)))
    
    def _load_colmap_data(self):
        """Load COLMAP reconstruction data"""
        # Simplified COLMAP loading - in practice would use COLMAP reader
        
        cameras_file = self.data_dir / 'cameras.txt'
        images_file = self.data_dir / 'images.txt'
        
        if not (cameras_file.exists() and images_file.exists()):
            raise ValueError("Neither poses.txt/intrinsics.txt nor COLMAP files found")
        
        # Parse COLMAP cameras
        cameras = {}
        with open(cameras_file, 'r') as f:
            for line in f:
                if line.startswith('#'):
                    continue
                parts = line.strip().split()
                camera_id = int(parts[0])
                model = parts[1]
                width, height = int(parts[2]), int(parts[3])
                params = [float(x) for x in parts[4:]]
                
                # Convert to intrinsic matrix (simplified for PINHOLE model)
                if model == 'PINHOLE':
                    fx, fy, cx, cy = params
                    K = np.array([
                        [fx, 0, cx],
                        [0, fy, cy], 
                        [0, 0, 1]
                    ], dtype=np.float32)
                    cameras[camera_id] = K
        
        # Parse COLMAP images
        self.poses = []
        self.intrinsics = []
        self.image_paths = []
        
        with open(images_file, 'r') as f:
            for line in f:
                if line.startswith('#'):
                    continue
                parts = line.strip().split()
                image_id = int(parts[0])
                qw, qx, qy, qz = [float(x) for x in parts[1:5]]
                tx, ty, tz = [float(x) for x in parts[5:8]]
                camera_id = int(parts[8])
                image_name = parts[9]
                
                # Convert quaternion and translation to pose matrix
                pose = self._quat_trans_to_matrix(qw, qx, qy, qz, tx, ty, tz)
                self.poses.append(torch.from_numpy(pose))
                self.intrinsics.append(torch.from_numpy(cameras[camera_id]))
                self.image_paths.append(self.data_dir / 'images' / image_name)
    
    def _quat_trans_to_matrix(self, qw: float, qx: float, qy: float, qz: float,
                             tx: float, ty: float, tz: float) -> np.ndarray:
        """Convert quaternion and translation to 4x4 pose matrix"""
        
        # Normalize quaternion
        norm = np.sqrt(qw*qw + qx*qx + qy*qy + qz*qz)
        qw, qx, qy, qz = qw/norm, qx/norm, qy/norm, qz/norm
        
        # Convert to rotation matrix
        R = np.array([
            [1-2*(qy*qy+qz*qz), 2*(qx*qy-qw*qz), 2*(qx*qz+qw*qy)],
            [2*(qx*qy+qw*qz), 1-2*(qx*qx+qz*qz), 2*(qy*qz-qw*qx)],
            [2*(qx*qz-qw*qy), 2*(qy*qz+qw*qx), 1-2*(qx*qx+qy*qy)]
        ], dtype=np.float32)
        
        # Create 4x4 pose matrix
        pose = np.eye(4, dtype=np.float32)
        pose[:3, :3] = R
        pose[:3, 3] = [tx, ty, tz]
        
        return pose
    
    def _load_high_resolution_images(self):
        """Load high-resolution images with memory optimization"""
        
        if not self.image_paths:
            # Find images in directory
            image_dir = self.data_dir / 'images'
            if not image_dir.exists():
                raise ValueError(f"Images directory not found: {image_dir}")
            
            # Supported image formats
            extensions = ['.jpg', '.jpeg', '.png', '.tif', '.tiff', '.exr']
            self.image_paths = []
            
            for ext in extensions:
                self.image_paths.extend(list(image_dir.glob(f'*{ext}')))
                self.image_paths.extend(list(image_dir.glob(f'*{ext.upper()}')))
            
            self.image_paths.sort()
        
        # Load images with appropriate handling for different formats
        self.images = []
        
        print(f"Loading {len(self.image_paths)} images...")
        
        for i, img_path in enumerate(self.image_paths):
            if i % 100 == 0:
                print(f"Loaded {i}/{len(self.image_paths)} images")
            
            try:
                img = self._load_single_image(img_path)
                self.images.append(img)
            except Exception as e:
                print(f"Failed to load image {img_path}: {e}")
                continue
        
        print(f"Successfully loaded {len(self.images)} images")
    
    def _load_single_image(self, img_path: Path) -> torch.Tensor:
        """Load a single high-resolution image"""
        
        ext = img_path.suffix.lower()
        
        if ext in ['.tif', '.tiff']:
            # Use tifffile for TIFF images
            img = tifffile.imread(str(img_path))
            if img.dtype == np.uint16:
                img = img.astype(np.float32) / 65535.0
            elif img.dtype == np.uint8:
                img = img.astype(np.float32) / 255.0
        elif ext == '.exr':
            # Use OpenEXR for EXR images (requires openexr package)
            try:
                import OpenEXR
                import Imath
                exr_file = OpenEXR.InputFile(str(img_path))
                header = exr_file.header()
                dw = header['dataWindow']
                size = (dw.max.x - dw.min.x + 1, dw.max.y - dw.min.y + 1)
                
                # Read RGB channels
                FLOAT = Imath.PixelType(Imath.PixelType.FLOAT)
                r_str = exr_file.channel('R', FLOAT)
                g_str = exr_file.channel('G', FLOAT)
                b_str = exr_file.channel('B', FLOAT)
                
                r = np.frombuffer(r_str, dtype=np.float32).reshape(size[1], size[0])
                g = np.frombuffer(g_str, dtype=np.float32).reshape(size[1], size[0])
                b = np.frombuffer(b_str, dtype=np.float32).reshape(size[1], size[0])
                
                img = np.stack([r, g, b], axis=-1)
            except ImportError:
                # Fallback to imageio
                img = imageio.imread(str(img_path))
                if img.dtype == np.uint8:
                    img = img.astype(np.float32) / 255.0
        else:
            # Use PIL for standard formats
            img = np.array(Image.open(img_path))
            if img.dtype == np.uint8:
                img = img.astype(np.float32) / 255.0
        
        # Handle different channel configurations
        if len(img.shape) == 2:
            # Grayscale - convert to RGB
            img = np.stack([img, img, img], axis=-1)
        elif img.shape[-1] == 4:
            # RGBA - use only RGB channels
            img = img[..., :3]
        
        # Downsample if image is too large
        h, w = img.shape[:2]
        if max(h, w) > self.max_image_resolution:
            scale = self.max_image_resolution / max(h, w)
            new_h, new_w = int(h * scale), int(w * scale)
            img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
        
        # Apply initial downsampling
        if self.downsample_factor > 1:
            h, w = img.shape[:2]
            new_h, new_w = h // self.downsample_factor, w // self.downsample_factor
            img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
        
        return torch.from_numpy(img)
    
    def _process_photogrammetric_metadata(self):
        """Process photogrammetric metadata"""
        
        # Extract metadata from poses and intrinsics
        for i, (pose, intrinsic) in enumerate(zip(self.poses, self.intrinsics)):
            # Camera position
            camera_pos = pose[:3, 3]
            
            # Viewing direction
            viewing_dir = pose[:3, 2]  # Camera's forward direction
            
            # Field of view
            fx, fy = intrinsic[0, 0], intrinsic[1, 1]
            img_h, img_w = self.images[i].shape[:2] if i < len(self.images) else (1024, 1024)
            fov_x = 2 * np.arctan(img_w / (2 * fx))
            fov_y = 2 * np.arctan(img_h / (2 * fy))
            
            self.photogrammetric_metadata[i] = {
                'camera_position': camera_pos,
                'viewing_direction': viewing_dir,
                'fov_x': fov_x,
                'fov_y': fov_y,
                'image_resolution': (img_w, img_h),
                'focal_length': (fx, fy)
            }
        
        # Compute scene bounds
        self._compute_scene_bounds()
    
    def _compute_scene_bounds(self):
        """Compute scene bounding box from camera positions"""
        
        if not self.poses:
            return
        
        # Extract camera positions
        camera_positions = torch.stack([pose[:3, 3] for pose in self.poses])
        
        # Compute bounds with margin
        mins = torch.min(camera_positions, dim=0)[0]
        maxs = torch.max(camera_positions, dim=0)[0]
        
        # Add margin based on scene size
        scene_size = maxs - mins
        margin = torch.max(scene_size) * 0.5  # 50% margin
        
        self.scene_bounds = torch.stack([mins - margin, maxs + margin])
        
        # Estimate near and far planes
        center = (mins + maxs) / 2
        max_distance = torch.max(torch.norm(camera_positions - center, dim=1))
        
        self.near = 0.1
        self.far = max_distance.item() * 2
    
    def _load_or_generate_rays(self):
        """Load cached rays or generate new ones"""
        
        cache_file = self.cache_dir / f'rays_{self.split}.pkl'
        
        if cache_file.exists() and self.use_cached_rays:
            print(f"Loading cached rays from {cache_file}")
            try:
                with open(cache_file, 'rb') as f:
                    cached_data = pickle.load(f)
                    self.rays = cached_data['rays']
                    self.rgbs = cached_data['rgbs']
                return
            except Exception as e:
                print(f"Failed to load cached rays: {e}")
        
        print("Generating rays...")
        self._generate_rays()
        
        # Cache rays for future use
        if self.use_cached_rays:
            print(f"Caching rays to {cache_file}")
            try:
                with open(cache_file, 'wb') as f:
                    pickle.dump({
                        'rays': self.rays,
                        'rgbs': self.rgbs
                    }, f)
            except Exception as e:
                print(f"Failed to cache rays: {e}")
    
    def _generate_rays(self):
        """Generate rays for all images"""
        
        all_rays = []
        all_rgbs = []
        
        for i, (img, pose, intrinsic) in enumerate(zip(self.images, self.poses, self.intrinsics)):
            if i % 10 == 0:
                print(f"Generating rays for image {i}/{len(self.images)}")
            
            h, w = img.shape[:2]
            
            # Generate pixel coordinates
            i_coords, j_coords = torch.meshgrid(
                torch.arange(w, dtype=torch.float32),
                torch.arange(h, dtype=torch.float32),
                indexing='xy'
            )
            
            # Convert to camera coordinates
            fx, fy = intrinsic[0, 0], intrinsic[1, 1]
            cx, cy = intrinsic[0, 2], intrinsic[1, 2]
            
            dirs = torch.stack([
                (i_coords - cx) / fx,
                -(j_coords - cy) / fy,
                -torch.ones_like(i_coords)
            ], dim=-1)
            
            # Transform to world coordinates
            dirs = torch.sum(dirs[..., None, :] * pose[:3, :3], dim=-1)
            origins = pose[:3, 3].expand(dirs.shape)
            
            # Flatten spatial dimensions
            rays_o = origins.reshape(-1, 3)
            rays_d = dirs.reshape(-1, 3)
            rgbs = img.reshape(-1, 3)
            
            # Store rays
            all_rays.append(torch.cat([rays_o, rays_d], dim=-1))
            all_rgbs.append(rgbs)
        
        self.rays = torch.cat(all_rays, dim=0)
        self.rgbs = torch.cat(all_rgbs, dim=0)


class LargeSceneDataset(PhotogrammetricDataset):
    """
    Dataset optimized for very large scenes with streaming capabilities
    """
    
    def __init__(self, data_dir: str, split: str = 'train',
                 partition_config: Optional[Dict] = None,
                 streaming_mode: bool = True,
                 **kwargs):
        """
        Args:
            partition_config: Configuration for spatial partitioning
            streaming_mode: Whether to use streaming data loading
        """
        self.partition_config = partition_config or {}
        self.streaming_mode = streaming_mode
        self.partitions = []
        self.current_partition = 0
        
        super().__init__(data_dir, split, **kwargs)
        
        if streaming_mode:
            self._setup_streaming()
    
    def _load_data(self):
        """Load data with partitioning for large scenes"""
        
        # Load metadata first
        super()._load_data()
        
        # Create spatial partitions
        if self.scene_bounds is not None:
            self._create_spatial_partitions()
    
    def _create_spatial_partitions(self):
        """Create spatial partitions for the scene"""
        
        from .spatial_partitioner import PhotogrammetricPartitioner, PartitionConfig
        
        # Create partitioner
        config = PartitionConfig(**self.partition_config)
        partitioner = PhotogrammetricPartitioner(config)
        
        # Extract camera data
        camera_positions = torch.stack([pose[:3, 3] for pose in self.poses])
        camera_orientations = torch.stack([pose[:3, :3] for pose in self.poses])
        
        # Get image resolutions and intrinsics
        image_resolutions = []
        intrinsics_tensor = []
        
        for i, intrinsic in enumerate(self.intrinsics):
            if i < len(self.images):
                h, w = self.images[i].shape[:2]
                image_resolutions.append([w, h])
            else:
                image_resolutions.append([1024, 1024])  # Default
            intrinsics_tensor.append(intrinsic)
        
        image_resolutions = torch.tensor(image_resolutions, dtype=torch.float32)
        intrinsics_tensor = torch.stack(intrinsics_tensor)
        
        # Create partitions
        self.partitions = partitioner.partition_scene(
            self.scene_bounds, camera_positions, camera_orientations,
            image_resolutions, intrinsics_tensor
        )
        
        print(f"Created {len(self.partitions)} spatial partitions")
    
    def _setup_streaming(self):
        """Setup streaming data loading"""
        
        if not self.partitions:
            return
        
        # For streaming mode, we'll load data partition by partition
        # This is a simplified implementation
        self.partition_data = {}
        
        for i, partition in enumerate(self.partitions):
            # Determine which images belong to this partition
            partition_images = self._get_partition_images(partition)
            
            self.partition_data[i] = {
                'image_indices': partition_images,
                'loaded': False,
                'rays': None,
                'rgbs': None
            }
    
    def _get_partition_images(self, partition: Dict) -> List[int]:
        """Get image indices that belong to a partition"""
        
        partition_bounds = partition['bounds']
        partition_images = []
        
        for i, pose in enumerate(self.poses):
            camera_pos = pose[:3, 3]
            
            # Check if camera is within partition bounds (with margin)
            margin = 0.1 * torch.norm(partition_bounds[1] - partition_bounds[0])
            expanded_bounds = torch.stack([
                partition_bounds[0] - margin,
                partition_bounds[1] + margin
            ])
            
            if (torch.all(camera_pos >= expanded_bounds[0]) and 
                torch.all(camera_pos <= expanded_bounds[1])):
                partition_images.append(i)
        
        return partition_images
    
    def get_partition_data(self, partition_idx: int) -> Dict[str, torch.Tensor]:
        """Get data for a specific partition"""
        
        if partition_idx not in self.partition_data:
            return {}
        
        partition_info = self.partition_data[partition_idx]
        
        # Load partition data if not already loaded
        if not partition_info['loaded']:
            self._load_partition_data(partition_idx)
        
        return {
            'rays': partition_info['rays'],
            'rgbs': partition_info['rgbs'],
            'partition': self.partitions[partition_idx]
        }
    
    def _load_partition_data(self, partition_idx: int):
        """Load data for a specific partition"""
        
        partition_info = self.partition_data[partition_idx]
        image_indices = partition_info['image_indices']
        
        if not image_indices:
            return
        
        print(f"Loading partition {partition_idx} with {len(image_indices)} images")
        
        # Generate rays for partition images
        partition_rays = []
        partition_rgbs = []
        
        for img_idx in image_indices:
            if img_idx >= len(self.images):
                continue
                
            img = self.images[img_idx]
            pose = self.poses[img_idx]
            intrinsic = self.intrinsics[img_idx]
            
            h, w = img.shape[:2]
            
            # Generate rays (same as before)
            i_coords, j_coords = torch.meshgrid(
                torch.arange(w, dtype=torch.float32),
                torch.arange(h, dtype=torch.float32),
                indexing='xy'
            )
            
            fx, fy = intrinsic[0, 0], intrinsic[1, 1]
            cx, cy = intrinsic[0, 2], intrinsic[1, 2]
            
            dirs = torch.stack([
                (i_coords - cx) / fx,
                -(j_coords - cy) / fy,
                -torch.ones_like(i_coords)
            ], dim=-1)
            
            dirs = torch.sum(dirs[..., None, :] * pose[:3, :3], dim=-1)
            origins = pose[:3, 3].expand(dirs.shape)
            
            rays_o = origins.reshape(-1, 3)
            rays_d = dirs.reshape(-1, 3)
            rgbs = img.reshape(-1, 3)
            
            partition_rays.append(torch.cat([rays_o, rays_d], dim=-1))
            partition_rgbs.append(rgbs)
        
        # Store partition data
        if partition_rays:
            partition_info['rays'] = torch.cat(partition_rays, dim=0)
            partition_info['rgbs'] = torch.cat(partition_rgbs, dim=0)
        else:
            partition_info['rays'] = torch.empty(0, 6)
            partition_info['rgbs'] = torch.empty(0, 3)
        
        partition_info['loaded'] = True


def create_meganerf_plus_dataset(data_dir: str, dataset_type: str = 'photogrammetric',
                                split: str = 'train', **kwargs) -> MegaNeRFPlusDataset:
    """
    Factory function to create Mega-NeRF++ datasets
    
    Args:
        data_dir: Path to dataset directory
        dataset_type: Type of dataset ('photogrammetric', 'large_scene')
        split: Dataset split
        **kwargs: Additional arguments
        
    Returns:
        Mega-NeRF++ dataset instance
    """
    
    if dataset_type.lower() == 'photogrammetric':
        return PhotogrammetricDataset(data_dir, split, **kwargs)
    elif dataset_type.lower() == 'large_scene':
        return LargeSceneDataset(data_dir, split, **kwargs)
    else:
        raise ValueError(f"Unknown dataset type: {dataset_type}")


def create_photogrammetric_dataloader(dataset: MegaNeRFPlusDataset,
                                    batch_size: int = 4096,
                                    shuffle: bool = True,
                                    num_workers: int = 4,
                                    pin_memory: bool = True) -> data.DataLoader:
    """
    Create optimized DataLoader for photogrammetric data
    
    Args:
        dataset: Mega-NeRF++ dataset
        batch_size: Batch size
        shuffle: Whether to shuffle data
        num_workers: Number of worker processes
        pin_memory: Whether to pin memory
        
    Returns:
        PyTorch DataLoader
    """
    
    return data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=photogrammetric_collate_fn,
        persistent_workers=True if num_workers > 0 else False
    )


def photogrammetric_collate_fn(batch: List[Dict]) -> Dict[str, torch.Tensor]:
    """
    Custom collate function for photogrammetric data
    
    Args:
        batch: List of data samples
        
    Returns:
        Batched data
    """
    
    if not batch:
        return {}
    
    # Check if batch contains rays or images
    if 'rays' in batch[0]:
        # Ray-based data
        rays = torch.stack([item['rays'] for item in batch])
        rgbs = torch.stack([item['rgbs'] for item in batch])
        
        return {
            'rays': rays,
            'rgbs': rgbs,
            'batch_size': len(batch)
        }
    else:
        # Image-based data
        images = torch.stack([item['image'] for item in batch])
        poses = torch.stack([item['pose'] for item in batch])
        intrinsics = torch.stack([item['intrinsics'] for item in batch])
        
        return {
            'images': images,
            'poses': poses,
            'intrinsics': intrinsics,
            'batch_size': len(batch)
        } 