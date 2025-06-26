"""
CNC-NeRF Dataset Module

This module implements dataset handling for Context-based NeRF Compression, supporting multi-resolution supervision and pyramid loss training.
"""

import torch
import torch.nn.functional as F
import numpy as np
import json
import cv2
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from torch.utils.data import Dataset, DataLoader

from .core import CNCNeRFConfig


@dataclass
class CNCNeRFDatasetConfig:
    """Configuration for CNC-NeRF dataset."""
    
    # Data paths
    data_root: str = "data"
    images_dir: str = "images"
    cameras_file: str = "cameras.json"
    
    # Image processing
    image_width: int = 800
    image_height: int = 600
    downscale_factor: float = 1.0
    
    # Multi-resolution supervision
    pyramid_levels: int = 4
    min_resolution: int = 64
    use_pyramid_loss: bool = True
    pyramid_loss_weights: list[float] = None
    
    # Training data
    train_split: float = 0.8
    val_split: float = 0.1
    test_split: float = 0.1
    
    # Sampling
    num_rays_per_batch: int = 4096
    precrop_fraction: float = 1.0
    precrop_iterations: int = 500
    
    # Near/far planes
    near_plane: float = 0.2
    far_plane: float = 1000.0
    
    def __post_init__(self):
        """Post-initialization validation."""
        if self.pyramid_loss_weights is None:
            # Default weights: higher weight for finer resolutions
            self.pyramid_loss_weights = [1.0 / (2**i) for i in range(self.pyramid_levels)]
        
        assert len(self.pyramid_loss_weights) == self.pyramid_levels, \
            "Pyramid loss weights must match number of levels"
        
        assert abs(self.train_split + self.val_split + self.test_split - 1.0) < 1e-6, \
            "Train/val/test splits must sum to 1.0"


class CNCNeRFDataset(Dataset):
    """Dataset for CNC-NeRF with multi-resolution supervision."""
    
    def __init__(self, config: CNCNeRFDatasetConfig, split: str = 'train'):
        super().__init__()
        self.config = config
        self.split = split
        
        # Load camera data
        self.cameras = self._load_cameras()
        self.images = self._load_images()
        
        # Create multi-resolution pyramid
        if config.use_pyramid_loss:
            self.image_pyramid = self._create_image_pyramid()
        
        # Split data
        self.indices = self._create_split()
        
        # Pre-compute rays
        self.rays, self.colors = self._precompute_rays()
        
        print(f"Loaded {len(self.indices)} {split} views, {len(self.rays)} rays")
    
    def _load_cameras(self) -> dict[str, Any]:
        """Load camera parameters from JSON file."""
        cameras_path = Path(self.config.data_root) / self.config.cameras_file
        
        with open(cameras_path, 'r') as f:
            cameras_data = json.load(f)
        
        # Convert to more convenient format
        cameras = {
            'intrinsics': [], 'extrinsics': [], 'image_names': []
        }
        
        for frame in cameras_data['frames']:
            # Intrinsic matrix
            if 'fl_x' in frame and 'fl_y' in frame:
                fx = frame['fl_x']
                fy = frame['fl_y']
                cx = frame.get('cx', self.config.image_width / 2)
                cy = frame.get('cy', self.config.image_height / 2)
            else:
                # Use camera_angle_x if available
                angle_x = cameras_data.get('camera_angle_x', 0.6911112070083618)
                fx = self.config.image_width / (2.0 * np.tan(angle_x / 2.0))
                fy = fx
                cx = self.config.image_width / 2
                cy = self.config.image_height / 2
            
            intrinsic = np.array([
                [fx, 0, cx], [0, fy, cy], [0, 0, 1]
            ])
            
            # Extrinsic matrix (camera-to-world)
            extrinsic = np.array(frame['transform_matrix'])
            
            cameras['intrinsics'].append(intrinsic)
            cameras['extrinsics'].append(extrinsic)
            cameras['image_names'].append(frame['file_path'])
        
        return cameras
    
    def _load_images(self) -> list[np.ndarray]:
        """Load and preprocess images."""
        images = []
        images_dir = Path(self.config.data_root) / self.config.images_dir
        
        for img_name in self.cameras['image_names']:
            # Handle different file extensions
            img_path = images_dir / img_name
            if not img_path.exists():
                # Try with .png extension
                img_path = images_dir / (Path(img_name).stem + '.png')
            if not img_path.exists():
                # Try with .jpg extension
                img_path = images_dir / (Path(img_name).stem + '.jpg')
            
            if not img_path.exists():
                raise FileNotFoundError(f"Image not found: {img_name}")
            
            # Load image
            img = cv2.imread(str(img_path))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            # Resize if needed
            if self.config.downscale_factor != 1.0:
                new_width = int(img.shape[1] * self.config.downscale_factor)
                new_height = int(img.shape[0] * self.config.downscale_factor)
                img = cv2.resize(img, (new_width, new_height))
            
            # Normalize to [0, 1]
            img = img.astype(np.float32) / 255.0
            
            images.append(img)
        
        return images
    
    def _create_image_pyramid(self) -> list[list[np.ndarray]]:
        """Create multi-resolution image pyramid for pyramid supervision."""
        pyramid = []
        
        for img in self.images:
            img_pyramid = [img]  # Original resolution
            
            current_img = img
            for level in range(1, self.config.pyramid_levels):
                # Downsample by factor of 2
                new_height = max(self.config.min_resolution, current_img.shape[0] // 2)
                new_width = max(self.config.min_resolution, current_img.shape[1] // 2)
                
                current_img = cv2.resize(current_img, (new_width, new_height))
                img_pyramid.append(current_img)
            
            pyramid.append(img_pyramid)
        
        return pyramid
    
    def _create_split(self) -> list[int]:
        """Create train/val/test split indices."""
        num_images = len(self.images)
        indices = list(range(num_images))
        np.random.shuffle(indices)
        
        if self.split == 'train':
            end_idx = int(num_images * self.config.train_split)
            return indices[:end_idx]
        elif self.split == 'val':
            start_idx = int(num_images * self.config.train_split)
            end_idx = start_idx + int(num_images * self.config.val_split)
            return indices[start_idx:end_idx]
        else:  # test
            start_idx = int(num_images * (self.config.train_split + self.config.val_split))
            return indices[start_idx:]
    
    def _precompute_rays(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Pre-compute rays and colors for all pixels."""
        all_rays = []
        all_colors = []
        
        for idx in self.indices:
            intrinsic = self.cameras['intrinsics'][idx]
            extrinsic = self.cameras['extrinsics'][idx]
            image = self.images[idx]
            
            rays, colors = self._generate_rays(intrinsic, extrinsic, image)
            all_rays.append(rays)
            all_colors.append(colors)
        
        # Concatenate all rays
        all_rays = torch.cat(all_rays, dim=0)
        all_colors = torch.cat(all_colors, dim=0)
        
        return all_rays, all_colors
    
    def _generate_rays(
        self,
        intrinsic: np.ndarray,
        extrinsic: np.ndarray,
        image: np.ndarray,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Generate rays for a single image."""
        H, W = image.shape[:2]
        
        # Create pixel coordinates
        i, j = np.meshgrid(np.arange(W), np.arange(H), indexing='xy')
        
        # Convert to camera coordinates
        fx, fy = intrinsic[0, 0], intrinsic[1, 1]
        cx, cy = intrinsic[0, 2], intrinsic[1, 2]
        
        dirs = np.stack([
            (i - cx) / fx, -(j - cy) / fy, # Negative y for right-handed coordinate system
            -np.ones_like(i)
        ], axis=-1)
        
        # Transform to world coordinates
        rays_d = np.sum(dirs[..., None, :] * extrinsic[:3, :3], axis=-1)
        rays_o = np.broadcast_to(extrinsic[:3, 3], rays_d.shape)
        
        # Normalize direction vectors
        rays_d = rays_d / np.linalg.norm(rays_d, axis=-1, keepdims=True)
        
        # Flatten
        rays_o = rays_o.reshape(-1, 3)
        rays_d = rays_d.reshape(-1, 3)
        colors = image.reshape(-1, 3)
        
        # Create ray bundle
        rays = np.concatenate([
            rays_o, # Origin (3)
            rays_d, # Direction (3)
            np.full((rays_o.shape[0], 1), self.config.near_plane), # Near (1)
            np.full((rays_o.shape[0], 1), self.config.far_plane)   # Far (1)
        ], axis=-1)
        
        return torch.from_numpy(rays).float(), torch.from_numpy(colors).float()
    
    def get_pyramid_targets(self, ray_indices: torch.Tensor) -> list[torch.Tensor]:
        """Get multi-resolution targets for pyramid supervision."""
        if not self.config.use_pyramid_loss:
            return [self.colors[ray_indices]]
        
        targets = []
        
        # Original resolution
        targets.append(self.colors[ray_indices])
        
        # Lower resolutions
        for level in range(1, self.config.pyramid_levels):
            # Find which image each ray belongs to
            # This is a simplified approach - in practice, you'd want to maintain
            # a mapping from ray indices to image/pixel coordinates
            downsampled_colors = F.interpolate(
                self.colors[ray_indices].unsqueeze(
                    0,
                )
            ).permute(0, 2, 1).squeeze(0)
            
            targets.append(downsampled_colors)
        
        return targets
    
    def __len__(self) -> int:
        """Get dataset size."""
        return len(self.rays) // self.config.num_rays_per_batch
    
    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        """Get a batch of rays and colors."""
        # Sample random rays
        if self.split == 'train':
            # Random sampling for training
            ray_indices = torch.randperm(len(self.rays))[:self.config.num_rays_per_batch]
        else:
            # Sequential sampling for validation/test
            start_idx = idx * self.config.num_rays_per_batch
            end_idx = min(start_idx + self.config.num_rays_per_batch, len(self.rays))
            ray_indices = torch.arange(start_idx, end_idx)
        
        rays = self.rays[ray_indices]
        colors = self.colors[ray_indices]
        
        batch = {
            'rays': rays, 'colors': colors, 'ray_indices': ray_indices
        }
        
        # Add pyramid targets if enabled
        if self.config.use_pyramid_loss:
            batch['pyramid_targets'] = self.get_pyramid_targets(ray_indices)
            batch['pyramid_weights'] = torch.tensor(self.config.pyramid_loss_weights)
        
        return batch


class MultiResolutionDataLoader:
    """Multi-resolution data loader for progressive training."""
    
    def __init__(
        self,
        dataset: CNCNeRFDataset,
        batch_size: int = 1,
        shuffle: bool = True,
        num_workers: int = 0,
    ) -> None:
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.num_workers = num_workers
        
        # Create base dataloader
        self.dataloader = DataLoader(
            dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, pin_memory=True
        )
    
    def __iter__(self):
        """Iterate over batches."""
        for batch in self.dataloader:
            yield batch
    
    def __len__(self):
        """Get number of batches."""
        return len(self.dataloader)


def create_synthetic_dataset(
    config: CNCNeRFDatasetConfig,
    scene_type: str = 'lego',
) -> CNCNeRFDataset:
    """Create a synthetic dataset for testing CNC-NeRF."""
    import os
    from pathlib import Path
    
    # Create data directory
    data_dir = Path(config.data_root)
    data_dir.mkdir(exist_ok=True)
    
    images_dir = data_dir / config.images_dir
    images_dir.mkdir(exist_ok=True)
    
    # Generate synthetic camera poses in a circle
    num_views = 100
    radius = 4.0
    cameras_data = {
        'camera_angle_x': 0.6911112070083618, 'frames': []
    }
    
    # Generate synthetic images and poses
    for i in range(num_views):
        angle = 2 * np.pi * i / num_views
        
        # Camera position on a circle
        x = radius * np.cos(angle)
        y = radius * np.sin(angle)
        z = 0.0
        
        # Look at origin
        look_at = np.array([0, 0, 0])
        camera_pos = np.array([x, y, z])
        
        # Compute camera matrix
        forward = look_at - camera_pos
        forward = forward / np.linalg.norm(forward)
        
        up = np.array([0, 0, 1])
        right = np.cross(forward, up)
        right = right / np.linalg.norm(right)
        
        up = np.cross(right, forward)
        
        # Camera-to-world matrix
        c2w = np.eye(4)
        c2w[:3, 0] = right
        c2w[:3, 1] = up
        c2w[:3, 2] = -forward
        c2w[:3, 3] = camera_pos
        
        # Generate synthetic image (simple colored pattern)
        img = np.zeros((config.image_height, config.image_width, 3), dtype=np.uint8)
        
        # Create a simple pattern based on scene type
        if scene_type == 'lego':
            # Red-blue checkerboard pattern
            for h in range(config.image_height):
                for w in range(config.image_width):
                    if (h // 50 + w // 50) % 2 == 0:
                        img[h, w] = [255, 0, 0]  # Red
                    else:
                        img[h, w] = [0, 0, 255]  # Blue
        else:
            # Gradient pattern
            img[:, :, 0] = np.linspace(0, 255, config.image_width)[None, :]
            img[:, :, 1] = np.linspace(0, 255, config.image_height)[:, None]
            img[:, :, 2] = 128
        
        # Save image
        img_name = f"frame_{i:04d}.png"
        img_path = images_dir / img_name
        cv2.imwrite(str(img_path), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        
        # Add to cameras data
        cameras_data['frames'].append({
            'file_path': img_name, 'transform_matrix': c2w.tolist()
        })
    
    # Save cameras file
    cameras_path = data_dir / config.cameras_file
    with open(cameras_path, 'w') as f:
        json.dump(cameras_data, f, indent=2)
    
    print(f"Created synthetic dataset with {num_views} views in {data_dir}")
    
    # Create and return dataset
    return CNCNeRFDataset(config, split='train')


def compute_pyramid_loss(
    predictions: list[torch.Tensor],
    targets: list[torch.Tensor],
    weights: torch.Tensor,
) -> torch.Tensor:
    """Compute multi-resolution pyramid loss."""
    total_loss = 0.0
    
    for i, (pred, target, weight) in enumerate(zip(predictions, targets, weights)):
        # MSE loss at this resolution
        mse_loss = F.mse_loss(pred, target)
        total_loss += weight * mse_loss
    
    return total_loss


def create_cnc_nerf_dataloader(
    config: CNCNeRFDatasetConfig,
    split: str = 'train',
    batch_size: int = 1,
) -> MultiResolutionDataLoader: 
    """Create a CNC-NeRF dataloader."""
    dataset = CNCNeRFDataset(config, split=split)
    
    return MultiResolutionDataLoader(
        dataset, batch_size=batch_size, shuffle=(
            split == 'train',
        )
    )


# Example usage
if __name__ == "__main__":
    # Create dataset config
    config = CNCNeRFDatasetConfig(
        data_root="test_data", image_width=400, image_height=300, pyramid_levels=3, use_pyramid_loss=True, num_rays_per_batch=1024
    )
    
    # Create synthetic dataset
    dataset = create_synthetic_dataset(config, scene_type='lego')
    
    # Create dataloader
    dataloader = create_cnc_nerf_dataloader(config, split='train')
    
    print(f"Dataset size: {len(dataset)}")
    print(f"Dataloader size: {len(dataloader)}")
    
    # Test batch
    for batch in dataloader:
        print(f"Batch rays shape: {batch['rays'].shape}")
        print(f"Batch colors shape: {batch['colors'].shape}")
        if 'pyramid_targets' in batch:
            print(f"Pyramid targets: {len(batch['pyramid_targets'])} levels")
        break 