from typing import Any, Optional, Union
"""
Dataset module for SVRaster.
"""

import os
import json
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
from PIL import Image
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)

@dataclass
class SVRasterDatasetConfig:
    """Configuration for SVRaster dataset."""
    
    # Data paths
    data_dir: str = "data/nerf_synthetic/lego"
    images_dir: str = "data/nerf_synthetic/lego/images"
    masks_dir: Optional[str] = None
    
    # Data format
    dataset_type: str = "colmap"
    image_format: str = "png"
    
    # Image processing
    image_height: int = 800
    image_width: int = 800
    downscale_factor: float = 1.0
    
    # Camera parameters
    camera_model: str = "pinhole"
    distortion_params: Optional[list[float]] = None
    
    # Data loading
    train_split: float = 0.8
    val_split: float = 0.1
    test_split: float = 0.1
    
    # Ray sampling
    num_rays_train: int = 1024
    num_rays_val: int = 512
    patch_size: int = 1
    
    # Data augmentation
    use_color_jitter: bool = False
    color_jitter_strength: float = 0.1
    use_random_background: bool = False
    
    # Scene bounds
    auto_scale_poses: bool = True
    scene_scale: float = 1.0
    scene_center: Optional[tuple[float, float, float]] = None
    
    # Background handling
    white_background: bool = False
    black_background: bool = False

class SVRasterDataset(Dataset):
    """Main dataset class for SVRaster."""
    
    def __init__(self, config: SVRasterDatasetConfig, split: str = "train"):
        self.config = config
        self.split = split
        
        # Load dataset based on type
        if config.dataset_type == "colmap":
            self._load_colmap_data()
        elif config.dataset_type == "blender":
            self._load_blender_data()
        else:
            raise ValueError(f"Unsupported dataset type: {config.dataset_type}")
        
        # Process and filter data
        self._process_images()
        self._split_data()
        self._setup_rays()
        
        logger.info(f"Loaded {len(self.images)} images for {split} split")
    
    def _load_colmap_data(self):
        """Load COLMAP format data."""
        images_path = os.path.join(self.config.data_dir, self.config.images_dir)
        image_files = sorted([f for f in os.listdir(images_path) 
                            if f.endswith(('.png', '.jpg', '.jpeg'))])
        
        self.images = []
        self.poses = []
        self.intrinsics = []
        
        # Load camera intrinsics (simplified)
        focal = 800.0
        cx, cy = self.config.image_width / 2, self.config.image_height / 2
        
        for img_file in image_files:
            # Load image
            img_path = os.path.join(images_path, img_file)
            image = Image.open(img_path).convert('RGB')
            image = image.resize((self.config.image_width, self.config.image_height))
            image = np.array(image) / 255.0
            self.images.append(image)
            
            # Generate dummy pose
            pose = np.eye(4)
            pose[:3, 3] = np.random.randn(3) * 2
            self.poses.append(pose)
            
            # Camera intrinsics
            K = np.array([
                [focal, 0, cx], [0, focal, cy], [0, 0, 1]
            ])
            self.intrinsics.append(K)
        
        self.images = np.array(self.images)
        self.poses = np.array(self.poses)
        self.intrinsics = np.array(self.intrinsics)
    
    def _load_blender_data(self):
        """Load Blender synthetic data format."""
        transforms_file = os.path.join(self.config.data_dir, f"transforms_{self.split}.json")
        
        with open(transforms_file, 'r') as f:
            transforms = json.load(f)
        
        self.images = []
        self.poses = []
        
        # Extract camera parameters
        camera_angle_x = transforms.get('camera_angle_x', 0.6911112070083618)
        focal = 0.5 * self.config.image_width / np.tan(0.5 * camera_angle_x)
        
        # Camera intrinsics
        K = np.array([
            [focal, 0, self.config.image_width / 2], [0, focal, self.config.image_height / 2], [0, 0, 1]
        ])
        
        for frame in transforms['frames']:
            # Load image
            img_path = os.path.join(self.config.data_dir, frame['file_path'] + '.png')
            if not os.path.exists(img_path):
                img_path = img_path.replace('.png', '.jpg')
            
            if os.path.exists(img_path):
                image = Image.open(img_path).convert('RGBA')
                image = image.resize((self.config.image_width, self.config.image_height))
                image = np.array(image) / 255.0
                
                # Handle alpha channel
                if image.shape[2] == 4:
                    alpha = image[..., 3:4]
                    rgb = image[..., :3]
                    
                    if self.config.white_background:
                        rgb = rgb * alpha + (1 - alpha)
                    elif self.config.black_background:
                        rgb = rgb * alpha
                    
                    image = rgb
                
                self.images.append(image)
                
                # Load pose
                pose = np.array(frame['transform_matrix'])
                self.poses.append(pose)
        
        self.images = np.array(self.images)
        self.poses = np.array(self.poses)
        self.intrinsics = np.tile(K[None, ...], (len(self.images), 1, 1))
    
    def _process_images(self):
        """Process loaded images."""
        if self.config.downscale_factor != 1.0:
            new_h = int(self.config.image_height / self.config.downscale_factor)
            new_w = int(self.config.image_width / self.config.downscale_factor)
            
            processed_images = []
            for img in self.images:
                img_tensor = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0)
                img_resized = F.interpolate(
                    img_tensor,
                    size=(new_h, new_w),
                    mode='bilinear',
                    align_corners=False,
                )
                img_resized = img_resized.squeeze(0).permute(1, 2, 0).numpy()
                processed_images.append(img_resized)
            
            self.images = np.array(processed_images)
            self.config.image_height = new_h
            self.config.image_width = new_w
    
    def _split_data(self):
        """Split data into train/val/test sets."""
        num_images = len(self.images)
        indices = np.arange(num_images)
        
        # Calculate split sizes
        train_size = int(num_images * self.config.train_split)
        val_size = int(num_images * self.config.val_split)
        
        if self.split == "train":
            self.indices = indices[:train_size]
        elif self.split == "val":
            self.indices = indices[train_size:train_size + val_size]
        elif self.split == "test":
            self.indices = indices[train_size + val_size:]
        else:
            self.indices = indices
        
        # Filter data
        self.images = self.images[self.indices]
        self.poses = self.poses[self.indices]
        self.intrinsics = self.intrinsics[self.indices]
    
    def _setup_rays(self):
        """Pre-compute ray information."""
        self.rays_o = []
        self.rays_d = []
        self.target_colors = []
        
        for i in range(len(self.images)):
            pose = self.poses[i]
            K = self.intrinsics[i]
            image = self.images[i]
            
            # Generate rays
            rays_o, rays_d = self._generate_rays(pose, K)
            
            # Flatten rays and colors
            rays_o_flat = rays_o.reshape(-1, 3)
            rays_d_flat = rays_d.reshape(-1, 3)
            colors_flat = image.reshape(-1, image.shape[-1])
            
            self.rays_o.append(rays_o_flat)
            self.rays_d.append(rays_d_flat)
            self.target_colors.append(colors_flat)
        
        # Concatenate all rays
        self.all_rays_o = np.concatenate(self.rays_o, axis=0)
        self.all_rays_d = np.concatenate(self.rays_d, axis=0)
        self.all_colors = np.concatenate(self.target_colors, axis=0)
    
    def _generate_rays(self, pose: np.ndarray, K: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Generate rays for a camera."""
        H, W = self.config.image_height, self.config.image_width
        
        # Pixel coordinates
        i, j = np.meshgrid(np.arange(W), np.arange(H), indexing='xy')
        
        # Camera coordinates
        dirs = np.stack([
            (i - K[0, 2]) / K[0, 0], -(j - K[1, 2]) / K[1, 1], -np.ones_like(i)
        ], axis=-1)
        
        # Transform to world coordinates
        rays_d = np.sum(dirs[..., None, :] * pose[:3, :3], axis=-1)
        rays_o = np.broadcast_to(pose[:3, 3], rays_d.shape)
        
        return rays_o, rays_d
    
    def __len__(self):
        """Return dataset size."""
        if self.split == "train":
            return len(self.all_rays_o) // self.config.num_rays_train
        else:
            return len(self.images)
    
    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        """Get a batch of data."""
        if self.split == "train":
            # Random ray sampling for training
            num_rays = self.config.num_rays_train
            ray_indices = np.random.choice(len(self.all_rays_o), num_rays, replace=False)
            
            rays_o = torch.from_numpy(self.all_rays_o[ray_indices]).float()
            rays_d = torch.from_numpy(self.all_rays_d[ray_indices]).float()
            colors = torch.from_numpy(self.all_colors[ray_indices]).float()
            
            return {
                'rays_o': rays_o, 'rays_d': rays_d, 'colors': colors
            }
        else:
            # Full image for validation/test
            image_idx = idx % len(self.images)
            
            rays_o = torch.from_numpy(self.rays_o[image_idx]).float()
            rays_d = torch.from_numpy(self.rays_d[image_idx]).float()
            colors = torch.from_numpy(self.target_colors[image_idx]).float()
            
            return {
                'rays_o': rays_o, 'rays_d': rays_d, 'colors': colors, 'image_idx': image_idx, 'pose': torch.from_numpy(
                    self.poses[image_idx],
                )
            }

    def get_dataset_info(self) -> dict[str, Any]:
        """Get dataset information."""

def create_svraster_dataloader(
    config: SVRasterDatasetConfig,
    split: str = "train",
    batch_size: int = 1,
    shuffle: bool = True,
    num_workers: int = 4,
) -> DataLoader:
    """Create a DataLoader for SVRaster dataset."""
    dataset = SVRasterDataset(config, split)
    
    return DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, pin_memory=True, drop_last=(
            split == "train",
        )
    )

def create_svraster_dataset(config: SVRasterDatasetConfig, split: str = "train") -> SVRasterDataset:
    """Create a SVRaster dataset."""
    return SVRasterDataset(config, split) 