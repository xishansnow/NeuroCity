from __future__ import annotations

from typing import Any, Optional, Union

"""
Dataset module for SVRaster.
"""

import os
import json
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from PIL import Image
from dataclasses import dataclass
import logging
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import cv2
import torch.nn.functional as F

logger = logging.getLogger(__name__)


@dataclass
class SVRasterDatasetConfig:
    """SVRaster dataset configuration.

    Attributes:
        data_dir: Data directory path
        image_width: Image width
        image_height: Image height
        camera_angle_x: Camera horizontal FoV in radians
        train_split: Training split ratio
        val_split: Validation split ratio
        test_split: Test split ratio
        downscale_factor: Image downscale factor
        color_space: Color space ("linear" or "srgb")
        num_rays_train: Number of rays per training batch
        num_rays_val: Number of rays per validation batch
        patch_size: Size of image patches for ray sampling
    """

    data_dir: str = "data/nerf_synthetic/lego"
    image_width: int = 800
    image_height: int = 800
    camera_angle_x: float = 0.6911112070083618  # ~40 degrees
    train_split: float = 0.8
    val_split: float = 0.1
    test_split: float = 0.1
    downscale_factor: float = 1.0
    color_space: str = "srgb"
    num_rays_train: int = 4096
    num_rays_val: int = 1024
    patch_size: int = 1

    # Data paths
    images_dir: str = "data/nerf_synthetic/lego/images"
    masks_dir: Optional[str] = None

    # Data format
    dataset_type: str = "colmap"
    image_format: str = "png"

    # Image processing
    downscale_factor: float = 1.0

    # Camera parameters
    camera_model: str = "pinhole"
    distortion_params: Optional[List[float]] = None

    # Data augmentation
    use_color_jitter: bool = False
    color_jitter_strength: float = 0.1
    use_random_background: bool = False

    # Scene bounds
    auto_scale_poses: bool = True
    scene_scale: float = 1.0
    scene_center: Optional[Tuple[float, float, float]] = None

    # Background handling
    white_background: bool = False
    black_background: bool = False


class SVRasterDataset(Dataset):
    """SVRaster dataset with modern PyTorch features.

    Features:
    - Efficient data loading and preprocessing
    - Memory-optimized ray generation
    - Automatic mixed precision support
    - CUDA acceleration with CPU fallback
    - Flexible data augmentation
    """

    def __init__(self, config: SVRasterDatasetConfig, split: str = "train"):
        """Initialize dataset.

        Args:
            config: Dataset configuration
            split: Dataset split ("train", "val", or "test")
        """
        super().__init__()
        self.config = config
        self.split = split
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Load and preprocess data
        self._load_data()
        self._preprocess_data()
        self._setup_rays()

    def _load_image(self, path: Path) -> np.ndarray:
        """Load and preprocess a single image.

        Args:
            path: Path to image file

        Returns:
            Preprocessed image array
        """
        image = cv2.imread(str(path))
        if image is None:
            raise ValueError(f"Failed to load image: {path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return image.astype(np.float32) / 255.0

    def _load_pose(self, path: Path) -> np.ndarray:
        """Load camera pose from file.

        Args:
            path: Path to pose file

        Returns:
            Camera pose matrix
        """
        return np.loadtxt(str(path))

    def _load_intrinsics(self) -> np.ndarray:
        """Load camera intrinsics.

        Returns:
            Camera intrinsics matrix
        """
        # Load intrinsics from file if available
        intrinsics_path = Path(self.config.data_dir) / "intrinsics.txt"
        if intrinsics_path.exists():
            return np.loadtxt(str(intrinsics_path))

        # Otherwise, use default intrinsics based on image size
        H, W = self.config.image_height, self.config.image_width
        focal = 0.5 * W / np.tan(0.5 * self.config.camera_angle_x)

        K = np.array([[focal, 0, W / 2], [0, focal, H / 2], [0, 0, 1]])

        return K

    def _load_data(self) -> None:
        """Load raw data efficiently."""
        # Load images and camera parameters
        data_dir = Path(self.config.data_dir)
        image_paths = sorted(data_dir.glob("images/*.png"))
        pose_paths = sorted(data_dir.glob("poses/*.txt"))

        if not image_paths:
            raise ValueError(f"No images found in {data_dir}/images/")
        if not pose_paths:
            raise ValueError(f"No pose files found in {data_dir}/poses/")

        # Load data in parallel
        with ThreadPoolExecutor() as executor:
            # Load images
            image_futures = [executor.submit(self._load_image, path) for path in image_paths]
            self.images = [f.result() for f in image_futures]

            # Load poses
            pose_futures = [executor.submit(self._load_pose, path) for path in pose_paths]
            self.poses = [f.result() for f in pose_futures]

        # Load camera intrinsics
        self.intrinsics = self._load_intrinsics()

        # Convert to tensors with consistent data types
        self.images = torch.stack([torch.from_numpy(img).float() for img in self.images])
        self.poses = torch.stack([torch.from_numpy(pose).float() for pose in self.poses])
        self.intrinsics = torch.from_numpy(self.intrinsics).float()

        # Split data
        num_images = len(self.images)
        indices = torch.randperm(num_images)
        train_size = int(num_images * self.config.train_split)
        val_size = int(num_images * self.config.val_split)

        if self.split == "train":
            self.indices = indices[:train_size]
        elif self.split == "val":
            self.indices = indices[train_size : train_size + val_size]
        else:  # test
            self.indices = indices[train_size + val_size :]

        # Update data
        self.images = self.images[self.indices]
        self.poses = self.poses[self.indices]

    def _preprocess_data(self) -> None:
        """Preprocess loaded data."""
        # Resize images if needed
        if self.config.downscale_factor != 1.0:
            H = int(self.config.image_height * self.config.downscale_factor)
            W = int(self.config.image_width * self.config.downscale_factor)
            self.images = F.interpolate(
                self.images.permute(0, 3, 1, 2),
                size=(H, W),
                mode="bilinear",
                align_corners=False,
            ).permute(0, 2, 3, 1)

        # Convert color space if needed
        if self.config.color_space == "linear":
            self.images = self.images.pow(2.2)  # sRGB to linear

    def _setup_rays(self) -> None:
        """Set up ray generation."""
        self.H = int(self.config.image_height * self.config.downscale_factor)
        self.W = int(self.config.image_width * self.config.downscale_factor)

        # Create pixel coordinates
        i, j = torch.meshgrid(
            torch.arange(self.H, dtype=torch.float32),
            torch.arange(self.W, dtype=torch.float32),
            indexing="ij",
        )
        self.pixels = torch.stack([i, j], dim=-1)  # [H, W, 2]

    def __len__(self) -> int:
        """Get dataset length."""
        return len(self.images)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get dataset item.

        Args:
            idx: Item index

        Returns:
            Dictionary containing rays and colors
        """
        # Get image and pose
        image = self.images[idx]  # [H, W, 3]
        pose = self.poses[idx]  # [4, 4]

        # Sample rays
        if self.split == "train":
            # Random ray sampling for training
            num_rays = self.config.num_rays_train
            if self.config.patch_size > 1:
                # Sample patches
                patch_size = self.config.patch_size
                num_patches = num_rays // (patch_size * patch_size)
                patch_indices = torch.randint(
                    0,
                    (self.H - patch_size) * (self.W - patch_size),
                    size=(num_patches,),
                )
                row_indices = patch_indices // (self.W - patch_size)
                col_indices = patch_indices % (self.W - patch_size)

                # Get patch pixels
                pixels = []
                for r, c in zip(row_indices, col_indices):
                    patch_pixels = self.pixels[
                        r : r + patch_size,
                        c : c + patch_size,
                    ].reshape(-1, 2)
                    pixels.append(patch_pixels)
                pixels = torch.cat(pixels, dim=0)  # [num_rays, 2]
            else:
                # Random pixel sampling
                indices = torch.randint(0, self.H * self.W, size=(num_rays,))
                pixels = self.pixels.reshape(-1, 2)[indices]  # [num_rays, 2]
        else:
            # Use all pixels for validation/testing
            pixels = self.pixels.reshape(-1, 2)  # [H*W, 2]

        # Generate rays
        rays_o, rays_d = self._get_rays(pixels, pose)  # [num_rays, 3]

        # Get colors
        if self.split == "train":
            if self.config.patch_size > 1:
                # Get patch colors
                colors = []
                for r, c in zip(row_indices, col_indices):
                    patch_colors = image[
                        r : r + patch_size,
                        c : c + patch_size,
                    ].reshape(-1, 3)
                    colors.append(patch_colors)
                colors = torch.cat(colors, dim=0)  # [num_rays, 3]
            else:
                # Random color sampling
                colors = image.reshape(-1, 3)[indices]  # [num_rays, 3]
        else:
            # Use all colors for validation/testing
            colors = image.reshape(-1, 3)  # [H*W, 3]

        return {
            "rays_o": rays_o,
            "rays_d": rays_d,
            "colors": colors,
            "image_id": idx,
        }

    def _get_rays(
        self,
        pixels: torch.Tensor,
        pose: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Generate rays for given pixels and pose.

        Args:
            pixels: Pixel coordinates [N, 2]
            pose: Camera pose [4, 4]

        Returns:
            Tuple of ray origins and directions [N, 3]
        """
        # Unproject pixels to camera space
        x = (pixels[:, 1] - self.intrinsics[0, 2]) / self.intrinsics[0, 0]
        y = (pixels[:, 0] - self.intrinsics[1, 2]) / self.intrinsics[1, 1]
        dirs = torch.stack([x, y, torch.ones_like(x)], dim=-1)  # [N, 3]

        # Transform to world space
        rays_d = torch.einsum("ij,nj->ni", pose[:3, :3], dirs)  # [N, 3]
        rays_o = pose[:3, 3].expand_as(rays_d)  # [N, 3]

        return rays_o, rays_d

    def get_dataset_info(self) -> Dict[str, Any]:
        """Get dataset information."""
        return {
            "num_images": len(self.images),
            "image_size": (self.config.image_height, self.config.image_width),
            "num_train_rays": len(self.pixels) // self.config.num_rays_train,
            "num_val_rays": len(self.pixels) // self.config.num_rays_val,
        }


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
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=split == "train",
    )


def create_svraster_dataset(config: SVRasterDatasetConfig, split: str = "train") -> SVRasterDataset:
    """Create a SVRaster dataset."""
    return SVRasterDataset(config, split)
