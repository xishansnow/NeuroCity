from __future__ import annotations

from typing import Any

"""
Dataset module for InfNeRF supporting large-scale scenes and multi-resolution supervision.

This module implements:
- Large-scale scene data loading
- Image pyramid generation for multi-scale supervision  
- Ray sampling with Level of Detail awareness
- Sparse point integration from SfM
"""

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
import cv2
import json
import os
from pathlib import Path
from dataclasses import dataclass
import math

from .core import InfNeRFConfig


@dataclass
class InfNeRFDatasetConfig:
    """Configuration for InfNeRF dataset."""

    # Data paths
    data_root: str  # Root directory of dataset
    images_path: str = "images"  # Relative path to images
    sparse_points_path: str = "sparse_points.ply"  # SfM sparse points
    cameras_path: str = "cameras.json"  # Camera parameters

    # Image processing
    image_scale: float = 1.0  # Image downscaling factor
    max_image_size: int = 2048  # Maximum image dimension
    min_image_size: int = 256  # Minimum image dimension

    # Multi-resolution parameters
    num_pyramid_levels: int = 4  # Number of pyramid levels
    pyramid_scale_factor: float = 2.0  # Scale factor between levels
    uniform_pixel_sampling: bool = True  # Uniform sampling across pyramid levels

    # Ray sampling
    rays_per_image: int = 1024  # Rays sampled per image
    batch_size: int = 4096  # Ray batch size
    patch_size: int = 32  # Patch sampling size (optional)
    use_patch_sampling: bool = False  # Use patch-based sampling

    # Scene bounds
    scene_scale: float = 1.0  # Scene scaling factor
    near_plane: float = 0.1  # Near clipping plane
    far_plane: float = 100.0  # Far clipping plane

    # Data augmentation
    use_color_jitter: bool = False  # Apply color jittering
    color_jitter_strength: float = 0.1  # Color jitter strength
    use_random_crop: bool = False  # Apply random cropping
    crop_ratio: float = 0.9  # Crop ratio for random cropping


class InfNeRFDataset(Dataset):
    """
    Dataset for InfNeRF training supporting large-scale scenes.
    """

    def __init__(self, config: InfNeRFDatasetConfig, split: str = "train"):
        """
        Initialize InfNeRF dataset.

        Args:
            config: Dataset configuration
            split: Dataset split ('train', 'val', 'test')
        """
        self.config = config
        self.split = split

        # Load dataset
        self._load_dataset()

        # Build image pyramids for multi-scale supervision
        self._build_image_pyramids()

        # Prepare ray sampling
        self._prepare_ray_sampling()

        print(
            f"Loaded InfNeRF dataset: {len(self.images)} images, "
            f"{self.num_pyramid_levels} pyramid levels"
        )

    def _load_dataset(self):
        """Load images, cameras, and sparse points."""
        data_root = Path(self.config.data_root)

        # Load camera parameters
        cameras_file = data_root / self.config.cameras_path
        with open(cameras_file, "r") as f:
            self.cameras = json.load(f)

        # Load images
        images_dir = data_root / self.config.images_path
        self.image_paths = sorted(list(images_dir.glob("*.jpg")) + list(images_dir.glob("*.png")))

        # Filter images based on split
        if self.split == "train":
            # Use most images for training (e.g., 80%)
            split_idx = int(len(self.image_paths) * 0.8)
            self.image_paths = self.image_paths[:split_idx]
        elif self.split == "val":
            # Use some images for validation (e.g., 10%)
            start_idx = int(len(self.image_paths) * 0.8)
            end_idx = int(len(self.image_paths) * 0.9)
            self.image_paths = self.image_paths[start_idx:end_idx]
        else:  # test
            # Use remaining images for testing (e.g., 10%)
            start_idx = int(len(self.image_paths) * 0.9)
            self.image_paths = self.image_paths[start_idx:]

        # Load images into memory (for smaller datasets)
        self.images = []
        self.intrinsics = []
        self.extrinsics = []

        for img_path in self.image_paths:
            # Load image
            img = cv2.imread(str(img_path))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = img.astype(np.float32) / 255.0

            # Resize if needed
            if max(img.shape[:2]) > self.config.max_image_size:
                scale = self.config.max_image_size / max(img.shape[:2])
                new_size = (int(img.shape[1] * scale), int(img.shape[0] * scale))
                img = cv2.resize(img, new_size)

            self.images.append(img)

            # Get camera parameters
            img_name = img_path.name
            if img_name in self.cameras:
                cam = self.cameras[img_name]
                self.intrinsics.append(np.array(cam["intrinsic"]))
                self.extrinsics.append(np.array(cam["extrinsic"]))
            else:
                # Use default camera parameters
                h, w = img.shape[:2]
                focal = max(h, w)  # Rough estimate
                intrinsic = np.array([[focal, 0, w / 2], [0, focal, h / 2], [0, 0, 1]])
                extrinsic = np.eye(4)  # Identity for now
                self.intrinsics.append(intrinsic)
                self.extrinsics.append(extrinsic)

        # Load sparse points from SfM
        sparse_points_file = data_root / self.config.sparse_points_path
        if sparse_points_file.exists():
            self.sparse_points = self._load_sparse_points(sparse_points_file)
        else:
            print(f"Warning: Sparse points file not found: {sparse_points_file}")
            self.sparse_points = np.array([]).reshape(0, 3)

        print(f"Loaded {len(self.sparse_points)} sparse points")

    def _load_sparse_points(self, ply_file: Path) -> np.ndarray:
        """Load sparse points from PLY file."""
        # Simplified PLY loading - in practice would use plyfile library
        points = []

        try:
            with open(ply_file, "r") as f:
                lines = f.readlines()

            # Find vertex count
            vertex_count = 0
            header_end = 0
            for i, line in enumerate(lines):
                if line.startswith("element vertex"):
                    vertex_count = int(line.split()[-1])
                elif line.startswith("end_header"):
                    header_end = i + 1
                    break

            # Read vertices
            for i in range(header_end, header_end + vertex_count):
                parts = lines[i].strip().split()
                if len(parts) >= 3:
                    x, y, z = float(parts[0]), float(parts[1]), float(parts[2])
                    points.append([x, y, z])

        except Exception as e:
            print(f"Error loading sparse points: {e}")
            return np.array([]).reshape(0, 3)

        return np.array(points) * self.config.scene_scale

    def _build_image_pyramids(self):
        """Build image pyramids for multi-scale supervision."""
        self.image_pyramids = []
        self.num_pyramid_levels = self.config.num_pyramid_levels

        for img in self.images:
            pyramid = [img]  # Original resolution
            current_img = img

            # Build pyramid by downsampling
            for level in range(1, self.num_pyramid_levels):
                scale = 1.0 / (self.config.pyramid_scale_factor**level)
                new_size = max(
                    1,
                    int,
                )
                current_img = cv2.resize(current_img, new_size)
                pyramid.append(current_img)

            self.image_pyramids.append(pyramid)

    def _prepare_ray_sampling(self):
        """Prepare data structures for efficient ray sampling."""
        self.num_images = len(self.images)

        # Calculate pixel weights for uniform sampling across pyramid levels
        if self.config.uniform_pixel_sampling:
            self.level_weights = []
            for level in range(self.num_pyramid_levels):
                # Weight inversely proportional to pixel count at this level
                scale = self.config.pyramid_scale_factor**level
                weight = scale**2  # Higher resolution gets lower weight
                self.level_weights.append(weight)

            # Normalize weights
            total_weight = sum(self.level_weights)
            self.level_weights = [w / total_weight for w in self.level_weights]
        else:
            # Equal weights for all levels
            self.level_weights = [1.0 / self.num_pyramid_levels] * self.num_pyramid_levels

    def __len__(self) -> int:
        """Return dataset size."""
        if self.split == "train":
            # For training, we can sample indefinitely
            return len(self.images) * 1000  # Arbitrary large number
        else:
            return len(self.images)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        """
        Get training sample.

        Args:
            idx: Sample index

        Returns:
            Dictionary containing rays and target colors
        """
        # Select image
        img_idx = idx % self.num_images

        # Sample pyramid level based on weights
        level = np.random.choice(self.num_pyramid_levels, p=self.level_weights)

        # Get image at selected level
        img = self.image_pyramids[img_idx][level]
        intrinsic = self.intrinsics[img_idx].copy()
        extrinsic = self.extrinsics[img_idx]

        # Adjust intrinsics for pyramid level
        scale = self.config.pyramid_scale_factor ** (-level)
        intrinsic[:2, :] *= scale

        # Sample rays
        rays_o, rays_d, target_rgb, pixel_coords = self._sample_rays(
            img, intrinsic, extrinsic, self.config.rays_per_image
        )

        # Calculate pixel width for anti-aliasing
        focal_length = (intrinsic[0, 0] + intrinsic[1, 1]) / 2
        pixel_width = 1.0  # Normalized pixel width

        return {
            "rays_o": torch.from_numpy(
                rays_o,
            ),
            "rays_d": torch.from_numpy(rays_d),
            "target_rgb": torch.from_numpy(target_rgb),
            "pixel_coords": torch.from_numpy(pixel_coords),
            "focal_length": focal_length,
            "pixel_width": pixel_width,
        }

    def _sample_rays(
        self,
        img: np.ndarray,
        intrinsic: np.ndarray,
        extrinsic: np.ndarray,
        num_rays: int,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Sample rays from image.

        Args:
            img: Image array [H, W, 3]
            intrinsic: Camera intrinsic matrix [3, 3]
            extrinsic: Camera extrinsic matrix [4, 4]
            num_rays: Number of rays to sample

        Returns:
            rays_o: Ray origins [num_rays, 3]
            rays_d: Ray directions [num_rays, 3]
            target_rgb: Target colors [num_rays, 3]
            pixel_coords: Pixel coordinates [num_rays, 2]
        """
        H, W = img.shape[:2]

        if self.config.use_patch_sampling:
            # Patch-based sampling
            patch_size = self.config.patch_size
            num_patches = num_rays // (patch_size * patch_size)

            pixel_coords = []
            for _ in range(num_patches):
                # Random patch top-left corner
                top = np.random.randint(0, H - patch_size)
                left = np.random.randint(0, W - patch_size)

                # Generate patch coordinates
                for dy in range(patch_size):
                    for dx in range(patch_size):
                        pixel_coords.append([left + dx, top + dy])

            pixel_coords = np.array(pixel_coords[:num_rays])
        else:
            # Random pixel sampling
            pixels_x = np.random.randint(0, W, num_rays)
            pixels_y = np.random.randint(0, H, num_rays)
            pixel_coords = np.stack([pixels_x, pixels_y], axis=1)

        # Get target colors
        target_rgb = img[pixel_coords[:, 1], pixel_coords[:, 0]]

        # Generate rays
        rays_o, rays_d = self._pixels_to_rays(pixel_coords, intrinsic, extrinsic)

        return rays_o, rays_d, target_rgb, pixel_coords

    def _pixels_to_rays(
        self,
        pixel_coords: np.ndarray,
        intrinsic: np.ndarray,
        extrinsic: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Convert pixel coordinates to 3D rays.

        Args:
            pixel_coords: Pixel coordinates [N, 2]
            intrinsic: Camera intrinsic matrix [3, 3]
            extrinsic: Camera extrinsic matrix [4, 4]

        Returns:
            rays_o: Ray origins [N, 3]
            rays_d: Ray directions [N, 3]
        """
        # Convert to homogeneous coordinates
        pixels_homo = np.ones((len(pixel_coords), 3))
        pixels_homo[:, :2] = pixel_coords

        # Back-project to camera coordinates
        cam_coords = np.linalg.inv(intrinsic) @ pixels_homo.T  # [3, N]
        cam_coords = cam_coords.T  # [N, 3]

        # Convert to world coordinates
        rotation = extrinsic[:3, :3]
        translation = extrinsic[:3, 3]

        # Ray directions in world coordinates
        rays_d = cam_coords @ rotation  # [N, 3]
        rays_d = rays_d / np.linalg.norm(rays_d, axis=1, keepdims=True)

        # Ray origins (camera center in world coordinates)
        camera_center = -rotation.T @ translation  # [3]
        rays_o = np.tile(camera_center, (len(pixel_coords), 1))  # [N, 3]

        return rays_o, rays_d

    def get_sparse_points(self) -> np.ndarray:
        """Get sparse points from SfM."""
        return self.sparse_points.copy()

    def get_all_camera_poses(self) -> list[np.ndarray]:
        """Get all camera poses for testing."""
        return [ext.copy() for ext in self.extrinsics]

    def get_image_at_level(self, img_idx: int, level: int) -> np.ndarray:
        """Get image at specific pyramid level."""
        if 0 <= img_idx < len(self.image_pyramids) and 0 <= level < self.num_pyramid_levels:
            return self.image_pyramids[img_idx][level]
        else:
            raise IndexError("Invalid image index or pyramid level")


def create_inf_nerf_dataloader(
    config: InfNeRFDatasetConfig,
    split: str = "train",
    batch_size: int | None = None,
    shuffle: bool = True,
    num_workers: int = 4,
) -> DataLoader:
    """
    Create DataLoader for InfNeRF dataset.

    Args:
        config: Dataset configuration
        split: Dataset split
        batch_size: Batch size (uses config.batch_size if None)
        shuffle: Whether to shuffle data
        num_workers: Number of worker processes

    Returns:
        DataLoader instance
    """
    dataset = InfNeRFDataset(config, split)

    if batch_size is None:
        batch_size = config.batch_size

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True if split == "train" else False,
    )

    return dataloader


# Utility functions for data preparation
def prepare_colmap_data(colmap_dir: str, output_dir: str):
    """
    Convert COLMAP reconstruction to InfNeRF format.

    Args:
        colmap_dir: Directory containing COLMAP reconstruction
        output_dir: Output directory for InfNeRF dataset
    """
    # This would contain code to convert COLMAP format to InfNeRF format
    # Including cameras.txt, images.txt, points3D.txt -> cameras.json, sparse_points.ply
    pass


def prepare_nerfstudio_data(nerfstudio_dir: str, output_dir: str):
    """
    Convert NeRFStudio dataset to InfNeRF format.

    Args:
        nerfstudio_dir: Directory containing NeRFStudio dataset
        output_dir: Output directory for InfNeRF dataset
    """
    # This would contain code to convert NeRFStudio format to InfNeRF format
    pass
