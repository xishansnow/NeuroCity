from __future__ import annotations

from typing import Any

"""
Nerfacto Dataset Module

This module implements dataset handling for Nerfacto, supporting various data formats
including COLMAP, BlenderNeRF, Instant-NGP format, and custom formats.

Nerfacto is designed to work with real-world captured data and synthetic datasets.
"""

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
import json
import os
from pathlib import Path
from dataclasses import dataclass
import cv2
from PIL import Image
import glob
from scipy.spatial.transform import Rotation


@dataclass
class NerfactoDatasetConfig:
    """Configuration for Nerfacto dataset."""

    # Data paths
    data_dir: str = "data"
    images_dir: str = "images"

    # Data format
    data_format: str = "colmap"  # colmap, blender, instant_ngp, custom

    # Image settings
    image_width: Optional[int] = None
    image_height: Optional[int] = None
    downscale_factor: int = 1

    # Camera settings
    camera_model: str = "perspective"  # perspective, fisheye, equirectangular
    distortion_params: Optional[list[float]] = None

    # Data splitting
    train_split_fraction: float = 0.9
    val_split_fraction: float = 0.1
    test_split_fraction: float = 0.0

    # Scene bounds
    scene_bounds: Optional[tuple[float, float, float, float, float, float]] = None
    auto_scale: bool = True

    # Sampling settings
    patch_size: Optional[int] = None  # For patch-based training
    num_rays_per_batch: int = 4096

    # Data augmentation
    color_jitter: bool = False
    random_crop: bool = False

    # Advanced settings
    load_normals: bool = False
    load_depths: bool = False
    load_masks: bool = False

    # Memory optimization
    cache_images: bool = False
    use_cached_dataset: bool = False


class CameraModel:
    """Camera model for handling different projection types."""

    def __init__(self, camera_type: str = "perspective"):
        self.camera_type = camera_type

    def project_points(self, points: torch.Tensor, intrinsics: torch.Tensor) -> torch.Tensor:
        """Project 3D points to 2D image coordinates."""
        if self.camera_type == "perspective":
            return self._perspective_projection(points, intrinsics)
        elif self.camera_type == "fisheye":
            return self._fisheye_projection(points, intrinsics)
        else:
            raise ValueError(f"Unsupported camera type: {self.camera_type}")

    def _perspective_projection(
        self,
        points: torch.Tensor,
        intrinsics: torch.Tensor,
    ) -> torch.Tensor:
        """Standard perspective projection."""
        fx, fy, cx, cy = intrinsics[0], intrinsics[1], intrinsics[2], intrinsics[3]

        # Project to image plane
        x = points[..., 0] / points[..., 2]
        y = points[..., 1] / points[..., 2]

        # Apply intrinsics
        u = fx * x + cx
        v = fy * y + cy

        return torch.stack([u, v], dim=-1)

    def _fisheye_projection(self, points: torch.Tensor, intrinsics: torch.Tensor) -> torch.Tensor:
        """Fisheye projection model."""
        fx, fy, cx, cy = intrinsics[0], intrinsics[1], intrinsics[2], intrinsics[3]

        # Convert to spherical coordinates
        r = torch.norm(points[..., :2], dim=-1)
        theta = torch.atan2(r, points[..., 2])
        phi = torch.atan2(points[..., 1], points[..., 0])

        # Fisheye distortion
        r_distorted = theta

        # Project to image
        u = fx * r_distorted * torch.cos(phi) + cx
        v = fy * r_distorted * torch.sin(phi) + cy

        return torch.stack([u, v], dim=-1)


class ImageProcessor:
    """Image processing utilities for Nerfacto dataset."""

    @staticmethod
    def load_image(image_path: str, target_size: Optional[tuple[int, int]] = None) -> torch.Tensor:
        """Load and preprocess image."""
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image not found: {image_path}")

        # Load image
        image = Image.open(image_path).convert("RGB")

        # Resize if needed
        if target_size is not None:
            image = image.resize(target_size, Image.LANCZOS)

        # Convert to tensor
        image_tensor = torch.from_numpy(np.array(image)).float() / 255.0

        return image_tensor

    @staticmethod
    def apply_color_jitter(
        image: torch.Tensor,
        brightness: float = 0.1,
        contrast: float = 0.1,
        saturation: float = 0.1,
        hue: float = 0.05,
    ) -> torch.Tensor:
        """Apply color jittering augmentation."""
        # Convert to PIL for color jittering
        from torchvision import transforms

        jitter = transforms.ColorJitter(
            brightness=brightness, contrast=contrast, saturation=saturation, hue=hue
        )

        # Convert tensor to PIL and back
        pil_image = transforms.ToPILImage()(image.permute(2, 0, 1))
        jittered = jitter(pil_image)

        return transforms.ToTensor()(jittered).permute(1, 2, 0)


class COLMAPDataParser:
    """Parser for COLMAP data format."""

    @staticmethod
    def parse_colmap_data(data_dir: str) -> dict[str, Any]:
        """Parse COLMAP reconstruction data."""
        # This is a simplified version - full COLMAP parsing would require
        # reading binary files and handling various camera models

        cameras_file = os.path.join(data_dir, "cameras.txt")
        images_file = os.path.join(data_dir, "images.txt")
        points3d_file = os.path.join(data_dir, "points3D.txt")

        cameras = {}
        images = {}

        # Parse cameras
        if os.path.exists(cameras_file):
            with open(cameras_file, "r") as f:
                for line in f:
                    if line.startswith("#"):
                        continue
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        cam_id = int(parts[0])
                        model = parts[1]
                        width = int(parts[2])
                        height = int(parts[3])
                        params = [float(p) for p in parts[4:]]

                        cameras[cam_id] = {
                            "model": model,
                            "width": width,
                            "height": height,
                            "params": params,
                        }

        # Parse images
        if os.path.exists(images_file):
            with open(images_file, "r") as f:
                lines = f.readlines()
                for i in range(0, len(lines), 2):  # Images are on every other line
                    line = lines[i]
                    if line.startswith("#"):
                        continue
                    parts = line.strip().split()
                    if len(parts) >= 10:
                        img_id = int(parts[0])
                        qw, qx, qy, qz = [float(p) for p in parts[1:5]]
                        tx, ty, tz = [float(p) for p in parts[5:8]]
                        cam_id = int(parts[8])
                        name = parts[9]

                        # Convert quaternion to rotation matrix
                        rot = Rotation.from_quat([qx, qy, qz, qw])
                        R = rot.as_matrix()
                        t = np.array([tx, ty, tz])

                        # COLMAP uses world-to-camera, we need camera-to-world
                        c2w = np.eye(4)
                        c2w[:3, :3] = R.T
                        c2w[:3, 3] = -R.T @ t

                        images[img_id] = {
                            "name": name,
                            "camera_id": cam_id,
                            "pose": c2w,
                            "quat": [qw, qx, qy, qz],
                            "trans": [tx, ty, tz],
                        }

        return {"cameras": cameras, "images": images}


class BlenderDataParser:
    """Parser for Blender NeRF data format."""

    @staticmethod
    def parse_blender_data(data_dir: str) -> dict[str, Any]:
        """Parse Blender NeRF format data."""
        transform_files = ["transforms_train.json", "transforms_val.json", "transforms_test.json"]

        all_data = {}

        for split_file in transform_files:
            file_path = os.path.join(data_dir, split_file)
            if os.path.exists(file_path):
                with open(file_path, "r") as f:
                    data = json.load(f)

                split_name = split_file.replace("transforms_", "").replace(".json", "")
                all_data[split_name] = data

        return all_data


class NerfactoDataset(Dataset):
    """Main Nerfacto dataset class."""

    def __init__(self, config: NerfactoDatasetConfig, split: str = "train"):
        self.config = config
        self.split = split
        self.camera_model = CameraModel(config.camera_model)

        # Initialize data containers
        self.images = []
        self.poses = []
        self.intrinsics = []
        self.image_paths = []

        # Load data
        self._load_data()

        # Auto-scale scene if needed
        if self.config.auto_scale:
            self._auto_scale_scene()

        # Cache images if requested
        if self.config.cache_images:
            self._cache_images()

    def _load_data(self):
        """Load data based on format."""
        if self.config.data_format == "colmap":
            self._load_colmap_data()
        elif self.config.data_format == "blender":
            self._load_blender_data()
        elif self.config.data_format == "instant_ngp":
            self._load_instant_ngp_data()
        else:
            raise ValueError(f"Unsupported data format: {self.config.data_format}")

    def _load_colmap_data(self):
        """Load COLMAP format data."""
        self.data = COLMAPDataParser.parse_colmap_data(self.config.data_dir)

        for img_id, img_data in self.data["images"].items():
            camera_id = img_data["camera_id"]
            camera = self.data["cameras"][camera_id]

            # Get image path
            image_name = img_data["name"]
            image_path = os.path.join(self.config.data_dir, self.config.images_dir, image_name)

            if os.path.exists(image_path):
                self.image_paths.append(image_path)
                self.poses.append(torch.from_numpy(img_data["pose"]).float())

                # Extract intrinsics (simplified for PINHOLE model)
                if camera["model"] in ["PINHOLE", "SIMPLE_PINHOLE"]:
                    if len(camera["params"]) >= 4:
                        fx, fy, cx, cy = camera["params"][:4]
                    else:
                        f, cx, cy = camera["params"][:3]
                        fx = fy = f

                    intrinsics = torch.tensor([fx, fy, cx, cy])
                    self.intrinsics.append(intrinsics)

    def _load_blender_data(self):
        """Load Blender NeRF format data."""
        self.data = BlenderDataParser.parse_blender_data(self.config.data_dir)

        for split_name, split_data in self.data.items():
            if split_name == self.split:
                frames = split_data["frames"]
                break

        # Get camera intrinsics
        if "camera_angle_x" in frames[0]:
            fov_x = frames[0]["camera_angle_x"]

            # Determine image size
            if self.config.image_width and self.config.image_height:
                w, h = self.config.image_width, self.config.image_height
            else:
                # Try to get from first image
                if frames:
                    first_frame = frames[0]
                    img_path = os.path.join(self.config.data_dir, first_frame["file_path"])
                    if img_path.endswith(".png") or img_path.endswith(".jpg"):
                        pass
                    else:
                        img_path += ".png"

                    if os.path.exists(img_path):
                        img = Image.open(img_path)
                        w, h = img.size
                    else:
                        w, h = 800, 600  # Default

            # Compute focal length
            focal = 0.5 * w / np.tan(0.5 * fov_x)
            cx, cy = w * 0.5, h * 0.5

            base_intrinsics = torch.tensor([focal, focal, cx, cy])

        for frame in frames:
            # Get pose matrix
            transform_matrix = np.array(frame["transform_matrix"])
            pose = torch.from_numpy(transform_matrix).float()

            # Get image path
            file_path = frame["file_path"]
            if not file_path.endswith((".png", ".jpg", ".jpeg")):
                file_path += ".png"

            image_path = os.path.join(self.config.data_dir, file_path)

            if os.path.exists(image_path):
                self.image_paths.append(image_path)
                self.poses.append(pose)
                self.intrinsics.append(base_intrinsics.clone())

    def _load_instant_ngp_data(self):
        """Load Instant-NGP format data."""
        self.data = self._load_instant_ngp_data()

        for frame in self.data["frames"]:
            # Get pose matrix
            transform_matrix = np.array(frame["transform_matrix"])
            pose = torch.from_numpy(transform_matrix).float()

            # Get image path
            file_path = frame["file_path"]
            if not file_path.endswith((".png", ".jpg", ".jpeg")):
                file_path += ".png"

            image_path = os.path.join(self.config.data_dir, file_path)

            if os.path.exists(image_path):
                self.image_paths.append(image_path)
                self.poses.append(pose)
                self.intrinsics.append(self.data["camera_angle_x"])

    def _auto_scale_scene(self):
        """Automatically scale scene to unit cube."""
        if len(self.poses) == 0:
            return

        # Extract camera positions
        positions = torch.stack([pose[:3, 3] for pose in self.poses])

        # Compute scene bounds
        min_bounds = torch.min(positions, dim=0)[0]
        max_bounds = torch.max(positions, dim=0)[0]
        scene_center = (min_bounds + max_bounds) * 0.5
        scene_scale = torch.max(max_bounds - min_bounds)

        # Apply scaling and centering
        for i in range(len(self.poses)):
            self.poses[i][:3, 3] = (self.poses[i][:3, 3] - scene_center) / scene_scale

    def _cache_images(self):
        """Cache all images in memory."""
        print("Caching images...")
        for i, image_path in enumerate(self.image_paths):
            target_size = None
            if self.config.image_width and self.config.image_height:
                target_size = (self.config.image_width, self.config.image_height)

            image = ImageProcessor.load_image(image_path, target_size)
            self.images.append(image)

        print(f"Cached {len(self.images)} images")

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        """Get dataset item."""
        # Load image
        if self.config.cache_images and idx < len(self.images):
            image = self.images[idx]
        else:
            target_size = None
            if self.config.image_width and self.config.image_height:
                target_size = (self.config.image_width, self.config.image_height)

            image = ImageProcessor.load_image(self.image_paths[idx], target_size)

        # Apply augmentations
        if self.split == "train" and self.config.color_jitter:
            image = ImageProcessor.apply_color_jitter(image)

        # Get pose and intrinsics
        pose = self.poses[idx]
        intrinsics = self.intrinsics[idx]

        # Generate rays if patch-based training
        if self.config.patch_size is not None:
            rays_o, rays_d, colors = self._generate_patch_rays(image, pose, intrinsics)
        else:
            rays_o, rays_d, colors = self._generate_all_rays(image, pose, intrinsics)

        return {
            "image": image,
            "pose": pose,
            "intrinsics": intrinsics,
            "rays_o": rays_o,
            "rays_d": rays_d,
            "colors": colors,
            "image_idx": torch.tensor(idx),
        }

    def _generate_all_rays(
        self,
        image: torch.Tensor,
        pose: torch.Tensor,
        intrinsics: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Generate rays for entire image."""
        H, W = image.shape[:2]

        # Create pixel coordinates
        i, j = torch.meshgrid(torch.arange(W), torch.arange(H), indexing="ij")
        i = i.t().float()
        j = j.t().float()

        # Convert to camera coordinates
        fx, fy, cx, cy = intrinsics[0], intrinsics[1], intrinsics[2], intrinsics[3]

        # Normalize pixel coordinates
        x = (i - cx) / fx
        y = (j - cy) / fy
        z = torch.ones_like(x)

        # Ray directions in camera coordinates
        dirs = torch.stack([x, y, z], dim=-1)  # [H, W, 3]

        # Transform to world coordinates
        rays_d = torch.sum(dirs[..., None, :] * pose[:3, :3], dim=-1)  # [H, W, 3]
        rays_o = pose[:3, 3].expand(rays_d.shape)  # [H, W, 3]

        # Flatten
        rays_o = rays_o.reshape(-1, 3)
        rays_d = rays_d.reshape(-1, 3)
        colors = image.reshape(-1, 3)

        return rays_o, rays_d, colors

    def _generate_patch_rays(
        self,
        image: torch.Tensor,
        pose: torch.Tensor,
        intrinsics: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Generate rays for random patch."""
        H, W = image.shape[:2]
        patch_size = self.config.patch_size

        # Random patch location
        top = torch.randint(0, H - patch_size + 1, (1,)).item()
        left = torch.randint(0, W - patch_size + 1, (1,)).item()

        # Extract patch
        patch = image[top : top + patch_size, left : left + patch_size]

        # Generate rays for patch
        i, j = torch.meshgrid(torch.arange(patch_size), torch.arange(patch_size), indexing="ij")
        i = i.t().float()
        j = j.t().float()

        fx, fy, cx, cy = intrinsics[0], intrinsics[1], intrinsics[2], intrinsics[3]

        x = (i - cx) / fx
        y = (j - cy) / fy
        z = torch.ones_like(x)

        dirs = torch.stack([x, y, z], dim=-1)
        rays_d = torch.sum(dirs[..., None, :] * pose[:3, :3], dim=-1)
        rays_o = pose[:3, 3].expand(rays_d.shape)

        rays_o = rays_o.reshape(-1, 3)
        rays_d = rays_d.reshape(-1, 3)
        colors = patch.reshape(-1, 3)

        return rays_o, rays_d, colors


def create_nerfacto_dataloader(
    config: NerfactoDatasetConfig,
    split: str = "train",
    batch_size: int = 1,
    shuffle: bool = True,
    num_workers: int = 4,
) -> DataLoader:
    """Create Nerfacto dataloader."""
    dataset = NerfactoDataset(config, split)

    return DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, pin_memory=True
    )


def create_nerfacto_dataset(
    data_dir: str,
    data_format: str = "colmap",
    image_size: Optional[tuple[int, int]] = None,
    **kwargs,
) -> dict[str, NerfactoDataset]:
    """Factory function to create Nerfacto datasets."""
    config = NerfactoDatasetConfig(data_dir=data_dir, data_format=data_format, **kwargs)

    if image_size:
        config.image_width, config.image_height = image_size

    datasets = {}

    # Create datasets for each split
    for split in ["train", "val", "test"]:
        try:
            dataset = NerfactoDataset(config, split)
            if len(dataset) > 0:
                datasets[split] = dataset
        except (FileNotFoundError, KeyError):
            # Skip splits that don't exist
            continue

    return datasets
