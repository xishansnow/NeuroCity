"""
Block-NeRF Dataset

This module provides dataset functionality for Block-NeRF training and evaluation.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
from PIL import Image

# Type aliases
Tensor = torch.Tensor
TensorDict = dict[str, Tensor]


@dataclass
class BlockNeRFDatasetConfig:
    """Configuration for Block-NeRF dataset."""

    # Data paths
    data_dir: str = "./data"
    images_dir: str = "images"
    poses_file: str = "poses.json"
    intrinsics_file: str = "intrinsics.json"

    # Dataset type
    dataset_type: str = "waymo"  # "waymo", "blender", "colmap", "llff"

    # Image settings
    image_width: int = 800
    image_height: int = 600
    downscale_factor: int = 1
    white_background: bool = True

    # Ray sampling
    num_rays: int = 1024
    precrop: bool = False
    precrop_frac: float = 0.5
    precrop_iters: int = 500

    # Appearance modeling
    use_appearance_ids: bool = True
    max_appearance_ids: int = 1000

    # Exposure modeling
    use_exposure: bool = True
    exposure_range: tuple[float, float] = (0.5, 2.0)

    # Data splits
    train_split: float = 0.8
    val_split: float = 0.1
    test_split: float = 0.1

    # Augmentation
    use_augmentation: bool = False
    color_jitter: bool = False
    rotation_augment: bool = False


class BlockNeRFDataset(Dataset):
    """Block-NeRF dataset for training and evaluation."""

    def __init__(
        self,
        config: BlockNeRFDatasetConfig,
        split: str = "train",
        transform=None,
    ):
        self.config = config
        self.split = split
        self.transform = transform

        # Load dataset
        self.load_dataset()

        # Setup ray sampling
        self.setup_ray_sampling()

    def load_dataset(self) -> None:
        """Load dataset based on type."""
        if self.config.dataset_type == "waymo":
            self.load_waymo_dataset()
        elif self.config.dataset_type == "blender":
            self.load_blender_dataset()
        elif self.config.dataset_type == "colmap":
            self.load_colmap_dataset()
        elif self.config.dataset_type == "llff":
            self.load_llff_dataset()
        else:
            raise ValueError(f"Unknown dataset type: {self.config.dataset_type}")

    def load_waymo_dataset(self) -> None:
        """Load Waymo Open Dataset format."""
        data_dir = Path(self.config.data_dir)

        # Check if data directory exists
        if not data_dir.exists():
            print(f"Warning: Data directory {data_dir} does not exist. Creating empty dataset.")
            self.images = []
            self.poses = []
            self.intrinsics = []
            self.appearance_ids = []
            self.exposure_values = []
            return

        # Load poses
        poses_file = data_dir / self.config.poses_file
        if not poses_file.exists():
            print(f"Warning: Poses file {poses_file} does not exist. Creating empty dataset.")
            self.images = []
            self.poses = []
            self.intrinsics = []
            self.appearance_ids = []
            self.exposure_values = []
            return

        with open(poses_file, "r") as f:
            poses_data = json.load(f)

        # Load intrinsics
        intrinsics_file = data_dir / self.config.intrinsics_file
        with open(intrinsics_file, "r") as f:
            intrinsics_data = json.load(f)

        # Process images and poses
        self.images = []
        self.poses = []
        self.intrinsics = []
        self.appearance_ids = []
        self.exposure_values = []

        images_dir = data_dir / self.config.images_dir

        for frame_data in poses_data["frames"]:
            # Load image
            image_path = images_dir / frame_data["file_path"]
            if image_path.exists():
                image = self.load_image(str(image_path))
                self.images.append(image)

                # Load pose
                pose = np.array(frame_data["transform_matrix"])
                self.poses.append(torch.from_numpy(pose).float())

                # Load intrinsics
                intrinsic = np.array(intrinsics_data["camera_matrix"])
                self.intrinsics.append(torch.from_numpy(intrinsic).float())

                # Appearance ID (could be based on lighting conditions, time of day, etc.)
                appearance_id = frame_data.get("appearance_id", 0)
                self.appearance_ids.append(appearance_id)

                # Exposure value
                exposure = frame_data.get("exposure", 1.0)
                self.exposure_values.append(exposure)

        # Convert to tensors
        self.poses = torch.stack(self.poses)
        self.intrinsics = torch.stack(self.intrinsics)
        self.appearance_ids = torch.tensor(self.appearance_ids, dtype=torch.long)
        self.exposure_values = torch.tensor(self.exposure_values, dtype=torch.float32)

        # Split dataset
        self.split_dataset()

    def load_blender_dataset(self) -> None:
        """Load Blender synthetic dataset format."""
        data_dir = Path(self.config.data_dir)

        # Load transforms file for this split
        transforms_file = data_dir / f"transforms_{self.split}.json"
        with open(transforms_file, "r") as f:
            transforms = json.load(f)

        # Camera settings
        camera_angle_x = transforms["camera_angle_x"]
        focal = 0.5 * self.config.image_width / np.tan(0.5 * camera_angle_x)

        # Create intrinsics matrix
        intrinsic = np.array(
            [
                [focal, 0, self.config.image_width / 2],
                [0, focal, self.config.image_height / 2],
                [0, 0, 1],
            ]
        )

        # Process frames
        self.images = []
        self.poses = []
        self.intrinsics = []
        self.appearance_ids = []
        self.exposure_values = []

        for frame in transforms["frames"]:
            # Load image
            image_path = data_dir / f"{frame['file_path']}.png"
            if image_path.exists():
                image = self.load_image(str(image_path))
                self.images.append(image)

                # Load pose (Blender to OpenGL coordinate system)
                pose = np.array(frame["transform_matrix"])
                pose = self.blender_to_opencv_pose(pose)
                self.poses.append(torch.from_numpy(pose).float())

                # Same intrinsics for all frames
                self.intrinsics.append(torch.from_numpy(intrinsic).float())

                # Default appearance and exposure
                self.appearance_ids.append(0)
                self.exposure_values.append(1.0)

        # Convert to tensors
        self.poses = torch.stack(self.poses)
        self.intrinsics = torch.stack(self.intrinsics)
        self.appearance_ids = torch.tensor(self.appearance_ids, dtype=torch.long)
        self.exposure_values = torch.tensor(self.exposure_values, dtype=torch.float32)

    def load_colmap_dataset(self) -> None:
        """Load COLMAP dataset format."""
        # Implementation for COLMAP format
        raise NotImplementedError("COLMAP dataset loading not implemented yet")

    def load_llff_dataset(self) -> None:
        """Load LLFF dataset format."""
        # Implementation for LLFF format
        raise NotImplementedError("LLFF dataset loading not implemented yet")

    def load_image(self, image_path: str) -> Tensor:
        """Load and preprocess image."""
        image = Image.open(image_path).convert("RGB")

        # Resize if needed
        if self.config.downscale_factor > 1:
            new_width = self.config.image_width // self.config.downscale_factor
            new_height = self.config.image_height // self.config.downscale_factor
            image = image.resize((new_width, new_height), Image.LANCZOS)

        # Convert to tensor
        image = torch.from_numpy(np.array(image)).float() / 255.0

        # Handle alpha channel for white background
        if image.shape[-1] == 4 and self.config.white_background:
            rgb = image[..., :3]
            alpha = image[..., 3:4]
            image = rgb * alpha + (1.0 - alpha)

        return image

    def blender_to_opencv_pose(self, pose: np.ndarray) -> np.ndarray:
        """Convert Blender pose to OpenCV convention."""
        # Blender: Y up, -Z forward
        # OpenCV: Y down, Z forward
        pose = pose.copy()
        pose[1:3] *= -1  # Flip Y and Z axes
        return pose

    def split_dataset(self) -> None:
        """Split dataset into train/val/test."""
        total_frames = len(self.images)

        # Calculate split indices
        train_end = int(total_frames * self.config.train_split)
        val_end = train_end + int(total_frames * self.config.val_split)

        if self.split == "train":
            indices = list(range(train_end))
        elif self.split == "val":
            indices = list(range(train_end, val_end))
        elif self.split == "test":
            indices = list(range(val_end, total_frames))
        else:
            indices = list(range(total_frames))

        # Filter data based on split
        self.images = [self.images[i] for i in indices]
        self.poses = self.poses[indices]
        self.intrinsics = self.intrinsics[indices]
        self.appearance_ids = self.appearance_ids[indices]
        self.exposure_values = self.exposure_values[indices]

    def setup_ray_sampling(self) -> None:
        """Setup ray sampling for training."""
        if not self.images:
            return

        # Get image dimensions
        sample_image = self.images[0]
        self.image_height, self.image_width = sample_image.shape[:2]

        # Create pixel coordinates
        i, j = torch.meshgrid(
            torch.linspace(0, self.image_width - 1, self.image_width),
            torch.linspace(0, self.image_height - 1, self.image_height),
            indexing="ij",
        )
        self.i = i.t()  # Transpose for correct orientation
        self.j = j.t()

        # Precompute ray directions for efficiency
        self.precompute_rays()

    def precompute_rays(self) -> None:
        """Precompute ray directions for all cameras."""
        self.ray_directions_list = []

        for intrinsic in self.intrinsics:
            # Convert pixel coordinates to ray directions
            dirs = torch.stack(
                [
                    (self.i - intrinsic[0, 2]) / intrinsic[0, 0],
                    -(self.j - intrinsic[1, 2]) / intrinsic[1, 1],
                    -torch.ones_like(self.i),
                ],
                dim=-1,
            )

            self.ray_directions_list.append(dirs)

    def get_rays(self, pose: Tensor, intrinsic: Tensor) -> tuple[Tensor, Tensor]:
        """Get ray origins and directions for a camera."""
        # Ray directions in camera coordinates
        dirs = torch.stack(
            [
                (self.i - intrinsic[0, 2]) / intrinsic[0, 0],
                -(self.j - intrinsic[1, 2]) / intrinsic[1, 1],
                -torch.ones_like(self.i),
            ],
            dim=-1,
        )

        # Transform to world coordinates
        ray_directions = torch.sum(dirs[..., None, :] * pose[:3, :3], dim=-1)

        # Normalize directions
        ray_directions = ray_directions / torch.norm(ray_directions, dim=-1, keepdim=True)

        # Ray origins are camera position
        ray_origins = pose[:3, 3].expand(ray_directions.shape)

        return ray_origins, ray_directions

    def sample_rays(self, image_idx: int, num_rays: int | None = None) -> TensorDict:
        """Sample rays from an image."""
        if num_rays is None:
            num_rays = self.config.num_rays

        image = self.images[image_idx]
        pose = self.poses[image_idx]
        intrinsic = self.intrinsics[image_idx]
        appearance_id = self.appearance_ids[image_idx]
        exposure_value = self.exposure_values[image_idx]

        # Get all rays for this image
        ray_origins, ray_directions = self.get_rays(pose, intrinsic)

        # Flatten image and rays
        image_flat = image.reshape(-1, 3)
        ray_origins_flat = ray_origins.reshape(-1, 3)
        ray_directions_flat = ray_directions.reshape(-1, 3)

        # Sample random rays
        total_pixels = image_flat.shape[0]

        if self.config.precrop and self.split == "train":
            # Crop center for initial training
            crop_size = int(self.config.precrop_frac * min(self.image_height, self.image_width))
            center_h = self.image_height // 2
            center_w = self.image_width // 2

            h_start = max(0, center_h - crop_size // 2)
            h_end = min(self.image_height, center_h + crop_size // 2)
            w_start = max(0, center_w - crop_size // 2)
            w_end = min(self.image_width, center_w + crop_size // 2)

            # Create mask for cropped region
            mask = torch.zeros(self.image_height, self.image_width, dtype=torch.bool)
            mask[h_start:h_end, w_start:w_end] = True
            mask_flat = mask.reshape(-1)

            # Sample from cropped region
            valid_indices = torch.where(mask_flat)[0]
            if len(valid_indices) >= num_rays:
                selected_indices = valid_indices[torch.randperm(len(valid_indices))[:num_rays]]
            else:
                selected_indices = torch.randperm(total_pixels)[:num_rays]
        else:
            # Sample from entire image
            selected_indices = torch.randperm(total_pixels)[:num_rays]

        # Extract sampled data
        rgb = image_flat[selected_indices]
        ray_origins_sampled = ray_origins_flat[selected_indices]
        ray_directions_sampled = ray_directions_flat[selected_indices]

        # Create appearance and exposure tensors
        appearance_ids = torch.full((num_rays,), appearance_id, dtype=torch.long)
        exposure_values_tensor = torch.full((num_rays, 1), exposure_value, dtype=torch.float32)

        return {
            "rgb": rgb,
            "ray_origins": ray_origins_sampled,
            "ray_directions": ray_directions_sampled,
            "appearance_ids": appearance_ids,
            "exposure_values": exposure_values_tensor,
            "camera_positions": pose[:3, 3].unsqueeze(0).expand(num_rays, -1),
            "image_idx": torch.full((num_rays,), image_idx, dtype=torch.long),
        }

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, idx: int) -> TensorDict:
        """Get a batch of rays from the dataset."""
        return self.sample_rays(idx)


def create_block_nerf_dataloader(
    config: BlockNeRFDatasetConfig,
    split: str = "train",
    batch_size: int = 1,
    num_workers: int = 4,
    shuffle: bool = None,
) -> DataLoader:
    """Create a Block-NeRF dataloader."""
    if shuffle is None:
        shuffle = split == "train"

    dataset = BlockNeRFDataset(config, split)

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    return dataloader


def create_block_nerf_dataset(
    config: BlockNeRFDatasetConfig,
    split: str = "train",
) -> BlockNeRFDataset:
    """Create a Block-NeRF dataset."""
    return BlockNeRFDataset(config, split)
