"""
Mega-NeRF Dataset Module

This module contains the data loading and preprocessing pipeline for Mega-NeRF including:
- Dataset configuration
- Main dataset class
- Camera information handling
- Data partitioning utilities
"""

from __future__ import annotations

import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from dataclasses import dataclass
from pathlib import Path
import logging
import json
import os
import pickle
import h5py

logger = logging.getLogger(__name__)

# 尝试导入可选依赖
try:
    import cv2

    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False
    logger.warning("cv2 not available for image processing")

try:
    from PIL import Image

    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False
    logger.warning("PIL not available for image loading")


@dataclass
class MegaNeRFDatasetConfig:
    """Mega-NeRF 数据集配置"""

    # 数据路径设置
    data_root: str = "data/mega_nerf"
    split: str = "train"  # "train", "val", "test"

    # 图像处理设置
    image_scale: float = 1.0
    load_images: bool = True
    use_cache: bool = True
    cache_dir: str | None = None

    # 光线生成设置
    ray_batch_size: int = 1024
    num_rays_per_image: int = 1024
    use_random_rays: bool = True

    # 数据增强设置
    use_data_augmentation: bool = False
    color_jitter: float = 0.1
    random_crop: bool = False

    # 分区设置
    num_partitions: int = 8
    partition_overlap: float = 0.1

    def __post_init__(self):
        """后初始化验证"""
        if self.image_scale <= 0:
            raise ValueError("image_scale must be positive")

        if self.ray_batch_size <= 0:
            raise ValueError("ray_batch_size must be positive")

        if self.split not in ["train", "val", "test"]:
            raise ValueError("split must be 'train', 'val', or 'test'")

        if self.partition_overlap < 0 or self.partition_overlap > 1:
            raise ValueError("partition_overlap must be between 0 and 1")


@dataclass
class CameraInfo:
    """相机信息结构"""

    transform_matrix: np.ndarray  # 4x4 相机到世界变换矩阵
    intrinsics: np.ndarray  # 3x3 内参矩阵
    image_path: str  # 图像文件路径
    image_id: int  # 唯一图像标识符
    width: int  # 图像宽度
    height: int  # 图像高度

    def get_position(self) -> np.ndarray:
        """获取相机在世界坐标系中的位置"""
        return self.transform_matrix[:3, 3]

    def get_rotation(self) -> np.ndarray:
        """获取相机旋转矩阵"""
        return self.transform_matrix[:3, :3]

    def get_focal_length(self) -> tuple[float, float]:
        """获取焦距 (fx, fy)"""
        return self.intrinsics[0, 0], self.intrinsics[1, 1]

    def get_principal_point(self) -> tuple[float, float]:
        """获取主点 (cx, cy)"""
        return self.intrinsics[0, 2], self.intrinsics[1, 2]


class CameraDataset:
    """管理相机信息和图像的数据集"""

    def __init__(
        self,
        data_root: str,
        split: str = "train",
        image_scale: float = 1.0,
        load_images: bool = True,
    ):
        """
        初始化相机数据集

        Args:
            data_root: 包含数据集的根目录
            split: 数据分割 ("train", "val", "test")
            image_scale: 图像缩放因子
            load_images: 是否加载图像数据
        """
        self.data_root = Path(data_root)
        self.split = split
        self.image_scale = image_scale
        self.load_images = load_images

        # 加载相机数据
        self.cameras = self._load_cameras()

        # 如果请求则加载图像
        self.images = {}
        if load_images:
            self._load_images()

        logger.info(f"Loaded {len(self.cameras)} cameras for {split} split")

    def _load_cameras(self) -> list[CameraInfo]:
        """从各种格式加载相机信息"""
        # 尝试不同的数据格式
        if (self.data_root / "transforms.json").exists():
            return self._load_nerf_format()
        elif (self.data_root / "sparse").exists():
            return self._load_colmap_format()
        elif (self.data_root / "poses_bounds.npy").exists():
            return self._load_llff_format()
        else:
            # 为演示创建合成数据
            return self._create_synthetic_cameras()

    def _load_nerf_format(self) -> list[CameraInfo]:
        """加载 NeRF 风格的 transforms.json 格式"""
        transforms_file = self.data_root / "transforms.json"

        with open(transforms_file, "r") as f:
            transforms = json.load(f)

        cameras = []

        # 获取相机内参
        if "fl_x" in transforms and "fl_y" in transforms:
            fx, fy = transforms["fl_x"], transforms["fl_y"]
        elif "camera_angle_x" in transforms:
            # 从视场角计算焦距
            w = transforms.get("w", 800)
            fx = fy = w / (2 * np.tan(transforms["camera_angle_x"] / 2))
        else:
            fx = fy = 800  # 默认焦距

        cx = transforms.get("cx", transforms.get("w", 800) / 2)
        cy = transforms.get("cy", transforms.get("h", 600) / 2)

        intrinsics = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])

        # 缩放内参
        intrinsics[:2] *= self.image_scale

        # 加载帧信息
        for i, frame in enumerate(transforms["frames"]):
            # 如果不在正确的分割中则跳过
            if self.split in frame.get("split", self.split):
                continue

            # 变换矩阵
            transform_matrix = np.array(frame["transform_matrix"])

            # 图像路径
            image_path = self.data_root / frame["file_path"]
            if not image_path.suffix:
                # 尝试常见扩展名
                for ext in [".png", ".jpg", ".jpeg"]:
                    if (self.data_root / (frame["file_path"] + ext)).exists():
                        image_path = self.data_root / (frame["file_path"] + ext)
                        break

            # 获取图像尺寸
            if image_path.exists() and PIL_AVAILABLE:
                with Image.open(image_path) as img:
                    width, height = img.size
                    width = int(width * self.image_scale)
                    height = int(height * self.image_scale)
            else:
                width, height = 800, 600

            camera = CameraInfo(
                transform_matrix=transform_matrix,
                intrinsics=intrinsics,
                image_path=str(image_path),
                image_id=i,
                width=width,
                height=height,
            )

            cameras.append(camera)

        return cameras

    def _load_colmap_format(self) -> list[CameraInfo]:
        """加载 COLMAP 格式数据"""
        try:
            from .utils.colmap_utils import read_cameras_binary, read_images_binary
        except ImportError:
            logger.error("COLMAP utils not available")
            return self._create_synthetic_cameras()

        sparse_dir = self.data_root / "sparse" / "0"
        images_dir = self.data_root / "images"

        # 读取 COLMAP 数据
        cameras_data = read_cameras_binary(sparse_dir / "cameras.bin")
        images_data = read_images_binary(sparse_dir / "images.bin")

        # 获取分割文件
        split_file = self.data_root / f"{self.split}.txt"
        if split_file.exists():
            with open(split_file, "r") as f:
                split_names = set(line.strip() for line in f.readlines())
        else:
            split_names = None

        cameras = []

        for img_id, img_data in images_data.items():
            # 检查分割
            if split_names and img_data.name not in split_names:
                continue

            # 获取相机参数
            camera_data = cameras_data[img_data.camera_id]

            # 构建内参矩阵
            if camera_data.model == "SIMPLE_PINHOLE":
                fx = camera_data.params[0]
                cx, cy = camera_data.params[1], camera_data.params[2]
                fy = fx
            elif camera_data.model == "PINHOLE":
                fx, fy = camera_data.params[0], camera_data.params[1]
                cx, cy = camera_data.params[2], camera_data.params[3]
            else:
                logger.warning(f"Unsupported camera model: {camera_data.model}")
                continue

            intrinsics = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])

            # 缩放内参
            intrinsics[:2] *= self.image_scale

            # 构建变换矩阵
            transform_matrix = np.eye(4)
            transform_matrix[:3, :3] = img_data.qvec2rotmat()
            transform_matrix[:3, 3] = img_data.tvec

            # 图像路径
            image_path = images_dir / img_data.name

            camera = CameraInfo(
                transform_matrix=transform_matrix,
                intrinsics=intrinsics,
                image_path=str(image_path),
                image_id=img_id,
                width=int(camera_data.width * self.image_scale),
                height=int(camera_data.height * self.image_scale),
            )

            cameras.append(camera)

        return cameras

    def _load_llff_format(self) -> list[CameraInfo]:
        """加载 LLFF 格式数据"""
        poses_bounds_file = self.data_root / "poses_bounds.npy"
        poses_bounds = np.load(poses_bounds_file)

        # 解析姿态和边界
        poses = poses_bounds[:, :15].reshape(-1, 3, 5)
        bounds = poses_bounds[:, 15:]

        cameras = []

        # 获取图像文件列表
        image_files = sorted(list((self.data_root / "images").glob("*.png")))
        if not image_files:
            image_files = sorted(list((self.data_root / "images").glob("*.jpg")))

        for i, (pose, image_file) in enumerate(zip(poses, image_files)):
            # 构建变换矩阵
            transform_matrix = np.eye(4)
            transform_matrix[:3, :4] = pose

            # 获取内参
            h, w = pose[:2, 4]
            fx = pose[0, 4]
            fy = pose[1, 4]
            cx = w / 2
            cy = h / 2

            intrinsics = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])

            # 缩放内参
            intrinsics[:2] *= self.image_scale

            camera = CameraInfo(
                transform_matrix=transform_matrix,
                intrinsics=intrinsics,
                image_path=str(image_file),
                image_id=i,
                width=int(w * self.image_scale),
                height=int(h * self.image_scale),
            )

            cameras.append(camera)

        return cameras

    def _create_synthetic_cameras(self) -> list[CameraInfo]:
        """创建合成相机数据用于演示"""
        logger.info("Creating synthetic camera data for demonstration")

        cameras = []
        num_cameras = 100

        # 创建圆形轨迹的相机
        radius = 5.0
        height = 2.0

        for i in range(num_cameras):
            # 相机位置
            angle = 2 * np.pi * i / num_cameras
            x = radius * np.cos(angle)
            y = radius * np.sin(angle)
            z = height

            # 相机朝向中心
            center = np.array([0, 0, 0])
            position = np.array([x, y, z])

            # 构建变换矩阵
            forward = center - position
            forward = forward / np.linalg.norm(forward)

            right = np.cross(forward, np.array([0, 0, 1]))
            right = right / np.linalg.norm(right)

            up = np.cross(right, forward)

            transform_matrix = np.eye(4)
            transform_matrix[:3, 0] = right
            transform_matrix[:3, 1] = up
            transform_matrix[:3, 2] = forward
            transform_matrix[:3, 3] = position

            # 内参矩阵
            intrinsics = np.array([[800, 0, 400], [0, 800, 300], [0, 0, 1]])

            camera = CameraInfo(
                transform_matrix=transform_matrix,
                intrinsics=intrinsics,
                image_path=f"synthetic_{i:04d}.png",
                image_id=i,
                width=800,
                height=600,
            )

            cameras.append(camera)

        return cameras

    def _load_images(self) -> None:
        """加载图像数据"""
        if not PIL_AVAILABLE:
            logger.warning("PIL not available, skipping image loading")
            return

        for camera in self.cameras:
            image_path = Path(camera.image_path)
            if image_path.exists():
                try:
                    with Image.open(image_path) as img:
                        # 转换为 RGB
                        if img.mode != "RGB":
                            img = img.convert("RGB")

                        # 调整大小
                        if self.image_scale != 1.0:
                            new_width = int(img.width * self.image_scale)
                            new_height = int(img.height * self.image_scale)
                            img = img.resize((new_width, new_height), Image.LANCZOS)

                        # 转换为 numpy 数组
                        image_array = np.array(img) / 255.0
                        self.images[camera.image_id] = image_array

                except Exception as e:
                    logger.warning(f"Failed to load image {image_path}: {e}")

    def get_camera_positions(self) -> np.ndarray:
        """获取所有相机位置"""
        return np.array([camera.get_position() for camera in self.cameras])

    def get_scene_bounds(self) -> tuple[np.ndarray, np.ndarray]:
        """获取场景边界"""
        positions = self.get_camera_positions()
        min_bounds = np.min(positions, axis=0)
        max_bounds = np.max(positions, axis=0)
        return min_bounds, max_bounds

    def get_camera(self, camera_id: int) -> CameraInfo:
        """获取指定相机"""
        return self.cameras[camera_id]

    def get_image(self, camera_id: int) -> np.ndarray | None:
        """获取指定图像"""
        return self.images.get(camera_id)


class MegaNeRFDataset(Dataset):
    """Mega-NeRF 主数据集类"""

    def __init__(
        self,
        data_root: str,
        config: MegaNeRFDatasetConfig,
        partitioner: object | None = None,
    ):
        """
        初始化数据集

        Args:
            data_root: 数据根目录
            config: 数据集配置
            partitioner: 空间分区器
        """
        self.config = config
        self.data_root = Path(data_root)
        self.partitioner = partitioner

        # 加载相机数据集
        self.camera_dataset = CameraDataset(
            data_root=data_root,
            split=config.split,
            image_scale=config.image_scale,
            load_images=config.load_images,
        )

        # 创建数据分区
        self.partitions = self._create_data_partitions()

        # 预计算光线
        if config.use_cache:
            self._precompute_rays()

        logger.info(f"MegaNeRFDataset initialized with {len(self.partitions)} partitions")

    def _create_data_partitions(self) -> list[dict[str, object]]:
        """创建数据分区"""
        partitions = []

        # 获取相机位置
        camera_positions = self.camera_dataset.get_camera_positions()

        # 如果没有分区器，创建简单的网格分区
        if self.partitioner is None:
            # 简单的网格分区
            num_partitions = self.config.num_partitions
            grid_size = int(np.sqrt(num_partitions))

            # 计算场景边界
            min_bounds, max_bounds = self.camera_dataset.get_scene_bounds()

            for i in range(num_partitions):
                row = i // grid_size
                col = i % grid_size

                # 计算分区边界
                x_step = (max_bounds[0] - min_bounds[0]) / grid_size
                y_step = (max_bounds[1] - min_bounds[1]) / grid_size

                partition_bounds = [
                    min_bounds[0] + col * x_step,
                    min_bounds[1] + row * y_step,
                    min_bounds[2],
                    min_bounds[0] + (col + 1) * x_step,
                    min_bounds[1] + (row + 1) * y_step,
                    max_bounds[2],
                ]

                # 找到属于此分区的相机
                partition_cameras = []
                for j, camera in enumerate(self.camera_dataset.cameras):
                    position = camera.get_position()
                    if (
                        partition_bounds[0] <= position[0] <= partition_bounds[3]
                        and partition_bounds[1] <= position[1] <= partition_bounds[4]
                        and partition_bounds[2] <= position[2] <= partition_bounds[5]
                    ):
                        partition_cameras.append(j)

                partitions.append(
                    {
                        "partition_id": i,
                        "bounds": partition_bounds,
                        "camera_indices": partition_cameras,
                        "rays": None,  # 将在预计算时填充
                    }
                )
        else:
            # 使用提供的分区器
            partition_assignments = self.partitioner.assign_cameras(camera_positions)

            for partition_id, camera_indices in enumerate(partition_assignments):
                if camera_indices:
                    # 计算分区边界
                    partition_positions = camera_positions[camera_indices]
                    min_bounds = np.min(partition_positions, axis=0)
                    max_bounds = np.max(partition_positions, axis=0)

                    partitions.append(
                        {
                            "partition_id": partition_id,
                            "bounds": [*min_bounds, *max_bounds],
                            "camera_indices": camera_indices,
                            "rays": None,
                        }
                    )

        return partitions

    def _precompute_rays(self) -> None:
        """预计算光线"""
        logger.info("Precomputing rays for all partitions")

        for partition in self.partitions:
            partition["rays"] = self._compute_partition_rays(partition)

    def _compute_partition_rays(self, partition: dict[str, object]) -> dict[str, np.ndarray]:
        """计算分区的光线"""
        camera_indices = partition["camera_indices"]

        if not camera_indices:
            return {}

        all_rays_o = []
        all_rays_d = []
        all_colors = []
        all_appearance_ids = []

        for camera_idx in camera_indices:
            camera = self.camera_dataset.cameras[camera_idx]

            # 生成相机光线
            rays_o, rays_d = self._generate_camera_rays(camera)

            # 随机采样光线
            if self.config.use_random_rays:
                num_rays = min(self.config.num_rays_per_image, len(rays_o))
                indices = np.random.choice(len(rays_o), num_rays, replace=False)
                rays_o = rays_o[indices]
                rays_d = rays_d[indices]

            all_rays_o.append(rays_o)
            all_rays_d.append(rays_d)

            # 获取颜色
            if camera.image_id in self.camera_dataset.images:
                image = self.camera_dataset.images[camera.image_id]
                colors = image.reshape(-1, 3)[: len(rays_o)]
                all_colors.append(colors)
            else:
                # 使用默认颜色
                colors = np.ones((len(rays_o), 3)) * 0.5
                all_colors.append(colors)

            # 外观 ID
            appearance_ids = np.full(len(rays_o), camera_idx, dtype=np.int32)
            all_appearance_ids.append(appearance_ids)

        return {
            "ray_origins": np.concatenate(all_rays_o, axis=0),
            "ray_directions": np.concatenate(all_rays_d, axis=0),
            "colors": np.concatenate(all_colors, axis=0),
            "appearance_ids": np.concatenate(all_appearance_ids, axis=0),
        }

    def _generate_camera_rays(self, camera: CameraInfo) -> tuple[np.ndarray, np.ndarray]:
        """为相机生成光线"""
        # 创建像素坐标网格
        i, j = np.meshgrid(np.arange(camera.height), np.arange(camera.width), indexing="ij")

        # 转换为相机坐标
        directions = np.stack(
            [
                (j - camera.intrinsics[0, 2]) / camera.intrinsics[0, 0],
                (i - camera.intrinsics[1, 2]) / camera.intrinsics[1, 1],
                np.ones_like(i),
            ],
            axis=-1,
        )

        # 转换到世界坐标
        directions = directions @ camera.get_rotation().T
        origins = camera.get_position()

        # 扁平化
        rays_o = origins.reshape(-1, 3)
        rays_d = directions.reshape(-1, 3)

        # 归一化方向
        rays_d = rays_d / np.linalg.norm(rays_d, axis=-1, keepdims=True)

        return rays_o, rays_d

    def get_partition_data(self, partition_idx: int) -> dict[str, object]:
        """获取分区数据"""
        if partition_idx >= len(self.partitions):
            raise ValueError(f"Invalid partition index: {partition_idx}")

        partition = self.partitions[partition_idx]

        # 如果没有预计算的光线，现在计算
        if partition["rays"] is None:
            partition["rays"] = self._compute_partition_rays(partition)

        return partition["rays"]

    def __len__(self) -> int:
        """数据集长度"""
        return len(self.partitions)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        """获取数据项"""
        partition_data = self.get_partition_data(idx)

        # 随机采样批次
        num_rays = len(partition_data["ray_origins"])
        batch_size = min(self.config.ray_batch_size, num_rays)

        if batch_size < num_rays:
            indices = np.random.choice(num_rays, batch_size, replace=False)
        else:
            indices = np.arange(num_rays)

        return {
            "ray_origins": torch.tensor(
                partition_data["ray_origins"][indices], dtype=torch.float32
            ),
            "ray_directions": torch.tensor(
                partition_data["ray_directions"][indices], dtype=torch.float32
            ),
            "colors": torch.tensor(partition_data["colors"][indices], dtype=torch.float32),
            "appearance_ids": torch.tensor(
                partition_data["appearance_ids"][indices], dtype=torch.long
            ),
        }

    def create_dataloader(
        self,
        batch_size: int = 1,
        shuffle: bool = True,
        num_workers: int = 4,
    ) -> DataLoader:
        """创建数据加载器"""
        return DataLoader(
            self, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, pin_memory=True
        )

    def get_validation_data(self, partition_idx: int) -> list[dict[str, torch.Tensor]]:
        """获取验证数据"""
        partition_data = self.get_partition_data(partition_idx)

        # 创建验证批次
        validation_batches = []
        num_rays = len(partition_data["ray_origins"])

        for i in range(0, num_rays, self.config.ray_batch_size):
            end_idx = min(i + self.config.ray_batch_size, num_rays)
            indices = np.arange(i, end_idx)

            batch = {
                "ray_origins": torch.tensor(
                    partition_data["ray_origins"][indices], dtype=torch.float32
                ),
                "ray_directions": torch.tensor(
                    partition_data["ray_directions"][indices], dtype=torch.float32
                ),
                "colors": torch.tensor(partition_data["colors"][indices], dtype=torch.float32),
                "appearance_ids": torch.tensor(
                    partition_data["appearance_ids"][indices], dtype=torch.long
                ),
            }
            validation_batches.append(batch)

        return validation_batches
