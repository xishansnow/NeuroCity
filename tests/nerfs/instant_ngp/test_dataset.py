"""
Test Instant NGP Dataset Components

This module tests the dataset-related components of Instant NGP:
- InstantNGPDataset
- InstantNGPDatasetConfig
"""

import pytest
import torch
import numpy as np
import tempfile
import os
import json

# Add the src directory to the path for imports
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "..", "src"))

try:
    from nerfs.instant_ngp import InstantNGP
    from nerfs.instant_ngp.dataset import InstantNGPDataset, InstantNGPDatasetConfig

    INSTANT_NGP_AVAILABLE = True
except ImportError as e:
    INSTANT_NGP_AVAILABLE = False
    IMPORT_ERROR = str(e)

from __future__ import annotations

from pathlib import Path
from PIL import Image
from torch.utils.data import DataLoader

from dataclasses import asdict

from nerfs.instant_ngp.dataset import (
    create_instant_ngp_dataloader,
)


class TestInstantNGPDataset:
    """Instant NGP 数据集测试类"""

    def setup_method(self):
        """每个测试方法前的设置"""
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # 创建临时数据目录
        self.temp_dir = tempfile.mkdtemp()
        self.data_dir = Path(self.temp_dir)

        # 创建虚拟数据集配置
        self.dataset_config = InstantNGPDatasetConfig(
            data_dir=str(self.data_dir),
            image_width=256,
            image_height=256,
            near_plane=0.1,
            far_plane=10.0,
            scene_scale=1.0,
        )

    def create_dummy_transforms_json(self) -> dict[str, Any]:
        """创建虚拟的 transforms.json 文件"""
        # 创建虚拟相机参数
        transforms_data: dict[str, Any] = {
            "camera_angle_x": 0.6911112070083618,
            "fl_x": 400.0,
            "fl_y": 400.0,
            "cx": 128.0,
            "cy": 128.0,
            "w": 256,
            "h": 256,
            "aabb_scale": 1,
            "frames": [],
        }

        # 添加虚拟帧数据
        num_frames = 10
        for i in range(num_frames):
            # 创建简单的相机轨迹
            angle = 2 * np.pi * i / num_frames
            radius = 3.0

            # 创建变换矩阵
            transform_matrix: list[list[float]] = [
                [np.cos(angle), 0, np.sin(angle), radius * np.sin(angle)],
                [0, 1, 0, 0],
                [-np.sin(angle), 0, np.cos(angle), radius * np.cos(angle)],
                [0, 0, 0, 1],
            ]

            frame_data: dict[str, Any] = {
                "file_path": f"./images/frame_{i:04d}.png",
                "transform_matrix": transform_matrix,
            }

            transforms_data["frames"].append(frame_data)

        return transforms_data

    def create_dummy_images(self, transforms_data: dict[str, Any]):
        """创建虚拟图像文件"""
        # 创建图像目录
        images_dir = self.data_dir / "images"
        images_dir.mkdir(exist_ok=True)

        # 生成虚拟图像
        for frame in transforms_data["frames"]:
            file_path = frame["file_path"]
            # 移除 "./" 前缀
            if file_path.startswith("./"):
                file_path = file_path[2:]

            image_path = self.data_dir / file_path

            # 创建随机彩色图像
            image_array = np.random.randint(
                0,
                256,
                (self.dataset_config.image_height, self.dataset_config.image_width, 3),
                dtype=np.uint8,
            )

            image = Image.fromarray(image_array)
            image.save(image_path)

    def test_dataset_config_initialization(self):
        """测试数据集配置初始化"""
        config = InstantNGPDatasetConfig()

        # 检查默认值
        assert config.image_width == 512
        assert config.image_height == 512
        assert config.near_plane == 0.1
        assert config.far_plane == 1000.0
        assert config.batch_size == 8192

        # 测试 Python 3.10 兼容性 - 使用内置容器
        config_dict: dict[str, Any] = asdict(config)
        assert isinstance(config_dict, dict)
        assert "image_width" in config_dict
        assert "batch_size" in config_dict

    def test_dataset_initialization_with_dummy_data(self):
        """测试数据集初始化（使用虚拟数据）"""
        # 创建虚拟数据
        transforms_data = self.create_dummy_transforms_json()
        self.create_dummy_images(transforms_data)

        # 保存 transforms.json
        transforms_file = self.data_dir / "transforms.json"
        with open(transforms_file, "w") as f:
            json.dump(transforms_data, f, indent=2)

        # 初始化数据集
        dataset = InstantNGPDataset(self.dataset_config)

        # 检查数据集属性
        assert dataset.config is self.dataset_config
        assert len(dataset.images) == len(transforms_data["frames"])
        assert len(dataset.poses) == len(transforms_data["frames"])

        # 检查图像数据类型 - 使用 Python 3.10 兼容的类型
        images: list[torch.Tensor] = dataset.images
        poses: list[torch.Tensor] = dataset.poses

        assert isinstance(images, list)
        assert isinstance(poses, list)
        assert all(isinstance(img, torch.Tensor) for img in images)
        assert all(isinstance(pose, torch.Tensor) for pose in poses)

    def test_dataset_getitem(self):
        """测试数据集项目获取"""
        # 创建虚拟数据
        transforms_data = self.create_dummy_transforms_json()
        self.create_dummy_images(transforms_data)

        transforms_file = self.data_dir / "transforms.json"
        with open(transforms_file, "w") as f:
            json.dump(transforms_data, f, indent=2)

        dataset = InstantNGPDataset(self.dataset_config)

        # 获取数据项
        item: dict[str, torch.Tensor] = dataset[0]

        # 检查数据项结构
        assert isinstance(item, dict)
        expected_keys = ["ray_origins", "ray_directions", "colors"]
        for key in expected_keys:
            assert key in item

        # 检查数据形状
        batch_size = self.dataset_config.batch_size
        assert item["ray_origins"].shape == (batch_size, 3)
        assert item["ray_directions"].shape == (batch_size, 3)
        assert item["colors"].shape == (batch_size, 3)

        # 检查数据类型和值范围
        assert item["colors"].dtype == torch.float32
        assert torch.all(item["colors"] >= 0) and torch.all(item["colors"] <= 1)

    def test_dataset_length(self):
        """测试数据集长度"""
        # 创建虚拟数据
        transforms_data = self.create_dummy_transforms_json()
        self.create_dummy_images(transforms_data)

        transforms_file = self.data_dir / "transforms.json"
        with open(transforms_file, "w") as f:
            json.dump(transforms_data, f, indent=2)

        dataset = InstantNGPDataset(self.dataset_config)

        # 检查数据集长度
        expected_length = len(transforms_data["frames"])
        assert len(dataset) == expected_length

    def test_ray_generation(self):
        """测试光线生成"""
        # 创建虚拟数据
        transforms_data = self.create_dummy_transforms_json()
        self.create_dummy_images(transforms_data)

        transforms_file = self.data_dir / "transforms.json"
        with open(transforms_file, "w") as f:
            json.dump(transforms_data, f, indent=2)

        dataset = InstantNGPDataset(self.dataset_config)

        # 获取第一个图像的光线
        image_idx = 0
        ray_origins, ray_directions = dataset.get_rays(image_idx)

        # 检查光线形状
        expected_num_rays = self.dataset_config.image_width * self.dataset_config.image_height
        assert ray_origins.shape == (expected_num_rays, 3)
        assert ray_directions.shape == (expected_num_rays, 3)

        # 检查光线方向归一化
        ray_norms = torch.norm(ray_directions, dim=-1)
        assert torch.allclose(ray_norms, torch.ones_like(ray_norms), atol=1e-6)

    def test_camera_intrinsics_parsing(self):
        """测试相机内参解析"""
        transforms_data = self.create_dummy_transforms_json()
        self.create_dummy_images(transforms_data)

        transforms_file = self.data_dir / "transforms.json"
        with open(transforms_file, "w") as f:
            json.dump(transforms_data, f, indent=2)

        dataset = InstantNGPDataset(self.dataset_config)

        # 检查相机内参
        assert hasattr(dataset, "intrinsics")
        intrinsics: dict[str, float] = dataset.intrinsics

        assert isinstance(intrinsics, dict)
        assert "fx" in intrinsics
        assert "fy" in intrinsics
        assert "cx" in intrinsics
        assert "cy" in intrinsics

        # 检查内参值
        assert intrinsics["fx"] == transforms_data["fl_x"]
        assert intrinsics["fy"] == transforms_data["fl_y"]
        assert intrinsics["cx"] == transforms_data["cx"]
        assert intrinsics["cy"] == transforms_data["cy"]

    def test_pose_matrix_parsing(self):
        """测试姿态矩阵解析"""
        transforms_data = self.create_dummy_transforms_json()
        self.create_dummy_images(transforms_data)

        transforms_file = self.data_dir / "transforms.json"
        with open(transforms_file, "w") as f:
            json.dump(transforms_data, f, indent=2)

        dataset = InstantNGPDataset(self.dataset_config)

        # 检查姿态矩阵
        poses: list[torch.Tensor] = dataset.poses

        for i, pose in enumerate(poses):
            # 检查姿态矩阵形状
            assert pose.shape == (4, 4)

            # 检查变换矩阵的最后一行
            expected_last_row = torch.tensor([0, 0, 0, 1], dtype=torch.float32)
            assert torch.allclose(pose[3], expected_last_row)

            # 检查与原始数据的一致性
            original_matrix = torch.tensor(
                transforms_data["frames"][i]["transform_matrix"], dtype=torch.float32
            )
            assert torch.allclose(pose, original_matrix, atol=1e-6)

    def test_image_loading_and_preprocessing(self):
        """测试图像加载和预处理"""
        transforms_data = self.create_dummy_transforms_json()
        self.create_dummy_images(transforms_data)

        transforms_file = self.data_dir / "transforms.json"
        with open(transforms_file, "w") as f:
            json.dump(transforms_data, f, indent=2)

        dataset = InstantNGPDataset(self.dataset_config)

        # 检查图像数据
        images: list[torch.Tensor] = dataset.images

        for image in images:
            # 检查图像形状
            expected_shape = (self.dataset_config.image_height, self.dataset_config.image_width, 3)
            assert image.shape == expected_shape

            # 检查数据类型和值范围
            assert image.dtype == torch.float32
            assert torch.all(image >= 0) and torch.all(image <= 1)

    def test_batch_sampling(self):
        """测试批次采样"""
        transforms_data = self.create_dummy_transforms_json()
        self.create_dummy_images(transforms_data)

        transforms_file = self.data_dir / "transforms.json"
        with open(transforms_file, "w") as f:
            json.dump(transforms_data, f, indent=2)

        dataset = InstantNGPDataset(self.dataset_config)

        # 获取多个批次
        batches: list[dict[str, torch.Tensor]] = []
        for i in range(min(3, len(dataset))):
            batch = dataset[i]
            batches.append(batch)

        # 检查批次一致性
        for batch in batches:
            assert "ray_origins" in batch
            assert "ray_directions" in batch
            assert "colors" in batch

            batch_size = self.dataset_config.batch_size
            assert batch["ray_origins"].shape[0] == batch_size
            assert batch["ray_directions"].shape[0] == batch_size
            assert batch["colors"].shape[0] == batch_size

    def test_dataloader_creation(self):
        """测试数据加载器创建"""
        transforms_data = self.create_dummy_transforms_json()
        self.create_dummy_images(transforms_data)

        transforms_file = self.data_dir / "transforms.json"
        with open(transforms_file, "w") as f:
            json.dump(transforms_data, f, indent=2)

        # 使用工厂函数创建数据加载器
        dataloader = create_instant_ngp_dataloader(
            self.dataset_config,
            batch_size=2,
            shuffle=True,
            num_workers=0,
        )

        # 检查数据加载器
        assert isinstance(dataloader, DataLoader)

        # 测试数据加载
        for batch in dataloader:
            # batch 是从 DataLoader 来的，可能包含多个数据集项
            assert isinstance(batch, (list, tuple))
            break  # 只测试第一个批次

    def test_scene_normalization(self):
        """测试场景归一化"""
        # 创建具有场景缩放的配置
        config_with_scale = InstantNGPDatasetConfig(
            data_dir=str(self.data_dir),
            scene_scale=2.0,  # 缩放场景
            image_width=128,
            image_height=128,
        )

        transforms_data = self.create_dummy_transforms_json()
        self.create_dummy_images(transforms_data)

        transforms_file = self.data_dir / "transforms.json"
        with open(transforms_file, "w") as f:
            json.dump(transforms_data, f, indent=2)

        dataset = InstantNGPDataset(config_with_scale)

        # 检查场景缩放是否应用
        poses: list[torch.Tensor] = dataset.poses

        for pose in poses:
            # 检查平移部分（前3列的第4行）是否受到缩放影响
            translation = pose[:3, 3]
            # 具体缩放逻辑取决于实现
            assert not torch.isnan(translation).any()

    def test_dataset_device_handling(self):
        """测试数据集设备处理"""
        transforms_data = self.create_dummy_transforms_json()
        self.create_dummy_images(transforms_data)

        transforms_file = self.data_dir / "transforms.json"
        with open(transforms_file, "w") as f:
            json.dump(transforms_data, f, indent=2)

        dataset = InstantNGPDataset(self.dataset_config)

        # 获取数据项
        item = dataset[0]

        # 检查数据是否在正确的设备上
        for key, tensor in item.items():
            assert isinstance(tensor, torch.Tensor)
            # 默认应该在 CPU 上
            assert tensor.device.type in ["cpu", "cuda"]

    def test_error_handling(self):
        """测试错误处理"""
        # 测试不存在的数据目录
        invalid_config = InstantNGPDatasetConfig(data_dir="/nonexistent/path")

        with pytest.raises((FileNotFoundError, OSError)):
            InstantNGPDataset(invalid_config)

    def test_dataset_statistics(self):
        """测试数据集统计信息"""
        transforms_data = self.create_dummy_transforms_json()
        self.create_dummy_images(transforms_data)

        transforms_file = self.data_dir / "transforms.json"
        with open(transforms_file, "w") as f:
            json.dump(transforms_data, f, indent=2)

        dataset = InstantNGPDataset(self.dataset_config)

        # 获取数据集统计信息
        stats: dict[str, Any] = dataset.get_statistics()

        assert isinstance(stats, dict)
        assert "num_images" in stats
        assert "image_resolution" in stats
        assert "scene_bounds" in stats

        # 检查统计值
        assert stats["num_images"] == len(transforms_data["frames"])
        assert stats["image_resolution"] == (
            self.dataset_config.image_width,
            self.dataset_config.image_height,
        )

    def teardown_method(self):
        """每个测试方法后的清理"""
        # 清理临时文件
        import shutil

        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)


class TestInstantNGPDatasetIntegration:
    """Instant NGP 数据集集成测试"""

    def setup_method(self):
        """设置集成测试"""
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.temp_dir = tempfile.mkdtemp()
        self.data_dir = Path(self.temp_dir)

    def test_dataset_with_dataloader_integration(self):
        """测试数据集与数据加载器的集成"""
        # 创建虚拟数据
        config = InstantNGPDatasetConfig(
            data_dir=str(self.data_dir),
            batch_size=1000,
            image_width=128,
            image_height=128,
        )

        # 创建虚拟 transforms.json 和图像
        transforms_data: dict[str, Any] = {
            "camera_angle_x": 0.6911112070083618,
            "fl_x": 300.0,
            "fl_y": 300.0,
            "cx": 64.0,
            "cy": 64.0,
            "w": 128,
            "h": 128,
            "frames": [],
        }

        # 创建少量帧用于快速测试
        for i in range(3):
            angle = 2 * np.pi * i / 3
            transform_matrix = [
                [np.cos(angle), 0, np.sin(angle), 2 * np.sin(angle)],
                [0, 1, 0, 0],
                [-np.sin(angle), 0, np.cos(angle), 2 * np.cos(angle)],
                [0, 0, 0, 1],
            ]

            frame_data = {
                "file_path": f"./images/frame_{i:04d}.png",
                "transform_matrix": transform_matrix,
            }
            transforms_data["frames"].append(frame_data)

        # 保存 transforms.json
        transforms_file = self.data_dir / "transforms.json"
        with open(transforms_file, "w") as f:
            json.dump(transforms_data, f, indent=2)

        # 创建图像
        images_dir = self.data_dir / "images"
        images_dir.mkdir(exist_ok=True)

        for frame in transforms_data["frames"]:
            file_path = frame["file_path"]
            if file_path.startswith("./"):
                file_path = file_path[2:]

            image_path = self.data_dir / file_path
            image_array = np.random.randint(0, 256, (128, 128, 3), dtype=np.uint8)
            image = Image.fromarray(image_array)
            image.save(image_path)

        # 创建数据加载器
        dataloader = create_instant_ngp_dataloader(
            config,
            batch_size=2,
            shuffle=True,
            num_workers=0,
        )

        # 测试数据加载
        batch_count = 0
        for batch in dataloader:
            batch_count += 1
            assert isinstance(batch, (list, tuple))
            if batch_count >= 2:  # 只测试前两个批次
                break

        assert batch_count > 0

    def teardown_method(self):
        """清理集成测试"""
        import shutil

        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)


if __name__ == "__main__":
    pytest.main([__file__])
