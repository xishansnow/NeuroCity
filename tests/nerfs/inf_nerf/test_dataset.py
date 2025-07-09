"""
InfNeRF 数据集测试

测试 InfNeRF 数据集的各种功能，包括：
- 数据集加载
- 数据预处理
- 相机参数处理
- 光线生成
- 数据增强
"""

import pytest
import torch
import numpy as np
from pathlib import Path
from unittest.mock import Mock, patch

from src.nerfs.inf_nerf import InfNeRFDataset, InfNeRFDatasetConfig


class TestInfNeRFDatasetConfig:
    """测试数据集配置"""

    def test_default_config(self):
        """测试默认配置"""
        config = InfNeRFDatasetConfig()

        assert config.data_dir == "data"
        assert config.image_size == 800
        assert config.batch_size == 1
        assert config.num_workers == 4
        assert config.pin_memory is True
        assert config.shuffle is True

    def test_custom_config(self):
        """测试自定义配置"""
        config = InfNeRFDatasetConfig(
            data_dir="custom_data",
            image_size=1024,
            batch_size=2,
            num_workers=8,
            pin_memory=False,
            shuffle=False,
        )

        assert config.data_dir == "custom_data"
        assert config.image_size == 1024
        assert config.batch_size == 2
        assert config.num_workers == 8
        assert config.pin_memory is False
        assert config.shuffle is False

    def test_config_validation(self):
        """测试配置验证"""
        with pytest.raises(ValueError):
            InfNeRFDatasetConfig(image_size=0)

        with pytest.raises(ValueError):
            InfNeRFDatasetConfig(batch_size=0)

        with pytest.raises(ValueError):
            InfNeRFDatasetConfig(num_workers=-1)


class TestInfNeRFDataset:
    """测试 InfNeRF 数据集"""

    def test_dataset_creation(self, temp_dir):
        """测试数据集创建"""
        # 创建模拟数据目录结构
        data_dir = temp_dir / "test_data"
        data_dir.mkdir()

        # 创建模拟的 transforms.json
        transforms = {
            "camera_angle_x": 0.857556,
            "frames": [
                {
                    "file_path": "images/image_000",
                    "transform_matrix": [
                        [1.0, 0.0, 0.0, 0.0],
                        [0.0, 1.0, 0.0, 0.0],
                        [0.0, 0.0, 1.0, 2.0],
                        [0.0, 0.0, 0.0, 1.0],
                    ],
                }
            ],
        }

        import json

        with open(data_dir / "transforms.json", "w") as f:
            json.dump(transforms, f)

        # 创建图像目录
        images_dir = data_dir / "images"
        images_dir.mkdir()

        # 创建模拟图像文件
        (images_dir / "image_000.png").touch()

        config = InfNeRFDatasetConfig(data_dir=str(data_dir))
        dataset = InfNeRFDataset(config)

        assert len(dataset) == 1
        assert dataset.config is config

    def test_dataset_loading(self, temp_dir):
        """测试数据集加载"""
        # 创建更完整的模拟数据
        data_dir = temp_dir / "test_data"
        data_dir.mkdir()

        # 创建多个帧的 transforms.json
        transforms = {"camera_angle_x": 0.857556, "frames": []}

        for i in range(3):
            frame = {
                "file_path": f"images/image_{i:03d}",
                "transform_matrix": [
                    [1.0, 0.0, 0.0, i * 2.0],
                    [0.0, 1.0, 0.0, 0.0],
                    [0.0, 0.0, 1.0, 2.0],
                    [0.0, 0.0, 0.0, 1.0],
                ],
            }
            transforms["frames"].append(frame)

        import json

        with open(data_dir / "transforms.json", "w") as f:
            json.dump(transforms, f)

        # 创建图像目录和文件
        images_dir = data_dir / "images"
        images_dir.mkdir()

        for i in range(3):
            (images_dir / f"image_{i:03d}.png").touch()

        config = InfNeRFDatasetConfig(data_dir=str(data_dir))
        dataset = InfNeRFDataset(config)

        assert len(dataset) == 3
        assert len(dataset.frames) == 3
        assert dataset.camera_angle_x == 0.857556

    def test_get_item(self, temp_dir):
        """测试获取数据项"""
        # 创建模拟数据
        data_dir = temp_dir / "test_data"
        data_dir.mkdir()

        transforms = {
            "camera_angle_x": 0.857556,
            "frames": [
                {
                    "file_path": "images/image_000",
                    "transform_matrix": [
                        [1.0, 0.0, 0.0, 0.0],
                        [0.0, 1.0, 0.0, 0.0],
                        [0.0, 0.0, 1.0, 2.0],
                        [0.0, 0.0, 0.0, 1.0],
                    ],
                }
            ],
        }

        import json

        with open(data_dir / "transforms.json", "w") as f:
            json.dump(transforms, f)

        images_dir = data_dir / "images"
        images_dir.mkdir()
        (images_dir / "image_000.png").touch()

        config = InfNeRFDatasetConfig(data_dir=str(data_dir))
        dataset = InfNeRFDataset(config)

        # 获取数据项
        item = dataset[0]

        assert isinstance(item, dict)
        assert "rays_o" in item
        assert "rays_d" in item
        assert "target_rgb" in item
        assert "near" in item
        assert "far" in item
        assert "focal_length" in item
        assert "pixel_width" in item

    def test_camera_matrix_processing(self, temp_dir):
        """测试相机矩阵处理"""
        data_dir = temp_dir / "test_data"
        data_dir.mkdir()

        # 创建测试相机矩阵
        test_matrix = [
            [1.0, 0.0, 0.0, 5.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 3.0],
            [0.0, 0.0, 0.0, 1.0],
        ]

        transforms = {
            "camera_angle_x": 0.857556,
            "frames": [{"file_path": "images/image_000", "transform_matrix": test_matrix}],
        }

        import json

        with open(data_dir / "transforms.json", "w") as f:
            json.dump(transforms, f)

        images_dir = data_dir / "images"
        images_dir.mkdir()
        (images_dir / "image_000.png").touch()

        config = InfNeRFDatasetConfig(data_dir=str(data_dir))
        dataset = InfNeRFDataset(config)

        # 检查相机矩阵是否正确加载
        frame = dataset.frames[0]
        matrix = np.array(frame["transform_matrix"])

        assert matrix.shape == (4, 4)
        assert matrix[0, 3] == 5.0  # x translation
        assert matrix[2, 3] == 3.0  # z translation

    def test_ray_generation(self, temp_dir):
        """测试光线生成"""
        data_dir = temp_dir / "test_data"
        data_dir.mkdir()

        transforms = {
            "camera_angle_x": 0.857556,
            "frames": [
                {
                    "file_path": "images/image_000",
                    "transform_matrix": [
                        [1.0, 0.0, 0.0, 0.0],
                        [0.0, 1.0, 0.0, 0.0],
                        [0.0, 0.0, 1.0, 2.0],
                        [0.0, 0.0, 0.0, 1.0],
                    ],
                }
            ],
        }

        import json

        with open(data_dir / "transforms.json", "w") as f:
            json.dump(transforms, f)

        images_dir = data_dir / "images"
        images_dir.mkdir()
        (images_dir / "image_000.png").touch()

        config = InfNeRFDatasetConfig(data_dir=str(data_dir), image_size=32)
        dataset = InfNeRFDataset(config)

        item = dataset[0]

        # 检查光线参数
        assert item["rays_o"].shape == (32 * 32, 3)
        assert item["rays_d"].shape == (32 * 32, 3)

        # 检查光线方向是否归一化
        directions_norm = torch.norm(item["rays_d"], dim=-1)
        assert torch.allclose(directions_norm, torch.ones_like(directions_norm), atol=1e-6)

    def test_data_augmentation(self, temp_dir):
        """测试数据增强"""
        data_dir = temp_dir / "test_data"
        data_dir.mkdir()

        transforms = {
            "camera_angle_x": 0.857556,
            "frames": [
                {
                    "file_path": "images/image_000",
                    "transform_matrix": [
                        [1.0, 0.0, 0.0, 0.0],
                        [0.0, 1.0, 0.0, 0.0],
                        [0.0, 0.0, 1.0, 2.0],
                        [0.0, 0.0, 0.0, 1.0],
                    ],
                }
            ],
        }

        import json

        with open(data_dir / "transforms.json", "w") as f:
            json.dump(transforms, f)

        images_dir = data_dir / "images"
        images_dir.mkdir()
        (images_dir / "image_000.png").touch()

        config = InfNeRFDatasetConfig(
            data_dir=str(data_dir), image_size=32, use_data_augmentation=True
        )
        dataset = InfNeRFDataset(config)

        # 获取两次数据项，应该有所不同（由于数据增强）
        item1 = dataset[0]
        item2 = dataset[0]

        # 检查数据增强是否生效
        if config.use_data_augmentation:
            # 光线应该有所不同
            assert not torch.allclose(item1["rays_o"], item2["rays_o"])

    def test_train_val_split(self, temp_dir):
        """测试训练验证集分割"""
        data_dir = temp_dir / "test_data"
        data_dir.mkdir()

        # 创建多个帧
        transforms = {"camera_angle_x": 0.857556, "frames": []}

        for i in range(10):
            frame = {
                "file_path": f"images/image_{i:03d}",
                "transform_matrix": [
                    [1.0, 0.0, 0.0, i * 0.5],
                    [0.0, 1.0, 0.0, 0.0],
                    [0.0, 0.0, 1.0, 2.0],
                    [0.0, 0.0, 0.0, 1.0],
                ],
            }
            transforms["frames"].append(frame)

        import json

        with open(data_dir / "transforms.json", "w") as f:
            json.dump(transforms, f)

        images_dir = data_dir / "images"
        images_dir.mkdir()

        for i in range(10):
            (images_dir / f"image_{i:03d}.png").touch()

        # 创建训练集
        train_config = InfNeRFDatasetConfig(data_dir=str(data_dir), split="train", train_ratio=0.8)
        train_dataset = InfNeRFDataset(train_config)

        # 创建验证集
        val_config = InfNeRFDatasetConfig(data_dir=str(data_dir), split="val", train_ratio=0.8)
        val_dataset = InfNeRFDataset(val_config)

        # 检查分割
        assert len(train_dataset) == 8  # 80% of 10
        assert len(val_dataset) == 2  # 20% of 10

        # 检查没有重叠
        train_indices = set(range(len(train_dataset)))
        val_indices = set(range(len(val_dataset)))
        assert len(train_indices.intersection(val_indices)) == 0

    def test_error_handling(self, temp_dir):
        """测试错误处理"""
        # 测试不存在的数据目录
        config = InfNeRFDatasetConfig(data_dir="nonexistent_dir")

        with pytest.raises(FileNotFoundError):
            InfNeRFDataset(config)

        # 测试无效的 transforms.json
        data_dir = temp_dir / "invalid_data"
        data_dir.mkdir()

        # 创建无效的 JSON 文件
        with open(data_dir / "transforms.json", "w") as f:
            f.write("invalid json")

        config = InfNeRFDatasetConfig(data_dir=str(data_dir))

        with pytest.raises((ValueError, json.JSONDecodeError)):
            InfNeRFDataset(config)

    def test_dataset_statistics(self, temp_dir):
        """测试数据集统计信息"""
        data_dir = temp_dir / "test_data"
        data_dir.mkdir()

        transforms = {"camera_angle_x": 0.857556, "frames": []}

        for i in range(5):
            frame = {
                "file_path": f"images/image_{i:03d}",
                "transform_matrix": [
                    [1.0, 0.0, 0.0, i * 1.0],
                    [0.0, 1.0, 0.0, 0.0],
                    [0.0, 0.0, 1.0, 2.0],
                    [0.0, 0.0, 0.0, 1.0],
                ],
            }
            transforms["frames"].append(frame)

        import json

        with open(data_dir / "transforms.json", "w") as f:
            json.dump(transforms, f)

        images_dir = data_dir / "images"
        images_dir.mkdir()

        for i in range(5):
            (images_dir / f"image_{i:03d}.png").touch()

        config = InfNeRFDatasetConfig(data_dir=str(data_dir))
        dataset = InfNeRFDataset(config)

        # 获取统计信息
        stats = dataset.get_statistics()

        assert isinstance(stats, dict)
        assert "num_frames" in stats
        assert "image_size" in stats
        assert "camera_angle_x" in stats
        assert stats["num_frames"] == 5
        assert stats["image_size"] == config.image_size
        assert stats["camera_angle_x"] == 0.857556

    def test_dataset_visualization(self, temp_dir):
        """测试数据集可视化"""
        data_dir = temp_dir / "test_data"
        data_dir.mkdir()

        transforms = {"camera_angle_x": 0.857556, "frames": []}

        for i in range(3):
            frame = {
                "file_path": f"images/image_{i:03d}",
                "transform_matrix": [
                    [1.0, 0.0, 0.0, i * 2.0],
                    [0.0, 1.0, 0.0, 0.0],
                    [0.0, 0.0, 1.0, 2.0],
                    [0.0, 0.0, 0.0, 1.0],
                ],
            }
            transforms["frames"].append(frame)

        import json

        with open(data_dir / "transforms.json", "w") as f:
            json.dump(transforms, f)

        images_dir = data_dir / "images"
        images_dir.mkdir()

        for i in range(3):
            (images_dir / f"image_{i:03d}.png").touch()

        config = InfNeRFDatasetConfig(data_dir=str(data_dir))
        dataset = InfNeRFDataset(config)

        # 测试可视化方法（应该不抛出异常）
        try:
            dataset.visualize_camera_positions()
        except Exception as e:
            # 如果可视化失败，至少不应该崩溃
            assert "matplotlib" in str(e) or "plotly" in str(e)

    def test_dataset_serialization(self, temp_dir):
        """测试数据集序列化"""
        data_dir = temp_dir / "test_data"
        data_dir.mkdir()

        transforms = {
            "camera_angle_x": 0.857556,
            "frames": [
                {
                    "file_path": "images/image_000",
                    "transform_matrix": [
                        [1.0, 0.0, 0.0, 0.0],
                        [0.0, 1.0, 0.0, 0.0],
                        [0.0, 0.0, 1.0, 2.0],
                        [0.0, 0.0, 0.0, 1.0],
                    ],
                }
            ],
        }

        import json

        with open(data_dir / "transforms.json", "w") as f:
            json.dump(transforms, f)

        images_dir = data_dir / "images"
        images_dir.mkdir()
        (images_dir / "image_000.png").touch()

        config = InfNeRFDatasetConfig(data_dir=str(data_dir))
        dataset = InfNeRFDataset(config)

        # 测试保存和加载数据集信息
        info_path = temp_dir / "dataset_info.json"
        dataset.save_info(info_path)

        assert info_path.exists()

        # 加载信息
        loaded_info = dataset.load_info(info_path)
        assert isinstance(loaded_info, dict)
        assert "num_frames" in loaded_info
        assert "camera_angle_x" in loaded_info
