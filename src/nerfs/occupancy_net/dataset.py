"""
from __future__ import annotations

from typing import Any

Occupancy Network Dataset Implementation
处理占用网络训练所需的数据
"""

import torch
import torch.utils.data as data
import numpy as np
import json
import os
import trimesh


class OccupancyDataset(data.Dataset):
    """占用网络数据集

    从3D网格生成点云和占用标签

    Args:
        data_root: 数据根目录
        split: 数据集分割 ('train', 'val', 'test')
        num_points: 每个样本的点数
        surface_sampling: 表面采样点比例
        uniform_sampling: 均匀采样点比例
        transform: 数据变换
    """

    def __init__(
        self,
        data_root: str,
        split: str = "train",
        num_points: int = 100000,
        surface_sampling: float = 0.5,
        uniform_sampling: float = 0.5,
        bbox_size: float = 1.1,
        transform: Any | None = None,
        near_far: tuple[float, float] | None = None,
    ):
        self.data_root = data_root
        self.split = split
        self.num_points = num_points
        self.surface_sampling = surface_sampling
        self.uniform_sampling = uniform_sampling
        self.bbox_size = bbox_size
        self.transform = transform
        self.near_far = near_far

        # 加载数据列表
        self.data_list = self._load_data_list()

        print(f"Loaded {len(self.data_list)} {split} samples")

    def _load_data_list(self) -> list[dict[str, str]]:
        """加载数据列表"""
        split_file = os.path.join(self.data_root, f"{self.split}.json")

        if os.path.exists(split_file):
            with open(split_file, "r") as f:
                data_list = json.load(f)
        else:
            # 如果没有分割文件，直接从目录加载
            data_list = []
            mesh_dir = os.path.join(self.data_root, "meshes")
            if os.path.exists(mesh_dir):
                for filename in os.listdir(mesh_dir):
                    if filename.endswith((".obj", ".ply", ".off")):
                        data_list.append(
                            {
                                "mesh_path": os.path.join(
                                    mesh_dir,
                                    filename,
                                )
                            }
                        )

        return data_list

    def __len__(self) -> int:
        return len(self.data_list)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        """获取数据样本

        Returns:
            sample: 包含points和occupancy的字典
        """
        data_info = self.data_list[idx]

        # 加载网格
        mesh_path = data_info["mesh_path"]
        mesh = self._load_mesh(mesh_path)

        # 生成采样点和占用标签
        points, occupancy = self._sample_points(mesh)

        # 构建样本
        sample = {
            "points": torch.from_numpy(
                points,
            )
        }

        if self.transform:
            sample = self.transform(sample)

        return sample

    def _load_mesh(self, mesh_path: str) -> trimesh.Trimesh:
        """加载3D网格"""
        try:
            mesh = trimesh.load(mesh_path)

            # 确保是三角网格
            if not isinstance(mesh, trimesh.Trimesh):
                if hasattr(mesh, "dump"):
                    mesh = mesh.dump().sum()
                else:
                    raise ValueError(f"Cannot load mesh from {mesh_path}")

            # 标准化网格
            mesh = self._normalize_mesh(mesh)

            return mesh
        except Exception as e:
            print(f"Error loading mesh {mesh_path}: {e}")
            # 返回默认立方体
            return trimesh.creation.box()

    def _normalize_mesh(self, mesh: trimesh.Trimesh) -> trimesh.Trimesh:
        """标准化网格到单位立方体"""
        # 居中
        center = mesh.bounds.mean(axis=0)
        mesh.vertices -= center

        # 缩放到[-1, 1]范围
        scale = np.max(mesh.bounds[1] - mesh.bounds[0])
        if scale > 0:
            mesh.vertices /= scale / 2.0

        return mesh

    def _sample_points(self, mesh: trimesh.Trimesh) -> tuple[np.ndarray, np.ndarray]:
        """采样点和占用标签

        Args:
            mesh: 三角网格

        Returns:
            points: 采样点 [N, 3]
            occupancy: 占用标签 [N, 1]
        """
        num_surface = int(self.num_points * self.surface_sampling)
        num_uniform = self.num_points - num_surface

        points_list = []
        occupancy_list = []

        # 表面采样（占用=1）
        if num_surface > 0:
            surface_points = self._sample_surface_points(mesh, num_surface)
            points_list.append(surface_points)
            occupancy_list.append(np.ones((num_surface, 1)))

        # 均匀随机采样
        if num_uniform > 0:
            uniform_points = self._sample_uniform_points(num_uniform)
            uniform_occupancy = self._compute_occupancy(mesh, uniform_points)
            points_list.append(uniform_points)
            occupancy_list.append(uniform_occupancy.reshape(-1, 1))

        # 合并所有点
        points = np.concatenate(points_list, axis=0)
        occupancy = np.concatenate(occupancy_list, axis=0)

        return points, occupancy

    def _sample_surface_points(self, mesh: trimesh.Trimesh, num_points: int) -> np.ndarray:
        """在表面采样点"""
        try:
            # 在表面采样
            surface_points, _ = trimesh.sample.sample_surface(mesh, num_points)

            # 添加小的随机偏移
            noise = np.random.normal(0, 0.01, surface_points.shape)
            surface_points += noise

            return surface_points
        except:
            # 如果采样失败，使用顶点
            vertices = mesh.vertices
            if len(vertices) >= num_points:
                indices = np.random.choice(len(vertices), num_points, replace=False)
                return vertices[indices]
            else:
                # 重复采样
                indices = np.random.choice(len(vertices), num_points, replace=True)
                return vertices[indices]

    def _sample_uniform_points(self, num_points: int) -> np.ndarray:
        """均匀采样空间点"""
        points = np.random.uniform(-self.bbox_size, self.bbox_size, (num_points, 3))
        return points

    def _compute_occupancy(self, mesh: trimesh.Trimesh, points: np.ndarray) -> np.ndarray:
        """计算点的占用标签"""
        try:
            # 使用trimesh的contains方法
            occupancy = mesh.contains(points)
            return occupancy.astype(np.float32)
        except:
            # 如果计算失败，使用简单的距离判断
            distances = np.linalg.norm(points, axis=1)
            occupancy = (distances < 1.0).astype(np.float32)
            return occupancy


class SyntheticOccupancyDataset(data.Dataset):
    """合成占用数据集

    生成简单几何形状的占用数据
    """

    def __init__(
        self,
        num_samples: int = 1000,
        num_points: int = 10000,
        shape_types: list[str] = ["sphere", "cube", "cylinder"],
        **kwargs,
    ):
        self.num_samples = num_samples
        self.num_points = num_points
        self.shape_types = shape_types

        print(f"Created synthetic dataset with {num_samples} samples")

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int) -> dict[str, Any]:
        """生成合成数据样本"""
        # 随机选择形状类型
        shape_type = np.random.choice(self.shape_types)

        # 生成形状参数
        scale = np.random.uniform(0.5, 1.0)
        center = np.random.uniform(-0.2, 0.2, 3)

        # 采样点
        points = np.random.uniform(-1.5, 1.5, (self.num_points, 3))

        # 计算占用
        occupancy = self._compute_synthetic_occupancy(points, shape_type, scale, center)

        return {
            "points": torch.from_numpy(points).float(),
            "occupancy": torch.from_numpy(occupancy).float(),
            "shape_id": f"{shape_type}",
        }

    def _compute_synthetic_occupancy(
        self, points: np.ndarray, shape_type: str, scale: float, center: np.ndarray
    ) -> np.ndarray:
        """计算合成形状的占用"""
        # 将点转换到形状坐标系
        local_points = (points - center) / scale

        if shape_type == "sphere":
            distances = np.linalg.norm(local_points, axis=1)
            occupancy = (distances <= 1.0).astype(np.float32)

        elif shape_type == "cube":
            occupancy = np.all(np.abs(local_points) <= 1.0, axis=1).astype(np.float32)

        elif shape_type == "cylinder":
            xy_dist = np.linalg.norm(local_points[:, :2], axis=1)
            z_dist = np.abs(local_points[:, 2])
            occupancy = ((xy_dist <= 1.0) & (z_dist <= 1.0)).astype(np.float32)

        else:
            # 默认球形
            distances = np.linalg.norm(local_points, axis=1)
            occupancy = (distances <= 1.0).astype(np.float32)

        return occupancy.reshape(-1, 1)


def create_occupancy_dataloader(
    dataset: data.Dataset, batch_size: int = 8, shuffle: bool = True, num_workers: int = 4, **kwargs
) -> data.DataLoader:
    """创建占用网络数据加载器"""

    def collate_fn(batch: list[dict[str, Any]]) -> dict[str, Any]:
        """自定义批处理函数"""
        points = torch.stack([item["points"] for item in batch])
        occupancy = torch.stack([item["occupancy"] for item in batch])
        shape_ids = [item["shape_id"] for item in batch]

        batch_dict = {"points": points, "occupancy": occupancy, "shape_ids": shape_ids}

        # 添加其他可能的键
        if "shape_type" in batch[0]:
            batch_dict["shape_types"] = [item["shape_type"] for item in batch]

        return batch_dict

    return data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_fn,
        **kwargs,
    )


# 数据变换类
class OccupancyTransform:
    """占用数据变换基类"""

    def __call__(self, sample: dict[str, Any]) -> dict[str, Any]:
        raise NotImplementedError


class RandomRotation(OccupancyTransform):
    """随机旋转变换"""

    def __init__(self, max_angle: float = np.pi):
        self.max_angle = max_angle

    def __call__(self, sample: dict[str, Any]) -> dict[str, Any]:
        # 生成随机旋转矩阵
        angle = np.random.uniform(-self.max_angle, self.max_angle)
        axis = np.random.randn(3)
        axis = axis / np.linalg.norm(axis)

        # 罗德里格旋转公式
        cos_angle = np.cos(angle)
        sin_angle = np.sin(angle)

        # 创建反对称矩阵
        K = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])

        # 旋转矩阵
        R = np.eye(3) + sin_angle * K + (1 - cos_angle) * np.dot(K, K)

        # 应用旋转
        points = sample["points"].numpy()
        points = np.dot(points, R.T)
        sample["points"] = torch.from_numpy(points).float()

        return sample


class RandomNoise(OccupancyTransform):
    """随机噪声变换"""

    def __init__(self, noise_std: float = 0.01):
        self.noise_std = noise_std

    def __call__(self, sample: dict[str, Any]) -> dict[str, Any]:
        points = sample["points"]
        noise = torch.randn_like(points) * self.noise_std
        sample["points"] = points + noise
        return sample


class Compose(OccupancyTransform):
    """组合多个变换"""

    def __init__(self, transforms: list[OccupancyTransform]):
        self.transforms = transforms

    def __call__(self, sample: dict[str, Any]) -> dict[str, Any]:
        for transform in self.transforms:
            sample = transform(sample)
        return sample
