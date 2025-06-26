"""
SDF Network Dataset Implementation
处理SDF网络训练所需的数据
"""

import torch
import torch.utils.data as data
import numpy as np
import json
import os
from typing import Dict, List, Optional, Tuple, Any, Union
import trimesh
from scipy.spatial.distance import cdist


class SDFDataset(data.Dataset):
    """SDF网络数据集
    
    从3D网格生成点云和SDF标签
    
    Args:
        data_root: 数据根目录
        split: 数据集分割 ('train', 'val', 'test')
        num_points: 每个样本的点数
        surface_sampling: 表面采样点比例
        near_surface_sampling: 近表面采样点比例
        uniform_sampling: 均匀采样点比例
        surface_std: 表面噪声标准差
        transform: 数据变换
    """
    
    def __init__(
        self, data_root: str, split: str = 'train', num_points: int = 100000, surface_sampling: float = 0.4, near_surface_sampling: float = 0.4, uniform_sampling: float = 0.2, surface_std: float = 0.01, near_surface_std: float = 0.1, bbox_size: float = 1.1, clamp_distance: float = 0.1, transform: Optional = None
    ):
        self.data_root = data_root
        self.split = split
        self.num_points = num_points
        self.surface_sampling = surface_sampling
        self.near_surface_sampling = near_surface_sampling
        self.uniform_sampling = uniform_sampling
        self.surface_std = surface_std
        self.near_surface_std = near_surface_std
        self.bbox_size = bbox_size
        self.clamp_distance = clamp_distance
        self.transform = transform
        
        # 确保采样比例之和为1
        total_sampling = surface_sampling + near_surface_sampling + uniform_sampling
        if abs(total_sampling - 1.0) > 1e-6:
            print(f"Warning: Sampling ratios sum to {total_sampling}, normalizing to 1.0")
            self.surface_sampling /= total_sampling
            self.near_surface_sampling /= total_sampling
            self.uniform_sampling /= total_sampling
        
        # 加载数据列表
        self.data_list = self._load_data_list()
        
        print(f"Loaded {len(self.data_list)} {split} samples")
    
    def _load_data_list(self) -> list[Dict]:
        """加载数据列表"""
        split_file = os.path.join(self.data_root, f'{self.split}.json')
        
        if os.path.exists(split_file):
            with open(split_file, 'r') as f:
                data_list = json.load(f)
        else:
            # 如果没有分割文件，直接从目录加载
            data_list = []
            mesh_dir = os.path.join(self.data_root, 'meshes')
            if os.path.exists(mesh_dir):
                for filename in os.listdir(mesh_dir):
                    if filename.endswith(('.obj', '.ply', '.off')):
                        data_list.append({
                            'mesh_path': os.path.join(
                                mesh_dir,
                                filename,
                            )
                        })
        
        return data_list
    
    def __len__(self) -> int:
        return len(self.data_list)
    
    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        """获取数据样本
        
        Returns:
            sample: 包含points和sdf的字典
        """
        data_info = self.data_list[idx]
        
        # 加载网格
        mesh_path = data_info['mesh_path']
        mesh = self._load_mesh(mesh_path)
        
        # 生成采样点和SDF标签
        points, sdf = self._sample_points(mesh)
        
        # 构建样本
        sample = {
            'points': torch.from_numpy(
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
                if hasattr(mesh, 'dump'):
                    mesh = mesh.dump().sum()
                else:
                    raise ValueError(f"Cannot load mesh from {mesh_path}")
            
            # 标准化网格
            mesh = self._normalize_mesh(mesh)
            
            return mesh
        except Exception as e:
            print(f"Error loading mesh {mesh_path}: {e}")
            # 返回默认球体
            return trimesh.creation.uv_sphere(radius=0.5)
    
    def _normalize_mesh(self, mesh: trimesh.Trimesh) -> trimesh.Trimesh:
        """标准化网格到单位立方体"""
        # 居中
        center = mesh.bounds.mean(axis=0)
        mesh.vertices -= center
        
        # 缩放到[-1, 1]范围
        scale = np.max(mesh.bounds[1] - mesh.bounds[0])
        if scale > 0:
            mesh.vertices /= (scale / 2.0)
        
        return mesh
    
    def _sample_points(
        self, mesh: trimesh.Trimesh
    ) -> tuple[np.ndarray, np.ndarray]:
        """采样点和SDF标签
        
        Args:
            mesh: 三角网格
            
        Returns:
            points: 采样点 [N, 3]
            sdf: SDF值 [N, 1]
        """
        num_surface = int(self.num_points * self.surface_sampling)
        num_near_surface = int(self.num_points * self.near_surface_sampling)
        num_uniform = self.num_points - num_surface - num_near_surface
        
        points_list = []
        sdf_list = []
        
        # 表面采样（SDF≈0）
        if num_surface > 0:
            surface_points = self._sample_surface_points(mesh, num_surface)
            surface_sdf = self._compute_sdf(mesh, surface_points)
            points_list.append(surface_points)
            sdf_list.append(surface_sdf)
        
        # 近表面采样
        if num_near_surface > 0:
            near_surface_points = self._sample_near_surface_points(mesh, num_near_surface)
            near_surface_sdf = self._compute_sdf(mesh, near_surface_points)
            points_list.append(near_surface_points)
            sdf_list.append(near_surface_sdf)
        
        # 均匀随机采样
        if num_uniform > 0:
            uniform_points = self._sample_uniform_points(num_uniform)
            uniform_sdf = self._compute_sdf(mesh, uniform_points)
            points_list.append(uniform_points)
            sdf_list.append(uniform_sdf)
        
        # 合并所有点
        points = np.concatenate(points_list, axis=0)
        sdf = np.concatenate(sdf_list, axis=0)
        
        # 限制SDF值范围
        sdf = np.clip(sdf, -self.clamp_distance, self.clamp_distance)
        
        return points, sdf.reshape(-1, 1)
    
    def _sample_surface_points(
        self, mesh: trimesh.Trimesh, num_points: int
    ) -> np.ndarray:
        """在表面采样点"""
        try:
            # 在表面采样
            surface_points, _ = trimesh.sample.sample_surface(mesh, num_points)
            
            # 添加小的随机偏移
            noise = np.random.normal(0, self.surface_std, surface_points.shape)
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
    
    def _sample_near_surface_points(
        self, mesh: trimesh.Trimesh, num_points: int
    ) -> np.ndarray:
        """在表面附近采样点"""
        try:
            # 先在表面采样
            surface_points, _ = trimesh.sample.sample_surface(mesh, num_points)
            
            # 添加较大的随机偏移
            noise = np.random.normal(0, self.near_surface_std, surface_points.shape)
            near_surface_points = surface_points + noise
            
            return near_surface_points
        except:
            # 如果失败，使用均匀采样
            return self._sample_uniform_points(num_points)
    
    def _sample_uniform_points(self, num_points: int) -> np.ndarray:
        """均匀采样空间点"""
        points = np.random.uniform(
            -self.bbox_size, self.bbox_size, (num_points, 3)
        )
        return points
    
    def _compute_sdf(
        self, mesh: trimesh.Trimesh, points: np.ndarray
    ) -> np.ndarray:
        """计算点的SDF值"""
        try:
            # 使用trimesh的nearest_point方法计算最近点
            closest_points, distances, face_indices = mesh.nearest.on_surface(points)
            
            # 判断点是否在网格内部
            try:
                is_inside = mesh.contains(points)
                # 内部点的距离为负
                sdf = np.where(is_inside, -distances, distances)
            except:
                # 如果contains方法失败，使用简单的距离判断
                center_distances = np.linalg.norm(points, axis=1)
                mesh_scale = np.max(np.linalg.norm(mesh.vertices, axis=1))
                is_inside = center_distances < mesh_scale * 0.8
                sdf = np.where(is_inside, -distances, distances)
            
            return sdf.astype(np.float32)
            
        except Exception as e:
            print(f"SDF computation failed: {e}, using fallback method")
            # 简单的球形SDF作为后备
            distances = np.linalg.norm(points, axis=1)
            sdf = distances - 0.5  # 假设单位球
            return sdf.astype(np.float32)


class SyntheticSDFDataset(data.Dataset):
    """合成SDF数据集
    
    生成简单几何形状的SDF数据
    """
    
    def __init__(
        self, num_samples: int = 1000, num_points: int = 10000, shape_types: list[str] = ['sphere', 'cube', 'cylinder'], **kwargs
    ):
        self.num_samples = num_samples
        self.num_points = num_points
        self.shape_types = shape_types
        
        print(f"Created synthetic SDF dataset with {num_samples} samples")
    
    def __len__(self) -> int:
        return self.num_samples
    
    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        """生成合成数据样本"""
        # 随机选择形状类型
        shape_type = np.random.choice(self.shape_types)
        
        # 生成形状参数
        scale = np.random.uniform(0.5, 1.0)
        center = np.random.uniform(-0.2, 0.2, 3)
        
        # 采样点
        points = np.random.uniform(-1.5, 1.5, (self.num_points, 3))
        
        # 计算SDF
        sdf = self._compute_synthetic_sdf(
            points, shape_type, scale, center
        )
        
        return {
            'points': torch.from_numpy(points).float(),
            'sdf': torch.from_numpy(sdf).float(),
            'shape_id': f"{shape_type}",
        }
    
    def _compute_synthetic_sdf(
        self, points: np.ndarray, shape_type: str, scale: float, center: np.ndarray
    ) -> np.ndarray:
        """计算合成形状的SDF"""
        # 将点转换到形状坐标系
        local_points = (points - center) / scale
        
        if shape_type == 'sphere':
            sdf = np.linalg.norm(local_points, axis=1) - 1.0
        
        elif shape_type == 'cube':
            # 立方体SDF
            d = np.abs(local_points) - 1.0
            sdf = (np.linalg.norm(np.maximum(d, 0), axis=1) + 
                   np.minimum(np.max(d, axis=1), 0))
        
        elif shape_type == 'cylinder':
            # 圆柱体SDF
            xy_dist = np.linalg.norm(local_points[:, :2], axis=1) - 1.0
            z_dist = np.abs(local_points[:, 2]) - 1.0
            
            # 组合距离
            outside_xy = np.maximum(xy_dist, 0)
            outside_z = np.maximum(z_dist, 0)
            sdf = np.sqrt(outside_xy**2 + outside_z**2) + np.minimum(np.maximum(xy_dist, z_dist), 0)
        
        else:
            # 默认球形
            sdf = np.linalg.norm(local_points, axis=1) - 1.0
        
        # 缩放回原始空间
        sdf *= scale
        
        return sdf.reshape(-1, 1)


class LatentSDFDataset(data.Dataset):
    """潜在SDF数据集
    
    支持形状编码的SDF数据集
    """
    
    def __init__(
        self, data_root: str, split: str = 'train', latent_codes_path: Optional[str] = None, **kwargs
    ):
        super().__init__()
        
        # 基础SDF数据集
        self.base_dataset = SDFDataset(data_root, split, **kwargs)
        
        # 加载潜在编码
        self.latent_codes = None
        if latent_codes_path and os.path.exists(latent_codes_path):
            self.latent_codes = torch.load(latent_codes_path)
            print(f"Loaded latent codes from {latent_codes_path}")
    
    def __len__(self) -> int:
        return len(self.base_dataset)
    
    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        """获取带潜在编码的数据样本"""
        sample = self.base_dataset[idx]
        
        # 添加潜在编码
        if self.latent_codes is not None:
            shape_id = sample['shape_id']
            if isinstance(shape_id, str):
                # 尝试从形状ID获取索引
                try:
                    shape_idx = int(shape_id.split('_')[-1]) if '_' in shape_id else idx
                except:
                    shape_idx = idx
            else:
                shape_idx = shape_id
            
            if shape_idx < len(self.latent_codes):
                sample['latent_code'] = self.latent_codes[shape_idx]
            else:
                # 如果没有对应的潜在编码，使用随机编码
                latent_dim = self.latent_codes.shape[1] if len(self.latent_codes) > 0 else 256
                sample['latent_code'] = torch.randn(latent_dim) * 0.01
        else:
            # 如果没有预训练的潜在编码，使用随机编码
            sample['latent_code'] = torch.randn(256) * 0.01
        
        return sample


def create_sdf_dataloader(
    dataset: data.Dataset, batch_size: int = 8, shuffle: bool = True, num_workers: int = 4, **kwargs
) -> data.DataLoader:
    """创建SDF网络数据加载器"""
    
    def collate_fn(batch):
        """自定义批处理函数"""
        points = torch.stack([item['points'] for item in batch])
        sdf = torch.stack([item['sdf'] for item in batch])
        shape_ids = [item['shape_id'] for item in batch]
        
        batch_dict = {
            'points': points, 'sdf': sdf, 'shape_ids': shape_ids
        }
        
        # 添加其他可能的键
        if 'latent_code' in batch[0]:
            batch_dict['latent_codes'] = torch.stack([item['latent_code'] for item in batch])
        
        if 'shape_type' in batch[0]:
            batch_dict['shape_types'] = [item['shape_type'] for item in batch]
        
        return batch_dict
    
    return data.DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, collate_fn=collate_fn, **kwargs
    )


# 数据变换类
class SDFTransform:
    """SDF数据变换基类"""
    
    def __call__(self, sample: Dict) -> Dict:
        raise NotImplementedError


class RandomRotationSDF(SDFTransform):
    """随机旋转变换（保持SDF值不变）"""
    
    def __init__(self, max_angle: float = np.pi):
        self.max_angle = max_angle
    
    def __call__(self, sample: Dict) -> Dict:
        # 生成随机旋转矩阵
        angle = np.random.uniform(-self.max_angle, self.max_angle)
        axis = np.random.randn(3)
        axis = axis / np.linalg.norm(axis)
        
        # 罗德里格旋转公式
        cos_angle = np.cos(angle)
        sin_angle = np.sin(angle)
        
        # 创建反对称矩阵
        K = np.array([
            [0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]
        ])
        
        # 旋转矩阵
        R = np.eye(3) + sin_angle * K + (1 - cos_angle) * np.dot(K, K)
        
        # 应用旋转
        points = sample['points'].numpy()
        points = np.dot(points, R.T)
        sample['points'] = torch.from_numpy(points).float()
        
        return sample


class RandomNoiseSDF(SDFTransform):
    """随机噪声变换"""
    
    def __init__(self, noise_std: float = 0.01):
        self.noise_std = noise_std
    
    def __call__(self, sample: Dict) -> Dict:
        points = sample['points']
        noise = torch.randn_like(points) * self.noise_std
        sample['points'] = points + noise
        return sample


class ComposeSDF(SDFTransform):
    """组合多个变换"""
    
    def __init__(self, transforms: list[SDFTransform]):
        self.transforms = transforms
    
    def __call__(self, sample: Dict) -> Dict:
        for transform in self.transforms:
            sample = transform(sample)
        return sample 