"""
DataGen 核心模块

提供数据生成的核心配置和管道功能。
"""

import torch
import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


@dataclass
class DataGenConfig:
    """数据生成配置类"""
    
    # 基础配置
    output_dir: str = "generated_data"
    random_seed: int = 42
    device: str = "auto"  # auto, cuda, cpu
    
    # 体素配置
    voxel_size: float = 1.0
    grid_size: tuple[int, int, int] = (64, 64, 64)
    scene_bounds: tuple[float, float, float, float, float, float] = (
        -32.0,
        -32.0,
        -32.0,
        32.0,
        32.0,
        32.0,
    )
    
    # 采样配置
    sample_ratio: float = 0.1
    samples_per_tile: int = 10000
    stratified_sampling: bool = True
    surface_sampling_ratio: float = 0.3
    
    # SDF配置
    sdf_truncation: float = 5.0
    sdf_clamp: tuple[float, float] = (-1.0, 1.0)
    
    # 训练配置
    batch_size: int = 1024
    num_workers: int = 4
    
    # 生成器配置
    generator_type: str = "neural"  # neural, analytical, hybrid
    network_depth: int = 8
    network_width: int = 256
    encoding_levels: int = 16
    
    def validate(self) -> bool:
        """验证配置参数"""
        if self.voxel_size <= 0:
            raise ValueError("voxel_size 必须大于 0")
        
        if not all(s > 0 for s in self.grid_size):
            raise ValueError("grid_size 的所有维度必须大于 0")
        
        if not (0 < self.sample_ratio <= 1):
            raise ValueError("sample_ratio 必须在 (0, 1] 范围内")
        
        if self.samples_per_tile <= 0:
            raise ValueError("samples_per_tile 必须大于 0")
        
        return True


class DataGenPipeline:
    """数据生成管道"""
    
    def __init__(self, config: DataGenConfig):
        """
        初始化数据生成管道
        
        Args:
            config: 数据生成配置
        """
        self.config = config
        config.validate()
        
        # 设置随机种子
        self._set_random_seed(config.random_seed)
        
        # 设置设备
        self.device = self._setup_device(config.device)
        
        # 创建输出目录
        self.output_dir = Path(config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # 初始化统计信息
        self.stats = {
            'total_samples': 0, 'processing_time': 0.0, 'memory_usage': 0.0
        }
        
        logger.info(f"DataGen管道初始化完成，设备: {self.device}")
    
    def _set_random_seed(self, seed: int) -> None:
        """设置随机种子"""
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
    
    def _setup_device(self, device: str) -> torch.device:
        """设置计算设备"""
        if device == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            return torch.device(device)
    
    def generate_voxel_data(
        self,
        scene_params: dict[str, Any],
        save_path: Optional[str] = None,
    ) -> dict[str, np.ndarray]:
        """生成体素数据"""
        logger.info("开始生成体素数据...")
        
        # 根据场景类型生成体素
        if scene_params.get('type') == 'sphere':
            voxel_data = self._generate_sphere_voxels(scene_params)
        elif scene_params.get('type') == 'cube':
            voxel_data = self._generate_cube_voxels(scene_params)
        elif scene_params.get('type') == 'complex':
            voxel_data = self._generate_complex_scene_voxels(scene_params)
        else:
            raise ValueError(f"未知场景类型: {scene_params.get('type')}")
        
        # 保存数据
        if save_path:
            self._save_voxel_data(voxel_data, save_path)
        
        # 更新统计
        self.stats['total_samples'] += np.prod(voxel_data['occupancy'].shape)
        
        logger.info(f"体素数据生成完成，形状: {voxel_data['occupancy'].shape}")
        return voxel_data
    
    def _generate_sphere_voxels(self, params: dict[str, Any]) -> dict[str, np.ndarray]:
        """生成球体体素"""
        center = params.get('center', (0, 0, 0))
        radius = params.get('radius', 10.0)
        
        # 创建坐标网格
        x = np.linspace(
            self.config.scene_bounds[0],
            self.config.scene_bounds[3],
            self.config.grid_size[0],
        )
        y = np.linspace(
            self.config.scene_bounds[1],
            self.config.scene_bounds[4],
            self.config.grid_size[1],
        )
        z = np.linspace(
            self.config.scene_bounds[2],
            self.config.scene_bounds[5],
            self.config.grid_size[2],
        )
        
        X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
        
        # 计算到球心的距离
        distance = np.sqrt((X - center[0])**2 + (Y - center[1])**2 + (Z - center[2])**2)
        
        # 生成占用和SDF
        occupancy = (distance <= radius).astype(np.float32)
        sdf = distance - radius
        sdf = np.clip(sdf, self.config.sdf_clamp[0], self.config.sdf_clamp[1])
        
        return {
            'occupancy': occupancy, 'sdf': sdf, 'coordinates': np.stack([X, Y, Z], axis=-1)
        }
    
    def _generate_cube_voxels(self, params: dict[str, Any]) -> dict[str, np.ndarray]:
        """生成立方体体素"""
        center = params.get('center', (0, 0, 0))
        size = params.get('size', 20.0)
        
        # 创建坐标网格
        x = np.linspace(
            self.config.scene_bounds[0],
            self.config.scene_bounds[3],
            self.config.grid_size[0],
        )
        y = np.linspace(
            self.config.scene_bounds[1],
            self.config.scene_bounds[4],
            self.config.grid_size[1],
        )
        z = np.linspace(
            self.config.scene_bounds[2],
            self.config.scene_bounds[5],
            self.config.grid_size[2],
        )
        
        X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
        
        # 计算到立方体边界的距离
        half_size = size / 2
        dx = np.maximum(np.abs(X - center[0]) - half_size, 0)
        dy = np.maximum(np.abs(Y - center[1]) - half_size, 0)
        dz = np.maximum(np.abs(Z - center[2]) - half_size, 0)
        
        # 外部距离
        external_dist = np.sqrt(dx**2 + dy**2 + dz**2)
        
        # 内部距离
        internal_dist = np.minimum(np.minimum(
            half_size - np.abs(X - center[0]), half_size - np.abs(Y - center[1])
        ), half_size - np.abs(Z - center[2]))
        
        # 组合SDF
        sdf = np.where(
            (np.abs(X - center[0]) <= half_size) & 
            (np.abs(Y - center[1]) <= half_size) & 
            (np.abs(Z - center[2]) <= half_size), -internal_dist, # 内部为负
            external_dist    # 外部为正
        )
        
        occupancy = (sdf <= 0).astype(np.float32)
        sdf = np.clip(sdf, self.config.sdf_clamp[0], self.config.sdf_clamp[1])
        
        return {
            'occupancy': occupancy, 'sdf': sdf, 'coordinates': np.stack([X, Y, Z], axis=-1)
        }
    
    def _generate_complex_scene_voxels(self, params: dict[str, Any]) -> dict[str, np.ndarray]:
        """生成复杂场景体素"""
        objects = params.get('objects', [])
        
        # 初始化空场景
        x = np.linspace(
            self.config.scene_bounds[0],
            self.config.scene_bounds[3],
            self.config.grid_size[0],
        )
        y = np.linspace(
            self.config.scene_bounds[1],
            self.config.scene_bounds[4],
            self.config.grid_size[1],
        )
        z = np.linspace(
            self.config.scene_bounds[2],
            self.config.scene_bounds[5],
            self.config.grid_size[2],
        )
        
        X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
        
        # 初始化为远距离
        combined_sdf = np.full(self.config.grid_size, float('inf'))
        
        # 组合多个对象
        for obj in objects:
            if obj['type'] == 'sphere':
                obj_data = self._generate_sphere_voxels(obj)
            elif obj['type'] == 'cube':
                obj_data = self._generate_cube_voxels(obj)
            else:
                continue
            
            # 使用最小值组合SDF（并集）
            combined_sdf = np.minimum(combined_sdf, obj_data['sdf'])
        
        # 处理无限值
        combined_sdf = np.where(
            combined_sdf == float,
        )
        
        occupancy = (combined_sdf <= 0).astype(np.float32)
        combined_sdf = np.clip(combined_sdf, self.config.sdf_clamp[0], self.config.sdf_clamp[1])
        
        return {
            'occupancy': occupancy, 'sdf': combined_sdf, 'coordinates': np.stack([X, Y, Z], axis=-1)
        }
    
    def _save_voxel_data(self, voxel_data: dict[str, np.ndarray], save_path: str) -> None:
        """保存体素数据"""
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        
        # 保存为npz格式
        np.savez_compressed(save_path, **voxel_data)
        logger.info(f"体素数据已保存到: {save_path}")
    
    def generate_training_samples(
        self,
        voxel_data: dict[str, np.ndarray],
        sampling_strategy: str = "stratified",
    ) -> dict[str, np.ndarray]:
        """从体素数据生成训练样本"""
        logger.info(f"使用{sampling_strategy}策略生成训练样本...")
        
        coordinates = voxel_data['coordinates']
        occupancy = voxel_data['occupancy']
        sdf = voxel_data.get('sdf', None)
        
        # 展平数据
        coords_flat = coordinates.reshape(-1, 3)
        occupancy_flat = occupancy.reshape(-1)
        
        if sdf is not None:
            sdf_flat = sdf.reshape(-1)
        
        # 根据策略采样
        if sampling_strategy == "uniform":
            indices = self._uniform_sampling(coords_flat, occupancy_flat)
        elif sampling_strategy == "stratified":
            indices = self._stratified_sampling(coords_flat, occupancy_flat)
        elif sampling_strategy == "surface":
            indices = self._surface_sampling(coords_flat, occupancy_flat)
        else:
            raise ValueError(f"未知采样策略: {sampling_strategy}")
        
        # 提取样本
        sample_coords = coords_flat[indices]
        sample_occupancy = occupancy_flat[indices]
        
        result = {
            'coordinates': sample_coords, 'occupancy': sample_occupancy
        }
        
        if sdf is not None:
            result['sdf'] = sdf_flat[indices]
        
        logger.info(f"生成了{len(sample_coords)}个训练样本")
        return result
    
    def _uniform_sampling(self, coords: np.ndarray, occupancy: np.ndarray) -> np.ndarray:
        """均匀采样"""
        n_samples = min(self.config.samples_per_tile, len(coords))
        return np.random.choice(len(coords), n_samples, replace=False)
    
    def _stratified_sampling(self, coords: np.ndarray, occupancy: np.ndarray) -> np.ndarray:
        """分层采样"""
        occupied_indices = np.where(occupancy > 0)[0]
        free_indices = np.where(occupancy == 0)[0]
        
        n_total = min(self.config.samples_per_tile, len(coords))
        n_occupied = min(int(n_total * self.config.surface_sampling_ratio), len(occupied_indices))
        n_free = n_total - n_occupied
        
        selected_indices = []
        
        if n_occupied > 0 and len(occupied_indices) > 0:
            selected_occupied = np.random.choice(occupied_indices, n_occupied, replace=False)
            selected_indices.extend(selected_occupied)
        
        if n_free > 0 and len(free_indices) > 0:
            selected_free = np.random.choice(free_indices, n_free, replace=False)
            selected_indices.extend(selected_free)
        
        return np.array(selected_indices)
    
    def _surface_sampling(self, coords: np.ndarray, occupancy: np.ndarray) -> np.ndarray:
        """表面采样（边界附近的点）"""
        # 这里简化实现，实际应该检测边界
        return self._stratified_sampling(coords, occupancy)
    
    def get_statistics(self) -> dict[str, Any]:
        """获取生成统计信息"""
        return self.stats.copy()
    
    def reset_statistics(self) -> None:
        """重置统计信息"""
        self.stats = {
            'total_samples': 0, 'processing_time': 0.0, 'memory_usage': 0.0
        } 