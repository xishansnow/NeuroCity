"""
占用网格生成器模块

用于生成占用网格数据，支持多种几何体和复杂场景。
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, import logging
import os

logger = logging.getLogger(__name__)


class OccupancyGenerator:
    """占用网格生成器"""
    
    def __init__(
        self,
        voxel_size: float = 1.0,
        grid_bounds: tuple[float,
        float,
        float,
        float,
        float,
        float] =,
    )
        """
        初始化占用网格生成器
        
        Args:
            voxel_size: 体素大小（米）
            grid_bounds: 网格边界 (xmin, ymin, zmin, xmax, ymax, zmax)
        """
        self.voxel_size = voxel_size
        self.grid_bounds = grid_bounds
        
        # 计算网格尺寸
        self.grid_size = tuple(
            int(np.ceil((grid_bounds[i+3] - grid_bounds[i]) / voxel_size))
            for i in range(3)
        )
        
        logger.info(f"占用网格生成器初始化完成，网格尺寸: {self.grid_size}")
    
    def generate_sphere_occupancy(
        self,
        center: tuple[float,
        float,
        float],
        radius: float,
        filled: bool = True,
    )
        """
        生成球体占用网格
        
        Args:
            center: 球心坐标
            radius: 半径
            filled: 是否填充（True）或仅表面（False）
            
        Returns:
            占用网格 [nx, ny, nz]
        """
        # 创建坐标网格
        x = np.linspace(self.grid_bounds[0], self.grid_bounds[3], self.grid_size[0])
        y = np.linspace(self.grid_bounds[1], self.grid_bounds[4], self.grid_size[1])
        z = np.linspace(self.grid_bounds[2], self.grid_bounds[5], self.grid_size[2])
        
        X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
        
        # 计算到球心的距离
        distances = np.sqrt(
            (X - center[0])**2 + 
            (Y - center[1])**2 + 
            (Z - center[2])**2
        )
        
        if filled:
            # 填充球体
            occupancy = (distances <= radius).astype(np.float32)
        else:
            # 仅表面
            thickness = self.voxel_size
            occupancy = ((distances <= radius + thickness/2) & 
                        (distances >= radius - thickness/2)).astype(np.float32)
        
        logger.info(f"生成球体占用网格，中心: {center}, 半径: {radius}, 占用体素数: {np.sum(occupancy)}")
        return occupancy
    
    def generate_box_occupancy(
        self,
        center: tuple[float,
        float,
        float],
        size: tuple[float,
        float,
        float],
        filled: bool = True,
    )
        """
        生成盒子占用网格
        
        Args:
            center: 盒子中心
            size: 盒子尺寸 (width, height, depth)
            filled: 是否填充
            
        Returns:
            占用网格
        """
        # 创建坐标网格
        x = np.linspace(self.grid_bounds[0], self.grid_bounds[3], self.grid_size[0])
        y = np.linspace(self.grid_bounds[1], self.grid_bounds[4], self.grid_size[1])
        z = np.linspace(self.grid_bounds[2], self.grid_bounds[5], self.grid_size[2])
        
        X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
        
        # 计算盒子边界
        half_size = np.array(size) / 2
        x_min, x_max = center[0] - half_size[0], center[0] + half_size[0]
        y_min, y_max = center[1] - half_size[1], center[1] + half_size[1]
        z_min, z_max = center[2] - half_size[2], center[2] + half_size[2]
        
        if filled:
            # 填充盒子
            occupancy = ((X >= x_min) & (X <= x_max) &
                        (Y >= y_min) & (Y <= y_max) &
                        (Z >= z_min) & (Z <= z_max)).astype(np.float32)
        else:
            # 仅表面
            thickness = self.voxel_size
            outer_mask = ((X >= x_min - thickness/2) & (X <= x_max + thickness/2) &
                         (Y >= y_min - thickness/2) & (Y <= y_max + thickness/2) &
                         (Z >= z_min - thickness/2) & (Z <= z_max + thickness/2))
            inner_mask = ((X >= x_min + thickness/2) & (X <= x_max - thickness/2) &
                         (Y >= y_min + thickness/2) & (Y <= y_max - thickness/2) &
                         (Z >= z_min + thickness/2) & (Z <= z_max - thickness/2))
            occupancy = (outer_mask & ~inner_mask).astype(np.float32)
        
        logger.info(f"生成盒子占用网格，中心: {center}, 尺寸: {size}, 占用体素数: {np.sum(occupancy)}")
        return occupancy
    
    def generate_cylinder_occupancy(
        self,
        center: tuple[float,
        float,
        float],
        radius: float,
        height: float,
        axis: int = 2,
        filled: bool = True,
    ) -> np.ndarray:
        """
        生成圆柱体占用网格
        
        Args:
            center: 圆柱体中心
            radius: 半径
            height: 高度
            axis: 轴向 (0=X, 1=Y, 2=Z)
            filled: 是否填充
            
        Returns:
            占用网格
        """
        # 创建坐标网格
        x = np.linspace(self.grid_bounds[0], self.grid_bounds[3], self.grid_size[0])
        y = np.linspace(self.grid_bounds[1], self.grid_bounds[4], self.grid_size[1])
        z = np.linspace(self.grid_bounds[2], self.grid_bounds[5], self.grid_size[2])
        
        X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
        
        # 根据轴向计算径向距离和轴向距离
        if axis == 2:  # Z轴
            radial_dist = np.sqrt((X - center[0])**2 + (Y - center[1])**2)
            axial_dist = np.abs(Z - center[2])
        elif axis == 1:  # Y轴
            radial_dist = np.sqrt((X - center[0])**2 + (Z - center[2])**2)
            axial_dist = np.abs(Y - center[1])
        else:  # X轴
            radial_dist = np.sqrt((Y - center[1])**2 + (Z - center[2])**2)
            axial_dist = np.abs(X - center[0])
        
        if filled:
            # 填充圆柱体
            occupancy = ((radial_dist <= radius) & 
                        (axial_dist <= height/2)).astype(np.float32)
        else:
            # 仅表面
            thickness = self.voxel_size
            # 侧面
            side_mask = ((radial_dist <= radius + thickness/2) & 
                        (radial_dist >= radius - thickness/2) & 
                        (axial_dist <= height/2))
            # 顶面和底面
            top_bottom_mask = ((radial_dist <= radius) & 
                              (axial_dist >= height/2 - thickness/2) & 
                              (axial_dist <= height/2 + thickness/2))
            occupancy = (side_mask | top_bottom_mask).astype(np.float32)
        
        logger.info(f"生成圆柱体占用网格，中心: {center}, 半径: {radius}, 高度: {height}, 占用体素数: {np.sum(occupancy)}")
        return occupancy
    
    def generate_complex_scene(
        self,
        objects: list[Dict],
        combination_mode: str = 'union',
    )
        """
        生成复杂场景占用网格
        
        Args:
            objects: 对象列表，每个对象包含类型和参数
            combination_mode: 组合模式 ('union', 'intersection', 'difference')
            
        Returns:
            复合占用网格
        """
        if not objects:
            return np.zeros(self.grid_size, dtype=np.float32)
        
        # 生成第一个对象
        first_obj = objects[0]
        result = self._generate_single_object(first_obj)
        
        # 组合其他对象
        for obj in objects[1:]:
            obj_occupancy = self._generate_single_object(obj)
            
            if combination_mode == 'union':
                result = np.maximum(result, obj_occupancy)
            elif combination_mode == 'intersection':
                result = np.minimum(result, obj_occupancy)
            elif combination_mode == 'difference':
                result = np.maximum(result - obj_occupancy, 0)
        
        logger.info(f"生成复杂场景，对象数: {len(objects)}, 总占用体素数: {np.sum(result)}")
        return result
    
    def _generate_single_object(self, obj_config: Dict) -> np.ndarray:
        """生成单个对象的占用网格"""
        obj_type = obj_config['type']
        params = obj_config.get('params', {})
        
        if obj_type == 'sphere':
            return self.generate_sphere_occupancy(
                center=params.get(
                    'center',
                )
            )
        elif obj_type == 'box':
            return self.generate_box_occupancy(
                center=params.get(
                    'center',
                )
            )
        elif obj_type == 'cylinder':
            return self.generate_cylinder_occupancy(
                center=params.get(
                    'center',
                )
            )
        else:
            raise ValueError(f"未知对象类型: {obj_type}")
    
    def generate_random_scene(
        self,
        num_objects: int = 5,
        object_types: list[str] = ['sphere',
        'box',
        'cylinder'],
        size_range: tuple[float,
        float] =,
    ) -> tuple[np.ndarray, list[Dict]]:
        """
        生成随机场景
        
        Args:
            num_objects: 对象数量
            object_types: 对象类型列表
            size_range: 尺寸范围
            
        Returns:
            (占用网格, 对象配置列表)
        """
        objects = []
        
        for i in range(num_objects):
            # 随机选择对象类型
            obj_type = np.random.choice(object_types)
            
            # 随机生成位置
            center = (
                np.random.uniform(
                    self.grid_bounds[0] + 20,
                    self.grid_bounds[3] - 20,
                )
            )
            
            # 随机生成尺寸
            size = np.random.uniform(size_range[0], size_range[1])
            
            if obj_type == 'sphere':
                obj_config = {
                    'type': 'sphere', 'params': {
                        'center': center, 'radius': size, 'filled': True
                    }
                }
            elif obj_type == 'box':
                box_size = (size, size, size * 0.5)
                obj_config = {
                    'type': 'box', 'params': {
                        'center': center, 'size': box_size, 'filled': True
                    }
                }
            elif obj_type == 'cylinder':
                obj_config = {
                    'type': 'cylinder', 'params': {
                        'center': center, 'radius': size * 0.7, 'height': size * 1.5, 'axis': 2, 'filled': True
                    }
                }
            
            objects.append(obj_config)
        
        # 生成复合场景
        occupancy = self.generate_complex_scene(objects, 'union')
        
        logger.info(f"生成随机场景，对象数: {num_objects}, 总占用体素数: {np.sum(occupancy)}")
        return occupancy, objects
    
    def save_occupancy_grid(
        self,
        occupancy: np.ndarray,
        filepath: str,
        metadata: Optional[Dict] = None,
    ) -> None:
        """
        保存占用网格
        
        Args:
            occupancy: 占用网格
            filepath: 保存路径
            metadata: 元数据
        """
        # 保存numpy数组
        np.save(filepath, occupancy)
        
        # 保存元数据
        if metadata is None:
            metadata = {}
        
        metadata.update({
            'voxel_size': self.voxel_size, 'grid_bounds': self.grid_bounds, 'grid_size': self.grid_size, 'occupied_voxels': int(
                np.sum,
            )
        })
        
        metadata_path = filepath.replace('.npy', '.json')
        import json
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        logger.info(f"占用网格已保存: {filepath}")
    
    def load_occupancy_grid(self, filepath: str) -> tuple[np.ndarray, Dict]:
        """
        加载占用网格
        
        Args:
            filepath: 文件路径
            
        Returns:
            (占用网格, 元数据)
        """
        occupancy = np.load(filepath)
        
        # 加载元数据
        metadata_path = filepath.replace('.npy', '.json')
        metadata = {}
        if os.path.exists(metadata_path):
            import json
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
        
        logger.info(f"占用网格已加载: {filepath}")
        return occupancy, metadata 