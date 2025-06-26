"""
Data Generators for NeuralVDB

This module contains data generation tools for creating
synthetic volumetric datasets for training and testing.
"""

import numpy as np
import json
import os
from typing import Dict, List, Optional, Tuple, Any
import random
import logging

logger = logging.getLogger(__name__)


class TileCityGenerator:
    """瓦片城市生成器 - 用于生成大规模城市场景"""
    
    def __init__(
        self,
        city_size: tuple[int,
        int,
        int] =,
    )
        """
        初始化瓦片城市生成器
        
        Args:
            city_size: 总城市尺寸 (x, y, z) 单位：米
            tile_size: 瓦片尺寸 (x, y) 单位：米
            voxel_size: 体素大小（米）
            output_dir: 输出目录
        """
        self.city_size = city_size
        self.tile_size = tile_size
        self.voxel_size = voxel_size
        self.output_dir = output_dir
        
        # 计算体素网格尺寸
        self.grid_size = (
            int(
                tile_size[0] / voxel_size,
            )
        )
        
        # 计算瓦片数量
        self.tiles_x = city_size[0] // tile_size[0]
        self.tiles_y = city_size[1] // tile_size[1]
        
        # 创建输出目录
        os.makedirs(self.output_dir, exist_ok=True)
        
        logger.info(f"瓦片城市生成器初始化完成")
        logger.info(f"城市尺寸: {city_size}, 瓦片尺寸: {tile_size}")
        logger.info(f"瓦片数量: {self.tiles_x} x {self.tiles_y} = {self.tiles_x * self.tiles_y}")
        logger.info(f"体素网格尺寸: {self.grid_size}")
    
    def create_building(
        self,
        center: tuple[float,
        float,
        float],
        size: tuple[float,
        float,
        float],
        building_type: str = 'residential',
        tile_origin: tuple[float,
        float] =,
    )
        """
        创建建筑物体素
        
        Args:
            center: 建筑中心 (x, y, z) 全局坐标
            size: 建筑尺寸 (width, length, height)
            building_type: 建筑类型
            tile_origin: 瓦片左下角全局坐标
            
        Returns:
            建筑体素网格（瓦片内坐标）
        """
        # 转换为瓦片内体素坐标
        center_voxel = (
            int(
            )
        )
        
        size_voxel = (
            int(
                size[0] / self.voxel_size,
            )
        )
        
        # 创建建筑体素网格
        building = np.zeros(self.grid_size, dtype=np.float32)
        
        # 计算建筑边界
        x1 = max(0, center_voxel[0] - size_voxel[0]//2)
        y1 = max(0, center_voxel[1] - size_voxel[1]//2)
        z1 = max(0, center_voxel[2] - size_voxel[2]//2)
        
        x2 = min(self.grid_size[0], center_voxel[0] + size_voxel[0]//2)
        y2 = min(self.grid_size[1], center_voxel[1] + size_voxel[1]//2)
        z2 = min(self.grid_size[2], center_voxel[2] + size_voxel[2]//2)
        
        # 填充建筑体素
        if building_type == 'residential':
            # 住宅建筑 - 实心
            building[x1:x2, y1:y2, z1:z2] = 1.0
        elif building_type == 'commercial':
            # 商业建筑 - 有内部空间
            building[x1:x2, y1:y2, z1:z2] = 1.0
            # 挖空内部（保留外壳）
            if x2-x1 > 4 and y2-y1 > 4 and z2-z1 > 2:
                building[x1+2:x2-2, y1+2:y2-2, z1+1:z2-1] = 0.3
        elif building_type == 'industrial':
            # 工业建筑 - 较低密度
            building[x1:x2, y1:y2, z1:z2] = 0.8
        else:
            # 默认建筑
            building[x1:x2, y1:y2, z1:z2] = 1.0
        
        return building
    
    def create_road_network(self, tile_origin: tuple[float, float]) -> np.ndarray:
        """
        创建瓦片内道路网络
        
        Args:
            tile_origin: 瓦片左下角全局坐标
            
        Returns:
            道路体素网格
        """
        roads = np.zeros(self.grid_size, dtype=np.float32)
        road_width = max(1, int(8 / self.voxel_size))  # 8米宽的道路
        road_height = max(1, int(2 / self.voxel_size))  # 2米高的道路
        
        # 主要道路间距
        major_road_spacing = int(200 / self.voxel_size)  # 每200米一条主路
        minor_road_spacing = int(100 / self.voxel_size)  # 每100米一条小路
        
        # 东西向道路
        for y in range(0, self.grid_size[1], major_road_spacing):
            y_start = max(0, y - road_width//2)
            y_end = min(self.grid_size[1], y + road_width//2)
            roads[:, y_start:y_end, :road_height] = 0.2  # 道路有低占用率
        
        # 南北向道路
        for x in range(0, self.grid_size[0], major_road_spacing):
            x_start = max(0, x - road_width//2)
            x_end = min(self.grid_size[0], x + road_width//2)
            roads[x_start:x_end, :, :road_height] = 0.2
        
        # 次要道路
        minor_road_width = max(1, road_width // 2)
        for y in range(minor_road_spacing//2, self.grid_size[1], minor_road_spacing):
            y_start = max(0, y - minor_road_width//2)
            y_end = min(self.grid_size[1], y + minor_road_width//2)
            roads[:, y_start:y_end, :road_height] = 0.1
        
        for x in range(minor_road_spacing//2, self.grid_size[0], minor_road_spacing):
            x_start = max(0, x - minor_road_width//2)
            x_end = min(self.grid_size[0], x + minor_road_width//2)
            roads[x_start:x_end, :, :road_height] = 0.1
        
        return roads
    
    def create_vegetation(self, tile_origin: tuple[float, float]) -> np.ndarray:
        """
        创建植被（公园、绿地等）
        
        Args:
            tile_origin: 瓦片左下角全局坐标
            
        Returns:
            植被体素网格
        """
        vegetation = np.zeros(self.grid_size, dtype=np.float32)
        
        # 随机生成一些公园区域
        num_parks = random.randint(1, 3)
        
        for _ in range(num_parks):
            # 随机公园位置和大小
            park_x = random.randint(50, self.grid_size[0] - 100)
            park_y = random.randint(50, self.grid_size[1] - 100)
            park_w = random.randint(30, 80)
            park_h = random.randint(30, 80)
            
            # 确保不超出边界
            park_w = min(park_w, self.grid_size[0] - park_x)
            park_h = min(park_h, self.grid_size[1] - park_y)
            
            # 树木高度
            tree_height = int(random.uniform(8, 15) / self.voxel_size)
            tree_height = min(tree_height, self.grid_size[2])
            
            # 填充植被
            vegetation[park_x:park_x+park_w, park_y:park_y+park_h, :tree_height] = 0.6
        
        return vegetation
    
    def generate_tile(
        self,
        tile_x: int,
        tile_y: int,
        buildings: list[Dict],
    )
        """
        生成单个瓦片的体素网格和建筑信息
        
        Args:
            tile_x, tile_y: 瓦片索引
            buildings: 所有建筑信息（全局坐标）
            
        Returns:
            (瓦片体素网格, 瓦片内建筑信息)
        """
        tile_origin = (tile_x * self.tile_size[0], tile_y * self.tile_size[1])
        tile_grid = np.zeros(self.grid_size, dtype=np.float32)
        tile_buildings = []
        
        # 筛选属于本瓦片的建筑
        for building in buildings:
            bx, by, bz = building['center']
            if (tile_origin[0] <= bx < tile_origin[0] + self.tile_size[0] and
                tile_origin[1] <= by < tile_origin[1] + self.tile_size[1]):
                
                # 创建建筑体素
                building_voxels = self.create_building(
                    building['center'], building['size'], building.get(
                        'type',
                        'residential',
                    )
                )
                
                # 添加到瓦片网格
                tile_grid = np.maximum(tile_grid, building_voxels)
                tile_buildings.append(building)
        
        # 添加道路网络
        roads = self.create_road_network(tile_origin)
        tile_grid = np.maximum(tile_grid, roads)
        
        # 添加植被
        vegetation = self.create_vegetation(tile_origin)
        tile_grid = np.maximum(tile_grid, vegetation)
        
        return tile_grid, tile_buildings
    
    def generate_global_buildings(self, density: str = 'medium') -> list[Dict]:
        """
        生成全局建筑信息
        
        Args:
            density: 建筑密度 ('low', 'medium', 'high')
            
        Returns:
            所有建筑信息（全局坐标）
        """
        # 根据密度确定每个瓦片的建筑数量
        if density == 'low':
            buildings_per_tile = random.randint(10, 20)
        elif density == 'medium':
            buildings_per_tile = random.randint(20, 40)
        else:  # high
            buildings_per_tile = random.randint(40, 80)
        
        buildings = []
        building_id = 0
        
        for tx in range(self.tiles_x):
            for ty in range(self.tiles_y):
                tile_origin = (tx * self.tile_size[0], ty * self.tile_size[1])
                
                # 为每个瓦片生成建筑
                n_buildings = random.randint(buildings_per_tile//2, buildings_per_tile)
                
                for i in range(n_buildings):
                    # 建筑位置（避免边界）
                    margin = 20  # 20米边界
                    x = random.uniform(
                        tile_origin[0] + margin,
                        tile_origin[0] + self.tile_size[0] - margin,
                    )
                    y = random.uniform(
                        tile_origin[1] + margin,
                        tile_origin[1] + self.tile_size[1] - margin,
                    )
                    z = 0  # 地面高度
                    
                    # 建筑类型和尺寸
                    building_type = random.choice(['residential', 'commercial', 'industrial'])
                    
                    if building_type == 'residential':
                        width = random.uniform(10, 25)
                        length = random.uniform(10, 25)
                        height = random.uniform(15, 40)
                    elif building_type == 'commercial':
                        width = random.uniform(20, 50)
                        length = random.uniform(20, 50)
                        height = random.uniform(20, 80)
                    else:  # industrial
                        width = random.uniform(30, 80)
                        length = random.uniform(30, 80)
                        height = random.uniform(8, 25)
                    
                    buildings.append({
                        'id': f"building_{
                            building_id,
                        }
                        'size': (width, length, height), 'tile_index': (tx, ty)
                    })
                    
                    building_id += 1
        
        logger.info(f"生成了 {len(buildings)} 个建筑，密度: {density}")
        return buildings
    
    def save_tile(
        self,
        tile_grid: np.ndarray,
        tile_buildings: list[Dict],
        tile_x: int,
        tile_y: int,
    )
        """
        保存瓦片体素和元数据
        
        Args:
            tile_grid: 瓦片体素网格
            tile_buildings: 瓦片内建筑信息
            tile_x, tile_y: 瓦片索引
        """
        # 保存体素数据
        npy_path = os.path.join(self.output_dir, f"tile_{tile_x}_{tile_y}.npy")
        np.save(npy_path, tile_grid)
        
        # 保存元数据
        json_path = os.path.join(self.output_dir, f"tile_{tile_x}_{tile_y}.json")
        metadata = {
            'tile_index': (
                tile_x,
                tile_y,
            )
                'total_voxels': tile_grid.size, 'occupied_voxels': np.count_nonzero(
                    tile_grid,
                )
            }
        }
        
        with open(json_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        logger.info(f"保存瓦片: {npy_path}, {json_path}")
    
    def generate_and_save_all_tiles(self, density: str = 'medium'):
        """
        生成并保存所有瓦片
        
        Args:
            density: 建筑密度
        """
        logger.info(f"开始生成城市瓦片，密度: {density}")
        
        # 生成全局建筑
        buildings = self.generate_global_buildings(density=density)
        
        # 生成每个瓦片
        total_tiles = self.tiles_x * self.tiles_y
        processed = 0
        
        for tx in range(self.tiles_x):
            for ty in range(self.tiles_y):
                logger.info(f"生成瓦片 ({tx}, {ty}) [{processed+1}/{total_tiles}]...")
                
                tile_grid, tile_buildings = self.generate_tile(tx, ty, buildings)
                self.save_tile(tile_grid, tile_buildings, tx, ty)
                
                processed += 1
        
        # 保存全局信息
        global_info = {
            'city_size': self.city_size, 'tile_size': self.tile_size, 'voxel_size': self.voxel_size, 'tiles_x': self.tiles_x, 'tiles_y': self.tiles_y, 'total_tiles': total_tiles, 'total_buildings': len(
                buildings,
            )
        }
        
        global_info_path = os.path.join(self.output_dir, 'city_info.json')
        with open(global_info_path, 'w') as f:
            json.dump(global_info, f, indent=2)
        
        logger.info(f"城市生成完成！共 {total_tiles} 个瓦片，{len(buildings)} 个建筑")


class SimpleVDBGenerator:
    """简单VDB生成器 - 用于生成基础几何体"""
    
    def __init__(self, grid_size: tuple[int, int, int] = (128, 128, 128), voxel_size: float = 1.0):
        """
        初始化简单VDB生成器
        
        Args:
            grid_size: 网格尺寸
            voxel_size: 体素大小
        """
        self.grid_size = grid_size
        self.voxel_size = voxel_size
        
        logger.info(f"简单VDB生成器初始化完成，网格尺寸: {grid_size}")
    
    def create_sphere(
        self,
        center: tuple[float,
        float,
        float],
        radius: float,
        density: float = 1.0,
    )
        """
        创建球体
        
        Args:
            center: 球心坐标
            radius: 半径
            density: 密度值
            
        Returns:
            球体体素网格
        """
        grid = np.zeros(self.grid_size, dtype=np.float32)
        
        # 创建坐标网格
        x, y, z = np.meshgrid(
            np.arange(
                self.grid_size[0],
            )
        )
        
        # 计算到球心的距离
        distances = np.sqrt((x - center[0])**2 + (y - center[1])**2 + (z - center[2])**2)
        
        # 创建球体
        sphere_mask = distances <= radius
        grid[sphere_mask] = density
        
        return grid
    
    def create_box(
        self,
        center: tuple[float,
        float,
        float],
        size: tuple[float,
        float,
        float],
        density: float = 1.0,
    )
        """
        创建立方体
        
        Args:
            center: 中心坐标
            size: 尺寸 (width, height, depth)
            density: 密度值
            
        Returns:
            立方体体素网格
        """
        grid = np.zeros(self.grid_size, dtype=np.float32)
        
        # 计算立方体边界
        half_size = np.array(size) / 2
        min_coords = np.array(center) - half_size
        max_coords = np.array(center) + half_size
        
        # 转换为体素坐标
        min_voxel = np.floor(min_coords / self.voxel_size).astype(int)
        max_voxel = np.ceil(max_coords / self.voxel_size).astype(int)
        
        # 确保在网格范围内
        min_voxel = np.maximum(min_voxel, 0)
        max_voxel = np.minimum(max_voxel, self.grid_size)
        
        # 填充立方体
        grid[min_voxel[0]:max_voxel[0], min_voxel[1]:max_voxel[1], min_voxel[2]:max_voxel[2]] = density
        
        return grid
    
    def create_cylinder(
        self,
        center: tuple[float,
        float,
        float],
        radius: float,
        height: float,
        axis: int = 2,
        density: float = 1.0,
    )
        """
        创建圆柱体
        
        Args:
            center: 中心坐标
            radius: 半径
            height: 高度
            axis: 圆柱轴向 (0=x, 1=y, 2=z)
            density: 密度值
            
        Returns:
            圆柱体体素网格
        """
        grid = np.zeros(self.grid_size, dtype=np.float32)
        
        # 创建坐标网格
        x, y, z = np.meshgrid(
            np.arange(
                self.grid_size[0],
            )
        )
        
        coords = [x, y, z]
        
        # 计算到轴的距离
        if axis == 0:  # x轴
            radial_dist = np.sqrt((coords[1] - center[1])**2 + (coords[2] - center[2])**2)
            axial_dist = np.abs(coords[0] - center[0])
        elif axis == 1:  # y轴
            radial_dist = np.sqrt((coords[0] - center[0])**2 + (coords[2] - center[2])**2)
            axial_dist = np.abs(coords[1] - center[1])
        else:  # z轴
            radial_dist = np.sqrt((coords[0] - center[0])**2 + (coords[1] - center[1])**2)
            axial_dist = np.abs(coords[2] - center[2])
        
        # 创建圆柱体
        cylinder_mask = (radial_dist <= radius) & (axial_dist <= height/2)
        grid[cylinder_mask] = density
        
        return grid
    
    def create_torus(
        self,
        center: tuple[float,
        float,
        float],
        major_radius: float,
        minor_radius: float,
        axis: int = 2,
        density: float = 1.0,
    )
        """
        创建环形体
        
        Args:
            center: 中心坐标
            major_radius: 主半径
            minor_radius: 次半径
            axis: 环形轴向 (0=x, 1=y, 2=z)
            density: 密度值
            
        Returns:
            环形体体素网格
        """
        grid = np.zeros(self.grid_size, dtype=np.float32)
        
        # 创建坐标网格
        x, y, z = np.meshgrid(
            np.arange(
                self.grid_size[0],
            )
        )
        
        # 相对坐标
        dx = x - center[0]
        dy = y - center[1]
        dz = z - center[2]
        
        if axis == 0:  # x轴
            # 到主环的距离
            R = np.sqrt(dy**2 + dz**2)
            # 环形距离
            torus_dist = np.sqrt((R - major_radius)**2 + dx**2)
        elif axis == 1:  # y轴
            R = np.sqrt(dx**2 + dz**2)
            torus_dist = np.sqrt((R - major_radius)**2 + dy**2)
        else:  # z轴
            R = np.sqrt(dx**2 + dy**2)
            torus_dist = np.sqrt((R - major_radius)**2 + dz**2)
        
        # 创建环形体
        torus_mask = torus_dist <= minor_radius
        grid[torus_mask] = density
        
        return grid
    
    def create_complex_scene(self, scene_type: str = 'mixed') -> np.ndarray:
        """
        创建复杂场景
        
        Args:
            scene_type: 场景类型 ('mixed', 'architectural', 'organic')
            
        Returns:
            场景体素网格
        """
        grid = np.zeros(self.grid_size, dtype=np.float32)
        
        # 计算场景中心和尺度
        center = np.array(self.grid_size) * self.voxel_size / 2
        scale = min(self.grid_size) * self.voxel_size / 4
        
        if scene_type == 'mixed':
            # 混合几何体
            # 中心球体
            sphere = self.create_sphere(center, scale * 0.3, 1.0)
            grid = np.maximum(grid, sphere)
            
            # 立方体
            box_center = center + np.array([scale * 0.8, 0, 0])
            box = self.create_box(box_center, (scale * 0.4, scale * 0.4, scale * 0.6), 0.8)
            grid = np.maximum(grid, box)
            
            # 圆柱体
            cyl_center = center + np.array([0, scale * 0.8, 0])
            cylinder = self.create_cylinder(
                cyl_center,
                scale * 0.2,
                scale * 0.8,
                axis=2,
                density=0.6,
            )
            grid = np.maximum(grid, cylinder)
            
        elif scene_type == 'architectural':
            # 建筑风格
            # 主建筑
            main_building = self.create_box(center, (scale * 0.6, scale * 0.6, scale * 1.2), 1.0)
            grid = np.maximum(grid, main_building)
            
            # 塔楼
            tower_center = center + np.array([scale * 0.4, scale * 0.4, scale * 0.3])
            tower = self.create_cylinder(
                tower_center,
                scale * 0.15,
                scale * 0.9,
                axis=2,
                density=0.9,
            )
            grid = np.maximum(grid, tower)
            
            # 附属建筑
            annex_center = center + np.array([-scale * 0.5, 0, -scale * 0.2])
            annex = self.create_box(annex_center, (scale * 0.3, scale * 0.4, scale * 0.4), 0.8)
            grid = np.maximum(grid, annex)
            
        elif scene_type == 'organic':
            # 有机形状
            # 主体
            main_sphere = self.create_sphere(center, scale * 0.4, 1.0)
            grid = np.maximum(grid, main_sphere)
            
            # 环形结构
            torus = self.create_torus(center, scale * 0.6, scale * 0.1, axis=2, density=0.7)
            grid = np.maximum(grid, torus)
            
            # 分支
            for i in range(5):
                angle = i * 2 * np.pi / 5
                branch_center = center + scale * 0.5 * np.array([np.cos(angle), np.sin(angle), 0])
                branch = self.create_sphere(branch_center, scale * 0.15, 0.6)
                grid = np.maximum(grid, branch)
        
        # 添加噪声
        noise = np.random.rand(*self.grid_size) * 0.1
        grid = np.clip(grid + noise, 0, 1)
        
        return grid
    
    def save_vdb(self, grid: np.ndarray, filepath: str, metadata: Optional[Dict] = None):
        """
        保存VDB数据
        
        Args:
            grid: 体素网格
            filepath: 保存路径
            metadata: 元数据
        """
        # 保存numpy数组
        np.save(filepath, grid)
        
        # 保存元数据
        if metadata is None:
            metadata = {}
        
        metadata.update({
            'grid_size': self.grid_size, 'voxel_size': self.voxel_size, 'total_voxels': grid.size, 'occupied_voxels': np.count_nonzero(
                grid,
            )
        })
        
        metadata_path = filepath.replace('.npy', '_metadata.json')
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        logger.info(f"VDB数据已保存到: {filepath}")
        logger.info(f"元数据已保存到: {metadata_path}")


def generate_sample_dataset(
    output_dir: str,
    num_samples: int = 1000,
    grid_size: tuple[int,
    int,
    int] =,
)
    """
    生成示例数据集
    
    Args:
        output_dir: 输出目录
        num_samples: 样本数量
        grid_size: 网格尺寸
        scene_types: 场景类型列表
    """
    os.makedirs(output_dir, exist_ok=True)
    
    generator = SimpleVDBGenerator(grid_size=grid_size)
    
    logger.info(f"开始生成示例数据集，样本数: {num_samples}")
    
    for i in range(num_samples):
        # 随机选择场景类型
        scene_type = random.choice(scene_types)
        
        # 生成场景
        grid = generator.create_complex_scene(scene_type)
        
        # 提取点和占用值
        indices = np.where(grid > 0)
        coords = np.column_stack(indices).astype(np.float32)
        occupancies = grid[indices]
        
        # 添加一些空白点
        num_empty = len(coords) // 2
        empty_coords = np.random.rand(num_empty, 3) * np.array(grid_size)
        empty_occupancies = np.zeros(num_empty, dtype=np.float32)
        
        # 合并数据
        all_coords = np.vstack([coords, empty_coords])
        all_occupancies = np.hstack([occupancies, empty_occupancies])
        
        # 保存样本
        sample_path = os.path.join(output_dir, f'sample_{i:04d}.npz')
        np.savez_compressed(
            sample_path,
            coords=all_coords,
            occupancies=all_occupancies,
            scene_type=scene_type,
        )
        
        if (i + 1) % 100 == 0:
            logger.info(f"已生成 {i + 1}/{num_samples} 个样本")
    
    # 保存数据集信息
    dataset_info = {
        'num_samples': num_samples, 'grid_size': grid_size, 'scene_types': scene_types, 'total_files': num_samples, 'file_format': 'npz', 'data_fields': ['coords', 'occupancies', 'scene_type']
    }
    
    info_path = os.path.join(output_dir, 'dataset_info.json')
    with open(info_path, 'w') as f:
        json.dump(dataset_info, f, indent=2)
    
    logger.info(f"示例数据集生成完成，保存在: {output_dir}") 