"""
合成场景生成器模块

用于生成复杂的合成场景，结合多种几何体和环境要素。
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, import logging
from .occupancy_generator import OccupancyGenerator
from .sdf_generator import SDFGenerator

logger = logging.getLogger(__name__)


class SyntheticSceneGenerator:
    """合成场景生成器"""
    
    def __init__(
        self,
        voxel_size: float = 1.0,
        scene_bounds: tuple[float,
        float,
        float,
        float,
        float,
        float] =,
    )
        """
        初始化合成场景生成器
        
        Args:
            voxel_size: 体素大小
            scene_bounds: 场景边界
        """
        self.voxel_size = voxel_size
        self.scene_bounds = scene_bounds
        
        # 创建子生成器
        self.occupancy_gen = OccupancyGenerator(voxel_size, scene_bounds)
        self.sdf_gen = SDFGenerator()
        
        logger.info(f"合成场景生成器初始化完成")
    
    def generate_urban_scene(
        self,
        num_buildings: int = 10,
        building_height_range: tuple[float,
        float] =,
    )
        """
        生成城市场景
        
        Args:
            num_buildings: 建筑数量
            building_height_range: 建筑高度范围
            building_size_range: 建筑尺寸范围
            
        Returns:
            场景数据字典
        """
        buildings = []
        
        for i in range(num_buildings):
            # 随机生成建筑位置
            x = np.random.uniform(self.scene_bounds[0] + 20, self.scene_bounds[3] - 20)
            y = np.random.uniform(self.scene_bounds[1] + 20, self.scene_bounds[4] - 20)
            z = np.random.uniform(building_height_range[0], building_height_range[1]) / 2
            
            # 随机生成建筑尺寸
            width = np.random.uniform(building_size_range[0], building_size_range[1])
            depth = np.random.uniform(building_size_range[0], building_size_range[1])
            height = np.random.uniform(building_height_range[0], building_height_range[1])
            
            building_config = {
                'type': 'box', 'params': {
                    'center': (x, y, z), 'size': (width, depth, height), 'filled': True
                }
            }
            
            buildings.append(building_config)
        
        # 生成场景
        scene_occupancy = self.occupancy_gen.generate_complex_scene(buildings, 'union')
        
        logger.info(f"生成城市场景，建筑数: {num_buildings}")
        
        return {
            'occupancy': scene_occupancy, 'buildings': buildings, 'scene_type': 'urban'
        }
    
    def generate_forest_scene(
        self,
        num_trees: int = 20,
        tree_height_range: tuple[float,
        float] =,
    )
        """
        生成森林场景
        
        Args:
            num_trees: 树木数量
            tree_height_range: 树木高度范围
            tree_radius_range: 树木半径范围
            
        Returns:
            场景数据字典
        """
        trees = []
        
        for i in range(num_trees):
            # 随机生成树木位置
            x = np.random.uniform(self.scene_bounds[0] + 10, self.scene_bounds[3] - 10)
            y = np.random.uniform(self.scene_bounds[1] + 10, self.scene_bounds[4] - 10)
            
            # 随机生成树木参数
            height = np.random.uniform(tree_height_range[0], tree_height_range[1])
            radius = np.random.uniform(tree_radius_range[0], tree_radius_range[1])
            z = height / 2
            
            tree_config = {
                'type': 'cylinder', 'params': {
                    'center': (
                        x,
                        y,
                        z,
                    )
                }
            }
            
            trees.append(tree_config)
        
        # 生成场景
        scene_occupancy = self.occupancy_gen.generate_complex_scene(trees, 'union')
        
        logger.info(f"生成森林场景，树木数: {num_trees}")
        
        return {
            'occupancy': scene_occupancy, 'trees': trees, 'scene_type': 'forest'
        }
    
    def generate_mixed_scene(self, scene_configs: list[Dict]) -> dict[str, np.ndarray]:
        """
        生成混合场景
        
        Args:
            scene_configs: 场景配置列表
            
        Returns:
            场景数据字典
        """
        all_objects = []
        scene_types = []
        
        for config in scene_configs:
            scene_type = config.get('type', 'urban')
            
            if scene_type == 'urban':
                urban_scene = self.generate_urban_scene(**config.get('params', {}))
                all_objects.extend(urban_scene['buildings'])
                scene_types.append('urban')
            elif scene_type == 'forest':
                forest_scene = self.generate_forest_scene(**config.get('params', {}))
                all_objects.extend(forest_scene['trees'])
                scene_types.append('forest')
        
        # 生成复合场景
        scene_occupancy = self.occupancy_gen.generate_complex_scene(all_objects, 'union')
        
        logger.info(f"生成混合场景，场景类型: {scene_types}")
        
        return {
            'occupancy': scene_occupancy, 'objects': all_objects, 'scene_types': scene_types, 'scene_type': 'mixed'
        }
    
    def add_noise_to_scene(self, occupancy: np.ndarray, noise_level: float = 0.01) -> np.ndarray:
        """
        向场景添加噪声
        
        Args:
            occupancy: 占用网格
            noise_level: 噪声级别
            
        Returns:
            添加噪声后的占用网格
        """
        noise = np.random.normal(0, noise_level, occupancy.shape)
        noisy_occupancy = np.clip(occupancy + noise, 0, 1)
        
        return noisy_occupancy.astype(np.float32)
    
    def save_scene(self, scene_data: Dict, output_path: str):
        """
        保存场景数据
        
        Args:
            scene_data: 场景数据
            output_path: 输出路径
        """
        # 保存占用网格
        self.occupancy_gen.save_occupancy_grid(
            scene_data['occupancy'], output_path, metadata={
                'scene_type': scene_data.get(
                    'scene_type',
                    'unknown',
                )
            }
        )
        
        logger.info(f"场景已保存: {output_path}")
    
    def load_scene(self, filepath: str) -> dict[str, np.ndarray]:
        """
        加载场景数据
        
        Args:
            filepath: 文件路径
            
        Returns:
            场景数据字典
        """
        occupancy, metadata = self.occupancy_gen.load_occupancy_grid(filepath)
        
        return {
            'occupancy': occupancy, 'metadata': metadata
        } 