#!/usr/bin/env python3
"""
生成小型城市建筑VDB测试集合
支持多种建筑类型和布局
"""

import numpy as np
import openvdb as vdb
from scipy.spatial import cKDTree
import os
import json
import random

class CityVDBGenerator:
    def __init__(self, city_size: tuple[int, int, int] = (1000, 1000, 200)):
        """
        初始化城市VDB生成器
        
        Args:
            city_size: 城市尺寸 (x, y, z) 单位：米
        """
        self.city_size = city_size
        self.voxel_size = 1.0  # 体素大小：1米
        self.grid_size = (
            int(
                city_size[0] / self.voxel_size,
            )
        )
        
    def create_building(
        self,
        center: tuple[float,
        float],
        size: tuple[float,
        float,
        float],
        building_type: str = "residential",
    )
        """
        创建单个建筑
        
        Args:
            center: 建筑中心坐标 (x, y)
            size: 建筑尺寸 (width, length, height)
            building_type: 建筑类型
            
        Returns:
            建筑的体素网格
        """
        # 转换为体素坐标
        center_voxel = (
            int(center[0] / self.voxel_size), int(center[1] / self.voxel_size)
        )
        size_voxel = (
            int(
                size[0] / self.voxel_size,
            )
        )
        
        # 创建建筑体素
        building = np.zeros(self.grid_size, dtype=np.float32)
        
        # 根据建筑类型调整形状
        if building_type == "residential":
            # 住宅建筑：矩形
            x1, y1 = center_voxel[0] - size_voxel[0]//2, center_voxel[1] - size_voxel[1]//2
            x2, y2 = center_voxel[0] + size_voxel[0]//2, center_voxel[1] + size_voxel[1]//2
            building[x1:x2, y1:y2, :size_voxel[2]] = 1.0
            
        elif building_type == "commercial":
            # 商业建筑：L形
            x1, y1 = center_voxel[0] - size_voxel[0]//2, center_voxel[1] - size_voxel[1]//2
            x2, y2 = center_voxel[0] + size_voxel[0]//2, center_voxel[1] + size_voxel[1]//2
            # 主体
            building[x1:x2, y1:y2, :size_voxel[2]] = 1.0
            # L形延伸
            building[x1:x1+size_voxel[0]//3, y2:y2+size_voxel[1]//3, :size_voxel[2]] = 1.0
            
        elif building_type == "industrial":
            # 工业建筑：大而低
            x1, y1 = center_voxel[0] - size_voxel[0]//2, center_voxel[1] - size_voxel[1]//2
            x2, y2 = center_voxel[0] + size_voxel[0]//2, center_voxel[1] + size_voxel[1]//2
            building[x1:x2, y1:y2, :size_voxel[2]//2] = 1.0
            
        elif building_type == "skyscraper":
            # 摩天大楼：高而细
            x1, y1 = center_voxel[0] - size_voxel[0]//4, center_voxel[1] - size_voxel[1]//4
            x2, y2 = center_voxel[0] + size_voxel[0]//4, center_voxel[1] + size_voxel[1]//4
            building[x1:x2, y1:y2, :size_voxel[2]] = 1.0
            
        return building
    
    def create_road_network(self) -> np.ndarray:
        """
        创建道路网络
        
        Returns:
            道路的体素网格
        """
        roads = np.zeros(self.grid_size, dtype=np.float32)
        
        # 主道路（东西向）
        for y in range(0, self.grid_size[1], 200):
            roads[:, y-5:y+5, :10] = 1.0
            
        # 主道路（南北向）
        for x in range(0, self.grid_size[0], 200):
            roads[x-5:x+5, :, :10] = 1.0
            
        # 次道路
        for y in range(100, self.grid_size[1], 200):
            roads[:, y-3:y+3, :8] = 1.0
            
        for x in range(100, self.grid_size[0], 200):
            roads[x-3:x+3, :, :8] = 1.0
            
        return roads
    
    def create_terrain(self) -> np.ndarray:
        """
        创建地形（地面）
        
        Returns:
            地形的体素网格
        """
        terrain = np.zeros(self.grid_size, dtype=np.float32)
        
        # 创建起伏的地形
        x_coords, y_coords = np.meshgrid(
            np.arange(self.grid_size[0]), np.arange(self.grid_size[1]), indexing='ij'
        )
        
        # 添加一些随机起伏
        height_map = np.random.normal(0, 2, self.grid_size[:2])
        height_map = np.clip(height_map, 0, 5)
        
        for i in range(self.grid_size[0]):
            for j in range(self.grid_size[1]):
                height = int(height_map[i, j])
                terrain[i, j, :height] = 1.0
                
        return terrain
    
    def generate_city_layout(self) -> list[dict]:
        """
        生成城市布局
        
        Returns:
            建筑信息列表
        """
        buildings = []
        
        # 住宅区
        for i in range(20):
            x = random.uniform(50, 450)
            y = random.uniform(50, 450)
            size = (
                random.uniform(15, 25), random.uniform(15, 25), random.uniform(20, 40)
            )
            buildings.append({
                'center': (x, y), 'size': size, 'type': 'residential'
            })
        
        # 商业区
        for i in range(8):
            x = random.uniform(550, 950)
            y = random.uniform(50, 450)
            size = (
                random.uniform(30, 50), random.uniform(30, 50), random.uniform(25, 45)
            )
            buildings.append({
                'center': (x, y), 'size': size, 'type': 'commercial'
            })
        
        # 工业区
        for i in range(5):
            x = random.uniform(50, 450)
            y = random.uniform(550, 950)
            size = (
                random.uniform(40, 80), random.uniform(40, 80), random.uniform(15, 30)
            )
            buildings.append({
                'center': (x, y), 'size': size, 'type': 'industrial'
            })
        
        # 摩天大楼
        for i in range(3):
            x = random.uniform(550, 950)
            y = random.uniform(550, 950)
            size = (
                random.uniform(20, 30), random.uniform(20, 30), random.uniform(60, 100)
            )
            buildings.append({
                'center': (x, y), 'size': size, 'type': 'skyscraper'
            })
            
        return buildings
    
    def generate_vdb(self, output_path: str = "test_city.vdb"):
        """
        生成完整的城市VDB文件
        
        Args:
            output_path: 输出文件路径
        """
        print("开始生成城市VDB...")
        
        # 创建基础网格
        city_grid = np.zeros(self.grid_size, dtype=np.float32)
        
        # 添加地形
        print("添加地形...")
        terrain = self.create_terrain()
        city_grid = np.maximum(city_grid, terrain)
        
        # 添加道路
        print("添加道路网络...")
        roads = self.create_road_network()
        city_grid = np.maximum(city_grid, roads)
        
        # 生成建筑布局
        print("生成建筑布局...")
        buildings = self.generate_city_layout()
        
        # 添加建筑
        print("添加建筑...")
        for i, building_info in enumerate(buildings):
            print(f"  添加建筑 {i+1}/{len(buildings)}: {building_info['type']}")
            building = self.create_building(
                building_info['center'], building_info['size'], building_info['type']
            )
            city_grid = np.maximum(city_grid, building)
        
        # 创建VDB网格
        print("创建VDB网格...")
        grid = vdb.FloatGrid()
        grid.copyFromArray(city_grid)
        
        # 设置网格元数据
        grid.name = "density"
        grid.transform = vdb.createLinearTransform(voxelSize=self.voxel_size)
        
        # 写入文件
        print(f"写入文件: {output_path}")
        vdb.write(output_path, grid)
        
        # 保存建筑信息
        metadata_path = output_path.replace('.vdb', '_metadata.json')
        with open(metadata_path, 'w') as f:
            json.dump({
                'city_size': self.city_size, 'voxel_size': self.voxel_size, 'grid_size': self.grid_size, 'buildings': buildings, 'building_count': len(
                    buildings,
                )
            }, f, indent=2)
        
        print(f"生成完成！")
        print(f"VDB文件: {output_path}")
        print(f"元数据: {metadata_path}")
        print(f"城市尺寸: {self.city_size[0]}m x {self.city_size[1]}m x {self.city_size[2]}m")
        print(f"建筑数量: {len(buildings)}")
        
        return output_path

def main():
    """主函数"""
    # 创建生成器
    generator = CityVDBGenerator(city_size=(1000, 1000, 200))
    
    # 生成VDB文件
    output_path = generator.generate_vdb("test_city.vdb")
    
    # 验证文件
    print("\n验证生成的文件...")
    if os.path.exists(output_path):
        grid = vdb.read(output_path)
        print(f"VDB文件读取成功")
        print(f"网格名称: {grid.name}")
        print(f"网格尺寸: {grid.evalActiveVoxelDim()}")
        print(f"体素数量: {grid.activeVoxelCount()}")
    else:
        print("错误：VDB文件生成失败")

if __name__ == "__main__":
    main() 