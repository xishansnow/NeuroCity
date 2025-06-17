#!/usr/bin/env python3
"""
大规模城市tile体素生成器
生成10km x 10km范围，每1km x 1km为一个tile，体素分辨率为1m
"""

import numpy as np
import json
import os
from typing import List, Tuple, Dict
import random

class TileCityGenerator:
    def __init__(self, 
                 city_size: Tuple[int, int, int] = (10000, 10000, 100),
                 tile_size: Tuple[int, int] = (1000, 1000),
                 voxel_size: float = 1.0,
                 output_dir: str = "tiles"):
        """
        初始化tile城市生成器
        Args:
            city_size: 总城市尺寸 (x, y, z) 单位：米
            tile_size: tile尺寸 (x, y) 单位：米
            voxel_size: 体素大小（米）
            output_dir: 输出目录
        """
        self.city_size = city_size
        self.tile_size = tile_size
        self.voxel_size = voxel_size
        self.output_dir = output_dir
        self.grid_size = (
            int(tile_size[0] / voxel_size),
            int(tile_size[1] / voxel_size),
            int(city_size[2] / voxel_size)
        )
        os.makedirs(self.output_dir, exist_ok=True)
        self.tiles_x = city_size[0] // tile_size[0]
        self.tiles_y = city_size[1] // tile_size[1]

    def create_simple_building(self, 
                              center: Tuple[float, float, float],
                              size: Tuple[float, float, float],
                              tile_origin: Tuple[float, float]) -> np.ndarray:
        """
        创建简单建筑，坐标为全局坐标
        Args:
            center: 建筑中心 (x, y, z) 全局坐标
            size: 建筑尺寸 (width, length, height)
            tile_origin: tile左下角全局坐标
        Returns:
            建筑体素网格（tile内）
        """
        # 转换为tile内体素坐标
        center_voxel = (
            int((center[0] - tile_origin[0]) / self.voxel_size),
            int((center[1] - tile_origin[1]) / self.voxel_size),
            int(center[2] / self.voxel_size)
        )
        size_voxel = (
            int(size[0] / self.voxel_size),
            int(size[1] / self.voxel_size),
            int(size[2] / self.voxel_size)
        )
        building = np.zeros(self.grid_size, dtype=np.float32)
        x1 = max(0, center_voxel[0] - size_voxel[0]//2)
        y1 = max(0, center_voxel[1] - size_voxel[1]//2)
        x2 = min(self.grid_size[0], center_voxel[0] + size_voxel[0]//2)
        y2 = min(self.grid_size[1], center_voxel[1] + size_voxel[1]//2)
        z_max = min(self.grid_size[2], center_voxel[2] + size_voxel[2]//2)
        z_min = max(0, center_voxel[2] - size_voxel[2]//2)
        building[x1:x2, y1:y2, z_min:z_max] = 1.0
        return building

    def create_road_network(self, tile_origin: Tuple[float, float]) -> np.ndarray:
        """创建tile内简单道路网络"""
        roads = np.zeros(self.grid_size, dtype=np.float32)
        road_width = int(8 / self.voxel_size)
        # 东西向主道路
        for y in range(0, self.grid_size[1], 200):
            roads[:, max(0, y-road_width):min(self.grid_size[1], y+road_width), :5] = 1.0
        # 南北向主道路
        for x in range(0, self.grid_size[0], 200):
            roads[max(0, x-road_width):min(self.grid_size[0], x+road_width), :, :5] = 1.0
        return roads

    def generate_tile(self, tile_x: int, tile_y: int, buildings: List[Dict]) -> Tuple[np.ndarray, List[Dict]]:
        """
        生成单个tile的体素网格和建筑信息
        Args:
            tile_x, tile_y: tile索引
            buildings: 所有建筑信息（全局坐标）
        Returns:
            (tile体素网格, tile内建筑信息)
        """
        tile_origin = (tile_x * self.tile_size[0], tile_y * self.tile_size[1])
        tile_grid = np.zeros(self.grid_size, dtype=np.float32)
        tile_buildings = []
        # 筛选属于本tile的建筑
        for b in buildings:
            cx, cy, cz = b['center']
            if (tile_origin[0] <= cx < tile_origin[0] + self.tile_size[0] and
                tile_origin[1] <= cy < tile_origin[1] + self.tile_size[1]):
                building = self.create_simple_building(b['center'], b['size'], tile_origin)
                tile_grid = np.maximum(tile_grid, building)
                tile_buildings.append(b)
        # 添加道路
        roads = self.create_road_network(tile_origin)
        tile_grid = np.maximum(tile_grid, roads)
        return tile_grid, tile_buildings

    def generate_global_buildings(self, n_per_tile: int = 20) -> List[Dict]:
        """
        生成全局建筑信息（均匀分布在所有tile）
        Returns:
            所有建筑信息（全局坐标）
        """
        buildings = []
        for tx in range(self.tiles_x):
            for ty in range(self.tiles_y):
                tile_origin = (tx * self.tile_size[0], ty * self.tile_size[1])
                for i in range(n_per_tile):
                    x = random.uniform(tile_origin[0]+50, tile_origin[0]+self.tile_size[0]-50)
                    y = random.uniform(tile_origin[1]+50, tile_origin[1]+self.tile_size[1]-50)
                    z = 0
                    width = random.uniform(15, 40)
                    length = random.uniform(15, 40)
                    height = random.uniform(10, 60)
                    buildings.append({
                        'type': 'building',
                        'center': (x, y, z),
                        'size': (width, length, height),
                        'id': f"b_{tx}_{ty}_{i}"
                    })
        return buildings

    def save_tile(self, tile_grid: np.ndarray, tile_buildings: List[Dict], tile_x: int, tile_y: int):
        """
        保存tile体素和元数据
        """
        npy_path = os.path.join(self.output_dir, f"tile_{tile_x}_{tile_y}.npy")
        json_path = os.path.join(self.output_dir, f"tile_{tile_x}_{tile_y}.json")
        np.save(npy_path, tile_grid)
        metadata = {
            'tile_index': (tile_x, tile_y),
            'tile_origin': (tile_x * self.tile_size[0], tile_y * self.tile_size[1]),
            'tile_size': self.tile_size,
            'voxel_size': self.voxel_size,
            'grid_size': self.grid_size,
            'buildings': tile_buildings,
            'building_count': len(tile_buildings)
        }
        with open(json_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        print(f"保存: {npy_path}, {json_path}")

    def generate_and_save_all_tiles(self, n_per_tile: int = 20):
        """
        生成并保存所有tile
        """
        print(f"生成全局建筑...")
        buildings = self.generate_global_buildings(n_per_tile=n_per_tile)
        print(f"总建筑数: {len(buildings)}")
        for tx in range(self.tiles_x):
            for ty in range(self.tiles_y):
                print(f"生成tile ({tx}, {ty}) ...")
                tile_grid, tile_buildings = self.generate_tile(tx, ty, buildings)
                self.save_tile(tile_grid, tile_buildings, tx, ty)
        print("全部tile生成完毕！")

def main():
    generator = TileCityGenerator(
        city_size=(10000, 10000, 100),
        tile_size=(1000, 1000),
        voxel_size=1.0,
        output_dir="tiles"
    )
    generator.generate_and_save_all_tiles(n_per_tile=20)

if __name__ == "__main__":
    main() 