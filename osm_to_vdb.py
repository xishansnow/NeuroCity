#!/usr/bin/env python3
"""
从OpenStreetMap下载建筑物数据并转换为VDB格式
支持多种数据源和体素化方法
"""

import os
import json
import numpy as np
import openvdb as vdb
from typing import List, Tuple, Dict, Optional
import requests
import xml.etree.ElementTree as ET
from shapely.geometry import Polygon, Point
from shapely.ops import unary_union
import trimesh
import pyproj
from scipy.spatial import cKDTree
import time
import logging

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class OSMVDBConverter:
    def __init__(self, 
                 bbox: Tuple[float, float, float, float],
                 voxel_size: float = 1.0,
                 max_height: float = 200.0):
        """
        初始化OSM到VDB转换器
        
        Args:
            bbox: 边界框 (min_lat, min_lon, max_lat, max_lon)
            voxel_size: 体素大小（米）
            max_height: 最大高度（米）
        """
        self.bbox = bbox
        self.voxel_size = voxel_size
        self.max_height = max_height
        
        # 计算网格尺寸
        self.grid_size = self._calculate_grid_size()
        
        # 坐标转换器
        self.transformer = pyproj.Transformer.from_crs(
            "EPSG:4326",  # WGS84
            "EPSG:3857",  # Web Mercator
            always_xy=True
        )
        
        self.transformer_back = pyproj.Transformer.from_crs(
            "EPSG:3857",  # Web Mercator
            "EPSG:4326",  # WGS84
            always_xy=True
        )
        
    def _calculate_grid_size(self) -> Tuple[int, int, int]:
        """计算网格尺寸"""
        # 将经纬度转换为米
        min_lat, min_lon, max_lat, max_lon = self.bbox
        
        # 计算边界框的米制尺寸
        x1, y1 = self.transformer.transform(min_lon, min_lat)
        x2, y2 = self.transformer.transform(max_lon, max_lat)
        
        width = abs(x2 - x1)
        height = abs(y2 - y1)
        
        grid_x = int(width / self.voxel_size)
        grid_y = int(height / self.voxel_size)
        grid_z = int(self.max_height / self.voxel_size)
        
        return (grid_x, grid_y, grid_z)
    
    def download_osm_data(self, output_path: str = "buildings.osm") -> str:
        """
        从OSM下载建筑物数据
        
        Args:
            output_path: 输出文件路径
            
        Returns:
            下载的文件路径
        """
        min_lat, min_lon, max_lat, max_lon = self.bbox
        
        # 构建Overpass API查询
        query = f"""
        [out:xml][timeout:25];
        (
          way["building"]({min_lat},{min_lon},{max_lat},{max_lon});
          relation["building"]({min_lat},{min_lon},{max_lat},{max_lon});
        );
        out body;
        >;
        out skel qt;
        """
        
        url = "https://overpass-api.de/api/interpreter"
        
        logger.info(f"正在从OSM下载数据...")
        logger.info(f"边界框: {self.bbox}")
        
        try:
            response = requests.post(url, data=query, timeout=60)
            response.raise_for_status()
            
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(response.text)
            
            logger.info(f"OSM数据已保存到: {output_path}")
            return output_path
            
        except requests.exceptions.RequestException as e:
            logger.error(f"下载OSM数据失败: {e}")
            raise
    
    def parse_osm_buildings(self, osm_file: str) -> List[Dict]:
        """
        解析OSM文件中的建筑物数据
        
        Args:
            osm_file: OSM文件路径
            
        Returns:
            建筑物信息列表
        """
        logger.info("解析OSM建筑物数据...")
        
        tree = ET.parse(osm_file)
        root = tree.getroot()
        
        # 提取节点
        nodes = {}
        for node in root.findall('.//node'):
            node_id = node.get('id')
            lat = float(node.get('lat'))
            lon = float(node.get('lon'))
            nodes[node_id] = (lat, lon)
        
        # 提取建筑物
        buildings = []
        for way in root.findall('.//way'):
            # 检查是否为建筑物
            building_tags = way.findall('.//tag[@k="building"]')
            if not building_tags:
                continue
            
            building_type = building_tags[0].get('v', 'yes')
            
            # 获取高度信息
            height = None
            height_tags = way.findall('.//tag[@k="height"]')
            if height_tags:
                try:
                    height = float(height_tags[0].get('v'))
                except ValueError:
                    pass
            
            # 获取楼层数
            levels = None
            level_tags = way.findall('.//tag[@k="building:levels"]')
            if level_tags:
                try:
                    levels = int(level_tags[0].get('v'))
                    if height is None:
                        height = levels * 3.0  # 估算高度
                except ValueError:
                    pass
            
            # 如果没有高度信息，使用默认值
            if height is None:
                height = 10.0
            
            # 获取节点坐标
            node_refs = way.findall('.//nd')
            if len(node_refs) < 3:  # 至少需要3个点形成多边形
                continue
            
            coordinates = []
            for nd in node_refs:
                node_id = nd.get('ref')
                if node_id in nodes:
                    lat, lon = nodes[node_id]
                    coordinates.append((lat, lon))
            
            if len(coordinates) >= 3:
                buildings.append({
                    'type': building_type,
                    'height': height,
                    'levels': levels,
                    'coordinates': coordinates
                })
        
        logger.info(f"找到 {len(buildings)} 个建筑物")
        return buildings
    
    def convert_coordinates_to_meters(self, buildings: List[Dict]) -> List[Dict]:
        """
        将建筑物坐标从经纬度转换为米
        
        Args:
            buildings: 建筑物列表
            
        Returns:
            转换后的建筑物列表
        """
        logger.info("转换坐标系统...")
        
        converted_buildings = []
        
        for building in buildings:
            # 转换坐标
            coords_meters = []
            for lat, lon in building['coordinates']:
                x, y = self.transformer.transform(lon, lat)
                coords_meters.append((x, y))
            
            # 计算边界框
            x_coords = [coord[0] for coord in coords_meters]
            y_coords = [coord[1] for coord in coords_meters]
            
            min_x, max_x = min(x_coords), max(x_coords)
            min_y, max_y = min(y_coords), max(y_coords)
            
            converted_buildings.append({
                **building,
                'coordinates_meters': coords_meters,
                'bbox': (min_x, min_y, max_x, max_y)
            })
        
        return converted_buildings
    
    def create_building_mesh(self, building: Dict) -> Optional[trimesh.Trimesh]:
        """
        从建筑物数据创建3D网格
        
        Args:
            building: 建筑物信息
            
        Returns:
            3D网格对象
        """
        try:
            # 创建底面多边形
            coords = building['coordinates_meters']
            if len(coords) < 3:
                return None
            
            # 创建底面
            polygon = Polygon(coords)
            if not polygon.is_valid:
                return None
            
            # 拉伸到指定高度
            height = building['height']
            
            # 使用trimesh创建3D网格
            vertices = []
            faces = []
            
            # 底面顶点
            for x, y in coords[:-1]:  # 去掉重复的最后一个点
                vertices.append([x, y, 0])
            
            # 顶面顶点
            for x, y in coords[:-1]:
                vertices.append([x, y, height])
            
            # 创建底面三角形
            if len(coords) > 3:
                # 三角化底面
                triangles = trimesh.triangulate_polygon(coords[:-1])
                for triangle in triangles:
                    faces.append(triangle)
            
            # 创建侧面三角形
            n = len(coords) - 1
            for i in range(n):
                j = (i + 1) % n
                # 第一个三角形
                faces.append([i, j, i + n])
                # 第二个三角形
                faces.append([j, j + n, i + n])
            
            # 创建顶面三角形
            if len(coords) > 3:
                for triangle in triangles:
                    faces.append([v + n for v in triangle])
            
            if len(vertices) > 0 and len(faces) > 0:
                mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
                return mesh
            
        except Exception as e:
            logger.warning(f"创建建筑物网格失败: {e}")
        
        return None
    
    def voxelize_mesh(self, mesh: trimesh.Trimesh, 
                     grid_size: Tuple[int, int, int]) -> np.ndarray:
        """
        将3D网格体素化
        
        Args:
            mesh: 3D网格
            grid_size: 网格尺寸
            
        Returns:
            体素网格
        """
        # 计算网格边界
        min_lat, min_lon, max_lat, max_lon = self.bbox
        x1, y1 = self.transformer.transform(min_lon, min_lat)
        x2, y2 = self.transformer.transform(max_lon, max_lat)
        
        bounds = [
            [x1, x2],
            [y1, y2],
            [0, self.max_height]
        ]
        
        # 体素化
        voxels = mesh.voxelized(pitch=self.voxel_size, bounds=bounds)
        
        if voxels is None:
            return np.zeros(grid_size, dtype=np.float32)
        
        # 转换为numpy数组
        voxel_array = voxels.matrix.astype(np.float32)
        
        # 确保尺寸匹配
        if voxel_array.shape != grid_size:
            # 调整尺寸
            target_array = np.zeros(grid_size, dtype=np.float32)
            min_shape = [min(a, b) for a, b in zip(voxel_array.shape, grid_size)]
            
            target_array[:min_shape[0], :min_shape[1], :min_shape[2]] = \
                voxel_array[:min_shape[0], :min_shape[1], :min_shape[2]]
            
            return target_array
        
        return voxel_array
    
    def create_vdb_from_buildings(self, buildings: List[Dict], 
                                 output_path: str = "osm_buildings.vdb") -> str:
        """
        从建筑物数据创建VDB文件
        
        Args:
            buildings: 建筑物列表
            output_path: 输出文件路径
            
        Returns:
            生成的VDB文件路径
        """
        logger.info("开始创建VDB文件...")
        
        # 创建空网格
        city_grid = np.zeros(self.grid_size, dtype=np.float32)
        
        # 处理每个建筑物
        successful_buildings = 0
        for i, building in enumerate(buildings):
            logger.info(f"处理建筑物 {i+1}/{len(buildings)}")
            
            # 创建3D网格
            mesh = self.create_building_mesh(building)
            if mesh is None:
                continue
            
            # 体素化
            building_voxels = self.voxelize_mesh(mesh, self.grid_size)
            
            # 合并到城市网格
            city_grid = np.maximum(city_grid, building_voxels)
            successful_buildings += 1
        
        logger.info(f"成功处理 {successful_buildings} 个建筑物")
        
        # 创建VDB网格
        grid = vdb.FloatGrid()
        grid.copyFromArray(city_grid)
        
        # 设置网格元数据
        grid.name = "density"
        grid.transform = vdb.createLinearTransform(voxelSize=self.voxel_size)
        
        # 写入文件
        logger.info(f"写入VDB文件: {output_path}")
        vdb.write(output_path, grid)
        
        # 保存元数据
        metadata_path = output_path.replace('.vdb', '_metadata.json')
        metadata = {
            'bbox': self.bbox,
            'voxel_size': self.voxel_size,
            'max_height': self.max_height,
            'grid_size': self.grid_size,
            'total_buildings': len(buildings),
            'successful_buildings': successful_buildings,
            'buildings': buildings
        }
        
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2, default=str)
        
        logger.info(f"VDB文件生成完成: {output_path}")
        logger.info(f"元数据保存到: {metadata_path}")
        
        return output_path
    
    def convert_osm_to_vdb(self, 
                          osm_file: Optional[str] = None,
                          output_path: str = "osm_buildings.vdb") -> str:
        """
        完整的OSM到VDB转换流程
        
        Args:
            osm_file: OSM文件路径（如果为None则自动下载）
            output_path: 输出VDB文件路径
            
        Returns:
            生成的VDB文件路径
        """
        # 下载OSM数据（如果需要）
        if osm_file is None:
            osm_file = "buildings.osm"
            self.download_osm_data(osm_file)
        
        # 解析建筑物数据
        buildings = self.parse_osm_buildings(osm_file)
        
        if not buildings:
            raise ValueError("未找到建筑物数据")
        
        # 转换坐标
        buildings = self.convert_coordinates_to_meters(buildings)
        
        # 创建VDB文件
        vdb_path = self.create_vdb_from_buildings(buildings, output_path)
        
        return vdb_path

def main():
    """主函数 - 示例用法"""
    # 定义边界框（北京天安门附近）
    bbox = (39.9, 116.3, 40.0, 116.4)  # (min_lat, min_lon, max_lat, max_lon)
    
    # 创建转换器
    converter = OSMVDBConverter(
        bbox=bbox,
        voxel_size=2.0,  # 2米体素
        max_height=100.0  # 最大高度100米
    )
    
    try:
        # 执行转换
        vdb_path = converter.convert_osm_to_vdb(output_path="beijing_buildings.vdb")
        
        # 验证结果
        if os.path.exists(vdb_path):
            grid = vdb.read(vdb_path)
            logger.info(f"VDB文件验证成功")
            logger.info(f"网格尺寸: {grid.evalActiveVoxelDim()}")
            logger.info(f"活跃体素数: {grid.activeVoxelCount()}")
        else:
            logger.error("VDB文件生成失败")
            
    except Exception as e:
        logger.error(f"转换过程中出现错误: {e}")

if __name__ == "__main__":
    main() 