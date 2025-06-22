"""
Coordinate Utilities - 坐标转换工具

This module provides coordinate transformation utilities for GFV library.
"""

import math
import numpy as np
import mercantile
import pyproj
from typing import Tuple, List, Optional


def lat_lon_to_mercator(lat: float, lon: float) -> Tuple[float, float]:
    """
    将经纬度转换为墨卡托投影坐标
    
    Args:
        lat: 纬度
        lon: 经度
        
    Returns:
        (x, y): 墨卡托坐标
    """
    # Web Mercator (EPSG:3857)
    x = lon * 20037508.34 / 180
    y = math.log(math.tan((90 + lat) * math.pi / 360)) / (math.pi / 180)
    y = y * 20037508.34 / 180
    return x, y


def mercator_to_lat_lon(x: float, y: float) -> Tuple[float, float]:
    """
    将墨卡托投影坐标转换为经纬度
    
    Args:
        x: 墨卡托X坐标
        y: 墨卡托Y坐标
        
    Returns:
        (lat, lon): 经纬度坐标
    """
    lon = (x / 20037508.34) * 180
    lat = (y / 20037508.34) * 180
    lat = 180 / math.pi * (2 * math.atan(math.exp(lat * math.pi / 180)) - math.pi / 2)
    return lat, lon


def lat_lon_to_tile(lat: float, lon: float, zoom: int) -> Tuple[int, int]:
    """
    将经纬度转换为瓦片坐标
    
    Args:
        lat: 纬度
        lon: 经度
        zoom: 缩放级别
        
    Returns:
        (x, y): 瓦片坐标
    """
    tile = mercantile.tile(lon, lat, zoom)
    return tile.x, tile.y


def tile_to_lat_lon(x: int, y: int, zoom: int) -> Tuple[float, float]:
    """
    将瓦片坐标转换为经纬度（瓦片中心点）
    
    Args:
        x: 瓦片X坐标
        y: 瓦片Y坐标
        zoom: 缩放级别
        
    Returns:
        (lat, lon): 经纬度坐标
    """
    bounds = mercantile.bounds(x, y, zoom)
    lat = (bounds.north + bounds.south) / 2
    lon = (bounds.west + bounds.east) / 2
    return lat, lon


def calculate_tile_bounds(x: int, y: int, zoom: int) -> Tuple[float, float, float, float]:
    """
    计算瓦片的地理边界
    
    Args:
        x: 瓦片X坐标
        y: 瓦片Y坐标
        zoom: 缩放级别
        
    Returns:
        (west, south, east, north): 地理边界
    """
    bounds = mercantile.bounds(x, y, zoom)
    return bounds.west, bounds.south, bounds.east, bounds.north


def degrees_to_radians(degrees: float) -> float:
    """角度转弧度"""
    return degrees * math.pi / 180


def radians_to_degrees(radians: float) -> float:
    """弧度转角度"""
    return radians * 180 / math.pi


def calculate_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """
    计算两点间的大圆距离（Haversine公式）
    
    Args:
        lat1, lon1: 第一个点的经纬度
        lat2, lon2: 第二个点的经纬度
        
    Returns:
        distance: 距离（公里）
    """
    R = 6371  # 地球半径（公里）
    
    # 转换为弧度
    lat1_rad = degrees_to_radians(lat1)
    lon1_rad = degrees_to_radians(lon1)
    lat2_rad = degrees_to_radians(lat2)
    lon2_rad = degrees_to_radians(lon2)
    
    # Haversine公式
    dlat = lat2_rad - lat1_rad
    dlon = lon2_rad - lon1_rad
    
    a = (math.sin(dlat/2)**2 + 
         math.cos(lat1_rad) * math.cos(lat2_rad) * math.sin(dlon/2)**2)
    c = 2 * math.asin(math.sqrt(a))
    
    return R * c


def get_tiles_in_bbox(west: float, south: float, east: float, north: float, 
                     zoom: int) -> List[Tuple[int, int]]:
    """
    获取边界框内的所有瓦片
    
    Args:
        west, south, east, north: 边界框坐标
        zoom: 缩放级别
        
    Returns:
        tiles: 瓦片坐标列表
    """
    tiles = list(mercantile.tiles(west, south, east, north, zoom))
    return [(tile.x, tile.y) for tile in tiles]


def normalize_coordinates(coords: np.ndarray, bounds: Tuple[float, float, float, float]) -> np.ndarray:
    """
    将坐标归一化到[0, 1]范围
    
    Args:
        coords: 坐标数组 [N, 2] (lat, lon)
        bounds: 边界 (west, south, east, north)
        
    Returns:
        normalized_coords: 归一化坐标
    """
    west, south, east, north = bounds
    
    # 归一化
    normalized = np.zeros_like(coords)
    normalized[:, 0] = (coords[:, 1] - west) / (east - west)  # lon -> x
    normalized[:, 1] = (coords[:, 0] - south) / (north - south)  # lat -> y
    
    return normalized


def denormalize_coordinates(normalized_coords: np.ndarray, 
                          bounds: Tuple[float, float, float, float]) -> np.ndarray:
    """
    将归一化坐标转换回地理坐标
    
    Args:
        normalized_coords: 归一化坐标 [N, 2]
        bounds: 边界 (west, south, east, north)
        
    Returns:
        coords: 地理坐标 (lat, lon)
    """
    west, south, east, north = bounds
    
    coords = np.zeros_like(normalized_coords)
    coords[:, 1] = normalized_coords[:, 0] * (east - west) + west  # x -> lon
    coords[:, 0] = normalized_coords[:, 1] * (north - south) + south  # y -> lat
    
    return coords


def calculate_zoom_level(bbox: Tuple[float, float, float, float], 
                        target_tiles: int = 100) -> int:
    """
    根据边界框和目标瓦片数计算合适的缩放级别
    
    Args:
        bbox: 边界框 (west, south, east, north)
        target_tiles: 目标瓦片数量
        
    Returns:
        zoom: 缩放级别
    """
    west, south, east, north = bbox
    
    for zoom in range(1, 20):
        tiles = get_tiles_in_bbox(west, south, east, north, zoom)
        if len(tiles) >= target_tiles:
            return max(1, zoom - 1)
    
    return 18  # 最大缩放级别


def project_coordinates(coords: np.ndarray, 
                       src_crs: str = "EPSG:4326", 
                       dst_crs: str = "EPSG:3857") -> np.ndarray:
    """
    坐标系投影转换
    
    Args:
        coords: 坐标数组 [N, 2]
        src_crs: 源坐标系
        dst_crs: 目标坐标系
        
    Returns:
        projected_coords: 投影后的坐标
    """
    transformer = pyproj.Transformer.from_crs(src_crs, dst_crs, always_xy=True)
    
    if coords.ndim == 1:
        x, y = transformer.transform(coords[0], coords[1])
        return np.array([x, y])
    else:
        x, y = transformer.transform(coords[:, 0], coords[:, 1])
        return np.column_stack([x, y]) 