from typing import Any, Optional
"""
Data Utilities - 数据处理工具

This module provides data processing utilities for GFV library.
"""

import numpy as np
import json
import pickle
import h5py
import os
import logging

logger = logging.getLogger(__name__)

def load_sdf_data(file_path: str) -> tuple[np.ndarray, np.ndarray]:
    """
    加载SDF数据
    
    Args:
        file_path: SDF数据文件路径
        
    Returns:
        coords: 坐标数组 [N, 3]
        sdf_values: SDF值数组 [N]
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"SDF数据文件不存在: {file_path}")
    
    # 根据文件扩展名选择加载方式
    ext = os.path.splitext(file_path)[1].lower()
    
    if ext == '.npy':
        data = np.load(file_path)
        if data.shape[1] != 4:
            raise ValueError(f"期望4列数据 (x, y, z, sdf)，但获得 {data.shape[1]} 列")
        coords = data[:, :3]
        sdf_values = data[:, 3]
    
    elif ext == '.npz':
        data = np.load(file_path)
        coords = data['coords']
        sdf_values = data['sdf_values']
    
    elif ext == '.h5' or ext == '.hdf5':
        with h5py.File(file_path, 'r') as f:
            coords = np.array(f['coords'])
            sdf_values = np.array(f['sdf_values'])
    
    else:
        raise ValueError(f"不支持的文件格式: {ext}")
    
    logger.info(f"已加载SDF数据: {len(coords)} 个点")
    return coords, sdf_values

def save_feature_cache(
    features: dict[str,
    np.ndarray],
    cache_path: str,
    format: str = 'npz',
)
    """
    保存特征缓存
    
    Args:
        features: 特征字典
        cache_path: 缓存文件路径
        format: 保存格式 ('npz', 'h5', 'pickle')
    """
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    
    if format == 'npz':
        # 使用字典解包操作符来传递特征数据
        save_dict = {k: v for k, v in features.items()}
        np.savez_compressed(cache_path, **save_dict)
    
    elif format == 'h5':
        with h5py.File(cache_path, 'w') as f:
            for key, value in features.items():
                f.create_dataset(key, data=value, compression='gzip')
    
    elif format == 'pickle':
        with open(cache_path, 'wb') as f:
            pickle.dump(features, f)
    
    else:
        raise ValueError(f"不支持的保存格式: {format}")
    
    logger.info(f"特征缓存已保存到: {cache_path}")

def load_feature_cache(cache_path: str) -> dict[str, np.ndarray]:
    """
    加载特征缓存
    
    Args:
        cache_path: 缓存文件路径
        
    Returns:
        features: 特征字典
    """
    if not os.path.exists(cache_path):
        raise FileNotFoundError(f"缓存文件不存在: {cache_path}")
    
    ext = os.path.splitext(cache_path)[1].lower()
    
    if ext == '.npz':
        data = np.load(cache_path)
        features = {key: data[key] for key in data.files}
    
    elif ext == '.h5' or ext == '.hdf5':
        features = {}
        with h5py.File(cache_path, 'r') as f:
            for key in f.keys():
                features[key] = np.array(f[key])
    
    elif ext in ['.pkl', '.pickle']:
        with open(cache_path, 'rb') as f:
            features = pickle.load(f)
    
    else:
        raise ValueError(f"不支持的文件格式: {ext}")
    
    logger.info(f"已加载特征缓存: {cache_path}, {len(features)} 个特征")
    return features

def export_features_to_json(
    features: dict[str,
    np.ndarray],
    coords: list[tuple[float,
    float]],
    output_path: str,
)
    """
    将特征导出为JSON格式
    
    Args:
        features: 特征字典
        coords: 坐标列表
        output_path: 输出文件路径
    """
    # 准备导出数据
    export_data = {
        'metadata': {
            'num_features': len(
                features,
            )
        }, 'coordinates': coords, 'features': {}
    }
    
    # 转换特征数据
    for key, feature_array in features.items():
        if isinstance(feature_array, np.ndarray):
            export_data['features'][key] = feature_array.tolist()
        else:
            export_data['features'][key] = feature_array
    
    # 保存JSON文件
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(export_data, f, indent=2, ensure_ascii=False)
    
    logger.info(f"特征已导出为JSON: {output_path}")

def load_geojson_features(
    geojson_path: str,
)
    """
    从GeoJSON文件加载特征
    
    Args:
        geojson_path: GeoJSON文件路径
        
    Returns:
        coords: 坐标列表
        properties: 属性列表
    """
    with open(geojson_path, 'r', encoding='utf-8') as f:
        geojson_data = json.load(f)
    
    coords = []
    properties = []
    
    for feature in geojson_data.get('features', []):
        geometry = feature.get('geometry', {})
        props = feature.get('properties', {})
        
        if geometry.get('type') == 'Point':
            lon, lat = geometry['coordinates']
            coords.append((lat, lon))
            properties.append(props)
    
    logger.info(f"从GeoJSON加载了 {len(coords)} 个特征点")
    return coords, properties

def preprocess_coordinates(
    coords: np.ndarray,
    bounds: tuple[float,
    float,
    float,
    float] | None = None,
    normalize: bool = True,
)
    """
    预处理坐标数据
    
    Args:
        coords: 坐标数组 [N, 2] 或 [N, 3]
        bounds: 边界 (west, south, east, north)
        normalize: 是否归一化
        
    Returns:
        processed_coords: 处理后的坐标数组
    """
    if bounds is not None:
        west, south, east, north = bounds
        
        # 裁剪到边界范围
        mask = ((coords[:, 0] >= south) & (coords[:, 0] <= north) &
                (coords[:, 1] >= west) & (coords[:, 1] <= east))
        coords = coords[mask]
    
    if normalize:
        # 归一化到[0, 1]范围
        coords_min = coords.min(axis=0)
        coords_max = coords.max(axis=0)
        coords = (coords - coords_min) / (coords_max - coords_min)
    
    return coords

def create_spatial_grid(
    bounds: tuple[float,
    float,
    float,
    float],
    resolution: int,
)
    """
    创建空间采样网格
    
    Args:
        bounds: 边界 (west, south, east, north)
        resolution: 网格分辨率
        
    Returns:
        lat_grid: 纬度网格
        lon_grid: 经度网格
    """
    west, south, east, north = bounds
    
    lats = np.linspace(south, north, resolution)
    lons = np.linspace(west, east, resolution)
    
    return np.meshgrid(lats, lons)

def interpolate_features(
    coords: np.ndarray,
    features: np.ndarray,
    target_coords: np.ndarray,
    method: str = 'nearest',
)
    """
    特征插值
    
    Args:
        coords: 原始坐标 [N, 2]
        features: 原始特征 [N, D]
        target_coords: 目标坐标 [M, 2]
        method: 插值方法 ('nearest', 'linear', 'cubic')
        
    Returns:
        interpolated_features: 插值后的特征 [M, D]
    """
    from scipy.interpolate import griddata
    
    return griddata(coords, features, target_coords, method=method, fill_value=0)

def compute_feature_statistics(features: np.ndarray) -> dict[str, Any]:
    """
    计算特征统计信息
    
    Args:
        features: 特征数组 [N, D]
        
    Returns:
        stats: 统计信息字典
    """
    stats = {
        'mean': np.mean(
            features,
            axis=0,
        )
    }
    
    return stats

def batch_process_coordinates(
    coords_list: list[np.ndarray],
    batch_size: int = 1000,
)
    """
    批量处理坐标
    
    Args:
        coords_list: 坐标数组列表
        batch_size: 批次大小
        
    Returns:
        processed_coords: 处理后的坐标列表
    """
    processed_coords = []
    
    for coords in coords_list:
        # 分批处理
        num_batches = (len(coords) + batch_size - 1) // batch_size
        batches = np.array_split(coords, num_batches)
        
        # 处理每个批次
        batch_results = []
        for batch in batches:
            processed_batch = preprocess_coordinates(batch)
            batch_results.append(processed_batch)
        
        # 合并结果
        processed_coords.append(np.concatenate(batch_results))
    
    return processed_coords

def validate_data_consistency(
    coords: list[tuple[float,
    float]],
    features: list[np.ndarray],
)
    """
    验证数据一致性
    
    Args:
        coords: 坐标列表
        features: 特征列表
        
    Returns:
        is_valid: 是否一致
    """
    # 检查长度是否匹配
    if len(coords) != len(features):
        logger.error(f"坐标数量 ({len(coords)}) 与特征数量 ({len(features)}) 不匹配")
        return False
    
    # 检查特征维度是否一致
    feature_dims = [f.shape[-1] for f in features if f is not None]
    if not feature_dims:
        logger.error("没有有效的特征数据")
        return False
    
    if not all(d == feature_dims[0] for d in feature_dims):
        logger.error("特征维度不一致")
        return False
    
    # 检查坐标格式
    for i, (lat, lon) in enumerate(coords):
        if not (-90 <= lat <= 90) or not (-180 <= lon <= 180):
            logger.error(f"无效的坐标: ({lat}, {lon}) at index {i}")
            return False
    
    return True 