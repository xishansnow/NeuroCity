"""
Data Utilities - 数据处理工具

This module provides data processing utilities for GFV library.
"""

import numpy as np
import json
import pickle
import h5py
import os
from typing import Dict, List, Optional, Tuple, Any, Union
import logging

logger = logging.getLogger(__name__)


def load_sdf_data(file_path: str) -> Tuple[np.ndarray, np.ndarray]:
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
            coords = f['coords'][:]
            sdf_values = f['sdf_values'][:]
    
    else:
        raise ValueError(f"不支持的文件格式: {ext}")
    
    logger.info(f"已加载SDF数据: {len(coords)} 个点")
    return coords, sdf_values


def save_feature_cache(features: Dict[str, np.ndarray], 
                      cache_path: str,
                      format: str = 'npz') -> None:
    """
    保存特征缓存
    
    Args:
        features: 特征字典
        cache_path: 缓存文件路径
        format: 保存格式 ('npz', 'h5', 'pickle')
    """
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    
    if format == 'npz':
        np.savez_compressed(cache_path, **features)
    
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


def load_feature_cache(cache_path: str) -> Dict[str, np.ndarray]:
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
                features[key] = f[key][:]
    
    elif ext in ['.pkl', '.pickle']:
        with open(cache_path, 'rb') as f:
            features = pickle.load(f)
    
    else:
        raise ValueError(f"不支持的文件格式: {ext}")
    
    logger.info(f"已加载特征缓存: {cache_path}, {len(features)} 个特征")
    return features


def export_features_to_json(features: Dict[str, np.ndarray],
                           coords: List[Tuple[float, float]],
                           output_path: str) -> None:
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
            'num_features': len(features),
            'num_coords': len(coords),
            'feature_keys': list(features.keys())
        },
        'coordinates': coords,
        'features': {}
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


def load_geojson_features(geojson_path: str) -> Tuple[List[Tuple[float, float]], List[Dict[str, Any]]]:
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


def preprocess_coordinates(coords: np.ndarray, 
                         bounds: Optional[Tuple[float, float, float, float]] = None,
                         normalize: bool = True) -> np.ndarray:
    """
    预处理坐标数据
    
    Args:
        coords: 坐标数组 [N, 2] 或 [N, 3]
        bounds: 边界 (west, south, east, north)
        normalize: 是否归一化
        
    Returns:
        processed_coords: 处理后的坐标
    """
    processed_coords = coords.copy()
    
    # 过滤边界外的点
    if bounds is not None:
        west, south, east, north = bounds
        if coords.shape[1] >= 2:
            mask = ((coords[:, 1] >= west) & (coords[:, 1] <= east) &  # lon
                   (coords[:, 0] >= south) & (coords[:, 0] <= north))  # lat
            processed_coords = processed_coords[mask]
            logger.info(f"边界过滤后保留 {len(processed_coords)} / {len(coords)} 个点")
    
    # 归一化
    if normalize:
        if bounds is not None:
            west, south, east, north = bounds
            processed_coords[:, 1] = (processed_coords[:, 1] - west) / (east - west)  # lon
            processed_coords[:, 0] = (processed_coords[:, 0] - south) / (north - south)  # lat
        else:
            # 使用数据自身的范围归一化
            min_vals = processed_coords.min(axis=0)
            max_vals = processed_coords.max(axis=0)
            processed_coords = (processed_coords - min_vals) / (max_vals - min_vals)
    
    return processed_coords


def create_spatial_grid(bounds: Tuple[float, float, float, float],
                       resolution: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    创建空间网格
    
    Args:
        bounds: 边界 (west, south, east, north)
        resolution: 网格分辨率
        
    Returns:
        grid_coords: 网格坐标 [resolution^2, 2]
        grid_shape: 网格形状 (resolution, resolution)
    """
    west, south, east, north = bounds
    
    # 创建网格
    lons = np.linspace(west, east, resolution)
    lats = np.linspace(south, north, resolution)
    
    lon_grid, lat_grid = np.meshgrid(lons, lats)
    grid_coords = np.stack([lat_grid.flatten(), lon_grid.flatten()], axis=1)
    
    return grid_coords, (resolution, resolution)


def interpolate_features(coords: np.ndarray,
                        features: np.ndarray,
                        target_coords: np.ndarray,
                        method: str = 'nearest') -> np.ndarray:
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
    
    if method == 'nearest':
        interpolated = griddata(coords, features, target_coords, method='nearest')
    elif method == 'linear':
        interpolated = griddata(coords, features, target_coords, method='linear')
        # 填充NaN值
        nan_mask = np.isnan(interpolated).any(axis=1)
        if nan_mask.any():
            nearest_interp = griddata(coords, features, target_coords[nan_mask], method='nearest')
            interpolated[nan_mask] = nearest_interp
    elif method == 'cubic':
        interpolated = griddata(coords, features, target_coords, method='cubic')
        # 填充NaN值
        nan_mask = np.isnan(interpolated).any(axis=1)
        if nan_mask.any():
            linear_interp = griddata(coords, features, target_coords[nan_mask], method='linear')
            interpolated[nan_mask] = linear_interp
            # 如果还有NaN，用nearest填充
            nan_mask = np.isnan(interpolated).any(axis=1)
            if nan_mask.any():
                nearest_interp = griddata(coords, features, target_coords[nan_mask], method='nearest')
                interpolated[nan_mask] = nearest_interp
    else:
        raise ValueError(f"不支持的插值方法: {method}")
    
    return interpolated


def compute_feature_statistics(features: np.ndarray) -> Dict[str, Any]:
    """
    计算特征统计信息
    
    Args:
        features: 特征数组 [N, D]
        
    Returns:
        stats: 统计信息字典
    """
    stats = {
        'shape': features.shape,
        'mean': np.mean(features, axis=0),
        'std': np.std(features, axis=0),
        'min': np.min(features, axis=0),
        'max': np.max(features, axis=0),
        'median': np.median(features, axis=0),
        'percentile_25': np.percentile(features, 25, axis=0),
        'percentile_75': np.percentile(features, 75, axis=0)
    }
    
    # 转换为可序列化的格式
    for key, value in stats.items():
        if isinstance(value, np.ndarray):
            stats[key] = value.tolist()
        elif isinstance(value, tuple):
            stats[key] = list(value)
    
    return stats


def batch_process_coordinates(coords_list: List[np.ndarray],
                            batch_size: int = 1000) -> List[np.ndarray]:
    """
    批量处理坐标
    
    Args:
        coords_list: 坐标数组列表
        batch_size: 批处理大小
        
    Returns:
        processed_coords_list: 处理后的坐标列表
    """
    processed_list = []
    
    for i, coords in enumerate(coords_list):
        if len(coords) > batch_size:
            # 分批处理
            batches = []
            for j in range(0, len(coords), batch_size):
                batch = coords[j:j+batch_size]
                # 这里可以添加具体的处理逻辑
                batches.append(batch)
            processed_coords = np.vstack(batches)
        else:
            processed_coords = coords
        
        processed_list.append(processed_coords)
        
        if (i + 1) % 100 == 0:
            logger.info(f"已处理 {i+1}/{len(coords_list)} 个坐标数组")
    
    return processed_list


def validate_data_consistency(coords: List[Tuple[float, float]],
                            features: List[np.ndarray]) -> bool:
    """
    验证数据一致性
    
    Args:
        coords: 坐标列表
        features: 特征列表
        
    Returns:
        is_valid: 数据是否一致
    """
    if len(coords) != len(features):
        logger.error(f"坐标数量 ({len(coords)}) 与特征数量 ({len(features)}) 不匹配")
        return False
    
    # 检查坐标有效性
    for i, (lat, lon) in enumerate(coords):
        if not (-90 <= lat <= 90) or not (-180 <= lon <= 180):
            logger.error(f"坐标 {i} 超出有效范围: ({lat}, {lon})")
            return False
    
    # 检查特征有效性
    feature_dims = []
    for i, feature in enumerate(features):
        if not isinstance(feature, np.ndarray):
            logger.error(f"特征 {i} 不是numpy数组")
            return False
        
        if np.isnan(feature).any() or np.isinf(feature).any():
            logger.warning(f"特征 {i} 包含NaN或Inf值")
        
        feature_dims.append(feature.shape)
    
    # 检查特征维度一致性
    if len(set(feature_dims)) > 1:
        logger.warning(f"特征维度不一致: {set(feature_dims)}")
    
    logger.info("数据一致性检查通过")
    return True 