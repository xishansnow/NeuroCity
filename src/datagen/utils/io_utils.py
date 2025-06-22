"""
IO工具函数

用于数据保存和加载的工具函数。
"""

import numpy as np
import json
import os
from typing import Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)


def save_numpy_data(data: np.ndarray, filepath: str, compressed: bool = True):
    """
    保存numpy数据
    
    Args:
        data: numpy数组
        filepath: 文件路径
        compressed: 是否压缩
    """
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    
    if compressed:
        np.savez_compressed(filepath, data=data)
    else:
        np.save(filepath, data)
    
    logger.info(f"数据已保存: {filepath}")


def load_numpy_data(filepath: str) -> np.ndarray:
    """
    加载numpy数据
    
    Args:
        filepath: 文件路径
        
    Returns:
        numpy数组
    """
    if filepath.endswith('.npz'):
        data = np.load(filepath)
        return data['data']
    else:
        return np.load(filepath)


def save_json_metadata(metadata: Dict[str, Any], filepath: str):
    """
    保存JSON元数据
    
    Args:
        metadata: 元数据字典
        filepath: 文件路径
    """
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    
    with open(filepath, 'w') as f:
        json.dump(metadata, f, indent=2, default=str)
    
    logger.info(f"元数据已保存: {filepath}")


def load_json_metadata(filepath: str) -> Dict[str, Any]:
    """
    加载JSON元数据
    
    Args:
        filepath: 文件路径
        
    Returns:
        元数据字典
    """
    with open(filepath, 'r') as f:
        metadata = json.load(f)
    
    return metadata


def create_output_directory(base_dir: str, experiment_name: str) -> str:
    """
    创建输出目录
    
    Args:
        base_dir: 基础目录
        experiment_name: 实验名称
        
    Returns:
        创建的目录路径
    """
    output_dir = os.path.join(base_dir, experiment_name)
    os.makedirs(output_dir, exist_ok=True)
    
    logger.info(f"输出目录已创建: {output_dir}")
    return output_dir


# 简化的几何工具函数
def compute_sdf_sphere(points, center=[0, 0, 0], radius=1.0):
    """计算球体SDF"""
    distances = np.linalg.norm(points - np.array(center), axis=1)
    return distances - radius

def compute_sdf_box(points, center=[0, 0, 0], size=[1, 1, 1]):
    """计算盒子SDF"""
    relative_points = points - np.array(center)
    d = np.abs(relative_points) - np.array(size) / 2
    outside_distance = np.linalg.norm(np.maximum(d, 0), axis=1)
    inside_distance = np.min(np.maximum(d, np.min(d, axis=1, keepdims=True)), axis=1)
    return outside_distance + np.minimum(inside_distance, 0)

def compute_sdf_cylinder(points, center=[0, 0, 0], radius=1.0, height=2.0, axis=2):
    """计算圆柱体SDF"""
    relative_points = points - np.array(center)
    
    if axis == 0:  # X轴
        radial_coords = relative_points[:, [1, 2]]
        axial_coord = relative_points[:, 0]
    elif axis == 1:  # Y轴
        radial_coords = relative_points[:, [0, 2]]
        axial_coord = relative_points[:, 1]
    else:  # Z轴
        radial_coords = relative_points[:, [0, 1]]
        axial_coord = relative_points[:, 2]
    
    radial_distance = np.linalg.norm(radial_coords, axis=1) - radius
    axial_distance = np.abs(axial_coord) - height / 2
    
    outside_distance = np.linalg.norm(
        np.column_stack([
            np.maximum(radial_distance, 0),
            np.maximum(axial_distance, 0)
        ]), axis=1
    )
    
    inside_distance = np.minimum(np.maximum(radial_distance, axial_distance), 0)
    
    return outside_distance + inside_distance

def mesh_to_sdf(points, vertices, faces):
    """从网格计算SDF值（简化版本）"""
    min_distances = []
    for point in points:
        distances_to_vertices = np.linalg.norm(vertices - point, axis=1)
        min_distance = np.min(distances_to_vertices)
        min_distances.append(min_distance)
    return np.array(min_distances)

def point_cloud_to_sdf(query_points, cloud_points, bandwidth=1.0):
    """从点云估计SDF值"""
    from scipy.spatial import cKDTree
    tree = cKDTree(cloud_points)
    distances, indices = tree.query(query_points)
    sdf_values = distances - bandwidth
    return sdf_values 