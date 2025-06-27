from typing import Any, Optional
"""
Utility Functions for NeuralVDB

This module contains utility functions for data generation, loading, saving, and other common operations.
"""

import numpy as np
import torch
import json
import os
import pickle
import logging
from torch.utils.data import DataLoader

logger = logging.getLogger(__name__)

def create_sample_data(
    n_points: int = 10000,
    scene_type: str = 'mixed',
) -> tuple[np.ndarray, np.ndarray]:
    """
    创建示例数据
    
    Args:
        n_points: 点数量
        scene_type: 场景类型 ('sphere', 'cube', 'mixed', 'urban')
        
    Returns:
        (points, occupancies)
    """
    logger.info(f"创建示例数据，类型: {scene_type}，点数: {n_points}")
    
    # 生成随机点
    points = np.random.rand(n_points, 3) * 100
    occupancies = np.zeros(n_points, dtype=np.float32)
    
    if scene_type == 'sphere':
        # 球体
        center = np.array([50, 50, 50])
        radius = 20
        distances = np.linalg.norm(points - center, axis=1)
        occupancies = (distances < radius).astype(np.float32)
        
    elif scene_type == 'cube':
        # 立方体
        center = np.array([50, 50, 50])
        size = 30
        in_cube = np.all(
            (points >= center - size/2) & (points < center + size/2), axis=1
        )
        occupancies = in_cube.astype(np.float32)
        
    elif scene_type == 'mixed':
        # 混合几何体
        # 球体
        center1 = np.array([30, 30, 30])
        radius1 = 15
        distances1 = np.linalg.norm(points - center1, axis=1)
        occupancies += (distances1 < radius1).astype(np.float32)
        
        # 立方体
        center2 = np.array([70, 70, 70])
        size2 = 20
        in_cube = np.all(
            (points >= center2 - size2/2) & (points < center2 + size2/2), axis=1
        )
        occupancies += in_cube.astype(np.float32)
        
        # 圆柱体
        center3 = np.array([20, 80, 50])
        radius3 = 12
        height3 = 30
        
        horizontal_dist = np.sqrt((points[:, 0] - center3[0])**2 + (points[:, 1] - center3[1])**2)
        vertical_dist = np.abs(points[:, 2] - center3[2])
        
        in_cylinder = (horizontal_dist < radius3) & (vertical_dist < height3/2)
        occupancies += in_cylinder.astype(np.float32)
        
    elif scene_type == 'urban':
        # 城市场景
        occupancies = _create_urban_scene(points)
    
    # 添加噪声
    noise = np.random.rand(n_points) * 0.1
    occupancies = np.clip(occupancies + noise, 0, 1)
    
    logger.info(f"示例数据创建完成，占用率: {np.mean(occupancies > 0.5):.3f}")
    
    return points, occupancies

def _create_urban_scene(points: np.ndarray) -> np.ndarray:
    """创建城市场景"""
    occupancies = np.zeros(len(points), dtype=np.float32)
    
    # 建筑物
    buildings = [
        {
            'center': [25,
            25,
            20],
            'size': [20,
            20,
            40],
        }
    ]
    
    for building in buildings:
        center = np.array(building['center'])
        size = np.array(building['size'])
        
        in_building = np.all(
            (points >= center - size/2) & (points < center + size/2), axis=1
        )
        occupancies += in_building.astype(np.float32)
    
    # 道路（负空间）
    # 水平道路
    road_width = 8
    for y in [20, 50, 80]:
        on_road = (np.abs(points[:, 1] - y) < road_width/2) & (points[:, 2] < 2)
        occupancies[on_road] = 0.1  # 道路有低占用率
    
    # 垂直道路
    for x in [20, 50, 80]:
        on_road = (np.abs(points[:, 0] - x) < road_width/2) & (points[:, 2] < 2)
        occupancies[on_road] = 0.1
    
    return occupancies

def load_training_data(
    samples_dir: str,
    task_type: str = 'occupancy',
    train_ratio: float = 0.8,
)
    """
    加载训练数据
    
    Args:
        samples_dir: 样本目录
        task_type: 任务类型 ('occupancy' 或 'sdf')
        train_ratio: 训练集比例
        
    Returns:
        (train_loader, val_loader)
    """
    logger.info(f"从 {samples_dir} 加载训练数据，任务类型: {task_type}")
    
    coords_list = []
    targets_list = []
    
    # 扫描样本文件
    for filename in os.listdir(samples_dir):
        if filename.endswith('.npz'):
            filepath = os.path.join(samples_dir, filename)
            data = np.load(filepath)
            
            coords = data['coords']
            
            if task_type == 'occupancy':
                if 'occupancies' in data:
                    targets = data['occupancies']
                elif 'labels' in data:
                    targets = data['labels']
                else:
                    logger.warning(f"文件 {filename} 中未找到占用数据")
                    continue
            elif task_type == 'sdf':
                if 'sdf_values' in data:
                    targets = data['sdf_values']
                else:
                    logger.warning(f"文件 {filename} 中未找到SDF数据")
                    continue
            else:
                raise ValueError(f"未知任务类型: {task_type}")
            
            coords_list.append(coords)
            targets_list.append(targets)
    
    if not coords_list:
        raise ValueError(f"在 {samples_dir} 中未找到有效的训练数据")
    
    # 合并数据
    all_coords = np.concatenate(coords_list, axis=0)
    all_targets = np.concatenate(targets_list, axis=0)
    
    logger.info(f"加载完成，总样本数: {len(all_coords)}")
    
    # 创建数据集
    from .dataset import VoxelDataset, create_data_loaders
    
    if task_type == 'occupancy':
        dataset = VoxelDataset(all_coords, labels=all_targets, task_type='occupancy')
    else:
        dataset = VoxelDataset(all_coords, sdf_values=all_targets, task_type='sdf')
    
    # 创建数据加载器
    train_loader, val_loader = create_data_loaders(
        dataset, train_ratio=train_ratio, batch_size=1024
    )
    
    return train_loader, val_loader

def save_vdb_data(data: dict[str, Any], filepath: str, format: str = 'npz') -> None:
    """
    保存VDB数据
    
    Args:
        data: 要保存的数据字典
        filepath: 保存路径
        format: 保存格式 ('npz', 'pickle', 'json')
    """
    logger.info(f"保存VDB数据到 {filepath}，格式: {format}")
    
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    
    if format == 'npz':
        # 分离numpy数组和其他数据
        arrays = {}
        metadata = {}
        
        for key, value in data.items():
            if isinstance(value, np.ndarray):
                arrays[key] = value
            else:
                metadata[key] = value
        
        # 保存数组
        np.savez_compressed(filepath, **arrays)
        
        # 保存元数据
        if metadata:
            metadata_path = filepath.replace('.npz', '_metadata.json')
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2, default=str)
                
    elif format == 'pickle':
        with open(filepath, 'wb') as f:
            pickle.dump(data, f)
            
    elif format == 'json':
        # 转换numpy数组为列表
        json_data = {}
        for key, value in data.items():
            if isinstance(value, np.ndarray):
                json_data[key] = value.tolist()
            else:
                json_data[key] = value
        
        with open(filepath, 'w') as f:
            json.dump(json_data, f, indent=2)
    else:
        raise ValueError(f"不支持的格式: {format}")
    
    logger.info(f"VDB数据保存完成")

def load_vdb_data(filepath: str, format: str = 'auto') -> dict[str, Any]:
    """
    加载VDB数据
    
    Args:
        filepath: 文件路径
        format: 文件格式 ('auto', 'npz', 'pickle', 'json')
        
    Returns:
        数据字典
    """
    logger.info(f"从 {filepath} 加载VDB数据")
    
    if format == 'auto':
        # 自动检测格式
        if filepath.endswith('.npz'):
            format = 'npz'
        elif filepath.endswith('.pkl') or filepath.endswith('.pickle'):
            format = 'pickle'
        elif filepath.endswith('.json'):
            format = 'json'
        else:
            raise ValueError(f"无法自动检测文件格式: {filepath}")
    
    data = {}
    
    if format == 'npz':
        # 加载数组
        arrays = np.load(filepath)
        for key in arrays.files:
            data[key] = arrays[key]
        
        # 加载元数据
        metadata_path = filepath.replace('.npz', '_metadata.json')
        if os.path.exists(metadata_path):
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
                data.update(metadata)
                
    elif format == 'pickle':
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
            
    elif format == 'json':
        with open(filepath, 'r') as f:
            json_data = json.load(f)
            
        # 转换列表为numpy数组（如果需要）
        for key, value in json_data.items():
            if isinstance(value, list) and key.endswith('_array'):
                data[key] = np.array(value)
            else:
                data[key] = value
    else:
        raise ValueError(f"不支持的格式: {format}")
    
    logger.info(f"VDB数据加载完成，包含 {len(data)} 个字段")
    
    return data

def compute_sdf_from_occupancy(occupancy_grid: np.ndarray, voxel_size: float = 1.0) -> np.ndarray:
    """
    从占用网格计算SDF
    
    Args:
        occupancy_grid: 占用网格 (H, W, D)
        voxel_size: 体素大小
        
    Returns:
        SDF网格 (H, W, D)
    """
    try:
        from scipy.ndimage import distance_transform_edt
    except ImportError:
        raise ImportError("需要安装scipy来计算SDF")
    
    logger.info("从占用网格计算SDF...")
    
    # 二值化占用网格
    binary_grid = (occupancy_grid > 0.5).astype(np.uint8)
    
    # 计算内部和外部的距离变换
    inside_distance = distance_transform_edt(binary_grid) * voxel_size
    outside_distance = distance_transform_edt(1 - binary_grid) * voxel_size
    
    # 组合为SDF（内部为负，外部为正）
    sdf_grid = outside_distance - inside_distance
    
    logger.info("SDF计算完成")
    
    return sdf_grid

def extract_surface_points(
    sdf_grid: np.ndarray,
    threshold: float = 0.1,
    voxel_size: float = 1.0,
)
    """
    从SDF网格提取表面点
    
    Args:
        sdf_grid: SDF网格
        threshold: 表面阈值
        voxel_size: 体素大小
        
    Returns:
        (surface_points, surface_sdf_values)
    """
    logger.info("从SDF网格提取表面点...")
    
    # 找到接近表面的点
    surface_mask = np.abs(sdf_grid) < threshold
    surface_indices = np.where(surface_mask)
    
    # 转换为世界坐标
    surface_points = np.column_stack(surface_indices).astype(np.float32) * voxel_size
    surface_sdf_values = sdf_grid[surface_mask]
    
    logger.info(f"提取到 {len(surface_points)} 个表面点")
    
    return surface_points, surface_sdf_values

def visualize_training_data(
    points: np.ndarray,
    occupancies: np.ndarray,
    save_path: Optional[str] = None,
    show_plot: bool = True,
)
    """
    可视化训练数据
    
    Args:
        points: 3D坐标点
        occupancies: 占用值
        save_path: 保存路径
        show_plot: 是否显示图形
    """
    try:
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D
    except ImportError:
        logger.warning("matplotlib未安装，无法可视化")
        return
    
    logger.info("可视化训练数据...")
    
    fig = plt.figure(figsize=(15, 5))
    
    # 分离占用和空闲点
    occupied_mask = occupancies > 0.5
    occupied_points = points[occupied_mask]
    empty_points = points[~occupied_mask]
    
    # 3D散点图
    ax1 = fig.add_subplot(131, projection='3d')
    if len(occupied_points) > 0:
        ax1.scatter(
            occupied_points[:,
            0],
            occupied_points[:,
            1],
            occupied_points[:,
            2],
            c='red',
            s=1,
            alpha=0.6,
            label='Occupied',
        )
    if len(empty_points) > 0:
        # 随机采样空闲点以减少绘制时间
        if len(empty_points) > 5000:
            indices = np.random.choice(len(empty_points), 5000, replace=False)
            empty_sample = empty_points[indices]
        else:
            empty_sample = empty_points
        
        ax1.scatter(
            empty_sample[:,
            0],
            empty_sample[:,
            1],
            empty_sample[:,
            2],
            c='blue',
            s=1,
            alpha=0.1,
            label='Empty',
        )
    
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.set_zlabel('Z')
    ax1.set_title('3D View')
    ax1.legend()
    
    # XY平面投影
    ax2 = fig.add_subplot(132)
    if len(occupied_points) > 0:
        ax2.scatter(occupied_points[:, 0], occupied_points[:, 1], c='red', s=1, alpha=0.6)
    if len(empty_points) > 0:
        ax2.scatter(empty_points[:, 0], empty_points[:, 1], c='blue', s=1, alpha=0.1)
    
    ax2.set_xlabel('X')
    ax2.set_ylabel('Y')
    ax2.set_title('XY Projection')
    ax2.set_aspect('equal')
    
    # 占用率分布
    ax3 = fig.add_subplot(133)
    ax3.hist(occupancies, bins=50, alpha=0.7, color='green')
    ax3.set_xlabel('Occupancy Value')
    ax3.set_ylabel('Frequency')
    ax3.set_title('Occupancy Distribution')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"可视化结果已保存到: {save_path}")
    
    if show_plot:
        plt.show()
    else:
        plt.close()

def visualize_predictions(
    points: np.ndarray,
    predictions: np.ndarray,
    ground_truth: Optional[np.ndarray] = None,
    save_path: Optional[str] = None,
    show_plot: bool = True,
)
    """
    可视化预测结果
    
    Args:
        points: 3D坐标点
        predictions: 预测值
        ground_truth: 真实值（可选）
        save_path: 保存路径
        show_plot: 是否显示图形
    """
    try:
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D
    except ImportError:
        logger.warning("matplotlib未安装，无法可视化")
        return
    
    logger.info("可视化预测结果...")
    
    if ground_truth is not None:
        fig = plt.figure(figsize=(20, 5))
        n_subplots = 4
    else:
        fig = plt.figure(figsize=(15, 5))
        n_subplots = 3
    
    # 预测值分布
    ax1 = fig.add_subplot(1, n_subplots, 1)
    ax1.hist(predictions, bins=50, alpha=0.7, color='orange')
    ax1.set_xlabel('Predicted Occupancy')
    ax1.set_ylabel('Frequency')
    ax1.set_title('Prediction Distribution')
    
    # 3D预测结果
    ax2 = fig.add_subplot(1, n_subplots, 2, projection='3d')
    
    # 随机采样以减少绘制时间
    if len(points) > 10000:
        indices = np.random.choice(len(points), 10000, replace=False)
        sample_points = points[indices]
        sample_predictions = predictions[indices]
    else:
        sample_points = points
        sample_predictions = predictions
    
    scatter = ax2.scatter(
        sample_points[:,
        0],
        sample_points[:,
        1],
        sample_points[:,
        2],
        c=sample_predictions,
        cmap='viridis',
        s=10,
        alpha=0.7,
    )
    
    ax2.set_xlabel('X')
    ax2.set_ylabel('Y')
    ax2.set_zlabel('Z')
    ax2.set_title('Prediction Results (3D)')
    
    # XY平面热力图
    ax3 = fig.add_subplot(1, n_subplots, 3)
    scatter_2d = ax3.scatter(
        sample_points[:,
        0],
        sample_points[:,
        1],
        c=sample_predictions,
        cmap='viridis',
        s=10,
        alpha=0.7,
    )
    ax3.set_xlabel('X')
    ax3.set_ylabel('Y')
    ax3.set_title('Prediction Heatmap (XY)')
    ax3.set_aspect('equal')
    plt.colorbar(scatter_2d, ax=ax3)
    
    # 如果有真实值，添加误差分析
    if ground_truth is not None:
        ax4 = fig.add_subplot(1, n_subplots, 4)
        
        errors = np.abs(predictions - ground_truth)
        ax4.hist(errors, bins=50, alpha=0.7, color='red')
        ax4.set_xlabel('Absolute Error')
        ax4.set_ylabel('Frequency')
        ax4.set_title(f'Error Distribution (MAE: {np.mean(errors):.4f})')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"预测可视化结果已保存到: {save_path}")
    
    if show_plot:
        plt.show()
    else:
        plt.close()

def compute_metrics(
    predictions: np.ndarray,
    ground_truth: np.ndarray,
    threshold: float = 0.5,
)
    """
    计算评估指标
    
    Args:
        predictions: 预测值
        ground_truth: 真实值
        threshold: 二值化阈值
        
    Returns:
        评估指标字典
    """
    # 基础回归指标
    mae = np.mean(np.abs(predictions - ground_truth))
    mse = np.mean((predictions - ground_truth) ** 2)
    rmse = np.sqrt(mse)
    
    # 二值化分类指标
    pred_binary = (predictions > threshold).astype(int)
    gt_binary = (ground_truth > threshold).astype(int)
    
    # 混淆矩阵
    tp = np.sum((pred_binary == 1) & (gt_binary == 1))
    fp = np.sum((pred_binary == 1) & (gt_binary == 0))
    tn = np.sum((pred_binary == 0) & (gt_binary == 0))
    fn = np.sum((pred_binary == 0) & (gt_binary == 1))
    
    # 分类指标
    accuracy = (tp + tn) / (tp + fp + tn + fn) if (tp + fp + tn + fn) > 0 else 0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    # IoU
    intersection = tp
    union = tp + fp + fn
    iou = intersection / union if union > 0 else 0
    
    return {
        'mae': float(
            mae,
        )
    } 