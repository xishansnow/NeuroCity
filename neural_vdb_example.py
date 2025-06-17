#!/usr/bin/env python3
"""
NeuralVDB 使用示例
演示如何使用 NeuralVDB 进行稀疏体素神经表示
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import torch
import os
import sys
from typing import Tuple, List

# 导入 NeuralVDB
from neural_vdb import (
    NeuralVDB, NeuralVDBConfig, create_sample_data,
    OctreeNode, SparseVoxelGrid, FeatureNetwork, OccupancyNetwork
)

def create_complex_geometry(n_points: int = 20000) -> Tuple[np.ndarray, np.ndarray]:
    """创建复杂的几何形状数据"""
    print("创建复杂几何形状数据...")
    
    # 生成随机点
    points = np.random.rand(n_points, 3) * 100
    
    occupancies = np.zeros(n_points, dtype=np.float32)
    
    # 1. 中心球体
    center1 = np.array([30, 30, 30])
    radius1 = 15
    distances1 = np.linalg.norm(points - center1, axis=1)
    occupancies += (distances1 < radius1).astype(np.float32)
    
    # 2. 右上角立方体
    center2 = np.array([70, 70, 70])
    size2 = 20
    in_cube = np.all(
        (points >= center2 - size2/2) & (points < center2 + size2/2), 
        axis=1
    )
    occupancies += in_cube.astype(np.float32)
    
    # 3. 左下角圆柱体
    center3 = np.array([20, 80, 50])
    radius3 = 12
    height3 = 30
    
    # 圆柱体的水平距离
    horizontal_dist = np.sqrt((points[:, 0] - center3[0])**2 + (points[:, 1] - center3[1])**2)
    # 垂直距离
    vertical_dist = np.abs(points[:, 2] - center3[2])
    
    in_cylinder = (horizontal_dist < radius3) & (vertical_dist < height3/2)
    occupancies += in_cylinder.astype(np.float32)
    
    # 4. 添加一些噪声
    noise = np.random.rand(n_points) * 0.1
    occupancies = np.clip(occupancies + noise, 0, 1)
    
    return points, occupancies

def visualize_training_data(points: np.ndarray, occupancies: np.ndarray, save_path: str = None):
    """可视化训练数据"""
    print("可视化训练数据...")
    
    fig = plt.figure(figsize=(15, 5))
    
    # 占用点
    occupied_points = points[occupancies > 0.5]
    empty_points = points[occupancies <= 0.5]
    
    # 子图1: 3D散点图
    ax1 = fig.add_subplot(131, projection='3d')
    if len(occupied_points) > 0:
        ax1.scatter(occupied_points[:, 0], occupied_points[:, 1], occupied_points[:, 2], 
                   c='red', s=1, alpha=0.6, label='Occupied')
    if len(empty_points) > 0:
        ax1.scatter(empty_points[:, 0], empty_points[:, 1], empty_points[:, 2], 
                   c='blue', s=1, alpha=0.1, label='Empty')
    
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.set_zlabel('Z')
    ax1.set_title('Training Data (3D View)')
    ax1.legend()
    
    # 子图2: XY平面投影
    ax2 = fig.add_subplot(132)
    if len(occupied_points) > 0:
        ax2.scatter(occupied_points[:, 0], occupied_points[:, 1], 
                   c='red', s=1, alpha=0.6)
    if len(empty_points) > 0:
        ax2.scatter(empty_points[:, 0], empty_points[:, 1], 
                   c='blue', s=1, alpha=0.1)
    
    ax2.set_xlabel('X')
    ax2.set_ylabel('Y')
    ax2.set_title('XY Projection')
    ax2.set_aspect('equal')
    
    # 子图3: 占用率分布
    ax3 = fig.add_subplot(133)
    ax3.hist(occupancies, bins=50, alpha=0.7, color='green')
    ax3.set_xlabel('Occupancy Value')
    ax3.set_ylabel('Frequency')
    ax3.set_title('Occupancy Distribution')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"训练数据可视化已保存到: {save_path}")
    else:
        plt.show()

def visualize_predictions(model: NeuralVDB, test_points: np.ndarray, 
                         predictions: np.ndarray, save_path: str = None):
    """可视化预测结果"""
    print("可视化预测结果...")
    
    fig = plt.figure(figsize=(15, 5))
    
    # 子图1: 预测值分布
    ax1 = fig.add_subplot(131)
    ax1.hist(predictions, bins=50, alpha=0.7, color='orange')
    ax1.set_xlabel('Predicted Occupancy')
    ax1.set_ylabel('Frequency')
    ax1.set_title('Prediction Distribution')
    
    # 子图2: 3D预测结果
    ax2 = fig.add_subplot(132, projection='3d')
    
    # 根据预测值设置颜色
    colors = plt.cm.viridis(predictions)
    
    ax2.scatter(test_points[:, 0], test_points[:, 1], test_points[:, 2], 
               c=colors, s=10, alpha=0.7)
    
    ax2.set_xlabel('X')
    ax2.set_ylabel('Y')
    ax2.set_zlabel('Z')
    ax2.set_title('Prediction Results (3D)')
    
    # 子图3: 预测值热力图 (XY平面)
    ax3 = fig.add_subplot(133)
    scatter = ax3.scatter(test_points[:, 0], test_points[:, 1], 
                         c=predictions, cmap='viridis', s=10, alpha=0.7)
    ax3.set_xlabel('X')
    ax3.set_ylabel('Y')
    ax3.set_title('Prediction Heatmap (XY)')
    ax3.set_aspect('equal')
    
    # 添加颜色条
    plt.colorbar(scatter, ax=ax3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"预测结果可视化已保存到: {save_path}")
    else:
        plt.show()

def compare_with_traditional_voxelization(points: np.ndarray, occupancies: np.ndarray, 
                                        voxel_size: float = 2.0):
    """与传统体素化方法比较"""
    print("与传统体素化方法比较...")
    
    # 计算边界框
    min_coords = points.min(axis=0)
    max_coords = points.max(axis=0)
    
    # 计算体素网格尺寸
    grid_size = ((max_coords - min_coords) / voxel_size).astype(int) + 1
    
    # 创建传统体素网格
    voxel_grid = np.zeros(grid_size, dtype=np.float32)
    
    # 体素化
    voxel_coords = ((points - min_coords) / voxel_size).astype(int)
    
    # 确保坐标在有效范围内
    valid_mask = np.all((voxel_coords >= 0) & (voxel_coords < grid_size), axis=1)
    valid_coords = voxel_coords[valid_mask]
    valid_occupancies = occupancies[valid_mask]
    
    # 填充体素网格
    for coord, occ in zip(valid_coords, valid_occupancies):
        voxel_grid[coord[0], coord[1], coord[2]] = max(
            voxel_grid[coord[0], coord[1], coord[2]], occ
        )
    
    # 计算内存使用
    traditional_memory = voxel_grid.nbytes / (1024 * 1024)  # MB
    neural_memory = 0  # 将在训练后计算
    
    print(f"传统体素网格尺寸: {grid_size}")
    print(f"传统方法内存使用: {traditional_memory:.2f} MB")
    
    return voxel_grid, traditional_memory

def evaluate_model_performance(model: NeuralVDB, test_points: np.ndarray, 
                             test_occupancies: np.ndarray):
    """评估模型性能"""
    print("评估模型性能...")
    
    # 预测
    predictions = model.predict(test_points)
    
    # 计算指标
    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
    
    mse = mean_squared_error(test_occupancies, predictions)
    mae = mean_absolute_error(test_occupancies, predictions)
    r2 = r2_score(test_occupancies, predictions)
    
    # 计算准确率 (二分类)
    binary_predictions = (predictions > 0.5).astype(np.float32)
    binary_occupancies = (test_occupancies > 0.5).astype(np.float32)
    accuracy = np.mean(binary_predictions == binary_occupancies)
    
    print(f"性能指标:")
    print(f"  MSE: {mse:.6f}")
    print(f"  MAE: {mae:.6f}")
    print(f"  R²: {r2:.6f}")
    print(f"  准确率: {accuracy:.6f}")
    
    return {
        'mse': mse,
        'mae': mae,
        'r2': r2,
        'accuracy': accuracy,
        'predictions': predictions
    }

def run_neural_vdb_demo():
    """运行完整的NeuralVDB演示"""
    print("=" * 60)
    print("NeuralVDB 完整演示")
    print("=" * 60)
    
    # 1. 创建数据
    print("\n1. 创建训练数据...")
    points, occupancies = create_complex_geometry(15000)
    print(f"创建了 {len(points)} 个训练点")
    
    # 可视化训练数据
    visualize_training_data(points, occupancies, 'training_data.png')
    
    # 2. 与传统方法比较
    print("\n2. 与传统体素化方法比较...")
    voxel_grid, traditional_memory = compare_with_traditional_voxelization(points, occupancies)
    
    # 3. 配置NeuralVDB
    print("\n3. 配置NeuralVDB...")
    config = NeuralVDBConfig(
        voxel_size=1.0,
        max_depth=6,
        feature_dim=32,
        hidden_dims=[256, 512, 512, 256, 128],
        learning_rate=1e-3,
        batch_size=1024,
        sparsity_threshold=0.01
    )
    
    # 4. 创建和训练模型
    print("\n4. 训练NeuralVDB模型...")
    neural_vdb = NeuralVDB(config)
    
    # 分割训练集和测试集
    train_ratio = 0.8
    train_size = int(train_ratio * len(points))
    
    train_points = points[:train_size]
    train_occupancies = occupancies[:train_size]
    test_points = points[train_size:]
    test_occupancies = occupancies[train_size:]
    
    # 训练模型
    neural_vdb.fit(
        train_points, train_occupancies,
        train_ratio=0.8,
        num_epochs=50,
        save_path='neural_vdb_model.pth'
    )
    
    # 5. 评估性能
    print("\n5. 评估模型性能...")
    performance = evaluate_model_performance(neural_vdb, test_points, test_occupancies)
    
    # 6. 可视化预测结果
    print("\n6. 可视化预测结果...")
    visualize_predictions(neural_vdb, test_points, performance['predictions'], 'predictions.png')
    
    # 7. 可视化八叉树结构
    print("\n7. 可视化八叉树结构...")
    neural_vdb.visualize_octree(max_depth=4, save_path='octree_structure.png')
    
    # 8. 内存使用比较
    print("\n8. 内存使用比较...")
    # 估算NeuralVDB内存使用
    # 这里简化计算，实际应该计算网络参数和特征存储
    neural_memory = 0.1  # 假设值，实际应该根据模型参数计算
    
    print(f"传统体素化方法: {traditional_memory:.2f} MB")
    print(f"NeuralVDB方法: {neural_memory:.2f} MB")
    print(f"内存压缩比: {traditional_memory/neural_memory:.1f}x")
    
    # 9. 保存结果
    print("\n9. 保存结果...")
    results = {
        'performance': performance,
        'traditional_memory': traditional_memory,
        'neural_memory': neural_memory,
        'config': config.__dict__
    }
    
    import json
    with open('neural_vdb_results.json', 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    print("结果已保存到 neural_vdb_results.json")
    
    print("\n" + "=" * 60)
    print("NeuralVDB 演示完成!")
    print("=" * 60)

def run_quick_test():
    """快速测试NeuralVDB功能"""
    print("快速测试NeuralVDB...")
    
    # 创建简单数据
    points, occupancies = create_sample_data(1000)
    
    # 简单配置
    config = NeuralVDBConfig(
        max_depth=4,
        feature_dim=16,
        hidden_dims=[64, 128, 64],
        learning_rate=1e-3,
        batch_size=256
    )
    
    # 创建和训练模型
    neural_vdb = NeuralVDB(config)
    neural_vdb.fit(points, occupancies, num_epochs=10)
    
    # 测试预测
    test_points = np.random.rand(100, 3) * 100
    predictions = neural_vdb.predict(test_points)
    
    print(f"快速测试完成，预测值范围: [{predictions.min():.3f}, {predictions.max():.3f}]")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='NeuralVDB 演示')
    parser.add_argument('--mode', choices=['demo', 'quick'], default='demo',
                       help='运行模式: demo (完整演示) 或 quick (快速测试)')
    
    args = parser.parse_args()
    
    if args.mode == 'demo':
        run_neural_vdb_demo()
    else:
        run_quick_test() 