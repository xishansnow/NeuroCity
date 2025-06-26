#!/usr/bin/env python3
"""
DataGen 数据生成软件包简化演示

展示基本的占用网格生成功能。
"""

import numpy as np
import os
import sys

# 添加源码路径
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.datagen.generators import OccupancyGenerator
from src.datagen.samplers import SurfaceSampler, PointCloudSampler

def demo_occupancy_generation():
    """演示占用网格生成"""
    print("=== 占用网格生成演示 ===")
    
    # 创建占用网格生成器
    generator = OccupancyGenerator(
        voxel_size=1.0, grid_bounds=(-50, -50, 0, 50, 50, 50)
    )
    
    # 生成球体占用网格
    print("生成球体占用网格...")
    sphere_occupancy = generator.generate_sphere_occupancy(
        center=(0, 0, 25), radius=15.0, filled=True
    )
    print(f"球体占用体素数: {np.sum(sphere_occupancy)}")
    
    # 生成立方体占用网格
    print("生成立方体占用网格...")
    box_occupancy = generator.generate_box_occupancy(
        center=(20, 20, 25), size=(20, 20, 20), filled=True
    )
    print(f"立方体占用体素数: {np.sum(box_occupancy)}")
    
    # 生成复合场景
    print("生成复合场景...")
    objects = [
        {
            'type': 'sphere', 'params': {'center': (-15, -15, 20), 'radius': 12, 'filled': True}
        }, {
            'type': 'cylinder', 'params': {
                'center': (15,
                -15,
                25),
                'radius': 8,
                'height': 30,
                'filled': True,
            }
        }
    ]
    
    scene_occupancy = generator.generate_complex_scene(objects, 'union')
    print(f"复合场景占用体素数: {np.sum(scene_occupancy)}")
    
    # 保存占用网格
    os.makedirs("demo_outputs", exist_ok=True)
    generator.save_occupancy_grid(
        scene_occupancy, "demo_outputs/demo_scene.npy", metadata={
            'description': 'Demo complex scene',
        }
    )
    
    print("场景数据已保存到 demo_outputs/demo_scene.npy")


def demo_surface_sampling():
    """演示表面采样"""
    print("\n=== 表面采样演示 ===")
    
    # 创建表面采样器
    surface_sampler = SurfaceSampler(
        surface_threshold=0.5, sampling_radius=3.0, adaptive_sampling=True
    )
    
    # 生成一些测试数据
    n_points = 5000
    coords = np.random.uniform(-20, 20, (n_points, 3))
    
    # 计算到球心的距离作为占用值
    distances = np.linalg.norm(coords, axis=1)
    occupancy = np.exp(-distances / 10.0)  # 软边界
    
    print(f"测试数据: {coords.shape}, 占用值范围: [{occupancy.min():.2f}, {occupancy.max():.2f}]")
    
    # 表面附近采样
    print("进行表面采样...")
    surface_samples = surface_sampler.sample_near_surface(
        coords, occupancy, n_samples=2000, noise_std=1.5
    )
    
    print(f"表面采样结果: {surface_samples['coordinates'].shape}")
    print(f"检测到的表面点数: {surface_samples['surface_points'].shape[0]}")


def demo_point_cloud_sampling():
    """演示点云采样"""
    print("\n=== 点云采样演示 ===")
    
    # 创建点云采样器
    pc_sampler = PointCloudSampler(
        downsample_ratio=0.1, noise_level=0.05, normal_estimation=True
    )
    
    # 生成模拟点云数据（球面点云）
    n_points = 8000
    theta = np.random.uniform(0, 2*np.pi, n_points)
    phi = np.random.uniform(0, np.pi, n_points)
    radius = 10 + np.random.normal(0, 0.5, n_points)
    
    points = np.column_stack([
        radius * np.sin(
            phi,
        )
    ])
    
    print(f"模拟点云数据: {points.shape}")
    
    # 从点云采样
    print("从点云采样...")
    pc_samples = pc_sampler.sample_from_point_cloud(
        points, n_samples=3000
    )
    
    print(f"点云采样结果: {pc_samples['coordinates'].shape}")
    if 'normals' in pc_samples:
        print(f"估计的法向量: {pc_samples['normals'].shape}")
    
    # 密度采样
    print("进行密度采样...")
    density_samples = pc_sampler.density_based_sampling(
        points, radius=3.0, min_samples=10
    )
    
    print(f"密度采样结果: {density_samples['coordinates'].shape}")
    print(f"密度标签分布: {np.bincount(density_samples['density_labels'])}")


def main():
    """主函数"""
    print("DataGen 数据生成软件包简化演示")
    print("=" * 50)
    
    try:
        # 运行各个演示
        demo_occupancy_generation()
        demo_surface_sampling()
        demo_point_cloud_sampling()
        
        print("\n=" * 50)
        print("简化演示完成！")
        print("输出文件保存在 demo_outputs/ 目录中")
        
    except Exception as e:
        print(f"演示过程中出现错误: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main() 