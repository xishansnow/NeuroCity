#!/usr/bin/env python3
"""
DataGen 数据生成软件包使用演示

展示如何使用DataGen包进行数据生成和采样。
"""

import numpy as np
import os
import sys

# 添加源码路径
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.datagen import DataGenConfig, DataGenPipeline
from src.datagen.generators import SDFGenerator, OccupancyGenerator
from src.datagen.samplers import VoxelSampler, SurfaceSampler, PointCloudSampler

def demo_basic_data_generation():
    """演示基础数据生成"""
    print("=== 基础数据生成演示 ===")
    
    # 创建配置
    config = DataGenConfig(
        output_dir="demo_outputs", voxel_size=0.5, grid_size=(
            200,
            200,
            100,
        )
    )
    
    # 创建数据生成管道
    pipeline = DataGenPipeline(config)
    
    # 生成球体数据
    print("生成球体数据...")
    coords, sdf_values = pipeline.generate_sphere_data(
        center=(0, 0, 25), radius=20.0, n_samples=10000
    )
    
    print(f"生成的球体数据: {coords.shape}, SDF范围: [{sdf_values.min():.2f}, {sdf_values.max():.2f}]")
    
    # 生成立方体数据
    print("生成立方体数据...")
    coords, sdf_values = pipeline.generate_box_data(
        center=(30, 30, 25), size=(15, 15, 15), n_samples=8000
    )
    
    print(f"生成的立方体数据: {coords.shape}, SDF范围: [{sdf_values.min():.2f}, {sdf_values.max():.2f}]")
    
    # 保存数据
    pipeline.save_training_data(coords, sdf_values, "demo_box_data")
    print("数据已保存到 demo_outputs/")


def demo_occupancy_generation():
    """演示占用网格生成"""
    print("\n=== 占用网格生成演示 ===")
    
    # 创建占用网格生成器
    generator = OccupancyGenerator(
        voxel_size=1.0, grid_bounds=(-50, -50, 0, 50, 50, 50)
    )
    
    # 生成球体占用网格
    print("生成球体占用网格...")
    sphere_occupancy = generator.generate_sphere_occupancy(
        center=(0, 0, 25), radius=15.0, filled=True
    )
    
    # 生成立方体占用网格
    print("生成立方体占用网格...")
    box_occupancy = generator.generate_box_occupancy(
        center=(20, 20, 25), size=(20, 20, 20), filled=True
    )
    
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
    
    # 保存占用网格
    os.makedirs("demo_outputs", exist_ok=True)
    generator.save_occupancy_grid(
        scene_occupancy, "demo_outputs/demo_scene.npy", metadata={
            'description': 'Demo complex scene',
        }
    )
    
    print(f"复合场景占用体素数: {np.sum(scene_occupancy)}")


def demo_sdf_generation():
    """演示SDF生成和训练"""
    print("\n=== SDF生成和训练演示 ===")
    
    # 创建SDF生成器
    sdf_gen = SDFGenerator(
        model_config={
            'hidden_dims': [128, 256, 256, 128], 'use_positional_encoding': True, 'encoding_freqs': 8
        }, learning_rate=1e-3
    )
    
    # 生成训练数据
    print("生成SDF训练数据...")
    geometry_configs = [
        {
            'type': 'sphere', 'params': {'center': [0, 0, 0], 'radius': 15.0}
        }, {
            'type': 'box', 'params': {'center': [20, 0, 0], 'size': [10, 10, 10]}
        }
    ]
    
    train_coords, train_sdf = sdf_gen.generate_training_data(
        scene_bounds=(
            -30,
            -30,
            -30,
            30,
            30,
            30,
        )
    )
    
    print(f"SDF训练数据: {train_coords.shape}, SDF范围: [{train_sdf.min():.2f}, {train_sdf.max():.2f}]")
    
    # 训练SDF网络（简短演示）
    print("训练SDF网络...")
    history = sdf_gen.train_sdf_network(
        train_coords, train_sdf, num_epochs=20, batch_size=512, verbose=True
    )
    
    print(f"训练完成，最终损失: {history['train_losses'][-1]:.6f}")
    
    # 保存模型
    os.makedirs("demo_outputs", exist_ok=True)
    sdf_gen.save_model("demo_outputs/demo_sdf_model.pth")


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
    
    # 多分辨率采样
    print("进行多分辨率采样...")
    multi_res_samples = surface_sampler.sample_multi_resolution(
        coords, occupancy, resolution_levels=[1.0, 0.5, 0.25], samples_per_level=1000
    )
    
    print(f"多分辨率采样结果: {multi_res_samples['coordinates'].shape}")


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
    print("DataGen 数据生成软件包演示")
    print("=" * 50)
    
    try:
        # 运行各个演示
        demo_basic_data_generation()
        demo_occupancy_generation()
        demo_sdf_generation()
        demo_surface_sampling()
        demo_point_cloud_sampling()
        
        print("\n=" * 50)
        print("所有演示完成！")
        print("输出文件保存在 demo_outputs/ 目录中")
        
    except Exception as e:
        print(f"演示过程中出现错误: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main() 