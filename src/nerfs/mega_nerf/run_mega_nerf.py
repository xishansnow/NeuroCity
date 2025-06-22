#!/usr/bin/env python3
"""
Mega-NeRF完整演示脚本
展示大规模场景的神经辐射场训练和渲染
"""

import os
import sys
import torch
import numpy as np
import matplotlib.pyplot as plt
import logging
from .mega_nerf import MegaNeRF, MegaNeRFConfig
from mega_nerf_trainer import MegaNeRFTrainer, InteractiveRenderer, create_sample_camera_path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def create_demo_scene():
    """创建演示场景数据"""
    logger.info("🏗️ 创建Mega-NeRF演示场景")
    
    # 配置参数 - 针对大规模场景优化
    config = MegaNeRFConfig(
        # 空间分解参数
        num_submodules=8,
        grid_size=(4, 2),  # 4x2网格分解
        overlap_factor=0.15,
        
        # 网络参数
        hidden_dim=256,
        num_layers=8,
        use_viewdirs=True,
        
        # 训练参数
        batch_size=512,
        learning_rate=5e-4,
        max_iterations=5000,  # 演示用较少迭代
        
        # 采样参数
        num_coarse=128,  # 减少采样点以加快演示
        num_fine=256,
        near=0.1,
        far=1000.0,
        
        # 外观嵌入
        use_appearance_embedding=True,
        appearance_dim=48,
        
        # 场景边界 - 大规模城市场景
        scene_bounds=(-200, -200, -20, 200, 200, 100),
        foreground_ratio=0.8
    )
    
    return config

def demonstrate_spatial_partitioning(config):
    """演示空间分解功能"""
    logger.info("📊 演示空间分解")
    
    # 创建模型
    model = MegaNeRF(config)
    
    # 可视化空间分解
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # 绘制场景边界和网格分解
    bounds = config.scene_bounds
    x_min, y_min, z_min, x_max, y_max, z_max = bounds
    
    # 网格中心点
    centroids = model.centroids
    
    # 左图：俯视图显示空间分解
    ax1.set_xlim(x_min, x_max)
    ax1.set_ylim(y_min, y_max)
    ax1.scatter(centroids[:, 0], centroids[:, 1], c='red', s=100, marker='s', alpha=0.7)
    
    # 绘制网格线
    grid_x, grid_y = config.grid_size
    x_lines = np.linspace(x_min, x_max, grid_x + 1)
    y_lines = np.linspace(y_min, y_max, grid_y + 1)
    
    for x in x_lines:
        ax1.axvline(x, color='blue', alpha=0.5, linestyle='--')
    for y in y_lines:
        ax1.axhline(y, color='blue', alpha=0.5, linestyle='--')
    
    ax1.set_xlabel('X (米)')
    ax1.set_ylabel('Y (米)')
    ax1.set_title('Mega-NeRF 空间分解 (俯视图)')
    ax1.grid(True, alpha=0.3)
    ax1.set_aspect('equal')
    
    # 右图：显示子模块参数分布
    param_counts = [sum(p.numel() for p in submodule.parameters()) for submodule in model.submodules]
    ax2.bar(range(len(param_counts)), param_counts, color='skyblue', alpha=0.7)
    ax2.set_xlabel('子模块索引')
    ax2.set_ylabel('参数数量')
    ax2.set_title('各子模块参数分布')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('output/mega_nerf/spatial_partitioning.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    logger.info(f"✅ 创建了 {len(model.submodules)} 个子模块")
    logger.info(f"📈 总参数数量: {sum(p.numel() for p in model.parameters()):,}")

def train_mega_nerf(config):
    """训练Mega-NeRF模型"""
    logger.info("🚀 开始训练Mega-NeRF")
    
    # 数据和输出路径
    data_dir = "data/mill19"
    output_dir = "outputs/mega_nerf"
    
    # 创建目录
    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)
    
    # 检查是否已有训练好的模型
    model_path = os.path.join(output_dir, "model.pth")
    
    if not os.path.exists(model_path):
        # 创建训练器
        trainer = MegaNeRFTrainer(config, data_dir, output_dir)
        
        # 显示训练信息
        logger.info(f"🎯 训练配置:")
        logger.info(f"   - 子模块数量: {len(trainer.model.submodules)}")
        logger.info(f"   - 网格大小: {config.grid_size}")
        logger.info(f"   - 最大迭代: {config.max_iterations}")
        logger.info(f"   - 批大小: {config.batch_size}")
        logger.info(f"   - 学习率: {config.learning_rate}")
        
        # 开始训练
        start_time = torch.cuda.Event(enable_timing=True) if torch.cuda.is_available() else None
        end_time = torch.cuda.Event(enable_timing=True) if torch.cuda.is_available() else None
        
        if start_time:
            start_time.record()
        
        trainer.train_sequential()
        
        if end_time:
            end_time.record()
            torch.cuda.synchronize()
            training_time = start_time.elapsed_time(end_time) / 1000.0  # 转换为秒
            logger.info(f"⏱️ 训练时间: {training_time:.2f} 秒")
        
        # 保存模型
        trainer.save_model(model_path)
        logger.info("💾 模型训练完成并保存")
    else:
        logger.info("📂 发现已训练的模型，跳过训练")
    
    return model_path

def render_and_visualize(config, model_path):
    """渲染和可视化结果"""
    logger.info("🎨 开始渲染和可视化")
    
    # 加载模型
    model = MegaNeRF(config)
    if os.path.exists(model_path):
        checkpoint = torch.load(model_path, map_location='cpu')
        model.load_state_dict(checkpoint['model_state_dict'])
        logger.info("✅ 模型加载成功")
    
    # 创建渲染器
    renderer = InteractiveRenderer(model, config)
    
    # 渲染多个视角
    logger.info("🖼️ 渲染多视角图像")
    
    # 创建不同的相机位置
    camera_positions = [
        ([0, 0, 50], "鸟瞰图"),
        ([100, 0, 30], "侧视图1"),
        ([0, 100, 30], "侧视图2"),
        ([70, 70, 40], "斜视图"),
        ([0, 0, 20], "低空视图")
    ]
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()
    
    for i, ((x, y, z), view_name) in enumerate(camera_positions):
        if i >= len(axes):
            break
            
        # 创建相机姿态
        camera_pose = np.eye(4)
        camera_pose[:3, 3] = [x, y, z]
        
        # 让相机朝向原点
        target = np.array([0, 0, 0])
        pos = np.array([x, y, z])
        up = np.array([0, 0, 1])
        
        forward = target - pos
        forward = forward / np.linalg.norm(forward)
        right = np.cross(forward, up)
        right = right / np.linalg.norm(right)
        up = np.cross(right, forward)
        
        camera_pose[:3, 0] = right
        camera_pose[:3, 1] = up
        camera_pose[:3, 2] = -forward
        
        # 渲染
        rgb = renderer.render_view(camera_pose, width=400, height=300)
        
        # 显示
        axes[i].imshow(rgb)
        axes[i].set_title(f'{view_name}\\n位置: ({x}, {y}, {z})')
        axes[i].axis('off')
    
    # 隐藏多余的子图
    for i in range(len(camera_positions), len(axes)):
        axes[i].axis('off')
    
    plt.suptitle('Mega-NeRF 多视角渲染结果', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig('output/mega_nerf/multi_view_renders.png', dpi=150, bbox_inches='tight')
    plt.show()

def create_flythrough_video(config, model_path):
    """创建飞行浏览视频"""
    logger.info("🎬 创建飞行浏览视频")
    
    # 加载模型
    model = MegaNeRF(config)
    if os.path.exists(model_path):
        checkpoint = torch.load(model_path, map_location='cpu')
        model.load_state_dict(checkpoint['model_state_dict'])
    
    # 创建渲染器
    renderer = InteractiveRenderer(model, config)
    
    # 创建复杂的飞行路径
    logger.info("📐 生成飞行路径")
    
    # 螺旋上升路径
    num_frames = 120
    angles = np.linspace(0, 4 * np.pi, num_frames)  # 两圈
    radius_start = 80
    radius_end = 40
    height_start = 10
    height_end = 60
    
    camera_path = []
    for i, angle in enumerate(angles):
        # 螺旋参数
        t = i / (num_frames - 1)
        radius = radius_start * (1 - t) + radius_end * t
        height = height_start * (1 - t) + height_end * t
        
        # 相机位置
        pos = np.array([
            radius * np.cos(angle),
            radius * np.sin(angle),
            height
        ])
        
        # 朝向中心，但稍微向上倾斜
        target = np.array([0, 0, height * 0.3])
        up = np.array([0, 0, 1])
        
        forward = target - pos
        forward = forward / np.linalg.norm(forward)
        right = np.cross(forward, up)
        right = right / np.linalg.norm(right)
        up = np.cross(right, forward)
        
        transform = np.eye(4)
        transform[:3, 0] = right
        transform[:3, 1] = up
        transform[:3, 2] = -forward
        transform[:3, 3] = pos
        
        camera_path.append(transform)
    
    # 渲染视频
    output_path = 'output/mega_nerf/mega_nerf_flythrough.mp4'
    renderer.create_flythrough(camera_path, output_path, fps=30)
    
    logger.info(f"🎥 飞行浏览视频已保存到: {output_path}")

def analyze_performance(config, model_path):
    """分析性能指标"""
    logger.info("📊 分析性能指标")
    
    # 加载模型
    model = MegaNeRF(config)
    if os.path.exists(model_path):
        checkpoint = torch.load(model_path, map_location='cpu')
        model.load_state_dict(checkpoint['model_state_dict'])
    
    # 创建渲染器
    renderer = InteractiveRenderer(model, config)
    
    # 性能测试
    camera_pose = np.eye(4)
    camera_pose[:3, 3] = [0, 0, 30]
    
    # 测试不同分辨率的渲染时间
    resolutions = [(200, 150), (400, 300), (800, 600), (1200, 900)]
    render_times = []
    
    logger.info("⏱️ 测试渲染性能")
    for width, height in resolutions:
        import time
        start_time = time.time()
        
        rgb = renderer.render_view(camera_pose, width=width, height=height)
        
        end_time = time.time()
        render_time = end_time - start_time
        render_times.append(render_time)
        
        logger.info(f"   {width}x{height}: {render_time:.3f}s")
    
    # 可视化性能
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # 渲染时间 vs 分辨率
    pixel_counts = [w * h for w, h in resolutions]
    ax1.plot(pixel_counts, render_times, 'bo-', linewidth=2, markersize=8)
    ax1.set_xlabel('像素数量')
    ax1.set_ylabel('渲染时间 (秒)')
    ax1.set_title('渲染性能分析')
    ax1.grid(True, alpha=0.3)
    
    # 模型复杂度分析
    param_counts = []
    module_names = []
    
    # 统计各部分参数
    for i, submodule in enumerate(model.submodules):
        params = sum(p.numel() for p in submodule.parameters())
        param_counts.append(params)
        module_names.append(f'子模块{i}')
    
    # 背景模块
    bg_params = sum(p.numel() for p in model.background_nerf.parameters())
    param_counts.append(bg_params)
    module_names.append('背景模块')
    
    ax2.bar(range(len(param_counts)), param_counts, color='lightcoral', alpha=0.7)
    ax2.set_xlabel('模块')
    ax2.set_ylabel('参数数量')
    ax2.set_title('模型复杂度分析')
    ax2.set_xticks(range(len(module_names)))
    ax2.set_xticklabels(module_names, rotation=45)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('output/mega_nerf/performance_analysis.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    # 输出总结
    total_params = sum(p.numel() for p in model.parameters())
    logger.info(f"📈 性能总结:")
    logger.info(f"   - 总参数数量: {total_params:,}")
    logger.info(f"   - 子模块数量: {len(model.submodules)}")
    logger.info(f"   - 平均渲染时间 (800x600): {render_times[2]:.3f}s")
    logger.info(f"   - 场景覆盖范围: {config.scene_bounds}")

def main():
    """主演示函数"""
    print("🌟 Mega-NeRF: 大规模神经辐射场演示")
    print("=" * 50)
    
    try:
        # 1. 创建演示场景
        config = create_demo_scene()
        
        # 2. 演示空间分解
        demonstrate_spatial_partitioning(config)
        
        # 3. 训练模型
        model_path = train_mega_nerf(config)
        
        # 4. 渲染和可视化
        render_and_visualize(config, model_path)
        
        # 5. 创建飞行浏览视频
        create_flythrough_video(config, model_path)
        
        # 6. 性能分析
        analyze_performance(config, model_path)
        
        print("\n🎉 Mega-NeRF演示完成！")
        print("📁 所有结果已保存到 output/mega_nerf/ 目录")
        print("\n📋 生成的文件:")
        print("   - spatial_partitioning.png: 空间分解可视化")
        print("   - multi_view_renders.png: 多视角渲染结果")
        print("   - mega_nerf_flythrough.mp4: 飞行浏览视频")
        print("   - performance_analysis.png: 性能分析图表")
        print("   - model.pth: 训练好的模型")
        
    except Exception as e:
        logger.error(f"❌ 演示过程中出现错误: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 