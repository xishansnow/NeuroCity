"""
SVRaster 耦合架构使用示例

这个示例展示了如何使用重构后的 SVRaster 架构：
1. SVRasterTrainer 与 VolumeRenderer 紧密耦合（训练阶段）
2. SVRasterRenderer 与 TrueVoxelRasterizer 紧密耦合（推理阶段）

这种设计确保了训练和推理阶段使用不同的渲染策略，符合 SVRaster 论文的设计思想。
"""

import torch
import numpy as np
from pathlib import Path
import logging

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def demo_training_pipeline():
    """
    演示训练流程 - SVRasterTrainer 与 VolumeRenderer 耦合
    """
    print("=" * 60)
    print("训练阶段：SVRasterTrainer ↔ VolumeRenderer")
    print("=" * 60)
    
    try:
        from src.nerfs.svraster.core import SVRasterModel, SVRasterConfig, VolumeRenderer
        from src.nerfs.svraster.trainer_refactored_coupled import SVRasterTrainer, SVRasterTrainerConfig
        
        # 1. 创建模型配置
        model_config = SVRasterConfig(
            grid_resolution=128,
            max_subdivision_level=8,
            density_activation="exp",
            color_activation="sigmoid"
        )
        
        # 2. 创建模型
        model = SVRasterModel(model_config)
        if torch.cuda.is_available():
            model = model.cuda()
        
        # 3. 创建体积渲染器（与训练器紧密耦合）
        volume_renderer = VolumeRenderer(model_config)
        
        # 4. 创建训练器配置
        trainer_config = SVRasterTrainerConfig(
            num_epochs=10,
            batch_size=1,
            learning_rate=1e-3,
            num_samples=64,
            num_importance_samples=128,
            background_color=(1.0, 1.0, 1.0)
        )
        
        # 5. 创建训练器（紧密耦合体积渲染器）
        trainer = SVRasterTrainer(
            model=model,
            volume_renderer=volume_renderer,
            config=trainer_config
        )
        
        print(f"✓ 训练器初始化成功")
        print(f"  - 模型参数数量: {sum(p.numel() for p in model.parameters()):,}")
        print(f"  - 设备: {model.device}")
        print(f"  - 体积渲染器: {type(volume_renderer).__name__}")
        print(f"  - 训练配置: {trainer_config.num_epochs} epochs")
        
        # 6. 创建模拟数据进行演示
        demo_batch = create_demo_training_batch(model.device)
        
        # 7. 执行一个训练步骤（演示）
        print(f"\n执行演示训练步骤...")
        losses = trainer.train_step(demo_batch)
        print(f"✓ 训练步骤完成")
        print(f"  - 损失: {losses}")
        
        # 8. 保存检查点
        trainer.save_checkpoint(0)
        print(f"✓ 检查点已保存")
        
    except ImportError as e:
        print(f"❌ 导入错误: {e}")
        print("请确保所有模块都已正确实现")
    except Exception as e:
        print(f"❌ 训练演示失败: {e}")


def demo_inference_pipeline():
    """
    演示推理流程 - SVRasterRenderer 与 TrueVoxelRasterizer 耦合
    """
    print("\n" + "=" * 60)
    print("推理阶段：SVRasterRenderer ↔ TrueVoxelRasterizer")
    print("=" * 60)
    
    try:
        from src.nerfs.svraster.renderer_coupled_final import (
            SVRasterRenderer, SVRasterRendererConfig, TrueVoxelRasterizerConfig
        )
        from src.nerfs.svraster.true_rasterizer import TrueVoxelRasterizer
        from src.nerfs.svraster.core import SVRasterModel, SVRasterConfig
        
        # 1. 检查检查点文件
        checkpoint_path = "checkpoints/checkpoint_epoch_0.pth"
        if not Path(checkpoint_path).exists():
            print(f"⚠️  检查点文件不存在: {checkpoint_path}")
            print("   创建演示检查点...")
            create_demo_checkpoint(checkpoint_path)
        
        # 2. 创建光栅化器配置
        rasterizer_config = TrueVoxelRasterizerConfig(
            background_color=(1.0, 1.0, 1.0),
            near_plane=0.1,
            far_plane=100.0,
            density_activation="exp",
            color_activation="sigmoid"
        )
        
        # 3. 创建渲染器配置
        renderer_config = SVRasterRendererConfig(
            image_width=400,
            image_height=300,
            render_batch_size=2048,
            background_color=(1.0, 1.0, 1.0)
        )
        
        # 4. 从检查点加载渲染器（紧密耦合光栅化器）
        renderer = SVRasterRenderer.from_checkpoint(
            checkpoint_path=checkpoint_path,
            rasterizer_config=rasterizer_config,
            renderer_config=renderer_config
        )
        
        print(f"✓ 渲染器初始化成功")
        print(f"  - 设备: {renderer.device}")
        print(f"  - 光栅化器: {type(renderer.rasterizer).__name__}")
        print(f"  - 渲染分辨率: {renderer_config.image_width}x{renderer_config.image_height}")
        
        # 5. 创建相机参数
        camera_pose = torch.eye(4, device=renderer.device)
        camera_pose[2, 3] = 5.0  # 向后移动相机
        
        intrinsics = torch.tensor([
            [400, 0, 200],
            [0, 400, 150],
            [0, 0, 1]
        ], dtype=torch.float32, device=renderer.device)
        
        # 6. 渲染单张图像
        print(f"\n执行演示渲染...")
        result = renderer.render_image(camera_pose, intrinsics)
        
        print(f"✓ 渲染完成")
        print(f"  - RGB 形状: {result['rgb'].shape}")
        print(f"  - 深度形状: {result['depth'].shape}")
        print(f"  - RGB 范围: [{result['rgb'].min():.3f}, {result['rgb'].max():.3f}]")
        print(f"  - 深度范围: [{result['depth'].min():.3f}, {result['depth'].max():.3f}]")
        
        # 7. 批量渲染演示
        print(f"\n执行批量渲染演示...")
        num_views = 4
        poses = create_demo_camera_trajectory(num_views, renderer.device)
        
        renders = renderer.render_batch(poses, intrinsics)
        print(f"✓ 批量渲染完成")
        print(f"  - 渲染图像数量: {len(renders)}")
        
        # 8. 保存渲染结果
        output_dir = "demo_outputs/coupled_architecture"
        renderer.save_renders(renders, output_dir, "coupled_demo")
        print(f"✓ 渲染结果已保存到: {output_dir}")
        
        # 9. 内存使用情况
        memory_info = renderer.get_memory_usage()
        if memory_info:
            print(f"\n内存使用情况:")
            for key, value in memory_info.items():
                print(f"  - {key}: {value:.2f}")
        
    except ImportError as e:
        print(f"❌ 导入错误: {e}")
        print("请确保所有模块都已正确实现")
    except Exception as e:
        print(f"❌ 推理演示失败: {e}")


def create_demo_training_batch(device: torch.device) -> dict:
    """
    创建演示训练批次
    """
    batch_size = 1
    num_rays = 1024
    
    # 创建随机光线
    rays_o = torch.randn(batch_size, num_rays, 3, device=device)
    rays_d = torch.randn(batch_size, num_rays, 3, device=device)
    rays_d = rays_d / torch.norm(rays_d, dim=-1, keepdim=True)  # 归一化方向
    
    # 创建目标颜色
    target_rgb = torch.rand(batch_size, num_rays, 3, device=device)
    
    return {
        'rays_o': rays_o,
        'rays_d': rays_d,
        'target_rgb': target_rgb
    }


def create_demo_checkpoint(checkpoint_path: str):
    """
    创建演示检查点文件
    """
    from src.nerfs.svraster.core import SVRasterModel, SVRasterConfig
    
    # 创建模型
    model_config = SVRasterConfig()
    model = SVRasterModel(model_config)
    
    # 创建检查点
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'model_config': model_config,
        'epoch': 0,
        'global_step': 0
    }
    
    # 确保目录存在
    Path(checkpoint_path).parent.mkdir(parents=True, exist_ok=True)
    
    # 保存检查点
    torch.save(checkpoint, checkpoint_path)
    print(f"✓ 演示检查点已创建: {checkpoint_path}")


def create_demo_camera_trajectory(num_views: int, device: torch.device) -> torch.Tensor:
    """
    创建演示相机轨迹
    """
    poses = []
    
    for i in range(num_views):
        angle = 2 * np.pi * i / num_views
        
        # 创建围绕原点的相机位置
        x = 3.0 * np.cos(angle)
        y = 3.0 * np.sin(angle)
        z = 1.0
        
        # 创建相机姿态矩阵
        pose = torch.eye(4, device=device)
        pose[0, 3] = x
        pose[1, 3] = y
        pose[2, 3] = z
        
        poses.append(pose)
    
    return torch.stack(poses)


def main():
    """
    主函数 - 展示完整的耦合架构
    """
    print("SVRaster 耦合架构演示")
    print("1. 训练阶段：SVRasterTrainer ↔ VolumeRenderer")
    print("2. 推理阶段：SVRasterRenderer ↔ TrueVoxelRasterizer")
    print()
    
    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    print()
    
    # 演示训练流程
    demo_training_pipeline()
    
    # 演示推理流程
    demo_inference_pipeline()
    
    print("\n" + "=" * 60)
    print("架构耦合设计总结：")
    print("=" * 60)
    print("✓ SVRasterTrainer 与 VolumeRenderer 紧密耦合")
    print("  - 训练阶段使用体积渲染")
    print("  - 支持梯度优化和损失计算")
    print("  - 符合 NeRF 训练范式")
    print()
    print("✓ SVRasterRenderer 与 TrueVoxelRasterizer 紧密耦合")
    print("  - 推理阶段使用光栅化")
    print("  - 快速高效的渲染性能")
    print("  - 符合 SVRaster 论文设计")
    print()
    print("✓ 清晰的职责分离和模块化设计")
    print("✓ 符合 SVRaster 论文的核心思想")
    print("✓ 易于维护和扩展")


if __name__ == "__main__":
    main()
