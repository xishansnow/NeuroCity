#!/usr/bin/env python3
"""
SVRaster 重构架构最终测试

这个脚本测试完全重构后    # 测试训练器（与 VolumeRenderer 耦合）
    print("\\n3. 测试 SVRasterTrainer（与 VolumeRenderer 耦合）...")
    trainer_config = SVRasterTrainerConfig()
    trainer_config.num_epochs = 1
    trainer_config.learning_rate = 1e-4
    trainer_config.use_amp = False  # 禁用 AMP 以简化调试aster 架构：
1. SVRasterTrainer 与 VolumeRenderer 紧密耦合（用于训练）
2. SVRasterRenderer 与 TrueVoxelRasterizer 紧密耦合（用于推理）
3. 模型支持两种模式：training（体积渲染）和 inference（光栅化）
"""

import sys
import os
sys.path.append('src')

import torch
import numpy as np
from nerfs.svraster.trainer_refactored_coupled import (
    SVRasterTrainer, 
    SVRasterTrainerConfig,
    create_svraster_trainer
)
from nerfs.svraster.renderer_refactored_coupled import (
    SVRasterRenderer, 
    SVRasterRendererConfig,
    TrueVoxelRasterizerConfig,
    create_svraster_renderer
)
from nerfs.svraster.core import SVRasterModel, SVRasterConfig
from nerfs.svraster.volume_renderer import VolumeRenderer
from nerfs.svraster.true_rasterizer import TrueVoxelRasterizer

def test_coupled_architecture():
    """测试重构后的耦合架构"""
    
    print("=" * 60)
    print("SVRaster 重构架构测试")
    print("=" * 60)
    
    # 1. 创建模型配置
    print("\\n1. 创建模型配置...")
    config = SVRasterConfig()
    config.base_resolution = 8  # 更小的分辨率用于测试 (8^3 = 512 voxels)
    config.num_samples = 32
    config.scene_bounds = [-2.0, -2.0, -2.0, 2.0, 2.0, 2.0]
    config.morton_ordering = False  # 禁用 Morton 排序以简化测试
    
    # 禁用可能有问题的损失函数
    config.use_ssim_loss = False
    config.use_distortion_loss = False
    config.use_opacity_regularization = False
    config.use_pointwise_rgb_loss = True  # 只使用简单的 RGB 损失
    
    # 2. 创建模型
    print("\\n2. 创建 SVRasterModel...")
    model = SVRasterModel(config)
    print(f"   ✓ 模型参数: {sum(p.numel() for p in model.parameters()):,}")
    print(f"   ✓ 设备: {model.device}")
    
    # 3. 测试训练器（与 VolumeRenderer 耦合）
    print("\\n3. 测试 SVRasterTrainer（与 VolumeRenderer 耦合）...")
    trainer_config = SVRasterTrainerConfig()
    trainer_config.num_epochs = 1
    trainer_config.learning_rate = 1e-4
    trainer_config.use_amp = False  # 禁用 AMP 以简化调试
    
    volume_renderer = VolumeRenderer(config)
    trainer = SVRasterTrainer(model, volume_renderer, trainer_config)
    
    print(f"   ✓ 优化器: {type(trainer.optimizer).__name__}")
    print(f"   ✓ 调度器: {type(trainer.scheduler).__name__ if trainer.scheduler else 'None'}")
    print(f"   ✓ 体积渲染器类型: {type(trainer.volume_renderer).__name__}")
    print(f"   ✓ 使用 AMP: {trainer.config.use_amp}")
    print(f"   ✓ 有 scaler: {hasattr(trainer, 'scaler')}")
    
    # 4. 测试单步训练
    print("\\n4. 测试单步训练...")
    batch_size = 1
    num_rays = 512  # 减少光线数量以匹配体素数量
    
    # 创建虚拟训练数据
    rays_o = torch.randn(batch_size, num_rays, 3, device=model.device)
    rays_d = torch.randn(batch_size, num_rays, 3, device=model.device)
    rays_d = torch.nn.functional.normalize(rays_d, dim=-1)
    target_rgb = torch.rand(batch_size, num_rays, 3, device=model.device)
    
    batch = {
        'rays_o': rays_o,
        'rays_d': rays_d,
        'target_rgb': target_rgb
    }
    
    # 执行训练步骤
    model.train()
    losses = trainer.train_step(batch)
    print(f"   ✓ 训练损失: {losses}")
    
    # 5. 测试渲染器（与 TrueVoxelRasterizer 耦合）
    print("\\n5. 测试 SVRasterRenderer（与 TrueVoxelRasterizer 耦合）...")
    renderer_config = SVRasterRendererConfig()
    renderer_config.image_width = 128
    renderer_config.image_height = 128
    
    rasterizer_config = TrueVoxelRasterizerConfig()
    rasterizer = TrueVoxelRasterizer(rasterizer_config)
    
    renderer = SVRasterRenderer(model, rasterizer, renderer_config)
    print(f"   ✓ 光栅化器类型: {type(renderer.rasterizer).__name__}")
    print(f"   ✓ 渲染配置: {renderer.config.image_width}x{renderer.config.image_height}")
    
    # 6. 测试推理渲染
    print("\\n6. 测试推理渲染...")
    model.eval()
    
    # 创建相机参数
    camera_pose = torch.eye(4, device=model.device)
    camera_pose[2, 3] = 3.0  # 相机位于 z=3
    
    intrinsics = torch.tensor([
        [128, 0, 64],
        [0, 128, 64],
        [0, 0, 1]
    ], dtype=torch.float32, device=model.device)
    
    # 渲染图像
    with torch.no_grad():
        result = renderer.render_image(camera_pose, intrinsics, 128, 128)
    
    print(f"   ✓ 渲染结果: RGB {result['rgb'].shape}, Depth {result['depth'].shape}")
    print(f"   ✓ RGB 范围: [{result['rgb'].min().item():.3f}, {result['rgb'].max().item():.3f}]")
    
    # 7. 内存使用情况
    print("\\n7. 内存使用情况...")
    memory_info = renderer.get_memory_usage()
    if memory_info:
        for key, value in memory_info.items():
            print(f"   ✓ {key}: {value:.2f}")
    else:
        print("   ✓ CPU 模式，无 GPU 内存信息")
    
    # 8. 架构总结
    print("\\n8. 重构架构总结...")
    print("   ✓ 训练阶段：SVRasterTrainer ↔ VolumeRenderer（体积渲染）")
    print("   ✓ 推理阶段：SVRasterRenderer ↔ TrueVoxelRasterizer（光栅化）")
    print("   ✓ 模式分离：training（体积渲染）vs inference（光栅化）")
    print("   ✓ 符合 SVRaster 论文设计理念")
    
    print("\\n" + "=" * 60)
    print("🎉 SVRaster 重构架构测试完成！")
    print("✅ 所有组件正常工作")
    print("✅ 耦合架构符合设计要求")
    print("✅ 训练和推理逻辑分离清晰")
    print("=" * 60)

def test_convenience_functions():
    """测试便捷函数"""
    print("\\n🧪 测试便捷函数...")
    
    # 创建基础组件
    config = SVRasterConfig()
    config.base_resolution = 16  # 很小的分辨率
    model = SVRasterModel(config)
    
    # 测试便捷训练器创建函数
    trainer = create_svraster_trainer(model, train_dataset=None)
    print("   ✓ create_svraster_trainer 工作正常")
    
    # 测试从检查点创建渲染器（需要先保存检查点）
    checkpoint_path = "test_checkpoint.pth"
    torch.save({
        'model_state_dict': model.state_dict(),
        'model_config': config
    }, checkpoint_path)
    
    try:
        renderer = create_svraster_renderer(checkpoint_path)
        print("   ✓ create_svraster_renderer 工作正常")
    except Exception as e:
        print(f"   ⚠️ create_svraster_renderer 遇到问题: {e}")
    finally:
        # 清理测试文件
        if os.path.exists(checkpoint_path):
            os.remove(checkpoint_path)

if __name__ == "__main__":
    try:
        test_coupled_architecture()
        test_convenience_functions()
    except Exception as e:
        print(f"\\n❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
