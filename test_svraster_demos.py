#!/usr/bin/env python3
"""
SVRaster 演示脚本快速测试

快速验证两个演示脚本的核心功能是否正常工作
"""

from __future__ import annotations

import sys
import torch
import time
from pathlib import Path

# 添加路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / 'demos'))

def test_training_demo():
    """测试训练演示的核心功能"""
    print("=== 测试训练演示 ===")
    
    try:
        from demo_svraster_training import SVRasterTrainingDemo
        
        # 创建演示
        demo = SVRasterTrainingDemo()
        print(f"✅ 训练演示创建成功 (设备: {demo.device})")
        
        # 快速设置（使用更小的配置）
        demo.config.base_resolution = 16  # 减小网格尺寸
        demo.config.image_width = 64
        demo.config.image_height = 48
        
        # 设置训练组件
        demo.setup_training()
        print("✅ 训练组件设置成功")
        print(f"   模型参数: {sum(p.numel() for p in demo.model.parameters()):,}")
        
        # 测试单个训练步骤
        batch_data = demo._get_training_batch(0)
        print("✅ 训练批次数据生成成功")
        print(f"   光线数量: {len(batch_data['ray_origins'])}")
        
        # 测试推理
        with torch.no_grad():
            outputs = demo.model(
                batch_data['ray_origins'][:100],  # 只测试100条光线
                batch_data['ray_directions'][:100],
                mode="training"
            )
        
        print("✅ 体积渲染测试成功")
        print(f"   输出键: {list(outputs.keys())}")
        
        return True
        
    except Exception as e:
        print(f"❌ 训练演示测试失败: {e}")
        return False


def test_rendering_demo():
    """测试渲染演示的核心功能"""
    print("\n=== 测试渲染演示 ===")
    
    try:
        from demo_svraster_rendering import SVRasterRenderingDemo
        
        # 创建演示
        demo = SVRasterRenderingDemo()
        print(f"✅ 渲染演示创建成功 (设备: {demo.device})")
        
        # 快速设置（使用更小的配置）
        demo.model_config.base_resolution = 32  # 减小网格尺寸
        demo.model_config.image_width = 128
        demo.model_config.image_height = 96
        demo.render_config.image_width = 128
        demo.render_config.image_height = 96
        demo.render_config.render_batch_size = 1024
        
        # 设置模型和渲染器
        demo.setup_model_and_renderers()
        print("✅ 模型和渲染器设置成功")
        print(f"   模型参数: {sum(p.numel() for p in demo.model.parameters()):,}")
        
        # 测试光线生成
        import numpy as np
        camera_pos = np.array([2.0, 0.0, 1.0])
        target = np.array([0.0, 0.0, 0.0])
        camera_forward = target - camera_pos
        camera_forward = camera_forward / np.linalg.norm(camera_forward)
        
        ray_origins, ray_directions = demo.generate_ray_batch(
            camera_pos, camera_forward, subset_ratio=0.1  # 只测试10%的光线
        )
        
        print("✅ 光线生成测试成功")
        print(f"   光线数量: {len(ray_origins)}")
        
        # 测试体积渲染
        start_time = time.time()
        with torch.no_grad():
            volume_outputs = demo.model(ray_origins, ray_directions, mode="training")
        volume_time = time.time() - start_time
        
        print("✅ 体积渲染测试成功")
        print(f"   渲染时间: {volume_time:.4f}s")
        
        # 测试光栅化渲染
        start_time = time.time()
        with torch.no_grad():
            raster_outputs = demo.model(ray_origins, ray_directions, mode="inference")
        raster_time = time.time() - start_time
        
        print("✅ 光栅化渲染测试成功")
        print(f"   渲染时间: {raster_time:.4f}s")
        
        # 计算加速比
        if raster_time > 0:
            speedup = volume_time / raster_time
            print(f"   加速比: {speedup:.2f}x")
        
        return True
        
    except Exception as e:
        print(f"❌ 渲染演示测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """主测试函数"""
    print("🧪 SVRaster 演示脚本快速测试")
    print("=" * 50)
    
    # 检查设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    
    # 测试训练演示
    training_ok = test_training_demo()
    
    # 测试渲染演示
    rendering_ok = test_rendering_demo()
    
    # 总结
    print("\n" + "=" * 50)
    print("测试结果总结:")
    print(f"{'✅' if training_ok else '❌'} 训练演示: {'通过' if training_ok else '失败'}")
    print(f"{'✅' if rendering_ok else '❌'} 渲染演示: {'通过' if rendering_ok else '失败'}")
    
    if training_ok and rendering_ok:
        print("\n🎉 所有演示脚本测试通过！")
        print("\n可以运行完整演示:")
        print("  python demos/demo_svraster_training.py")
        print("  python demos/demo_svraster_rendering.py")
    else:
        print("\n⚠️  部分演示脚本测试失败，请检查配置")
    
    return training_ok and rendering_ok


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
