"""
SVRaster 直接演示

直接导入核心组件，绕过有问题的模块
"""

import torch
import numpy as np
import time
import sys
import os

# 添加项目路径
sys.path.append('/home/xishansnow/3DVision/NeuroCity')

# 直接导入核心组件，不通过 __init__.py
try:
    from src.nerfs.svraster.core import SVRasterModel, SVRasterConfig
    print("✅ 成功导入核心组件")
except ImportError as e:
    print(f"❌ 导入核心组件失败: {e}")
    # 尝试直接导入
    try:
        sys.path.append('/home/xishansnow/3DVision/NeuroCity/src/nerfs/svraster')
        from core import SVRasterModel, SVRasterConfig
        print("✅ 通过直接路径导入成功")
    except ImportError as e2:
        print(f"❌ 直接导入也失败: {e2}")
        sys.exit(1)


def test_config():
    """测试配置创建"""
    print("\n=== 测试配置创建 ===")
    
    try:
        # 创建默认配置
        config = SVRasterConfig()
        print(f"✅ 默认配置创建成功")
        print(f"   - 图像尺寸: {config.image_width}x{config.image_height}")
        print(f"   - 场景边界: {config.scene_bounds}")
        print(f"   - 网格分辨率: {config.grid_resolution}")
        
        # 创建自定义配置
        custom_config = SVRasterConfig(
            image_width=400,
            image_height=300,
            scene_bounds=(-1, -1, -1, 1, 1, 1),
            grid_resolution=64
        )
        print(f"✅ 自定义配置创建成功")
        print(f"   - 图像尺寸: {custom_config.image_width}x{custom_config.image_height}")
        print(f"   - 场景边界: {custom_config.scene_bounds}")
        print(f"   - 网格分辨率: {custom_config.grid_resolution}")
        
        return config
        
    except Exception as e:
        print(f"❌ 配置创建失败: {e}")
        import traceback
        traceback.print_exc()
        return None


def test_model(config):
    """测试模型创建"""
    print("\n=== 测试模型创建 ===")
    
    try:
        # 创建模型
        model = SVRasterModel(config)
        print(f"✅ 模型创建成功")
        print(f"   - 设备: {model.device}")
        print(f"   - 体素网格形状: {model.voxel_grid.shape}")
        print(f"   - 特征网格形状: {model.feature_grid.shape}")
        
        return model
        
    except Exception as e:
        print(f"❌ 模型创建失败: {e}")
        import traceback
        traceback.print_exc()
        return None


def test_inference(model):
    """测试推理功能"""
    print("\n=== 测试推理功能 ===")
    
    try:
        # 准备测试数据
        num_rays = 5
        device = model.device
        
        # 生成测试光线
        rays_o = torch.randn(num_rays, 3, device=device) * 0.1
        rays_d = torch.randn(num_rays, 3, device=device)
        rays_d = rays_d / torch.norm(rays_d, dim=1, keepdim=True)
        
        print(f"   - 测试光线数量: {num_rays}")
        print(f"   - 光线起点形状: {rays_o.shape}")
        print(f"   - 光线方向形状: {rays_d.shape}")
        
        # 测试推理模式
        print("\n   测试推理模式...")
        with torch.no_grad():
            start_time = time.time()
            outputs = model(rays_o, rays_d, mode="inference")
            inference_time = time.time() - start_time
        
        print(f"   ✅ 推理模式成功!")
        print(f"   - 渲染时间: {inference_time:.3f}秒")
        print(f"   - 输出键: {list(outputs.keys())}")
        
        for key, value in outputs.items():
            if torch.is_tensor(value):
                print(f"   - {key}: {value.shape}, 范围 [{value.min():.3f}, {value.max():.3f}]")
        
        # 测试训练模式
        print("\n   测试训练模式...")
        with torch.no_grad():
            start_time = time.time()
            outputs = model(rays_o, rays_d, mode="training")
            training_time = time.time() - start_time
        
        print(f"   ✅ 训练模式成功!")
        print(f"   - 渲染时间: {training_time:.3f}秒")
        print(f"   - 输出键: {list(outputs.keys())}")
        
        for key, value in outputs.items():
            if torch.is_tensor(value):
                print(f"   - {key}: {value.shape}, 范围 [{value.min():.3f}, {value.max():.3f}]")
        
        if training_time > 0 and inference_time > 0:
            speedup = training_time / inference_time
            print(f"   - 速度提升: {speedup:.2f}x")
            
    except Exception as e:
        print(f"❌ 推理测试失败: {e}")
        import traceback
        traceback.print_exc()


def show_usage_guide():
    """显示使用指南"""
    print("\n=== SVRaster 使用指南 ===\n")
    
    print("🎯 SVRaster 是什么？")
    print("   SVRaster 是一个高效的神经辐射场渲染器，支持两种模式：")
    print("   - 训练模式：使用体积渲染，精确但较慢")
    print("   - 推理模式：使用光栅化，快速但近似")
    print()
    
    print("📋 基本使用步骤：")
    print("   1️⃣ 创建配置")
    print("   2️⃣ 初始化模型")
    print("   3️⃣ 准备光线数据")
    print("   4️⃣ 选择渲染模式")
    print("   5️⃣ 获取渲染结果")
    print()
    
    print("💡 代码示例：")
    print("""
# 步骤1: 创建配置
config = SVRasterConfig(
    image_width=800,
    image_height=600,
    scene_bounds=(-2, -2, -2, 2, 2, 2),
    grid_resolution=128
)

# 步骤2: 初始化模型
model = SVRasterModel(config)

# 步骤3: 准备光线数据
rays_o = torch.randn(1000, 3)  # 光线起点
rays_d = torch.randn(1000, 3)  # 光线方向
rays_d = rays_d / torch.norm(rays_d, dim=1, keepdim=True)  # 归一化方向

# 步骤4&5: 渲染
with torch.no_grad():
    # 快速推理
    outputs = model(rays_o, rays_d, mode="inference")
    
    # 精确训练
    outputs = model(rays_o, rays_d, mode="training")

# 获取结果
rgb = outputs['rgb']      # 颜色 [N, 3]
depth = outputs['depth']  # 深度 [N]
""")
    
    print("⚡ 性能优化建议：")
    print("   - 推理时使用 torch.no_grad() 减少内存")
    print("   - 批量处理光线提高效率")
    print("   - 调整 grid_resolution 平衡质量和速度")
    print("   - 推理模式用于实时渲染")
    print("   - 训练模式用于高质量渲染")
    print()
    
    print("🔧 关键参数说明：")
    print("   - image_width/height: 输出图像分辨率")
    print("   - scene_bounds: 场景边界 (x_min, y_min, z_min, x_max, y_max, z_max)")
    print("   - grid_resolution: 体素网格分辨率 (影响质量和速度)")
    print("   - sh_degree: 球谐函数阶数 (影响光照质量)")


def main():
    """主函数"""
    print("=== SVRaster 演示程序 ===")
    print("这个程序演示了 SVRaster 的基本使用方法")
    
    # 检查设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    
    # 测试配置
    config = test_config()
    if config is None:
        return
    
    # 测试模型
    model = test_model(config)
    if model is None:
        return
    
    # 测试推理
    test_inference(model)
    
    # 显示使用指南
    show_usage_guide()
    
    print("\n🎉 演示完成！")
    print("现在您可以开始使用 SVRaster 进行神经辐射场渲染了！")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⏹️  用户中断演示")
    except Exception as e:
        print(f"\n❌ 演示过程中出现错误: {e}")
        import traceback
        traceback.print_exc()
