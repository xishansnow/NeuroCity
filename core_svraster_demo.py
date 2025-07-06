"""
SVRaster 核心演示

直接使用核心文件，不导入有问题的模块
"""

import torch
import numpy as np
import time
import sys
import os
from pathlib import Path

# 添加项目路径
project_root = Path('/home/xishansnow/3DVision/NeuroCity')
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / 'src'))

print("Python 路径:")
for p in sys.path[:5]:  # 只显示前5个路径
    print(f"  {p}")

# 尝试导入
try:
    # 直接从文件导入
    import importlib.util
    
    # 导入配置类
    spec = importlib.util.spec_from_file_location(
        "svraster_core", 
        "/home/xishansnow/3DVision/NeuroCity/src/nerfs/svraster/core.py"
    )
    svraster_core = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(svraster_core)
    
    SVRasterConfig = svraster_core.SVRasterConfig
    SVRasterModel = svraster_core.SVRasterModel
    
    print("✅ 成功导入核心组件")
    
except Exception as e:
    print(f"❌ 导入失败: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)


def run_demo():
    """运行演示"""
    print("\n=== SVRaster 核心演示 ===")
    
    # 检查 PyTorch 和设备
    print(f"PyTorch 版本: {torch.__version__}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    
    # 1. 创建配置
    print("\n1. 创建配置...")
    try:
        config = SVRasterConfig(
            image_width=200,
            image_height=150,
            scene_bounds=(-1.0, -1.0, -1.0, 1.0, 1.0, 1.0),
            grid_resolution=32  # 小尺寸用于快速测试
        )
        print("✅ 配置创建成功")
        print(f"   - 图像尺寸: {config.image_width}x{config.image_height}")
        print(f"   - 场景边界: {config.scene_bounds}")
        print(f"   - 网格分辨率: {config.grid_resolution}")
    except Exception as e:
        print(f"❌ 配置创建失败: {e}")
        return
    
    # 2. 创建模型
    print("\n2. 创建模型...")
    try:
        model = SVRasterModel(config)
        print("✅ 模型创建成功")
        print(f"   - 设备: {model.device}")
        print(f"   - 体素网格: {model.voxel_grid.shape}")
        print(f"   - 特征网格: {model.feature_grid.shape}")
    except Exception as e:
        print(f"❌ 模型创建失败: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # 3. 准备测试数据
    print("\n3. 准备测试数据...")
    try:
        # 创建一个简单的场景
        with torch.no_grad():
            # 在网格中心创建一个密度球
            res = config.grid_resolution
            coords = torch.stack(torch.meshgrid(
                torch.linspace(-1, 1, res),
                torch.linspace(-1, 1, res),
                torch.linspace(-1, 1, res),
                indexing='ij'
            ), dim=-1)
            
            # 距离中心的距离
            distances = torch.norm(coords, dim=-1)
            
            # 创建高斯球体
            densities = torch.exp(-distances * 3.0)
            
            # 设置体素数据
            model.voxel_grid = densities.unsqueeze(-1).to(model.device)
            
            # 创建彩色特征（简单的位置编码）
            features = torch.zeros(*coords.shape[:-1], model.feature_grid.shape[-1])
            features[..., 0] = (coords[..., 0] + 1) / 2  # 红色通道
            features[..., 1] = (coords[..., 1] + 1) / 2  # 绿色通道
            features[..., 2] = (coords[..., 2] + 1) / 2  # 蓝色通道
            
            model.feature_grid = features.to(model.device)
        
        print("✅ 测试场景创建成功")
        print(f"   - 密度范围: [{model.voxel_grid.min():.3f}, {model.voxel_grid.max():.3f}]")
    except Exception as e:
        print(f"❌ 场景创建失败: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # 4. 生成测试光线
    print("\n4. 生成测试光线...")
    try:
        num_rays = 100
        device = model.device
        
        # 从相机位置生成光线
        camera_pos = torch.tensor([0.0, 0.0, 2.0], device=device)
        
        # 生成朝向场景的光线
        rays_o = camera_pos.unsqueeze(0).expand(num_rays, -1)
        
        # 随机方向（朝向场景中心附近）
        target_offsets = torch.randn(num_rays, 3, device=device) * 0.5
        rays_d = torch.tensor([0.0, 0.0, -1.0], device=device) + target_offsets
        rays_d = rays_d / torch.norm(rays_d, dim=1, keepdim=True)
        
        print(f"✅ 生成了 {num_rays} 条测试光线")
        print(f"   - 相机位置: {camera_pos}")
        print(f"   - 光线起点: {rays_o.shape}")
        print(f"   - 光线方向: {rays_d.shape}")
    except Exception as e:
        print(f"❌ 光线生成失败: {e}")
        return
    
    # 5. 测试推理模式
    print("\n5. 测试推理模式...")
    try:
        start_time = time.time()
        with torch.no_grad():
            inference_outputs = model(rays_o, rays_d, mode="inference")
        inference_time = time.time() - start_time
        
        print(f"✅ 推理模式成功!")
        print(f"   - 渲染时间: {inference_time:.4f}秒")
        print(f"   - 输出键: {list(inference_outputs.keys())}")
        
        for key, value in inference_outputs.items():
            if torch.is_tensor(value):
                print(f"   - {key}: {value.shape}, [{value.min():.3f}, {value.max():.3f}]")
    except Exception as e:
        print(f"❌ 推理模式失败: {e}")
        import traceback
        traceback.print_exc()
    
    # 6. 测试训练模式
    print("\n6. 测试训练模式...")
    try:
        start_time = time.time()
        with torch.no_grad():
            training_outputs = model(rays_o, rays_d, mode="training")
        training_time = time.time() - start_time
        
        print(f"✅ 训练模式成功!")
        print(f"   - 渲染时间: {training_time:.4f}秒")
        print(f"   - 输出键: {list(training_outputs.keys())}")
        
        for key, value in training_outputs.items():
            if torch.is_tensor(value):
                print(f"   - {key}: {value.shape}, [{value.min():.3f}, {value.max():.3f}]")
    except Exception as e:
        print(f"❌ 训练模式失败: {e}")
        import traceback
        traceback.print_exc()
    
    # 7. 性能比较
    print("\n7. 性能比较...")
    try:
        if 'inference_time' in locals() and 'training_time' in locals():
            if training_time > 0 and inference_time > 0:
                speedup = training_time / inference_time
                print(f"   - 速度提升: {speedup:.2f}x")
            
            print(f"   - 推理模式: {inference_time:.4f}秒")
            print(f"   - 训练模式: {training_time:.4f}秒")
    except Exception as e:
        print(f"❌ 性能比较失败: {e}")
    
    print("\n✅ 核心演示完成!")


def show_usage():
    """显示使用方法"""
    print("\n=== SVRaster 使用方法 ===")
    
    print("""
🎯 SVRaster 是一个高效的神经辐射场渲染系统

📋 基本使用步骤：
1. 创建配置
2. 初始化模型
3. 准备光线数据
4. 选择渲染模式
5. 获取结果

💡 代码示例：
```python
# 1. 创建配置
config = SVRasterConfig(
    image_width=800,
    image_height=600,
    scene_bounds=(-2, -2, -2, 2, 2, 2),
    grid_resolution=128
)

# 2. 初始化模型
model = SVRasterModel(config)

# 3. 准备光线数据
rays_o = torch.randn(1000, 3)  # 光线起点
rays_d = torch.randn(1000, 3)  # 光线方向
rays_d = rays_d / torch.norm(rays_d, dim=1, keepdim=True)  # 归一化

# 4. 渲染
with torch.no_grad():
    # 快速推理 (光栅化)
    outputs = model(rays_o, rays_d, mode="inference")
    
    # 精确训练 (体积渲染)
    outputs = model(rays_o, rays_d, mode="training")

# 5. 获取结果
rgb = outputs['rgb']      # 颜色 [N, 3]
depth = outputs['depth']  # 深度 [N]
```

🔧 重要参数：
- mode="inference": 光栅化渲染，快速
- mode="training": 体积渲染，精确
- grid_resolution: 体素网格分辨率
- scene_bounds: 场景边界

⚡ 性能建议：
- 推理时使用 torch.no_grad()
- 批量处理光线
- 调整网格分辨率平衡质量和速度
""")


if __name__ == "__main__":
    try:
        run_demo()
        show_usage()
        print("\n🎉 演示完成!")
    except KeyboardInterrupt:
        print("\n⏹️  用户中断")
    except Exception as e:
        print(f"\n❌ 演示失败: {e}")
        import traceback
        traceback.print_exc()
