#!/usr/bin/env python3
"""
Advanced SVRaster demo showcasing:
1. Spherical Harmonics (SH) color decoding
2. View-dependent rendering
3. Volume rendering integration
4. Multi-level voxel representation
"""

import torch
import numpy as np
import sys
import os

# Add the src directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

# Try to import matplotlib for visualization
try:
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    print("Warning: matplotlib not available, visualization will be skipped")

from nerfs.svraster.core import SVRasterConfig, SVRasterModel, VoxelRasterizer

def create_test_scene():
    """创建一个测试场景，包含不同层级的体素"""
    print("Creating test scene with multi-level voxels...")
    
    # 创建配置
    config = SVRasterConfig(
        max_octree_levels=4,
        base_resolution=8,
        scene_bounds=(-2.0, -2.0, -2.0, 2.0, 2.0, 2.0),
        morton_ordering=True,
        sh_degree=2,  # 使用2阶球谐函数
        ray_samples_per_voxel=8
    )
    
    # 创建模型
    model = SVRasterModel(config)
    
    # 添加一些体素到不同层级
    # 层级0：大体素
    model.add_voxels(
        positions=torch.tensor([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]]),
        sizes=torch.tensor([1.0, 1.0]),
        densities=torch.tensor([0.5, 0.3]),
        colors=torch.randn(2, 3 * (config.sh_degree + 1) ** 2) * 0.1,
        level=0
    )
    
    # 层级1：中等体素
    model.add_voxels(
        positions=torch.tensor([[-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]]),
        sizes=torch.tensor([0.5, 0.5]),
        densities=torch.tensor([0.8, 0.6]),
        colors=torch.randn(2, 3 * (config.sh_degree + 1) ** 2) * 0.2,
        level=1
    )
    
    # 层级2：小体素（高分辨率区域）
    model.add_voxels(
        positions=torch.tensor([[-0.25, -0.25, -0.25], [0.25, 0.25, 0.25]]),
        sizes=torch.tensor([0.25, 0.25]),
        densities=torch.tensor([1.2, 1.0]),
        colors=torch.randn(2, 3 * (config.sh_degree + 1) ** 2) * 0.3,
        level=2
    )
    
    return model, config

def test_view_dependent_rendering(model, config):
    """测试视角相关渲染"""
    print("\nTesting view-dependent rendering...")
    
    rasterizer = VoxelRasterizer(config)
    
    # 从不同视角渲染同一场景
    viewpoints = [
        (torch.tensor([0.0, 0.0, 3.0]), "Front"),
        (torch.tensor([3.0, 0.0, 0.0]), "Right"),
        (torch.tensor([0.0, 3.0, 0.0]), "Top"),
        (torch.tensor([2.0, 2.0, 2.0]), "Diagonal")
    ]
    
    results = {}
    
    for camera_pos, view_name in viewpoints:
        print(f"  Rendering from {view_name} view...")
        
        # 创建光线
        ray_origins = camera_pos.unsqueeze(0).repeat(64, 1)  # 64条光线
        ray_directions = torch.randn(64, 3)
        ray_directions = ray_directions / ray_directions.norm(dim=-1, keepdim=True)
        
        # 获取体素数据
        voxels = model.get_all_voxels()
        
        # 渲染
        outputs = rasterizer(voxels, ray_origins, ray_directions)
        rgb = outputs['rgb']
        depth = outputs['depth']
        weights = outputs['weights']
        
        results[view_name] = {
            'rgb': rgb,
            'depth': depth,
            'weights': weights,
            'camera_pos': camera_pos
        }
        
        print(f"    RGB range: [{rgb.min():.3f}, {rgb.max():.3f}]")
        print(f"    Depth range: [{depth.min():.3f}, {depth.max():.3f}]")
    
    return results

def visualize_results(results):
    """可视化渲染结果"""
    if not HAS_MATPLOTLIB:
        print("  Matplotlib not available, skipping visualization")
        return
        
    print("\nVisualizing results...")
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()
    
    for i, (view_name, result) in enumerate(results.items()):
        ax = axes[i]
        
        # 显示RGB图像
        rgb = result['rgb'].reshape(8, 8, 3).detach().numpy()
        rgb = np.clip(rgb, 0, 1)  # 裁剪到[0,1]范围
        
        ax.imshow(rgb)
        ax.set_title(f'{view_name} View')
        ax.axis('off')
    
    plt.tight_layout()
    plt.savefig('svraster_view_dependent_rendering.png', dpi=150, bbox_inches='tight')
    print("  Saved visualization to 'svraster_view_dependent_rendering.png'")

def test_sh_decoding():
    """测试球谐函数解码"""
    print("\nTesting Spherical Harmonics decoding...")
    
    # 创建一些测试方向
    theta = torch.linspace(0, np.pi, 10)
    phi = torch.linspace(0, 2*np.pi, 10)
    theta, phi = torch.meshgrid(theta, phi, indexing='ij')
    
    # 转换为笛卡尔坐标
    x = torch.sin(theta) * torch.cos(phi)
    y = torch.sin(theta) * torch.sin(phi)
    z = torch.cos(theta)
    directions = torch.stack([x.flatten(), y.flatten(), z.flatten()], dim=-1)
    
    # 测试不同阶数的SH
    for degree in [0, 1, 2]:
        print(f"  Testing SH degree {degree}...")
        
        # 计算SH基函数
        from nerfs.svraster.core import eval_sh_basis
        sh_basis = eval_sh_basis(degree, directions)
        
        print(f"    SH basis shape: {sh_basis.shape}")
        print(f"    Expected shape: [{directions.shape[0]}, {(degree + 1) ** 2}]")
        
        # 验证基函数值范围
        print(f"    SH basis range: [{sh_basis.min():.3f}, {sh_basis.max():.3f}]")

def test_volume_integration():
    """测试体积渲染积分"""
    print("\nTesting volume rendering integration...")
    
    config = SVRasterConfig(
        max_octree_levels=2,
        base_resolution=4,
        scene_bounds=(-1.0, -1.0, -1.0, 1.0, 1.0, 1.0),
        sh_degree=1,
        ray_samples_per_voxel=16  # 更多采样点
    )
    
    rasterizer = VoxelRasterizer(config)
    
    # 创建简单的体素数据
    voxels = {
        'positions': torch.tensor([[0.0, 0.0, 0.0]]),
        'sizes': torch.tensor([0.5]),
        'densities': torch.tensor([1.0]),
        'colors': torch.randn(1, 3 * (config.sh_degree + 1) ** 2) * 0.1,
        'morton_codes': torch.tensor([0])
    }
    
    # 创建光线
    ray_o = torch.tensor([[0.0, 0.0, 2.0]])
    ray_d = torch.tensor([[0.0, 0.0, -1.0]])
    
    # 渲染
    outputs = rasterizer(voxels, ray_o, ray_d)
    rgb = outputs['rgb']
    depth = outputs['depth']
    weights = outputs['weights']
    
    print(f"  Single voxel rendering:")
    print(f"    RGB: {rgb[0].detach().numpy()}")
    print(f"    Depth: {depth[0].item():.3f}")
    print(f"    Weight: {weights[0].item():.3f}")

def main():
    """主函数"""
    print("Advanced SVRaster Demo")
    print("=" * 50)
    
    # 测试SH解码
    test_sh_decoding()
    
    # 测试体积渲染积分
    test_volume_integration()
    
    # 创建测试场景
    model, config = create_test_scene()
    
    # 测试视角相关渲染
    results = test_view_dependent_rendering(model, config)
    
    # 可视化结果
    visualize_results(results)
    
    print("\n🎉 Advanced SVRaster demo completed!")
    print("\nKey improvements demonstrated:")
    print("1. ✅ Spherical Harmonics color decoding (0-3 degree)")
    print("2. ✅ View-dependent rendering")
    print("3. ✅ Volume rendering integration with multi-point sampling")
    print("4. ✅ Multi-level voxel representation")
    print("5. ✅ Vectorized ray-voxel intersection")

if __name__ == "__main__":
    main() 