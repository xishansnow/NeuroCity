#!/usr/bin/env python3
"""
NeRFs Package Usage Demo

This demo shows how to use the unified NeRFs package to access
different Neural Radiance Fields implementations.

Author: NeuroCity Team
"""

import sys
import os

# Add the src directory to Python path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

def demo_nerfs_package():
    """Demonstrate basic usage of the NeRFs package."""
    print("=" * 60)
    print("NeRFs Package Usage Demo")
    print("=" * 60)
    
    try:
        # Import the main NeRFs package
        from nerfs import list_available_nerfs, get_nerf_info, get_nerf_module
        
        print("\n1. 列出所有可用的 NeRF 实现:")
        print("-" * 40)
        available_nerfs = list_available_nerfs()
        for i, nerf_name in enumerate(available_nerfs, 1):
            print(f"{i:2d}. {nerf_name}")
        
        print(f"\n总共有 {len(available_nerfs)} 种 NeRF 实现可用。")
        
        print("\n2. 获取 NeRF 实现的详细信息:")
        print("-" * 40)
        nerf_info = get_nerf_info()
        
        # 显示前几个 NeRF 的信息
        showcase_nerfs = ['classic_nerf', 'instant_ngp', 'mip_nerf', 'block_nerf', 'svraster']
        for nerf_name in showcase_nerfs:
            if nerf_name in nerf_info:
                print(f"\n{nerf_name.upper()}:")
                print(f"  描述: {nerf_info[nerf_name]}")
        
        print("\n3. 动态加载特定的 NeRF 模块:")
        print("-" * 40)
        
        # 演示加载不同的 NeRF 模块
        demo_modules = ['classic_nerf', 'instant_ngp', 'svraster']
        
        for nerf_name in demo_modules:
            try:
                print(f"\n加载 {nerf_name} 模块...")
                nerf_module = get_nerf_module(nerf_name)
                print(f"✓ 成功加载 {nerf_name}")
                print(f"  模块路径: {nerf_module.__name__}")
                
                # 尝试获取模块的主要组件
                if hasattr(nerf_module, '__file__'):
                    print(f"  文件位置: {nerf_module.__file__}")
                
            except Exception as e:
                print(f"✗ 加载 {nerf_name} 失败: {e}")
        
        print("\n4. 展示具体模块的使用方法:")
        print("-" * 40)
        
        # 演示 Classic NeRF
        try:
            print("\n演示 Classic NeRF:")
            classic_nerf = get_nerf_module('classic_nerf')
            
            # 检查可用的类和函数
            if hasattr(classic_nerf, 'core'):
                print("  - 包含核心模块 (core.py)")
            if hasattr(classic_nerf, 'dataset'):
                print("  - 包含数据集模块 (dataset.py)")
            if hasattr(classic_nerf, 'trainer'):
                print("  - 包含训练器模块 (trainer.py)")
                
        except Exception as e:
            print(f"  Classic NeRF 演示失败: {e}")
        
        # 演示 Instant-NGP
        try:
            print("\n演示 Instant-NGP:")
            instant_ngp = get_nerf_module('instant_ngp')
            
            # 检查可用的类和函数
            if hasattr(instant_ngp, 'core'):
                print("  - 包含核心模块 (core.py)")
            if hasattr(instant_ngp, 'dataset'):
                print("  - 包含数据集模块 (dataset.py)")
            if hasattr(instant_ngp, 'trainer'):
                print("  - 包含训练器模块 (trainer.py)")
                
        except Exception as e:
            print(f"  Instant-NGP 演示失败: {e}")
        
        print("\n5. NeRF 选择建议:")
        print("-" * 40)
        
        recommendations = {
            "小规模场景学习": ["classic_nerf", "nerfacto"], "实时渲染应用": ["instant_ngp", "svraster", "plenoxels"], "大规模城市场景": ["block_nerf", "mega_nerf", "mega_nerf_plus"], "高质量渲染": ["mip_nerf", "pyramid_nerf"], "快速原型开发": ["instant_ngp", "nerfacto"], "几何约束场景": ["dnmp_nerf"], "渐进式训练": ["bungee_nerf"], "多尺度场景": ["mip_nerf", "pyramid_nerf"]
        }
        
        for use_case, recommended_nerfs in recommendations.items():
            print(f"\n{use_case}:")
            for nerf in recommended_nerfs:
                print(f"  - {nerf}")
        
        print("\n6. 性能特点总结:")
        print("-" * 40)
        
        performance_info = [
            (
                "训练速度最快",
                "instant_ngp",
            )
        ]
        
        for feature, nerf_name in performance_info:
            print(f"{feature}: {nerf_name}")
        
    except ImportError as e:
        print(f"❌ 导入 NeRFs 包失败: {e}")
        print("请确保已正确安装所有依赖项")
        return False
    
    except Exception as e:
        print(f"❌ 演示过程中出现错误: {e}")
        return False
    
    print("\n" + "=" * 60)
    print("NeRFs Package 演示完成！")
    print("=" * 60)
    return True

def demo_specific_nerf_usage():
    """Demonstrate usage of specific NeRF implementations."""
    print("\n" + "=" * 60)
    print("具体 NeRF 实现使用演示")
    print("=" * 60)
    
    try:
        from nerfs import get_nerf_module
        
        print("\n1. Classic NeRF 基本用法:")
        print("-" * 30)
        
        # 演示 Classic NeRF 的基本用法
        try:
            classic_nerf = get_nerf_module('classic_nerf')
            print("✓ Classic NeRF 模块加载成功")
            
            # 这里可以添加具体的使用示例
            print("  可用组件:")
            for attr in dir(classic_nerf):
                if not attr.startswith('_'):
                    print(f"    - {attr}")
                    
        except Exception as e:
            print(f"✗ Classic NeRF 演示失败: {e}")
        
        print("\n2. Instant-NGP 基本用法:")
        print("-" * 30)
        
        try:
            instant_ngp = get_nerf_module('instant_ngp')
            print("✓ Instant-NGP 模块加载成功")
            
            print("  可用组件:")
            for attr in dir(instant_ngp):
                if not attr.startswith('_'):
                    print(f"    - {attr}")
                    
        except Exception as e:
            print(f"✗ Instant-NGP 演示失败: {e}")
        
        print("\n3. SVRaster 基本用法:")
        print("-" * 30)
        
        try:
            svraster = get_nerf_module('svraster')
            print("✓ SVRaster 模块加载成功")
            
            print("  可用组件:")
            for attr in dir(svraster):
                if not attr.startswith('_'):
                    print(f"    - {attr}")
                    
        except Exception as e:
            print(f"✗ SVRaster 演示失败: {e}")
            
    except Exception as e:
        print(f"❌ 具体 NeRF 演示失败: {e}")
        return False
    
    return True

def main():
    """Main demo function."""
    print("🚀 NeuroCity NeRFs Package 综合演示")
    print("=" * 60)
    
    # 基本包使用演示
    success1 = demo_nerfs_package()
    
    # 具体实现使用演示
    success2 = demo_specific_nerf_usage()
    
    if success1 and success2:
        print("\n🎉 所有演示都成功完成！")
        print("\n💡 提示:")
        print("  - 使用 list_available_nerfs() 查看所有可用的 NeRF 实现")
        print("  - 使用 get_nerf_info() 获取详细信息")
        print("  - 使用 get_nerf_module(name) 加载特定的 NeRF 模块")
        print("  - 查看各模块的 README.md 文件了解具体用法")
    else:
        print("\n⚠️  部分演示未能成功完成，请检查环境配置")

if __name__ == "__main__":
    main() 