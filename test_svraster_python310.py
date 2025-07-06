#!/usr/bin/env python3
"""
SVRaster Python 3.10 兼容性验证脚本

检查所有 SVRaster 组件是否符合 Python 3.10 标准要求
"""

from __future__ import annotations

import sys
import traceback
from pathlib import Path
import importlib
import ast
import re


def check_python_version():
    """检查 Python 版本"""
    version = sys.version_info
    print(f"Python 版本: {version.major}.{version.minor}.{version.micro}")
    
    if version.major != 3 or version.minor < 10:
        print("⚠️  警告: 建议使用 Python 3.10 或更高版本")
    else:
        print("✅ Python 版本兼容")
    
    return version


def check_syntax_compatibility(file_path: Path) -> bool:
    """检查文件的语法兼容性"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # 检查是否使用了新式 Union 语法 (|)
        union_pattern = r':\s*[A-Za-z_][A-Za-z0-9_\[\],\s]*\s\|\s[A-Za-z_][A-Za-z0-9_\[\],\s]*'
        if re.search(union_pattern, content):
            print(f"❌ {file_path}: 发现新式 Union 语法 (|)")
            return False
        
        # 尝试编译代码
        ast.parse(content)
        return True
    except SyntaxError as e:
        print(f"❌ {file_path}: 语法错误 - {e}")
        return False
    except Exception as e:
        print(f"❌ {file_path}: 检查失败 - {e}")
        return False


def check_imports_compatibility():
    """检查导入兼容性"""
    print("\n=== 检查导入兼容性 ===")
    
    test_imports = [
        # 核心组件
        ("from src.nerfs.svraster.core import SVRasterConfig", "SVRasterConfig"),
        ("from src.nerfs.svraster.core import SVRasterModel", "SVRasterModel"),
        ("from src.nerfs.svraster.core import SVRasterLoss", "SVRasterLoss"),
        
        # 渲染器
        ("from src.nerfs.svraster.volume_renderer import VolumeRenderer", "VolumeRenderer"),
        ("from src.nerfs.svraster.true_rasterizer import TrueVoxelRasterizer", "TrueVoxelRasterizer"),
        
        # 工具函数
        ("from src.nerfs.svraster.spherical_harmonics import eval_sh_basis", "eval_sh_basis"),
        
        # 训练和渲染
        ("from src.nerfs.svraster.trainer import SVRasterTrainer", "SVRasterTrainer"),
        ("from src.nerfs.svraster.renderer import SVRasterRenderer", "SVRasterRenderer"),
        
        # 数据集
        ("from src.nerfs.svraster.dataset import SVRasterDataset", "SVRasterDataset"),
        
        # 包级别导入
        ("from src.nerfs.svraster import SVRasterConfig, SVRasterModel", "package imports"),
    ]
    
    success_count = 0
    for import_stmt, component_name in test_imports:
        try:
            exec(import_stmt)
            print(f"✅ {component_name}")
            success_count += 1
        except Exception as e:
            print(f"❌ {component_name}: {e}")
    
    print(f"\n导入测试结果: {success_count}/{len(test_imports)} 成功")
    return success_count == len(test_imports)


def check_type_annotations():
    """检查类型注解兼容性"""
    print("\n=== 检查类型注解兼容性 ===")
    
    svraster_dir = Path("src/nerfs/svraster")
    python_files = list(svraster_dir.rglob("*.py"))
    
    compatible_files = 0
    total_files = len(python_files)
    
    for file_path in python_files:
        if file_path.name.startswith('.'):
            continue
            
        if check_syntax_compatibility(file_path):
            compatible_files += 1
        else:
            print(f"   需要修复: {file_path}")
    
    print(f"\n语法兼容性: {compatible_files}/{total_files} 文件通过")
    return compatible_files == total_files


def check_instantiation():
    """检查组件实例化"""
    print("\n=== 检查组件实例化 ===")
    
    try:
        # 导入必要组件
        from src.nerfs.svraster.core import SVRasterConfig, SVRasterModel
        from src.nerfs.svraster.volume_renderer import VolumeRenderer
        from src.nerfs.svraster.spherical_harmonics import eval_sh_basis
        
        # 创建配置
        config = SVRasterConfig(
            image_width=64,
            image_height=48,
            base_resolution=16  # 小尺寸快速测试
        )
        print("✅ SVRasterConfig 实例化成功")
        
        # 创建模型
        model = SVRasterModel(config)
        print("✅ SVRasterModel 实例化成功")
        
        # 创建体积渲染器
        volume_renderer = VolumeRenderer(config)
        print("✅ VolumeRenderer 实例化成功")
        
        # 测试球谐函数
        import torch
        directions = torch.randn(10, 3)
        directions = directions / directions.norm(dim=-1, keepdim=True)
        sh_basis = eval_sh_basis(2, directions)
        print(f"✅ eval_sh_basis 计算成功: {sh_basis.shape}")
        
        return True
        
    except Exception as e:
        print(f"❌ 实例化失败: {e}")
        traceback.print_exc()
        return False


def check_modern_features():
    """检查现代 Python 特性使用"""
    print("\n=== 检查现代 Python 特性 ===")
    
    # 检查是否正确使用了 future annotations
    svraster_dir = Path("src/nerfs/svraster")
    files_with_future = 0
    files_with_typing = 0
    
    for file_path in svraster_dir.rglob("*.py"):
        if file_path.name.startswith('.'):
            continue
            
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            if "from __future__ import annotations" in content:
                files_with_future += 1
            
            if any(typing_import in content for typing_import in 
                   ["from typing import", "import typing"]):
                files_with_typing += 1
                
        except Exception:
            continue
    
    print(f"✅ {files_with_future} 文件使用 future annotations")
    print(f"✅ {files_with_typing} 文件使用 typing 模块")
    
    return True


def main():
    """主测试函数"""
    print("=" * 70)
    print("SVRaster Python 3.10 兼容性验证")
    print("=" * 70)
    
    # 检查 Python 版本
    version = check_python_version()
    
    # 检查语法兼容性
    syntax_ok = check_type_annotations()
    
    # 检查导入兼容性
    imports_ok = check_imports_compatibility()
    
    # 检查实例化
    instantiation_ok = check_instantiation()
    
    # 检查现代特性
    modern_ok = check_modern_features()
    
    # 总结
    print("\n" + "=" * 70)
    print("兼容性检查总结")
    print("=" * 70)
    
    checks = [
        ("语法兼容性", syntax_ok),
        ("导入兼容性", imports_ok),
        ("实例化测试", instantiation_ok),
        ("现代特性", modern_ok),
    ]
    
    passed = sum(1 for _, ok in checks if ok)
    total = len(checks)
    
    for name, ok in checks:
        status = "✅ 通过" if ok else "❌ 失败"
        print(f"{name:.<20} {status}")
    
    if passed == total:
        print(f"\n🎉 所有检查通过! SVRaster 完全兼容 Python 3.10+")
        return True
    else:
        print(f"\n⚠️  {passed}/{total} 项检查通过，需要进一步修复")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
