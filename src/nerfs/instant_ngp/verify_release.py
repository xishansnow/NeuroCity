#!/usr/bin/env python3
"""
Instant NGP Package Release Verification Script

This script performs a comprehensive check of the instant_ngp package
to ensure it's ready for official release.
"""

import os
import sys
import importlib
from pathlib import Path
import subprocess

def check_file_exists(file_path, description):
    """Check if a file exists."""
    if os.path.exists(file_path):
        print(f"✓ {description}: {file_path}")
        return True
    else:
        print(f"❌ {description}: {file_path} - NOT FOUND")
        return False

def check_imports():
    """Check if all major components can be imported."""
    print("\n=== 检查包导入 ===")
    try:
        # Add src to path
        sys.path.insert(0, str(Path(__file__).parent.parent.parent))
        
        # Test core imports
        from nerfs.instant_ngp import (
            InstantNGPConfig, InstantNGPModel, InstantNGPLoss,
            InstantNGPTrainer, InstantNGPTrainerConfig,
            InstantNGPInferenceRenderer, InstantNGPRendererConfig,
            InstantNGPDataset, InstantNGPDatasetConfig,
            create_instant_ngp_dataloader
        )
        print("✓ 核心组件导入成功")
        
        # Test CLI imports
        from nerfs.instant_ngp.cli import train_cli, render_cli
        print("✓ CLI 组件导入成功")
        
        # Test utils imports
        from nerfs.instant_ngp import utils
        print("✓ 工具组件导入成功")
        
        # Test backward compatibility
        from nerfs.instant_ngp import InstantNGP
        print("✓ 向后兼容性导入成功")
        
        return True
        
    except Exception as e:
        print(f"❌ 导入失败: {e}")
        return False

def check_package_structure():
    """Check package structure and essential files."""
    print("\n=== 检查包结构 ===")
    
    base_path = Path(__file__).parent
    
    # Essential files
    essential_files = [
        ("__init__.py", "包初始化文件"),
        ("core.py", "核心模型文件"),
        ("trainer_new.py", "训练管道文件"),
        ("renderer_new.py", "推理管道文件"),
        ("dataset.py", "数据集工具文件"),
        ("utils.py", "工具函数文件"),
        ("cli.py", "命令行接口文件"),
        ("pyproject.toml", "项目配置文件"),
        ("setup.py", "安装脚本"),
        ("MANIFEST.in", "包文件清单"),
        ("LICENSE", "许可证文件"),
        ("CHANGELOG.md", "变更日志"),
        ("README.md", "说明文档"),
        ("README_cn.md", "中文说明文档"),
        ("requirements.txt", "依赖文件"),
        ("RELEASE_CHECKLIST.md", "发布检查清单"),
        ("REFACTOR_SUMMARY.md", "重构总结"),
        ("RELEASE_SUMMARY.md", "发布总结"),
    ]
    
    all_files_exist = True
    for filename, description in essential_files:
        file_path = base_path / filename
        if not check_file_exists(file_path, description):
            all_files_exist = False
    
    # Essential directories
    essential_dirs = [
        ("cuda", "CUDA 扩展目录"),
        ("utils", "工具模块目录"),
        ("tests", "测试目录"),
    ]
    
    for dirname, description in essential_dirs:
        dir_path = base_path / dirname
        if not check_file_exists(dir_path, description):
            all_files_exist = False
    
    return all_files_exist

def check_metadata():
    """Check package metadata."""
    print("\n=== 检查包元数据 ===")
    
    try:
        # Check pyproject.toml
        pyproject_path = Path(__file__).parent / "pyproject.toml"
        if pyproject_path.exists():
            with open(pyproject_path) as f:
                content = f.read()
                if "name = \"instant_ngp\"" in content:
                    print("✓ pyproject.toml 包名配置正确")
                else:
                    print("❌ pyproject.toml 包名配置错误")
                    
                if "version = \"1.0.0\"" in content:
                    print("✓ pyproject.toml 版本配置正确")
                else:
                    print("❌ pyproject.toml 版本配置错误")
                    
                if "train_cli = \"nerfs.instant_ngp.cli:train_cli\"" in content:
                    print("✓ CLI 入口点配置正确")
                else:
                    print("❌ CLI 入口点配置错误")
        
        # Check setup.py
        setup_path = Path(__file__).parent / "setup.py"
        if setup_path.exists():
            with open(setup_path) as f:
                content = f.read()
                if "name=\"instant_ngp\"" in content:
                    print("✓ setup.py 包名配置正确")
                else:
                    print("❌ setup.py 包名配置错误")
        
        return True
        
    except Exception as e:
        print(f"❌ 元数据检查失败: {e}")
        return False

def check_cuda_components():
    """Check CUDA components."""
    print("\n=== 检查 CUDA 组件 ===")
    
    cuda_path = Path(__file__).parent / "cuda"
    
    if not cuda_path.exists():
        print("❌ CUDA 目录不存在")
        return False
    
    cuda_files = [
        "instant_ngp_cuda.cpp",
        "hash_encoding_kernel.cu",
        "setup.py",
        "build_cuda.sh",
    ]
    
    all_cuda_files_exist = True
    for filename in cuda_files:
        file_path = cuda_path / filename
        if not check_file_exists(file_path, f"CUDA 文件"):
            all_cuda_files_exist = False
    
    return all_cuda_files_exist

def check_utils_components():
    """Check utils components."""
    print("\n=== 检查工具组件 ===")
    
    utils_path = Path(__file__).parent / "utils"
    
    if not utils_path.exists():
        print("❌ utils 目录不存在")
        return False
    
    utils_files = [
        "__init__.py",
        "coordinate_utils.py",
        "geometry_utils.py",
        "hash_utils.py",
        "regularization_utils.py",
        "sampling_utils.py",
        "visualization_utils.py",
    ]
    
    all_utils_files_exist = True
    for filename in utils_files:
        file_path = utils_path / filename
        if not check_file_exists(file_path, f"工具文件"):
            all_utils_files_exist = False
    
    return all_utils_files_exist

def count_code_lines():
    """Count lines of code."""
    print("\n=== 代码统计 ===")
    
    try:
        result = subprocess.run(
            ["find", ".", "-name", "*.py", "-exec", "wc", "-l", "{}", "+"],
            cwd=Path(__file__).parent,
            capture_output=True,
            text=True
        )
        
        if result.returncode == 0:
            lines = result.stdout.strip().split('\n')
            total_line = lines[-1]
            total_count = int(total_line.split()[0])
            print(f"✓ 总代码行数: {total_count}")
            
            # Count python files
            py_result = subprocess.run(
                ["find", ".", "-name", "*.py", "-type", "f"],
                cwd=Path(__file__).parent,
                capture_output=True,
                text=True
            )
            
            if py_result.returncode == 0:
                py_count = len(py_result.stdout.strip().split('\n'))
                print(f"✓ Python 文件数量: {py_count}")
        
        return True
        
    except Exception as e:
        print(f"❌ 代码统计失败: {e}")
        return False

def main():
    """Main verification function."""
    print("🚀 Instant NGP Package Release Verification")
    print("=" * 50)
    
    checks = [
        ("包结构检查", check_package_structure),
        ("导入检查", check_imports),
        ("元数据检查", check_metadata),
        ("CUDA 组件检查", check_cuda_components),
        ("工具组件检查", check_utils_components),
        ("代码统计", count_code_lines),
    ]
    
    all_passed = True
    
    for check_name, check_func in checks:
        print(f"\n{'=' * 20} {check_name} {'=' * 20}")
        
        try:
            result = check_func()
            if not result:
                all_passed = False
                print(f"❌ {check_name} 失败")
            else:
                print(f"✓ {check_name} 通过")
        except Exception as e:
            all_passed = False
            print(f"❌ {check_name} 异常: {e}")
    
    print("\n" + "=" * 50)
    if all_passed:
        print("🎉 所有检查通过! instant_ngp 包已准备好发布!")
        print("\n发布建议:")
        print("1. 确保所有 CUDA 扩展可以正常编译")
        print("2. 运行完整的测试套件")
        print("3. 检查依赖版本兼容性")
        print("4. 更新版本号和发布说明")
        print("5. 创建发布标签和文档")
        return 0
    else:
        print("❌ 部分检查失败，请修复后再发布")
        return 1

if __name__ == "__main__":
    sys.exit(main())
