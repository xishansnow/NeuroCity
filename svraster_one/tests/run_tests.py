#!/usr/bin/env python3
"""
SVRaster One 测试运行脚本

提供多种测试运行选项，包括单元测试、集成测试、CUDA 测试等。
"""

import sys
import os
import argparse
import subprocess
import time
from pathlib import Path

# 添加项目根目录到 Python 路径
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))


def run_command(cmd, description=""):
    """运行命令并显示结果"""
    print(f"\n{'='*60}")
    print(f"运行: {description}")
    print(f"命令: {' '.join(cmd)}")
    print(f"{'='*60}")
    
    start_time = time.time()
    
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        end_time = time.time()
        
        print("✅ 成功!")
        print(f"耗时: {end_time - start_time:.2f} 秒")
        
        if result.stdout:
            print("\n输出:")
            print(result.stdout)
        
        return True
        
    except subprocess.CalledProcessError as e:
        end_time = time.time()
        
        print("❌ 失败!")
        print(f"耗时: {end_time - start_time:.2f} 秒")
        print(f"返回码: {e.returncode}")
        
        if e.stdout:
            print("\n标准输出:")
            print(e.stdout)
        
        if e.stderr:
            print("\n错误输出:")
            print(e.stderr)
        
        return False


def run_unit_tests():
    """运行单元测试"""
    tests_dir = Path(__file__).parent
    
    # 单元测试文件
    unit_test_files = [
        "test_config.py",
        "test_voxels.py", 
        "test_renderer.py",
        "test_losses.py",
        "test_trainer.py",
        "test_core.py",
    ]
    
    success_count = 0
    total_count = len(unit_test_files)
    
    for test_file in unit_test_files:
        test_path = tests_dir / test_file
        if test_path.exists():
            cmd = [sys.executable, "-m", "pytest", str(test_path), "-v"]
            if run_command(cmd, f"单元测试: {test_file}"):
                success_count += 1
        else:
            print(f"⚠️  测试文件不存在: {test_file}")
    
    print(f"\n单元测试结果: {success_count}/{total_count} 通过")
    return success_count == total_count


def run_integration_tests():
    """运行集成测试"""
    tests_dir = Path(__file__).parent
    test_path = tests_dir / "test_integration.py"
    
    if test_path.exists():
        cmd = [sys.executable, "-m", "pytest", str(test_path), "-v", "-m", "integration"]
        return run_command(cmd, "集成测试")
    else:
        print("⚠️  集成测试文件不存在")
        return False


def run_cuda_tests():
    """运行 CUDA 测试"""
    tests_dir = Path(__file__).parent
    test_path = tests_dir / "test_cuda.py"
    
    if test_path.exists():
        cmd = [sys.executable, "-m", "pytest", str(test_path), "-v", "-m", "cuda"]
        return run_command(cmd, "CUDA 测试")
    else:
        print("⚠️  CUDA 测试文件不存在")
        return False


def run_all_tests():
    """运行所有测试"""
    tests_dir = Path(__file__).parent
    
    cmd = [sys.executable, "-m", "pytest", str(tests_dir), "-v"]
    return run_command(cmd, "所有测试")


def run_specific_test(test_name):
    """运行特定测试"""
    tests_dir = Path(__file__).parent
    test_path = tests_dir / f"test_{test_name}.py"
    
    if test_path.exists():
        cmd = [sys.executable, "-m", "pytest", str(test_path), "-v"]
        return run_command(cmd, f"特定测试: {test_name}")
    else:
        print(f"⚠️  测试文件不存在: test_{test_name}.py")
        return False


def run_coverage_tests():
    """运行覆盖率测试"""
    tests_dir = Path(__file__).parent
    src_dir = Path(__file__).parent.parent
    
    cmd = [
        sys.executable, "-m", "pytest", 
        str(tests_dir), 
        "--cov", str(src_dir),
        "--cov-report", "html",
        "--cov-report", "term-missing",
        "-v"
    ]
    
    return run_command(cmd, "覆盖率测试")


def run_performance_tests():
    """运行性能测试"""
    tests_dir = Path(__file__).parent
    
    cmd = [
        sys.executable, "-m", "pytest", 
        str(tests_dir), 
        "-m", "slow",
        "-v"
    ]
    
    return run_command(cmd, "性能测试")


def check_test_environment():
    """检查测试环境"""
    print("🔍 检查测试环境...")
    
    # 检查 Python 版本
    python_version = sys.version_info
    print(f"Python 版本: {python_version.major}.{python_version.minor}.{python_version.micro}")
    
    # 检查必要的包
    required_packages = ["torch", "pytest", "numpy"]
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
            print(f"✅ {package}: 已安装")
        except ImportError:
            print(f"❌ {package}: 未安装")
            missing_packages.append(package)
    
    # 检查 CUDA
    try:
        import torch
        if torch.cuda.is_available():
            print(f"✅ CUDA: 可用 ({torch.cuda.device_count()} 个设备)")
            print(f"   CUDA 版本: {torch.version.cuda}")
        else:
            print("⚠️  CUDA: 不可用")
    except ImportError:
        print("❌ PyTorch: 未安装")
        missing_packages.append("torch")
    
    # 检查项目结构
    project_files = [
        "config.py",
        "core.py", 
        "voxels.py",
        "renderer.py",
        "losses.py",
        "trainer.py"
    ]
    
    src_dir = Path(__file__).parent.parent
    missing_files = []
    
    for file in project_files:
        file_path = src_dir / file
        if file_path.exists():
            print(f"✅ {file}: 存在")
        else:
            print(f"❌ {file}: 不存在")
            missing_files.append(file)
    
    if missing_packages:
        print(f"\n⚠️  缺少包: {', '.join(missing_packages)}")
        print("请运行: pip install " + " ".join(missing_packages))
    
    if missing_files:
        print(f"\n⚠️  缺少文件: {', '.join(missing_files)}")
    
    return len(missing_packages) == 0 and len(missing_files) == 0


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="SVRaster One 测试运行器")
    parser.add_argument(
        "--test-type", 
        choices=["unit", "integration", "cuda", "all", "coverage", "performance"],
        default="unit",
        help="测试类型"
    )
    parser.add_argument(
        "--specific", 
        type=str,
        help="运行特定测试 (例如: config, voxels, renderer)"
    )
    parser.add_argument(
        "--check-env", 
        action="store_true",
        help="检查测试环境"
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true", 
        help="详细输出"
    )
    
    args = parser.parse_args()
    
    # 检查环境
    if args.check_env:
        check_test_environment()
        return
    
    # 设置详细输出
    if args.verbose:
        os.environ["PYTEST_ADDOPTS"] = "-v -s"
    
    # 运行测试
    success = False
    
    if args.specific:
        success = run_specific_test(args.specific)
    elif args.test_type == "unit":
        success = run_unit_tests()
    elif args.test_type == "integration":
        success = run_integration_tests()
    elif args.test_type == "cuda":
        success = run_cuda_tests()
    elif args.test_type == "all":
        success = run_all_tests()
    elif args.test_type == "coverage":
        success = run_coverage_tests()
    elif args.test_type == "performance":
        success = run_performance_tests()
    
    # 输出结果
    print(f"\n{'='*60}")
    if success:
        print("🎉 所有测试通过!")
        sys.exit(0)
    else:
        print("💥 部分测试失败!")
        sys.exit(1)


if __name__ == "__main__":
    main() 