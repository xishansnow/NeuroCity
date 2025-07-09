#!/usr/bin/env python3
"""
InfNeRF 测试运行脚本

运行 InfNeRF 的所有测试用例，包括：
- 单元测试
- 集成测试
- 性能测试
- 内存测试
"""

import sys
import os
import subprocess
import argparse
from pathlib import Path

# 添加项目根目录到 Python 路径
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))


def run_tests(test_type="all", verbose=False, coverage=False, device="auto"):
    """
    运行测试

    Args:
        test_type: 测试类型 ("all", "unit", "integration", "performance")
        verbose: 是否详细输出
        coverage: 是否生成覆盖率报告
        device: 测试设备 ("auto", "cpu", "cuda")
    """

    # 设置测试目录
    test_dir = Path(__file__).parent

    # 构建 pytest 命令
    cmd = ["python", "-m", "pytest"]

    # 添加测试目录
    cmd.append(str(test_dir))

    # 添加详细输出
    if verbose:
        cmd.append("-v")

    # 添加覆盖率
    if coverage:
        cmd.append("--cov=src.nerfs.inf_nerf")
        cmd.append("--cov-report=html")
        cmd.append("--cov-report=term-missing")

    # 根据测试类型选择测试文件
    if test_type == "unit":
        cmd.extend(
            [
                "test_core.py",
                "test_trainer.py",
                "test_renderer.py",
                "test_dataset.py",
                "test_utils.py",
            ]
        )
    elif test_type == "integration":
        cmd.append("test_integration.py")
    elif test_type == "performance":
        cmd.append("-m")
        cmd.append("performance")
    elif test_type == "gpu":
        cmd.append("-m")
        cmd.append("gpu")
    elif test_type == "slow":
        cmd.append("-m")
        cmd.append("slow")

    # 设置环境变量
    env = os.environ.copy()

    # 设置设备
    if device == "cpu":
        env["CUDA_VISIBLE_DEVICES"] = ""
    elif device == "cuda":
        env["CUDA_VISIBLE_DEVICES"] = "0"

    # 设置测试环境
    env["PYTHONPATH"] = f"{project_root}:{env.get('PYTHONPATH', '')}"
    env["TESTING"] = "1"

    print(f"运行测试命令: {' '.join(cmd)}")
    print(f"测试目录: {test_dir}")
    print(f"设备: {device}")
    print(f"覆盖率: {coverage}")
    print("-" * 50)

    try:
        # 运行测试
        result = subprocess.run(cmd, env=env, cwd=project_root)

        if result.returncode == 0:
            print("\n✅ 所有测试通过!")
        else:
            print(f"\n❌ 测试失败，退出码: {result.returncode}")
            return False

    except Exception as e:
        print(f"\n❌ 运行测试时出错: {e}")
        return False

    return True


def run_specific_test(test_file, verbose=False):
    """
    运行特定测试文件

    Args:
        test_file: 测试文件名
        verbose: 是否详细输出
    """
    test_dir = Path(__file__).parent
    test_path = test_dir / test_file

    if not test_path.exists():
        print(f"❌ 测试文件不存在: {test_path}")
        return False

    cmd = ["python", "-m", "pytest", str(test_path)]

    if verbose:
        cmd.append("-v")

    print(f"运行特定测试: {test_file}")
    print(f"命令: {' '.join(cmd)}")
    print("-" * 50)

    try:
        result = subprocess.run(cmd, cwd=project_root)
        return result.returncode == 0
    except Exception as e:
        print(f"❌ 运行测试时出错: {e}")
        return False


def check_test_environment():
    """检查测试环境"""
    print("检查测试环境...")

    # 检查 Python 版本
    python_version = sys.version_info
    print(f"Python 版本: {python_version.major}.{python_version.minor}.{python_version.micro}")

    if python_version < (3, 8):
        print("⚠️  警告: 建议使用 Python 3.8 或更高版本")

    # 检查必要的包
    required_packages = ["torch", "numpy", "pytest"]
    missing_packages = []

    for package in required_packages:
        try:
            __import__(package)
            print(f"✅ {package}: 已安装")
        except ImportError:
            missing_packages.append(package)
            print(f"❌ {package}: 未安装")

    if missing_packages:
        print(f"\n请安装缺失的包: pip install {' '.join(missing_packages)}")
        return False

    # 检查 CUDA 可用性
    try:
        import torch

        if torch.cuda.is_available():
            print(f"✅ CUDA: 可用 ({torch.cuda.device_count()} 个设备)")
            print(f"   CUDA 版本: {torch.version.cuda}")
        else:
            print("⚠️  CUDA: 不可用 (将使用 CPU 进行测试)")
    except ImportError:
        print("❌ PyTorch: 未安装")
        return False

    print("✅ 测试环境检查完成")
    return True


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="运行 InfNeRF 测试")
    parser.add_argument(
        "--type",
        "-t",
        choices=["all", "unit", "integration", "performance", "gpu", "slow"],
        default="all",
        help="测试类型",
    )
    parser.add_argument("--file", "-f", help="运行特定测试文件")
    parser.add_argument("--verbose", "-v", action="store_true", help="详细输出")
    parser.add_argument("--coverage", "-c", action="store_true", help="生成覆盖率报告")
    parser.add_argument(
        "--device", "-d", choices=["auto", "cpu", "cuda"], default="auto", help="测试设备"
    )
    parser.add_argument("--check-env", action="store_true", help="检查测试环境")

    args = parser.parse_args()

    # 检查环境
    if args.check_env:
        if not check_test_environment():
            sys.exit(1)
        return

    # 运行特定测试文件
    if args.file:
        success = run_specific_test(args.file, args.verbose)
        sys.exit(0 if success else 1)

    # 运行测试
    success = run_tests(
        test_type=args.type, verbose=args.verbose, coverage=args.coverage, device=args.device
    )

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
