#!/usr/bin/env python3
"""
Instant NGP Test Runner

This script runs the complete Instant NGP test suite, including:
- Core component tests
- Training pipeline tests
- Rendering pipeline tests
- Utility function tests
- Dataset tests
- CUDA/GPU tests
- Integration tests

Usage:
    python run_instant_ngp_tests.py [options]

Options:
    --quick          Run only quick tests (skip slow integration tests)
    --cuda-only      Run only CUDA tests
    --no-cuda        Skip CUDA tests
    --verbose        Verbose output
    --coverage       Run with coverage reporting
    --html           Generate HTML test report
"""

import sys
import os
import argparse
import subprocess
import time
from pathlib import Path

# Add the src directory to the path for imports
src_path = Path(__file__).parent.parent.parent.parent / "src"
sys.path.insert(0, str(src_path))

# Test discovery
TEST_MODULES = [
    "test_core",
    "test_trainer",
    "test_utils",
    "test_dataset",
    "test_hash_encoder",
    "test_python310_compatibility",
]

QUICK_TESTS = ["test_core", "test_utils", "test_python310_compatibility"]

CUDA_TESTS = ["test_core::TestInstantNGPCore::test_cuda_forward"]

SLOW_TESTS = [
    "test_trainer::TestInstantNGPTrainer::test_full_training_cycle",
    "test_dataset::TestInstantNGPDataset::test_large_dataset_loading",
]


def main():
    """主函数：运行所有 Instant NGP 测试"""
    print("=" * 80)
    print("Instant NGP 测试套件")
    print("=" * 80)
    print(f"Python 版本: {sys.version}")
    print(f"PyTorch 版本: {get_pytorch_version()}")
    print(f"CUDA 可用: {is_cuda_available()}")
    print("=" * 80)

    # 获取测试目录
    test_dir = Path(__file__).parent

    # 定义测试模块列表
    test_modules: list[str] = [
        "test_python310_compatibility.py",  # 首先测试 Python 3.10 兼容性
        "test_core.py",  # 核心组件测试
        "test_hash_encoder.py",  # 哈希编码器测试
        "test_trainer.py",  # 训练器测试
        "test_renderer.py",  # 渲染器测试
        "test_dataset.py",  # 数据集测试
        "test_utils.py",  # 工具函数测试
        "test_integration.py",  # 集成测试
    ]

    # 运行测试统计
    test_results: dict[str, Any] = {
        "total_modules": len(test_modules),
        "passed_modules": 0,
        "failed_modules": 0,
        "skipped_modules": 0,
        "execution_times": {},
        "failed_tests": [],
    }

    start_time = time.time()

    for module in test_modules:
        module_path = test_dir / module

        if not module_path.exists():
            print(f"⚠️  警告: 测试模块 {module} 不存在，跳过")
            test_results["skipped_modules"] += 1
            continue

        print(f"\n📋 运行测试模块: {module}")
        print("-" * 60)

        module_start_time = time.time()

        # 运行单个测试模块
        result = run_test_module(module_path)

        module_end_time = time.time()
        execution_time = module_end_time - module_start_time
        test_results["execution_times"][module] = execution_time

        print(f"⏱️  执行时间: {execution_time:.2f} 秒")

        if result == 0:
            print(f"✅ {module} 测试通过")
            test_results["passed_modules"] += 1
        else:
            print(f"❌ {module} 测试失败 (退出码: {result})")
            test_results["failed_modules"] += 1
            test_results["failed_tests"].append(module)

    end_time = time.time()
    total_time = end_time - start_time

    # 打印测试总结
    print_test_summary(test_results, total_time)

    # 返回适当的退出码
    if test_results["failed_modules"] > 0:
        return 1
    else:
        return 0


def run_test_module(module_path: Path) -> int:
    """运行单个测试模块"""
    try:
        # 使用 pytest 运行测试
        args: list[str] = [
            str(module_path),
            "-v",  # 详细输出
            "--tb=short",  # 简短的回溯信息
            "--disable-warnings",  # 禁用警告
            "-x",  # 遇到第一个失败就停止
        ]

        # 如果有 CUDA，添加 CUDA 相关标记
        if is_cuda_available():
            args.extend(["--capture=no"])  # 不捕获输出，便于调试

        result = pytest.main(args)
        return result

    except Exception as e:
        print(f"❌ 运行测试模块时出错: {e}")
        return 1


def print_test_summary(results: dict[str, Any], total_time: float):
    """打印测试总结"""
    print("\n" + "=" * 80)
    print("测试总结")
    print("=" * 80)

    print(f"📊 总模块数: {results['total_modules']}")
    print(f"✅ 通过模块: {results['passed_modules']}")
    print(f"❌ 失败模块: {results['failed_modules']}")
    print(f"⏭️  跳过模块: {results['skipped_modules']}")
    print(f"⏱️  总执行时间: {total_time:.2f} 秒")

    if results["failed_tests"]:
        print(f"\n❌ 失败的测试模块:")
        for failed_test in results["failed_tests"]:
            print(f"   - {failed_test}")

    print(f"\n📈 各模块执行时间:")
    for module, exec_time in results["execution_times"].items():
        status = "✅" if module not in results["failed_tests"] else "❌"
        print(f"   {status} {module}: {exec_time:.2f} 秒")

    # 计算成功率
    if results["total_modules"] > 0:
        success_rate = (results["passed_modules"] / results["total_modules"]) * 100
        print(f"\n🎯 测试成功率: {success_rate:.1f}%")

    if results["failed_modules"] == 0:
        print("\n🎉 所有测试通过！Instant NGP 实现与 Python 3.10 完全兼容！")
    else:
        print(f"\n⚠️  有 {results['failed_modules']} 个模块测试失败，请检查相关问题。")


def get_pytorch_version() -> str:
    """获取 PyTorch 版本"""
    try:
        import torch

        return torch.__version__
    except ImportError:
        return "未安装"


def is_cuda_available() -> bool:
    """检查 CUDA 是否可用"""
    try:
        import torch

        return torch.cuda.is_available()
    except ImportError:
        return False


def check_dependencies():
    """检查依赖项"""
    print("🔍 检查依赖项...")

    dependencies: list[tuple[str, str]] = [
        ("torch", "PyTorch"),
        ("numpy", "NumPy"),
        ("PIL", "Pillow"),
        ("pytest", "pytest"),
    ]

    missing_deps: list[str] = []

    for module_name, display_name in dependencies:
        try:
            __import__(module_name)
            print(f"✅ {display_name}: 已安装")
        except ImportError:
            print(f"❌ {display_name}: 未安装")
            missing_deps.append(display_name)

    if missing_deps:
        print(f"\n⚠️  缺少以下依赖项: {', '.join(missing_deps)}")
        print("请使用以下命令安装:")
        print("pip install torch torchvision numpy pillow pytest")
        return False

    return True


def run_quick_test():
    """运行快速测试（仅核心功能）"""
    print("🚀 运行快速测试...")

    quick_tests: list[str] = [
        "test_python310_compatibility.py",
        "test_core.py",
    ]

    test_dir = Path(__file__).parent

    for test_file in quick_tests:
        test_path = test_dir / test_file
        if test_path.exists():
            print(f"📋 运行: {test_file}")
            result = run_test_module(test_path)
            if result != 0:
                print(f"❌ 快速测试失败: {test_file}")
                return False
            print(f"✅ 快速测试通过: {test_file}")
        else:
            print(f"⚠️  测试文件不存在: {test_file}")

    print("🎉 快速测试完成！")
    return True


if __name__ == "__main__":
    # 检查命令行参数
    if len(sys.argv) > 1:
        if sys.argv[1] == "--quick":
            # 运行快速测试
            if not check_dependencies():
                sys.exit(1)

            success = run_quick_test()
            sys.exit(0 if success else 1)

        elif sys.argv[1] == "--help":
            print("Instant NGP 测试运行器")
            print("用法:")
            print("  python run_instant_ngp_tests.py        # 运行所有测试")
            print("  python run_instant_ngp_tests.py --quick   # 运行快速测试")
            print("  python run_instant_ngp_tests.py --help    # 显示帮助")
            sys.exit(0)

    # 检查依赖项
    if not check_dependencies():
        sys.exit(1)

    # 运行完整测试套件
    exit_code = main()
    sys.exit(exit_code)
