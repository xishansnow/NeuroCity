#!/usr/bin/env python3
"""
SVRaster Test Runner

This script runs the complete SVRaster test suite, including:
- Core component tests
- Training pipeline tests
- Rendering pipeline tests
- Utility function tests
- Dataset tests
- CUDA/GPU tests
- Integration tests (including validation/test split evaluation)

Usage:
    python run_svraster_tests.py [options]

Options:
    --quick          Run only quick tests (skip slow integration tests)
    --cuda-only      Run only CUDA tests
    --no-cuda        Skip CUDA tests
    --verbose        Verbose output
    --coverage       Run with coverage reporting
    --html           Generate HTML test report

Note:
    Integration tests now include end-to-end validation and test split evaluation (see test_validate_and_test_pipeline in test_integration.py).
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
    "test_training",
    "test_rendering",
    "test_utils",
    "test_dataset",
    "test_cuda",
    "test_integration",
]

QUICK_TESTS = ["test_core", "test_utils", "test_cuda::TestCUDAAvailability"]

CUDA_TESTS = ["test_cuda"]

SLOW_TESTS = [
    "test_integration",
    "test_training::TestSVRasterTrainer::test_trainer_with_dummy_dataset",
    "test_dataset::TestSVRasterDataset::test_dataset_creation_with_dummy_data",
]


def check_svraster_availability():
    """Check if SVRaster is available for testing"""
    try:
        import nerfs.svraster as svraster

        print(f"✓ SVRaster available (version: {svraster.__version__})")

        # Get device info
        device_info = svraster.get_device_info()
        print(f"  PyTorch version: {device_info['torch_version']}")
        print(f"  CUDA available: {device_info['cuda_available']}")
        print(f"  SVRaster CUDA: {device_info['svraster_cuda']}")

        if device_info["cuda_available"]:
            print(f"  GPU devices: {device_info['device_count']}")
            for i, device in enumerate(device_info.get("devices", [])):
                print(f"    Device {i}: {device['name']} ({device['memory_total']}GB)")

        return True

    except ImportError as e:
        print(f"✗ SVRaster not available: {e}")
        print("  Make sure SVRaster is properly installed")
        return False


def run_pytest(test_files, args):
    """Run pytest with specified test files and arguments"""

    # Build pytest command
    cmd = ["python", "-m", "pytest"]

    # Add test files
    for test_file in test_files:
        cmd.append(f"tests/nerfs/svraster/{test_file}.py")

    # Add pytest arguments
    if args.verbose:
        cmd.extend(["-v", "-s"])
    else:
        cmd.append("-v")

    if args.coverage:
        cmd.extend(
            ["--cov=nerfs.svraster", "--cov-report=term-missing", "--cov-report=html:htmlcov"]
        )

    if args.html:
        cmd.extend(["--html=test_report.html", "--self-contained-html"])

    # Add performance markers
    cmd.extend(["-m", "not slow" if args.quick else "not never"])

    # Add warnings
    cmd.extend(["-W", "ignore::DeprecationWarning"])

    print(f"Running command: {' '.join(cmd)}")
    print("=" * 80)

    # Run tests
    start_time = time.time()
    result = subprocess.run(cmd, cwd=os.getcwd())
    end_time = time.time()

    print("=" * 80)
    print(f"Tests completed in {end_time - start_time:.2f} seconds")

    return result.returncode


def main():
    """Main test runner function"""
    parser = argparse.ArgumentParser(description="SVRaster Test Runner")
    parser.add_argument("--quick", action="store_true", help="Run only quick tests")
    parser.add_argument("--cuda-only", action="store_true", help="Run only CUDA tests")
    parser.add_argument("--no-cuda", action="store_true", help="Skip CUDA tests")
    parser.add_argument("--verbose", action="store_true", help="Verbose output")
    parser.add_argument("--coverage", action="store_true", help="Run with coverage reporting")
    parser.add_argument("--html", action="store_true", help="Generate HTML test report")
    parser.add_argument("--module", action="append", help="Run specific test module(s)")

    args = parser.parse_args()

    print("SVRaster Test Suite")
    print("=" * 50)
    print(
        "Integration tests now include validation/test split evaluation (test_validate_and_test_pipeline).\n"
    )

    # Check SVRaster availability
    if not check_svraster_availability():
        print("\nCannot run tests without SVRaster. Exiting.")
        return 1

    print()

    # Determine which tests to run
    if args.module:
        test_files = args.module
    elif args.cuda_only:
        test_files = CUDA_TESTS
    elif args.quick:
        test_files = ["test_core", "test_utils"]
        if not args.no_cuda:
            test_files.append("test_cuda")
    else:
        test_files = TEST_MODULES.copy()
        if args.no_cuda:
            test_files = [t for t in test_files if t != "test_cuda"]

    print(f"Running test modules: {', '.join(test_files)}")
    print()

    # Check for required dependencies
    missing_deps = []
    try:
        import pytest
    except ImportError:
        missing_deps.append("pytest")

    if args.coverage:
        try:
            import pytest_cov
        except ImportError:
            missing_deps.append("pytest-cov")

    if args.html:
        try:
            import pytest_html
        except ImportError:
            missing_deps.append("pytest-html")

    if missing_deps:
        print(f"Missing dependencies: {', '.join(missing_deps)}")
        print("Install with: pip install " + " ".join(missing_deps))
        return 1

    # Run tests
    return_code = run_pytest(test_files, args)

    # Summary
    if return_code == 0:
        print("\n✓ All tests passed!")
    else:
        print(f"\n✗ Tests failed with code {return_code}")

    # Additional information
    if args.coverage:
        print("\nCoverage report generated in htmlcov/index.html")

    if args.html:
        print("HTML test report generated: test_report.html")

    return return_code


if __name__ == "__main__":
    sys.exit(main())
