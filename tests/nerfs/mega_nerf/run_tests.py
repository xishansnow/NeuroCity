#!/usr/bin/env python3
"""
Mega-NeRF Test Runner

This script runs the Mega-NeRF test suite with various options.
"""

import argparse
import sys
import subprocess
import os
from pathlib import Path


def run_command(cmd, description):
    """Run a command and handle errors."""
    print(f"\n{'='*60}")
    print(f"Running: {description}")
    print(f"Command: {' '.join(cmd)}")
    print(f"{'='*60}")

    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print("✅ Success!")
        if result.stdout:
            print("Output:")
            print(result.stdout)
        return True
    except subprocess.CalledProcessError as e:
        print("❌ Failed!")
        print(f"Error code: {e.returncode}")
        if e.stdout:
            print("Stdout:")
            print(e.stdout)
        if e.stderr:
            print("Stderr:")
            print(e.stderr)
        return False


def check_environment():
    """Check if the testing environment is properly set up."""
    print("🔍 Checking testing environment...")

    # Check Python version
    python_version = sys.version_info
    if python_version.major != 3 or python_version.minor < 10:
        print(f"❌ Python 3.10+ required, found {python_version.major}.{python_version.minor}")
        return False
    print(f"✅ Python {python_version.major}.{python_version.minor}.{python_version.micro}")

    # Check required packages
    required_packages = ["torch", "numpy", "pytest", "pytest-cov"]
    for package in required_packages:
        try:
            __import__(package)
            print(f"✅ {package}")
        except ImportError:
            print(f"❌ {package} not found")
            return False

    # Check CUDA availability
    try:
        import torch

        if torch.cuda.is_available():
            print(f"✅ CUDA available: {torch.cuda.get_device_name(0)}")
        else:
            print("⚠️  CUDA not available, tests will run on CPU")
    except Exception as e:
        print(f"⚠️  Could not check CUDA: {e}")

    return True


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Run Mega-NeRF tests")
    parser.add_argument(
        "--test-type",
        choices=["unit", "integration", "performance", "all"],
        default="unit",
        help="Type of tests to run",
    )
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    parser.add_argument("--coverage", action="store_true", help="Run with coverage reporting")
    parser.add_argument(
        "--device", choices=["cpu", "cuda", "auto"], default="auto", help="Device to run tests on"
    )
    parser.add_argument("--markers", nargs="+", help="Run only tests with specific markers")
    parser.add_argument("--timeout", type=int, default=300, help="Test timeout in seconds")
    parser.add_argument("--parallel", type=int, default=1, help="Number of parallel processes")

    args = parser.parse_args()

    # Check environment first
    if not check_environment():
        print("\n❌ Environment check failed. Please install required dependencies.")
        sys.exit(1)

    # Set device environment variable
    if args.device == "cpu":
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
    elif args.device == "cuda":
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    # Build pytest command
    test_dir = Path(__file__).parent
    cmd = ["python", "-m", "pytest", str(test_dir)]

    # Add verbosity
    if args.verbose:
        cmd.append("-v")

    # Add coverage
    if args.coverage:
        cmd.extend(
            ["--cov=src.nerfs.mega_nerf", "--cov-report=html:htmlcov", "--cov-report=term-missing"]
        )

    # Add markers
    if args.markers:
        for marker in args.markers:
            cmd.extend(["-m", marker])

    # Add test type filter
    if args.test_type == "unit":
        cmd.extend(["-m", "not integration and not performance"])
    elif args.test_type == "integration":
        cmd.extend(["-m", "integration"])
    elif args.test_type == "performance":
        cmd.extend(["-m", "performance"])
    # "all" runs everything

    # Add timeout
    cmd.extend(["--timeout", str(args.timeout)])

    # Add parallel execution
    if args.parallel > 1:
        cmd.extend(["-n", str(args.parallel)])

    # Run tests
    success = run_command(cmd, f"Mega-NeRF {args.test_type} tests")

    if success:
        print("\n🎉 All tests passed!")
        if args.coverage:
            print("📊 Coverage report generated in htmlcov/")
        sys.exit(0)
    else:
        print("\n💥 Some tests failed!")
        sys.exit(1)


if __name__ == "__main__":
    main()
