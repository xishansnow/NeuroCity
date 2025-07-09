#!/usr/bin/env python3
"""
Test runner script for Block-NeRF tests.

Usage:
    python test_block_nerf_runner.py [options]

Options:
    --fast          Run only fast tests (skip slow tests)
    --cuda          Include CUDA tests
    --integration   Run only integration tests
    --unit          Run only unit tests
    --coverage      Generate coverage report
    --verbose       Verbose output
"""

import sys
import argparse
import subprocess
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Run Block-NeRF tests")
    
    parser.add_argument(
        "--fast",
        action="store_true",
        help="Run only fast tests (skip slow tests)"
    )
    
    parser.add_argument(
        "--cuda",
        action="store_true",
        help="Include CUDA tests"
    )
    
    parser.add_argument(
        "--integration",
        action="store_true",
        help="Run only integration tests"
    )
    
    parser.add_argument(
        "--unit",
        action="store_true",
        help="Run only unit tests"
    )
    
    parser.add_argument(
        "--coverage",
        action="store_true",
        help="Generate coverage report"
    )
    
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Verbose output"
    )
    
    parser.add_argument(
        "--module",
        type=str,
        help="Run tests for specific module (core, trainer, renderer, dataset, integrations)"
    )
    
    args = parser.parse_args()
    
    # Build pytest command
    cmd = ["python", "-m", "pytest"]
    
    # Test directory
    test_dir = Path(__file__).parent
    
    if args.module:
        # Run specific module tests
        test_file = test_dir / f"test_{args.module}.py"
        if not test_file.exists():
            print(f"Error: Test file {test_file} does not exist")
            return 1
        cmd.append(str(test_file))
    else:
        # Run all tests in the directory
        cmd.append(str(test_dir))
    
    # Add options
    if args.verbose:
        cmd.extend(["-v", "-s"])
    
    if not args.fast:
        cmd.append("--runslow")
    
    if args.cuda:
        cmd.append("--runcuda")
    
    if args.integration:
        cmd.extend(["-m", "integration"])
    elif args.unit:
        cmd.extend(["-m", "unit"])
    
    if args.coverage:
        cmd.extend([
            "--cov=src.nerfs.block_nerf",
            "--cov-report=html",
            "--cov-report=term-missing"
        ])
    
    # Additional pytest options for better output
    cmd.extend([
        "--tb=short",
        "--strict-markers",
        "--disable-warnings"
    ])
    
    print(f"Running command: {' '.join(cmd)}")
    
    # Run tests
    try:
        result = subprocess.run(cmd, check=False)
        return result.returncode
    except KeyboardInterrupt:
        print("\nTests interrupted by user")
        return 1
    except Exception as e:
        print(f"Error running tests: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
