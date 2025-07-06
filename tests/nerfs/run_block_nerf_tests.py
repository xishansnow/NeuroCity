"""Script to run Block-NeRF tests.

This script provides a convenient way to run Block-NeRF tests with different configurations:
- Unit tests only
- Integration tests only
- All tests
- Specific test classes or methods
"""

import os
import sys
import pytest
import argparse
from pathlib import Path


def main():
    """Run Block-NeRF tests."""
    parser = argparse.ArgumentParser(description="Run Block-NeRF tests")
    parser.add_argument(
        "--test-type",
        choices=["unit", "integration", "all"],
        default="all",
        help="Type of tests to run",
    )
    parser.add_argument(
        "--test-path",
        type=str,
        help="Specific test path to run (e.g., TestBlockNeRF::test_forward)",
    )
    parser.add_argument("--verbose", "-v", action="store_true", help="Enable verbose output")
    parser.add_argument("--cuda", action="store_true", help="Run tests on CUDA device if available")

    args = parser.parse_args()

    # Prepare test arguments
    test_args = [
        os.path.join(os.path.dirname(__file__), "test_block_nerf.py"),
        "-v" if args.verbose else "",
    ]

    # Add test selection based on type
    if args.test_type == "unit":
        test_args.extend(["-k", "not TestBlockNeRFIntegration"])
    elif args.test_type == "integration":
        test_args.extend(["-k", "TestBlockNeRFIntegration"])

    # Add specific test path if provided
    if args.test_path:
        test_args.extend(["-k", args.test_path])

    # Set CUDA environment variable
    if args.cuda:
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = ""

    # Run tests
    sys.exit(pytest.main(test_args))


if __name__ == "__main__":
    main()
