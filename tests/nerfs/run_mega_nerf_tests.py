#!/usr/bin/env python3
"""
Run Mega-NeRF Tests

This script runs the Mega-NeRF test suite.
"""

import subprocess
import sys
from pathlib import Path


def main():
    """Run Mega-NeRF tests."""
    test_dir = Path(__file__).parent / "mega_nerf"

    if not test_dir.exists():
        print(f"❌ Test directory not found: {test_dir}")
        sys.exit(1)

    # Run the test runner
    cmd = [sys.executable, str(test_dir / "run_tests.py")] + sys.argv[1:]

    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        print(f"❌ Tests failed with exit code {e.returncode}")
        sys.exit(e.returncode)


if __name__ == "__main__":
    main()
