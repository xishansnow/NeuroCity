#!/usr/bin/env python3
"""
Main Plenoxels Test Runner

This script serves as the main entry point for running all Plenoxels tests.
It can be run from the project root or from the tests directory.
"""

import os
import sys
import unittest
from pathlib import Path

# Ensure we can import from the plenoxels test package
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir.parent.parent.parent / "src"))

def discover_and_run_tests():
    """Discover and run all Plenoxels tests"""
    # Set up test discovery
    loader = unittest.TestLoader()
    
    # Discover tests in the current directory
    test_suite = loader.discover(
        start_dir=str(current_dir),
        pattern='test_*.py',
        top_level_dir=str(current_dir.parent.parent.parent)
    )
    
    # Run tests with detailed output
    runner = unittest.TextTestRunner(
        verbosity=2,
        buffer=True,
        failfast=False
    )
    
    print("🧪 Running Plenoxels Test Suite")
    print("=" * 60)
    print(f"Test directory: {current_dir}")
    print(f"Tests discovered: {test_suite.countTestCases()}")
    print("=" * 60)
    
    result = runner.run(test_suite)
    
    # Print summary
    print("\n" + "=" * 60)
    print("🎯 TEST SUMMARY")
    print("=" * 60)
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print(f"Skipped: {len(result.skipped)}")
    
    if result.wasSuccessful():
        print("✅ All tests passed!")
        return 0
    else:
        print("❌ Some tests failed")
        return 1

if __name__ == "__main__":
    sys.exit(discover_and_run_tests())
