#!/usr/bin/env python3
"""
Test runner for Instant NGP package

Usage:
    python run_tests.py [--cuda] [--verbose] [--pattern TEST_PATTERN]
"""

import sys
import unittest
import argparse
from pathlib import Path

# Add package to path
sys.path.insert(0, str(Path(__file__).parent))

def main():
    """Main test runner."""
    parser = argparse.ArgumentParser(description="Run Instant NGP tests")
    parser.add_argument("--cuda", action="store_true", 
                       help="Include CUDA-specific tests")
    parser.add_argument("--verbose", "-v", action="store_true",
                       help="Verbose output")
    parser.add_argument("--pattern", type=str, default="test_*.py",
                       help="Test file pattern")
    parser.add_argument("--module", type=str,
                       help="Specific test module to run")
    
    args = parser.parse_args()
    
    # Set up test discovery
    test_dir = Path(__file__).parent / "tests"
    
    if args.module:
        # Run specific module
        suite = unittest.TestLoader().loadTestsFromName(f"tests.{args.module}")
    else:
        # Discover tests
        suite = unittest.TestLoader().discover(
            str(test_dir),
            pattern=args.pattern,
            top_level_dir=str(Path(__file__).parent)
        )
    
    # Set verbosity
    verbosity = 2 if args.verbose else 1
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=verbosity)
    result = runner.run(suite)
    
    # Exit with appropriate code
    sys.exit(0 if result.wasSuccessful() else 1)

if __name__ == "__main__":
    main()
