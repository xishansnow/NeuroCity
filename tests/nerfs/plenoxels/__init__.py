"""
Plenoxels Test Package

This package contains comprehensive tests for the Plenoxels neural rendering implementation.

Test Modules:
- test_plenoxels.py: Original Plenoxels tests
- test_plenoxels_comprehensive.py: Comprehensive functionality tests
- test_plenoxels_cuda_extensions.py: CUDA extensions tests
- test_plenoxels_integration.py: Integration and end-to-end tests
- test_refactored_package.py: Refactored package tests

Test Utilities:
- run_plenoxels_tests.py: Test runner with reporting
- quick_test_plenoxels.py: Quick validation tests
- plenoxels_test_config.yaml: Test configuration
- plenoxels_test_summary.py: Test suite summary

Usage:
    # Run all tests
    python run_plenoxels_tests.py
    
    # Quick tests
    python quick_test_plenoxels.py
    
    # Specific test modules
    python -m unittest test_plenoxels_comprehensive -v
"""

from .test_plenoxels_comprehensive import *
from .test_plenoxels_cuda_extensions import *
from .test_plenoxels_integration import *

__version__ = "2.0.0"
__author__ = "NeuroCity Development Team"
