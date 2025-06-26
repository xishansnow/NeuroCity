"""
NeuroCity Test Suite

This package contains all test files for the NeuroCity project, organized by module.

Test Structure:
- tests/nerfs/ - Tests for all NeRF implementations
- tests/demos/ - Tests for demo scripts and examples  
- tests/datagen/ - Tests for data generation functionality
- tests/gfv/ - Tests for GFV (Geometric Feature Vector) functionality
- tests/neuralvdb/ - Tests for NeuralVDB functionality

Usage:
    Run all tests: python -m pytest tests/
    Run specific module: python -m pytest tests/nerfs/
    Run specific test: python -m pytest tests/nerfs/test_instant_ngp.py

Author: NeuroCity Team
Version: 1.0.0
"""

__version__ = "1.0.0"
__author__ = "NeuroCity Team"

# Available test modules
TEST_MODULES = [
    'nerfs', 'demos', 'datagen', 'gfv', 'neuralvdb'
]

def list_test_modules() -> list[str]:
    """List all available test modules."""
    return TEST_MODULES

def get_test_info() -> dict[str, str]:
    """Get information about test modules."""
    return {
        'nerfs': 'Tests for all Neural Radiance Fields implementations',
        'demos': 'Tests for demo scripts and examples',
        'datagen': 'Tests for data generation and sampling functionality',
        'gfv': 'Tests for Geometric Feature Vector processing',
        'neuralvdb': 'Tests for Neural VDB functionality'
    } 