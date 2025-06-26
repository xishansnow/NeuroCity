"""
Demo Tests Package

Tests for demo scripts and examples in the NeuroCity project.

Available Tests:
- test_gfv_basic.py - Basic GFV functionality tests
- quick_test.py - Quick functionality tests

Usage:
    Run all demo tests: python -m pytest tests/demos/
    Run specific test: python tests/demos/test_gfv_basic.py

Author: NeuroCity Team
"""

# Available demo tests
DEMO_TESTS = [
    'test_gfv_basic', 'quick_test'
]

def list_demo_tests() -> list[str]:
    """List all available demo tests."""
    return DEMO_TESTS 