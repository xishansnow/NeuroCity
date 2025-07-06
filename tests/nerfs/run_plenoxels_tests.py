"""Run tests for Plenoxels."""

import unittest
import sys
import os

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from tests.nerfs.test_plenoxels import (
    TestPlenoxels,
    TestVolumetricRenderer,
    TestPlenoxelModel,
    TestPlenoxelLoss,
    TestPlenoxelTrainer,
)


def create_test_suite():
    """Create a test suite containing all Plenoxels tests."""
    suite = unittest.TestSuite()

    # Add all test cases
    test_cases = [
        TestPlenoxels,  # Basic functionality tests
        TestVolumetricRenderer,  # Rendering tests
        TestPlenoxelModel,  # Model tests
        TestPlenoxelLoss,  # Loss function tests
        TestPlenoxelTrainer,  # Training tests
    ]

    for test_case in test_cases:
        suite.addTests(unittest.TestLoader().loadTestsFromTestCase(test_case))

    return suite


if __name__ == "__main__":
    # Create and run test suite
    suite = create_test_suite()
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    # Exit with non-zero code if tests failed
    sys.exit(not result.wasSuccessful())
