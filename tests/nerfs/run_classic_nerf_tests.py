"""Run tests for Classic NeRF."""

import unittest
import sys
import os

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from tests.nerfs.test_classic_nerf import TestClassicNeRF

if __name__ == "__main__":
    # Create test suite
    suite = unittest.TestLoader().loadTestsFromTestCase(TestClassicNeRF)

    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    runner.run(suite)
