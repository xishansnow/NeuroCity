"""Run tests for MIP-NeRF."""

import unittest
import sys
import os

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from tests.nerfs.test_mip_nerf import TestMIPNeRF

if __name__ == "__main__":
    # Create test suite
    suite = unittest.TestLoader().loadTestsFromTestCase(TestMIPNeRF)

    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    runner.run(suite)
