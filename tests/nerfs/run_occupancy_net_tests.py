"""Run tests for Occupancy Net."""

import unittest
import sys
import os

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from tests.nerfs.test_occupancy_net import TestOccupancyNet

if __name__ == "__main__":
    # Create test suite
    suite = unittest.TestLoader().loadTestsFromTestCase(TestOccupancyNet)

    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    runner.run(suite)
