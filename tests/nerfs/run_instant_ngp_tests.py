"""Run tests for Instant NGP."""

import unittest
import sys
import os

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from tests.nerfs.test_instant_ngp import TestInstantNGP

if __name__ == "__main__":
    # Create test suite
    suite = unittest.TestLoader().loadTestsFromTestCase(TestInstantNGP)

    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    runner.run(suite)
