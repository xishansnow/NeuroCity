"""Run tests for NeRFacto."""

import unittest
import sys
import os

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from tests.nerfs.test_nerfacto import TestNeRFacto

if __name__ == "__main__":
    # Create test suite
    suite = unittest.TestLoader().loadTestsFromTestCase(TestNeRFacto)

    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    runner.run(suite)
