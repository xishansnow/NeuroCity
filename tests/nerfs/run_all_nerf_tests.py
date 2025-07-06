"""Run all tests for NeRF variants."""

import unittest
import sys
import os

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

# Import all test cases
from tests.nerfs.test_mega_nerf import TestMegaNeRF
from tests.nerfs.test_inf_nerf import TestInfNeRF
from tests.nerfs.test_cnc_nerf import TestCNCNeRF
from tests.nerfs.test_dnmp_nerf import TestDNMPNeRF
from tests.nerfs.test_sdf_net import TestSDFNet
from tests.nerfs.test_occupancy_net import TestOccupancyNet
from tests.nerfs.test_classic_nerf import TestClassicNeRF
from tests.nerfs.test_mega_nerf_plus import TestMegaNeRFPlus
from tests.nerfs.test_mip_nerf import TestMIPNeRF
from tests.nerfs.test_nerfacto import TestNeRFacto
from tests.nerfs.test_pyramid_nerf import TestPyramidNeRF
from tests.nerfs.test_plenoxels import TestPlenoxels
from tests.nerfs.test_instant_ngp import TestInstantNGP
from tests.nerfs.test_bungee_nerf import TestBungeeNeRF
from tests.nerfs.test_block_nerf import TestBlockNeRF
from tests.nerfs.test_svraster import TestSVRaster


def create_test_suite():
    """Create a test suite containing all NeRF variant tests."""
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()

    # Add all test cases to suite
    test_cases = [
        TestMegaNeRF,
        TestInfNeRF,
        TestCNCNeRF,
        TestDNMPNeRF,
        TestSDFNet,
        TestOccupancyNet,
        TestClassicNeRF,
        TestMegaNeRFPlus,
        TestMIPNeRF,
        TestNeRFacto,
        TestPyramidNeRF,
        TestPlenoxels,
        TestInstantNGP,
        TestBungeeNeRF,
        TestBlockNeRF,
        TestSVRaster,
    ]

    for test_case in test_cases:
        suite.addTests(loader.loadTestsFromTestCase(test_case))

    return suite


if __name__ == "__main__":
    # Create and run test suite
    suite = create_test_suite()
    runner = unittest.TextTestRunner(verbosity=2)
    runner.run(suite)
