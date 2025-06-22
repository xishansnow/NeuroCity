"""
NeRF Tests Package

Tests for all Neural Radiance Fields implementations in the NeuroCity project.

Available Tests:
- test_classic_nerf.py - Classic NeRF implementation tests
- test_instant_ngp.py - Instant-NGP tests  
- test_mip_nerf.py - Mip-NeRF tests
- test_grid_nerf.py - Grid-NeRF tests
- test_svraster.py - SVRaster tests
- test_plenoxels.py - Plenoxels tests
- test_bungee_nerf.py - Bungee-NeRF tests
- test_pyramid_nerf.py - Pyramid-NeRF tests
- test_nerfacto.py - Nerfacto tests
- test_mega_nerf_plus.py - Mega-NeRF Plus tests

Usage:
    Run all NeRF tests: python -m pytest tests/nerfs/
    Run specific test: python tests/nerfs/test_instant_ngp.py

Author: NeuroCity Team
"""

# Available NeRF tests
NERF_TESTS = [
    'test_classic_nerf',
    'test_instant_ngp',
    'test_mip_nerf', 
    'test_grid_nerf',
    'test_svraster',
    'test_plenoxels',
    'test_bungee_nerf',
    'test_pyramid_nerf',
    'test_nerfacto',
    'test_mega_nerf_plus'
]

def list_nerf_tests():
    """List all available NeRF tests."""
    return NERF_TESTS 