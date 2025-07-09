"""Test configuration for pytest."""

import os
import sys
import pytest

# Add the package root to Python path
package_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, package_root)

@pytest.fixture(scope="session")
def cuda_device():
    """Provide CUDA device if available."""
    import torch
    if torch.cuda.is_available():
        return torch.device('cuda')
    else:
        pytest.skip("CUDA not available")

@pytest.fixture
def sample_config():
    """Provide a sample configuration for testing."""
    from block_nerf import BlockNeRFConfig
    return BlockNeRFConfig(
        block_size=32,
        max_blocks=10,
        appearance_embedding_dim=16,
        pose_refinement_steps=100
    )
