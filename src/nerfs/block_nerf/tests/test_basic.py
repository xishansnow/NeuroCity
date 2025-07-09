"""Test suite for Block-NeRF package."""

import pytest
import torch
import numpy as np
from block_nerf import (
    BlockNeRFConfig,
    BlockNeRFModel,
    check_compatibility,
    get_device_info,
)


def test_import():
    """Test that all modules can be imported."""
    import block_nerf
    assert hasattr(block_nerf, '__version__')
    assert hasattr(block_nerf, 'BlockNeRFConfig')
    assert hasattr(block_nerf, 'BlockNeRFModel')


def test_config_creation():
    """Test BlockNeRF configuration creation."""
    config = BlockNeRFConfig()
    assert isinstance(config, BlockNeRFConfig)
    
    # Test with custom parameters
    config = BlockNeRFConfig(
        block_size=64,
        max_blocks=100,
        appearance_embedding_dim=32
    )
    assert config.block_size == 64
    assert config.max_blocks == 100


def test_device_info():
    """Test device information retrieval."""
    device_info = get_device_info()
    assert 'device' in device_info
    assert 'device_name' in device_info
    assert 'memory_total' in device_info


def test_compatibility_check():
    """Test compatibility checking."""
    result = check_compatibility()
    assert isinstance(result, dict)
    assert 'torch_version' in result
    assert 'cuda_available' in result


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_cuda_functionality():
    """Test CUDA-specific functionality."""
    device = torch.device('cuda')
    tensor = torch.randn(100, 3, device=device)
    assert tensor.is_cuda
    assert tensor.device.type == 'cuda'


def test_model_creation():
    """Test BlockNeRF model creation."""
    config = BlockNeRFConfig(
        block_size=32,
        max_blocks=10,
        appearance_embedding_dim=16
    )
    
    model = BlockNeRFModel(config)
    assert isinstance(model, BlockNeRFModel)
    assert model.config == config


if __name__ == "__main__":
    pytest.main([__file__])
