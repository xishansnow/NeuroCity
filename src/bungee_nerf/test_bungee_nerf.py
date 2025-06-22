"""
Test script for BungeeNeRF
"""

import torch
import numpy as np
import logging
import sys
import os

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from bungee_nerf import (
    BungeeNeRF, BungeeNeRFConfig, ProgressiveBlock,
    ProgressivePositionalEncoder, MultiScaleEncoder,
    MultiScaleRenderer, LevelOfDetailRenderer,
    BungeeNeRFDataset, MultiScaleDataset,
    BungeeNeRFTrainer, ProgressiveTrainer,
    compute_scale_factor, get_level_of_detail,
    progressive_positional_encoding, multiscale_sampling,
    compute_multiscale_loss, save_bungee_model, load_bungee_model
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_config():
    """Test BungeeNeRF configuration"""
    logger.info("Testing BungeeNeRF configuration...")
    
    config = BungeeNeRFConfig(
        num_stages=4,
        base_resolution=16,
        max_resolution=2048,
        hidden_dim=256,
        num_layers=8
    )
    
    assert config.num_stages == 4
    assert config.base_resolution == 16
    assert config.max_resolution == 2048
    assert config.hidden_dim == 256
    assert config.num_layers == 8
    assert len(config.skip_layers) > 0
    assert len(config.scale_weights) == 4
    assert len(config.distance_thresholds) == 4
    
    logger.info("✓ Configuration test passed")


def test_progressive_encoder():
    """Test progressive positional encoder"""
    logger.info("Testing progressive positional encoder...")
    
    encoder = ProgressivePositionalEncoder(
        num_freqs_base=4,
        num_freqs_max=10,
        include_input=True
    )
    
    # Test encoding
    batch_size = 100
    x = torch.randn(batch_size, 3)
    
    # Test different stages
    for stage in range(4):
        encoder.set_current_stage(stage)
        encoded = encoder(x)
        
        expected_dim = encoder.get_output_dim()
        assert encoded.shape == (batch_size, expected_dim)
        assert not torch.isnan(encoded).any()
        assert not torch.isinf(encoded).any()
    
    logger.info("✓ Progressive encoder test passed")


def test_multiscale_encoder():
    """Test multi-scale encoder"""
    logger.info("Testing multi-scale encoder...")
    
    encoder = MultiScaleEncoder(
        num_scales=4,
        base_freqs=4,
        max_freqs=10
    )
    
    batch_size = 100
    x = torch.randn(batch_size, 3)
    distances = torch.rand(batch_size) * 100
    
    # Test different scales
    for scale in range(4):
        encoded = encoder(x, scale=scale)
        assert encoded.shape[0] == batch_size
        assert not torch.isnan(encoded).any()
    
    # Test adaptive encoding
    encoded = encoder(x, distances=distances)
    assert encoded.shape[0] == batch_size
    assert not torch.isnan(encoded).any()
    
    logger.info("✓ Multi-scale encoder test passed")


def test_progressive_block():
    """Test progressive block"""
    logger.info("Testing progressive block...")
    
    block = ProgressiveBlock(
        input_dim=256,
        hidden_dim=128,
        num_layers=4,
        output_dim=3
    )
    
    batch_size = 100
    x = torch.randn(batch_size, 256)
    
    output = block(x)
    assert output.shape == (batch_size, 3)
    assert not torch.isnan(output).any()
    assert not torch.isinf(output).any()
    
    logger.info("✓ Progressive block test passed")


def test_bungee_nerf_model():
    """Test BungeeNeRF model"""
    logger.info("Testing BungeeNeRF model...")
    
    config = BungeeNeRFConfig(
        num_stages=2,  # Smaller for testing
        hidden_dim=128,
        num_layers=4,
        num_samples=32
    )
    
    model = BungeeNeRF(config)
    
    # Test model info
    info = model.get_progressive_info()
    assert "current_stage" in info
    assert "num_stages" in info
    assert "total_parameters" in info
    
    # Test forward pass
    batch_size = 10
    rays_o = torch.randn(batch_size, 3)
    rays_d = torch.randn(batch_size, 3)
    rays_d = rays_d / torch.norm(rays_d, dim=-1, keepdim=True)  # Normalize
    bounds = torch.tensor([[2.0, 6.0]]).expand(batch_size, 2)
    distances = torch.rand(batch_size) * 10 + 1
    
    outputs = model(rays_o, rays_d, bounds, distances)
    
    assert "rgb" in outputs
    assert "depth" in outputs
    assert outputs["rgb"].shape == (batch_size, 3)
    assert outputs["depth"].shape == (batch_size,)
    assert not torch.isnan(outputs["rgb"]).any()
    assert not torch.isnan(outputs["depth"]).any()
    
    # Test progressive stages
    for stage in range(2):
        model.set_current_stage(stage)
        outputs = model(rays_o, rays_d, bounds, distances)
        assert "rgb" in outputs
        assert outputs["rgb"].shape == (batch_size, 3)
    
    logger.info("✓ BungeeNeRF model test passed")


def test_multiscale_renderer():
    """Test multi-scale renderer"""
    logger.info("Testing multi-scale renderer...")
    
    config = BungeeNeRFConfig()
    renderer = MultiScaleRenderer(config)
    
    batch_size = 10
    num_samples = 64
    
    rgb = torch.rand(batch_size, num_samples, 3)
    sigma = torch.rand(batch_size, num_samples)
    z_vals = torch.linspace(2.0, 6.0, num_samples).expand(batch_size, num_samples)
    rays_d = torch.randn(batch_size, 3)
    rays_d = rays_d / torch.norm(rays_d, dim=-1, keepdim=True)
    
    outputs = renderer.render(rgb, sigma, z_vals, rays_d)
    
    assert "rgb" in outputs
    assert "depth" in outputs
    assert "acc" in outputs
    assert outputs["rgb"].shape == (batch_size, 3)
    assert outputs["depth"].shape == (batch_size,)
    assert outputs["acc"].shape == (batch_size,)
    
    logger.info("✓ Multi-scale renderer test passed")


def test_utility_functions():
    """Test utility functions"""
    logger.info("Testing utility functions...")
    
    # Test scale factor computation
    distances = torch.tensor([150.0, 75.0, 30.0, 5.0])
    scale_thresholds = [100.0, 50.0, 25.0, 10.0]
    
    scale_factors = compute_scale_factor(distances, scale_thresholds)
    assert scale_factors.shape == distances.shape
    assert (scale_factors > 0).all()
    
    # Test level of detail
    lod_levels = get_level_of_detail(distances, scale_thresholds)
    assert lod_levels.shape == distances.shape
    assert (lod_levels >= 0).all()
    assert (lod_levels <= len(scale_thresholds)).all()
    
    # Test progressive positional encoding
    x = torch.randn(100, 3)
    encoded = progressive_positional_encoding(x, num_freqs=6, stage=2, max_stages=4)
    assert encoded.shape[0] == 100
    assert encoded.shape[1] > 3  # Should be larger than input
    
    logger.info("✓ Utility functions test passed")


def test_model_save_load():
    """Test model save/load functionality"""
    logger.info("Testing model save/load...")
    
    config = BungeeNeRFConfig(hidden_dim=64, num_layers=4)  # Small model for testing
    model = BungeeNeRF(config)
    
    # Save model
    save_path = "/tmp/test_bungee_model.pth"
    save_bungee_model(model, config.__dict__, save_path, stage=1, epoch=10)
    
    # Load model
    new_model = BungeeNeRF(config)
    loaded_model, loaded_config = load_bungee_model(new_model, save_path, device="cpu")
    
    # Check if models are equivalent
    for p1, p2 in zip(model.parameters(), loaded_model.parameters()):
        assert torch.allclose(p1, p2, atol=1e-6)
    
    # Clean up
    os.remove(save_path)
    
    logger.info("✓ Model save/load test passed")


def run_all_tests():
    """Run all tests"""
    logger.info("Running BungeeNeRF tests...")
    
    try:
        test_config()
        test_progressive_encoder()
        test_multiscale_encoder()
        test_progressive_block()
        test_bungee_nerf_model()
        test_multiscale_renderer()
        test_utility_functions()
        test_model_save_load()
        
        logger.info("🎉 All tests passed!")
        return True
        
    except Exception as e:
        logger.error(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
