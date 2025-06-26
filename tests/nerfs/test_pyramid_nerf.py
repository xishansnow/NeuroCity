#!/usr/bin/env python3
"""
Test script for PyNeRF package
"""

import torch
import numpy as np
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

def test_imports():
    """Test if all modules can be imported"""
    print("Testing imports...")
    
    try:
        import sys
        import os
        sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'src'))
        
        from nerfs.pyramid_nerf import (
            PyNeRF, PyNeRFConfig, PyramidEncoder, PyramidRenderer, PyNeRFDataset, MultiScaleDataset, PyNeRFTrainer, compute_psnr, compute_ssim, create_pyramid_hierarchy
        )
        print("✓ All imports successful")
        return True
    except Exception as e:
        print(f"✗ Import failed: {e}")
        return False


def test_config():
    """Test configuration creation"""
    print("Testing configuration...")
    
    try:
        from nerfs.pyramid_nerf import PyNeRFConfig
        
        config = PyNeRFConfig()
        print(f"✓ Default config created: {config.num_levels} levels")
        
        # Test custom config
        custom_config = PyNeRFConfig(
            num_levels=6, base_resolution=32, max_resolution=1024
        )
        print(f"✓ Custom config created: {custom_config.num_levels} levels")
        return True
    except Exception as e:
        print(f"✗ Config test failed: {e}")
        return False


def test_pyramid_encoder():
    """Test pyramid encoder"""
    print("Testing pyramid encoder...")
    
    try:
        from nerfs.pyramid_nerf import PyramidEncoder
        
        encoder = PyramidEncoder(
            num_levels=4, base_resolution=16, max_resolution=256, features_per_level=4
        )
        
        # Test forward pass
        batch_size = 1024
        positions = torch.randn(batch_size, 3)
        
        features = encoder(positions)
        expected_features = 4 * 4  # 4 levels * 4 features per level
        
        assert features.shape == (batch_size, expected_features)
        print(f"✓ Encoder output shape: {features.shape}")
        return True
    except Exception as e:
        print(f"✗ Encoder test failed: {e}")
        return False


def test_model():
    """Test PyNeRF model"""
    print("Testing PyNeRF model...")
    
    try:
        from nerfs.pyramid_nerf import PyNeRF, PyNeRFConfig
        
        config = PyNeRFConfig(
            num_levels=4, base_resolution=16, max_resolution=256, batch_size=512
        )
        
        model = PyNeRF(config)
        print(f"✓ Model created with {sum(p.numel() for p in model.parameters()):, } parameters")
        
        # Test forward pass
        batch_size = 512
        rays_o = torch.randn(batch_size, 3)
        rays_d = torch.randn(batch_size, 3)
        rays_d = rays_d / torch.norm(rays_d, dim=-1, keepdim=True)  # Normalize
        bounds = torch.tensor([[2.0, 6.0]]).expand(batch_size, 2)
        
        with torch.no_grad():
            outputs = model(rays_o, rays_d, bounds)
        
        assert "rgb" in outputs
        assert outputs["rgb"].shape == (batch_size, 3)
        print(f"✓ Model forward pass successful: RGB shape {outputs['rgb'].shape}")
        return True
    except Exception as e:
        print(f"✗ Model test failed: {e}")
        return False


def test_utilities():
    """Test utility functions"""
    print("Testing utilities...")
    
    try:
        from nerfs.pyramid_nerf import (
            create_pyramid_hierarchy, compute_psnr, compute_ssim, compute_sample_area, get_pyramid_level
        )
        
        # Test pyramid hierarchy
        hierarchy = create_pyramid_hierarchy(
            num_levels=6, base_resolution=16, scale_factor=2.0, max_resolution=512
        )
        expected = [16, 32, 64, 128, 256, 512]
        assert hierarchy == expected
        print(f"✓ Pyramid hierarchy: {hierarchy}")
        
        # Test metrics
        img1 = torch.rand(64, 64, 3)
        img2 = torch.rand(64, 64, 3)
        
        psnr = compute_psnr(img1, img2)
        ssim = compute_ssim(img1, img2)
        
        print(f"✓ PSNR: {psnr:.4f}, SSIM: {ssim:.4f}")
        
        # Test sample area computation
        rays_o = torch.randn(100, 3)
        rays_d = torch.randn(100, 3)
        z_vals = torch.linspace(2.0, 6.0, 64).expand(100, 64)
        
        sample_areas = compute_sample_area(rays_o, rays_d, z_vals)
        assert sample_areas.shape == z_vals.shape
        print(f"✓ Sample areas shape: {sample_areas.shape}")
        
        return True
    except Exception as e:
        print(f"✗ Utilities test failed: {e}")
        return False


def test_renderer():
    """Test pyramid renderer"""
    print("Testing pyramid renderer...")
    
    try:
        from nerfs.pyramid_nerf import PyramidRenderer, PyNeRFConfig
        
        config = PyNeRFConfig()
        renderer = PyramidRenderer(config)
        
        # Test volume rendering
        batch_size = 256
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
        
        print(f"✓ Renderer output shapes: RGB {
            outputs['rgb'].shape,
        }
        return True
    except Exception as e:
        print(f"✗ Renderer test failed: {e}")
        return False


def main():
    """Run all tests"""
    print("=" * 50)
    print("PyNeRF Package Test Suite")
    print("=" * 50)
    
    tests = [
        test_imports, test_config, test_pyramid_encoder, test_model, test_utilities, test_renderer
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
        print()
    
    print("=" * 50)
    print(f"Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! PyNeRF package is working correctly.")
        return 0
    else:
        print("❌ Some tests failed. Please check the implementation.")
        return 1


if __name__ == "__main__":
    exit(main())
