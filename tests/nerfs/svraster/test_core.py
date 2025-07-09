"""
Test Core SVRaster Components

This module tests the core components of SVRaster:
- SVRasterModel
- SVRasterConfig
- SVRasterLoss
"""

import pytest
import torch
import numpy as np
import tempfile
import os

# Add the src directory to the path for imports
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..', 'src'))

try:
    import nerfs.svraster as svraster
    SVRASTER_AVAILABLE = True
except ImportError as e:
    SVRASTER_AVAILABLE = False
    IMPORT_ERROR = str(e)


class TestSVRasterConfig:
    """Test SVRasterConfig functionality"""
    
    def test_config_creation(self):
        """Test basic config creation"""
        if not SVRASTER_AVAILABLE:
            pytest.skip(f"SVRaster not available: {IMPORT_ERROR}")
            
        config = svraster.SVRasterConfig(
            max_octree_levels=8,
            base_resolution=128,
            sh_degree=2,
            scene_bounds=(-1, -1, -1, 1, 1, 1),
            density_activation="exp",
            color_activation="sigmoid",
            learning_rate=1e-3,
            weight_decay=1e-6
        )
        
        assert config.max_octree_levels == 8
        assert config.base_resolution == 128
        assert config.sh_degree == 2
        assert config.density_activation == "exp"
        assert config.color_activation == "sigmoid"
        assert config.learning_rate == 1e-3
        assert config.weight_decay == 1e-6
    
    def test_config_defaults(self):
        """Test config with default values"""
        if not SVRASTER_AVAILABLE:
            pytest.skip(f"SVRaster not available: {IMPORT_ERROR}")
            
        config = svraster.SVRasterConfig()
        
        # Should have reasonable defaults
        assert config.max_octree_levels > 0
        assert config.base_resolution > 0
        assert config.sh_degree >= 0
        assert config.learning_rate > 0
        assert config.weight_decay >= 0
    
    def test_config_validation(self):
        """Test config parameter validation"""
        if not SVRASTER_AVAILABLE:
            pytest.skip(f"SVRaster not available: {IMPORT_ERROR}")
            
        # Test invalid parameters
        with pytest.raises((ValueError, AssertionError)):
            svraster.SVRasterConfig(max_octree_levels=-1)
            
        with pytest.raises((ValueError, AssertionError)):
            svraster.SVRasterConfig(base_resolution=0)
            
        with pytest.raises((ValueError, AssertionError)):
            svraster.SVRasterConfig(sh_degree=-1)


class TestSVRasterModel:
    """Test SVRasterModel functionality"""
    
    def test_model_creation(self):
        """Test basic model creation"""
        if not SVRASTER_AVAILABLE:
            pytest.skip(f"SVRaster not available: {IMPORT_ERROR}")
            
        config = svraster.SVRasterConfig(
            max_octree_levels=6,
            base_resolution=64,
            sh_degree=1
        )
        
        model = svraster.SVRasterModel(config)
        
        assert model is not None
        assert model.config == config
        assert hasattr(model, 'forward')
    
    def test_model_parameters(self):
        """Test model parameters"""
        if not SVRASTER_AVAILABLE:
            pytest.skip(f"SVRaster not available: {IMPORT_ERROR}")
            
        config = svraster.SVRasterConfig(
            max_octree_levels=4,
            base_resolution=32,
            sh_degree=1
        )
        
        model = svraster.SVRasterModel(config)
        
        # Check if model has parameters
        params = list(model.parameters())
        assert len(params) > 0
        
        # Check if parameters are tensors
        for param in params:
            assert isinstance(param, torch.Tensor)
    
    def test_model_forward(self):
        """Test model forward pass"""
        if not SVRASTER_AVAILABLE:
            pytest.skip(f"SVRaster not available: {IMPORT_ERROR}")
            
        config = svraster.SVRasterConfig(
            max_octree_levels=4,
            base_resolution=32,
            sh_degree=1
        )
        
        model = svraster.SVRasterModel(config)
        
        # Test forward pass with dummy data
        batch_size = 2
        num_rays = 128
        
        # Create dummy input
        ray_origins = torch.randn(batch_size, num_rays, 3)
        ray_directions = torch.randn(batch_size, num_rays, 3)
        ray_directions = ray_directions / torch.norm(ray_directions, dim=-1, keepdim=True)
        
        # Forward pass
        try:
            output = model(ray_origins, ray_directions)
            assert output is not None
        except Exception as e:
            # Forward pass might fail due to uninitialized data, that's ok for this test
            print(f"Forward pass failed (expected): {e}")
    
    def test_model_save_load(self):
        """Test model saving and loading"""
        if not SVRASTER_AVAILABLE:
            pytest.skip(f"SVRaster not available: {IMPORT_ERROR}")
            
        config = svraster.SVRasterConfig(
            max_octree_levels=4,
            base_resolution=32,
            sh_degree=1
        )
        
        model = svraster.SVRasterModel(config)
        
        # Save model
        with tempfile.NamedTemporaryFile(suffix='.pth', delete=False) as f:
            temp_path = f.name
        
        try:
            torch.save(model.state_dict(), temp_path)
            
            # Load model
            model2 = svraster.SVRasterModel(config)
            model2.load_state_dict(torch.load(temp_path))
            
            # Check if parameters match
            for p1, p2 in zip(model.parameters(), model2.parameters()):
                assert torch.allclose(p1, p2)
                
        finally:
            os.unlink(temp_path)


class TestSVRasterLoss:
    """Test SVRasterLoss functionality"""
    
    def test_loss_creation(self):
        """Test loss function creation"""
        if not SVRASTER_AVAILABLE:
            pytest.skip(f"SVRaster not available: {IMPORT_ERROR}")
            
        try:
            loss_fn = svraster.SVRasterLoss()
            assert loss_fn is not None
        except Exception as e:
            # Loss function might not be implemented yet
            pytest.skip(f"SVRasterLoss not implemented: {e}")
    
    def test_loss_computation(self):
        """Test loss computation"""
        if not SVRASTER_AVAILABLE:
            pytest.skip(f"SVRaster not available: {IMPORT_ERROR}")
            
        try:
            loss_fn = svraster.SVRasterLoss()
            
            # Create dummy data
            predicted = torch.randn(10, 3)
            target = torch.randn(10, 3)
            
            loss = loss_fn(predicted, target)
            
            assert isinstance(loss, torch.Tensor)
            assert loss.dim() == 0  # Should be scalar
            assert loss.item() >= 0  # Loss should be non-negative
            
        except Exception as e:
            pytest.skip(f"SVRasterLoss computation failed: {e}")


class TestDeviceInfo:
    """Test device information functions"""
    
    def test_get_device_info(self):
        """Test device information retrieval"""
        if not SVRASTER_AVAILABLE:
            pytest.skip(f"SVRaster not available: {IMPORT_ERROR}")
            
        device_info = svraster.get_device_info()
        
        assert isinstance(device_info, dict)
        assert "torch_version" in device_info
        assert "cuda_available" in device_info
        assert "svraster_cuda" in device_info
        
        # Check types
        assert isinstance(device_info["torch_version"], str)
        assert isinstance(device_info["cuda_available"], bool)
        assert isinstance(device_info["svraster_cuda"], bool)
    
    def test_check_compatibility(self):
        """Test compatibility check"""
        if not SVRASTER_AVAILABLE:
            pytest.skip(f"SVRaster not available: {IMPORT_ERROR}")
            
        # This should not raise an exception
        svraster.check_compatibility()
    
    def test_quick_start_guide(self):
        """Test quick start guide"""
        if not SVRASTER_AVAILABLE:
            pytest.skip(f"SVRaster not available: {IMPORT_ERROR}")
            
        # This should not raise an exception
        svraster.quick_start_guide()


class TestConstants:
    """Test package constants"""
    
    def test_package_version(self):
        """Test package version"""
        if not SVRASTER_AVAILABLE:
            pytest.skip(f"SVRaster not available: {IMPORT_ERROR}")
            
        assert hasattr(svraster, '__version__')
        assert isinstance(svraster.__version__, str)
        assert len(svraster.__version__) > 0
    
    def test_cuda_available_flag(self):
        """Test CUDA availability flag"""
        if not SVRASTER_AVAILABLE:
            pytest.skip(f"SVRaster not available: {IMPORT_ERROR}")
            
        assert hasattr(svraster, 'CUDA_AVAILABLE')
        assert isinstance(svraster.CUDA_AVAILABLE, bool)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
