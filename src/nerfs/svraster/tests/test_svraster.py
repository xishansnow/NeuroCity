"""
Basic tests for SVRaster package.
"""

import pytest
import torch
import numpy as np
import sys
import os

# Add the parent directory to the Python path to import svraster directly
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from __init__ import (
    SVRasterConfig,
    SVRasterModel,
    SVRasterTrainerConfig,
    SVRasterRendererConfig,
    SVRasterDatasetConfig,
)


class TestSVRasterConfig:
    """Test SVRaster configuration."""
    
    def test_default_config(self):
        """Test default configuration creation."""
        config = SVRasterConfig()
        assert config.max_octree_levels > 0
        assert config.base_resolution > 0
        assert len(config.scene_bounds) == 6
    
    def test_custom_config(self):
        """Test custom configuration."""
        config = SVRasterConfig(
            max_octree_levels=8,
            base_resolution=128,
            scene_bounds=(-2.0, -2.0, -2.0, 2.0, 2.0, 2.0)
        )
        assert config.max_octree_levels == 8
        assert config.base_resolution == 128
        assert config.scene_bounds == (-2.0, -2.0, -2.0, 2.0, 2.0, 2.0)


class TestSVRasterModel:
    """Test SVRaster model."""
    
    def test_model_creation(self):
        """Test model creation."""
        config = SVRasterConfig(
            max_octree_levels=4,  # Small for testing
            base_resolution=32
        )
        model = SVRasterModel(config)
        assert model is not None
        assert model.config == config
    
    def test_model_forward(self):
        """Test model forward pass."""
        config = SVRasterConfig(
            max_octree_levels=4,
            base_resolution=32
        )
        model = SVRasterModel(config)
        
        # Test forward pass
        batch_size = 100
        ray_origins = torch.randn(batch_size, 3)
        ray_directions = torch.randn(batch_size, 3)
        ray_directions = ray_directions / ray_directions.norm(dim=-1, keepdim=True)
        
        outputs = model(ray_origins, ray_directions)
        
        assert "rgb" in outputs
        assert "depth" in outputs
        assert outputs["rgb"].shape == (batch_size, 3)
        assert outputs["depth"].shape == (batch_size,)


class TestSVRasterImports:
    """Test that all components can be imported."""
    
    def test_core_imports(self):
        """Test core component imports."""
        from svraster import (
            SVRasterModel,
            SVRasterConfig,
            SVRasterLoss,
        )
        assert SVRasterModel is not None
        assert SVRasterConfig is not None
        assert SVRasterLoss is not None
    
    def test_trainer_imports(self):
        """Test trainer component imports."""
        from svraster import (
            SVRasterTrainer,
            SVRasterTrainerConfig,
        )
        assert SVRasterTrainer is not None
        assert SVRasterTrainerConfig is not None
    
    def test_renderer_imports(self):
        """Test renderer component imports."""
        from svraster import (
            SVRasterRenderer,
            SVRasterRendererConfig,
        )
        assert SVRasterRenderer is not None
        assert SVRasterRendererConfig is not None
    
    def test_dataset_imports(self):
        """Test dataset component imports."""
        from svraster import (
            SVRasterDataset,
            SVRasterDatasetConfig,
        )
        assert SVRasterDataset is not None
        assert SVRasterDatasetConfig is not None
    
    def test_utils_imports(self):
        """Test utility function imports."""
        from svraster import (
            morton_encode_3d,
            morton_decode_3d,
            octree_subdivision,
            octree_pruning,
        )
        assert morton_encode_3d is not None
        assert morton_decode_3d is not None
        assert octree_subdivision is not None
        assert octree_pruning is not None
    
    def test_cuda_imports(self):
        """Test CUDA component imports (if available)."""
        try:
            from svraster.cuda import (
                SVRasterGPU,
                SVRasterGPUTrainer,
                EMAModel,
            )
            assert SVRasterGPU is not None
            assert SVRasterGPUTrainer is not None
            assert EMAModel is not None
        except ImportError:
            # CUDA components not available, skip test
            pytest.skip("CUDA components not available")


class TestSVRasterConfigurations:
    """Test different configuration combinations."""
    
    def test_trainer_config(self):
        """Test trainer configuration."""
        config = SVRasterTrainerConfig(
            num_epochs=10,
            learning_rate=1e-3,
            batch_size=2,
        )
        assert config.num_epochs == 10
        assert config.learning_rate == 1e-3
        assert config.batch_size == 2
    
    def test_renderer_config(self):
        """Test renderer configuration."""
        config = SVRasterRendererConfig(
            render_method="rasterization",
            image_height=256,
            image_width=256,
        )
        assert config.render_method == "rasterization"
        assert config.image_height == 256
        assert config.image_width == 256
    
    def test_dataset_config(self):
        """Test dataset configuration."""
        config = SVRasterDatasetConfig(
            data_dir="./test_data",
            dataset_type="blender",
            image_height=128,
            image_width=128,
        )
        assert config.data_dir == "./test_data"
        assert config.dataset_type == "blender"
        assert config.image_height == 128
        assert config.image_width == 128


def test_package_version():
    """Test package version is accessible."""
    assert hasattr(svraster, "__version__")
    assert isinstance(svraster.__version__, str)
    assert len(svraster.__version__) > 0


def test_cuda_availability():
    """Test CUDA availability detection."""
    assert hasattr(svraster, "CUDA_AVAILABLE")
    assert isinstance(svraster.CUDA_AVAILABLE, bool)


if __name__ == "__main__":
    pytest.main([__file__])
