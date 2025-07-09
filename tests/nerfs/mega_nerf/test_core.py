"""
Test Mega-NeRF Core Module

This module tests the core Mega-NeRF components including:
- MegaNeRFConfig
- PositionalEncoding
- MegaNeRFSubmodule
- MegaNeRF
"""

import pytest
import torch
import numpy as np
from src.nerfs.mega_nerf.core import MegaNeRFConfig, PositionalEncoding, MegaNeRFSubmodule, MegaNeRF


class TestMegaNeRFConfig:
    """Test MegaNeRFConfig class."""

    def test_default_config(self):
        """Test default configuration creation."""
        config = MegaNeRFConfig()

        assert config.num_submodules == 8
        assert config.grid_size == (4, 2)
        assert config.hidden_dim == 256
        assert config.num_layers == 8
        assert config.use_viewdirs is True
        assert config.skip_connections == [4]

    def test_custom_config(self):
        """Test custom configuration creation."""
        config = MegaNeRFConfig(
            num_submodules=4,
            grid_size=(2, 2),
            hidden_dim=128,
            num_layers=6,
            use_viewdirs=False,
            skip_connections=[2, 4],
        )

        assert config.num_submodules == 4
        assert config.grid_size == (2, 2)
        assert config.hidden_dim == 128
        assert config.num_layers == 6
        assert config.use_viewdirs is False
        assert config.skip_connections == [2, 4]

    def test_validation(self):
        """Test configuration validation."""
        # Test invalid num_submodules
        with pytest.raises(ValueError, match="num_submodules must be positive"):
            MegaNeRFConfig(num_submodules=0)

        # Test invalid hidden_dim
        with pytest.raises(ValueError, match="hidden_dim must be positive"):
            MegaNeRFConfig(hidden_dim=0)

        # Test invalid overlap_factor
        with pytest.raises(ValueError, match="overlap_factor must be between 0 and 1"):
            MegaNeRFConfig(overlap_factor=1.5)

        with pytest.raises(ValueError, match="overlap_factor must be between 0 and 1"):
            MegaNeRFConfig(overlap_factor=-0.1)


class TestPositionalEncoding:
    """Test PositionalEncoding class."""

    def test_initialization(self):
        """Test positional encoding initialization."""
        encoder = PositionalEncoding(input_dim=3, max_freq_log2=8, num_freqs=8)

        assert encoder.input_dim == 3
        assert encoder.max_freq_log2 == 8
        assert encoder.num_freqs == 8
        assert encoder.output_dim == 3 * (1 + 2 * 8)  # 51

    def test_forward(self, device):
        """Test positional encoding forward pass."""
        encoder = PositionalEncoding(input_dim=3, max_freq_log2=4, num_freqs=4)
        encoder = encoder.to(device)

        x = torch.randn(10, 3, device=device)
        encoded = encoder(x)

        assert encoded.shape == (10, 3 * (1 + 2 * 4))  # (10, 27)
        assert not torch.isnan(encoded).any()
        assert not torch.isinf(encoded).any()

    def test_frequency_bands(self):
        """Test frequency bands creation."""
        encoder = PositionalEncoding(input_dim=2, max_freq_log2=2, num_freqs=3)

        expected_freqs = 2.0 ** torch.linspace(0.0, 2.0, 3)
        assert torch.allclose(encoder.freq_bands, expected_freqs)


class TestMegaNeRFSubmodule:
    """Test MegaNeRFSubmodule class."""

    def test_initialization(self, mega_nerf_config, device):
        """Test submodule initialization."""
        centroid = np.array([1.0, 2.0, 3.0])
        submodule = MegaNeRFSubmodule(mega_nerf_config, centroid)
        submodule = submodule.to(device)

        assert submodule.config == mega_nerf_config
        assert torch.allclose(submodule.centroid, torch.tensor(centroid))

        # Check network components
        assert hasattr(submodule, "pos_encoder")
        assert hasattr(submodule, "dir_encoder")
        assert hasattr(submodule, "main_layers")
        assert hasattr(submodule, "density_head")

    def test_forward_with_viewdirs(self, mega_nerf_config, device, sample_points, sample_viewdirs):
        """Test submodule forward pass with view directions."""
        centroid = np.array([0.0, 0.0, 0.0])
        submodule = MegaNeRFSubmodule(mega_nerf_config, centroid)
        submodule = submodule.to(device)

        points = sample_points.to(device)
        viewdirs = sample_viewdirs.to(device)

        density, color = submodule(points, viewdirs)

        assert density.shape == (points.shape[0], 1)
        assert color.shape == (points.shape[0], 3)
        assert not torch.isnan(density).any()
        assert not torch.isnan(color).any()
        assert torch.all(density >= 0)  # Density should be non-negative
        assert torch.all(color >= 0) and torch.all(color <= 1)  # Color should be in [0, 1]

    def test_forward_without_viewdirs(self, mega_nerf_config, device, sample_points):
        """Test submodule forward pass without view directions."""
        # Create config without view directions
        config = MegaNeRFConfig(
            num_submodules=4, grid_size=(2, 2), hidden_dim=64, num_layers=4, use_viewdirs=False
        )

        centroid = np.array([0.0, 0.0, 0.0])
        submodule = MegaNeRFSubmodule(config, centroid)
        submodule = submodule.to(device)

        points = sample_points.to(device)

        density, color = submodule(points)

        assert density.shape == (points.shape[0], 1)
        assert color.shape == (points.shape[0], 3)
        assert not torch.isnan(density).any()
        assert not torch.isnan(color).any()

    def test_centroid_operations(self, mega_nerf_config, device):
        """Test centroid getter and setter."""
        centroid = np.array([1.0, 2.0, 3.0])
        submodule = MegaNeRFSubmodule(mega_nerf_config, centroid)
        submodule = submodule.to(device)

        # Test getter
        retrieved_centroid = submodule.get_centroid()
        assert torch.allclose(retrieved_centroid, torch.tensor(centroid))

        # Test setter
        new_centroid = torch.tensor([4.0, 5.0, 6.0], device=device)
        submodule.set_centroid(new_centroid)
        assert torch.allclose(submodule.centroid, new_centroid)


class TestMegaNeRF:
    """Test MegaNeRF class."""

    def test_initialization(self, mega_nerf_config, device):
        """Test MegaNeRF initialization."""
        model = MegaNeRF(mega_nerf_config)
        model = model.to(device)

        assert model.config == mega_nerf_config
        assert len(model.submodules) == mega_nerf_config.num_submodules
        assert hasattr(model, "centroids")
        assert hasattr(model, "foreground_bounds")

    def test_spatial_grid_creation(self, mega_nerf_config):
        """Test spatial grid creation."""
        model = MegaNeRF(mega_nerf_config)

        # Check that centroids are created correctly
        assert len(model.centroids) == mega_nerf_config.num_submodules

        # Check that centroids are within scene bounds
        x_min, y_min, z_min, x_max, y_max, z_max = mega_nerf_config.scene_bounds
        for centroid in model.centroids:
            assert x_min <= centroid[0] <= x_max
            assert y_min <= centroid[1] <= y_max
            assert z_min <= centroid[2] <= z_max

    def test_foreground_bounds(self, mega_nerf_config):
        """Test foreground bounds computation."""
        model = MegaNeRF(mega_nerf_config)

        assert len(model.foreground_bounds) == 6
        x_min, y_min, z_min, x_max, y_max, z_max = model.foreground_bounds

        # Foreground bounds should be smaller than scene bounds
        scene_x_min, scene_y_min, scene_z_min, scene_x_max, scene_y_max, scene_z_max = (
            mega_nerf_config.scene_bounds
        )
        assert x_min > scene_x_min
        assert y_min > scene_y_min
        assert z_min > scene_z_min
        assert x_max < scene_x_max
        assert y_max < scene_y_max
        assert z_max < scene_z_max

    def test_forward(self, model, device, sample_points, sample_viewdirs):
        """Test MegaNeRF forward pass."""
        model = model.to(device)
        points = sample_points.to(device)
        viewdirs = sample_viewdirs.to(device)

        density, color = model(points, viewdirs)

        assert density.shape == (points.shape[0], 1)
        assert color.shape == (points.shape[0], 3)
        assert not torch.isnan(density).any()
        assert not torch.isnan(color).any()

    def test_forward_without_viewdirs(self, mega_nerf_config, device, sample_points):
        """Test MegaNeRF forward pass without view directions."""
        config = MegaNeRFConfig(
            num_submodules=4, grid_size=(2, 2), hidden_dim=64, num_layers=4, use_viewdirs=False
        )

        model = MegaNeRF(config)
        model = model.to(device)
        points = sample_points.to(device)

        density, color = model(points)

        assert density.shape == (points.shape[0], 1)
        assert color.shape == (points.shape[0], 3)
        assert not torch.isnan(density).any()
        assert not torch.isnan(color).any()

    def test_submodule_centroids(self, model):
        """Test getting submodule centroids."""
        centroids = model.get_submodule_centroids()

        assert len(centroids) == len(model.submodules)
        for i, centroid in enumerate(centroids):
            assert np.array_equal(centroid, model.centroids[i])

    def test_submodule_bounds(self, model):
        """Test getting submodule bounds."""
        for i in range(len(model.submodules)):
            bounds = model.get_submodule_bounds(i)

            assert len(bounds) == 6
            x_min, y_min, z_min, x_max, y_max, z_max = bounds

            # Bounds should be valid
            assert x_min < x_max
            assert y_min < y_max
            assert z_min < z_max

    def test_relevant_submodules(self, model, device):
        """Test finding relevant submodules."""
        camera_position = torch.tensor([0.0, 0.0, 0.0], device=device)

        relevant_modules = model.get_relevant_submodules(camera_position, max_distance=10.0)

        assert isinstance(relevant_modules, list)
        assert all(isinstance(idx, int) for idx in relevant_modules)
        assert all(0 <= idx < len(model.submodules) for idx in relevant_modules)

    def test_model_info(self, model):
        """Test getting model information."""
        info = model.get_model_info()

        assert "num_submodules" in info
        assert "total_parameters" in info
        assert "trainable_parameters" in info
        assert "grid_size" in info
        assert "scene_bounds" in info
        assert "foreground_bounds" in info

        assert info["num_submodules"] == len(model.submodules)
        assert info["total_parameters"] > 0
        assert info["trainable_parameters"] > 0

    def test_invalid_submodule_index(self, model):
        """Test error handling for invalid submodule indices."""
        with pytest.raises(ValueError, match="Invalid submodule index"):
            model.get_submodule_bounds(len(model.submodules))

        with pytest.raises(ValueError, match="Invalid submodule index"):
            model.save_submodule(len(model.submodules), "dummy.pth")

        with pytest.raises(ValueError, match="Invalid submodule index"):
            model.load_submodule(len(model.submodules), "dummy.pth")
