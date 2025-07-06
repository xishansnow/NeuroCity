"""Test suite for Block-NeRF implementation.

This module contains unit tests and integration tests for Block-NeRF components:
- Network architecture tests
- Forward pass tests
- Device handling tests
- Training utilities tests
"""

import os
import torch
import pytest
import numpy as np
from pathlib import Path
from typing import Any, Generator

from src.nerfs.block_nerf.block_nerf_model import (
    BlockNeRFNetwork,
    BlockNeRF,
    positional_encoding,
    integrated_positional_encoding,
)

# Type aliases
Tensor = torch.Tensor

# Test configurations
TEST_CONFIGS = {
    "network": {
        "pos_encoding_levels": 8,
        "dir_encoding_levels": 4,
        "appearance_dim": 16,
        "exposure_dim": 4,
        "hidden_dim": 128,
        "num_layers": 4,
        "skip_connections": [2],
        "use_integrated_encoding": True,
    },
    "block": {
        "block_radius": 1.0,
        "num_appearance_embeddings": 100,
        "appearance_dim": 16,
        "exposure_dim": 4,
    },
}


# Fixtures
@pytest.fixture
def device() -> torch.device:
    """Get default device for testing."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@pytest.fixture
def network(device: torch.device) -> BlockNeRFNetwork:
    """Create BlockNeRFNetwork instance for testing."""
    return BlockNeRFNetwork(**TEST_CONFIGS["network"]).to(device)


@pytest.fixture
def block_nerf(device: torch.device) -> BlockNeRF:
    """Create BlockNeRF instance for testing."""
    block_center = torch.tensor([0.0, 0.0, 0.0], device=device)
    return BlockNeRF(
        network_config=TEST_CONFIGS["network"], block_center=block_center, **TEST_CONFIGS["block"]
    ).to(device)


@pytest.fixture
def sample_batch(device: torch.device) -> dict[str, Tensor]:
    """Create sample batch for testing."""
    batch_size = 32
    return {
        "positions": torch.randn(batch_size, 3, device=device),
        "directions": torch.randn(batch_size, 3, device=device),
        "appearance_ids": torch.randint(0, 100, (batch_size,), device=device),
        "exposure_values": torch.rand(batch_size, 1, device=device),
        "position_covs": torch.randn(batch_size, 3, 3, device=device),
    }


# Unit Tests
class TestPositionalEncoding:
    """Test positional encoding functions."""

    def test_positional_encoding(self, device: torch.device) -> None:
        """Test basic positional encoding."""
        x = torch.randn(10, 3, device=device)
        L = 4
        encoded = positional_encoding(x, L)

        assert encoded.shape == (10, 3 * 2 * L)
        assert torch.isfinite(encoded).all()

    def test_integrated_encoding(self, device: torch.device) -> None:
        """Test integrated positional encoding."""
        means = torch.randn(10, 3, device=device)
        covs = torch.randn(10, 3, 3, device=device)
        L = 4
        encoded = integrated_positional_encoding(means, covs, L)

        assert encoded.shape == (10, 3 * 2 * L)
        assert torch.isfinite(encoded).all()


class TestBlockNeRFNetwork:
    """Test BlockNeRFNetwork implementation."""

    def test_init(self, network: BlockNeRFNetwork) -> None:
        """Test network initialization."""
        assert isinstance(network, BlockNeRFNetwork)
        assert network.pos_encoding_levels == TEST_CONFIGS["network"]["pos_encoding_levels"]
        assert network.skip_connections == TEST_CONFIGS["network"]["skip_connections"]

    def test_encode_position(self, network: BlockNeRFNetwork, device: torch.device) -> None:
        """Test position encoding."""
        positions = torch.randn(10, 3, device=device)
        covs = torch.randn(10, 3, 3, device=device)

        # Test with covariance
        encoded = network.encode_position(positions, covs)
        assert torch.isfinite(encoded).all()

        # Test without covariance
        encoded = network.encode_position(positions)
        assert torch.isfinite(encoded).all()

    def test_encode_direction(self, network: BlockNeRFNetwork, device: torch.device) -> None:
        """Test direction encoding."""
        directions = torch.randn(10, 3, device=device)
        encoded = network.encode_direction(directions)
        assert torch.isfinite(encoded).all()

    def test_forward(self, network: BlockNeRFNetwork, sample_batch: dict[str, Tensor]) -> None:
        """Test network forward pass."""
        outputs = network(
            sample_batch["positions"],
            sample_batch["directions"],
            torch.randn_like(sample_batch["positions"]),  # appearance embedding
            torch.randn_like(sample_batch["positions"]),  # exposure
            sample_batch["position_covs"],
        )

        assert "density" in outputs
        assert "color" in outputs
        assert "features" in outputs
        assert torch.isfinite(outputs["density"]).all()
        assert torch.isfinite(outputs["color"]).all()
        assert (outputs["color"] >= 0).all() and (outputs["color"] <= 1).all()


class TestBlockNeRF:
    """Test BlockNeRF implementation."""

    def test_init(self, block_nerf: BlockNeRF) -> None:
        """Test model initialization."""
        assert isinstance(block_nerf, BlockNeRF)
        assert hasattr(block_nerf, "network")
        assert hasattr(block_nerf, "appearance_embeddings")
        assert hasattr(block_nerf, "scaler")

    def test_encode_exposure(self, block_nerf: BlockNeRF, device: torch.device) -> None:
        """Test exposure encoding."""
        exposure = torch.rand(10, 1, device=device)
        encoded = block_nerf.encode_exposure(exposure)
        assert torch.isfinite(encoded).all()

    def test_get_appearance_embedding(self, block_nerf: BlockNeRF, device: torch.device) -> None:
        """Test appearance embedding lookup."""
        ids = torch.randint(0, 100, (10,), device=device)
        embeddings = block_nerf.get_appearance_embedding(ids)
        assert embeddings.shape == (10, TEST_CONFIGS["block"]["appearance_dim"])
        assert torch.isfinite(embeddings).all()

    def test_is_in_block(self, block_nerf: BlockNeRF, device: torch.device) -> None:
        """Test block boundary checking."""
        # Test points inside block
        inside_points = torch.randn(10, 3, device=device) * 0.5  # within unit sphere
        inside_mask = block_nerf.is_in_block(inside_points)
        assert inside_mask.sum() > 0

        # Test points outside block
        outside_points = torch.randn(10, 3, device=device) * 2.0  # outside unit sphere
        outside_mask = block_nerf.is_in_block(outside_points)
        assert (~outside_mask).sum() > 0

    def test_forward(self, block_nerf: BlockNeRF, sample_batch: dict[str, Tensor]) -> None:
        """Test model forward pass."""
        outputs = block_nerf(
            sample_batch["positions"],
            sample_batch["directions"],
            sample_batch["appearance_ids"],
            sample_batch["exposure_values"],
            sample_batch["position_covs"],
        )

        assert "density" in outputs
        assert "color" in outputs
        assert torch.isfinite(outputs["density"]).all()
        assert torch.isfinite(outputs["color"]).all()
        assert (outputs["color"] >= 0).all() and (outputs["color"] <= 1).all()

    def test_get_block_info(self, block_nerf: BlockNeRF) -> None:
        """Test block info retrieval."""
        info = block_nerf.get_block_info()
        assert "center" in info
        assert "radius" in info
        assert torch.isfinite(info["center"]).all()
        assert info["radius"] == TEST_CONFIGS["block"]["block_radius"]


# Integration Tests
@pytest.mark.slow
class TestBlockNeRFIntegration:
    """Integration tests for Block-NeRF."""

    def test_training_step(self, block_nerf: BlockNeRF, sample_batch: dict[str, Tensor]) -> None:
        """Test training step with autocast and gradient scaling."""
        block_nerf.train()

        # Forward pass with autocast
        device_type = "cuda" if torch.cuda.is_available() else "cpu"
        with torch.amp.autocast(device_type=device_type):
            outputs = block_nerf(
                sample_batch["positions"],
                sample_batch["directions"],
                sample_batch["appearance_ids"],
                sample_batch["exposure_values"],
                sample_batch["position_covs"],
            )

        # Compute dummy loss
        loss = outputs["density"].mean() + outputs["color"].mean()

        # Backward pass with gradient scaling
        block_nerf.scaler.scale(loss).backward()

        # Check gradients
        for param in block_nerf.parameters():
            if param.requires_grad:
                assert param.grad is not None
                assert torch.isfinite(param.grad).all()

    def test_device_transfer(self, block_nerf: BlockNeRF, sample_batch: dict[str, Tensor]) -> None:
        """Test device transfer and non-blocking operations."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")

        # Move model to CPU
        block_nerf = block_nerf.cpu()

        # Move to CUDA non-blocking
        block_nerf = block_nerf.cuda(non_blocking=True)

        # Test forward pass
        outputs = block_nerf(
            sample_batch["positions"],
            sample_batch["directions"],
            sample_batch["appearance_ids"],
            sample_batch["exposure_values"],
            sample_batch["position_covs"],
        )

        assert all(v.device.type == "cuda" for v in outputs.values())

    @pytest.mark.parametrize("batch_size", [1, 4, 16])
    def test_batch_processing(
        self, block_nerf: BlockNeRF, device: torch.device, batch_size: int
    ) -> None:
        """Test processing different batch sizes."""
        batch = {
            "positions": torch.randn(batch_size, 3, device=device),
            "directions": torch.randn(batch_size, 3, device=device),
            "appearance_ids": torch.randint(0, 100, (batch_size,), device=device),
            "exposure_values": torch.rand(batch_size, 1, device=device),
            "position_covs": torch.randn(batch_size, 3, 3, device=device),
        }

        outputs = block_nerf(**batch)
        assert all(v.shape[0] == batch_size for v in outputs.values())


if __name__ == "__main__":
    pytest.main([__file__])
