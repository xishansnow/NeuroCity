"""
Block-NeRF Test Suite

This package contains comprehensive tests for the Block-NeRF implementation,
including unit tests, integration tests, and performance benchmarks.

Test Categories:
- Unit Tests: Individual component testing
- Integration Tests: End-to-end workflow testing
- Performance Tests: Benchmark and profiling tests
- Compatibility Tests: Device and version compatibility

Test Structure:
- test_core.py: Core model and configuration tests
- test_trainer.py: Training pipeline tests
- test_renderer.py: Rendering pipeline tests
- test_dataset.py: Dataset loading and processing tests
- test_rasterizer.py: Block rasterization tests
- test_volume_renderer.py: Volume rendering tests
- test_block_manager.py: Block management tests
- test_integrations.py: End-to-end integration tests
- test_performance.py: Performance benchmarks
- test_cuda.py: CUDA-specific tests
"""

__version__ = "1.0.0"
__author__ = "NeuroCity Team"

# Test configuration
TEST_CONFIG = {
    "device": "auto",  # auto, cpu, cuda
    "precision": "float32",  # float16, float32, float64
    "batch_size": 4,
    "num_rays": 1024,
    "scene_bounds": (-10, -10, -2, 10, 10, 2),
    "block_size": 5.0,
    "max_blocks": 8,
    "appearance_dim": 32,
    "num_epochs": 2,
    "learning_rate": 5e-4,
}

# Test utilities
import torch
import pytest
import numpy as np
from pathlib import Path


def get_test_device() -> torch.device:
    """Get the best available device for testing."""
    if torch.cuda.is_available() and TEST_CONFIG["device"] != "cpu":
        return torch.device("cuda")
    return torch.device("cpu")


def create_test_data(
    batch_size: int = None, num_rays: int = None, device: torch.device = None
) -> dict[str, torch.Tensor]:
    """Create synthetic test data for Block-NeRF testing."""
    if batch_size is None:
        batch_size = TEST_CONFIG["batch_size"]
    if num_rays is None:
        num_rays = TEST_CONFIG["num_rays"]
    if device is None:
        device = get_test_device()

    return {
        "rays_o": torch.randn(batch_size, num_rays, 3, device=device),
        "rays_d": torch.randn(batch_size, num_rays, 3, device=device),
        "viewdirs": torch.randn(batch_size, num_rays, 3, device=device),
        "camera_ids": torch.randint(0, 10, (batch_size,), device=device),
        "exposure": torch.randn(batch_size, 1, device=device),
        "block_ids": torch.randint(0, 8, (batch_size,), device=device),
        "pixel_coords": torch.randint(0, 800, (batch_size, num_rays, 2), device=device),
        "gt_rgb": torch.rand(batch_size, num_rays, 3, device=device),
        "gt_depth": torch.rand(batch_size, num_rays, 1, device=device),
    }


def create_test_camera() -> dict[str, torch.Tensor]:
    """Create test camera parameters."""
    device = get_test_device()
    return {
        "intrinsics": torch.tensor(
            [[800.0, 0.0, 400.0], [0.0, 800.0, 300.0], [0.0, 0.0, 1.0]], device=device
        ),
        "pose": torch.eye(4, device=device),
        "image_size": (600, 800),
    }


def skip_if_no_cuda():
    """Skip test if CUDA is not available."""
    return pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")


def skip_if_slow():
    """Skip test if slow tests are disabled."""
    return pytest.mark.slow


# Test data paths
TEST_DATA_DIR = Path(__file__).parent / "test_data"
TEST_DATA_DIR.mkdir(exist_ok=True)

# Export test utilities
__all__ = [
    "TEST_CONFIG",
    "get_test_device",
    "create_test_data",
    "create_test_camera",
    "skip_if_no_cuda",
    "skip_if_slow",
    "TEST_DATA_DIR",
]
