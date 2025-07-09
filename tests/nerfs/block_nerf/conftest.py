"""
pytest configuration for Block-NeRF tests.
"""

import pytest
import torch


def pytest_addoption(parser):
    """Add custom pytest options."""
    parser.addoption(
        "--runslow", 
        action="store_true", 
        default=False, 
        help="run slow tests"
    )
    parser.addoption(
        "--runcuda", 
        action="store_true", 
        default=False, 
        help="run CUDA tests"
    )


def pytest_collection_modifyitems(config, items):
    """Modify test collection based on options."""
    if config.getoption("--runslow"):
        # Don't skip slow tests
        return
    
    skip_slow = pytest.mark.skip(reason="need --runslow option to run")
    for item in items:
        if "slow" in item.keywords:
            item.add_marker(skip_slow)
    
    if not config.getoption("--runcuda"):
        skip_cuda = pytest.mark.skip(reason="need --runcuda option to run")
        for item in items:
            if "cuda" in item.keywords:
                item.add_marker(skip_cuda)


@pytest.fixture(scope="session", autouse=True)
def setup_torch():
    """Setup torch for testing."""
    # Set deterministic behavior for reproducible tests
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)
        torch.cuda.manual_seed_all(42)
    
    # Set number of threads for CPU testing
    torch.set_num_threads(1)
    
    # Disable gradient computation by default (enable in specific tests)
    torch.set_grad_enabled(True)


@pytest.fixture(scope="session")
def cuda_device():
    """Get CUDA device if available."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    else:
        pytest.skip("CUDA not available")


@pytest.fixture(scope="session")
def cpu_device():
    """Get CPU device."""
    return torch.device("cpu")


# Custom markers
pytest_markers = [
    "slow: marks tests as slow (deselect with '-m \"not slow\"')",
    "cuda: marks tests as requiring CUDA (deselect with '-m \"not cuda\"')",
    "integration: marks tests as integration tests",
    "unit: marks tests as unit tests",
]

for marker in pytest_markers:
    pytest.mark.__dict__.setdefault(marker.split(":")[0], marker)
