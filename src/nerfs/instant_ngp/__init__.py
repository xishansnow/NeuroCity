"""
Instant NGP: Instant Neural Graphics Primitives

This package implements the Instant NGP model from:
"Instant Neural Graphics Primitives with a Multiresolution Hash Encoding"
by Thomas Müller et al. (SIGGRAPH 2022)

The implementation includes:
- Multiresolution hash encoding for efficient feature lookup
- Small MLP networks for fast inference
- Spherical harmonics encoding for view directions
- Volume rendering for NeRF-style applications
- CUDA-optimized operations (with PyTorch fallback)

Key advantages over classic NeRF:
- 10-100x faster training and inference
- Compact hash-based scene representation
- Real-time rendering capabilities
- Reduced memory footprint

Example usage:
    ```python
    from .core import InstantNGPConfig, InstantNGPModel, InstantNGPTrainer

    # Create configuration
    config = InstantNGPConfig(
        num_levels=16, level_dim=2, base_resolution=16, desired_resolution=2048
    )

    # Create model
    model = InstantNGPModel(config)

    # Train model
    trainer = InstantNGPTrainer(config)
    trainer.train(train_loader, num_epochs=20)  # Much faster than classic NeRF!
    ```
"""

from .core import (
    InstantNGPConfig,
    InstantNGPModel,
    HashEncoder,
    SHEncoder,
    InstantNGPLoss,
    InstantNGPRenderer,
)

# Add alias for backward compatibility
InstantNGP = InstantNGPModel

from .trainer import InstantNGPTrainer

from .dataset import InstantNGPDataset, create_instant_ngp_dataloader

from .utils import (
    contract_to_unisphere,
    uncontract_from_unisphere,
    morton_encode_3d,
    compute_tv_loss,
    adaptive_sampling,
    estimate_normals,
    compute_hash_grid_size,
)

__version__ = "1.0.0"
__author__ = "NeuroCity Team"
__email__ = "team@neurocity.ai"

__all__ = [
    # Core components
    "InstantNGPConfig",
    "InstantNGPModel",
    "InstantNGP",  # Alias for backward compatibility
    "HashEncoder",
    "SHEncoder",
    "InstantNGPLoss",
    "InstantNGPRenderer",  # Training
    "InstantNGPTrainer",  # Dataset
    "InstantNGPDataset",
    "create_instant_ngp_dataloader",  # Utils
    "contract_to_unisphere",
    "uncontract_from_unisphere",
    "morton_encode_3d",
    "compute_tv_loss",
    "adaptive_sampling",
    "estimate_normals",
    "compute_hash_grid_size",
]
