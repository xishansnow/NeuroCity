"""
CNC-NeRF: Context-based NeRF Compression

This module implements the Context-based NeRF Compression (CNC) framework as described in:
"How Far Can We Compress Instant-NGP-Based NeRF?" by Yihang Chen et al.

Key Features:
- Level-wise and dimension-wise context models
- Hash collision fusion with occupancy grid
- Entropy-based compression with arithmetic coding
- 100x+ compression ratio over baseline Instant-NGP
- Storage-friendly NeRF representation with O(log n) space complexity

The CNC framework leverages highly efficient context models to compress multi-resolution 
hash embeddings while maintaining high fidelity and rendering speed.
"""

from .core import (
    CNCNeRF, CNCNeRFConfig, HashEmbeddingEncoder, ContextModel, LevelWiseContextModel, DimensionWiseContextModel, EntropyEstimator, ArithmeticCoder, OccupancyGrid, CNCRenderer
)

from .dataset import (
    CNCNeRFDataset, CNCNeRFDatasetConfig, create_synthetic_dataset
)

from .trainer import (
    CNCNeRFTrainer, CNCNeRFTrainerConfig, create_cnc_nerf_trainer
)

from .example_usage import (
    basic_usage_example, training_example, compression_analysis_example, rendering_speed_benchmark
)

__all__ = [
    # Core components
    "CNCNeRF", "CNCNeRFConfig", "HashEmbeddingEncoder", "ContextModel", "LevelWiseContextModel", "DimensionWiseContextModel", "EntropyEstimator", "ArithmeticCoder", "OccupancyGrid", "CNCRenderer", # Dataset
    "CNCNeRFDataset", "CNCNeRFDatasetConfig", "create_synthetic_dataset", # Training
    "CNCNeRFTrainer", "CNCNeRFTrainerConfig", "create_cnc_nerf_trainer", # Examples and demos
    "basic_usage_example", "training_example", "compression_analysis_example", "rendering_speed_benchmark", "main"
] 