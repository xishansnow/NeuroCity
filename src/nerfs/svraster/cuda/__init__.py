"""
SVRaster CUDA/GPU implementation.

This module provides GPU-accelerated implementations of SVRaster components
using CUDA kernels for improved performance.

The module is organized into two main components:
- trainer: GPU-accelerated training components (volume rendering, loss computation)
- renderer: GPU-accelerated rendering components (voxel rasterization, projection)
"""

try:
    # Import trainer components
    from .trainer import SVRasterGPU, SVRasterGPUTrainer
    from .trainer.svraster_optimized_kernels import SVRasterOptimizedKernels

    # Import renderer components
    from .renderer import VoxelRasterizerGPU, benchmark_rasterizer

    # Import shared components
    from .ema import EMAModel

    __all__ = [
        # Trainer components
        "SVRasterGPU",
        "SVRasterGPUTrainer",
        "SVRasterOptimizedKernels",
        # Renderer components
        "VoxelRasterizerGPU",
        "benchmark_rasterizer",
        # Shared components
        "EMAModel",
    ]
except ImportError:
    # CUDA modules not available
    __all__ = []
    print("Warning: SVRaster CUDA modules not available. GPU acceleration disabled.")
