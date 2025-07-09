"""
SVRaster Trainer CUDA Implementation

This module provides GPU-accelerated training components for SVRaster,
including volume rendering, loss computation, and optimization kernels.
"""

try:
    from .svraster_gpu import SVRasterGPU, SVRasterGPUTrainer
    from .svraster_optimized_kernels import SVRasterOptimizedKernels

    __all__ = ["SVRasterGPU", "SVRasterGPUTrainer", "SVRasterOptimizedKernels"]
except ImportError as e:
    # CUDA modules not available
    __all__ = []
    print(
        f"Warning: SVRaster Trainer CUDA modules not available. GPU training disabled. Error: {e}"
    )
