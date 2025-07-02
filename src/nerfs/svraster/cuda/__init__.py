"""
SVRaster CUDA/GPU implementation.

This module provides GPU-accelerated implementations of SVRaster components
using CUDA kernels for improved performance.
"""

try:
    from .svraster_gpu import SVRasterGPU, SVRasterGPUTrainer
    __all__ = ['SVRasterGPU', 'SVRasterGPUTrainer']
except ImportError:
    # CUDA modules not available
    __all__ = []
    print("Warning: SVRaster CUDA modules not available. GPU acceleration disabled.") 