"""
SVRaster Renderer CUDA Implementation

This module provides GPU-accelerated rendering components for SVRaster,
including voxel rasterization, projection, and visualization kernels.
"""

try:
    from .voxel_rasterizer_gpu import VoxelRasterizerGPU, benchmark_rasterizer

    __all__ = ["VoxelRasterizerGPU", "benchmark_rasterizer"]
except ImportError:
    # CUDA modules not available
    __all__ = []
    print("Warning: SVRaster Renderer CUDA modules not available. GPU rendering disabled.")
