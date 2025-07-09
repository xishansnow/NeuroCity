"""
SVRaster Test Suite

This module contains comprehensive tests for the SVRaster package, covering:
- Core components (model, config, loss)
- Training pipeline (trainer, volume renderer)
- Inference pipeline (renderer, rasterizer)
- Dataset utilities
- Utility functions (morton coding, octree operations, spherical harmonics)
- CUDA acceleration (if available)
- Integration tests

Usage:
    python -m pytest tests/nerfs/svraster/
    python tests/nerfs/svraster/run_svraster_tests.py
"""

__version__ = "1.0.0"
