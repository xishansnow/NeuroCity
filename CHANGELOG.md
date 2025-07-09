# Changelog

All notable changes to the Plenoxels package will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [2.0.0] - 2024-07-07

### Added
- Complete refactored Plenoxels implementation with modern architecture
- New PlenoxelTrainer class with advanced training features
- New PlenoxelRenderer class for optimized inference
- CUDA accelerated volume rendering kernels
- Spherical harmonics support for view-dependent effects
- Coarse-to-fine training strategy
- Professional configuration management system
- Comprehensive examples and documentation
- Integration with NeuroCity neural rendering toolkit

### Features
- **Sparse Voxel Grid**: Efficient memory usage with sparse data structures
- **CUDA Acceleration**: Custom CUDA kernels for volume rendering and interpolation
- **Spherical Harmonics**: Advanced view-dependent appearance modeling
- **Flexible Training**: Configurable training loops with modern PyTorch patterns
- **High-Quality Rendering**: Optimized inference pipeline for production use
- **Easy Integration**: Clean API design following modern Python practices

### Technical Specifications
- Python 3.10+ support
- PyTorch 2.0+ compatibility
- CUDA 12.1+ support
- Memory-efficient sparse voxel grids
- Real-time rendering capabilities
- Professional documentation and examples

### Dependencies
- torch>=2.1.0
- numpy>=1.26.4
- opencv-python>=4.8.1
- scipy>=1.11.3
- CUDA runtime libraries

## [1.0.0] - 2024-01-01

### Added
- Initial release of Plenoxels implementation
- Basic volume rendering functionality
- Training and inference capabilities
