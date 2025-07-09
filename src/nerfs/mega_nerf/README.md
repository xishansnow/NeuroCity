# Mega-NeRF

A scalable, modular implementation of Mega-NeRF for large-scale neural scene representation and rendering.

## Overview

Mega-NeRF is designed for efficient, distributed neural radiance field (NeRF) training and rendering on large scenes. This implementation is modular, extensible, and supports both research and production use cases.

## Features
- **Modular Design**: Decoupled core, trainer, renderer, dataset, and utilities for easy extension.
- **Scalable Partitioning**: Supports spatial partitioning and distributed training.
- **Efficient Rendering**: Fast batch and video rendering with chunked processing.
- **Flexible Dataset Support**: Compatible with NeRF-style datasets and custom data.
- **Mixed Precision & Multi-GPU**: Supports AMP and distributed training.
- **Comprehensive Testing**: Full test suite for all modules.

## Directory Structure

```
src/nerfs/mega_nerf/
├── __init__.py           # Main API and exports
├── core.py               # Core model and config
├── trainer.py            # Training logic and config
├── renderer.py           # Rendering logic and config
├── dataset.py            # Dataset loading and camera handling
├── utils/                # Utility functions (partitioning, metrics, etc.)
```

## Quick Start

### Installation
```bash
# Install dependencies
conda activate neurocity
pip install -r requirements.txt
```

### Training
```python
from src.nerfs.mega_nerf import MegaNeRF, MegaNeRFConfig, MegaNeRFTrainer, MegaNeRFTrainerConfig

# Configure model and trainer
model_config = MegaNeRFConfig()
trainer_config = MegaNeRFTrainerConfig()

# Initialize model and trainer
model = MegaNeRF(model_config)
trainer = MegaNeRFTrainer(model, trainer_config)

# Start training
trainer.train()
```

### Rendering
```python
from src.nerfs.mega_nerf import MegaNeRFRenderer, MegaNeRFRendererConfig

renderer_config = MegaNeRFRendererConfig()
renderer = MegaNeRFRenderer(model, renderer_config)

# Render an image
image = renderer.render_image(camera_pose, intrinsics)
```

### Testing
```bash
# Run all tests
python tests/nerfs/mega_nerf/run_tests.py
```

## Environment
- Python 3.10+
- PyTorch 2.0+
- NumPy 1.20+
- pytest 7.0+
- CUDA (optional, for GPU)

## Contributing
- Follow modular structure and type annotation conventions.
- Add/Update tests for new features.
- Document all public APIs.
- See `tests/nerfs/mega_nerf/README.md` for test guidelines.

## Citation
If you use this codebase, please cite the original Mega-NeRF paper and this repository.

```
@article{turki2022megenerf,
  title={Mega-NeRF: Scalable Construction of Large-Scale NeRFs for Virtual Fly-Throughs},
  author={Turki, H and others},
  journal={arXiv preprint arXiv:2112.10703},
  year={2022}
}
```

## License
This project is part of the NeuroCity suite and is licensed under the same terms as the main repository. 