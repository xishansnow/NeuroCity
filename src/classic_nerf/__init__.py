"""
Classic NeRF: Neural Radiance Fields for View Synthesis

This package implements the original NeRF model from:
"NeRF: Representing Scenes as Neural Radiance Fields for View Synthesis"
by Ben Mildenhall et al. (ECCV 2020)

The implementation includes:
- Positional encoding for high-frequency details
- Multi-layer perceptron (MLP) for scene representation
- Hierarchical volume sampling with coarse and fine networks
- Volume rendering with neural radiance fields
- Support for Blender synthetic scenes
- Training and evaluation utilities

Example usage:
    ```python
    from classic_nerf import NeRFConfig, NeRF, NeRFTrainer
    from classic_nerf.dataset import create_nerf_dataloader
    
    # Create configuration
    config = NeRFConfig(
        netdepth=8,
        netwidth=256,
        N_samples=64,
        N_importance=128
    )
    
    # Create data loader
    train_loader = create_nerf_dataloader(
        'blender', 
        'path/to/dataset', 
        split='train'
    )
    
    # Create trainer and train
    trainer = NeRFTrainer(config)
    trainer.train(train_loader, num_epochs=100)
    ```
"""

from .core import (
    NeRFConfig,
    NeRF, 
    Embedder,
    NeRFRenderer,
    NeRFLoss,
    raw2outputs,
    sample_pdf
)

from .dataset import (
    BlenderDataset,
    create_nerf_dataloader,
    get_rays_np,
    pose_spherical
)

from .trainer import (
    NeRFTrainer
)

from .utils import (
    to8b,
    img2mse, 
    mse2psnr,
    get_rays,
    render_path,
    create_spherical_poses,
    visualize_depth,
    compute_ssim,
    save_image_grid,
    load_config_from_args
)

__version__ = "1.0.0"
__author__ = "NeuroCity Team"
__email__ = "team@neurocity.ai"

__all__ = [
    # Core components
    'NeRFConfig',
    'NeRF',
    'Embedder', 
    'NeRFRenderer',
    'NeRFLoss',
    'raw2outputs',
    'sample_pdf',
    
    # Dataset
    'BlenderDataset',
    'create_nerf_dataloader',
    'get_rays_np',
    'pose_spherical',
    
    # Training
    'NeRFTrainer',
    
    # Utils
    'to8b',
    'img2mse',
    'mse2psnr', 
    'get_rays',
    'render_path',
    'create_spherical_poses',
    'visualize_depth',
    'compute_ssim',
    'save_image_grid',
    'load_config_from_args'
]
