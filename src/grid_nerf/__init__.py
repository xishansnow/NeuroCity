"""
Grid-NeRF: Grid-guided Neural Radiance Fields for Large Urban Scenes

This package implements Grid-guided Neural Radiance Fields, a scalable approach
for rendering large urban environments using hierarchical voxel grids to guide
neural rendering.

Main Components:
- GridNeRF: Main model class
- GridNeRFConfig: Configuration class
- GridNeRFTrainer: Training pipeline
- GridNeRFDataset: Dataset handling
- Utilities for rendering, evaluation, and visualization

Based on the paper: "Grid-guided neural radiance fields for large urban scenes"
"""

from .core import (
    GridNeRF,
    GridNeRFConfig, 
    GridNeRFLoss,
    GridNeRFRenderer,
    GridGuidedMLP,
    HierarchicalGrid
)

from .dataset import (
    GridNeRFDataset,
    KITTI360GridDataset,
    create_dataset,
    create_dataloader
)

from .trainer import (
    GridNeRFTrainer,
    main_worker,
    setup_distributed_training
)

from .utils import (
    # Image metrics
    compute_psnr,
    compute_ssim,
    compute_lpips,
    
    # Visualization and I/O
    save_image,
    load_image,
    create_video_from_images,
    create_comparison_grid,
    
    # Configuration
    load_config,
    save_config,
    setup_logging,
    
    # Learning rate scheduling
    get_learning_rate_scheduler,
    CosineAnnealingWarmRestarts,
    
    # Mathematical utilities
    positional_encoding,
    safe_normalize,
    get_ray_directions,
    sample_along_rays,
    volume_rendering
)

# Version info
__version__ = "1.0.0"
__author__ = "Grid-NeRF Implementation Team"
__description__ = "Grid-guided Neural Radiance Fields for Large Urban Scenes"

# Package metadata
__all__ = [
    # Core classes
    "GridNeRF",
    "GridNeRFConfig",
    "GridNeRFLoss", 
    "GridNeRFRenderer",
    "GridGuidedMLP",
    "HierarchicalGrid",
    
    # Dataset classes
    "GridNeRFDataset",
    "KITTI360GridDataset",
    "create_dataset",
    "create_dataloader",
    
    # Training
    "GridNeRFTrainer",
    "main_worker",
    "setup_distributed_training",
    
    # Utilities - Metrics
    "compute_psnr",
    "compute_ssim", 
    "compute_lpips",
    
    # Utilities - I/O
    "save_image",
    "load_image",
    "create_video_from_images",
    "create_comparison_grid",
    
    # Utilities - Config
    "load_config",
    "save_config",
    "setup_logging",
    
    # Utilities - Scheduling
    "get_learning_rate_scheduler",
    "CosineAnnealingWarmRestarts",
    
    # Utilities - Math
    "positional_encoding",
    "safe_normalize",
    "get_ray_directions",
    "sample_along_rays",
    "volume_rendering"
]

# Default configuration
DEFAULT_CONFIG = {
    # Scene bounds
    "scene_bounds": {
        "min_bound": [-100, -100, -10],
        "max_bound": [100, 100, 50]
    },
    
    # Grid configuration
    "grid_levels": 4,
    "base_resolution": 64,
    "resolution_multiplier": 2,
    "grid_feature_dim": 32,
    
    # Network architecture
    "density_layers": 3,
    "density_hidden_dim": 256,
    "color_layers": 2,
    "color_hidden_dim": 128,
    "position_encoding_levels": 10,
    "direction_encoding_levels": 4,
    
    # Rendering
    "num_samples": 64,
    "num_importance_samples": 128,
    "perturb": True,
    "white_background": False,
    
    # Training
    "batch_size": 1024,
    "num_epochs": 200,
    "grid_lr": 1e-2,
    "mlp_lr": 5e-4,
    "weight_decay": 1e-6,
    "grad_clip_norm": 1.0,
    
    # Loss weights
    "color_weight": 1.0,
    "depth_weight": 0.1,
    "grid_regularization_weight": 1e-4,
    
    # Evaluation
    "eval_batch_size": 256,
    "chunk_size": 1024,
    
    # Logging
    "log_every_n_steps": 100,
    "eval_every_n_epochs": 5,
    "save_every_n_epochs": 10,
    "render_every_n_epochs": 20
}


def get_default_config() -> dict:
    """Get default Grid-NeRF configuration."""
    return DEFAULT_CONFIG.copy()


def create_grid_nerf_model(config: dict = None, device: str = "cuda") -> GridNeRF:
    """
    Create a Grid-NeRF model with default or custom configuration.
    
    Args:
        config: Configuration dictionary (uses defaults if None)
        device: Device to create model on
        
    Returns:
        GridNeRF model instance
    """
    if config is None:
        config = get_default_config()
    
    grid_config = GridNeRFConfig(**config)
    model = GridNeRF(grid_config)
    
    if device:
        model = model.to(device)
    
    return model


def quick_setup(
    data_path: str,
    output_dir: str,
    config: dict = None,
    device: str = "cuda"
) -> tuple:
    """
    Quick setup for Grid-NeRF training.
    
    Args:
        data_path: Path to training data
        output_dir: Output directory for results
        config: Configuration dictionary
        device: Device to use
        
    Returns:
        Tuple of (model, trainer, dataset)
    """
    import torch
    
    # Setup configuration
    if config is None:
        config = get_default_config()
    
    grid_config = GridNeRFConfig(**config)
    
    # Create model
    model = create_grid_nerf_model(config, device)
    
    # Create dataset
    dataset = create_dataset(
        data_path=data_path,
        split='train',
        config=grid_config
    )
    
    # Create trainer
    trainer = GridNeRFTrainer(
        config=grid_config,
        output_dir=output_dir,
        device=torch.device(device)
    )
    
    return model, trainer, dataset


# Import checks
def check_dependencies():
    """Check if all required dependencies are available."""
    missing_deps = []
    
    try:
        import torch
        if not torch.cuda.is_available():
            print("Warning: CUDA not available, will use CPU")
    except ImportError:
        missing_deps.append("torch")
    
    try:
        import numpy
    except ImportError:
        missing_deps.append("numpy")
    
    try:
        import cv2
    except ImportError:
        missing_deps.append("opencv-python")
    
    try:
        import matplotlib
    except ImportError:
        missing_deps.append("matplotlib")
    
    try:
        import yaml
    except ImportError:
        missing_deps.append("pyyaml")
    
    if missing_deps:
        print(f"Warning: Missing dependencies: {', '.join(missing_deps)}")
        print("Please install with: pip install " + " ".join(missing_deps))
    
    return len(missing_deps) == 0


# Run dependency check on import
check_dependencies() 