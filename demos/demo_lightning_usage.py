"""
Example usage of PyTorch Lightning with SVRaster.

This script demonstrates how to use the Lightning-based training framework
for SVRaster models with various configurations and advanced features.
"""

import sys
import os

# 添加项目根目录到Python路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger, WandbLogger
from pytorch_lightning.profilers import SimpleProfiler

from src.nerfs.svraster.core import SVRasterConfig
from src.nerfs.svraster.dataset import SVRasterDataset, SVRasterDatasetConfig
from src.nerfs.svraster.lightning_trainer import (
    SVRasterLightningConfig, SVRasterLightningModule, create_lightning_trainer, train_svraster_lightning
)


def example_basic_training():
    """Basic example of training SVRaster with Lightning."""
    print("=== Basic Lightning Training Example ===")
    
    # Model configuration
    model_config = SVRasterConfig(
        max_octree_levels=12, base_resolution=32, scene_bounds=(
            -2.0,
            -2.0,
            -2.0,
            2.0,
            2.0,
            2.0,
        )
    )
    
    # Lightning configuration
    lightning_config = SVRasterLightningConfig(
        model_config=model_config, learning_rate=1e-3, optimizer_type="adamw", scheduler_type="cosine", enable_subdivision=True, enable_pruning=True, use_ema=True
    )
    
    # Create datasets (mock data for example)
    dataset_config = SVRasterDatasetConfig(
        data_dir="data/mock_scene", image_height=512, image_width=512, num_rays_train=4096
    )
    
    # In practice, you would load real datasets
    # train_dataset = SVRasterDataset(dataset_config, split='train')
    # val_dataset = SVRasterDataset(dataset_config, split='val')
    
    print("Datasets would be loaded here...")
    print("Model config:", model_config)
    print("Lightning config:", lightning_config)
    
    # Training would be started with:
    # trained_model = train_svraster_lightning(
    #     model_config=model_config, #     lightning_config=lightning_config, #     train_dataset=train_dataset, #     val_dataset=val_dataset, #     max_epochs=100, #     gpus=1
    # )


def example_advanced_training():
    """Advanced example with multiple GPUs, custom callbacks, and logging."""
    print("\n=== Advanced Lightning Training Example ===")
    
    # Model configuration
    model_config = SVRasterConfig(
        max_octree_levels=16, base_resolution=64, scene_bounds=(
            -5.0,
            -5.0,
            -5.0,
            5.0,
            5.0,
            5.0,
        )
    )
    
    # Lightning configuration with advanced features
    lightning_config = SVRasterLightningConfig(
        model_config=model_config, learning_rate=5e-4, weight_decay=1e-5, optimizer_type="adamw", scheduler_type="cosine", scheduler_params={
            "T_max": 200,
            "eta_min": 1e-6,
        }
    )
    
    # Create Lightning module
    lightning_module = SVRasterLightningModule(lightning_config)
    
    # Setup advanced callbacks
    callbacks = [
        # Model checkpointing
        ModelCheckpoint(
            dirpath="checkpoints/advanced_svraster", filename="svraster-{
                epoch:02d,
            }
        ), # Early stopping
        EarlyStopping(
            monitor="val/psnr", mode="max", patience=30, min_delta=0.001, verbose=True
        ), # Learning rate monitoring
        LearningRateMonitor(
            logging_interval="step", log_momentum=True
        )
    ]
    
    # Setup logger (choose one)
    # TensorBoard logger
    tb_logger = TensorBoardLogger(
        save_dir="logs", name="svraster_advanced", version="v1.0"
    )
    
    # W&B logger (alternative)
    # wandb_logger = WandbLogger(
    #     project="neurocity-svraster", #     name="advanced_experiment", #     tags=["svraster", "nerf", "voxels"], #     log_model=True
    # )
    
    # Setup trainer with advanced features
    trainer = pl.Trainer(
        max_epochs=200, devices=1, # Use 1 GPU (adjust based on available hardware)
        accelerator="auto", # Auto-detect available accelerator
        precision="16-mixed", # Mixed precision training
        logger=tb_logger, callbacks=callbacks, gradient_clip_val=lightning_config.gradient_clip_val, gradient_clip_algorithm="norm", log_every_n_steps=25, val_check_interval=0.25, # Validate 4 times per epoch
        limit_val_batches=0.5, # Use 50% of validation data
        profiler=SimpleProfiler(
        )
    )
    
    print("Advanced trainer configured with:")
    print(f"- GPU training on {trainer.num_devices} device(s)")
    print(f"- Mixed precision: {trainer.precision}")
    print(f"- Gradient clipping: {lightning_config.gradient_clip_val}")
    print(f"- EMA updates: {lightning_config.use_ema}")
    print(f"- Adaptive subdivision: {lightning_config.enable_subdivision}")
    print(f"- Voxel pruning: {lightning_config.enable_pruning}")
    
    # In practice, you would start training with:
    # trainer.fit(lightning_module, train_loader, val_loader)


def example_inference_and_evaluation():
    """Example of using trained Lightning model for inference."""
    print("\n=== Inference and Evaluation Example ===")
    
    # Load trained model
    checkpoint_path = "checkpoints/svraster-epoch=99-val_psnr=32.50.ckpt"
    
    # Method 1: Load from checkpoint
    try:
        model = SVRasterLightningModule.load_from_checkpoint(checkpoint_path)
        model.eval()
        print("Model loaded from checkpoint")
    except FileNotFoundError:
        print(f"Checkpoint not found: {checkpoint_path}")
        print("Using mock model for demonstration")
        
        # Create model for demonstration
        model_config = SVRasterConfig()
        lightning_config = SVRasterLightningConfig(model_config=model_config)
        model = SVRasterLightningModule(lightning_config)
        model.eval()
    
    # Method 2: Extract core model for deployment
    core_model = model.model
    
    # Example inference
    with torch.no_grad():
        # Mock ray data
        ray_origins = torch.randn(1000, 3)
        ray_directions = torch.randn(1000, 3)
        ray_directions = ray_directions / ray_directions.norm(dim=1, keepdim=True)
        
        # Inference
        outputs = core_model(ray_origins, ray_directions)
        
        print(f"Inference completed:")
        print(f"- Input rays: {ray_origins.shape[0]}")
        print(f"- Output RGB: {outputs['rgb'].shape}")
        print(f"- Output depth: {outputs['depth'].shape}")
        print(f"- Output alpha: {outputs['alpha'].shape}")
    
    # Get model statistics
    voxel_stats = core_model.get_voxel_statistics()
    print(f"\nModel statistics:")
    for key, value in voxel_stats.items():
        print(f"- {key}: {value}")


def example_custom_callbacks():
    """Example of creating custom callbacks for specialized functionality."""
    print("\n=== Custom Callbacks Example ===")
    
    class VoxelStatisticsCallback(pl.Callback):
        """Custom callback to log voxel statistics during training."""
        
        def on_train_epoch_end(self, trainer, pl_module):
            if trainer.current_epoch % 10 == 0:
                stats = pl_module.model.get_voxel_statistics()
                for key, value in stats.items():
                    trainer.logger.experiment.add_scalar(
                        f"voxel_stats/{key}", value, trainer.current_epoch
                    )
    
    class AdaptiveSubdivisionCallback(pl.Callback):
        """Custom callback for more sophisticated subdivision logic."""
        
        def __init__(self, subdivision_schedule):
            self.subdivision_schedule = subdivision_schedule
        
        def on_train_epoch_end(self, trainer, pl_module):
            epoch = trainer.current_epoch
            if epoch in self.subdivision_schedule:
                # Implement custom subdivision logic
                print(f"Custom subdivision at epoch {epoch}")
                # pl_module._perform_subdivision()
    
    class RenderingCallback(pl.Callback):
        """Custom callback to render test images during training."""
        
        def on_validation_epoch_end(self, trainer, pl_module):
            if trainer.current_epoch % 20 == 0:
                # Render test images
                print(f"Rendering test images at epoch {trainer.current_epoch}")
                # Implement rendering logic
    
    # Example usage of custom callbacks
    custom_callbacks = [
        VoxelStatisticsCallback(
        )
    ]
    
    print("Custom callbacks created:")
    for callback in custom_callbacks:
        print(f"- {callback.__class__.__name__}")


def example_experiment_management():
    """Example of experiment management and hyperparameter tuning."""
    print("\n=== Experiment Management Example ===")
    
    # Define experiment configurations
    experiments = [
        {
            "name": "baseline", "config": SVRasterLightningConfig(
                learning_rate=1e-3, optimizer_type="adam", scheduler_type="cosine", subdivision_threshold=0.01
            )
        }, {
            "name": "high_lr", "config": SVRasterLightningConfig(
                learning_rate=5e-3, optimizer_type="adamw", scheduler_type="cosine", subdivision_threshold=0.01
            )
        }, {
            "name": "aggressive_subdivision", "config": SVRasterLightningConfig(
                learning_rate=1e-3, optimizer_type="adamw", scheduler_type="cosine", subdivision_threshold=0.005, subdivision_start_epoch=5, subdivision_interval=3
            )
        }
    ]
    
    print("Experiment configurations:")
    for exp in experiments:
        print(f"- {exp['name']}: lr={exp['config'].learning_rate}, "
              f"opt={exp['config'].optimizer_type}, "
              f"sub_thresh={exp['config'].subdivision_threshold}")
    
    # In practice, you would run each experiment:
    # for exp in experiments:
    #     lightning_config = exp['config']
    #     lightning_config.model_config = SVRasterConfig()
    #     
    #     trainer = create_lightning_trainer(
    #         lightning_config, #         train_dataset, #         val_dataset, #         experiment_name=exp['name'], #         max_epochs=100
    #     )
    #     
    #     trainer.fit(...)


if __name__ == "__main__":
    # Set random seed for reproducibility
    pl.seed_everything(42)
    
    # Run examples
    example_basic_training()
    example_advanced_training()
    example_inference_and_evaluation()
    example_custom_callbacks()
    example_experiment_management()
    
    print("\n=== Summary ===")
    print("PyTorch Lightning provides the following benefits for this NeRF project:")
    print("1. 🚀 Simplified training loop with automatic optimization")
    print("2. 🔄 Built-in distributed training support")
    print("3. 📊 Automatic logging and metrics tracking")
    print("4. 💾 Checkpoint management and resuming")
    print("5. 🎯 Early stopping and learning rate scheduling")
    print("6. 🔧 Easy hyperparameter tuning and experiment management")
    print("7. 📈 Integration with TensorBoard and W&B")
    print("8. 🚦 Custom callbacks for specialized functionality")
    print("9. ⚡ Mixed precision training for efficiency")
    print("10. 🧪 Built-in profiling and debugging tools") 