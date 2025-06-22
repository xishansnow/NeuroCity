"""
Comprehensive example demonstrating PyTorch Lightning usage with multiple NeRF models.

This script shows how to train different NeRF variants using PyTorch Lightning,
including SVRaster, Grid-NeRF, Instant-NGP, and MIP-NeRF.
"""

import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger, WandbLogger
from pathlib import Path
import argparse
import yaml
from typing import Dict, Any, Optional

# Import different NeRF models and their Lightning trainers
from src.svraster.core import SVRasterConfig
from src.svraster.lightning_trainer import (
    SVRasterLightningConfig, 
    SVRasterLightningModule,
    train_svraster_lightning
)

from src.grid_nerf.core import GridNeRFConfig
from src.grid_nerf.lightning_trainer import (
    GridNeRFLightningConfig,
    GridNeRFLightningModule,
    train_grid_nerf_lightning
)

from src.instant_ngp.core import InstantNGPConfig
from src.instant_ngp.lightning_trainer import (
    InstantNGPLightningConfig,
    InstantNGPLightningModule,
    train_instant_ngp_lightning
)

from src.mip_nerf.core import MipNeRFConfig
from src.mip_nerf.lightning_trainer import (
    MipNeRFLightningConfig,
    MipNeRFLightningModule,
    train_mip_nerf_lightning
)


def create_mock_dataset(model_type: str, size: int = 1000):
    """Create a mock dataset for testing different models."""
    class MockDataset(torch.utils.data.Dataset):
        def __init__(self, size: int):
            self.size = size
            
        def __len__(self):
            return self.size
            
        def __getitem__(self, idx):
            # Generate random rays and colors
            rays_o = torch.randn(3)
            rays_d = torch.randn(3)
            rays_d = rays_d / torch.norm(rays_d)  # Normalize direction
            colors = torch.rand(3)
            
            data = {
                'rays_o': rays_o,
                'rays_d': rays_d,
                'colors': colors,
                'near': 0.1,
                'far': 100.0,
            }
            
            # Add model-specific data
            if model_type == "grid_nerf":
                data['image'] = torch.rand(64, 64, 3)  # Mock image
            elif model_type == "mip_nerf":
                data['viewdirs'] = rays_d
                data['image_shape'] = (64, 64)
            elif model_type == "instant_ngp":
                data['image_shape'] = (64, 64)
                
            return data
    
    return MockDataset(size)


def train_svraster_example():
    """Example of training SVRaster with Lightning."""
    print("🚀 Training SVRaster with PyTorch Lightning")
    
    # Create configurations
    model_config = SVRasterConfig(
        max_octree_levels=12,
        base_resolution=32,
        ray_samples_per_voxel=4
    )
    
    lightning_config = SVRasterLightningConfig(
        model_config=model_config,
        learning_rate=5e-4,
        ray_batch_size=2048,
        use_ema=True,
        enable_subdivision=True,
        enable_pruning=True
    )
    
    # Create datasets
    train_dataset = create_mock_dataset("svraster", 800)
    val_dataset = create_mock_dataset("svraster", 200)
    
    # Train model
    trained_model = train_svraster_lightning(
        model_config=model_config,
        lightning_config=lightning_config,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        max_epochs=10,
        gpus=1,
        logger_type="tensorboard",
        experiment_name="svraster_example"
    )
    
    print(f"✅ SVRaster training completed. Final validation PSNR: {trained_model.val_psnr.compute():.2f}")
    return trained_model


def train_grid_nerf_example():
    """Example of training Grid-NeRF with Lightning."""
    print("🏙️ Training Grid-NeRF with PyTorch Lightning")
    
    # Create configurations
    model_config = GridNeRFConfig(
        base_grid_resolution=32,
        max_grid_resolution=256,
        num_grid_levels=3,
        grid_feature_dim=16
    )
    
    lightning_config = GridNeRFLightningConfig(
        model_config=model_config,
        learning_rate=5e-4,
        grid_lr=1e-3,
        ray_batch_size=2048,
        enable_grid_pruning=True
    )
    
    # Create datasets
    train_dataset = create_mock_dataset("grid_nerf", 800)
    val_dataset = create_mock_dataset("grid_nerf", 200)
    
    # Train model
    trained_model = train_grid_nerf_lightning(
        model_config=model_config,
        lightning_config=lightning_config,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        max_epochs=10,
        gpus=1,
        logger_type="tensorboard",
        experiment_name="grid_nerf_example"
    )
    
    print(f"✅ Grid-NeRF training completed. Final validation PSNR: {trained_model.val_psnr.compute():.2f}")
    return trained_model


def train_instant_ngp_example():
    """Example of training Instant-NGP with Lightning."""
    print("⚡ Training Instant-NGP with PyTorch Lightning")
    
    # Create configurations
    model_config = InstantNGPConfig(
        num_levels=12,
        level_dim=2,
        base_resolution=16,
        log2_hashmap_size=18,
        hidden_dim=64
    )
    
    lightning_config = InstantNGPLightningConfig(
        model_config=model_config,
        learning_rate=1e-2,
        hash_lr_factor=1.0,
        network_lr_factor=0.1,
        ray_batch_size=2048,
        use_mixed_precision=True
    )
    
    # Create datasets
    train_dataset = create_mock_dataset("instant_ngp", 800)
    val_dataset = create_mock_dataset("instant_ngp", 200)
    
    # Train model
    trained_model = train_instant_ngp_lightning(
        model_config=model_config,
        lightning_config=lightning_config,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        max_epochs=10,
        gpus=1,
        logger_type="tensorboard",
        experiment_name="instant_ngp_example"
    )
    
    print(f"✅ Instant-NGP training completed. Final validation PSNR: {trained_model.val_psnr.compute():.2f}")
    return trained_model


def train_mip_nerf_example():
    """Example of training MIP-NeRF with Lightning."""
    print("🎯 Training MIP-NeRF with PyTorch Lightning")
    
    # Create configurations
    model_config = MipNeRFConfig(
        netdepth=6,
        netwidth=128,
        num_samples=32,
        num_importance=64,
        multires=8,
        use_viewdirs=True
    )
    
    lightning_config = MipNeRFLightningConfig(
        model_config=model_config,
        learning_rate=5e-4,
        final_learning_rate=5e-6,
        ray_batch_size=1024,
        use_hierarchical_sampling=True,
        pixel_radius=1.0
    )
    
    # Create datasets
    train_dataset = create_mock_dataset("mip_nerf", 800)
    val_dataset = create_mock_dataset("mip_nerf", 200)
    
    # Train model
    trained_model = train_mip_nerf_lightning(
        model_config=model_config,
        lightning_config=lightning_config,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        max_epochs=10,
        gpus=1,
        logger_type="tensorboard",
        experiment_name="mip_nerf_example"
    )
    
    print(f"✅ MIP-NeRF training completed. Final validation PSNR: {trained_model.val_psnr.compute():.2f}")
    return trained_model


def compare_models():
    """Compare different NeRF models using Lightning."""
    print("🔍 Comparing different NeRF models with PyTorch Lightning")
    
    results = {}
    
    # Train all models
    models = {
        'SVRaster': train_svraster_example,
        'Grid-NeRF': train_grid_nerf_example,
        'Instant-NGP': train_instant_ngp_example,
        'MIP-NeRF': train_mip_nerf_example
    }
    
    for model_name, train_func in models.items():
        try:
            print(f"\n{'='*50}")
            print(f"Training {model_name}")
            print(f"{'='*50}")
            
            trained_model = train_func()
            results[model_name] = {
                'model': trained_model,
                'final_psnr': trained_model.val_psnr.compute().item()
            }
            
        except Exception as e:
            print(f"❌ Error training {model_name}: {e}")
            results[model_name] = {'error': str(e)}
    
    # Print comparison results
    print(f"\n{'='*50}")
    print("📊 TRAINING RESULTS COMPARISON")
    print(f"{'='*50}")
    
    for model_name, result in results.items():
        if 'error' in result:
            print(f"{model_name:15}: ❌ Error - {result['error']}")
        else:
            print(f"{model_name:15}: ✅ PSNR = {result['final_psnr']:.2f} dB")
    
    return results


def advanced_lightning_features_demo():
    """Demonstrate advanced PyTorch Lightning features."""
    print("🚀 Demonstrating Advanced PyTorch Lightning Features")
    
    # Create a more advanced trainer configuration
    model_config = SVRasterConfig()
    lightning_config = SVRasterLightningConfig(
        model_config=model_config,
        learning_rate=5e-4,
        use_ema=True,
        enable_subdivision=True
    )
    
    # Create Lightning module
    lightning_module = SVRasterLightningModule(lightning_config)
    
    # Advanced logger with multiple backends
    loggers = [
        TensorBoardLogger(
            save_dir="logs",
            name="advanced_demo",
            version="tensorboard"
        )
    ]
    
    # Advanced callbacks
    callbacks = [
        ModelCheckpoint(
            dirpath="checkpoints/advanced_demo",
            filename="best-{epoch:02d}-{val/psnr:.2f}",
            monitor="val/psnr",
            mode="max",
            save_top_k=3,
            save_last=True
        ),
        EarlyStopping(
            monitor="val/psnr",
            mode="max",
            patience=20,
            verbose=True
        ),
        LearningRateMonitor(logging_interval="step"),
    ]
    
    # Advanced trainer with multiple features
    trainer = pl.Trainer(
        max_epochs=20,
        devices=1,
        accelerator="auto",
        logger=loggers,
        callbacks=callbacks,
        precision="16-mixed",
        gradient_clip_val=1.0,
        gradient_clip_algorithm="norm",
        log_every_n_steps=10,
        val_check_interval=0.5,
        enable_checkpointing=True,
        enable_progress_bar=True,
        enable_model_summary=True,
        detect_anomaly=False,  # Set to True for debugging
        profiler="simple",  # Can use "advanced" or "pytorch" for detailed profiling
    )
    
    # Create datasets
    train_dataset = create_mock_dataset("svraster", 500)
    val_dataset = create_mock_dataset("svraster", 100)
    
    # Create data loaders
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=32, shuffle=True, num_workers=2
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=32, shuffle=False, num_workers=2
    )
    
    print("Starting advanced training with the following features:")
    print("- ✅ Mixed precision training (16-bit)")
    print("- ✅ Gradient clipping")
    print("- ✅ Multiple checkpoints")
    print("- ✅ Early stopping")
    print("- ✅ Learning rate monitoring")
    print("- ✅ TensorBoard logging")
    print("- ✅ Model profiling")
    print("- ✅ EMA model updates")
    print("- ✅ Adaptive voxel subdivision")
    
    # Start training
    trainer.fit(lightning_module, train_loader, val_loader)
    
    print("✅ Advanced training completed!")
    return lightning_module, trainer


def main():
    """Main function to run different examples."""
    parser = argparse.ArgumentParser(description="PyTorch Lightning NeRF Examples")
    parser.add_argument("--mode", type=str, default="compare",
                       choices=["svraster", "grid_nerf", "instant_ngp", "mip_nerf", 
                               "compare", "advanced"],
                       help="Training mode")
    parser.add_argument("--gpus", type=int, default=1, help="Number of GPUs to use")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs")
    
    args = parser.parse_args()
    
    print("🌟 PyTorch Lightning NeRF Training Examples 🌟")
    print(f"Mode: {args.mode}")
    print(f"GPUs: {args.gpus}")
    print(f"Epochs: {args.epochs}")
    print()
    
    if args.mode == "svraster":
        train_svraster_example()
    elif args.mode == "grid_nerf":
        train_grid_nerf_example()
    elif args.mode == "instant_ngp":
        train_instant_ngp_example()
    elif args.mode == "mip_nerf":
        train_mip_nerf_example()
    elif args.mode == "compare":
        compare_models()
    elif args.mode == "advanced":
        advanced_lightning_features_demo()
    
    print("\n🎉 All examples completed successfully!")


if __name__ == "__main__":
    main() 