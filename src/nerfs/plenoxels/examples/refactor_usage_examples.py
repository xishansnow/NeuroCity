"""
Plenoxels Refactored Usage Examples

This file demonstrates how to use the refactored Plenoxels package with
separate training and inference classes.
"""

import torch
import numpy as np
from pathlib import Path

# Import the new refactored components
from nerfs.plenoxels import (
    PlenoxelTrainer,
    PlenoxelRenderer,
    PlenoxelTrainingConfig,
    PlenoxelInferenceConfig,
    ExampleConfigs,
    quick_train,
    quick_render,
)

# Import dataset utilities
from nerfs.plenoxels.dataset import PlenoxelDataset, PlenoxelDatasetConfig


def example_training():
    """Example of training a Plenoxel model using the new interface."""
    print("=== Plenoxels Training Example ===")
    
    # 1. Create training configuration
    config = PlenoxelTrainingConfig(
        # Model parameters
        grid_resolution=(256, 256, 256),
        sh_degree=2,
        
        # Training parameters
        num_epochs=10000,
        learning_rate=0.1,
        batch_size=4096,
        
        # Coarse-to-fine training
        use_coarse_to_fine=True,
        coarse_resolutions=[(128, 128, 128), (256, 256, 256)],
        coarse_epochs=[3000, 10000],
        
        # Regularization
        tv_lambda=1e-6,
        l1_lambda=1e-8,
        pruning_interval=1000,
        python3 run_svraster_tests.py

        # Logging
        experiment_name="plenoxel_example",
        use_tensorboard=True,
        log_interval=100,
        save_interval=2000,
    )
    
    # 2. Setup dataset (example)
    dataset_config = PlenoxelDatasetConfig(
        data_root="data/nerf_synthetic/lego",
        scene_bounds=torch.tensor([-1.5, -1.5, -1.5, 1.5, 1.5, 1.5]),
        white_background=True,
    )
    
    train_dataset = PlenoxelDataset(dataset_config, split="train")
    val_dataset = PlenoxelDataset(dataset_config, split="val")
    
    print(f"Training dataset: {len(train_dataset)} images")
    print(f"Validation dataset: {len(val_dataset)} images")
    
    # 3. Create trainer and train
    trainer = PlenoxelTrainer(config, train_dataset, val_dataset)
    
    print("Starting training...")
    renderer = trainer.train()  # Returns a trained renderer
    
    print("Training completed!")
    print(f"Best PSNR: {trainer.best_psnr:.2f}")
    
    return renderer


def example_inference(renderer=None, checkpoint_path=None):
    """Example of inference using the trained model."""
    print("\n=== Plenoxels Inference Example ===")
    
    # Load renderer from checkpoint if not provided
    if renderer is None:
        if checkpoint_path is None:
            checkpoint_path = "checkpoints/plenoxel_example/best.pth"
        
        print(f"Loading renderer from: {checkpoint_path}")
        
        # Option 1: Load with default inference config
        renderer = PlenoxelRenderer.from_checkpoint(checkpoint_path)
        
        # Option 2: Load with custom inference config
        # inference_config = PlenoxelInferenceConfig(
        #     high_quality=True,
        #     render_depth=True,
        #     render_normals=True,
        # )
        # renderer = PlenoxelRenderer.from_checkpoint(checkpoint_path, inference_config)
    
    # Print model info
    model_info = renderer.get_model_info()
    print(f"Model resolution: {model_info['resolution']}")
    print(f"Occupancy: {model_info['occupancy_stats']['sparsity']:.1%}")
    
    # 1. Render single image
    print("\nRendering single image...")
    
    # Example camera parameters (you would get these from your dataset)
    camera_matrix = torch.tensor([
        [512.0, 0.0, 256.0],
        [0.0, 512.0, 256.0], 
        [0.0, 0.0, 1.0]
    ])
    
    camera_pose = torch.eye(4)  # Identity pose for example
    camera_pose[2, 3] = -3.0   # Move camera back
    
    outputs = renderer.render_image(
        height=512,
        width=512,
        camera_matrix=camera_matrix,
        camera_pose=camera_pose,
        return_depth=True,
    )
    
    rgb_image = outputs["rgb"]
    depth_map = outputs["depth"]
    
    print(f"Rendered RGB image: {rgb_image.shape}")
    print(f"Rendered depth map: {depth_map.shape}")
    
    # 2. Render batch of rays
    print("\nRendering batch of rays...")
    
    # Generate some example rays
    num_rays = 1000
    rays_o = torch.randn(num_rays, 3) * 0.1  # Random origins near origin
    rays_d = torch.randn(num_rays, 3)
    rays_d = torch.nn.functional.normalize(rays_d, dim=-1)  # Normalize directions
    
    ray_outputs = renderer.render_rays(rays_o, rays_d, return_depth=True)
    
    print(f"Rendered {num_rays} rays")
    print(f"RGB colors: {ray_outputs['rgb'].shape}")
    print(f"Depths: {ray_outputs['depth'].shape}")
    
    # 3. Render video (example trajectory)
    print("\nRendering video...")
    
    # Create circular camera trajectory
    num_frames = 30
    radius = 3.0
    
    camera_trajectory = []
    for i in range(num_frames):
        angle = 2 * np.pi * i / num_frames
        
        # Camera pose (circular trajectory)
        pose = torch.eye(4)
        pose[0, 3] = radius * np.cos(angle)  # X position
        pose[2, 3] = radius * np.sin(angle)  # Z position
        pose[1, 3] = 0.5  # Slight elevation
        
        # Look at origin
        # (In practice, you'd compute proper look-at matrices)
        
        camera_trajectory.append((camera_matrix, pose))
    
    frames = renderer.render_video(
        camera_trajectory,
        height=256,
        width=256,
        output_path="outputs/plenoxel_video.mp4",
    )
    
    print(f"Rendered {len(frames)} video frames")
    
    # 4. Save model for later use
    print("\nSaving model...")
    renderer.save_model("outputs/plenoxel_model.pth")
    print("Model saved!")
    
    return outputs


def example_quick_interface():
    """Example using the quick interface functions."""
    print("\n=== Quick Interface Example ===")
    
    # Quick training (with sensible defaults)
    dataset_config = PlenoxelDatasetConfig(
        data_root="data/nerf_synthetic/lego",
    )
    train_dataset = PlenoxelDataset(dataset_config, split="train")
    val_dataset = PlenoxelDataset(dataset_config, split="val")
    
    print("Quick training...")
    renderer = quick_train(
        train_dataset,
        val_dataset,
        num_epochs=1000,  # Fast training for demo
        grid_resolution=(128, 128, 128),
    )
    
    # Quick rendering
    print("Quick rendering...")
    camera_matrix = torch.tensor([
        [256.0, 0.0, 128.0],
        [0.0, 256.0, 128.0],
        [0.0, 0.0, 1.0]
    ])
    camera_pose = torch.eye(4)
    camera_pose[2, 3] = -2.0
    
    outputs = quick_render(
        "checkpoints/quick_train/final.pth",  # Would be created by quick_train
        height=256,
        width=256,
        camera_matrix=camera_matrix,
        camera_pose=camera_pose,
    )
    
    print(f"Quick render output: {outputs['rgb'].shape}")


def example_preset_configs():
    """Example using preset configurations."""
    print("\n=== Preset Configurations Example ===")
    
    # Fast training configuration
    fast_config = ExampleConfigs.fast_training()
    print(f"Fast training config: {fast_config.grid_resolution}, {fast_config.num_epochs} epochs")
    
    # High quality training configuration  
    hq_config = ExampleConfigs.high_quality_training()
    print(f"HQ training config: {hq_config.grid_resolution}, {hq_config.num_epochs} epochs")
    
    # Fast inference configuration
    fast_inference = ExampleConfigs.fast_inference()
    print(f"Fast inference: samples={fast_inference.num_samples}, half_precision={fast_inference.use_half_precision}")
    
    # High quality inference configuration
    hq_inference = ExampleConfigs.high_quality_inference()
    print(f"HQ inference: max_samples={hq_inference.max_samples_per_ray}, render_depth={hq_inference.render_depth}")


def compare_with_svraster():
    """Compare the new Plenoxels interface with SVRaster interface."""
    print("\n=== Interface Comparison ===")
    
    print("SVRaster style (for comparison):")
    print("""
    # SVRaster Training
    from nerfs.svraster import SVRasterTrainer, SVRasterConfig
    config = SVRasterConfig(...)
    trainer = SVRasterTrainer(config, dataset)
    renderer = trainer.train()
    
    # SVRaster Inference  
    from nerfs.svraster import SVRasterRenderer
    renderer = SVRasterRenderer.from_checkpoint("checkpoint.pth")
    outputs = renderer.render_rays(rays_o, rays_d)
    """)
    
    print("Plenoxels style (new):")
    print("""
    # Plenoxels Training
    from nerfs.plenoxels import PlenoxelTrainer, PlenoxelTrainingConfig
    config = PlenoxelTrainingConfig(...)
    trainer = PlenoxelTrainer(config, train_dataset, val_dataset)
    renderer = trainer.train()
    
    # Plenoxels Inference
    from nerfs.plenoxels import PlenoxelRenderer
    renderer = PlenoxelRenderer.from_checkpoint("checkpoint.pth")
    outputs = renderer.render_rays(rays_o, rays_d)
    """)
    
    print("✅ Consistent API design across NeRF implementations!")


if __name__ == "__main__":
    print("Plenoxels Refactored Usage Examples")
    print("=" * 50)
    
    # Show preset configurations
    example_preset_configs()
    
    # Show interface comparison
    compare_with_svraster()
    
    # Note: The actual training and inference examples would require real datasets
    print("\nNote: Training and inference examples require real datasets.")
    print("See the individual functions for complete usage patterns.")
    
    print("\nRefactoring completed! 🎉")
    print("- ✅ Training and inference are now cleanly separated")
    print("- ✅ Consistent API with SVRaster")
    print("- ✅ Professional configuration management")
    print("- ✅ Comprehensive documentation and examples")
    print("- ✅ Backward compatibility maintained")
