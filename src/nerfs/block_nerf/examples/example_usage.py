"""
Block-NeRF Example Usage

This example demonstrates the refactored Block-NeRF architecture,
following the SVRaster pattern with dual rendering systems.
"""

import torch
import numpy as np
from pathlib import Path

def example_training():
    """Example of Block-NeRF training with volume rendering."""
    print("=== Block-NeRF Training Example ===")
    
    # Import refactored components
    from src.nerfs.block_nerf import (
        BlockNeRFConfig, BlockNeRFTrainerConfig, VolumeRendererConfig,
        BlockNeRFDatasetConfig, create_block_nerf_trainer,
        create_volume_renderer, create_block_nerf_dataloader
    )
    
    # 1. Create model configuration
    model_config = BlockNeRFConfig(
        scene_bounds=(-100, -100, -10, 100, 100, 10),
        block_size=75.0,
        overlap_ratio=0.1,
        pos_encoding_levels=16,
        dir_encoding_levels=4,
        hidden_dim=256,
        num_layers=8,
        use_integrated_encoding=True,
        device="cuda" if torch.cuda.is_available() else "cpu"
    )
    
    # 2. Create trainer configuration
    trainer_config = BlockNeRFTrainerConfig(
        num_epochs=10,  # Reduced for example
        batch_size=1,
        learning_rate=5e-4,
        training_strategy="adaptive",
        blocks_per_iteration=4,
        checkpoint_every=1000,
        val_every=500,
        device=model_config.device
    )
    
    # 3. Create volume renderer configuration (for training)
    volume_renderer_config = VolumeRendererConfig(
        num_samples=64,
        num_importance_samples=64,
        use_hierarchical_sampling=True,
        white_background=True
    )
    
    # 4. Create volume renderer (tightly coupled with trainer)
    volume_renderer = create_volume_renderer(volume_renderer_config)
    
    # 5. Create trainer (with integrated volume renderer)
    trainer = create_block_nerf_trainer(
        model_config=model_config,
        trainer_config=trainer_config,
        volume_renderer=volume_renderer,
        device=model_config.device
    )
    
    print(f"✓ Created trainer with {len(trainer.block_manager.block_centers)} blocks")
    print(f"✓ Volume renderer configured for training")
    print(f"✓ Using device: {trainer.device}")
    
    # 6. Create dataset configuration
    dataset_config = BlockNeRFDatasetConfig(
        data_dir="./demo_data",
        dataset_type="blender",
        image_width=400,
        image_height=400,
        num_rays=1024,
        use_appearance_ids=True,
        use_exposure=True
    )
    
    # 7. Create mock training data for demo
    create_mock_training_data(dataset_config.data_dir)
    
    try:
        # 8. Create data loaders
        train_loader = create_block_nerf_dataloader(
            dataset_config, split="train", batch_size=trainer_config.batch_size
        )
        val_loader = create_block_nerf_dataloader(
            dataset_config, split="val", batch_size=trainer_config.batch_size
        )
        
        print(f"✓ Created data loaders with {len(train_loader)} training batches")
        
        # 9. Start training (demo with reduced iterations)
        print("🚀 Starting Block-NeRF training...")
        trainer.train(train_loader, val_loader)
        
        print("✓ Training completed successfully!")
        return trainer
        
    except Exception as e:
        print(f"⚠️ Training failed: {e}")
        print("This is expected for the demo without real data")
        return trainer


def example_inference():
    """Example of Block-NeRF inference with block rasterization."""
    print("\n=== Block-NeRF Inference Example ===")
    
    # Import refactored components
    from src.nerfs.block_nerf import (
        BlockNeRFConfig, BlockNeRFRendererConfig, BlockRasterizerConfig,
        create_block_nerf_renderer, create_block_rasterizer
    )
    
    # 1. Create model configuration (same as training)
    model_config = BlockNeRFConfig(
        scene_bounds=(-100, -100, -10, 100, 100, 10),
        block_size=75.0,
        overlap_ratio=0.1,
        device="cuda" if torch.cuda.is_available() else "cpu"
    )
    
    # 2. Create renderer configuration
    renderer_config = BlockNeRFRendererConfig(
        image_width=800,
        image_height=600,
        chunk_size=1024,
        use_cached_blocks=True,
        max_cached_blocks=8,
        background_color=(1.0, 1.0, 1.0),
        device=model_config.device
    )
    
    # 3. Create rasterizer configuration (for inference)
    rasterizer_config = BlockRasterizerConfig(
        max_blocks_per_ray=8,
        samples_per_block=32,
        adaptive_sampling=True,
        early_termination=True,
        use_block_culling=True,
        use_morton_ordering=True
    )
    
    # 4. Create block rasterizer (tightly coupled with renderer)
    rasterizer = create_block_rasterizer(rasterizer_config)
    
    # 5. Create renderer (with integrated rasterizer)
    renderer = create_block_nerf_renderer(
        model_config=model_config,
        renderer_config=renderer_config,
        rasterizer=rasterizer,
        device=model_config.device
    )
    
    print(f"✓ Created renderer with rasterization")
    print(f"✓ Block rasterizer configured for inference")
    print(f"✓ Using device: {renderer.device}")
    
    # 6. Create mock checkpoint for demo
    checkpoint_dir = "./demo_checkpoints"
    create_mock_checkpoint(checkpoint_dir, model_config)
    
    try:
        # 7. Load trained blocks
        renderer.load_blocks(checkpoint_dir)
        print(f"✓ Loaded {len(renderer.block_manager.blocks)} blocks")
        
        # 8. Create camera pose and intrinsics
        camera_pose = torch.eye(4, device=renderer.device, dtype=torch.float32)
        camera_pose[2, 3] = 5.0  # Move camera back
        
        focal = renderer_config.image_width * 0.8
        intrinsics = torch.tensor([
            [focal, 0, renderer_config.image_width/2],
            [0, focal, renderer_config.image_height/2],
            [0, 0, 1]
        ], device=renderer.device, dtype=torch.float32)
        
        # 9. Render image
        print("🎨 Rendering image...")
        result = renderer.render_image(
            camera_pose=camera_pose,
            intrinsics=intrinsics,
            width=renderer_config.image_width,
            height=renderer_config.image_height,
            appearance_id=0,
            exposure_value=1.0
        )
        
        print(f"✓ Rendered image: {result['rgb'].shape}")
        print(f"✓ Depth map: {result['depth'].shape}")
        print(f"✓ Alpha channel: {result['alpha'].shape}")
        
        # 10. Save rendered image
        save_rendered_image(result["rgb"], "./demo_output_image.png")
        
        return renderer
        
    except Exception as e:
        print(f"⚠️ Inference failed: {e}")
        print("This is expected for the demo without real checkpoints")
        return renderer


def create_mock_training_data(data_dir: str):
    """Create mock training data for demonstration."""
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    
    # Create mock transforms file
    transforms = {
        "camera_angle_x": 0.6911112070083618,
        "frames": []
    }
    
    # Add a few mock frames
    for i in range(5):
        frame = {
            "file_path": f"frame_{i:03d}",
            "transform_matrix": [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 5.0],
                [0.0, 0.0, 0.0, 1.0]
            ]
        }
        transforms["frames"].append(frame)
    
    # Save transforms for each split
    for split in ["train", "val", "test"]:
        transforms_file = data_dir / f"transforms_{split}.json"
        import json
        with open(transforms_file, 'w') as f:
            json.dump(transforms, f, indent=2)
    
    print(f"✓ Created mock training data in {data_dir}")


def create_mock_checkpoint(checkpoint_dir: str, model_config):
    """Create mock checkpoint for demonstration."""
    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    # Import here to avoid circular imports
    from src.nerfs.block_nerf import BlockNeRFModel, create_block_manager
    
    # Create block manager
    block_manager = create_block_manager(
        scene_bounds=model_config.scene_bounds,
        block_size=model_config.block_size,
        device=model_config.device
    )
    
    # Create a few mock blocks
    block_names = list(block_manager.block_centers.keys())[:4]  # Use first 4 blocks
    
    for block_name in block_names:
        # Create mock block model
        block = BlockNeRFModel(model_config)
        
        # Save mock checkpoint
        checkpoint_path = checkpoint_dir / f"{block_name}.pth"
        torch.save({
            'model_state_dict': block.state_dict(),
            'metadata': block_manager.block_metadata[block_name]
        }, checkpoint_path)
    
    # Save block layout
    layout_path = checkpoint_dir / "block_layout.json"
    block_manager.save_block_layout(str(layout_path))
    
    print(f"✓ Created mock checkpoint in {checkpoint_dir}")


def save_rendered_image(rgb_tensor: torch.Tensor, output_path: str):
    """Save rendered image to file."""
    try:
        import imageio
        
        # Convert to numpy and scale to [0, 255]
        rgb_np = rgb_tensor.cpu().numpy()
        rgb_np = np.clip(rgb_np * 255, 0, 255).astype(np.uint8)
        
        imageio.imsave(output_path, rgb_np)
        print(f"✓ Saved rendered image to {output_path}")
        
    except ImportError:
        print("⚠️ imageio not available for saving image")
    except Exception as e:
        print(f"⚠️ Failed to save image: {e}")


def architecture_demonstration():
    """Demonstrate the dual architecture design."""
    print("\n=== Block-NeRF Architecture Demonstration ===")
    
    print("""
🏗️ Refactored Block-NeRF Architecture (following SVRaster pattern):

📚 TRAINING PHASE (Volume Rendering):
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│ BlockNeRFTrainer│◄──►│  VolumeRenderer  │◄──►│ BlockNeRFModel  │
│                 │    │                  │    │    (Blocks)     │
│ - Block sampling│    │ - Ray sampling   │    │ - Neural nets   │
│ - Loss comp.    │    │ - Alpha blending │    │ - Embeddings    │
│ - Optimization  │    │ - Hierarchical   │    │ - Pose refine.  │
└─────────────────┘    └──────────────────┘    └─────────────────┘
        │                        │                        │
        └────────── Tightly Coupled for Stable Training ───┘

🎨 INFERENCE PHASE (Block Rasterization):
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│BlockNeRFRenderer│◄──►│ BlockRasterizer  │◄──►│ BlockNeRFModel  │
│                 │    │                  │    │    (Blocks)     │
│ - Image render  │    │ - Ray marching   │    │ - Cached models │
│ - Video render  │    │ - Block culling  │    │ - Fast forward  │
│ - Camera mgmt   │    │ - Interpolation  │    │ - LOD support   │
└─────────────────┘    └──────────────────┘    └─────────────────┘
        │                        │                        │
        └────────── Tightly Coupled for Fast Inference ────┘

🔄 SHARED COMPONENTS:
- BlockManager: Spatial decomposition and block coordination
- BlockNeRFConfig: Unified configuration system
- BlockNeRFDataset: Data loading and preprocessing

✨ KEY BENEFITS:
- Training stability through volume rendering
- Inference speed through rasterization
- Clean separation of concerns
- Scalable to city-scale scenes
""")


def quick_compatibility_check():
    """Quick compatibility check for Block-NeRF."""
    print("\n=== Block-NeRF Compatibility Check ===")
    
    try:
        import torch
        print(f"✓ PyTorch: {torch.__version__}")
        print(f"✓ CUDA Available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"✓ CUDA Device: {torch.cuda.get_device_name()}")
    except ImportError:
        print("❌ PyTorch not found")
        return False
    
    try:
        from src.nerfs.block_nerf import check_compatibility
        check_compatibility()
        return True
    except ImportError as e:
        print(f"❌ Block-NeRF import failed: {e}")
        return False


def main():
    """Main demonstration function."""
    print("🚀 Block-NeRF Refactored Demo")
    print("=" * 50)
    
    # Check compatibility
    if not quick_compatibility_check():
        print("❌ Compatibility check failed")
        return
    
    # Show architecture
    architecture_demonstration()
    
    # Training example
    trainer = example_training()
    
    # Inference example
    renderer = example_inference()
    
    print("\n" + "=" * 50)
    print("🎉 Block-NeRF Demo Complete!")
    print("✓ Training pipeline: Trainer ↔ VolumeRenderer")
    print("✓ Inference pipeline: Renderer ↔ BlockRasterizer")
    print("✓ Architecture follows SVRaster refactoring pattern")
    print("✓ Ready for city-scale scene reconstruction")


if __name__ == "__main__":
    main()
