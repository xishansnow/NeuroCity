"""
Example usage of Classic NeRF.

This script demonstrates how to:
1. Create and configure a NeRF model
2. Load data and create data loaders
3. Train the model
4. Render novel views
5. Evaluate the model
"""

import os
import torch
import numpy as np
import imageio
from pathlib import Path

from classic_nerf import (
    NeRFConfig, 
    NeRF, 
    NeRFTrainer,
    create_nerf_dataloader,
    pose_spherical,
    to8b,
    create_spherical_poses
)


def create_demo_config():
    """Create a demo configuration for NeRF."""
    config = NeRFConfig(
        # Network architecture
        netdepth=8,
        netwidth=256,
        netdepth_fine=8,
        netwidth_fine=256,
        
        # Positional encoding
        multires=10,
        multires_views=4,
        
        # Sampling
        N_samples=64,
        N_importance=128,
        perturb=True,
        use_viewdirs=True,
        
        # Rendering
        raw_noise_std=0.0,
        white_bkgd=True,
        
        # Training
        learning_rate=5e-4,
        lrate_decay=250,
        
        # Scene bounds
        near=2.0,
        far=6.0
    )
    return config


def train_demo():
    """Demonstrate training a NeRF model."""
    print("=== Classic NeRF Training Demo ===")
    
    # Create configuration
    config = create_demo_config()
    print(f"Created NeRF config:")
    print(f"  Network depth: {config.netdepth}")
    print(f"  Network width: {config.netwidth}")
    print(f"  Coarse samples: {config.N_samples}")
    print(f"  Fine samples: {config.N_importance}")
    
    # Create data loaders (using synthetic data for demo)
    dataset_path = "data/demo_blender_scene"
    os.makedirs(dataset_path, exist_ok=True)
    
    try:
        train_loader = create_nerf_dataloader(
            dataset_type='blender',
            basedir=dataset_path,
            split='train',
            batch_size=1024,
            shuffle=True,
            white_bkgd=config.white_bkgd
        )
        
        val_loader = create_nerf_dataloader(
            dataset_type='blender', 
            basedir=dataset_path,
            split='val',
            batch_size=1024,
            shuffle=False,
            white_bkgd=config.white_bkgd
        )
        
        print(f"Created data loaders:")
        print(f"  Train samples: {len(train_loader.dataset):,}")
        print(f"  Val samples: {len(val_loader.dataset):,}")
        
    except Exception as e:
        print(f"Warning: Could not load real data ({e})")
        print("Using synthetic data for demonstration...")
        train_loader = create_synthetic_dataloader(batch_size=512)
        val_loader = create_synthetic_dataloader(batch_size=512)
    
    # Create trainer
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    trainer = NeRFTrainer(config, device=device)
    
    print(f"Created trainer on device: {device}")
    print(f"Model parameters:")
    print(f"  Coarse network: {sum(p.numel() for p in trainer.model_coarse.parameters()):,}")
    if trainer.model_fine is not None:
        print(f"  Fine network: {sum(p.numel() for p in trainer.model_fine.parameters()):,}")
    
    # Train for a few epochs (demo)
    output_dir = "output/classic_nerf_demo"
    os.makedirs(output_dir, exist_ok=True)
    
    trainer.train(
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=5,  # Short demo
        log_dir=f"{output_dir}/logs",
        ckpt_dir=f"{output_dir}/checkpoints",
        val_interval=2,
        save_interval=5
    )
    
    print("Training completed!")
    return trainer, config


def render_demo(trainer, config):
    """Demonstrate rendering with trained NeRF."""
    print("\n=== Rendering Demo ===")
    
    # Create spherical camera poses
    render_poses = create_spherical_poses(radius=4.0, n_poses=8)
    
    # Camera parameters
    H, W = 400, 400
    focal = 555.5  # Typical focal length for Blender scenes
    K = np.array([
        [focal, 0, W/2],
        [0, focal, H/2], 
        [0, 0, 1]
    ])
    
    print(f"Rendering {len(render_poses)} views...")
    print(f"Image size: {H}x{W}")
    
    # Render images
    output_dir = "output/classic_nerf_demo/renders"
    os.makedirs(output_dir, exist_ok=True)
    
    rgbs = []
    for i, c2w in enumerate(render_poses):
        print(f"Rendering view {i+1}/{len(render_poses)}...")
        
        rgb = trainer.render_test_image(H, W, K, c2w.numpy())
        rgb8 = to8b(rgb)
        
        # Save image
        filename = os.path.join(output_dir, f'render_{i:03d}.png')
        imageio.imwrite(filename, rgb8)
        rgbs.append(rgb)
    
    print(f"Rendered images saved to: {output_dir}")
    
    # Create video
    video_path = os.path.join(output_dir, 'render_video.mp4')
    try:
        rgb8_video = [to8b(rgb) for rgb in rgbs]
        imageio.mimwrite(video_path, rgb8_video, fps=5, quality=8)
        print(f"Video saved to: {video_path}")
    except:
        print("Could not create video (imageio-ffmpeg not available)")
    
    return rgbs


def evaluate_demo(trainer, config):
    """Demonstrate evaluation of NeRF model."""
    print("\n=== Evaluation Demo ===")
    
    # Create test data
    test_loader = create_synthetic_dataloader(batch_size=1024)
    
    # Evaluate
    val_metrics = trainer.validate(test_loader)
    
    print("Evaluation results:")
    print(f"  Loss: {val_metrics['val_loss']:.4f}")
    print(f"  PSNR: {val_metrics['val_psnr']:.2f} dB")
    
    return val_metrics


def create_synthetic_dataloader(batch_size=1024):
    """Create a synthetic dataloader for demonstration."""
    from torch.utils.data import DataLoader, TensorDataset
    
    # Create random ray data
    num_rays = 10000
    rays_o = torch.randn(num_rays, 3) * 2
    rays_d = torch.randn(num_rays, 3)
    rays_d = rays_d / torch.norm(rays_d, dim=-1, keepdim=True)
    targets = torch.rand(num_rays, 3)
    
    dataset = TensorDataset(rays_o, rays_d, targets)
    
    def collate_fn(batch):
        rays_o, rays_d, targets = zip(*batch)
        return {
            'rays_o': torch.stack(rays_o),
            'rays_d': torch.stack(rays_d),
            'targets': torch.stack(targets)
        }
    
    return DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)


def model_analysis_demo():
    """Demonstrate model analysis and visualization."""
    print("\n=== Model Analysis Demo ===")
    
    config = create_demo_config()
    
    # Create model
    model = NeRF(config)
    
    print("Model architecture:")
    print(f"  Total parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Analyze positional encoding
    print(f"  Positional encoding dimensions:")
    print(f"    Coordinate embedding: {model.embed_fn.out_dim}")
    if config.use_viewdirs:
        print(f"    Direction embedding: {model.embeddirs_fn.out_dim}")
    
    # Test forward pass
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    # Create test input
    batch_size = 100
    pts_embedded = torch.randn(batch_size, model.embed_fn.out_dim).to(device)
    
    if config.use_viewdirs:
        dirs_embedded = torch.randn(batch_size, model.embeddirs_fn.out_dim).to(device)
        test_input = torch.cat([pts_embedded, dirs_embedded], -1)
    else:
        test_input = pts_embedded
    
    # Forward pass
    with torch.no_grad():
        output = model(test_input)
    
    print(f"  Test forward pass:")
    print(f"    Input shape: {test_input.shape}")
    print(f"    Output shape: {output.shape}")
    print(f"    RGB range: [{output[:, :3].min().item():.3f}, {output[:, :3].max().item():.3f}]")
    print(f"    Density range: [{output[:, 3].min().item():.3f}, {output[:, 3].max().item():.3f}]")


def main():
    """Run all demos."""
    print("Classic NeRF Demo Script")
    print("=" * 50)
    
    # Model analysis
    model_analysis_demo()
    
    # Training demo  
    trainer, config = train_demo()
    
    # Rendering demo
    rgbs = render_demo(trainer, config)
    
    # Evaluation demo
    metrics = evaluate_demo(trainer, config)
    
    print("\n" + "=" * 50)
    print("Demo completed successfully!")
    print(f"Check output directory: output/classic_nerf_demo/")


if __name__ == "__main__":
    main()
