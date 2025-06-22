"""
Example usage of Mip-NeRF

This script demonstrates how to use the Mip-NeRF implementation for training and inference.
"""

import torch
import numpy as np
from pathlib import Path

from .core import MipNeRF, MipNeRFConfig
from .dataset import create_mip_nerf_dataset
from .trainer import MipNeRFTrainer


def train_mip_nerf_example():
    """Example of training Mip-NeRF on a synthetic dataset"""
    
    # Configuration
    config = MipNeRFConfig(
        netdepth=8,
        netwidth=256,
        num_samples=64,
        num_importance=128,
        use_viewdirs=True,
        lr_init=5e-4,
        lr_final=5e-6,
        lr_decay=250
    )
    
    # Create datasets
    data_dir = "path/to/blender/dataset"  # Replace with actual path
    train_dataset = create_mip_nerf_dataset(
        data_dir, 
        dataset_type='blender', 
        split='train',
        white_bkgd=True,
        half_res=True
    )
    
    val_dataset = create_mip_nerf_dataset(
        data_dir,
        dataset_type='blender',
        split='val',
        white_bkgd=True,
        half_res=True
    )
    
    # Create model
    model = MipNeRF(config)
    
    # Create trainer
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    trainer = MipNeRFTrainer(
        config=config,
        model=model,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        device=device,
        log_dir='./logs/mip_nerf_experiment'
    )
    
    # Train
    trainer.train(
        num_epochs=100,
        save_freq=1000,
        val_freq=500,
        log_freq=100
    )
    
    print("Training completed!")


def inference_example():
    """Example of using trained Mip-NeRF for inference"""
    
    # Load configuration and model
    config = MipNeRFConfig()
    model = MipNeRF(config)
    
    # Load trained weights
    checkpoint_path = "path/to/trained/model.pth"
    if Path(checkpoint_path).exists():
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        model.load_state_dict(checkpoint['model_state_dict'])
        print("Model loaded successfully")
    else:
        print("Checkpoint not found, using random weights")
    
    model.eval()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = model.to(device)
    
    # Example ray casting
    batch_size = 1024
    rays_o = torch.randn(batch_size, 3, device=device)  # Ray origins
    rays_d = torch.randn(batch_size, 3, device=device)  # Ray directions
    rays_d = torch.nn.functional.normalize(rays_d, dim=-1)  # Normalize directions
    viewdirs = rays_d  # Use ray directions as view directions
    
    # Render
    with torch.no_grad():
        results = model(
            origins=rays_o,
            directions=rays_d,
            viewdirs=viewdirs,
            near=2.0,
            far=6.0,
            pixel_radius=0.001
        )
    
    # Extract results
    if 'fine' in results:
        rgb = results['fine']['rgb']
        depth = results['fine']['depth']
    else:
        rgb = results['coarse']['rgb']
        depth = results['coarse']['depth']
    
    print(f"Rendered RGB shape: {rgb.shape}")
    print(f"Rendered depth shape: {depth.shape}")
    print(f"RGB range: [{rgb.min():.3f}, {rgb.max():.3f}]")
    print(f"Depth range: [{depth.min():.3f}, {depth.max():.3f}]")


def render_novel_views_example():
    """Example of rendering novel views from trained model"""
    
    # Load model (similar to inference_example)
    config = MipNeRFConfig()
    model = MipNeRF(config)
    
    # Generate spiral camera path
    def generate_spiral_poses(num_frames=30, radius=4.0):
        """Generate spiral camera poses"""
        poses = []
        
        for i in range(num_frames):
            angle = 2 * np.pi * i / num_frames
            
            # Camera position
            cam_pos = np.array([
                radius * np.cos(angle),
                0.0,
                radius * np.sin(angle)
            ])
            
            # Look at origin
            forward = -cam_pos / np.linalg.norm(cam_pos)
            up = np.array([0., 1., 0.])
            right = np.cross(forward, up)
            right = right / np.linalg.norm(right)
            up = np.cross(right, forward)
            
            # Create pose matrix
            pose = np.eye(4)
            pose[:3, 0] = right
            pose[:3, 1] = up
            pose[:3, 2] = forward
            pose[:3, 3] = cam_pos
            
            poses.append(torch.from_numpy(pose.astype(np.float32)))
        
        return torch.stack(poses)
    
    # Generate poses
    spiral_poses = generate_spiral_poses(num_frames=60)
    
    # Note: This would require a trained model and proper dataset setup
    print(f"Generated {len(spiral_poses)} spiral poses for novel view synthesis")
    print("To actually render, you would need:")
    print("1. A trained MipNeRF model")
    print("2. Proper camera intrinsics")
    print("3. Call trainer.render_spiral_video(spiral_poses, 'output.mp4')")


def debug_integrated_positional_encoding():
    """Debug example for integrated positional encoding"""
    
    from .core import IntegratedPositionalEncoder
    
    # Create encoder
    encoder = IntegratedPositionalEncoder(min_deg=0, max_deg=10)
    
    # Example Gaussian parameters
    batch_size = 100
    means = torch.randn(batch_size, 3)  # 3D coordinates
    vars = torch.abs(torch.randn(batch_size, 3)) * 0.1  # Positive variances
    
    # Encode
    encoding = encoder(means, vars)
    
    print(f"Input means shape: {means.shape}")
    print(f"Input vars shape: {vars.shape}")
    print(f"Output encoding shape: {encoding.shape}")
    print(f"Expected output dim: {2 * 3 * (10 - 0)} = {2 * 3 * 10}")
    
    # Compare with regular positional encoding
    regular_encoding = torch.cat([
        torch.sin(means * (2**i)) for i in range(10)
    ] + [
        torch.cos(means * (2**i)) for i in range(10)
    ], dim=-1)
    
    print(f"Regular encoding shape: {regular_encoding.shape}")
    print(f"IPE handles variance, regular PE doesn't")


def main():
    """Main example function"""
    print("Mip-NeRF Examples")
    print("================")
    
    print("\n1. Debug Integrated Positional Encoding:")
    debug_integrated_positional_encoding()
    
    print("\n2. Model Inference Example:")
    inference_example()
    
    print("\n3. Novel View Synthesis Setup:")
    render_novel_views_example()
    
    print("\nTo run training, uncomment and modify the train_mip_nerf_example() call")
    print("Make sure to set the correct data_dir path")
    
    # Uncomment to run training (requires dataset)
    # train_mip_nerf_example()


if __name__ == "__main__":
    main() 