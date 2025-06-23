"""
Example usage of InfNeRF for large-scale scene reconstruction.

This module demonstrates:
- InfNeRF model setup and configuration
- Dataset preparation and loading
- Training with pyramid supervision
- Inference and rendering
- Performance analysis and visualization
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import json
import time
from typing import Dict, List, Optional, Tuple, Any

from .core import InfNeRF, InfNeRFConfig
from .dataset import InfNeRFDataset, InfNeRFDatasetConfig
from .trainer import InfNeRFTrainer, InfNeRFTrainerConfig
from .utils.octree_utils import OctreeBuilder, visualize_octree, analyze_octree_memory
from .utils.lod_utils import pyramid_supervision, frustum_culling
from .utils.rendering_utils import memory_efficient_rendering, rendering_profiler


def demo_inf_nerf(data_path: str = "data/sample_scene",
                 output_path: str = "outputs/inf_nerf_demo",
                 device: str = "cuda"):
    """
    Complete demo of InfNeRF training and inference.
    
    Args:
        data_path: Path to dataset
        output_path: Path for outputs
        device: Device to use ('cuda' or 'cpu')
    """
    print("🚀 Starting InfNeRF Demo")
    print("=" * 50)
    
    # Create output directory
    output_dir = Path(output_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Setup configuration
    print("\n📋 Setting up configuration...")
    config = create_demo_config()
    print(f"   - Scene bound: {config.scene_bound}")
    print(f"   - Max depth: {config.max_depth}")
    print(f"   - GSD range: {config.min_gsd:.4f} - {config.max_gsd:.4f}")
    
    # 2. Load dataset
    print("\n📁 Loading dataset...")
    dataset_config = create_demo_dataset_config(data_path)
    train_dataset = InfNeRFDataset(dataset_config, split='train')
    val_dataset = InfNeRFDataset(dataset_config, split='val')
    
    print(f"   - Training images: {len(train_dataset.images)}")
    print(f"   - Validation images: {len(val_dataset.images)}")
    print(f"   - Sparse points: {len(train_dataset.get_sparse_points())}")
    
    # 3. Create model
    print("\n🏗️ Creating InfNeRF model...")
    model = InfNeRF(config)
    model.to(device)
    
    # Build octree from sparse points
    sparse_points = train_dataset.get_sparse_points()
    if len(sparse_points) > 0:
        model.build_octree(sparse_points)
        print(f"   - Built octree with {len(model.all_nodes)} nodes")
        
        # Analyze memory usage
        memory_stats = analyze_octree_memory(model.root_node)
        print(f"   - Total memory: {memory_stats['total_memory_mb']:.1f} MB")
        print(f"   - Nodes by level: {memory_stats['nodes_by_level']}")
    else:
        print("   - Warning: No sparse points found, using default octree")
    
    # 4. Visualize octree (optional)
    print("\n🎨 Visualizing octree structure...")
    try:
        viz_path = output_dir / "octree_structure.png"
        visualize_octree(model.root_node, max_depth=4, save_path=str(viz_path))
        print(f"   - Saved octree visualization: {viz_path}")
    except Exception as e:
        print(f"   - Visualization failed: {e}")
    
    # 5. Setup trainer
    print("\n🎯 Setting up trainer...")
    trainer_config = create_demo_trainer_config(output_path)
    trainer = InfNeRFTrainer(model, train_dataset, trainer_config, val_dataset)
    
    # 6. Training
    print("\n🚂 Starting training...")
    print(f"   - Epochs: {trainer_config.num_epochs}")
    print(f"   - Learning rate: {trainer_config.lr_init:.2e}")
    
    start_time = time.time()
    trainer.train()
    training_time = time.time() - start_time
    
    print(f"   - Training completed in {training_time:.1f} seconds")
    print(f"   - Best PSNR: {trainer.best_psnr:.2f}")
    
    # 7. Inference and rendering
    print("\n🎬 Running inference...")
    demo_inference(model, val_dataset, output_dir, device)
    
    # 8. Performance analysis
    print("\n📊 Performance analysis...")
    demo_performance_analysis(model, config, output_dir)
    
    print("\n✅ Demo completed successfully!")
    print(f"📁 Check outputs in: {output_path}")


def create_demo_config() -> InfNeRFConfig:
    """Create demo configuration for InfNeRF."""
    return InfNeRFConfig(
        # Octree parameters
        max_depth=6,
        min_depth=2,
        adaptive_depth=True,
        
        # LoD parameters
        base_resolution=32,
        grid_size=512,  # Reduced for demo
        max_gsd=1.0,
        min_gsd=0.01,
        
        # Network parameters
        hidden_dim=64,
        geo_feat_dim=15,
        num_layers=2,
        num_layers_color=3,
        
        # Hash encoding
        num_levels=12,  # Reduced for demo
        level_dim=2,
        per_level_scale=2.0,
        log2_hashmap_size=18,  # Reduced for demo
        
        # Sampling
        num_samples=64,
        num_importance=128,
        perturb_radius=True,
        
        # Training
        learning_rate=5e-3,
        weight_decay=1e-6,
        batch_size=1024,  # Reduced for demo
        
        # Scene
        scene_bound=2.0,
        
        # Pruning
        use_pruning=True,
        sparse_point_threshold=5
    )


def create_demo_dataset_config(data_path: str) -> InfNeRFDatasetConfig:
    """Create demo dataset configuration."""
    return InfNeRFDatasetConfig(
        data_root=data_path,
        images_path="images",
        sparse_points_path="sparse_points.ply",
        cameras_path="cameras.json",
        
        # Image processing
        image_scale=0.5,  # Downsample for demo
        max_image_size=1024,
        min_image_size=256,
        
        # Multi-resolution
        num_pyramid_levels=3,  # Reduced for demo
        pyramid_scale_factor=2.0,
        uniform_pixel_sampling=True,
        
        # Ray sampling
        rays_per_image=512,  # Reduced for demo
        batch_size=1024,
        use_patch_sampling=False,
        
        # Scene bounds
        scene_scale=1.0,
        near_plane=0.1,
        far_plane=100.0
    )


def create_demo_trainer_config(output_path: str) -> InfNeRFTrainerConfig:
    """Create demo trainer configuration."""
    return InfNeRFTrainerConfig(
        # Training
        num_epochs=20,  # Reduced for demo
        lr_init=5e-3,
        lr_final=1e-4,
        lr_decay_steps=5000,
        
        # Loss weights
        lambda_rgb=1.0,
        lambda_depth=0.1,
        lambda_distortion=0.01,
        lambda_transparency=1e-3,
        lambda_regularization=1e-4,
        
        # Strategy
        coarse_epochs=5,
        fine_epochs=15,
        progressive_training=True,
        
        # Memory
        max_batch_size=1024,
        gradient_accumulation_steps=1,
        mixed_precision=True,
        max_memory_gb=8.0,
        
        # Checkpointing
        checkpoint_dir=f"{output_path}/checkpoints",
        log_dir=f"{output_path}/logs",
        save_interval=1000,
        eval_interval=500,
        log_interval=50,
        
        # Validation
        num_val_images=3,
        
        # Wandb (disabled for demo)
        use_wandb=False
    )


def demo_inference(model: InfNeRF, 
                  dataset: InfNeRFDataset,
                  output_dir: Path,
                  device: str):
    """Demonstrate inference and rendering."""
    model.eval()
    
    # Get a test image
    test_sample = dataset[0]
    
    rays_o = test_sample['rays_o'][:256].to(device)  # Reduced for demo
    rays_d = test_sample['rays_d'][:256].to(device)
    target_rgb = test_sample['target_rgb'][:256]
    
    print(f"   - Rendering {len(rays_o)} rays...")
    
    # Render with profiling
    with rendering_profiler.profile("inference"):
        with torch.no_grad():
            rendered = model(
                rays_o=rays_o,
                rays_d=rays_d,
                near=test_sample['near'].item(),
                far=test_sample['far'].item(),
                focal_length=test_sample['focal_length'].item(),
                pixel_width=test_sample['pixel_width'].item()
            )
    
    # Calculate metrics
    pred_rgb = rendered['rgb'].cpu()
    mse = torch.mean((pred_rgb - target_rgb) ** 2)
    psnr = -10 * torch.log10(mse)
    
    print(f"   - PSNR: {psnr:.2f} dB")
    print(f"   - MSE: {mse:.6f}")
    
    # Save rendered image (if we can reshape it)
    try:
        if len(pred_rgb) == 256:  # 16x16 for demo
            img_size = 16
            pred_img = pred_rgb.view(img_size, img_size, 3).numpy()
            target_img = target_rgb.view(img_size, img_size, 3).numpy()
            
            # Create comparison plot
            fig, axes = plt.subplots(1, 2, figsize=(10, 5))
            axes[0].imshow(target_img)
            axes[0].set_title('Target')
            axes[0].axis('off')
            
            axes[1].imshow(np.clip(pred_img, 0, 1))
            axes[1].set_title(f'Predicted (PSNR: {psnr:.2f})')
            axes[1].axis('off')
            
            plt.tight_layout()
            plt.savefig(output_dir / "rendered_comparison.png", dpi=150, bbox_inches='tight')
            plt.close()
            
            print(f"   - Saved comparison: {output_dir}/rendered_comparison.png")
    except Exception as e:
        print(f"   - Could not save image: {e}")


def demo_performance_analysis(model: InfNeRF,
                            config: InfNeRFConfig,
                            output_dir: Path):
    """Analyze and visualize performance characteristics."""
    
    # Memory analysis
    memory_stats = model.get_memory_usage()
    print(f"   - Total model memory: {memory_stats['total_mb']:.1f} MB")
    print(f"   - Number of nodes: {memory_stats['num_nodes']}")
    print(f"   - Max octree level: {memory_stats['max_level']}")
    
    # Rendering profiler stats
    profiler_stats = rendering_profiler.get_stats()
    if profiler_stats:
        print("\n   Rendering Performance:")
        for name, stats in profiler_stats.items():
            print(f"     - {name}: {stats['avg_time']:.3f}s "
                  f"(memory: {stats['avg_memory_mb']:.1f} MB)")
    
    # Create performance visualization
    try:
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # Memory by level
        memory_by_level = memory_stats['by_level_mb']
        levels = list(memory_by_level.keys())
        memories = list(memory_by_level.values())
        
        axes[0, 0].bar(levels, memories)
        axes[0, 0].set_title('Memory Usage by Octree Level')
        axes[0, 0].set_xlabel('Level')
        axes[0, 0].set_ylabel('Memory (MB)')
        
        # GSD by level
        gsds = []
        for level in levels:
            gsd = config.max_gsd / (2 ** level)
            gsds.append(max(gsd, config.min_gsd))
        
        axes[0, 1].semilogy(levels, gsds, 'o-')
        axes[0, 1].set_title('Ground Sampling Distance by Level')
        axes[0, 1].set_xlabel('Level')
        axes[0, 1].set_ylabel('GSD (meters)')
        axes[0, 1].grid(True)
        
        # Octree node distribution
        node_counts = [1]  # Root
        for level in range(1, memory_stats['max_level'] + 1):
            if level in memory_by_level:
                # Estimate node count from memory
                estimated_nodes = max(1, int(memory_by_level[level] / 10))  # Rough estimate
                node_counts.append(estimated_nodes)
            else:
                node_counts.append(0)
        
        axes[1, 0].bar(range(len(node_counts)), node_counts)
        axes[1, 0].set_title('Node Count by Level')
        axes[1, 0].set_xlabel('Level')
        axes[1, 0].set_ylabel('Number of Nodes')
        
        # Performance comparison (theoretical)
        max_nodes = sum(8**i for i in range(memory_stats['max_level'] + 1))
        active_nodes = memory_stats['num_nodes']
        compression_ratio = active_nodes / max_nodes
        
        categories = ['Full Octree', 'InfNeRF (Pruned)', 'Traditional NeRF']
        memory_usage = [max_nodes * 10, active_nodes * 10, max_nodes * 20]  # Rough estimates
        
        axes[1, 1].bar(categories, memory_usage)
        axes[1, 1].set_title('Memory Usage Comparison')
        axes[1, 1].set_ylabel('Memory (MB)')
        axes[1, 1].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.savefig(output_dir / "performance_analysis.png", dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"   - Saved performance analysis: {output_dir}/performance_analysis.png")
        
        # Save stats to JSON
        stats_dict = {
            'memory_stats': memory_stats,
            'profiler_stats': profiler_stats,
            'compression_ratio': compression_ratio,
            'config': config.__dict__
        }
        
        with open(output_dir / "performance_stats.json", 'w') as f:
            json.dump(stats_dict, f, indent=2, default=str)
        
        print(f"   - Saved stats: {output_dir}/performance_stats.json")
        
    except Exception as e:
        print(f"   - Performance visualization failed: {e}")


def create_inf_nerf_demo(scene_path: str = "data/demo_scene") -> Dict[str, Any]:
    """
    Create a simple demo scene and run InfNeRF.
    
    Args:
        scene_path: Path to create demo scene
        
    Returns:
        Demo results and statistics
    """
    print("🎮 Creating InfNeRF demo scene...")
    
    # Create demo scene directory
    scene_dir = Path(scene_path)
    scene_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate synthetic data
    demo_data = generate_synthetic_demo_data()
    
    # Save demo data
    save_demo_data(demo_data, scene_dir)
    
    # Run demo
    demo_results = {
        'scene_path': str(scene_dir),
        'num_images': len(demo_data['images']),
        'num_sparse_points': len(demo_data['sparse_points']),
        'scene_bounds': demo_data['scene_bounds']
    }
    
    print(f"   - Created demo scene with {demo_results['num_images']} images")
    print(f"   - Generated {demo_results['num_sparse_points']} sparse points")
    
    return demo_results


def generate_synthetic_demo_data() -> Dict[str, Any]:
    """Generate synthetic data for demo."""
    
    # Simple synthetic scene: colored cube
    np.random.seed(42)
    
    # Generate sparse points (corners and some random points in a cube)
    cube_corners = np.array([
        [-1, -1, -1], [1, -1, -1], [1, 1, -1], [-1, 1, -1],
        [-1, -1, 1], [1, -1, 1], [1, 1, 1], [-1, 1, 1]
    ]) * 0.5
    
    random_points = np.random.uniform(-0.5, 0.5, (20, 3))
    sparse_points = np.vstack([cube_corners, random_points])
    
    # Generate cameras in a circle around the scene
    num_cameras = 8
    radius = 3.0
    images = []
    cameras = {}
    
    for i in range(num_cameras):
        angle = 2 * np.pi * i / num_cameras
        
        # Camera position
        cam_pos = np.array([
            radius * np.cos(angle),
            radius * np.sin(angle),
            1.0
        ])
        
        # Look at origin
        look_at = np.array([0.0, 0.0, 0.0])
        up = np.array([0.0, 0.0, 1.0])
        
        # Create camera matrix
        forward = look_at - cam_pos
        forward = forward / np.linalg.norm(forward)
        right = np.cross(forward, up)
        right = right / np.linalg.norm(right)
        up = np.cross(right, forward)
        
        # Extrinsic matrix
        rotation = np.column_stack([right, up, -forward])
        extrinsic = np.eye(4)
        extrinsic[:3, :3] = rotation.T
        extrinsic[:3, 3] = -rotation.T @ cam_pos
        
        # Simple synthetic image (gradient)
        img_size = 64
        img = np.zeros((img_size, img_size, 3))
        for y in range(img_size):
            for x in range(img_size):
                img[y, x, 0] = x / img_size  # Red gradient
                img[y, x, 1] = y / img_size  # Green gradient
                img[y, x, 2] = 0.5  # Blue constant
        
        images.append(img)
        
        # Camera parameters
        focal_length = img_size * 0.8  # Rough focal length
        cameras[f"image_{i:03d}.png"] = {
            'intrinsic': [
                [focal_length, 0, img_size/2],
                [0, focal_length, img_size/2],
                [0, 0, 1]
            ],
            'extrinsic': extrinsic.tolist()
        }
    
    return {
        'images': images,
        'cameras': cameras,
        'sparse_points': sparse_points,
        'scene_bounds': (np.array([-1, -1, -1]), np.array([1, 1, 1]))
    }


def save_demo_data(demo_data: Dict[str, Any], scene_dir: Path):
    """Save demo data to files."""
    
    # Create subdirectories
    (scene_dir / "images").mkdir(exist_ok=True)
    
    # Save images
    for i, img in enumerate(demo_data['images']):
        img_path = scene_dir / "images" / f"image_{i:03d}.png"
        # Convert to uint8 and save
        img_uint8 = (np.clip(img, 0, 1) * 255).astype(np.uint8)
        try:
            from PIL import Image
            Image.fromarray(img_uint8).save(img_path)
        except ImportError:
            # Fallback: save as numpy array
            np.save(str(img_path).replace('.png', '.npy'), img)
    
    # Save cameras
    with open(scene_dir / "cameras.json", 'w') as f:
        json.dump(demo_data['cameras'], f, indent=2)
    
    # Save sparse points as simple text file (PLY format would be better)
    with open(scene_dir / "sparse_points.ply", 'w') as f:
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {len(demo_data['sparse_points'])}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        f.write("end_header\n")
        
        for point in demo_data['sparse_points']:
            f.write(f"{point[0]:.6f} {point[1]:.6f} {point[2]:.6f}\n")


if __name__ == "__main__":
    """Run demo when script is executed directly."""
    
    # Check if CUDA is available
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Create synthetic demo scene
    demo_scene = create_inf_nerf_demo("data/synthetic_demo")
    
    # Run full demo
    demo_inf_nerf(
        data_path=demo_scene['scene_path'],
        output_path="outputs/inf_nerf_synthetic_demo",
        device=device
    )
    
    print("\n🎉 Demo completed! Check the outputs directory for results.") 