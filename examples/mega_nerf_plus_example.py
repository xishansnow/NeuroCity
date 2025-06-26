"""
Mega-NeRF+ Example Usage

This script demonstrates how to use Mega-NeRF+ for enhanced large-scale 
scene reconstruction with improved clustering and optimization.
"""

import os
import sys
import torch
import argparse
from pathlib import Path

# Add the project root to Python path to enable imports
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

try:
    from src.nerfs.mega_nerf_plus import (
        MegaNeRFPlusConfig, MegaNeRFPlusModel, MegaNeRFPlusTrainer, create_mega_nerf_plus_dataloader, create_mega_nerf_plus_dataset
    )
except ImportError as e:
    print(f"Import error: {e}")
    print("Please make sure you're running from the project root directory")
    sys.exit(1)


def basic_example():
    """Basic Mega-NeRF+ example with default configuration."""
    print("=== Basic Mega-NeRF+ Example ===")
    
    config = MegaNeRFPlusConfig(
        # Enhanced spatial decomposition
        num_clusters=8, cluster_method="adaptive_kmeans", overlap_ratio=0.15, # Network architecture
        netdepth=8, netwidth=256, # Advanced features
        use_attention_mechanism=True, use_feature_fusion=True, # Training settings
        learning_rate=5e-4, weight_decay=1e-6, # Scene bounds
        near_plane=0.1, far_plane=1000.0
    )
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = MegaNeRFPlusModel(config).to(device)
    
    print(f"Created Mega-NeRF+ model on {device}")
    print(f"Enhanced clusters: {config.num_clusters}")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):, }")
    
    return model, config


def enhanced_large_scale_example(data_path: str, output_dir: str):
    """Example for enhanced large-scale reconstruction."""
    print("=== Enhanced Large-Scale Mega-NeRF+ Training ===")
    
    config = MegaNeRFPlusConfig(
        # Advanced spatial decomposition
        num_clusters=32, cluster_method="hierarchical_adaptive", overlap_ratio=0.2, # Network architecture
        netdepth=8, netwidth=256, netdepth_fine=8, netwidth_fine=256, # Enhanced features
        use_attention_mechanism=True, use_feature_fusion=True, use_cross_cluster_attention=True, attention_heads=8, # Training settings
        learning_rate=1e-3, weight_decay=1e-6, scheduler_type="cosine_with_restarts", # Scene bounds
        near_plane=0.01, far_plane=10000.0, # Loss weights
        rgb_loss_weight=1.0, depth_loss_weight=0.1, cluster_consistency_weight=0.05, attention_regularization_weight=0.01, # Advanced optimization
        use_mixed_precision=True, gradient_accumulation_steps=4, max_clusters_per_batch=8
    )
    
    # Create dataset
    dataset = create_mega_nerf_plus_dataset(
        data_path=data_path, dataset_type="large_scale", config=config
    )
    
    train_loader = create_mega_nerf_plus_dataloader(
        dataset=dataset, split='train', batch_size=1, shuffle=True
    )
    
    val_loader = create_mega_nerf_plus_dataloader(
        dataset=dataset, split='val', batch_size=1, shuffle=False
    )
    
    # Create model and trainer
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = MegaNeRFPlusModel(config).to(device)
    trainer = MegaNeRFPlusTrainer(config)
    
    print(f"Training enhanced Mega-NeRF+ with {config.num_clusters} clusters")
    print(f"Using attention mechanism with {config.attention_heads} heads")
    print(f"Output directory: {output_dir}")
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Train model
    trainer.train(
        train_loader=train_loader, val_loader=val_loader, output_dir=output_dir, device=device
    )
    
    return model, trainer


def main():
    """Main function to run examples."""
    parser = argparse.ArgumentParser(description="Mega-NeRF+ Example Usage")
    parser.add_argument(
        "--example",
        type=str,
        default="basic",
        choices=["basic",
        "enhanced_large_scale"],
        help="Example to run",
    )
    parser.add_argument(
        "--data_path",
        type=str,
        default="./data/mega_nerf_plus/scene",
        help="Path to dataset",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./outputs/mega_nerf_plus",
        help="Output directory",
    )
    
    args = parser.parse_args()
    
    if args.example == "basic":
        model, config = basic_example()
        print("Basic example completed successfully!")
        
    elif args.example == "enhanced_large_scale":
        if not os.path.exists(args.data_path):
            print(f"Error: Data path {args.data_path} does not exist")
            return
        
        model, trainer = enhanced_large_scale_example(args.data_path, args.output_dir)
        print("Enhanced large-scale training completed!")


if __name__ == "__main__":
    main() 