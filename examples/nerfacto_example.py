"""
Nerfacto Example Usage

This script demonstrates how to use Nerfacto for neural radiance field training
with hash encoding and proposal networks.
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
    from src.nerfs.nerfacto import (
        NeRFactoConfig,
        NerfactoModel,
        NerfactoTrainer,
        create_nerfacto_dataloader,
    )
except ImportError as e:
    print(f"Import error: {e}")
    print("Please make sure you're running from the project root directory")
    sys.exit(1)


def basic_example():
    """Basic Nerfacto example with default settings."""
    print("=== Basic Nerfacto Example ===")

    # Create configuration with correct parameters
    config = NeRFactoConfig(
        # Model architecture
        num_layers=2,
        hidden_dim=64,
        geo_feat_dim=15,
        num_layers_color=3,
        hidden_dim_color=64,  # Hash encoding
        num_levels=16,
        base_resolution=16,
        max_resolution=2048,
        log2_hashmap_size=19,
        features_per_level=2,  # Proposal networks
        num_proposal_samples_per_ray=(256, 96),
        num_images=100,  # Adjust based on your dataset
        appearance_embed_dim=32,
        use_appearance_embedding=True,  # Background
        background_color="random",  # Loss weights
        distortion_loss_mult=0.002,
        interlevel_loss_mult=1.0,
        orientation_loss_mult=0.0001,
        pred_normal_loss_mult=0.001,  # Rendering
        near_plane=0.05,
        far_plane=1000.0,
        use_single_jitter=True,
        disable_scene_contraction=False,  # Training
        max_num_iterations=30000,
        proposal_net_args_list=[
            {
                "num_output_coords": 8,
                "num_levels": 5,
                "max_resolution": 128,
                "base_resolution": 16,
            }
        ],
    )

    print(f"Configuration: {config}")

    # Create scene bounding box (required for NerfactoModel)
    scene_box = torch.tensor([[-1, -1, -1], [1, 1, 1]], dtype=torch.float32)

    # Create model
    model = NerfactoModel(config, scene_box)
    print(f"Model created with {sum(p.numel() for p in model.parameters())} parameters")

    # Test forward pass with dummy ray bundle
    batch_size = 1024
    ray_bundle = {
        "origins": torch.randn(batch_size, 3),
        "directions": torch.randn(batch_size, 3),
        "camera_indices": torch.zeros(batch_size, 1, dtype=torch.long),  # All rays from camera 0
    }

    # Normalize directions
    ray_bundle["directions"] = ray_bundle["directions"] / torch.norm(
        ray_bundle["directions"], dim=-1, keepdim=True
    )

    with torch.no_grad():
        outputs = model(ray_bundle)
        print(f"Forward pass successful!")
        print(f"Output keys: {list(outputs.keys())}")
        if "rgb" in outputs:
            print(f"RGB shape: {outputs['rgb'].shape}")
            print(f"RGB range: [{outputs['rgb'].min():.4f}, {outputs['rgb'].max():.4f}]")
        if "accumulation" in outputs:
            print(f"Accumulation shape: {outputs['accumulation'].shape}")
            print(
                f"Accumulation range: [{outputs['accumulation'].min():.4f}, {outputs['accumulation'].max():.4f}]"
            )

    return model, config


def advanced_example():
    """Advanced Nerfacto example with custom settings."""
    print("\n=== Advanced Nerfacto Example ===")

    # Create configuration with custom parameters
    config = NeRFactoConfig(
        # Larger model architecture
        num_layers=4,
        hidden_dim=128,
        geo_feat_dim=31,
        num_layers_color=4,
        hidden_dim_color=128,  # Higher resolution hash encoding
        num_levels=20,
        base_resolution=32,
        max_resolution=4096,
        log2_hashmap_size=20,
        features_per_level=4,  # More proposal samples
        num_proposal_samples_per_ray=(
            512,
            192,
        ),
        num_images=1000,
        appearance_embed_dim=64,
        use_appearance_embedding=True,  # Different background
        background_color="white",  # Adjusted loss weights
        distortion_loss_mult=0.001,
        interlevel_loss_mult=2.0,
        orientation_loss_mult=0.0005,
        pred_normal_loss_mult=0.005,  # Extended rendering range
        near_plane=0.01,
        far_plane=2000.0,
        use_single_jitter=False,
        disable_scene_contraction=False,  # Longer training
        max_num_iterations=100000,
        proposal_net_args_list=[
            {
                "num_output_coords": 16,
                "num_levels": 8,
                "max_resolution": 256,
                "base_resolution": 32,
            }
        ],
    )

    print(f"Advanced configuration: {config}")

    # Create larger scene bounding box
    scene_box = torch.tensor([[-5, -5, -5], [5, 5, 5]], dtype=torch.float32)

    # Create model
    model = NerfactoModel(config, scene_box)
    print(f"Advanced model created with {sum(p.numel() for p in model.parameters())} parameters")

    return model, config


def training_example(data_path=None):
    """Example of training Nerfacto."""
    print("\n=== Training Example ===")

    if data_path is None:
        print("No data path provided, using synthetic data")
        return

    # Create configuration
    config = NeRFactoConfig(
        num_images=None,  # Will be set by dataset
        max_num_iterations=10000,
    )

    try:
        # Create data loader
        train_loader = create_nerfacto_dataloader(
            data_path, split="train", batch_size=4096, num_workers=4
        )

        # Update config with dataset info
        if hasattr(train_loader.dataset, "num_images"):
            config.num_images = train_loader.dataset.num_images

        # Create scene box from dataset
        scene_box = torch.tensor([[-2, -2, -2], [2, 2, 2]], dtype=torch.float32)

        # Create trainer
        trainer = NerfactoTrainer(config, scene_box)

        # Train for a few steps
        print("Starting training...")
        trainer.train(train_loader, num_epochs=1, max_steps=10, val_loader=None, test_loader=None)
        print("Training completed!")

    except Exception as e:
        print(f"Training failed: {e}")


def evaluation_example(model_path=None):
    """Example of evaluating Nerfacto."""
    print("\n=== Evaluation Example ===")

    if model_path is None:
        print("No model path provided, using basic model")
        model, config = basic_example()
    else:
        # Load trained model
        config = NeRFactoConfig()
        scene_box = torch.tensor([[-1, -1, -1], [1, 1, 1]], dtype=torch.float32)
        model = NerfactoModel(config, scene_box)
        model.load_state_dict(torch.load(model_path))
        print(f"Model loaded from {model_path}")

    # Evaluation metrics
    model.eval()
    with torch.no_grad():
        # Test on random ray bundle
        batch_size = 100
        ray_bundle = {
            "origins": torch.randn(
                batch_size,
                3,
            ),
            "directions": torch.randn(
                batch_size,
                3,
            ),
        }

        # Normalize directions
        ray_bundle["directions"] = ray_bundle["directions"] / torch.norm(
            ray_bundle["directions"], dim=-1, keepdim=True
        )

        outputs = model(ray_bundle)

        print(f"Evaluation completed!")
        if "rgb" in outputs:
            print(f"Average RGB: {outputs['rgb'].mean(dim=0)}")
        if "accumulation" in outputs:
            print(f"Average accumulation: {outputs['accumulation'].mean():.4f}")


def main():
    """Main function to run examples."""
    parser = argparse.ArgumentParser(description="Nerfacto Examples")
    parser.add_argument(
        "--example",
        type=str,
        default="basic",
        choices=["basic", "advanced", "training", "evaluation"],
        help="Type of example to run",
    )
    parser.add_argument("--data_path", type=str, help="Path to training data")
    parser.add_argument("--model_path", type=str, help="Path to trained model")
    parser.add_argument(
        "--output_dir",
        type=str,
        default="outputs/nerfacto",
        help="Output directory",
    )

    args = parser.parse_args()

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    print("🚀 Nerfacto Example Runner")
    print(f"Example type: {args.example}")
    print(f"Output directory: {args.output_dir}")

    if args.example == "basic":
        model, config = basic_example()
    elif args.example == "advanced":
        model, config = advanced_example()
    elif args.example == "training":
        training_example(args.data_path)
    elif args.example == "evaluation":
        evaluation_example(args.model_path)

    print("\n✅ Example completed successfully!")


if __name__ == "__main__":
    main()
