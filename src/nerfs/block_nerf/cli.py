"""
Command Line Interface for Block-NeRF

Provides command-line tools for training and rendering with Block-NeRF.
"""

import argparse
import sys
import torch
from pathlib import Path
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def train_cli():
    """CLI for training Block-NeRF models."""
    parser = argparse.ArgumentParser(description="Train Block-NeRF model")

    # Data arguments
    parser.add_argument(
        "--data-dir", type=str, required=True, help="Path to training data directory"
    )
    parser.add_argument(
        "--dataset-type",
        type=str,
        default="waymo",
        choices=["waymo", "colmap", "llff"],
        help="Dataset type",
    )

    # Model arguments
    parser.add_argument("--num-blocks", type=int, default=8, help="Number of blocks")
    parser.add_argument("--block-size", type=float, default=50.0, help="Block size")
    parser.add_argument("--hidden-dim", type=int, default=256, help="Hidden dimension")
    parser.add_argument("--num-layers", type=int, default=8, help="Number of layers")
    parser.add_argument("--num-samples", type=int, default=64, help="Number of samples per ray")
    parser.add_argument("--near", type=float, default=0.1, help="Near plane")
    parser.add_argument("--far", type=float, default=10.0, help="Far plane")

    # Training arguments
    parser.add_argument("--num-epochs", type=int, default=100, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=1024, help="Ray batch size")
    parser.add_argument("--learning-rate", type=float, default=5e-4, help="Learning rate")
    parser.add_argument("--log-interval", type=int, default=100, help="Logging interval")

    # Output arguments
    parser.add_argument("--output-dir", type=str, default="./outputs", help="Output directory")
    parser.add_argument(
        "--checkpoint-dir", type=str, default="./checkpoints", help="Checkpoint directory"
    )

    # Device arguments
    parser.add_argument(
        "--device", type=str, default="auto", choices=["auto", "cuda", "cpu"], help="Device to use"
    )

    args = parser.parse_args()

    # Import here to avoid circular imports
    try:
        from ..block_nerf import (
            BlockNeRFConfig,
            BlockNeRF,
            BlockNeRFTrainer,
            BlockNeRFTrainerConfig,
        )
        from ..block_nerf.dataset import BlockNeRFDataset, BlockNeRFDatasetConfig
    except ImportError as e:
        logger.error(f"Failed to import block_nerf: {e}")
        sys.exit(1)

    # Setup device
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    logger.info(f"Using device: {device}")

    # Create model configuration
    model_config = BlockNeRFConfig(
        num_blocks=args.num_blocks,
        block_size=args.block_size,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        num_samples=args.num_samples,
        near=args.near,
        far=args.far,
    )

    # Create model
    model = BlockNeRF(model_config)
    logger.info(f"Model created with {sum(p.numel() for p in model.parameters()):,} parameters")

    # Create trainer configuration
    trainer_config = BlockNeRFTrainerConfig(
        num_epochs=args.num_epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        log_interval=args.log_interval,
        checkpoint_dir=args.checkpoint_dir,
    )

    # Create trainer
    trainer = BlockNeRFTrainer(model, trainer_config)
    trainer = trainer.to(device)

    # Create dataset
    try:
        dataset_config = BlockNeRFDatasetConfig(
            data_root=args.data_dir,
            split="train",
        )

        train_dataset = BlockNeRFDataset(dataset_config)
        val_dataset = BlockNeRFDataset(BlockNeRFDatasetConfig(data_root=args.data_dir, split="val"))

        from torch.utils.data import DataLoader

        train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)

        # Start training
        logger.info("Starting training...")
        trainer.train(train_loader, val_loader)

        logger.info("Training completed successfully!")

    except Exception as e:
        logger.error(f"Training failed: {e}")
        sys.exit(1)


def render_cli():
    """CLI for rendering with trained Block-NeRF models."""
    parser = argparse.ArgumentParser(description="Render with Block-NeRF model")

    # Model arguments
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--config", type=str, help="Path to model config (if separate)")

    # Rendering arguments
    parser.add_argument(
        "--output-dir", type=str, default="./renders", help="Output directory for renders"
    )
    parser.add_argument("--width", type=int, default=512, help="Render width")
    parser.add_argument("--height", type=int, default=512, help="Render height")
    parser.add_argument("--num-samples", type=int, default=64, help="Number of samples per ray")

    # Camera arguments
    parser.add_argument("--camera-poses", type=str, help="Path to camera poses file")
    parser.add_argument("--render-video", action="store_true", help="Render video sequence")
    parser.add_argument("--fps", type=int, default=30, help="Video frame rate")

    # Device arguments
    parser.add_argument(
        "--device", type=str, default="auto", choices=["auto", "cuda", "cpu"], help="Device to use"
    )

    args = parser.parse_args()

    # Import here to avoid circular imports
    try:
        from ..block_nerf import (
            BlockNeRF,
            BlockNeRFRenderer,
            BlockNeRFRendererConfig,
        )
    except ImportError as e:
        logger.error(f"Failed to import block_nerf: {e}")
        sys.exit(1)

    # Setup device
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    logger.info(f"Using device: {device}")

    # Load model
    try:
        checkpoint = torch.load(args.checkpoint, map_location=device)

        # Create model
        model_config = checkpoint.get("model_config")
        if model_config is None:
            logger.error("Model config not found in checkpoint")
            sys.exit(1)

        model = BlockNeRF(model_config)
        model.load_state_dict(checkpoint["model_state_dict"])
        model.to(device)

        logger.info("Model loaded successfully")

    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        sys.exit(1)

    # Create renderer
    renderer_config = BlockNeRFRendererConfig(
        image_width=args.width,
        image_height=args.height,
        num_samples=args.num_samples,
        output_dir=args.output_dir,
    )

    renderer = BlockNeRFRenderer(model, renderer_config)
    renderer = renderer.to(device)

    # Create output directory
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    if args.render_video and args.camera_poses:
        # Render video
        import json

        with open(args.camera_poses, "r") as f:
            poses_data = json.load(f)

        # Convert poses and render video
        poses = [torch.tensor(pose, device=device) for pose in poses_data["poses"]]
        intrinsics = torch.tensor(poses_data["intrinsics"], device=device)

        output_path = Path(args.output_dir) / "rendered_video.mp4"

        try:
            renderer.render_video(poses, intrinsics, output_path)
            logger.info(f"Video rendered to {output_path}")

        except Exception as e:
            logger.error(f"Video rendering failed: {e}")
            sys.exit(1)

    else:
        # Render single image with default camera
        import numpy as np

        # Create default camera pose
        camera_pose = torch.eye(4, device=device)
        camera_pose[2, 3] = 3.0  # Move camera back

        # Create intrinsics
        focal = args.width * 0.8
        intrinsics = torch.tensor(
            [[focal, 0, args.width / 2], [0, focal, args.height / 2], [0, 0, 1]],
            device=device,
            dtype=torch.float32,
        )

        try:
            image = renderer.render_image(camera_pose, intrinsics)

            # Save image
            import imageio

            rgb_image = image.cpu().numpy()
            rgb_image = (rgb_image * 255).astype(np.uint8)

            output_path = Path(args.output_dir) / "rendered_image.png"
            imageio.imsave(output_path, rgb_image)

            logger.info(f"Image rendered to {output_path}")

        except Exception as e:
            logger.error(f"Image rendering failed: {e}")
            sys.exit(1)


if __name__ == "__main__":
    # This allows the module to be run directly
    if len(sys.argv) > 1:
        if sys.argv[1] == "train":
            sys.argv = [sys.argv[0]] + sys.argv[2:]  # Remove 'train' from args
            train_cli()
        elif sys.argv[1] == "render":
            sys.argv = [sys.argv[0]] + sys.argv[2:]  # Remove 'render' from args
            render_cli()
        else:
            print("Usage: python -m block_nerf.cli [train|render] [args...]")
            sys.exit(1)
    else:
        print("Usage: python -m block_nerf.cli [train|render] [args...]")
        sys.exit(1)
