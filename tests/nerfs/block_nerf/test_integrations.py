"""
Test Block NeRF Integration Components

This module tests the integration of Block NeRF components:
- End-to-end training pipeline
- Model saving and loading
- Dataset integration
"""

import pytest
import torch
import numpy as np
import tempfile
import os


# Add the src directory to the path for imports
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "..", "src"))

try:
    from nerfs.block_nerf import BlockNeRF
    from nerfs.block_nerf.trainer import BlockNeRFTrainer
    from nerfs.block_nerf.dataset import BlockNeRFDataset
    from nerfs.block_nerf.renderer import BlockNeRFRenderer

    BLOCK_NERF_AVAILABLE = True
except ImportError as e:
    BLOCK_NERF_AVAILABLE = False
    IMPORT_ERROR = str(e)

from pathlib import Path
from unittest.mock import patch, MagicMock

from src.nerfs.block_nerf import (
    BlockNeRFConfig,
    BlockNeRFModel,
    BlockNeRFTrainerConfig,
    BlockNeRFDatasetConfig,
    create_block_nerf_trainer,
    create_block_nerf_renderer,
    create_block_nerf_dataset,
    create_block_nerf_dataloader,
)
from . import (
    TEST_CONFIG,
    get_test_device,
    create_test_camera,
    skip_if_no_cuda,
    skip_if_slow,
)


class TestEndToEndWorkflow:
    """Test complete Block-NeRF workflow."""

    @pytest.fixture
    def mock_dataset_dir(self, tmp_path):
        """Create mock dataset directory."""
        import json

        data_dir = tmp_path / "integration_test_data"
        data_dir.mkdir()

        # Create transforms.json
        frames = []
        for i in range(8):  # Small dataset for testing
            frame = {
                "file_path": f"./images/frame_{i:03d}.png",
                "transform_matrix": np.eye(4).tolist(),
                "camera_angle_x": 0.8,
                "camera_id": i % 4,
                "exposure": 0.5 + 0.1 * np.sin(i * 0.5),
            }
            # Add some pose variation
            frame["transform_matrix"][0][3] = 2.0 * np.sin(i * 0.5)
            frame["transform_matrix"][2][3] = 2.0 * np.cos(i * 0.5)
            frames.append(frame)

        transforms = {
            "frames": frames,
            "camera_angle_x": 0.8,
            "scene_bounds": TEST_CONFIG["scene_bounds"],
        }

        with open(data_dir / "transforms.json", "w") as f:
            json.dump(transforms, f)

        # Create images directory
        images_dir = data_dir / "images"
        images_dir.mkdir()

        for i in range(8):
            (images_dir / f"frame_{i:03d}.png").touch()

        return data_dir

    @pytest.fixture
    def model_config(self):
        """Create model configuration for integration tests."""
        return BlockNeRFConfig(
            scene_bounds=TEST_CONFIG["scene_bounds"],
            block_size=TEST_CONFIG["block_size"],
            max_blocks=4,
            appearance_dim=16,
            hidden_dim=32,  # Small for fast testing
            num_layers=2,
            num_encoding_levels=6,
        )

    @pytest.fixture
    def trainer_config(self):
        """Create trainer configuration for integration tests."""
        return BlockNeRFTrainerConfig(
            num_epochs=2,
            learning_rate=1e-3,
            batch_size=2,
            num_rays=128,
            save_every=1,
            eval_every=1,
            use_amp=False,  # Disable for CPU testing
        )

    @pytest.fixture
    def dataset_config(self, mock_dataset_dir):
        """Create dataset configuration for integration tests."""
        return BlockNeRFDatasetConfig(
            data_dir=str(mock_dataset_dir),
            scene_bounds=TEST_CONFIG["scene_bounds"],
            block_size=TEST_CONFIG["block_size"],
            image_downscale=8,  # Very small images for testing
            num_rays=128,
            batch_size=2,
            val_split=0.5,
        )

    @pytest.fixture
    def renderer_config(self):
        """Create renderer configuration for integration tests."""
        return BlockNeRFRendererConfig(
            chunk_size=256,
            num_samples=16,
            num_importance_samples=8,
            perturb=0.0,  # Deterministic for testing
        )

    @skip_if_slow()
    def test_complete_training_workflow(
        self, model_config, trainer_config, dataset_config, tmp_path
    ):
        """Test complete training workflow."""
        # Mock image loading
        with patch("PIL.Image.open") as mock_open:
            mock_image = MagicMock()
            mock_image.size = (800, 600)
            mock_image.convert.return_value = mock_image
            mock_open.return_value = mock_image

            with patch("numpy.array") as mock_array:
                mock_array.return_value = np.random.rand(75, 100, 3)  # Downscaled image

                # Create dataset
                train_dataset = create_block_nerf_dataset(dataset_config, split="train")
                val_dataset = create_block_nerf_dataset(dataset_config, split="val")

                # Create data loaders
                train_loader = create_block_nerf_dataloader(
                    dataset=train_dataset, config=dataset_config, split="train"
                )
                val_loader = create_block_nerf_dataloader(
                    dataset=val_dataset, config=dataset_config, split="val"
                )

                # Create trainer
                trainer = create_block_nerf_trainer(
                    model_config=model_config,
                    trainer_config=trainer_config,
                    output_dir=str(tmp_path),
                )

                # Train for a few steps
                for epoch in range(trainer_config.num_epochs):
                    # Training epoch
                    trainer.model.train()
                    train_losses = []
                    for batch_idx, batch in enumerate(train_loader):
                        if batch_idx >= 2:  # Limit to 2 batches for testing
                            break

                        loss_dict = trainer.training_step(batch)
                        train_losses.append(loss_dict["total_loss"].item())

                    # Validation epoch
                    trainer.model.eval()
                    val_losses = []
                    with torch.no_grad():
                        for batch_idx, batch in enumerate(val_loader):
                            if batch_idx >= 1:  # Limit to 1 batch for testing
                                break

                            val_metrics = trainer.validation_step(batch)
                            val_losses.append(val_metrics["val_loss"].item())

                    # Check that losses are reasonable
                    assert all(loss >= 0 for loss in train_losses)
                    assert all(loss >= 0 for loss in val_losses)

                    # Save checkpoint
                    if (epoch + 1) % trainer_config.save_every == 0:
                        checkpoint_path = tmp_path / f"checkpoint_epoch_{epoch}.pth"
                        trainer.save_checkpoint(str(checkpoint_path), epoch)
                        assert checkpoint_path.exists()

    def test_training_to_inference_workflow(
        self, model_config, trainer_config, dataset_config, renderer_config, tmp_path
    ):
        """Test training followed by inference."""
        # Mock image loading
        with patch("PIL.Image.open") as mock_open:
            mock_image = MagicMock()
            mock_image.size = (800, 600)
            mock_image.convert.return_value = mock_image
            mock_open.return_value = mock_image

            with patch("numpy.array") as mock_array:
                mock_array.return_value = np.random.rand(75, 100, 3)

                # Create dataset and dataloader
                train_dataset = create_block_nerf_dataset(dataset_config, split="train")
                train_loader = create_block_nerf_dataloader(
                    dataset=train_dataset, config=dataset_config, split="train"
                )

                # Create and train model
                trainer = create_block_nerf_trainer(
                    model_config=model_config,
                    trainer_config=trainer_config,
                    output_dir=str(tmp_path),
                )

                # Quick training step
                batch = next(iter(train_loader))
                trainer.training_step(batch)

                # Save model
                model_path = tmp_path / "trained_model.pth"
                torch.save(
                    {
                        "model_state_dict": trainer.model.state_dict(),
                        "config": model_config.to_dict(),
                    },
                    model_path,
                )

                # Create renderer from trained model
                renderer = create_block_nerf_renderer(
                    model_config=model_config,
                    renderer_config=renderer_config,
                    checkpoint_path=str(model_path),
                )

                # Test inference
                camera_data = create_test_camera()

                with torch.no_grad():
                    rendered = renderer.render_image(
                        intrinsics=camera_data["intrinsics"],
                        pose=camera_data["pose"],
                        image_size=(32, 40),  # Small for testing
                        camera_id=torch.tensor(0, device=get_test_device()),
                        exposure=torch.tensor([0.5], device=get_test_device()),
                        chunk_size=256,
                    )

                assert "rgb" in rendered
                assert "depth" in rendered
                assert rendered["rgb"].shape == (32, 40, 3)
                assert rendered["depth"].shape == (32, 40, 1)

    def test_multi_block_training(self, model_config, trainer_config, dataset_config, tmp_path):
        """Test training with multiple blocks."""
        # Increase scene bounds to span multiple blocks
        model_config.scene_bounds = (-10, -10, -2, 10, 10, 2)
        model_config.block_size = 5.0
        model_config.max_blocks = 8

        dataset_config.scene_bounds = model_config.scene_bounds
        dataset_config.block_size = model_config.block_size

        # Mock image loading
        with patch("PIL.Image.open") as mock_open:
            mock_image = MagicMock()
            mock_image.size = (800, 600)
            mock_image.convert.return_value = mock_image
            mock_open.return_value = mock_image

            with patch("numpy.array") as mock_array:
                mock_array.return_value = np.random.rand(75, 100, 3)

                # Create dataset
                train_dataset = create_block_nerf_dataset(dataset_config, split="train")
                train_loader = create_block_nerf_dataloader(
                    dataset=train_dataset, config=dataset_config, split="train"
                )

                # Create trainer
                trainer = create_block_nerf_trainer(
                    model_config=model_config,
                    trainer_config=trainer_config,
                    output_dir=str(tmp_path),
                )

                # Train with multiple blocks
                batch = next(iter(train_loader))

                # Check that batch contains multiple blocks
                block_ids = batch["block_id"]
                if len(torch.unique(block_ids)) > 1:
                    # Multi-block batch
                    loss_dict = trainer.training_step(batch)
                    assert loss_dict["total_loss"] >= 0

    def test_appearance_embedding_learning(
        self, model_config, trainer_config, dataset_config, tmp_path
    ):
        """Test appearance embedding learning."""
        # Enable appearance embeddings
        model_config.appearance_dim = 32
        model_config.num_appearance_embeddings = 10

        # Mock image loading
        with patch("PIL.Image.open") as mock_open:
            mock_image = MagicMock()
            mock_image.size = (800, 600)
            mock_image.convert.return_value = mock_image
            mock_open.return_value = mock_image

            with patch("numpy.array") as mock_array:
                mock_array.return_value = np.random.rand(75, 100, 3)

                # Create dataset
                train_dataset = create_block_nerf_dataset(dataset_config, split="train")
                train_loader = create_block_nerf_dataloader(
                    dataset=train_dataset, config=dataset_config, split="train"
                )

                # Create trainer
                trainer = create_block_nerf_trainer(
                    model_config=model_config,
                    trainer_config=trainer_config,
                    output_dir=str(tmp_path),
                )

                # Check that model has appearance embeddings
                assert hasattr(trainer.model, "appearance_embedding")

                # Train step
                batch = next(iter(train_loader))
                initial_embeddings = trainer.model.appearance_embedding.weight.clone()

                trainer.training_step(batch)

                # Check that embeddings changed
                final_embeddings = trainer.model.appearance_embedding.weight
                assert not torch.equal(initial_embeddings, final_embeddings)

    def test_exposure_conditioning(self, model_config, trainer_config, dataset_config, tmp_path):
        """Test exposure conditioning."""
        # Enable exposure conditioning
        model_config.exposure_dim = 8

        # Mock image loading
        with patch("PIL.Image.open") as mock_open:
            mock_image = MagicMock()
            mock_image.size = (800, 600)
            mock_image.convert.return_value = mock_image
            mock_open.return_value = mock_image

            with patch("numpy.array") as mock_array:
                mock_array.return_value = np.random.rand(75, 100, 3)

                # Create dataset
                train_dataset = create_block_nerf_dataset(dataset_config, split="train")
                train_loader = create_block_nerf_dataloader(
                    dataset=train_dataset, config=dataset_config, split="train"
                )

                # Create trainer
                trainer = create_block_nerf_trainer(
                    model_config=model_config,
                    trainer_config=trainer_config,
                    output_dir=str(tmp_path),
                )

                # Check that model has exposure encoder
                assert hasattr(trainer.model, "exposure_encoder")

                # Train step
                batch = next(iter(train_loader))
                loss_dict = trainer.training_step(batch)

                assert loss_dict["total_loss"] >= 0

    @skip_if_no_cuda()
    def test_cuda_training_workflow(self, model_config, trainer_config, dataset_config, tmp_path):
        """Test CUDA training workflow."""
        # Enable mixed precision
        trainer_config.use_amp = True

        # Mock image loading
        with patch("PIL.Image.open") as mock_open:
            mock_image = MagicMock()
            mock_image.size = (800, 600)
            mock_image.convert.return_value = mock_image
            mock_open.return_value = mock_image

            with patch("numpy.array") as mock_array:
                mock_array.return_value = np.random.rand(75, 100, 3)

                # Create dataset
                train_dataset = create_block_nerf_dataset(dataset_config, split="train")
                train_loader = create_block_nerf_dataloader(
                    dataset=train_dataset, config=dataset_config, split="train"
                )

                # Create trainer
                trainer = create_block_nerf_trainer(
                    model_config=model_config,
                    trainer_config=trainer_config,
                    output_dir=str(tmp_path),
                )

                # Move to CUDA
                trainer.model = trainer.model.cuda()

                # Train step
                batch = next(iter(train_loader))

                # Move batch to CUDA
                batch = {
                    k: v.cuda() if isinstance(v, torch.Tensor) else v for k, v in batch.items()
                }

                loss_dict = trainer.training_step(batch)

                assert loss_dict["total_loss"].device.type == "cuda"
                assert loss_dict["total_loss"] >= 0

    def test_checkpoint_resumption(self, model_config, trainer_config, dataset_config, tmp_path):
        """Test checkpoint saving and resumption."""
        # Mock image loading
        with patch("PIL.Image.open") as mock_open:
            mock_image = MagicMock()
            mock_image.size = (800, 600)
            mock_image.convert.return_value = mock_image
            mock_open.return_value = mock_image

            with patch("numpy.array") as mock_array:
                mock_array.return_value = np.random.rand(75, 100, 3)

                # Create dataset
                train_dataset = create_block_nerf_dataset(dataset_config, split="train")
                train_loader = create_block_nerf_dataloader(
                    dataset=train_dataset, config=dataset_config, split="train"
                )

                # Create and train first trainer
                trainer1 = create_block_nerf_trainer(
                    model_config=model_config,
                    trainer_config=trainer_config,
                    output_dir=str(tmp_path),
                )

                # Train for a step
                batch = next(iter(train_loader))
                trainer1.training_step(batch)

                # Save checkpoint
                checkpoint_path = tmp_path / "checkpoint.pth"
                trainer1.save_checkpoint(str(checkpoint_path), epoch=0)

                # Create second trainer and load checkpoint
                trainer2 = create_block_nerf_trainer(
                    model_config=model_config,
                    trainer_config=trainer_config,
                    output_dir=str(tmp_path),
                )

                trainer2.load_checkpoint(str(checkpoint_path))

                # Check that parameters match
                for p1, p2 in zip(trainer1.model.parameters(), trainer2.model.parameters()):
                    assert torch.allclose(p1, p2, atol=1e-6)

    def test_evaluation_metrics(self, model_config, trainer_config, dataset_config, tmp_path):
        """Test evaluation metrics computation."""
        # Mock image loading
        with patch("PIL.Image.open") as mock_open:
            mock_image = MagicMock()
            mock_image.size = (800, 600)
            mock_image.convert.return_value = mock_image
            mock_open.return_value = mock_image

            with patch("numpy.array") as mock_array:
                mock_array.return_value = np.random.rand(75, 100, 3)

                # Create dataset
                val_dataset = create_block_nerf_dataset(dataset_config, split="val")
                val_loader = create_block_nerf_dataloader(
                    dataset=val_dataset, config=dataset_config, split="val"
                )

                # Create trainer
                trainer = create_block_nerf_trainer(
                    model_config=model_config,
                    trainer_config=trainer_config,
                    output_dir=str(tmp_path),
                )

                # Evaluate
                trainer.model.eval()
                with torch.no_grad():
                    batch = next(iter(val_loader))
                    metrics = trainer.validation_step(batch)

                # Check metrics
                assert "val_loss" in metrics
                assert "psnr" in metrics
                assert "ssim" in metrics
                assert "lpips" in metrics

                assert metrics["val_loss"] >= 0
                assert metrics["psnr"] >= 0
                assert 0 <= metrics["ssim"] <= 1
                assert metrics["lpips"] >= 0


class TestModelCompatibility:
    """Test model compatibility across different configurations."""

    def test_different_encoding_levels(self):
        """Test different encoding levels."""
        device = get_test_device()

        for levels in [4, 8, 12]:
            config = BlockNeRFConfig(
                scene_bounds=TEST_CONFIG["scene_bounds"],
                block_size=TEST_CONFIG["block_size"],
                num_encoding_levels=levels,
                hidden_dim=32,
                num_layers=2,
            )

            model = BlockNeRFModel(config).to(device)

            # Test forward pass
            batch_size = 2
            num_rays = 32

            positions = torch.randn(batch_size, num_rays, 3, device=device)
            directions = torch.randn(batch_size, num_rays, 3, device=device)
            camera_ids = torch.randint(0, 5, (batch_size,), device=device)
            exposure = torch.randn(batch_size, 1, device=device)

            with torch.no_grad():
                outputs = model(
                    positions=positions,
                    directions=directions,
                    camera_ids=camera_ids,
                    exposure=exposure,
                )

            assert torch.isfinite(outputs["density"]).all()
            assert torch.isfinite(outputs["color"]).all()

    def test_different_network_architectures(self):
        """Test different network architectures."""
        device = get_test_device()

        configs = [
            # Small network
            {"hidden_dim": 32, "num_layers": 2},
            # Medium network
            {"hidden_dim": 64, "num_layers": 4},
            # Network with skip connections
            {"hidden_dim": 64, "num_layers": 6, "skip_connections": [2, 4]},
        ]

        for config_params in configs:
            config = BlockNeRFConfig(
                scene_bounds=TEST_CONFIG["scene_bounds"],
                block_size=TEST_CONFIG["block_size"],
                **config_params,
            )

            model = BlockNeRFModel(config).to(device)

            # Test that model can be instantiated and run
            batch_size = 1
            num_rays = 16

            positions = torch.randn(batch_size, num_rays, 3, device=device)
            directions = torch.randn(batch_size, num_rays, 3, device=device)
            camera_ids = torch.randint(0, 5, (batch_size,), device=device)
            exposure = torch.randn(batch_size, 1, device=device)

            with torch.no_grad():
                outputs = model(
                    positions=positions,
                    directions=directions,
                    camera_ids=camera_ids,
                    exposure=exposure,
                )

            assert outputs["density"].shape == (batch_size, num_rays, 1)
            assert outputs["color"].shape == (batch_size, num_rays, 3)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
