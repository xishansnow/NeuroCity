from __future__ import annotations

"""
Plenoxels Trainer Module

This module implements the trainer for Plenoxels with coarse-to-fine optimization
and sparsity regularization.

Key features:
- Coarse-to-fine voxel grid training
- Sparsity regularization with pruning
- Total variation loss for smoothness
- Progressive resolution increases
- Efficient ray batching
"""

from typing import Any, Optional


import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from dataclasses import dataclass
import logging
from tqdm import tqdm
import wandb
from pathlib import Path
import shutil

from .core import PlenoxelModel, PlenoxelConfig, PlenoxelLoss
from .dataset import PlenoxelDataset, PlenoxelDatasetConfig, create_plenoxel_dataloader

logger = logging.getLogger(__name__)

__all__ = ["PlenoxelTrainerConfig", "PlenoxelTrainer", "create_plenoxel_trainer"]


@dataclass
class PlenoxelTrainerConfig:
    """Configuration for Plenoxel trainer."""

    # Training settings
    max_epochs: int = 10000
    learning_rate: float = 0.1
    weight_decay: float = 0.0

    # Coarse-to-fine settings
    coarse_to_fine: bool = True
    resolution_schedule: List[Tuple[int, int, int]] = None
    resolution_epochs: List[int] = None

    # Loss weights
    color_loss_weight: float = 1.0
    tv_loss_weight: float = 1e-6
    l1_loss_weight: float = 1e-8

    # Sparsity and pruning
    pruning_threshold: float = 0.01
    pruning_interval: int = 1000

    # Evaluation
    eval_interval: int = 1000
    save_interval: int = 5000

    # Logging
    log_interval: int = 100
    use_wandb: bool = False
    use_tensorboard: bool = True

    # Optimization
    use_adam: bool = False
    adam_beta1: float = 0.9
    adam_beta2: float = 0.999
    lr_decay: float = 0.1
    lr_decay_steps: List[int] = None

    # Output
    experiment_name: str = "plenoxel_experiment"
    output_dir: str = "outputs"
    resume_from: Optional[str] = None


class PlenoxelTrainer:
    """Main trainer class for Plenoxels."""

    def __init__(
        self,
        model_config: PlenoxelConfig,
        trainer_config: PlenoxelTrainerConfig,
        dataset_config: PlenoxelDatasetConfig,
    ) -> None:
        """
        Initialize Plenoxel trainer.

        Args:
            model_config: Model configuration
            trainer_config: Trainer configuration
            dataset_config: Dataset configuration
        """
        self.model_config = model_config
        self.trainer_config = trainer_config
        self.dataset_config = dataset_config

        # Setup device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {self.device}")

        # Initialize model
        self.model = PlenoxelModel(model_config).to(self.device)
        self.loss_fn = PlenoxelLoss(model_config)

        # Setup optimizer
        self._setup_optimizer()

        # Setup datasets
        self._setup_datasets()

        # Setup logging
        self._setup_logging()

        # Training state
        self.epoch = 0
        self.step = 0
        self.best_psnr = 0.0
        self.current_resolution_level = 0

        # Resume from checkpoint if specified
        if trainer_config.resume_from:
            self.load_checkpoint(trainer_config.resume_from)

    def _setup_optimizer(self):
        """Setup optimizer and learning rate scheduler."""
        if self.trainer_config.use_adam:
            self.optimizer = optim.Adam(self.model.parameters())
        else:
            # Use SGD for Plenoxels (as recommended in paper)
            self.optimizer = optim.SGD(self.model.parameters())

        # Learning rate scheduler
        if self.trainer_config.lr_decay_steps:
            self.lr_scheduler = optim.lr_scheduler.MultiStepLR(
                self.optimizer,
                milestones=self.trainer_config.lr_decay_steps,
                gamma=self.trainer_config.lr_decay,
            )
        else:
            self.lr_scheduler = None

    def _setup_datasets(self):
        """Setup training and validation datasets."""
        # Training dataset
        self.train_dataset = PlenoxelDataset(self.dataset_config, split="train")
        self.train_loader = create_plenoxel_dataloader(
            self.dataset_config, split="train", shuffle=True
        )

        # Validation dataset
        self.val_dataset = PlenoxelDataset(self.dataset_config, split="val")
        self.val_loader = create_plenoxel_dataloader(
            self.dataset_config, split="val", shuffle=False
        )

        logger.info(f"Training dataset: {len(self.train_dataset)} samples")
        logger.info(f"Validation dataset: {len(self.val_dataset)} samples")

    def _setup_logging(self):
        """Setup logging with TensorBoard and optionally W&B."""
        # Create output directory
        self.exp_dir = os.path.join(
            self.trainer_config.output_dir, self.trainer_config.experiment_name
        )
        os.makedirs(self.exp_dir, exist_ok=True)

        # TensorBoard
        if self.trainer_config.use_tensorboard:
            self.tb_writer = SummaryWriter(log_dir=os.path.join(self.exp_dir, "tensorboard"))
        else:
            self.tb_writer = None

        # Weights & Biases
        if self.trainer_config.use_wandb:
            wandb.init(
                project="plenoxels",
                name=self.trainer_config.experiment_name,
                config={
                    **self.model_config.__dict__,
                    **self.trainer_config.__dict__,
                    **self.dataset_config.__dict__,
                },
            )

        logger.info(f"Experiment directory: {self.exp_dir}")

    def train(self, dataloader=None, num_epochs=None):
        """Main training loop."""
        logger.info("Starting Plenoxel training...")

        start_time = time.time()

        for epoch in range(self.epoch, self.trainer_config.max_epochs):
            self.epoch = epoch

            # Update resolution for coarse-to-fine training
            self._update_resolution()

            # Training epoch
            train_metrics = self._train_epoch()

            # Learning rate scheduling
            if self.lr_scheduler:
                self.lr_scheduler.step()

            # Validation
            if epoch % self.trainer_config.eval_interval == 0:
                val_metrics = self.validate()

                # Update best PSNR
                if val_metrics["val_psnr"] > self.best_psnr:
                    self.best_psnr = val_metrics["val_psnr"]
                    self._save_checkpoint("best.pth")

            # Pruning
            if epoch % self.trainer_config.pruning_interval == 0 and epoch > 0:
                self._prune_voxels()

            # Logging
            if epoch % self.trainer_config.log_interval == 0:
                self._log_metrics(train_metrics, val_metrics if "val_metrics" in locals() else None)

            # Save checkpoint
            if epoch % self.trainer_config.save_interval == 0:
                self._save_checkpoint(f"epoch_{epoch}.pth")

        # Final save
        self._save_checkpoint("final.pth")

        total_time = time.time() - start_time
        logger.info(f"Training completed in {total_time:.2f} seconds")

        # Cleanup
        if self.tb_writer:
            self.tb_writer.close()
        if self.trainer_config.use_wandb:
            wandb.finish()

    def _train_epoch(self, dataloader=None) -> Dict[str, float]:
        """Train for one epoch.

        Args:
            dataloader: Optional dataloader to use for training. If not provided, uses self.train_loader.
        """
        self.model.train()

        total_loss = 0.0
        total_color_loss = 0.0
        total_tv_loss = 0.0
        total_l1_loss = 0.0
        num_batches = 0

        # Use provided dataloader or default to self.train_loader
        train_loader = dataloader if dataloader is not None else self.train_loader
        pbar = tqdm(train_loader, desc=f"Epoch {self.epoch}")

        for batch_idx, batch in enumerate(pbar):
            # Move batch to device
            batch = {
                k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()
            }

            # Forward pass
            rays_o = batch["rays_o"].squeeze(0)  # Remove batch dimension
            rays_d = batch["rays_d"].squeeze(0)
            target_colors = batch["colors"].squeeze(0)

            outputs = self.model(rays_o, rays_d)

            # Compute losses
            losses = self._compute_losses(outputs, {"colors": target_colors})

            # Compute total loss
            total_loss_batch = (
                self.trainer_config.color_loss_weight * losses["color_loss"]
                + self.trainer_config.tv_loss_weight * self.model.voxel_grid.total_variation_loss()
                + self.trainer_config.l1_loss_weight * self.model.voxel_grid.l1_loss()
            )

            # Backward pass
            self.optimizer.zero_grad()
            total_loss_batch.backward()
            self.optimizer.step()

            # Accumulate metrics
            total_loss += total_loss_batch.item()
            total_color_loss += losses["color_loss"].item()
            total_tv_loss += self.model.voxel_grid.total_variation_loss().item()
            total_l1_loss += self.model.voxel_grid.l1_loss().item()
            num_batches += 1

            # Update progress bar
            pbar.set_postfix({"loss": total_loss_batch.item()})

            self.step += 1

        return {
            "total_loss": total_loss / num_batches,
            "color_loss": total_color_loss / num_batches,
            "tv_loss": total_tv_loss / num_batches,
            "l1_loss": total_l1_loss / num_batches,
        }

    def validate(self, dataloader=None) -> Dict[str, float]:
        """Validate on validation set."""
        self.model.eval()

        total_loss = 0.0
        total_psnr = 0.0
        num_samples = 0

        with torch.no_grad():
            # Use provided dataloader or default to self.val_loader
            val_loader = dataloader if dataloader is not None else self.val_loader
            for batch in tqdm(val_loader, desc="Validation"):
                # Move batch to device
                if isinstance(batch, dict):
                    batch = {
                        k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                        for k, v in batch.items()
                    }
                    rays_o = batch["rays_o"].squeeze(0)  # [H, W, 3]
                    rays_d = batch["rays_d"].squeeze(0)  # [H, W, 3]
                    target_colors = batch["colors"].squeeze(0)  # [H, W, 3]
                else:
                    # Handle test dataset format (TensorDataset)
                    rays_o, rays_d, target_colors = [x.to(self.device) for x in batch]

                # Get batch shape
                batch_size = rays_o.shape[0]

                # Render in chunks to avoid memory issues
                chunk_size = 1024
                rgb_chunks = []

                for i in range(0, batch_size, chunk_size):
                    rays_o_chunk = rays_o[i : i + chunk_size]
                    rays_d_chunk = rays_d[i : i + chunk_size]

                    outputs_chunk = self.model(rays_o_chunk, rays_d_chunk)
                    rgb_chunks.append(outputs_chunk["rgb"])

                # Combine chunks
                rgb_pred = torch.cat(rgb_chunks, dim=0)

                # Compute metrics
                mse = torch.mean((rgb_pred - target_colors) ** 2)
                psnr = -10.0 * torch.log10(mse)

                total_loss += mse.item()
                total_psnr += psnr.item()
                num_samples += 1

        return {"val_loss": total_loss / num_samples, "val_psnr": total_psnr / num_samples}

    def _compute_losses(
        self,
        outputs: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        """Compute training losses."""
        losses = {}

        # Color reconstruction loss
        if "rgb" in outputs and "colors" in targets:
            losses["color_loss"] = torch.mean((outputs["rgb"] - targets["colors"]) ** 2)

        return losses

    def _update_resolution(self):
        """Update voxel grid resolution for coarse-to-fine training."""
        if not self.trainer_config.coarse_to_fine:
            return

        # Default resolution schedule if not provided
        if self.trainer_config.resolution_schedule is None:
            schedule = self.model_config.coarse_resolutions
            epochs = self.model_config.coarse_epochs
        else:
            schedule = self.trainer_config.resolution_schedule
            epochs = self.trainer_config.resolution_epochs

        # Check if we need to update resolution
        for i, epoch_threshold in enumerate(epochs):
            if self.epoch >= epoch_threshold and self.current_resolution_level < len(schedule):
                if self.current_resolution_level <= i:
                    new_resolution = schedule[i]
                    self.model.update_resolution(new_resolution)
                    self.current_resolution_level = i + 1
                    logger.info(f"Updated resolution to {new_resolution} at epoch {self.epoch}")
                    break

    def _prune_voxels(self):
        """Prune low-density voxels."""
        before_stats = self.model.get_occupancy_stats()
        self.model.prune_voxels(self.trainer_config.pruning_threshold)
        after_stats = self.model.get_occupancy_stats()

        logger.info(
            f"Pruning: {before_stats['occupied_voxels']} -> {after_stats['occupied_voxels']}"
        )

    def _log_metrics(
        self,
        train_metrics: Dict[str, float],
        val_metrics: Optional[Dict[str, float]] = None,
    ) -> None:
        """Log training and validation metrics."""
        # TensorBoard logging
        if self.tb_writer:
            for key, value in train_metrics.items():
                self.tb_writer.add_scalar(f"train/{key}", value, self.epoch)

            if val_metrics:
                for key, value in val_metrics.items():
                    self.tb_writer.add_scalar(f"val/{key}", value, self.epoch)

            # Log occupancy statistics
            occupancy_stats = self.model.get_occupancy_stats()
            for key, value in occupancy_stats.items():
                self.tb_writer.add_scalar(f"occupancy/{key}", value, self.epoch)

        # W&B logging
        if self.trainer_config.use_wandb:
            log_dict = {f"train/{k}": v for k, v in train_metrics.items()}
            if val_metrics:
                log_dict.update({f"val/{k}": v for k, v in val_metrics.items()})

            occupancy_stats = self.model.get_occupancy_stats()
            log_dict.update({f"occupancy/{k}": v for k, v in occupancy_stats.items()})

            wandb.log(log_dict, step=self.epoch)

        # Console logging
        train_str = " | ".join([f"{k}: {v:.6f}" for k, v in train_metrics.items()])
        logger.info(f"Epoch {self.epoch} | Train: {train_str}")

        if val_metrics:
            val_str = " | ".join([f"{k}: {v:.6f}" for k, v in val_metrics.items()])
            logger.info(f"Epoch {self.epoch} | Val: {val_str}")

    def _save_checkpoint(self, filename: str):
        """Save model checkpoint."""
        checkpoint = {
            "epoch": self.epoch,
            "step": self.step,
            "model_state": self.model.state_dict(),
            "optimizer_state": self.optimizer.state_dict(),
            "best_psnr": self.best_psnr,
            "current_resolution_level": self.current_resolution_level,
        }

        if self.lr_scheduler is not None:
            checkpoint["scheduler_state"] = self.lr_scheduler.state_dict()

        filepath = os.path.join(self.exp_dir, filename)
        torch.save(checkpoint, filepath)
        logger.info(f"Saved checkpoint to {filepath}")

    def load_checkpoint(self, checkpoint_path: str | Path) -> None:
        """Load checkpoint with improved error handling.

        Args:
            checkpoint_path: Path to checkpoint file

        Raises:
            FileNotFoundError: If checkpoint file does not exist
            RuntimeError: If checkpoint is incompatible
        """
        checkpoint_path = Path(checkpoint_path)
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

        logger.info(f"Loading checkpoint from {checkpoint_path}")

        try:
            # Load checkpoint to CPU first to avoid GPU OOM
            # Set weights_only=False since we need to load custom classes
            checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

            # Validate checkpoint contents
            required_keys = {
                "model_state",
                "optimizer_state",
                "epoch",
                "step",
                "best_psnr",
                "current_resolution_level",
            }
            missing_keys = required_keys - set(checkpoint.keys())
            if missing_keys:
                raise RuntimeError(f"Checkpoint missing required keys: {missing_keys}")

            # Load model state
            try:
                # Move model state to device before loading
                model_state = {k: v.to(self.device) for k, v in checkpoint["model_state"].items()}
                self.model.load_state_dict(model_state)
            except Exception as e:
                raise RuntimeError(f"Failed to load model state: {e}")

            # Load optimizer state
            try:
                self.optimizer.load_state_dict(checkpoint["optimizer_state"])
                # Move optimizer state to device
                for state in self.optimizer.state.values():
                    for k, v in state.items():
                        if isinstance(v, torch.Tensor):
                            state[k] = v.to(self.device)
            except Exception as e:
                raise RuntimeError(f"Failed to load optimizer state: {e}")

            # Load scheduler state if available
            if self.lr_scheduler is not None and "scheduler_state" in checkpoint:
                try:
                    self.lr_scheduler.load_state_dict(checkpoint["scheduler_state"])
                except Exception as e:
                    logger.warning(f"Failed to load scheduler state: {e}")

            # Load training state
            self.epoch = checkpoint["epoch"]
            self.step = checkpoint["step"]
            self.best_psnr = checkpoint["best_psnr"]
            self.current_resolution_level = checkpoint["current_resolution_level"]

            logger.info(
                f"Successfully loaded checkpoint from step {self.step} "
                f"(best PSNR: {self.best_psnr:.2f})"
            )

        except Exception as e:
            raise RuntimeError(f"Failed to load checkpoint: {e}")

    def render_novel_views(self, poses: np.ndarray, output_dir: str = None) -> List[np.ndarray]:
        """Render novel views given camera poses."""
        self.model.eval()

        if output_dir is None:
            output_dir = os.path.join(self.exp_dir, "novel_views")
        os.makedirs(output_dir, exist_ok=True)

        rendered_images = []

        with torch.no_grad():
            for i, pose in enumerate(tqdm(poses, desc="Rendering novel views")):
                # Generate rays for this pose
                pose_tensor = torch.from_numpy(pose).float().to(self.device)

                # Use validation dataset parameters
                H, W = self.val_dataset.H, self.val_dataset.W
                focal = self.val_dataset.focal

                # Generate rays
                i_coords, j_coords = np.meshgrid(
                    np.arange(W, dtype=np.float32), np.arange(H, dtype=np.float32), indexing="xy"
                )

                dirs = np.stack([(i_coords - W * 0.5,)], -1)

                rays_d = torch.from_numpy(dirs @ pose[:3, :3].T).float().to(self.device)
                rays_o = torch.from_numpy(
                    np.broadcast_to,
                )

                # Render in chunks
                chunk_size = 1024
                rgb_chunks = []

                rays_o_flat = rays_o.view(-1, 3)
                rays_d_flat = rays_d.view(-1, 3)

                for j in range(0, len(rays_o_flat), chunk_size):
                    rays_o_chunk = rays_o_flat[j : j + chunk_size]
                    rays_d_chunk = rays_d_flat[j : j + chunk_size]

                    outputs = self.model(rays_o_chunk, rays_d_chunk)
                    rgb_chunks.append(outputs["rgb"].cpu())

                # Combine and reshape
                rgb = torch.cat(rgb_chunks, dim=0).view(H, W, 3)
                rgb_np = rgb.numpy()
                rgb_np = np.clip(rgb_np, 0, 1)

                rendered_images.append(rgb_np)

                # Save image
                rgb_uint8 = (rgb_np * 255).astype(np.uint8)
                import imageio

                imageio.imwrite(os.path.join(output_dir, f"render_{i:03d}.png"), rgb_uint8)

        return rendered_images

    def save_checkpoint(self, checkpoint_path: str | Path, is_best: bool = False) -> None:
        """Save checkpoint with improved error handling.

        Args:
            checkpoint_path: Path to save checkpoint
            is_best: Whether this is the best checkpoint so far
        """
        checkpoint_path = Path(checkpoint_path)
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

        # Prepare checkpoint contents
        checkpoint = {
            "model_state": self.model.state_dict(),
            "optimizer_state": self.optimizer.state_dict(),
            "scheduler_state": (
                self.lr_scheduler.state_dict() if self.lr_scheduler is not None else None
            ),
            "config": self.model_config,
            "epoch": self.epoch,
            "step": self.step,
            "best_psnr": self.best_psnr,
            "current_resolution_level": self.current_resolution_level,
        }

        # Save checkpoint
        try:
            torch.save(checkpoint, checkpoint_path)
            logger.info(f"Saved checkpoint to {checkpoint_path}")

            if is_best:
                best_path = checkpoint_path.parent / "best.pt"
                shutil.copy(checkpoint_path, best_path)
                logger.info(f"Saved best checkpoint to {best_path}")

        except Exception as e:
            logger.error(f"Failed to save checkpoint: {e}")
            raise


def create_plenoxel_trainer(
    model_config: PlenoxelConfig,
    trainer_config: PlenoxelTrainerConfig,
    dataset_config: PlenoxelDatasetConfig,
) -> PlenoxelTrainer:
    """Create a Plenoxel trainer."""
    return PlenoxelTrainer(model_config, trainer_config, dataset_config)
