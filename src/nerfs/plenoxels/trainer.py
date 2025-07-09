"""
Plenoxels Refactored Trainer - Training-Only Implementation

This module implements a dedicated trainer class for Plenoxels that focuses
exclusively on training functionality, following the same pattern as SVRaster.
"""

from __future__ import annotations

import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from torch.amp import autocast, GradScaler
import numpy as np
import logging
from tqdm import tqdm
from pathlib import Path

try:
    import wandb

    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False

from .config import PlenoxelTrainingConfig
from .core import PlenoxelModel, PlenoxelLoss, VoxelGrid, VolumetricRenderer
from .dataset import PlenoxelDataset, create_plenoxel_dataloader
from .renderer import PlenoxelRenderer

logger = logging.getLogger(__name__)


class PlenoxelTrainer:
    """Dedicated trainer class for Plenoxels.

    This class handles all training-related functionality including:
    - Progressive resolution training (coarse-to-fine)
    - Voxel pruning and sparsification
    - Loss computation and optimization
    - Training monitoring and logging
    - Checkpoint management

    After training, it can export a PlenoxelRenderer for inference.
    """

    def __init__(
        self,
        config: PlenoxelTrainingConfig,
        train_dataset: PlenoxelDataset | None = None,
        val_dataset: PlenoxelDataset | None = None,
    ):
        """Initialize the Plenoxel trainer.

        Args:
            config: Training configuration
            train_dataset: Optional training dataset
            val_dataset: Optional validation dataset
        """
        self.config = config
        self.device = config.device

        # Initialize model components
        self.voxel_grid = VoxelGrid(
            resolution=config.grid_resolution,
            scene_bounds=config.scene_bounds,
            num_sh_coeffs=(config.sh_degree + 1) ** 2,
            device=config.device,
        )

        self.volumetric_renderer = VolumetricRenderer(config)
        self.loss_fn = PlenoxelLoss(config)

        # Setup optimizer and scheduler
        self._setup_optimizer()

        # Setup datasets if provided
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.train_loader = None
        self.val_loader = None

        if train_dataset:
            self.train_loader = create_plenoxel_dataloader(
                train_dataset, split="train", shuffle=True
            )
        if val_dataset:
            self.val_loader = create_plenoxel_dataloader(val_dataset, split="val", shuffle=False)

        # Training state
        self.epoch = 0
        self.step = 0
        self.best_psnr = 0.0
        self.current_resolution_level = 0

        # Setup logging
        self._setup_logging()

        # Mixed precision training
        self.scaler = GradScaler()
        self.use_amp = config.use_cuda and torch.cuda.is_available()

        logger.info(f"PlenoxelTrainer initialized with device: {self.device}")
        logger.info(f"Mixed precision training: {self.use_amp}")

    def _setup_optimizer(self):
        """Setup optimizer and learning rate scheduler."""
        params = [
            {"params": self.voxel_grid.parameters(), "lr": self.config.learning_rate},
        ]

        if self.config.use_adam:
            self.optimizer = optim.Adam(params, weight_decay=self.config.weight_decay)
        else:
            # SGD is recommended for Plenoxels in the original paper
            self.optimizer = optim.SGD(params, weight_decay=self.config.weight_decay)

        # Learning rate scheduler
        if self.config.lr_decay_steps:
            self.lr_scheduler = optim.lr_scheduler.MultiStepLR(
                self.optimizer,
                milestones=self.config.lr_decay_steps,
                gamma=self.config.lr_decay_gamma,
            )
        else:
            self.lr_scheduler = None

    def _setup_logging(self):
        """Setup logging infrastructure."""
        # Create experiment directory
        self.exp_dir = Path(self.config.checkpoint_dir) / self.config.experiment_name
        self.exp_dir.mkdir(parents=True, exist_ok=True)

        # TensorBoard
        if self.config.use_tensorboard:
            self.tb_writer = SummaryWriter(log_dir=self.exp_dir / "tensorboard")
        else:
            self.tb_writer = None

        # Weights & Biases
        if self.config.use_wandb and WANDB_AVAILABLE:
            wandb.init(
                project="plenoxels",
                name=self.config.experiment_name,
                config=self.config.__dict__,
            )
            self.use_wandb = True
        else:
            self.use_wandb = False
            if self.config.use_wandb:
                logger.warning("W&B requested but not available")

        logger.info(f"Experiment directory: {self.exp_dir}")

    def train(self) -> PlenoxelRenderer:
        """Execute the complete training pipeline.

        Returns:
            PlenoxelRenderer: Trained renderer ready for inference
        """
        logger.info("Starting Plenoxel training...")
        start_time = time.time()

        try:
            for epoch in range(self.epoch, self.config.num_epochs):
                self.epoch = epoch

                # Update resolution for coarse-to-fine training
                self._update_resolution()

                # Training epoch
                train_metrics = self._train_epoch()

                # Learning rate scheduling
                if self.lr_scheduler:
                    self.lr_scheduler.step()

                # Validation
                val_metrics = {}
                if epoch % self.config.eval_interval == 0 and self.val_loader:
                    val_metrics = self._validate_epoch()

                    # Update best model
                    if val_metrics.get("psnr", 0) > self.best_psnr:
                        self.best_psnr = val_metrics["psnr"]
                        self._save_checkpoint("best.pth")

                # Pruning
                if (
                    epoch % self.config.pruning_interval == 0
                    and epoch > 0
                    and self.config.pruning_threshold > 0
                ):
                    self._prune_voxels()

                # Logging
                if epoch % self.config.log_interval == 0:
                    self._log_metrics(train_metrics, val_metrics)

                # Checkpointing
                if epoch % self.config.save_interval == 0:
                    self._save_checkpoint(f"epoch_{epoch}.pth")

            # Final checkpoint
            self._save_checkpoint("final.pth")

            total_time = time.time() - start_time
            logger.info(f"Training completed in {total_time:.2f} seconds")

            # Export renderer
            return self.export_renderer()

        except KeyboardInterrupt:
            logger.info("Training interrupted by user")
            self._save_checkpoint("interrupted.pth")
            return self.export_renderer()

        finally:
            self._cleanup_logging()

    def _train_epoch(self) -> dict[str, float]:
        """Train for one epoch."""
        if not self.train_loader:
            raise ValueError("No training dataset provided")

        self.voxel_grid.train()

        total_loss = 0.0
        total_rgb_loss = 0.0
        total_tv_loss = 0.0
        total_l1_loss = 0.0
        num_batches = 0

        pbar = tqdm(self.train_loader, desc=f"Epoch {self.epoch}")

        for batch_idx, batch in enumerate(pbar):
            # Move batch to device
            batch = {
                k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()
            }

            self.optimizer.zero_grad()

            with autocast(enabled=self.use_amp):
                # Render rays
                outputs = self._render_batch(batch)

                # Compute losses
                losses = self._compute_losses(outputs, batch)
                total_loss_batch = losses["total_loss"]

            # Backward pass with mixed precision
            if self.use_amp:
                self.scaler.scale(total_loss_batch).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                total_loss_batch.backward()
                self.optimizer.step()

            # Accumulate metrics
            total_loss += total_loss_batch.item()
            total_rgb_loss += losses.get("rgb_loss", 0.0)
            total_tv_loss += losses.get("tv_loss", 0.0)
            total_l1_loss += losses.get("l1_loss", 0.0)
            num_batches += 1

            # Update progress bar
            pbar.set_postfix(
                {
                    "loss": f"{total_loss_batch.item():.4f}",
                    "rgb": f"{losses.get('rgb_loss', 0.0):.4f}",
                    "lr": f"{self.optimizer.param_groups[0]['lr']:.6f}",
                }
            )

            self.step += 1

        return {
            "epoch": self.epoch,
            "loss": total_loss / num_batches,
            "rgb_loss": total_rgb_loss / num_batches,
            "tv_loss": total_tv_loss / num_batches,
            "l1_loss": total_l1_loss / num_batches,
            "learning_rate": self.optimizer.param_groups[0]["lr"],
        }

    def _validate_epoch(self) -> dict[str, float]:
        """Validate for one epoch."""
        if not self.val_loader:
            return {}

        self.voxel_grid.eval()

        total_loss = 0.0
        total_psnr = 0.0
        num_batches = 0

        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc="Validation"):
                batch = {
                    k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                    for k, v in batch.items()
                }

                outputs = self._render_batch(batch)
                losses = self._compute_losses(outputs, batch)

                # Compute PSNR
                mse = torch.mean((outputs["rgb"] - batch["colors"]) ** 2)
                psnr = -10 * torch.log10(mse)

                total_loss += losses["total_loss"].item()
                total_psnr += psnr.item()
                num_batches += 1

        return {
            "val_loss": total_loss / num_batches,
            "psnr": total_psnr / num_batches,
        }

    def _render_batch(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """Render a batch of rays."""
        rays_o = batch["rays_o"].view(-1, 3)
        rays_d = batch["rays_d"].view(-1, 3)

        # Render using volumetric renderer
        outputs = self.volumetric_renderer.render_rays(
            self.voxel_grid,
            rays_o,
            rays_d,
            num_samples=self.config.num_samples,
            near=self.config.near_plane,
            far=self.config.far_plane,
            chunk_size=self.config.chunk_size,
        )

        return outputs

    def _compute_losses(
        self, outputs: dict[str, torch.Tensor], batch: dict[str, torch.Tensor]
    ) -> dict[str, torch.Tensor]:
        """Compute all training losses."""
        target_colors = batch["colors"].view(-1, 3)

        # RGB loss
        rgb_loss = self.loss_fn.compute_rgb_loss(outputs["rgb"], target_colors)

        # Regularization losses
        tv_loss = self.voxel_grid.total_variation_loss() * self.config.tv_lambda
        l1_loss = self.voxel_grid.l1_loss() * self.config.l1_lambda

        # Total loss
        total_loss = rgb_loss + tv_loss + l1_loss

        return {
            "rgb_loss": rgb_loss,
            "tv_loss": tv_loss,
            "l1_loss": l1_loss,
            "total_loss": total_loss,
        }

    def _update_resolution(self):
        """Update voxel grid resolution for coarse-to-fine training."""
        if not self.config.use_coarse_to_fine:
            return

        # Determine current resolution level
        for i, epoch_threshold in enumerate(self.config.coarse_epochs):
            if self.epoch < epoch_threshold:
                resolution_level = i
                break
        else:
            resolution_level = len(self.config.coarse_epochs) - 1

        # Update resolution if changed
        if resolution_level != self.current_resolution_level:
            new_resolution = self.config.coarse_resolutions[resolution_level]
            logger.info(f"Updating resolution to {new_resolution}")
            self.voxel_grid.update_resolution(new_resolution)
            self.current_resolution_level = resolution_level

    def _prune_voxels(self):
        """Prune low-density voxels."""
        before_count = self.voxel_grid.get_occupied_voxel_count()
        self.voxel_grid.prune_voxels(self.config.pruning_threshold)
        after_count = self.voxel_grid.get_occupied_voxel_count()

        pruned = before_count - after_count
        logger.info(f"Pruned {pruned} voxels ({before_count} -> {after_count})")

    def _log_metrics(
        self, train_metrics: dict[str, float], val_metrics: dict[str, float] | None = None
    ):
        """Log training metrics."""
        # TensorBoard logging
        if self.tb_writer:
            for key, value in train_metrics.items():
                self.tb_writer.add_scalar(f"train/{key}", value, self.epoch)

            if val_metrics:
                for key, value in val_metrics.items():
                    self.tb_writer.add_scalar(f"val/{key}", value, self.epoch)

        # W&B logging
        if self.use_wandb:
            log_dict = {f"train/{k}": v for k, v in train_metrics.items()}
            if val_metrics:
                log_dict.update({f"val/{k}": v for k, v in val_metrics.items()})
            wandb.log(log_dict, step=self.epoch)

        # Console logging
        log_str = f"Epoch {self.epoch}: " + " | ".join(
            [f"{k}: {v:.4f}" for k, v in train_metrics.items()]
        )
        if val_metrics:
            log_str += " | " + " | ".join([f"val_{k}: {v:.4f}" for k, v in val_metrics.items()])
        logger.info(log_str)

    def _save_checkpoint(self, filename: str):
        """Save training checkpoint."""
        checkpoint = {
            "epoch": self.epoch,
            "step": self.step,
            "voxel_grid_state": self.voxel_grid.state_dict(),
            "optimizer_state": self.optimizer.state_dict(),
            "scaler_state": self.scaler.state_dict(),
            "config": self.config,
            "best_psnr": self.best_psnr,
            "current_resolution_level": self.current_resolution_level,
        }

        if self.lr_scheduler:
            checkpoint["scheduler_state"] = self.lr_scheduler.state_dict()

        checkpoint_path = self.exp_dir / filename
        torch.save(checkpoint, checkpoint_path)
        logger.info(f"Checkpoint saved: {checkpoint_path}")

    def load_checkpoint(self, checkpoint_path: str):
        """Load training checkpoint."""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        self.epoch = checkpoint["epoch"]
        self.step = checkpoint["step"]
        self.voxel_grid.load_state_dict(checkpoint["voxel_grid_state"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state"])
        self.scaler.load_state_dict(checkpoint["scaler_state"])
        self.best_psnr = checkpoint["best_psnr"]
        self.current_resolution_level = checkpoint["current_resolution_level"]

        if "scheduler_state" in checkpoint and self.lr_scheduler:
            self.lr_scheduler.load_state_dict(checkpoint["scheduler_state"])

        logger.info(f"Checkpoint loaded from: {checkpoint_path}")

    def export_renderer(self) -> PlenoxelRenderer:
        """Export a trained renderer for inference.

        Returns:
            PlenoxelRenderer: Configured renderer ready for inference
        """
        # Create inference config from training config
        from .config import PlenoxelInferenceConfig

        inference_config = PlenoxelInferenceConfig(
            grid_resolution=self.voxel_grid.resolution,
            scene_bounds=self.config.scene_bounds,
            sh_degree=self.config.sh_degree,
            near_plane=self.config.near_plane,
            far_plane=self.config.far_plane,
            step_size=self.config.step_size,
            sigma_thresh=self.config.sigma_thresh,
            stop_thresh=self.config.stop_thresh,
            num_samples=self.config.num_samples,
            device=self.config.device,
        )

        # Create and configure renderer
        renderer = PlenoxelRenderer(inference_config)
        renderer.load_voxel_grid(self.voxel_grid)

        logger.info("Exported PlenoxelRenderer for inference")
        return renderer

    def _cleanup_logging(self):
        """Cleanup logging resources."""
        if self.tb_writer:
            self.tb_writer.close()

        if self.use_wandb:
            wandb.finish()


def create_plenoxel_trainer(
    config: PlenoxelTrainingConfig,
    train_dataset: PlenoxelDataset | None = None,
    val_dataset: PlenoxelDataset | None = None,
) -> PlenoxelTrainer:
    """Factory function to create a Plenoxel trainer.

    Args:
        config: Training configuration
        train_dataset: Optional training dataset
        val_dataset: Optional validation dataset

    Returns:
        PlenoxelTrainerRefactored: Configured trainer instance
    """
    return PlenoxelTrainer(config, train_dataset, val_dataset)
