from __future__ import annotations

"""
Grid-NeRF Trainer Module

This module provides comprehensive training functionality for Grid-NeRF models, including multi-GPU support, checkpointing, evaluation, and visualization.
"""

from typing import Any, Optional, Union
import os
import sys
import time
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import logging
from pathlib import Path
import json
from tqdm import tqdm

from .core import GridNeRF, GridNeRFConfig
from .dataset import create_dataset, create_dataloader, GridNeRFDataset
from .utils.metrics_utils import compute_psnr, compute_ssim, compute_lpips
from .utils.io_utils import save_image, create_video_from_images
from .utils.training_utils import setup_logging, get_learning_rate_scheduler


class GridNeRFTrainer:
    """
    Comprehensive trainer for Grid-NeRF models with support for:
    - Multi-GPU distributed training
    - Checkpointing and resuming
    - Evaluation and visualization
    - Learning rate scheduling
    - Loss tracking and logging
    """

    def __init__(
        self,
        config: GridNeRFConfig,
        output_dir: str,
        device: torch.device,
        rank: int = 0,
        world_size: int = 1,
        use_tensorboard: bool = True,
    ):
        self.config = config
        self.output_dir = Path(output_dir)
        self.device = device
        self.rank = rank
        self.world_size = world_size

        # Create output directories
        self.checkpoint_dir = self.output_dir / "checkpoints"
        self.log_dir = self.output_dir / "logs"
        self.eval_dir = self.output_dir / "evaluation"
        self.render_dir = self.output_dir / "renders"

        for dir_path in [self.checkpoint_dir, self.log_dir, self.eval_dir, self.render_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)

        # Setup logging
        setup_logging(self.log_dir / f"training_rank_{rank}.log")
        self.logger = logging.getLogger(__name__)

        # Initialize tensorboard writer
        if use_tensorboard and rank == 0:
            self.writer = SummaryWriter(log_dir=str(self.log_dir))
        else:
            self.writer = None

        # Initialize model and loss
        self.model = None
        self.loss_fn = None
        self.optimizer = None
        self.scheduler = None

        # Training state
        self.current_epoch = 0
        self.current_step = 0
        self.best_psnr = 0.0
        self.training_losses = []
        self.validation_metrics = []

    def setup_model(self) -> None:
        """Initialize the Grid-NeRF model and loss function."""
        self.model = GridNeRF(self.config).to(self.device)
        self.loss_fn = self.model.compute_loss

        # Setup distributed training
        if self.world_size > 1:
            self.model = DDP(self.model, device_ids=[self.rank])

        # Count parameters
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)

        if self.rank == 0:
            self.logger.info(f"Model initialized with {total_params:, } total parameters")
            self.logger.info(f"Trainable parameters: {trainable_params:, }")

    def setup_optimizer(self) -> None:
        """Initialize optimizer and learning rate scheduler."""
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
        )

        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=self.config.max_steps,
            eta_min=self.config.learning_rate * 0.1,
        )

        if self.rank == 0:
            self.logger.info(
                f"Optimizer initialized with {len(self.optimizer.param_groups)} parameter groups"
            )

    def load_checkpoint(self, checkpoint_path: str) -> None:
        """Load model and training state from checkpoint."""
        if not os.path.exists(checkpoint_path):
            self.logger.warning(f"Checkpoint not found: {checkpoint_path}")
            return

        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        # Load model state
        if self.world_size > 1:
            self.model.module.load_state_dict(checkpoint["model_state_dict"])
        else:
            self.model.load_state_dict(checkpoint["model_state_dict"])

        # Load optimizer and scheduler state
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

        # Load training state
        self.current_epoch = checkpoint.get("epoch", 0)
        self.current_step = checkpoint.get("step", 0)
        self.best_psnr = checkpoint.get("best_psnr", 0.0)

        if self.rank == 0:
            self.logger.info(f"Loaded checkpoint from {checkpoint_path}")
            self.logger.info(f"Resuming from epoch {self.current_epoch}, step {self.current_step}")

    def save_checkpoint(self, is_best: bool = False) -> None:
        """Save model and training state to checkpoint."""
        if self.rank != 0:
            return

        checkpoint = {
            "epoch": self.current_epoch,
            "step": self.current_step,
            "model_state_dict": (
                self.model.module.state_dict() if self.world_size > 1 else self.model.state_dict()
            ),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "best_psnr": self.best_psnr,
        }

        # Save latest checkpoint
        latest_path = self.checkpoint_dir / "latest.pth"
        torch.save(checkpoint, latest_path)

        # Save best checkpoint
        if is_best:
            best_path = self.checkpoint_dir / "best.pth"
            torch.save(checkpoint, best_path)
            self.logger.info(f"Saved best checkpoint with PSNR: {self.best_psnr:.2f}")

        # Save periodic checkpoint
        if self.current_epoch % self.config.save_every_n_epochs == 0:
            epoch_path = self.checkpoint_dir / f"epoch_{self.current_epoch:04d}.pth"
            torch.save(checkpoint, epoch_path)

    def train_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """Perform a single training step."""
        self.model.train()
        self.optimizer.zero_grad()

        # Forward pass
        rays_o = batch["rays_o"].to(self.device)
        rays_d = batch["rays_d"].to(self.device)
        target_rgb = batch["target_rgb"].to(self.device)

        outputs = self.model(rays_o, rays_d)

        # Compute loss
        loss_dict = self.loss_fn(outputs, target_rgb, batch)
        total_loss = loss_dict["total_loss"]

        # Backward pass
        total_loss.backward()

        # Gradient clipping
        if self.config.grad_clip_norm > 0:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.grad_clip_norm)

        self.optimizer.step()
        self.scheduler.step()

        # Convert losses to CPU for logging
        loss_dict_cpu = {k: v.item() if torch.is_tensor(v) else v for k, v in loss_dict.items()}

        return loss_dict_cpu

    def validate(self, val_dataloader) -> Dict[str, float]:
        """Validate the model on validation set."""
        self.model.eval()
        total_loss = 0.0
        total_psnr = 0.0
        total_ssim = 0.0
        num_batches = 0

        with torch.no_grad():
            for batch in val_dataloader:
                rays_o = batch["rays_o"].to(self.device)
                rays_d = batch["rays_d"].to(self.device)
                target_rgb = batch["target_rgb"].to(self.device)

                outputs = self.model(rays_o, rays_d)

                # Compute loss
                loss_dict = self.loss_fn(outputs, target_rgb, batch)
                total_loss += loss_dict["total_loss"].item()

                # Compute metrics
                pred_rgb = outputs["rgb"]
                psnr = compute_psnr(pred_rgb, target_rgb)
                ssim = compute_ssim(pred_rgb, target_rgb)

                total_psnr += psnr
                total_ssim += ssim
                num_batches += 1

        # Average metrics
        avg_loss = total_loss / num_batches
        avg_psnr = total_psnr / num_batches
        avg_ssim = total_ssim / num_batches

        return {"val_loss": avg_loss, "val_psnr": avg_psnr, "val_ssim": avg_ssim}

    def render_test_images(self, test_dataset, num_images: int = 8) -> None:
        """Render test images for visualization."""
        if self.rank != 0:
            return

        self.model.eval()
        render_epoch_dir = self.render_dir / f"epoch_{self.current_epoch:04d}"
        render_epoch_dir.mkdir(exist_ok=True)

        with torch.no_grad():
            for i in range(min(num_images, len(test_dataset))):
                sample = test_dataset[i]
                H, W = sample["image_height"], sample["image_width"]

                # Generate rays for full image
                rays_o, rays_d = test_dataset.get_rays(i)
                rays_o = rays_o.reshape(-1, 3).to(self.device)
                rays_d = rays_d.reshape(-1, 3).to(self.device)

                # Render in chunks to avoid memory issues
                chunk_size = self.config.chunk_size
                rgb_chunks = []

                for j in range(0, rays_o.shape[0], chunk_size):
                    rays_o_chunk = rays_o[j : j + chunk_size]
                    rays_d_chunk = rays_d[j : j + chunk_size]

                    outputs = self.model(rays_o_chunk, rays_d_chunk)
                    rgb_chunks.append(outputs["rgb"].cpu())

                # Combine chunks and reshape
                rgb_pred = torch.cat(rgb_chunks, dim=0).reshape(H, W, 3)
                rgb_gt = sample["target_rgb"]

                # Save images
                save_image(rgb_pred, render_epoch_dir / f"pred_{i:03d}.png")
                save_image(rgb_gt, render_epoch_dir / f"gt_{i:03d}.png")

        self.logger.info(f"Rendered {num_images} test images to {render_epoch_dir}")

    def train(
        self, train_dataset, val_dataset=None, test_dataset=None, resume_from: Optional[str] = None
    ) -> None:
        """Main training loop."""
        # Setup model and optimizer
        self.setup_model()
        self.setup_optimizer()

        # Load checkpoint if specified
        if resume_from:
            self.load_checkpoint(resume_from)

        # Create data loaders
        train_dataloader = create_dataloader(
            train_dataset,
            batch_size=self.config.batch_size,
            num_workers=self.config.num_workers,
            shuffle=True,
            distributed=(self.world_size > 1,),
        )

        if val_dataset:
            val_dataloader = create_dataloader(
                val_dataset,
                batch_size=self.config.eval_batch_size,
                num_workers=self.config.num_workers,
                shuffle=False,
                distributed=False,
            )

        if self.rank == 0:
            self.logger.info("Starting training...")
            self.logger.info(f"Total epochs: {self.config.num_epochs}")
            self.logger.info(f"Steps per epoch: {len(train_dataloader)}")

        # Training loop
        for epoch in range(self.current_epoch, self.config.num_epochs):
            self.current_epoch = epoch

            # Set epoch for distributed sampler
            if hasattr(train_dataloader.sampler, "set_epoch"):
                train_dataloader.sampler.set_epoch(epoch)

            # Training phase
            epoch_losses = []
            epoch_start_time = time.time()

            for batch_idx, batch in enumerate(train_dataloader):
                step_start_time = time.time()

                # Training step
                loss_dict = self.train_step(batch)
                epoch_losses.append(loss_dict)

                self.current_step += 1

                # Logging
                if (batch_idx + 1) % self.config.log_every_n_steps == 0 and self.rank == 0:
                    step_time = time.time() - step_start_time
                    lr = self.scheduler.get_last_lr()[0]

                    self.logger.info(
                        f"Epoch {epoch+1}/{self.config.num_epochs}, "
                        f"Step {batch_idx+1}/{len(train_dataloader)}, "
                        f"Loss: {loss_dict['total_loss']:.6f}, "
                        f"LR: {lr:.6f}, "
                        f"Time: {step_time:.3f}s"
                    )

                    # Tensorboard logging
                    if self.writer:
                        for key, value in loss_dict.items():
                            self.writer.add_scalar(f"train/{key}", value, self.current_step)
                        self.writer.add_scalar("lr", lr, self.current_step)

                # Early stopping check
                if self.current_step >= self.config.max_steps:
                    break

            # Validation phase
            if val_dataset and (epoch + 1) % self.config.eval_every_n_epochs == 0:
                if self.rank == 0:
                    self.logger.info("Running validation...")

                val_metrics = self.validate(val_dataloader)

                if self.rank == 0:
                    self.logger.info(
                        f"Validation - Loss: {val_metrics['val_loss']:.6f}, "
                        f"PSNR: {val_metrics['val_psnr']:.2f}, "
                        f"SSIM: {val_metrics['val_ssim']:.4f}"
                    )

                    # Tensorboard logging
                    if self.writer:
                        for key, value in val_metrics.items():
                            self.writer.add_scalar(f"val/{key}", value, self.current_step)

                    # Check if best model
                    is_best = val_metrics["val_psnr"] > self.best_psnr
                    if is_best:
                        self.best_psnr = val_metrics["val_psnr"]

                    # Save checkpoint
                    self.save_checkpoint(is_best=is_best)

            # Render test images
            if (
                test_dataset
                and self.rank == 0
                and (epoch + 1) % self.config.render_every_n_epochs == 0
            ):
                self.render_test_images(test_dataset)

            # Log epoch summary
            if self.rank == 0:
                epoch_time = time.time() - epoch_start_time
                avg_loss = np.mean([l["total_loss"] for l in epoch_losses])

                self.logger.info(
                    f"Epoch {epoch+1} completed in {epoch_time:.2f}s, "
                    f"Average loss: {avg_loss:.6f}"
                )

            # Early stopping
            if self.current_step >= self.config.max_steps:
                break

        if self.rank == 0:
            self.logger.info("Training completed!")
            if self.writer:
                self.writer.close()

    def cleanup(self) -> None:
        """Cleanup resources."""
        if self.writer:
            self.writer.close()

        if self.world_size > 1:
            dist.destroy_process_group()


def setup_distributed_training(rank: int, world_size: int, backend: str = "nccl") -> None:
    """Setup distributed training environment."""
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"

    dist.init_process_group(backend, rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)


def main_worker(
    rank: int, world_size: int, config: GridNeRFConfig, output_dir: str, data_config: Dict[str, Any]
) -> None:
    """Main worker function for distributed training."""
    # Setup distributed training
    if world_size > 1:
        setup_distributed_training(rank, world_size)

    # Set device
    if torch.cuda.is_available():
        device = torch.device(f"cuda:{rank}")
    else:
        device = torch.device("cpu")

    # Create trainer
    trainer = GridNeRFTrainer(
        config=config, output_dir=output_dir, device=device, rank=rank, world_size=world_size
    )

    try:
        # Create datasets
        train_dataset = create_dataset(
            data_config["train_data_path"],
            split="train",
            config=config,
            **data_config.get(
                "train_kwargs",
                {},
            ),
        )

        val_dataset = None
        if "val_data_path" in data_config:
            val_dataset = create_dataset(
                data_config["val_data_path"],
                split="val",
                config=config,
                **data_config.get(
                    "val_kwargs",
                    {},
                ),
            )

        test_dataset = None
        if "test_data_path" in data_config:
            test_dataset = create_dataset(
                data_config["test_data_path"],
                split="test",
                config=config,
                **data_config.get(
                    "test_kwargs",
                    {},
                ),
            )

        # Start training
        trainer.train(
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            test_dataset=test_dataset,
            resume_from=data_config.get(
                "resume_from",
            ),
        )

    finally:
        trainer.cleanup()


if __name__ == "__main__":
    # Example usage - this would typically be called from a separate script
    pass
