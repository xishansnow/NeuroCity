"""
Block-NeRF Trainer

This module provides training functionality for Block-NeRF, tightly coupled
with volume rendering for stable training.
"""

from __future__ import annotations

import os
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

from .core import BlockNeRFConfig, BlockNeRFModel, BlockNeRFLoss
from .volume_renderer import VolumeRenderer, VolumeRendererConfig
from .block_manager import BlockManager

# Type aliases
Tensor = torch.Tensor
TensorDict = dict[str, Tensor]


@dataclass
class BlockNeRFTrainerConfig:
    """Configuration for Block-NeRF trainer."""

    # Training parameters
    num_epochs: int = 100
    batch_size: int = 1024
    learning_rate: float = 5e-4
    learning_rate_decay: float = 0.1
    learning_rate_decay_steps: int = 250000
    weight_decay: float = 1e-6

    # Optimization
    optimizer_type: str = "adam"  # "adam" or "adamw"
    use_amp: bool = True
    gradient_clip_val: float = 0.0

    # Loss weights
    rgb_loss_weight: float = 1.0
    depth_loss_weight: float = 0.1
    eikonal_loss_weight: float = 0.1

    # Block training strategy
    training_strategy: str = "sequential"  # "sequential", "random", "adaptive"
    blocks_per_iteration: int = 4
    block_sampling_prob: float = 0.8

    # Validation
    val_every: int = 1000
    val_batch_size: int = 512

    # Checkpointing
    checkpoint_every: int = 5000
    checkpoint_dir: str = "./checkpoints"
    log_dir: str = "./logs"

    # Pose refinement
    use_pose_refinement: bool = False
    pose_refinement_lr: float = 1e-4
    pose_refinement_start_step: int = 10000

    # Progressive training
    use_progressive_training: bool = True
    coarse_epochs: int = 20
    fine_epochs: int = 80

    # Device
    device: str = "cuda"
    num_workers: int = 4


class BlockNeRFTrainer:
    """
    Block-NeRF Trainer with volume rendering integration.

    This trainer is tightly coupled with the VolumeRenderer for stable training.
    """

    def __init__(
        self,
        model_config: BlockNeRFConfig,
        trainer_config: BlockNeRFTrainerConfig,
        volume_renderer: VolumeRenderer,
        device: str | None = None,
    ):
        self.model_config = model_config
        self.config = trainer_config
        self.device = torch.device(device or trainer_config.device)

        # Volume renderer (tightly coupled)
        self.volume_renderer = volume_renderer.to(self.device)

        # Block manager
        self.block_manager = BlockManager(
            scene_bounds=model_config.scene_bounds,
            block_size=model_config.block_size,
            overlap_ratio=model_config.overlap_ratio,
            device=self.device,
        )

        # Loss function
        self.loss_fn = BlockNeRFLoss(model_config).to(self.device)

        # Training state
        self.step = 0
        self.epoch = 0
        self.best_val_loss = float("inf")

        # Setup logging
        self.setup_logging()

        # Optimizers (will be created when training starts)
        self.optimizers = {}
        self.schedulers = {}

    def setup_logging(self) -> None:
        """Setup logging and checkpointing directories."""
        self.checkpoint_dir = Path(self.config.checkpoint_dir)
        self.log_dir = Path(self.config.log_dir)

        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        self.writer = SummaryWriter(self.log_dir)

    def create_optimizers(self, blocks: dict[str, BlockNeRFModel]) -> None:
        """Create optimizers for all blocks."""
        self.optimizers = {}
        self.schedulers = {}

        for block_name, block in blocks.items():
            # Create optimizer for this block
            if self.config.optimizer_type == "adam":
                optimizer = optim.Adam(
                    block.parameters(),
                    lr=self.config.learning_rate,
                    weight_decay=self.config.weight_decay,
                )
            elif self.config.optimizer_type == "adamw":
                optimizer = optim.AdamW(
                    block.parameters(),
                    lr=self.config.learning_rate,
                    weight_decay=self.config.weight_decay,
                )
            else:
                raise ValueError(f"Unknown optimizer type: {self.config.optimizer_type}")

            self.optimizers[block_name] = optimizer

            # Create learning rate scheduler
            scheduler = optim.lr_scheduler.ExponentialLR(
                optimizer,
                gamma=self.config.learning_rate_decay
                ** (1.0 / self.config.learning_rate_decay_steps),
            )
            self.schedulers[block_name] = scheduler

    def select_training_blocks(self, camera_positions: Tensor) -> list[str]:
        """Select blocks for training based on camera positions."""
        if self.config.training_strategy == "sequential":
            # Train blocks in sequence
            block_names = list(self.block_manager.blocks.keys())
            blocks_per_iter = min(self.config.blocks_per_iteration, len(block_names))
            start_idx = (self.step // 100) % (len(block_names) - blocks_per_iter + 1)
            return block_names[start_idx : start_idx + blocks_per_iter]

        elif self.config.training_strategy == "random":
            # Randomly select blocks
            block_names = list(self.block_manager.blocks.keys())
            num_blocks = min(self.config.blocks_per_iteration, len(block_names))
            indices = torch.randperm(len(block_names))[:num_blocks]
            return [block_names[i] for i in indices]

        elif self.config.training_strategy == "adaptive":
            # Select blocks based on camera visibility
            relevant_blocks = []
            for cam_pos in camera_positions:
                blocks = self.block_manager.get_blocks_for_camera(
                    cam_pos, torch.zeros_like(cam_pos)  # dummy direction
                )
                relevant_blocks.extend(blocks)

            # Remove duplicates and limit
            unique_blocks = list(set(relevant_blocks))
            return unique_blocks[: self.config.blocks_per_iteration]

        else:
            raise ValueError(f"Unknown training strategy: {self.config.training_strategy}")

    def train_step(
        self,
        batch: TensorDict,
        active_blocks: list[str],
    ) -> TensorDict:
        """Perform one training step."""
        # Extract batch data
        ray_origins = batch["ray_origins"].to(self.device)
        ray_directions = batch["ray_directions"].to(self.device)
        target_rgb = batch["rgb"].to(self.device)
        appearance_ids = batch["appearance_ids"].to(self.device)
        exposure_values = batch["exposure_values"].to(self.device)
        camera_positions = batch.get("camera_positions", ray_origins)

        total_loss = 0.0
        losses_dict = {}

        # Train each active block
        for block_name in active_blocks:
            if block_name not in self.block_manager.blocks:
                continue

            block = self.block_manager.blocks[block_name]
            optimizer = self.optimizers[block_name]
            scheduler = self.schedulers[block_name]

            # Zero gradients
            optimizer.zero_grad()

            # Forward pass through volume renderer
            with torch.cuda.amp.autocast(enabled=self.config.use_amp):
                outputs = self.volume_renderer(
                    block,
                    ray_origins,
                    ray_directions,
                    appearance_ids,
                    exposure_values,
                )

                # Compute loss
                targets = {"rgb": target_rgb}
                if "depth" in batch:
                    targets["depth"] = batch["depth"].to(self.device)

                losses = self.loss_fn(outputs, targets)

            # Backward pass
            if self.config.use_amp:
                from torch.cuda.amp import GradScaler

                if not hasattr(self, "scaler"):
                    self.scaler = GradScaler()
                self.scaler.scale(losses["total_loss"]).backward()

                if self.config.gradient_clip_val > 0:
                    self.scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        block.parameters(), self.config.gradient_clip_val
                    )

                self.scaler.step(optimizer)
                self.scaler.update()
            else:
                losses["total_loss"].backward()

                if self.config.gradient_clip_val > 0:
                    torch.nn.utils.clip_grad_norm_(
                        block.parameters(), self.config.gradient_clip_val
                    )

                optimizer.step()

            scheduler.step()

            # Accumulate losses
            total_loss += losses["total_loss"].item()
            for key, value in losses.items():
                if key not in losses_dict:
                    losses_dict[key] = 0.0
                losses_dict[key] += value.item()

        # Average losses across blocks
        if active_blocks:
            total_loss /= len(active_blocks)
            for key in losses_dict:
                losses_dict[key] /= len(active_blocks)

        return {
            "total_loss": total_loss,
            **losses_dict,
        }

    def validate(self, val_loader) -> dict[str, float]:
        """Run validation."""
        self.block_manager.set_eval_mode()

        total_loss = 0.0
        total_samples = 0

        with torch.no_grad():
            for batch in val_loader:
                ray_origins = batch["ray_origins"].to(self.device)
                ray_directions = batch["ray_directions"].to(self.device)
                target_rgb = batch["rgb"].to(self.device)
                appearance_ids = batch["appearance_ids"].to(self.device)
                exposure_values = batch["exposure_values"].to(self.device)

                # Select all blocks for validation
                camera_positions = batch.get("camera_positions", ray_origins)
                active_blocks = self.select_training_blocks(camera_positions[:1])

                if not active_blocks:
                    continue

                # Use first active block for validation
                block = self.block_manager.blocks[active_blocks[0]]

                outputs = self.volume_renderer(
                    block,
                    ray_origins,
                    ray_directions,
                    appearance_ids,
                    exposure_values,
                )

                targets = {"rgb": target_rgb}
                losses = self.loss_fn(outputs, targets)

                total_loss += losses["total_loss"].item() * ray_origins.shape[0]
                total_samples += ray_origins.shape[0]

        self.block_manager.set_train_mode()

        return {
            "val_loss": total_loss / max(total_samples, 1),
        }

    def save_checkpoint(self, filename: str) -> None:
        """Save training checkpoint."""
        checkpoint = {
            "step": self.step,
            "epoch": self.epoch,
            "best_val_loss": self.best_val_loss,
            "model_config": self.model_config,
            "trainer_config": self.config,
            "blocks": {},
            "optimizers": {},
            "schedulers": {},
        }

        # Save all blocks
        for block_name, block in self.block_manager.blocks.items():
            checkpoint["blocks"][block_name] = block.state_dict()
            if block_name in self.optimizers:
                checkpoint["optimizers"][block_name] = self.optimizers[block_name].state_dict()
            if block_name in self.schedulers:
                checkpoint["schedulers"][block_name] = self.schedulers[block_name].state_dict()

        # Save block manager state
        checkpoint["block_manager"] = self.block_manager.get_state_dict()

        checkpoint_path = self.checkpoint_dir / filename
        torch.save(checkpoint, checkpoint_path)
        print(f"Saved checkpoint to {checkpoint_path}")

    def load_checkpoint(self, filename: str) -> None:
        """Load training checkpoint."""
        checkpoint_path = self.checkpoint_dir / filename
        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        self.step = checkpoint["step"]
        self.epoch = checkpoint["epoch"]
        self.best_val_loss = checkpoint["best_val_loss"]

        # Load blocks
        for block_name, state_dict in checkpoint["blocks"].items():
            if block_name in self.block_manager.blocks:
                self.block_manager.blocks[block_name].load_state_dict(state_dict)

        # Load optimizers and schedulers
        if "optimizers" in checkpoint:
            for block_name, state_dict in checkpoint["optimizers"].items():
                if block_name in self.optimizers:
                    self.optimizers[block_name].load_state_dict(state_dict)

        if "schedulers" in checkpoint:
            for block_name, state_dict in checkpoint["schedulers"].items():
                if block_name in self.schedulers:
                    self.schedulers[block_name].load_state_dict(state_dict)

        # Load block manager state
        if "block_manager" in checkpoint:
            self.block_manager.load_state_dict(checkpoint["block_manager"])

        print(f"Loaded checkpoint from {checkpoint_path}")

    def train(self, train_loader, val_loader=None) -> None:
        """Main training loop."""
        print(f"Starting Block-NeRF training for {self.config.num_epochs} epochs")
        print(f"Device: {self.device}")
        print(f"Number of blocks: {len(self.block_manager.blocks)}")

        # Create optimizers
        self.create_optimizers(self.block_manager.blocks)

        # Training loop
        for epoch in range(self.config.num_epochs):
            self.epoch = epoch
            epoch_start_time = time.time()
            epoch_loss = 0.0
            num_batches = 0

            # Set training mode
            self.block_manager.set_train_mode()

            for batch_idx, batch in enumerate(train_loader):
                step_start_time = time.time()

                # Select blocks for training
                camera_positions = batch.get("camera_positions", batch["ray_origins"])
                active_blocks = self.select_training_blocks(camera_positions)

                if not active_blocks:
                    continue

                # Training step
                losses = self.train_step(batch, active_blocks)

                # Update statistics
                epoch_loss += losses["total_loss"]
                num_batches += 1
                self.step += 1

                # Logging
                if self.step % 100 == 0:
                    step_time = time.time() - step_start_time
                    print(
                        f"Epoch {epoch}, Step {self.step}: Loss = {losses['total_loss']:.6f}, Time = {step_time:.3f}s"
                    )

                    # Log to tensorboard
                    for key, value in losses.items():
                        self.writer.add_scalar(f"train/{key}", value, self.step)

                    self.writer.add_scalar("train/step_time", step_time, self.step)

                # Validation
                if val_loader is not None and self.step % self.config.val_every == 0:
                    val_metrics = self.validate(val_loader)
                    val_loss = val_metrics["val_loss"]

                    print(f"Validation Loss: {val_loss:.6f}")
                    self.writer.add_scalar("val/loss", val_loss, self.step)

                    # Save best model
                    if val_loss < self.best_val_loss:
                        self.best_val_loss = val_loss
                        self.save_checkpoint("best_model.pth")

                # Checkpointing
                if self.step % self.config.checkpoint_every == 0:
                    self.save_checkpoint(f"checkpoint_step_{self.step}.pth")

            # End of epoch
            epoch_time = time.time() - epoch_start_time
            avg_loss = epoch_loss / max(num_batches, 1)

            print(f"Epoch {epoch} completed: Avg Loss = {avg_loss:.6f}, Time = {epoch_time:.1f}s")
            self.writer.add_scalar("train/epoch_loss", avg_loss, epoch)
            self.writer.add_scalar("train/epoch_time", epoch_time, epoch)

        # Save final model
        self.save_checkpoint("final_model.pth")
        print("Training completed!")

    def set_train_mode(self) -> None:
        """Set all components to training mode."""
        self.block_manager.set_train_mode()
        self.volume_renderer.train()

    def set_eval_mode(self) -> None:
        """Set all components to evaluation mode."""
        self.block_manager.set_eval_mode()
        self.volume_renderer.eval()


def create_block_nerf_trainer(
    model_config: BlockNeRFConfig,
    trainer_config: BlockNeRFTrainerConfig | None = None,
    volume_renderer_config: VolumeRendererConfig | None = None,
    device: str | None = None,
) -> BlockNeRFTrainer:
    """Create a Block-NeRF trainer with default configurations."""
    if trainer_config is None:
        trainer_config = BlockNeRFTrainerConfig()

    if volume_renderer_config is None:
        volume_renderer_config = VolumeRendererConfig()

    volume_renderer = VolumeRenderer(volume_renderer_config)

    return BlockNeRFTrainer(model_config, trainer_config, volume_renderer, device)
