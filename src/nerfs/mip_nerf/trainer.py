from __future__ import annotations

from typing import Any, Optional

"""
Trainer module for Mip-NeRF

This module implements the training pipeline for Mip-NeRF, including
the training loop, evaluation, checkpointing, and logging.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import os
import time
from pathlib import Path
import json
from tqdm import tqdm

from .core import MipNeRF, MipNeRFConfig, ConicalFrustum
from .dataset import MipNeRFDataset, MipNeRFRayDataset, MipNeRFImageDataset


class MipNeRFTrainer:
    """
    Trainer class for Mip-NeRF model

    Handles training loop, validation, checkpointing, and logging.
    """

    def __init__(
        self,
        config: MipNeRFConfig,
        model: MipNeRF,
        train_dataset: MipNeRFDataset,
        val_dataset: Optional[MipNeRFDataset] = None,
        test_dataset: Optional[MipNeRFDataset] = None,
        device: str = "cuda",
        log_dir: str = "./logs",
    ) -> None:
        """
        Args:
            config: Mip-NeRF configuration
            model: Mip-NeRF model
            train_dataset: Training dataset
            val_dataset: Validation dataset (optional)
            test_dataset: Test dataset (optional)
            device: Device to use for training
            log_dir: Directory for logging
        """
        self.config = config
        self.model = model.to(device)
        self.device = device

        # Datasets
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.test_dataset = test_dataset

        # Create ray datasets for efficient training
        self.train_ray_dataset = MipNeRFRayDataset(train_dataset, batch_size=4096)
        self.train_loader = DataLoader(
            self.train_ray_dataset,
            batch_size=1,
            shuffle=True,
            num_workers=4,
        )

        if val_dataset is not None:
            self.val_image_dataset = MipNeRFImageDataset(val_dataset)
            self.val_loader = DataLoader(self.val_image_dataset, batch_size=1, shuffle=False)

        # Loss function
        # Loss function is now part of the model

        # Optimizer
        self.optimizer = optim.Adam(self.model.parameters(), lr=config.lr_init)

        # Learning rate scheduler
        self.scheduler = optim.lr_scheduler.ExponentialLR(
            self.optimizer, gamma=(config.lr_final / config.lr_init,)
        )

        # Logging
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.writer = SummaryWriter(self.log_dir)

        # Training state
        self.global_step = 0
        self.epoch = 0
        self.best_psnr = 0.0

        # Metrics tracking
        self.train_losses = []
        self.val_psnrs = []
        self.learning_rates = []

    def train_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """
        Single training step

        Args:
            batch: Batch of rays and target colors

        Returns:
            Dictionary of loss values
        """
        self.model.train()
        self.optimizer.zero_grad()

        # Move data to device
        rays_o = batch["rays_o"].to(self.device)
        rays_d = batch["rays_d"].to(self.device)
        radii = batch["radii"].to(self.device)
        target_rgb = batch["rgbs"].to(self.device)

        # Forward pass
        pred = self.model(
            rays_o,
            rays_d,
            rays_d,
            near=self.train_dataset.near,
            far=self.train_dataset.far,
            pixel_radius=radii,
        )

        # Compute loss
        losses = self.loss_fn(pred, target_rgb)
        total_loss = losses["total_loss"]

        # Backward pass
        total_loss.backward()

        # Gradient clipping
        if self.config.grad_max_norm > 0:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.grad_max_norm)
        if self.config.grad_max_val > 0:
            torch.nn.utils.clip_grad_value_(self.model.parameters(), self.config.grad_max_val)

        self.optimizer.step()

        # Convert losses to float
        loss_dict = {k: v.item() for k, v in losses.items()}

        return loss_dict

    def validate(self) -> Dict[str, float]:
        """
        Validate the model on validation set

        Returns:
            Dictionary of validation metrics
        """
        if self.val_dataset is None:
            return {}

        self.model.eval()
        total_psnr = 0.0
        total_loss = 0.0
        num_images = 0

        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc="Validating"):
                target_img = batch["image"].squeeze(0).to(self.device)  # [H, W, 3]
                rays_o = batch["rays_o"].squeeze(0).to(self.device)  # [H, W, 3]
                rays_d = batch["rays_d"].squeeze(0).to(self.device)  # [H, W, 3]
                radii = batch["radii"].squeeze(0).to(self.device)  # [H, W]

                # Render image in chunks
                pred_img = self.render_image(rays_o, rays_d, radii)

                # Compute metrics
                mse = torch.mean((pred_img - target_img) ** 2)
                psnr = -10.0 * torch.log10(mse)

                total_psnr += psnr.item()
                total_loss += mse.item()
                num_images += 1

        avg_psnr = total_psnr / num_images
        avg_loss = total_loss / num_images

        return {"val_psnr": avg_psnr, "val_loss": avg_loss}

    def render_image(
        self,
        rays_o: torch.Tensor,
        rays_d: torch.Tensor,
        radii: torch.Tensor,
        chunk_size: int = 1024,
    ) -> torch.Tensor:
        """
        Render a full image by processing rays in chunks

        Args:
            rays_o: [H, W, 3] ray origins
            rays_d: [H, W, 3] ray directions
            radii: [H, W] pixel radii
            chunk_size: Number of rays to process at once

        Returns:
            [H, W, 3] rendered image
        """
        H, W = rays_o.shape[:2]

        # Flatten rays
        rays_o_flat = rays_o.reshape(-1, 3)
        rays_d_flat = rays_d.reshape(-1, 3)
        radii_flat = radii.reshape(-1)

        # Process in chunks
        rgb_chunks = []

        for i in range(0, rays_o_flat.shape[0], chunk_size):
            chunk_rays_o = rays_o_flat[i : i + chunk_size]
            chunk_rays_d = rays_d_flat[i : i + chunk_size]
            chunk_radii = radii_flat[i : i + chunk_size]

            # Render chunk
            pred = self.model(
                chunk_rays_o,
                chunk_rays_d,
                chunk_rays_d,
                near=self.train_dataset.near,
                far=self.train_dataset.far,
                pixel_radius=chunk_radii,
            )

            # Extract RGB (prefer fine if available)
            if "fine" in pred:
                chunk_rgb = pred["fine"]["rgb"]
            else:
                chunk_rgb = pred["coarse"]["rgb"]

            rgb_chunks.append(chunk_rgb)

        # Combine chunks and reshape
        rgb_flat = torch.cat(rgb_chunks, dim=0)
        rgb_image = rgb_flat.reshape(H, W, 3)

        return rgb_image

    def train(
        self,
        num_epochs: int,
        save_freq: int = 1000,
        val_freq: int = 1000,
        log_freq: int = 100,
    ) -> None:
        """
        Main training loop

        Args:
            num_epochs: Number of epochs to train
            save_freq: Frequency of saving checkpoints (in steps)
            val_freq: Frequency of validation (in steps)
            log_freq: Frequency of logging (in steps)
        """
        print(f"Starting training for {num_epochs} epochs")
        print(f"Training on {len(self.train_ray_dataset)} ray batches per epoch")

        start_time = time.time()

        for epoch in range(num_epochs):
            self.epoch = epoch
            epoch_losses = []

            # Training loop
            pbar = tqdm(self.train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")
            for batch in pbar:
                # Training step
                losses = self.train_step(batch)
                epoch_losses.append(losses["total_loss"])

                # Update learning rate
                self.scheduler.step()
                current_lr = self.optimizer.param_groups[0]["lr"]
                self.learning_rates.append(current_lr)

                # Logging
                if self.global_step % log_freq == 0:
                    self.log_metrics(losses, current_lr)

                    # Update progress bar
                    pbar.set_postfix({"loss": f"{losses['total_loss']:.4f}"})

                # Validation
                if self.global_step % val_freq == 0 and self.val_dataset is not None:
                    val_metrics = self.validate()
                    self.log_validation_metrics(val_metrics)

                    # Save best model
                    if val_metrics.get("val_psnr", 0) > self.best_psnr:
                        self.best_psnr = val_metrics["val_psnr"]
                        self.save_checkpoint("best_model.pth")

                    self.val_psnrs.append(val_metrics.get("val_psnr", 0))

                # Save checkpoint
                if self.global_step % save_freq == 0:
                    self.save_checkpoint(f"checkpoint_{self.global_step}.pth")

                self.global_step += 1

            # Epoch summary
            avg_loss = np.mean(epoch_losses)
            self.train_losses.append(avg_loss)

            elapsed_time = time.time() - start_time
            print(
                f"Epoch {epoch+1} completed. Avg loss: {avg_loss:.4f}, "
                f"Time: {elapsed_time/60:.1f}m"
            )

        # Final checkpoint
        self.save_checkpoint("final_model.pth")
        print("Training completed!")

    def log_metrics(self, losses: Dict[str, float], learning_rate: float):
        """Log training metrics"""
        for name, value in losses.items():
            self.writer.add_scalar(f"train/{name}", value, self.global_step)

        self.writer.add_scalar("train/learning_rate", learning_rate, self.global_step)

    def log_validation_metrics(self, metrics: Dict[str, float]):
        """Log validation metrics"""
        for name, value in metrics.items():
            self.writer.add_scalar(f"val/{name}", value, self.global_step)

    def save_checkpoint(self, filename: str):
        """Save model checkpoint"""
        checkpoint = {
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "global_step": self.global_step,
            "epoch": self.epoch,
            "best_psnr": self.best_psnr,
            "train_losses": self.train_losses,
            "val_psnrs": self.val_psnrs,
            "learning_rates": self.learning_rates,
        }

        torch.save(checkpoint, self.log_dir / filename)
        print(f"Checkpoint saved: {filename}")

    def load_checkpoint(self, checkpoint_path: str):
        """Load model checkpoint"""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

        self.global_step = checkpoint["global_step"]
        self.epoch = checkpoint["epoch"]
        self.best_psnr = checkpoint["best_psnr"]
        self.train_losses = checkpoint.get("train_losses", [])
        self.val_psnrs = checkpoint.get("val_psnrs", [])
        self.learning_rates = checkpoint.get("learning_rates", [])

        print(f"Checkpoint loaded from {checkpoint_path}")
        print(f"Resumed at step {self.global_step}, epoch {self.epoch}")

    def test(self, save_images: bool = True) -> Dict[str, float]:
        """
        Test the model on test set

        Args:
            save_images: Whether to save rendered images

        Returns:
            Dictionary of test metrics
        """
        if self.test_dataset is None:
            print("No test dataset provided")
            return {}

        self.model.eval()
        test_image_dataset = MipNeRFImageDataset(self.test_dataset)
        test_loader = DataLoader(test_image_dataset, batch_size=1, shuffle=False)

        total_psnr = 0.0
        total_ssim = 0.0
        total_lpips = 0.0
        num_images = 0

        results_dir = self.log_dir / "test_results"
        if save_images:
            results_dir.mkdir(exist_ok=True)

        with torch.no_grad():
            for i, batch in enumerate(tqdm(test_loader, desc="Testing")):
                target_img = batch["image"].squeeze(0).to(self.device)
                rays_o = batch["rays_o"].squeeze(0).to(self.device)
                rays_d = batch["rays_d"].squeeze(0).to(self.device)
                radii = batch["radii"].squeeze(0).to(self.device)

                # Render image
                pred_img = self.render_image(rays_o, rays_d, radii)

                # Compute metrics
                mse = torch.mean((pred_img - target_img) ** 2)
                psnr = -10.0 * torch.log10(mse)

                total_psnr += psnr.item()
                num_images += 1

                # Save images
                if save_images:
                    pred_np = pred_img.cpu().numpy()
                    target_np = target_img.cpu().numpy()

                    # Convert to uint8
                    pred_uint8 = (np.clip(pred_np, 0, 1) * 255).astype(np.uint8)
                    target_uint8 = (np.clip(target_np, 0, 1) * 255).astype(np.uint8)

                    # Save images
                    import cv2

                    cv2.imwrite(str(results_dir / f"pred_{i:03d}.png"), pred_uint8)
                    cv2.imwrite(str(results_dir / f"target_{i:03d}.png"), target_uint8)

        avg_psnr = total_psnr / num_images

        test_metrics = {"test_psnr": avg_psnr, "num_test_images": num_images}

        # Save test results
        with open(self.log_dir / "test_results.json", "w") as f:
            json.dump(test_metrics, f, indent=2)

        print(f"Test completed. Average PSNR: {avg_psnr:.2f} dB")
        return test_metrics

    def render_spiral_video(self, poses: torch.Tensor, output_path: str, fps: int = 30):
        """
        Render a spiral video from camera poses

        Args:
            poses: [num_frames, 4, 4] camera poses
            output_path: Path to save the video
            fps: Frames per second
        """
        self.model.eval()
        frames = []

        # Get image dimensions from training data
        H, W = self.train_dataset.images[0].shape[:2]
        focal = self.train_dataset.focal

        with torch.no_grad():
            for i, pose in enumerate(tqdm(poses, desc="Rendering video")):
                # Generate rays
                rays_o, rays_d, radii = self.generate_rays_from_pose(pose, H, W, focal)

                # Render image
                pred_img = self.render_image(rays_o, rays_d, radii)

                # Convert to numpy
                frame = pred_img.cpu().numpy()
                frame = (np.clip(frame, 0, 1) * 255).astype(np.uint8)
                frames.append(frame)

        # Save video
        import imageio

        imageio.mimsave(output_path, frames, fps=fps)
        print(f"Video saved to {output_path}")

    def generate_rays_from_pose(self, pose: torch.Tensor, H: int, W: int, focal: float):
        """Generate rays from camera pose"""
        # Create pixel coordinates
        i, j = torch.meshgrid(
            torch.arange(W, dtype=torch.float32, device=self.device),
            torch.arange(H, dtype=torch.float32, device=self.device),
            indexing="ij",
        )

        # Convert to camera coordinates
        dirs = torch.stack(
            [(i - W * 0.5) / focal, -(j - H * 0.5) / focal, -torch.ones_like(i)], dim=-1
        )

        # Transform to world coordinates
        rays_d = torch.sum(dirs[..., None, :] * pose[:3, :3], dim=-1)
        rays_o = pose[:3, 3].expand(rays_d.shape)

        # Compute pixel radii
        pixel_radius = 1.0 / focal * np.sqrt(2) / 2
        radii = torch.full(rays_d.shape[:-1], pixel_radius, device=self.device)

        return rays_o, rays_d, radii
