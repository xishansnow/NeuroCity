from typing import Any, Optional
"""
Training module for PyNeRF
Implements the training loop and optimization strategies
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import os
import time
import logging

from .core import PyNeRF, PyNeRFConfig, PyNeRFLoss
from .dataset import PyNeRFDataset, MultiScaleDataset
from .utils import (
    compute_psnr, compute_ssim, log_pyramid_stats, save_pyramid_model, apply_learning_rate_schedule, create_training_schedule
)

logger = logging.getLogger(__name__)

class PyNeRFTrainer:
    """
    Trainer class for PyNeRF model
    """
    
    def __init__(
        self, model: PyNeRF, config: PyNeRFConfig, train_dataset: PyNeRFDataset, val_dataset: Optional[PyNeRFDataset] = None, device: str = "cuda", log_dir: str = "./logs", checkpoint_dir: str = "./checkpoints"
    ):
        self.model = model.to(device)
        self.config = config
        self.device = device
        self.log_dir = log_dir
        self.checkpoint_dir = checkpoint_dir
        
        # Datasets
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        
        # Create data loaders
        self.train_loader = DataLoader(
            train_dataset, batch_size=1, # Process one image at a time
            shuffle=True, num_workers=4, pin_memory=True
        )
        
        if val_dataset is not None:
            self.val_loader = DataLoader(
                val_dataset, batch_size=1, shuffle=False, num_workers=2, pin_memory=True
            )
        
        # Loss function
        self.criterion = PyNeRFLoss(config)
        
        # Optimizer
        self.optimizer = optim.Adam(
            self.model.parameters(), lr=config.learning_rate, betas=(0.9, 0.999)
        )
        
        # Learning rate scheduler
        self.lr_schedule = create_training_schedule(
            max_steps=config.max_steps, warmup_steps=1000, decay_steps=5000, decay_rate=0.1
        )
        
        # Tensorboard writer
        os.makedirs(log_dir, exist_ok=True)
        self.writer = SummaryWriter(log_dir)
        
        # Training state
        self.global_step = 0
        self.epoch = 0
        self.best_psnr = 0.0
        
        # Create checkpoint directory
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        logger.info("PyNeRF Trainer initialized")
    
    def train_step(self, batch: dict[str, torch.Tensor]) -> dict[str, float]:
        """
        Single training step
        
        Args:
            batch: Training batch
            
        Returns:
            Dictionary of loss values
        """
        self.model.train()
        self.optimizer.zero_grad()
        
        # Move data to device
        for key in batch:
            if isinstance(batch[key], torch.Tensor):
                batch[key] = batch[key].to(self.device)
        
        # Get rays and target image
        rays_o = batch["rays_o"].squeeze(0)  # Remove batch dimension
        rays_d = batch["rays_d"].squeeze(0)
        target_image = batch["image"].squeeze(0)
        bounds = batch["bounds"].squeeze(0)
        
        # Sample random rays for training
        H, W = target_image.shape[:2]
        coords = torch.stack(
            torch.meshgrid(
                torch.arange(
                    H,
                    device=self.device,
                )
            ), dim=-1
        ).reshape(-1, 2)
        
        # Random sampling
        select_inds = torch.randperm(coords.shape[0], device=self.device)[:self.config.batch_size]
        select_coords = coords[select_inds]
        
        # Get selected rays and targets
        rays_o_batch = rays_o[select_coords[:, 0], select_coords[:, 1]]
        rays_d_batch = rays_d[select_coords[:, 0], select_coords[:, 1]]
        target_rgb = target_image[select_coords[:, 0], select_coords[:, 1]]
        bounds_batch = bounds[select_coords[:, 0], select_coords[:, 1]]
        
        # Forward pass
        outputs = self.model(rays_o_batch, rays_d_batch, bounds_batch)
        
        # Compute loss
        targets = {"rgb": target_rgb}
        losses = self.criterion(outputs, targets)
        
        # Backward pass
        losses["total_loss"].backward()
        self.optimizer.step()
        
        # Update learning rate
        current_lr = apply_learning_rate_schedule(
            self.optimizer, self.global_step, self.lr_schedule, self.config.learning_rate
        )
        
        # Convert to float for logging
        loss_dict = {k: v.item() for k, v in losses.items()}
        loss_dict["learning_rate"] = current_lr
        
        return loss_dict
    
    def validate(self) -> dict[str, float]:
        """
        Validation step
        
        Returns:
            Dictionary of validation metrics
        """
        if self.val_dataset is None:
            return {}
        
        self.model.eval()
        val_losses = []
        val_psnrs = []
        val_ssims = []
        
        with torch.no_grad():
            for batch in self.val_loader:
                # Move data to device
                for key in batch:
                    if isinstance(batch[key], torch.Tensor):
                        batch[key] = batch[key].to(self.device)
                
                # Get full image rays
                rays_o = batch["rays_o"].squeeze(0)
                rays_d = batch["rays_d"].squeeze(0)
                target_image = batch["image"].squeeze(0)
                bounds = batch["bounds"].squeeze(0)
                
                H, W = target_image.shape[:2]
                
                # Render full image in chunks
                chunk_size = 1024
                rendered_image = torch.zeros_like(target_image)
                
                for i in range(0, H * W, chunk_size):
                    end_i = min(i + chunk_size, H * W)
                    
                    # Get chunk coordinates
                    coords = torch.arange(i, end_i, device=self.device)
                    y_coords = coords // W
                    x_coords = coords % W
                    
                    # Get chunk rays
                    chunk_rays_o = rays_o[y_coords, x_coords]
                    chunk_rays_d = rays_d[y_coords, x_coords]
                    chunk_bounds = bounds[y_coords, x_coords]
                    
                    # Forward pass
                    chunk_outputs = self.model(chunk_rays_o, chunk_rays_d, chunk_bounds)
                    
                    # Store results
                    rendered_image.view(-1, 3)[i:end_i] = chunk_outputs["rgb"]
                
                # Compute metrics
                target_flat = target_image.view(-1, 3)
                rendered_flat = rendered_image.view(-1, 3)
                
                # Loss
                targets = {"rgb": target_flat}
                outputs = {"rgb": rendered_flat}
                losses = self.criterion(outputs, targets)
                val_losses.append(losses["total_loss"].item())
                
                # PSNR
                psnr = compute_psnr(rendered_image, target_image)
                val_psnrs.append(psnr)
                
                # SSIM
                ssim = compute_ssim(rendered_image, target_image)
                val_ssims.append(ssim)
        
        return {
            "val_loss": np.mean(
                val_losses,
            )
        }
    
    def save_checkpoint(self, is_best: bool = False):
        """
        Save model checkpoint
        
        Args:
            is_best: Whether this is the best model so far
        """
        checkpoint_path = os.path.join(
            self.checkpoint_dir, f"checkpoint_step_{self.global_step}.pth"
        )
        
        save_pyramid_model(
            self.model, self.config.__dict__, checkpoint_path, epoch=self.epoch, optimizer_state=self.optimizer.state_dict(
            )
        )
        
        if is_best:
            best_path = os.path.join(self.checkpoint_dir, "best_model.pth")
            save_pyramid_model(
                self.model, self.config.__dict__, best_path, epoch=self.epoch, optimizer_state=self.optimizer.state_dict(
                )
            )
    
    def train(self):
        """
        Main training loop
        """
        logger.info("Starting PyNeRF training")
        start_time = time.time()
        
        while self.global_step < self.config.max_steps:
            epoch_start_time = time.time()
            
            for batch_idx, batch in enumerate(self.train_loader):
                # Training step
                losses = self.train_step(batch)
                
                # Logging
                if self.global_step % 100 == 0:
                    log_pyramid_stats(self.model, self.global_step, losses)
                    
                    # Tensorboard logging
                    for key, value in losses.items():
                        self.writer.add_scalar(f"train/{key}", value, self.global_step)
                
                # Validation
                if self.global_step % 1000 == 0:
                    val_metrics = self.validate()
                    
                    if val_metrics:
                        logger.info(f"Validation - Step {self.global_step}:")
                        for key, value in val_metrics.items():
                            logger.info(f"  {key}: {value:.6f}")
                            self.writer.add_scalar(f"val/{key}", value, self.global_step)
                        
                        # Save best model
                        if val_metrics.get("val_psnr", 0) > self.best_psnr:
                            self.best_psnr = val_metrics["val_psnr"]
                            self.save_checkpoint(is_best=True)
                
                # Save checkpoint
                if self.global_step % 5000 == 0:
                    self.save_checkpoint()
                
                self.global_step += 1
                
                if self.global_step >= self.config.max_steps:
                    break
            
            self.epoch += 1
            epoch_time = time.time() - epoch_start_time
            logger.info(f"Epoch {self.epoch} completed in {epoch_time:.2f}s")
        
        # Final save
        self.save_checkpoint()
        
        total_time = time.time() - start_time
        logger.info(f"Training completed in {total_time:.2f}s")
        logger.info(f"Best validation PSNR: {self.best_psnr:.4f}")
        
        self.writer.close()

class MultiScaleTrainer(PyNeRFTrainer):
    """
    Multi-scale trainer for coarse-to-fine training
    """
    
    def __init__(
        self, model: PyNeRF, config: PyNeRFConfig, train_dataset: MultiScaleDataset, val_dataset: Optional[MultiScaleDataset] = None, scale_schedule: Optional[dict[int, int]] = None, **kwargs
    ):
        super().__init__(model, config, train_dataset, val_dataset, **kwargs)
        
        # Scale schedule: {step: scale}
        if scale_schedule is None:
            self.scale_schedule = {
                0: 8, # Start with 8x downscale
                2000: 4, # 4x downscale
                5000: 2, # 2x downscale
                10000: 1   # Full resolution
            }
        else:
            self.scale_schedule = scale_schedule
        
        self.current_scale = max(self.scale_schedule.values())
        
        logger.info(f"Multi-scale training schedule: {self.scale_schedule}")
    
    def get_current_scale(self) -> int:
        """Get current training scale based on step"""
        scale = self.current_scale
        for step, new_scale in sorted(self.scale_schedule.items()):
            if self.global_step >= step:
                scale = new_scale
        return scale
    
    def train_step(self, batch: dict[str, torch.Tensor]) -> dict[str, float]:
        """
        Multi-scale training step
        
        Args:
            batch: Training batch with multi-scale data
            
        Returns:
            Dictionary of loss values
        """
        # Update current scale
        self.current_scale = self.get_current_scale()
        
        self.model.train()
        self.optimizer.zero_grad()
        
        # Move data to device
        for key in batch:
            if isinstance(batch[key], torch.Tensor):
                batch[key] = batch[key].to(self.device)
            elif isinstance(batch[key], dict):
                for sub_key in batch[key]:
                    if isinstance(batch[key][sub_key], torch.Tensor):
                        batch[key][sub_key] = batch[key][sub_key].to(self.device)
        
        # Get current scale data
        rays_o = batch["multiscale_rays_o"][self.current_scale].squeeze(0)
        rays_d = batch["multiscale_rays_d"][self.current_scale].squeeze(0)
        target_image = batch["multiscale_images"][self.current_scale].squeeze(0)
        bounds = batch["multiscale_bounds"][self.current_scale].squeeze(0)
        
        # Sample random rays
        H, W = target_image.shape[:2]
        coords = torch.stack(
            torch.meshgrid(
                torch.arange(
                    H,
                    device=self.device,
                )
            ), dim=-1
        ).reshape(-1, 2)
        
        select_inds = torch.randperm(coords.shape[0], device=self.device)[:self.config.batch_size]
        select_coords = coords[select_inds]
        
        # Get selected rays and targets
        rays_o_batch = rays_o[select_coords[:, 0], select_coords[:, 1]]
        rays_d_batch = rays_d[select_coords[:, 0], select_coords[:, 1]]
        target_rgb = target_image[select_coords[:, 0], select_coords[:, 1]]
        bounds_batch = bounds[select_coords[:, 0], select_coords[:, 1]]
        
        # Forward pass
        outputs = self.model(rays_o_batch, rays_d_batch, bounds_batch)
        
        # Compute loss
        targets = {"rgb": target_rgb}
        losses = self.criterion(outputs, targets)
        
        # Backward pass
        losses["total_loss"].backward()
        self.optimizer.step()
        
        # Update learning rate
        current_lr = apply_learning_rate_schedule(
            self.optimizer, self.global_step, self.lr_schedule, self.config.learning_rate
        )
        
        # Convert to float for logging
        loss_dict = {k: v.item() for k, v in losses.items()}
        loss_dict["learning_rate"] = current_lr
        loss_dict["current_scale"] = self.current_scale
        
        return loss_dict
