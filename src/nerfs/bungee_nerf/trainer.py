"""
Trainer module for BungeeNeRF
Implements progressive training strategy
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from typing import Dict, List, Optional, Tuple, Any, Union 
import numpy as np
import os
import time
import logging
from tqdm import tqdm

from .core import BungeeNeRF, BungeeNeRFConfig, BungeeNeRFLoss
from .dataset import BungeeNeRFDataset, MultiScaleDataset
from .utils import (
    compute_multiscale_loss, compute_psnr, compute_ssim, save_bungee_model, load_bungee_model, create_progressive_schedule, apply_progressive_schedule
)

logger = logging.getLogger(__name__)


class BungeeNeRFTrainer:
    """
    Base trainer for BungeeNeRF
    """
    
    def __init__(
        self, model: BungeeNeRF, config: BungeeNeRFConfig, train_dataset: BungeeNeRFDataset, val_dataset: Optional[BungeeNeRFDataset] = None, device: str = "cuda"
    ):
        self.model = model.to(device)
        self.config = config
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.device = device
        
        # Loss function
        self.criterion = BungeeNeRFLoss(config)
        
        # Optimizer
        self.optimizer = optim.Adam(
            self.model.parameters(), lr=config.learning_rate, betas=(0.9, 0.999), eps=1e-8
        )
        
        # Learning rate scheduler
        self.scheduler = optim.lr_scheduler.ExponentialLR(
            self.optimizer, gamma=0.1 ** (1.0 / config.max_steps)
        )
        
        # Training state
        self.current_step = 0
        self.current_epoch = 0
        self.current_stage = 0
        self.best_psnr = 0.0
        
        # Logging
        self.log_dir = None
        self.writer = None
        
        logger.info("BungeeNeRF trainer initialized")
    
    def setup_logging(self, log_dir: str):
        """Setup logging and tensorboard"""
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)
        
        self.writer = SummaryWriter(log_dir)
        logger.info(f"Logging setup at {log_dir}")
    
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
        
        # Move batch to device
        batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                for k, v in batch.items()}
        
        # Forward pass
        rays_o = batch["rays_o"].reshape(-1, 3)
        rays_d = batch["rays_d"].reshape(-1, 3)
        bounds = batch["bounds"].reshape(-1, 2)
        distances = batch["distance"].expand(rays_o.shape[0])
        
        outputs = self.model(rays_o, rays_d, bounds, distances)
        
        # Prepare targets
        targets = {
            "rgb": batch["image"].reshape(-1, 3)
        }
        
        if "depth" in batch:
            targets["depth"] = batch["depth"].reshape(-1)
        
        # Compute loss
        losses = self.criterion(outputs, targets, self.current_stage)
        
        # Backward pass
        losses["total_loss"].backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        
        self.optimizer.step()
        self.scheduler.step()
        
        # Convert to float for logging
        loss_dict = {k: v.item() for k, v in losses.items()}
        
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
            for i in range(min(len(self.val_dataset), 10)):  # Validate on subset
                batch = self.val_dataset[i]
                
                # Move to device
                batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                        for k, v in batch.items()}
                
                # Forward pass
                rays_o = batch["rays_o"].reshape(-1, 3)
                rays_d = batch["rays_d"].reshape(-1, 3)
                bounds = batch["bounds"].reshape(-1, 2)
                distances = batch["distance"].expand(rays_o.shape[0])
                
                outputs = self.model(rays_o, rays_d, bounds, distances)
                
                # Prepare targets
                targets = {
                    "rgb": batch["image"].reshape(-1, 3)
                }
                
                # Compute loss
                losses = self.criterion(outputs, targets, self.current_stage)
                val_losses.append(losses["total_loss"].item())
                
                # Compute metrics
                img_pred = outputs["rgb"].reshape(batch["image"].shape)
                img_gt = batch["image"]
                
                psnr = compute_psnr(img_pred, img_gt)
                ssim = compute_ssim(img_pred, img_gt)
                
                val_psnrs.append(psnr)
                val_ssims.append(ssim)
        
        metrics = {
            "val_loss": np.mean(
                val_losses,
            )
        }
        
        return metrics
    
    def train(
        self, num_epochs: int, batch_size: int = 1, log_interval: int = 100, val_interval: int = 1000, save_interval: int = 5000, save_dir: str = "./checkpoints"
    ):
        """
        Train the model
        
        Args:
            num_epochs: Number of training epochs
            batch_size: Batch size
            log_interval: Logging interval
            val_interval: Validation interval
            save_interval: Model saving interval
            save_dir: Directory to save checkpoints
        """
        os.makedirs(save_dir, exist_ok=True)
        
        # Create data loader
        train_loader = DataLoader(
            self.train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True
        )
        
        logger.info(f"Starting training for {num_epochs} epochs")
        
        for epoch in range(num_epochs):
            self.current_epoch = epoch
            
            # Training loop
            epoch_losses = []
            
            pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")
            
            for batch_idx, batch in enumerate(pbar):
                # Training step
                loss_dict = self.train_step(batch)
                epoch_losses.append(loss_dict["total_loss"])
                
                self.current_step += 1
                
                # Update progress bar
                pbar.set_postfix({
                    "loss": f"{loss_dict['total_loss']:.4f}"
                })
                
                # Logging
                if self.current_step % log_interval == 0:
                    self._log_training(loss_dict)
                
                # Validation
                if self.current_step % val_interval == 0:
                    val_metrics = self.validate()
                    self._log_validation(val_metrics)
                    
                    # Update best PSNR
                    if "val_psnr" in val_metrics:
                        if val_metrics["val_psnr"] > self.best_psnr:
                            self.best_psnr = val_metrics["val_psnr"]
                            self._save_checkpoint(save_dir, "best")
                
                # Save checkpoint
                if self.current_step % save_interval == 0:
                    self._save_checkpoint(save_dir, f"step_{self.current_step}")
            
            # End of epoch logging
            avg_loss = np.mean(epoch_losses)
            logger.info(f"Epoch {epoch+1} completed. Average loss: {avg_loss:.4f}")
            
            # Save epoch checkpoint
            self._save_checkpoint(save_dir, f"epoch_{epoch+1}")
        
        logger.info("Training completed")
    
    def _log_training(self, loss_dict: dict[str, float]):
        """Log training metrics"""
        if self.writer is not None:
            for key, value in loss_dict.items():
                self.writer.add_scalar(f"train/{key}", value, self.current_step)
            
            self.writer.add_scalar(
                "train/learning_rate",
                self.optimizer.param_groups[0]['lr'],
                self.current_step,
            )
            
            self.writer.add_scalar("train/stage", self.current_stage, self.current_step)
    
    def _log_validation(self, val_metrics: dict[str, float]):
        """Log validation metrics"""
        if self.writer is not None and val_metrics:
            for key, value in val_metrics.items():
                self.writer.add_scalar(key, value, self.current_step)
            
            logger.info(f"Step {self.current_step}: " + 
                       ", ".join([f"{k}: {v:.4f}" for k, v in val_metrics.items()]))
    
    def _save_checkpoint(self, save_dir: str, name: str):
        """Save model checkpoint"""
        save_path = os.path.join(save_dir, f"{name}.pth")
        
        save_bungee_model(
            self.model, self.config.__dict__, save_path, stage=self.current_stage, epoch=self.current_epoch, optimizer_state=self.optimizer.state_dict(
            )
        )
    
    def load_checkpoint(self, checkpoint_path: str):
        """Load model checkpoint"""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        
        if 'optimizer_state_dict' in checkpoint:
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        if 'stage' in checkpoint:
            self.current_stage = checkpoint['stage']
            self.model.set_current_stage(self.current_stage)
        
        if 'epoch' in checkpoint:
            self.current_epoch = checkpoint['epoch']
        
        logger.info(f"Loaded checkpoint from {checkpoint_path}")


class ProgressiveTrainer(BungeeNeRFTrainer):
    """
    Progressive trainer for BungeeNeRF with multi-stage training
    """
    
    def __init__(
        self, model: BungeeNeRF, config: BungeeNeRFConfig, train_dataset: MultiScaleDataset, val_dataset: Optional[BungeeNeRFDataset] = None, device: str = "cuda", progressive_schedule: Optional[Dict] = None
    ):
        super().__init__(model, config, train_dataset, val_dataset, device)
        
        # Progressive training schedule
        if progressive_schedule is None:
            progressive_schedule = create_progressive_schedule(
                num_stages=config.num_stages, steps_per_stage=config.max_steps // config.num_stages
            )
        
        self.progressive_schedule = progressive_schedule
        
        logger.info(f"Progressive trainer initialized with {config.num_stages} stages")
    
    def update_training_stage(self):
        """Update training stage based on current step"""
        new_stage = apply_progressive_schedule(self.current_step, self.progressive_schedule)
        
        if new_stage != self.current_stage:
            logger.info(f"Advancing to stage {new_stage}")
            self.current_stage = new_stage
            self.model.set_current_stage(new_stage)
            
            # Optionally adjust learning rate for new stage
            for param_group in self.optimizer.param_groups:
                param_group['lr'] *= 0.8  # Reduce learning rate for new stage
    
    def get_stage_data(self) -> list[int]:
        """Get training data indices for current stage"""
        if hasattr(self.train_dataset, 'get_progressive_data'):
            return self.train_dataset.get_progressive_data(self.current_stage)
        else:
            return list(range(len(self.train_dataset)))
    
    def train_step(self, batch: dict[str, torch.Tensor]) -> dict[str, float]:
        """
        Progressive training step
        
        Args:
            batch: Training batch
            
        Returns:
            Dictionary of loss values
        """
        # Update stage if needed
        self.update_training_stage()
        
        # Standard training step
        return super().train_step(batch)
    
    def train(
        self, num_epochs: int, batch_size: int = 1, log_interval: int = 100, val_interval: int = 1000, save_interval: int = 5000, save_dir: str = "./checkpoints"
    ):
        """
        Progressive training with stage-based data selection
        """
        os.makedirs(save_dir, exist_ok=True)
        
        logger.info(f"Starting progressive training for {num_epochs} epochs")
        logger.info(f"Progressive schedule: {self.progressive_schedule}")
        
        for epoch in range(num_epochs):
            self.current_epoch = epoch
            
            # Get data for current stage
            stage_indices = self.get_stage_data()
            
            # Create subset dataset for current stage
            stage_dataset = torch.utils.data.Subset(self.train_dataset, stage_indices)
            
            # Create data loader
            train_loader = DataLoader(
                stage_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True
            )
            
            # Training loop
            epoch_losses = []
            
            pbar = tqdm(
                train_loader,
                desc=f"Epoch {epoch+1}/{num_epochs}"
            )
            
            for batch_idx, batch in enumerate(pbar):
                # Training step
                loss_dict = self.train_step(batch)
                epoch_losses.append(loss_dict["total_loss"])
                
                self.current_step += 1
                
                # Update progress bar
                pbar.set_postfix({
                    "loss": f"{loss_dict['total_loss']:.4f}"
                })
                
                # Logging
                if self.current_step % log_interval == 0:
                    self._log_training(loss_dict)
                
                # Validation
                if self.current_step % val_interval == 0:
                    val_metrics = self.validate()
                    self._log_validation(val_metrics)
                    
                    # Update best PSNR
                    if "val_psnr" in val_metrics:
                        if val_metrics["val_psnr"] > self.best_psnr:
                            self.best_psnr = val_metrics["val_psnr"]
                            self._save_checkpoint(save_dir, "best")
                
                # Save checkpoint
                if self.current_step % save_interval == 0:
                    self._save_checkpoint(save_dir, f"step_{self.current_step}")
            
            # End of epoch logging
            avg_loss = np.mean(epoch_losses)
            logger.info(f"Epoch {epoch+1}")
            
            # Save epoch checkpoint
            self._save_checkpoint(save_dir, f"epoch_{epoch+1}")
        
        logger.info("Progressive training completed")


class MultiScaleTrainer(ProgressiveTrainer):
    """
    Multi-scale trainer with adaptive sampling
    """
    
    def __init__(
        self, model: BungeeNeRF, config: BungeeNeRFConfig, train_dataset: MultiScaleDataset, val_dataset: Optional[BungeeNeRFDataset] = None, device: str = "cuda", scale_weights: Optional[list[float]] = None
    ):
        super().__init__(model, config, train_dataset, val_dataset, device)
        
        # Scale weights for multi-scale loss
        if scale_weights is None:
            scale_weights = [1.0, 0.8, 0.6, 0.4]
        
        self.scale_weights = scale_weights
        
        logger.info("Multi-scale trainer initialized")
    
    def compute_scale_weighted_loss(
        self, outputs: dict[str, torch.Tensor], targets: dict[str, torch.Tensor], distances: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        """
        Compute scale-weighted loss
        
        Args:
            outputs: Model outputs
            targets: Ground truth targets
            distances: Distance to camera
            
        Returns:
            Dictionary of loss values
        """
        # Use multi-scale loss from utils
        return compute_multiscale_loss(
            outputs, targets, distances, self.current_stage, self.config
        )
    
    def train_step(self, batch: dict[str, torch.Tensor]) -> dict[str, float]:
        """
        Multi-scale training step
        
        Args:
            batch: Training batch
            
        Returns:
            Dictionary of loss values
        """
        # Update stage if needed
        self.update_training_stage()
        
        self.model.train()
        self.optimizer.zero_grad()
        
        # Move batch to device
        batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                for k, v in batch.items()}
        
        # Forward pass
        rays_o = batch["rays_o"].reshape(-1, 3)
        rays_d = batch["rays_d"].reshape(-1, 3)
        bounds = batch["bounds"].reshape(-1, 2)
        distances = batch["distance"].expand(rays_o.shape[0])
        
        outputs = self.model(rays_o, rays_d, bounds, distances)
        
        # Prepare targets
        targets = {
            "rgb": batch["image"].reshape(-1, 3)
        }
        
        if "depth" in batch:
            targets["depth"] = batch["depth"].reshape(-1)
        
        # Compute multi-scale loss
        losses = self.compute_scale_weighted_loss(outputs, targets, distances)
        
        # Backward pass
        losses["total_loss"].backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        
        self.optimizer.step()
        self.scheduler.step()
        
        # Convert to float for logging
        loss_dict = {k: v.item() for k, v in losses.items()}
        
        return loss_dict
