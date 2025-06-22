"""
PyTorch Lightning trainer for Instant-NGP.

This module provides a Lightning-based training framework for Instant-NGP models,
optimized for fast neural radiance field training.
"""

import torch
import torch.nn.functional as F
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor, EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger, WandbLogger
from pytorch_lightning.strategies import DDPStrategy
import torchmetrics
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass
import numpy as np
import logging

from .core import InstantNGP, InstantNGPConfig, InstantNGPLoss, InstantNGPRenderer
from .dataset import InstantNGPDataset

logger = logging.getLogger(__name__)


@dataclass
class InstantNGPLightningConfig:
    """Configuration for Instant-NGP Lightning training."""
    
    # Model config
    model_config: InstantNGPConfig = None
    
    # Training parameters
    learning_rate: float = 1e-2
    weight_decay: float = 1e-6
    optimizer_type: str = "adam"  # adam, adamw, sgd
    
    # Learning rate scheduling
    lr_decay_steps: int = 1000
    lr_decay_factor: float = 0.33
    min_lr: float = 1e-4
    
    # Hash encoding specific learning rates
    hash_lr_factor: float = 1.0    # Multiplier for hash encoding parameters
    network_lr_factor: float = 1.0  # Multiplier for network parameters
    
    # Loss weights
    color_loss_weight: float = 1.0
    entropy_loss_weight: float = 1e-4
    tv_loss_weight: float = 1e-4
    
    # Sampling settings
    ray_batch_size: int = 4096
    chunk_size: int = 8192
    num_samples: int = 128
    importance_sampling: bool = True
    num_importance_samples: int = 64
    
    # Training optimization
    gradient_clip_val: float = 1.0
    use_mixed_precision: bool = True
    
    # Validation settings
    val_ray_batch_size: int = 1024
    render_test_freq: int = 10
    
    # Advanced features
    adaptive_ray_sampling: bool = True
    error_threshold: float = 0.01
    
    # Hash grid optimization
    hash_decay_rate: float = 0.95
    hash_update_freq: int = 100


class InstantNGPLightningModule(pl.LightningModule):
    """PyTorch Lightning module for Instant-NGP training."""
    
    def __init__(self, config: InstantNGPLightningConfig):
        super().__init__()
        self.config = config
        self.save_hyperparameters()
        
        # Initialize model, loss, and renderer
        self.model = InstantNGP(config.model_config)
        self.loss_fn = InstantNGPLoss(config.model_config)
        self.renderer = InstantNGPRenderer(config.model_config)
        
        # Initialize metrics
        self.train_psnr = torchmetrics.image.PeakSignalNoiseRatio()
        self.val_psnr = torchmetrics.image.PeakSignalNoiseRatio()
        self.val_ssim = torchmetrics.image.StructuralSimilarityIndexMeasure()
        
        # Training state
        self.automatic_optimization = True
        self.lr_scale_factor = 1.0
        
        # Ray sampling statistics
        self.ray_sampling_stats = {
            'total_rays': 0,
            'high_error_rays': 0
        }
    
    def forward(self, positions: torch.Tensor, directions: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Forward pass through the model."""
        return self.model(positions, directions)
    
    def training_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        """Training step."""
        # Extract batch data
        rays_o = batch['rays_o']  # [N, 3]
        rays_d = batch['rays_d']  # [N, 3]
        target_colors = batch['colors']  # [N, 3]
        near = batch.get('near', 0.1)
        far = batch.get('far', 1000.0)
        
        # Sample rays if batch is too large
        if rays_o.shape[0] > self.config.ray_batch_size:
            indices = torch.randperm(rays_o.shape[0])[:self.config.ray_batch_size]
            rays_o = rays_o[indices]
            rays_d = rays_d[indices]
            target_colors = target_colors[indices]
        
        # Render rays
        outputs = self.renderer.render_rays(
            self.model, rays_o, rays_d, near, far, 
            num_samples=self.config.num_samples
        )
        
        pred_colors = outputs['rgb']
        pred_density = outputs.get('density', None)
        positions = outputs.get('positions', None)
        
        # Compute losses
        losses = self.loss_fn(pred_colors, target_colors, pred_density, positions)
        
        # Log training metrics
        self.log('train/loss', losses['total_loss'], on_step=True, on_epoch=True, prog_bar=True)
        self.log('train/color_loss', losses['color_loss'], on_step=True, on_epoch=True)
        
        if 'entropy_loss' in losses:
            self.log('train/entropy_loss', losses['entropy_loss'], on_step=True, on_epoch=True)
        if 'tv_loss' in losses:
            self.log('train/tv_loss', losses['tv_loss'], on_step=True, on_epoch=True)
        
        # Compute PSNR
        psnr = self.train_psnr(pred_colors, target_colors)
        self.log('train/psnr', psnr, on_step=True, on_epoch=True, prog_bar=True)
        
        # Log learning rate
        self.log('train/lr', self.optimizers().param_groups[0]['lr'], on_step=True)
        
        # Update ray sampling statistics
        if self.config.adaptive_ray_sampling:
            self._update_ray_sampling_stats(pred_colors, target_colors)
        
        # Log hash grid statistics periodically
        if batch_idx % 100 == 0:
            self._log_hash_statistics()
        
        return losses['total_loss']
    
    def validation_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> Dict[str, torch.Tensor]:
        """Validation step."""
        # Extract batch data
        rays_o = batch['rays_o']
        rays_d = batch['rays_d']
        target_colors = batch['colors']
        near = batch.get('near', 0.1)
        far = batch.get('far', 1000.0)
        
        # Process in chunks to avoid memory issues
        chunk_size = self.config.val_ray_batch_size
        all_rgb = []
        all_depth = []
        
        with torch.no_grad():
            for i in range(0, rays_o.shape[0], chunk_size):
                end_i = min(i + chunk_size, rays_o.shape[0])
                chunk_rays_o = rays_o[i:end_i]
                chunk_rays_d = rays_d[i:end_i]
                
                chunk_outputs = self.renderer.render_rays(
                    self.model, chunk_rays_o, chunk_rays_d, near, far,
                    num_samples=self.config.num_samples
                )
                
                all_rgb.append(chunk_outputs['rgb'])
                all_depth.append(chunk_outputs['depth'])
        
        # Concatenate results
        pred_rgb = torch.cat(all_rgb, dim=0)
        pred_depth = torch.cat(all_depth, dim=0)
        
        # Compute losses
        losses = self.loss_fn(pred_rgb, target_colors)
        
        # Compute metrics
        psnr = self.val_psnr(pred_rgb, target_colors)
        
        # For image-level metrics, try to reshape if possible
        if 'image_shape' in batch:
            H, W = batch['image_shape']
            if pred_rgb.shape[0] == H * W:
                pred_image = pred_rgb.view(1, H, W, 3).permute(0, 3, 1, 2)
                target_image = target_colors.view(1, H, W, 3).permute(0, 3, 1, 2)
                ssim = self.val_ssim(pred_image, target_image)
            else:
                ssim = torch.tensor(0.0, device=self.device)
        else:
            ssim = torch.tensor(0.0, device=self.device)
        
        return {
            'val_loss': losses['total_loss'],
            'val_psnr': psnr,
            'val_ssim': ssim,
            'pred_rgb': pred_rgb[:100],  # Log first 100 pixels
            'target_rgb': target_colors[:100]
        }
    
    def validation_epoch_end(self, outputs: List[Dict[str, torch.Tensor]]) -> None:
        """Aggregate validation results."""
        # Average metrics
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        avg_psnr = torch.stack([x['val_psnr'] for x in outputs]).mean()
        avg_ssim = torch.stack([x['val_ssim'] for x in outputs]).mean()
        
        # Log metrics
        self.log('val/loss', avg_loss, prog_bar=True)
        self.log('val/psnr', avg_psnr, prog_bar=True)
        self.log('val/ssim', avg_ssim, prog_bar=True)
        
        # Log hash grid statistics
        self._log_hash_statistics(prefix='val_')
        
        # Log ray sampling statistics
        if self.config.adaptive_ray_sampling:
            self._log_ray_sampling_stats()
    
    def configure_optimizers(self) -> Dict[str, Any]:
        """Configure optimizers and schedulers."""
        # Group parameters with different learning rates
        hash_params = []
        network_params = []
        
        for name, param in self.model.named_parameters():
            if 'hash_encoder' in name or 'embeddings' in name:
                hash_params.append(param)
            else:
                network_params.append(param)
        
        param_groups = [
            {
                'params': hash_params,
                'lr': self.config.learning_rate * self.config.hash_lr_factor,
                'name': 'hash_encoding'
            },
            {
                'params': network_params,
                'lr': self.config.learning_rate * self.config.network_lr_factor,
                'name': 'networks'
            }
        ]
        
        # Create optimizer
        if self.config.optimizer_type == "adam":
            optimizer = torch.optim.Adam(param_groups, weight_decay=self.config.weight_decay)
        elif self.config.optimizer_type == "adamw":
            optimizer = torch.optim.AdamW(param_groups, weight_decay=self.config.weight_decay)
        elif self.config.optimizer_type == "sgd":
            optimizer = torch.optim.SGD(param_groups, momentum=0.9, 
                                      weight_decay=self.config.weight_decay)
        else:
            raise ValueError(f"Unsupported optimizer: {self.config.optimizer_type}")
        
        # Configure scheduler with exponential decay
        scheduler = torch.optim.lr_scheduler.ExponentialLR(
            optimizer, 
            gamma=self.config.lr_decay_factor ** (1.0 / self.config.lr_decay_steps)
        )
        
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
                "frequency": 1
            }
        }
    
    def on_train_epoch_end(self) -> None:
        """Called at the end of each training epoch."""
        # Update hash grid if needed
        if self.current_epoch % self.config.hash_update_freq == 0:
            self._update_hash_grid()
        
        # Reset ray sampling statistics
        if self.config.adaptive_ray_sampling:
            self.ray_sampling_stats = {
                'total_rays': 0,
                'high_error_rays': 0
            }
    
    def _update_ray_sampling_stats(self, pred_colors: torch.Tensor, target_colors: torch.Tensor):
        """Update ray sampling statistics for adaptive sampling."""
        errors = torch.mean((pred_colors - target_colors) ** 2, dim=-1)
        high_error_rays = (errors > self.config.error_threshold).sum().item()
        
        self.ray_sampling_stats['total_rays'] += pred_colors.shape[0]
        self.ray_sampling_stats['high_error_rays'] += high_error_rays
    
    def _log_ray_sampling_stats(self):
        """Log ray sampling statistics."""
        if self.ray_sampling_stats['total_rays'] > 0:
            error_ratio = self.ray_sampling_stats['high_error_rays'] / self.ray_sampling_stats['total_rays']
            self.log('sampling/high_error_ratio', error_ratio)
    
    def _log_hash_statistics(self, prefix: str = 'train_'):
        """Log hash encoding statistics."""
        # Log hash table utilization for each level
        for level_idx, embedding in enumerate(self.model.hash_encoder.embeddings):
            # Compute hash table statistics
            weights = embedding.weight.data
            active_entries = (torch.abs(weights).sum(dim=1) > 1e-6).float().mean()
            weight_magnitude = torch.abs(weights).mean()
            
            self.log(f'{prefix}hash/level_{level_idx}_utilization', active_entries)
            self.log(f'{prefix}hash/level_{level_idx}_magnitude', weight_magnitude)
        
        # Log overall hash encoding statistics
        total_params = sum(emb.num_embeddings * emb.embedding_dim 
                          for emb in self.model.hash_encoder.embeddings)
        self.log(f'{prefix}hash/total_parameters', total_params)
    
    def _update_hash_grid(self):
        """Update hash grid parameters."""
        logger.info(f"Updating hash grid at epoch {self.current_epoch}")
        
        # Apply decay to hash table entries with low gradients
        for embedding in self.model.hash_encoder.embeddings:
            with torch.no_grad():
                # Get gradient magnitude
                if embedding.weight.grad is not None:
                    grad_magnitude = torch.abs(embedding.weight.grad).sum(dim=1, keepdim=True)
                    
                    # Apply decay to entries with low gradients
                    low_grad_mask = grad_magnitude < 1e-6
                    embedding.weight.data[low_grad_mask.squeeze()] *= self.config.hash_decay_rate


def create_instant_ngp_lightning_trainer(
    config: InstantNGPLightningConfig,
    train_dataset,
    val_dataset = None,
    max_epochs: int = 100,
    gpus: Union[int, List[int]] = 1,
    logger_type: str = "tensorboard",
    project_name: str = "instant_ngp",
    experiment_name: str = "default",
    checkpoint_dir: str = "checkpoints",
    **trainer_kwargs
) -> Tuple[InstantNGPLightningModule, pl.Trainer]:
    """
    Create Instant-NGP Lightning module and trainer.
    
    Args:
        config: Lightning configuration
        train_dataset: Training dataset
        val_dataset: Validation dataset
        max_epochs: Maximum training epochs
        gpus: GPU configuration
        logger_type: Logger type (tensorboard/wandb)
        project_name: Project name for logging
        experiment_name: Experiment name
        checkpoint_dir: Checkpoint directory
        **trainer_kwargs: Additional trainer arguments
        
    Returns:
        Tuple of (lightning_module, trainer)
    """
    # Create Lightning module
    lightning_module = InstantNGPLightningModule(config)
    
    # Setup logger
    if logger_type == "tensorboard":
        logger = TensorBoardLogger(
            save_dir="logs",
            name=project_name,
            version=experiment_name
        )
    elif logger_type == "wandb":
        logger = WandbLogger(
            project=project_name,
            name=experiment_name,
            save_dir="logs",
            tags=["instant-ngp", "nerf", "hash-encoding"]
        )
    else:
        logger = None
    
    # Setup callbacks
    callbacks = [
        ModelCheckpoint(
            dirpath=checkpoint_dir,
            filename=f"{experiment_name}-{{epoch:02d}}-{{val/psnr:.2f}}",
            monitor="val/psnr",
            mode="max",
            save_top_k=3,
            save_last=True
        ),
        LearningRateMonitor(logging_interval="step"),
        EarlyStopping(
            monitor="val/psnr",
            mode="max",
            patience=50,  # Longer patience for Instant-NGP
            verbose=True
        )
    ]
    
    # Setup strategy for multi-GPU
    strategy = None
    if isinstance(gpus, list) and len(gpus) > 1:
        strategy = DDPStrategy(find_unused_parameters=False)
    elif isinstance(gpus, int) and gpus > 1:
        strategy = DDPStrategy(find_unused_parameters=False)
    
    # Create trainer
    trainer = pl.Trainer(
        max_epochs=max_epochs,
        devices=gpus,
        logger=logger,
        callbacks=callbacks,
        strategy=strategy,
        gradient_clip_val=config.gradient_clip_val,
        precision="16-mixed" if config.use_mixed_precision else 32,
        log_every_n_steps=25,
        val_check_interval=0.5,
        **trainer_kwargs
    )
    
    return lightning_module, trainer


def train_instant_ngp_lightning(
    model_config: InstantNGPConfig,
    lightning_config: InstantNGPLightningConfig,
    train_dataset,
    val_dataset = None,
    **trainer_kwargs
) -> InstantNGPLightningModule:
    """
    Simplified training function for Instant-NGP using Lightning.
    
    Args:
        model_config: Instant-NGP model configuration
        lightning_config: Lightning training configuration
        train_dataset: Training dataset
        val_dataset: Validation dataset
        **trainer_kwargs: Additional trainer arguments
        
    Returns:
        Trained Lightning module
    """
    # Set model config in lightning config
    lightning_config.model_config = model_config
    
    # Create Lightning module and trainer
    lightning_module, trainer = create_instant_ngp_lightning_trainer(
        lightning_config,
        train_dataset,
        val_dataset,
        **trainer_kwargs
    )
    
    # Create data loaders
    from torch.utils.data import DataLoader
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=lightning_config.ray_batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )
    
    val_loader = None
    if val_dataset is not None:
        val_loader = DataLoader(
            val_dataset,
            batch_size=lightning_config.val_ray_batch_size,
            shuffle=False,
            num_workers=2,
            pin_memory=True
        )
    
    # Start training
    trainer.fit(lightning_module, train_loader, val_loader)
    
    return lightning_module 