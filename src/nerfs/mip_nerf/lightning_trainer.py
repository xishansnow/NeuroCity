"""
PyTorch Lightning trainer for Mip-NeRF.

This module provides a Lightning-based training framework for Mip-NeRF models, with support for integrated positional encoding and anti-aliased rendering.
"""

import torch
import torch.nn.functional as F
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor, EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger, WandbLogger
from pytorch_lightning.strategies import DDPStrategy
import torchmetrics
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
import numpy as np
import logging

from .core import MipNeRF, MipNeRFConfig, MipNeRFLoss, ConicalFrustum
from .dataset import MipNeRFDataset

logger = logging.getLogger(__name__)


@dataclass
class MipNeRFLightningConfig:
    """Configuration for Mip-NeRF Lightning training."""
    
    # Model config
    model_config: MipNeRFConfig = None
    
    # Training parameters
    learning_rate: float = 5e-4
    final_learning_rate: float = 5e-6
    lr_decay_steps: int = 250000
    weight_decay: float = 0.0
    optimizer_type: str = "adam"  # adam, adamw
    
    # Scheduler settings
    scheduler_type: str = "exponential"  # exponential, cosine, none
    warmup_steps: int = 0
    
    # Loss weights
    coarse_loss_weight: float = 1.0
    fine_loss_weight: float = 1.0
    
    # Sampling settings
    ray_batch_size: int = 1024
    chunk_size: int = 1024 * 32
    num_coarse_samples: int = 64
    num_fine_samples: int = 128
    use_hierarchical_sampling: bool = True
    
    # Training optimization
    gradient_clip_val: float = 0.0
    gradient_clip_algorithm: str = "norm"  # norm, value
    use_mixed_precision: bool = True
    
    # Validation settings
    val_chunk_size: int = 1024 * 8
    render_test_freq: int = 25
    
    # Mip-NeRF specific
    pixel_radius: float = 1.0
    perturb_samples: bool = True
    white_background: bool = False
    
    # Advanced features
    use_multiscale_loss: bool = True
    stop_level_grad: bool = True


class MipNeRFLightningModule(pl.LightningModule):
    """PyTorch Lightning module for Mip-NeRF training."""
    
    def __init__(self, config: MipNeRFLightningConfig):
        super().__init__()
        self.config = config
        self.save_hyperparameters()
        
        # Initialize model and loss
        self.model = MipNeRF(config.model_config)
        self.loss_fn = MipNeRFLoss(config.model_config)
        
        # Initialize metrics
        self.train_psnr = torchmetrics.image.PeakSignalNoiseRatio()
        self.val_psnr = torchmetrics.image.PeakSignalNoiseRatio()
        self.val_ssim = torchmetrics.image.StructuralSimilarityIndexMeasure()
        
        # Training state
        self.automatic_optimization = True
        
        # Background color
        bg_color = 1.0 if config.white_background else 0.0
        self.register_buffer('background_color', torch.tensor([bg_color, bg_color, bg_color]))
    
    def forward(
        self,
        origins: torch.Tensor,
        directions: torch.Tensor,
        viewdirs: torch.Tensor,
        near: float,
        far: float,
        pixel_radius: float = None,
    )
        """Forward pass through the model."""
        if pixel_radius is None:
            pixel_radius = self.config.pixel_radius
        return self.model(origins, directions, viewdirs, near, far, pixel_radius)
    
    def training_step(self, batch: dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        """Training step."""
        # Extract batch data
        rays_o = batch['rays_o']  # [N, 3]
        rays_d = batch['rays_d']  # [N, 3]
        viewdirs = batch.get('viewdirs', rays_d)  # [N, 3]
        target_colors = batch['colors']  # [N, 3]
        near = batch.get('near', 2.0)
        far = batch.get('far', 6.0)
        
        # Sample rays if batch is too large
        if rays_o.shape[0] > self.config.ray_batch_size:
            indices = torch.randperm(rays_o.shape[0])[:self.config.ray_batch_size]
            rays_o = rays_o[indices]
            rays_d = rays_d[indices]
            viewdirs = viewdirs[indices]
            target_colors = target_colors[indices]
        
        # Process in chunks if needed
        chunk_size = self.config.chunk_size
        if rays_o.shape[0] > chunk_size:
            outputs = self._process_in_chunks(
                rays_o, rays_d, viewdirs, near, far, chunk_size, train=True
            )
        else:
            outputs = self.model(rays_o, rays_d, viewdirs, near, far, self.config.pixel_radius)
        
        # Compute losses
        losses = self.loss_fn(outputs, target_colors)
        
        # Log training metrics
        self.log('train/loss', losses['total_loss'], on_step=True, on_epoch=True, prog_bar=True)
        self.log('train/coarse_loss', losses['coarse_loss'], on_step=True, on_epoch=True)
        
        if 'fine_loss' in losses:
            self.log('train/fine_loss', losses['fine_loss'], on_step=True, on_epoch=True)
        
        # Compute PSNR for coarse and fine outputs
        coarse_psnr = self.train_psnr(outputs['rgb_coarse'], target_colors)
        self.log('train/coarse_psnr', coarse_psnr, on_step=True, on_epoch=True)
        
        if 'rgb_fine' in outputs:
            fine_psnr = -10 * torch.log10(F.mse_loss(outputs['rgb_fine'], target_colors))
            self.log('train/fine_psnr', fine_psnr, on_step=True, on_epoch=True, prog_bar=True)
        
        # Log learning rate
        self.log('train/lr', self.optimizers().param_groups[0]['lr'], on_step=True)
        
        return losses['total_loss']
    
    def validation_step(
        self,
        batch: dict[str,
        torch.Tensor],
        batch_idx: int,
    )
        """Validation step."""
        # Extract batch data
        rays_o = batch['rays_o']
        rays_d = batch['rays_d']
        viewdirs = batch.get('viewdirs', rays_d)
        target_colors = batch['colors']
        near = batch.get('near', 2.0)
        far = batch.get('far', 6.0)
        
        # Process in chunks to avoid memory issues
        chunk_size = self.config.val_chunk_size
        outputs = self._process_in_chunks(
            rays_o, rays_d, viewdirs, near, far, chunk_size, train=False
        )
        
        # Compute losses
        losses = self.loss_fn(outputs, target_colors)
        
        # Compute metrics
        coarse_psnr = self.val_psnr(outputs['rgb_coarse'], target_colors)
        
        # Use fine output if available, otherwise coarse
        pred_rgb = outputs.get('rgb_fine', outputs['rgb_coarse'])
        fine_psnr = -10 * torch.log10(F.mse_loss(pred_rgb, target_colors))
        
        # For image-level metrics, try to reshape if possible
        if 'image_shape' in batch:
            H, W = batch['image_shape']
            if pred_rgb.shape[0] == H * W:
                pred_image = pred_rgb.view(1, H, W, 3).permute(0, 3, 1, 2).clamp(0, 1)
                target_image = target_colors.view(1, H, W, 3).permute(0, 3, 1, 2).clamp(0, 1)
                ssim = self.val_ssim(pred_image, target_image)
            else:
                ssim = torch.tensor(0.0, device=self.device)
        else:
            ssim = torch.tensor(0.0, device=self.device)
        
        return {
            'val_loss': losses['total_loss'], 'val_coarse_psnr': coarse_psnr, 'val_fine_psnr': fine_psnr, 'val_ssim': ssim, 'pred_rgb': pred_rgb[:100], # Log first 100 pixels
            'target_rgb': target_colors[:100], 'coarse_rgb': outputs['rgb_coarse'][:100]
        }
    
    def validation_epoch_end(self, outputs: list[dict[str, torch.Tensor]]) -> None:
        """Aggregate validation results."""
        # Average metrics
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        avg_coarse_psnr = torch.stack([x['val_coarse_psnr'] for x in outputs]).mean()
        avg_fine_psnr = torch.stack([x['val_fine_psnr'] for x in outputs]).mean()
        avg_ssim = torch.stack([x['val_ssim'] for x in outputs]).mean()
        
        # Log metrics
        self.log('val/loss', avg_loss, prog_bar=True)
        self.log('val/coarse_psnr', avg_coarse_psnr, prog_bar=True)
        self.log('val/fine_psnr', avg_fine_psnr, prog_bar=True)
        self.log('val/ssim', avg_ssim, prog_bar=True)
        
        # Log sample images
        if len(outputs) > 0 and self.current_epoch % 5 == 0:
            self._log_sample_images(outputs[0])
    
    def configure_optimizers(self) -> dict[str, Any]:
        """Configure optimizers and schedulers."""
        # Create optimizer
        if self.config.optimizer_type == "adam":
            optimizer = torch.optim.Adam(
                self.parameters(
                )
            )
        elif self.config.optimizer_type == "adamw":
            optimizer = torch.optim.AdamW(
                self.parameters(
                )
            )
        else:
            raise ValueError(f"Unsupported optimizer: {self.config.optimizer_type}")
        
        # Configure scheduler
        if self.config.scheduler_type == "none":
            return optimizer
        
        scheduler_config = {"optimizer": optimizer}
        
        if self.config.scheduler_type == "exponential":
            # Exponential decay from initial to final learning rate
            decay_rate = (self.config.final_learning_rate / self.config.learning_rate) ** (
                1.0 / self.config.lr_decay_steps
            )
            scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=decay_rate)
            scheduler_config["lr_scheduler"] = scheduler
            scheduler_config["interval"] = "step"
            scheduler_config["frequency"] = 1
            
        elif self.config.scheduler_type == "cosine":
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=self.trainer.max_epochs
            )
            scheduler_config["lr_scheduler"] = scheduler
            scheduler_config["interval"] = "epoch"
        
        return scheduler_config
    
    def _process_in_chunks(
        self,
        rays_o: torch.Tensor,
        rays_d: torch.Tensor,
        viewdirs: torch.Tensor,
        near: float,
        far: float,
        chunk_size: int,
        train: bool = True,
    )
        """Process rays in chunks to avoid memory issues."""
        all_outputs = {}
        
        for i in range(0, rays_o.shape[0], chunk_size):
            end_i = min(i + chunk_size, rays_o.shape[0])
            chunk_rays_o = rays_o[i:end_i]
            chunk_rays_d = rays_d[i:end_i]
            chunk_viewdirs = viewdirs[i:end_i]
            
            if train:
                chunk_outputs = self.model(
                    chunk_rays_o, chunk_rays_d, chunk_viewdirs, near, far, self.config.pixel_radius
                )
            else:
                with torch.no_grad():
                    chunk_outputs = self.model(
                        chunk_rays_o, chunk_rays_d, chunk_viewdirs, near, far, self.config.pixel_radius
                    )
            
            # Accumulate outputs
            for key, value in chunk_outputs.items():
                if key not in all_outputs:
                    all_outputs[key] = []
                all_outputs[key].append(value)
        
        # Concatenate all chunks
        for key in all_outputs:
            all_outputs[key] = torch.cat(all_outputs[key], dim=0)
        
        return all_outputs
    
    def _log_sample_images(self, sample_output: dict[str, torch.Tensor]):
        """Log sample rendered images to tensorboard."""
        if self.logger is not None and hasattr(self.logger, 'experiment'):
            # Log predicted vs target vs coarse
            pred_rgb = sample_output['pred_rgb']
            target_rgb = sample_output['target_rgb']
            coarse_rgb = sample_output['coarse_rgb']
            
            # Ensure images are in [0, 1] range
            pred_rgb = torch.clamp(pred_rgb, 0, 1)
            target_rgb = torch.clamp(target_rgb, 0, 1)
            coarse_rgb = torch.clamp(coarse_rgb, 0, 1)
            
            # Create comparison image
            comparison = torch.cat([coarse_rgb, pred_rgb, target_rgb], dim=0)  # [300, 3]
            
            self.logger.experiment.add_histogram(
                'val/rgb_distribution', pred_rgb, self.current_epoch
            )
            
            # Log as text for now (would need proper image reconstruction for full images)
            self.log('val/sample_pred_mean', pred_rgb.mean())
            self.log('val/sample_target_mean', target_rgb.mean())
            self.log('val/sample_coarse_mean', coarse_rgb.mean())


def create_mip_nerf_lightning_trainer(
    config: MipNeRFLightningConfig, train_dataset, val_dataset = None, max_epochs: int = 200, gpus: int | list[int] = 1, logger_type: str = "tensorboard", project_name: str = "mip_nerf", experiment_name: str = "default", checkpoint_dir: str = "checkpoints", **trainer_kwargs
) -> tuple[MipNeRFLightningModule, pl.Trainer]:
    """
    Create Mip-NeRF Lightning module and trainer.
    
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
    lightning_module = MipNeRFLightningModule(config)
    
    # Setup logger
    if logger_type == "tensorboard":
        logger = TensorBoardLogger(
            save_dir="logs", name=project_name, version=experiment_name
        )
    elif logger_type == "wandb":
        logger = WandbLogger(
            project=project_name, name=experiment_name, save_dir="logs", tags=["mip-nerf", "nerf", "anti-aliasing"]
        )
    else:
        logger = None
    
    # Setup callbacks
    callbacks = [
        ModelCheckpoint(
            dirpath=checkpoint_dir, filename=f"{
                experiment_name,
            }
        ), LearningRateMonitor(logging_interval="step"), EarlyStopping(
            monitor="val/fine_psnr", mode="max", patience=50, verbose=True
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
        max_epochs=max_epochs, devices=gpus, logger=logger, callbacks=callbacks, strategy=strategy, gradient_clip_val=config.gradient_clip_val if config.gradient_clip_val > 0 else None, gradient_clip_algorithm=config.gradient_clip_algorithm, precision="16-mixed" if config.use_mixed_precision else 32, log_every_n_steps=50, val_check_interval=0.25, **trainer_kwargs
    )
    
    return lightning_module, trainer


def train_mip_nerf_lightning(
    model_config: MipNeRFConfig, lightning_config: MipNeRFLightningConfig, train_dataset, val_dataset = None, **trainer_kwargs
) -> MipNeRFLightningModule:
    """
    Simplified training function for Mip-NeRF using Lightning.
    
    Args:
        model_config: Mip-NeRF model configuration
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
    lightning_module, trainer = create_mip_nerf_lightning_trainer(
        lightning_config, train_dataset, val_dataset, **trainer_kwargs
    )
    
    # Create data loaders
    from torch.utils.data import DataLoader
    
    train_loader = DataLoader(
        train_dataset, batch_size=lightning_config.ray_batch_size, shuffle=True, num_workers=4, pin_memory=True
    )
    
    val_loader = None
    if val_dataset is not None:
        val_loader = DataLoader(
            val_dataset, batch_size=lightning_config.ray_batch_size, shuffle=False, num_workers=2, pin_memory=True
        )
    
    # Start training
    trainer.fit(lightning_module, train_loader, val_loader)
    
    return lightning_module 