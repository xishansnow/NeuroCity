"""
PyTorch Lightning trainer for SVRaster.

This module provides a Lightning-based training framework for SVRaster models, offering automatic optimization, distributed training, checkpointing, and logging.
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

from .core import SVRasterModel, SVRasterConfig, SVRasterLoss
from .dataset import SVRasterDataset, SVRasterDatasetConfig

logger = logging.getLogger(__name__)


@dataclass
class SVRasterLightningConfig:
    """Configuration for SVRaster Lightning training."""
    
    # Model config
    model_config: SVRasterConfig = None
    
    # Training parameters
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    optimizer_type: str = "adamw"  # adam, adamw, sgd
    
    # Scheduler settings
    scheduler_type: str = "cosine"  # cosine, step, exponential, none
    scheduler_params: dict[str, Any] = None
    warmup_steps: int = 1000
    
    # Loss weights
    rgb_loss_weight: float = 1.0
    depth_loss_weight: float = 0.1
    opacity_reg_weight: float = 0.01
    
    # Adaptive subdivision
    enable_subdivision: bool = True
    subdivision_start_epoch: int = 10
    subdivision_interval: int = 5
    subdivision_threshold: float = 0.01
    max_subdivision_level: int = 12
    
    # Pruning
    enable_pruning: bool = True
    pruning_start_epoch: int = 20
    pruning_interval: int = 10
    pruning_threshold: float = 0.001
    
    # Rendering settings
    render_batch_size: int = 4096
    render_chunk_size: int = 1024
    
    # Advanced training options
    gradient_clip_val: float = 1.0
    use_ema: bool = True
    ema_decay: float = 0.999


class SVRasterLightningModule(pl.LightningModule):
    """PyTorch Lightning module for SVRaster training."""
    
    def __init__(self, config: SVRasterLightningConfig):
        super().__init__()
        self.config = config
        self.save_hyperparameters()
        
        # Initialize model and loss
        self.model = SVRasterModel(config.model_config)
        self.loss_fn = SVRasterLoss(config.model_config)
        
        # Initialize metrics
        self.train_psnr = torchmetrics.image.PeakSignalNoiseRatio()
        self.val_psnr = torchmetrics.image.PeakSignalNoiseRatio()
        self.val_ssim = torchmetrics.image.StructuralSimilarityIndexMeasure()
        # Note: LPIPS requires lpips package, using a placeholder for now
        try:
            self.val_lpips = torchmetrics.image.LearnedPerceptualImagePatchSimilarity()
        except:
            self.val_lpips = None
        
        # EMA model for better validation performance
        if config.use_ema:
            self.ema_model = SVRasterModel(config.model_config)
            self.ema_model.load_state_dict(self.model.state_dict())
            self.ema_model.eval()
            for param in self.ema_model.parameters():
                param.requires_grad_(False)
        else:
            self.ema_model = None
        
        # Training state
        self.automatic_optimization = True
        
    def forward(
        self,
        ray_origins: torch.Tensor,
        ray_directions: torch.Tensor,
        camera_params: Optional[dict[str,
        torch.Tensor]] = None,
    )
        """Forward pass through the model."""
        return self.model(ray_origins, ray_directions, camera_params)
    
    def training_step(self, batch: dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        """Training step."""
        # Extract batch data
        ray_origins = batch['ray_origins']  # [B, N, 3]
        ray_directions = batch['ray_directions']  # [B, N, 3]
        target_colors = batch['colors']  # [B, N, 3]
        target_depth = batch.get('depth', None)  # [B, N] optional
        
        # Flatten for processing
        B, N = ray_origins.shape[:2]
        ray_origins = ray_origins.view(-1, 3)
        ray_directions = ray_directions.view(-1, 3)
        target_colors = target_colors.view(-1, 3)
        if target_depth is not None:
            target_depth = target_depth.view(-1)
        
        # Forward pass
        outputs = self.model(ray_origins, ray_directions)
        
        # Compute losses
        targets = {'colors': target_colors}
        if target_depth is not None:
            targets['depth'] = target_depth
            
        losses = self.loss_fn(outputs, targets, self.model)
        
        # Log training metrics
        self.log('train/loss', losses['total_loss'], on_step=True, on_epoch=True, prog_bar=True)
        self.log('train/rgb_loss', losses['rgb_loss'], on_step=True, on_epoch=True)
        
        if 'depth_loss' in losses:
            self.log('train/depth_loss', losses['depth_loss'], on_step=True, on_epoch=True)
        if 'opacity_reg' in losses:
            self.log('train/opacity_reg', losses['opacity_reg'], on_step=True, on_epoch=True)
        
        # Compute PSNR
        psnr = self.train_psnr(outputs['rgb'], target_colors)
        self.log('train/psnr', psnr, on_step=True, on_epoch=True, prog_bar=True)
        
        # Log voxel statistics
        if batch_idx % 100 == 0:
            voxel_stats = self.model.get_voxel_statistics()
            for key, value in voxel_stats.items():
                self.log(f'voxels/{key}', value, on_step=True)
        
        return losses['total_loss']
    
    def validation_step(
        self,
        batch: dict[str,
        torch.Tensor],
        batch_idx: int,
    )
        """Validation step."""
        # Use EMA model if available
        model = self.ema_model if self.ema_model is not None else self.model
        
        # Extract batch data
        ray_origins = batch['ray_origins']
        ray_directions = batch['ray_directions']
        target_colors = batch['colors']
        target_depth = batch.get('depth', None)
        
        # Flatten for processing
        B, N = ray_origins.shape[:2]
        ray_origins = ray_origins.view(-1, 3)
        ray_directions = ray_directions.view(-1, 3)
        target_colors = target_colors.view(-1, 3)
        
        # Forward pass
        with torch.no_grad():
            outputs = model(ray_origins, ray_directions)
        
        # Compute losses
        targets = {'colors': target_colors}
        if target_depth is not None:
            targets['depth'] = target_depth.view(-1)
            
        losses = self.loss_fn(outputs, targets, model)
        
        # Compute metrics
        psnr = self.val_psnr(outputs['rgb'], target_colors)
        
        # For image-level metrics, reshape outputs
        pred_image = outputs['rgb'].view(B, N, 3)
        target_image = target_colors.view(B, N, 3)
        
        # Assuming square images for SSIM/LPIPS
        H = W = int(np.sqrt(N))
        if H * W == N:
            pred_img = pred_image.view(B, H, W, 3).permute(0, 3, 1, 2)  # [B, 3, H, W]
            target_img = target_image.view(B, H, W, 3).permute(0, 3, 1, 2)
            
            ssim = self.val_ssim(pred_img, target_img)
            lpips = self.val_lpips(
                pred_img,
                target_img,
            )
        else:
            ssim = torch.tensor(0.0, device=self.device)
            lpips = torch.tensor(0.0, device=self.device)
        
        return {
            'val_loss': losses['total_loss'], 'val_psnr': psnr, 'val_ssim': ssim, 'val_lpips': lpips, 'pred_rgb': outputs['rgb'][:100], # Log first 100 pixels
            'target_rgb': target_colors[:100]
        }
    
    def validation_epoch_end(self, outputs: list[dict[str, torch.Tensor]]) -> None:
        """Aggregate validation results."""
        # Average metrics
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        avg_psnr = torch.stack([x['val_psnr'] for x in outputs]).mean()
        avg_ssim = torch.stack([x['val_ssim'] for x in outputs]).mean()
        avg_lpips = torch.stack([x['val_lpips'] for x in outputs]).mean()
        
        # Log metrics
        self.log('val/loss', avg_loss, prog_bar=True)
        self.log('val/psnr', avg_psnr, prog_bar=True)
        self.log('val/ssim', avg_ssim, prog_bar=True)
        self.log('val/lpips', avg_lpips, prog_bar=True)
        
        # Log voxel statistics
        voxel_stats = self.model.get_voxel_statistics()
        for key, value in voxel_stats.items():
            self.log(f'val_voxels/{key}', value)
    
    def configure_optimizers(self) -> dict[str, Any]:
        """Configure optimizers and schedulers."""
        # Group parameters for different learning rates
        voxel_params = []
        other_params = []
        
        for name, param in self.model.named_parameters():
            if 'sparse_voxels' in name:
                voxel_params.append(param)
            else:
                other_params.append(param)
        
        param_groups = [
            {
                'params': voxel_params,
                'lr': self.config.learning_rate,
                'name': 'voxels',
            }
        ]
        
        # Create optimizer
        if self.config.optimizer_type == "adam":
            optimizer = torch.optim.Adam(
                param_groups,
                lr=self.config.learning_rate,
                weight_decay=self.config.weight_decay,
            )
        elif self.config.optimizer_type == "adamw":
            optimizer = torch.optim.AdamW(
                param_groups,
                lr=self.config.learning_rate,
                weight_decay=self.config.weight_decay,
            )
        elif self.config.optimizer_type == "sgd":
            optimizer = torch.optim.SGD(
                param_groups,
                lr=self.config.learning_rate,
                momentum=0.9,
                weight_decay=self.config.weight_decay,
            )
        else:
            raise ValueError(f"Unsupported optimizer: {self.config.optimizer_type}")
        
        # Configure scheduler
        if self.config.scheduler_type == "none":
            return optimizer
        
        scheduler_config = {"optimizer": optimizer}
        
        if self.config.scheduler_type == "cosine":
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=self.trainer.max_epochs)
            scheduler_config["lr_scheduler"] = scheduler
            scheduler_config["interval"] = "epoch"
            
        elif self.config.scheduler_type == "step":
            params = self.config.scheduler_params or {"step_size": 30, "gamma": 0.1}
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, **params)
            scheduler_config["lr_scheduler"] = scheduler
            scheduler_config["interval"] = "epoch"
            
        elif self.config.scheduler_type == "exponential":
            params = self.config.scheduler_params or {"gamma": 0.95}
            scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, **params)
            scheduler_config["lr_scheduler"] = scheduler
            scheduler_config["interval"] = "epoch"
        
        return scheduler_config
    
    def on_train_epoch_end(self) -> None:
        """Called at the end of each training epoch."""
        # Update EMA model
        if self.ema_model is not None:
            self._update_ema_model()
        
        # Adaptive subdivision
        if (self.config.enable_subdivision and 
            self.current_epoch >= self.config.subdivision_start_epoch and
            self.current_epoch % self.config.subdivision_interval == 0):
            self._perform_subdivision()
        
        # Pruning
        if (self.config.enable_pruning and 
            self.current_epoch >= self.config.pruning_start_epoch and
            self.current_epoch % self.config.pruning_interval == 0):
            self._perform_pruning()
    
    def _update_ema_model(self) -> None:
        """Update EMA model parameters."""
        decay = self.config.ema_decay
        with torch.no_grad():
            for ema_param, param in zip(self.ema_model.parameters(), self.model.parameters()):
                ema_param.data.mul_(decay).add_(param.data, alpha=1 - decay)
    
    def _perform_subdivision(self) -> None:
        """Perform adaptive voxel subdivision."""
        # This is a simplified version - you would implement more sophisticated
        # subdivision criteria based on gradients or reconstruction error
        logger.info(f"Performing voxel subdivision at epoch {self.current_epoch}")
        
        # Example: subdivide voxels with high gradient magnitude
        # In practice, you'd analyze gradients from recent training steps
        for level_idx in range(len(self.model.sparse_voxels.voxel_positions)):
            num_voxels = self.model.sparse_voxels.voxel_positions[level_idx].shape[0]
            if num_voxels > 0:
                # Simple random subdivision for demonstration
                subdivision_mask = torch.rand(num_voxels) < self.config.subdivision_threshold
                if subdivision_mask.any():
                    self.model.sparse_voxels.subdivide_voxels(subdivision_mask, level_idx)
                    logger.info(f"Subdivided {subdivision_mask.sum().item()} voxels at level {level_idx}")
    
    def _perform_pruning(self) -> None:
        """Perform voxel pruning."""
        logger.info(f"Performing voxel pruning at epoch {self.current_epoch}")
        initial_count = self.model.sparse_voxels.get_total_voxel_count()
        self.model.sparse_voxels.prune_voxels(self.config.pruning_threshold)
        final_count = self.model.sparse_voxels.get_total_voxel_count()
        logger.info(f"Pruned {initial_count - final_count} voxels")


def create_lightning_trainer(
    config: SVRasterLightningConfig, train_dataset: SVRasterDataset, val_dataset: Optional[SVRasterDataset] = None, max_epochs: int = 100, gpus: int | list[int] = 1, logger_type: str = "tensorboard", # tensorboard, wandb
    project_name: str = "svraster", experiment_name: str = "default", checkpoint_dir: str = "checkpoints", **trainer_kwargs
) -> tuple[SVRasterLightningModule, pl.Trainer]:
    """
    Create SVRaster Lightning module and trainer.
    
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
    lightning_module = SVRasterLightningModule(config)
    
    # Setup logger
    if logger_type == "tensorboard":
        logger = TensorBoardLogger(
            save_dir="logs", name=project_name, version=experiment_name
        )
    elif logger_type == "wandb":
        logger = WandbLogger(
            project=project_name, name=experiment_name, save_dir="logs"
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
            monitor="val/psnr", mode="max", patience=20, verbose=True
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
        max_epochs=max_epochs, devices=gpus, logger=logger, callbacks=callbacks, strategy=strategy, gradient_clip_val=config.gradient_clip_val, precision="16-mixed", # Use mixed precision for efficiency
        log_every_n_steps=50, val_check_interval=0.5, # Validate twice per epoch
        **trainer_kwargs
    )
    
    return lightning_module, trainer


def train_svraster_lightning(
    model_config: SVRasterConfig, lightning_config: SVRasterLightningConfig, train_dataset: SVRasterDataset, val_dataset: Optional[SVRasterDataset] = None, **trainer_kwargs
) -> SVRasterLightningModule:
    """
    Simplified training function using Lightning.
    
    Args:
        model_config: SVRaster model configuration
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
    lightning_module, trainer = create_lightning_trainer(
        lightning_config, train_dataset, val_dataset, **trainer_kwargs
    )
    
    # Create data loaders
    from torch.utils.data import DataLoader
    
    train_loader = DataLoader(
        train_dataset, batch_size=1, # SVRaster typically processes one image at a time
        shuffle=True, num_workers=4, pin_memory=True
    )
    
    val_loader = None
    if val_dataset is not None:
        val_loader = DataLoader(
            val_dataset, batch_size=1, shuffle=False, num_workers=2, pin_memory=True
        )
    
    # Start training
    trainer.fit(lightning_module, train_loader, val_loader)
    
    return lightning_module 