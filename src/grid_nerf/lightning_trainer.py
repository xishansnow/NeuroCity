"""
PyTorch Lightning trainer for Grid-NeRF.

This module provides a Lightning-based training framework for Grid-NeRF models,
optimized for large-scale urban scene reconstruction.
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

from .core import GridNeRF, GridNeRFConfig, GridNeRFLoss
from .dataset import GridNeRFDataset, create_dataloader

logger = logging.getLogger(__name__)


@dataclass
class GridNeRFLightningConfig:
    """Configuration for Grid-NeRF Lightning training."""
    
    # Model config
    model_config: GridNeRFConfig = None
    
    # Training parameters
    learning_rate: float = 5e-4
    grid_lr: float = 1e-3          # Higher learning rate for grid features
    mlp_lr: float = 5e-4           # Learning rate for MLP networks
    weight_decay: float = 1e-6
    optimizer_type: str = "adam"    # adam, adamw, sgd
    
    # Scheduler settings
    scheduler_type: str = "exponential"  # exponential, cosine, step, none
    scheduler_params: Dict[str, Any] = None
    warmup_steps: int = 500
    max_steps: int = 100000
    
    # Loss weights
    color_loss_weight: float = 1.0
    depth_loss_weight: float = 0.1
    grid_reg_weight: float = 0.001
    
    # Grid-specific settings
    grid_update_freq: int = 100     # Update grid features every N steps
    density_threshold: float = 0.01
    enable_grid_pruning: bool = True
    pruning_start_step: int = 5000
    pruning_freq: int = 1000
    
    # Sampling settings
    ray_batch_size: int = 4096
    chunk_size: int = 1024
    num_samples_coarse: int = 64
    num_samples_fine: int = 128
    
    # Advanced training options
    gradient_clip_val: float = 1.0
    use_ema: bool = True
    ema_decay: float = 0.995
    
    # Rendering settings
    render_test_freq: int = 20      # Render test images every N epochs


class GridNeRFLightningModule(pl.LightningModule):
    """PyTorch Lightning module for Grid-NeRF training."""
    
    def __init__(self, config: GridNeRFLightningConfig):
        super().__init__()
        self.config = config
        self.save_hyperparameters()
        
        # Initialize model and loss
        self.model = GridNeRF(config.model_config)
        self.loss_fn = GridNeRFLoss(config.model_config)
        
        # Initialize metrics
        self.train_psnr = torchmetrics.image.PeakSignalNoiseRatio()
        self.val_psnr = torchmetrics.image.PeakSignalNoiseRatio()
        self.val_ssim = torchmetrics.image.StructuralSimilarityIndexMeasure()
        
        # EMA model for better validation performance
        if config.use_ema:
            self.ema_model = GridNeRF(config.model_config)
            self.ema_model.load_state_dict(self.model.state_dict())
            self.ema_model.eval()
            for param in self.ema_model.parameters():
                param.requires_grad_(False)
        else:
            self.ema_model = None
        
        # Training state
        self.automatic_optimization = True
        
        # Grid statistics tracking
        self.grid_update_count = 0
        
    def forward(self, ray_origins: torch.Tensor, ray_directions: torch.Tensor,
                background_color: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """Forward pass through the model."""
        return self.model(ray_origins, ray_directions, background_color)
    
    def training_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        """Training step."""
        # Extract batch data
        images = batch['image']  # [B, H, W, 3]
        rays_o = batch['rays_o']  # [B, H, W, 3]
        rays_d = batch['rays_d']  # [B, H, W, 3]
        target_depth = batch.get('depth', None)  # [B, H, W] optional
        
        # Reshape for processing
        B, H, W, _ = rays_o.shape
        total_rays = B * H * W
        
        # Sample random rays for training efficiency
        if total_rays > self.config.ray_batch_size:
            ray_indices = torch.randperm(total_rays)[:self.config.ray_batch_size]
            
            rays_o_flat = rays_o.view(-1, 3)[ray_indices]
            rays_d_flat = rays_d.view(-1, 3)[ray_indices]
            target_colors = images.view(-1, 3)[ray_indices]
            
            if target_depth is not None:
                target_depth_flat = target_depth.view(-1)[ray_indices]
            else:
                target_depth_flat = None
        else:
            rays_o_flat = rays_o.view(-1, 3)
            rays_d_flat = rays_d.view(-1, 3)
            target_colors = images.view(-1, 3)
            target_depth_flat = target_depth.view(-1) if target_depth is not None else None
        
        # Forward pass
        outputs = self.model(rays_o_flat, rays_d_flat)
        
        # Compute losses
        targets = {'colors': target_colors}
        if target_depth_flat is not None:
            targets['depth'] = target_depth_flat
            
        losses = self.loss_fn(outputs, targets)
        
        # Add grid regularization
        if self.config.grid_reg_weight > 0:
            grid_reg = self._compute_grid_regularization()
            losses['grid_reg'] = self.config.grid_reg_weight * grid_reg
            losses['total_loss'] = losses['total_loss'] + losses['grid_reg']
        
        # Log training metrics
        self.log('train/loss', losses['total_loss'], on_step=True, on_epoch=True, prog_bar=True)
        self.log('train/color_loss', losses['color_loss'], on_step=True, on_epoch=True)
        
        if 'depth_loss' in losses:
            self.log('train/depth_loss', losses['depth_loss'], on_step=True, on_epoch=True)
        if 'grid_reg' in losses:
            self.log('train/grid_reg', losses['grid_reg'], on_step=True, on_epoch=True)
        
        # Compute PSNR
        psnr = self.train_psnr(outputs['rgb'], target_colors)
        self.log('train/psnr', psnr, on_step=True, on_epoch=True, prog_bar=True)
        
        # Log grid statistics periodically
        if batch_idx % 100 == 0:
            self._log_grid_statistics()
        
        return losses['total_loss']
    
    def validation_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> Dict[str, torch.Tensor]:
        """Validation step."""
        # Use EMA model if available
        model = self.ema_model if self.ema_model is not None else self.model
        
        # Extract batch data
        images = batch['image']
        rays_o = batch['rays_o']
        rays_d = batch['rays_d']
        target_depth = batch.get('depth', None)
        
        # Process in chunks to avoid memory issues
        B, H, W, _ = rays_o.shape
        rays_o_flat = rays_o.view(-1, 3)
        rays_d_flat = rays_d.view(-1, 3)
        target_colors = images.view(-1, 3)
        
        # Render in chunks
        chunk_size = self.config.chunk_size
        all_rgb = []
        all_depth = []
        
        with torch.no_grad():
            for i in range(0, rays_o_flat.shape[0], chunk_size):
                end_i = min(i + chunk_size, rays_o_flat.shape[0])
                chunk_rays_o = rays_o_flat[i:end_i]
                chunk_rays_d = rays_d_flat[i:end_i]
                
                chunk_outputs = model(chunk_rays_o, chunk_rays_d)
                all_rgb.append(chunk_outputs['rgb'])
                all_depth.append(chunk_outputs['depth'])
        
        # Concatenate results
        pred_rgb = torch.cat(all_rgb, dim=0)
        pred_depth = torch.cat(all_depth, dim=0)
        
        outputs = {'rgb': pred_rgb, 'depth': pred_depth}
        
        # Compute losses
        targets = {'colors': target_colors}
        if target_depth is not None:
            targets['depth'] = target_depth.view(-1)
            
        losses = self.loss_fn(outputs, targets)
        
        # Compute metrics
        psnr = self.val_psnr(pred_rgb, target_colors)
        
        # Compute SSIM for image-level metrics
        pred_image = pred_rgb.view(B, H, W, 3).permute(0, 3, 1, 2)  # [B, 3, H, W]
        target_image = target_colors.view(B, H, W, 3).permute(0, 3, 1, 2)
        ssim = self.val_ssim(pred_image, target_image)
        
        return {
            'val_loss': losses['total_loss'],
            'val_psnr': psnr,
            'val_ssim': ssim,
            'pred_rgb': pred_rgb[:100],  # Log first 100 pixels
            'target_rgb': target_colors[:100],
            'pred_image': pred_image[0],  # Log first image
            'target_image': target_image[0]
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
        
        # Log grid statistics
        self._log_grid_statistics(prefix='val_')
        
        # Log sample images
        if len(outputs) > 0 and self.current_epoch % 5 == 0:
            self._log_sample_images(outputs[0])
    
    def configure_optimizers(self) -> Dict[str, Any]:
        """Configure optimizers and schedulers."""
        # Group parameters for different learning rates
        grid_params = []
        mlp_params = []
        other_params = []
        
        for name, param in self.model.named_parameters():
            if 'hierarchical_grid' in name or 'grid' in name:
                grid_params.append(param)
            elif 'mlp' in name or 'network' in name:
                mlp_params.append(param)
            else:
                other_params.append(param)
        
        param_groups = [
            {'params': grid_params, 'lr': self.config.grid_lr, 'name': 'grid'},
            {'params': mlp_params, 'lr': self.config.mlp_lr, 'name': 'mlp'},
            {'params': other_params, 'lr': self.config.learning_rate, 'name': 'other'}
        ]
        
        # Create optimizer
        if self.config.optimizer_type == "adam":
            optimizer = torch.optim.Adam(param_groups,
                                       weight_decay=self.config.weight_decay)
        elif self.config.optimizer_type == "adamw":
            optimizer = torch.optim.AdamW(param_groups,
                                        weight_decay=self.config.weight_decay)
        elif self.config.optimizer_type == "sgd":
            optimizer = torch.optim.SGD(param_groups,
                                      momentum=0.9,
                                      weight_decay=self.config.weight_decay)
        else:
            raise ValueError(f"Unsupported optimizer: {self.config.optimizer_type}")
        
        # Configure scheduler
        if self.config.scheduler_type == "none":
            return optimizer
        
        scheduler_config = {"optimizer": optimizer}
        
        if self.config.scheduler_type == "exponential":
            params = self.config.scheduler_params or {"gamma": 0.95}
            scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, **params)
            scheduler_config["lr_scheduler"] = scheduler
            scheduler_config["interval"] = "epoch"
            
        elif self.config.scheduler_type == "cosine":
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=self.trainer.max_epochs)
            scheduler_config["lr_scheduler"] = scheduler
            scheduler_config["interval"] = "epoch"
            
        elif self.config.scheduler_type == "step":
            params = self.config.scheduler_params or {"step_size": 50, "gamma": 0.5}
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, **params)
            scheduler_config["lr_scheduler"] = scheduler
            scheduler_config["interval"] = "epoch"
        
        return scheduler_config
    
    def on_train_epoch_end(self) -> None:
        """Called at the end of each training epoch."""
        # Update EMA model
        if self.ema_model is not None:
            self._update_ema_model()
        
        # Grid updates and pruning
        if self.current_epoch % self.config.grid_update_freq == 0:
            self._update_grid_features()
        
        if (self.config.enable_grid_pruning and 
            self.global_step >= self.config.pruning_start_step and
            self.global_step % self.config.pruning_freq == 0):
            self._prune_grid_features()
    
    def _update_ema_model(self) -> None:
        """Update EMA model parameters."""
        decay = self.config.ema_decay
        with torch.no_grad():
            for ema_param, param in zip(self.ema_model.parameters(), self.model.parameters()):
                ema_param.data.mul_(decay).add_(param.data, alpha=1 - decay)
    
    def _compute_grid_regularization(self) -> torch.Tensor:
        """Compute grid regularization loss."""
        reg_loss = 0.0
        count = 0
        
        for grid in self.model.hierarchical_grid.grids:
            # L2 regularization on grid features
            reg_loss += torch.mean(grid ** 2)
            count += 1
        
        return reg_loss / count if count > 0 else torch.tensor(0.0, device=self.device)
    
    def _log_grid_statistics(self, prefix: str = 'train_'):
        """Log grid-related statistics."""
        # Count occupied cells for each grid level
        for level in range(self.config.model_config.num_grid_levels):
            occupied_cells = self.model.hierarchical_grid.get_occupied_cells(
                level, self.config.density_threshold)
            occupied_ratio = occupied_cells.float().mean()
            
            self.log(f'{prefix}grid/level_{level}_occupied_ratio', 
                    occupied_ratio, on_step=True)
        
        # Log grid feature statistics
        for level, grid in enumerate(self.model.hierarchical_grid.grids):
            feature_mean = torch.mean(torch.abs(grid))
            feature_std = torch.std(grid)
            
            self.log(f'{prefix}grid/level_{level}_feature_mean', feature_mean, on_step=True)
            self.log(f'{prefix}grid/level_{level}_feature_std', feature_std, on_step=True)
    
    def _log_sample_images(self, sample_output: Dict[str, torch.Tensor]):
        """Log sample rendered images to tensorboard."""
        if self.logger is not None and hasattr(self.logger, 'experiment'):
            # Log predicted vs target image
            pred_img = sample_output['pred_image']
            target_img = sample_output['target_image']
            
            # Ensure images are in [0, 1] range
            pred_img = torch.clamp(pred_img, 0, 1)
            target_img = torch.clamp(target_img, 0, 1)
            
            self.logger.experiment.add_image(
                'val/pred_image', pred_img, self.current_epoch
            )
            self.logger.experiment.add_image(
                'val/target_image', target_img, self.current_epoch
            )
    
    def _update_grid_features(self):
        """Update grid features based on occupancy."""
        logger.info(f"Updating grid features at epoch {self.current_epoch}")
        self.model.update_grid_features(self.config.density_threshold)
        self.grid_update_count += 1
        self.log('grid/update_count', self.grid_update_count)
    
    def _prune_grid_features(self):
        """Prune grid features with low occupancy."""
        logger.info(f"Pruning grid features at step {self.global_step}")
        
        # Count features before pruning
        total_features_before = sum(grid.numel() for grid in self.model.hierarchical_grid.grids)
        
        # Perform pruning (implementation depends on specific grid structure)
        for level in range(self.config.model_config.num_grid_levels):
            occupied_mask = self.model.hierarchical_grid.get_occupied_cells(
                level, self.config.density_threshold)
            
            # Zero out features for unoccupied cells
            grid = self.model.hierarchical_grid.grids[level]
            grid.data[~occupied_mask] *= 0.1  # Reduce rather than zero to maintain gradients
        
        # Count features after pruning
        total_features_after = sum(
            torch.sum(torch.abs(grid) > 1e-6).item() 
            for grid in self.model.hierarchical_grid.grids
        )
        
        pruning_ratio = (total_features_before - total_features_after) / total_features_before
        self.log('grid/pruning_ratio', pruning_ratio)
        logger.info(f"Pruned {pruning_ratio:.2%} of grid features")


def create_grid_nerf_lightning_trainer(
    config: GridNeRFLightningConfig,
    train_dataset: GridNeRFDataset,
    val_dataset: Optional[GridNeRFDataset] = None,
    max_epochs: int = 100,
    gpus: Union[int, List[int]] = 1,
    logger_type: str = "tensorboard",
    project_name: str = "grid_nerf",
    experiment_name: str = "default",
    checkpoint_dir: str = "checkpoints",
    **trainer_kwargs
) -> Tuple[GridNeRFLightningModule, pl.Trainer]:
    """
    Create Grid-NeRF Lightning module and trainer.
    
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
    lightning_module = GridNeRFLightningModule(config)
    
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
            tags=["grid-nerf", "nerf", "urban-scenes"]
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
            patience=30,
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
        precision="16-mixed",
        log_every_n_steps=50,
        val_check_interval=0.5,
        **trainer_kwargs
    )
    
    return lightning_module, trainer


def train_grid_nerf_lightning(
    model_config: GridNeRFConfig,
    lightning_config: GridNeRFLightningConfig,
    train_dataset: GridNeRFDataset,
    val_dataset: Optional[GridNeRFDataset] = None,
    **trainer_kwargs
) -> GridNeRFLightningModule:
    """
    Simplified training function for Grid-NeRF using Lightning.
    
    Args:
        model_config: Grid-NeRF model configuration
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
    lightning_module, trainer = create_grid_nerf_lightning_trainer(
        lightning_config,
        train_dataset,
        val_dataset,
        **trainer_kwargs
    )
    
    # Create data loaders
    train_loader = create_dataloader(
        train_dataset,
        batch_size=1,  # Grid-NeRF typically processes one image at a time
        shuffle=True,
        num_workers=4,
        ray_batch_size=lightning_config.ray_batch_size
    )
    
    val_loader = None
    if val_dataset is not None:
        val_loader = create_dataloader(
            val_dataset,
            batch_size=1,
            shuffle=False,
            num_workers=2,
            ray_batch_size=lightning_config.ray_batch_size
        )
    
    # Start training
    trainer.fit(lightning_module, train_loader, val_loader)
    
    return lightning_module 