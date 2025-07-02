from typing import Any, Optional
"""
Trainer module for SVRaster.
"""

import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from dataclasses import dataclass
import logging
from tqdm import tqdm
import torch.nn.functional as F

from .core import SVRasterModel, SVRasterConfig, SVRasterLoss
from .dataset import SVRasterDataset, SVRasterDatasetConfig

logger = logging.getLogger(__name__)

@dataclass
class SVRasterTrainerConfig:
    """Configuration for SVRaster trainer."""
    
    # Training parameters
    num_epochs: int = 100
    batch_size: int = 1
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    
    # Optimizer settings
    optimizer_type: str = "adam"
    scheduler_type: str = "cosine"
    scheduler_params: Optional[dict[str, Any]] = None
    
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
    
    # Validation and logging
    val_interval: int = 5
    log_interval: int = 100
    save_interval: int = 1000
    
    # Paths
    checkpoint_dir: str = "checkpoints"
    log_dir: str = "logs"
    
    # Device and precision
    device: str = "cuda"
    use_mixed_precision: bool = True
    gradient_clip_norm: float = 1.0
    
    # Rendering settings
    render_batch_size: int = 4096
    render_chunk_size: int = 1024
    
    def __post_init__(self):
        """Set default values after initialization."""
        if self.scheduler_params is None:
            self.scheduler_params = {}

class SVRasterTrainer:
    """Main trainer class for SVRaster model."""
    
    def __init__(
        self,
        model_config: SVRasterConfig,
        trainer_config: SVRasterTrainerConfig,
        train_dataset: SVRasterDataset,
        val_dataset: Optional[SVRasterDataset] = None,
    ) -> None:
        self.model_config = model_config
        self.config = trainer_config
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        
        # Setup device
        self.device = torch.device(trainer_config.device)
        
        # Initialize model
        self.model = SVRasterModel(model_config).to(self.device)
        self.loss_fn = SVRasterLoss(model_config)
        
        # Setup optimizer
        self._setup_optimizer()
        
        # Setup data loaders
        self._setup_data_loaders()
        
        # Setup logging
        self._setup_logging()
        
        # Mixed precision training
        if self.config.use_mixed_precision:
            self.scaler = torch.cuda.amp.GradScaler()
        
        # Training state
        self.current_epoch = 0
        self.global_step = 0
        self.best_val_psnr = 0.0
        
        logger.info(f"Initialized SVRaster trainer")
    
    def _setup_optimizer(self):
        """Setup optimizer and scheduler."""
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
            }
        ]
        
        # Create optimizer
        if self.config.optimizer_type == "adam":
            self.optimizer = optim.Adam(
                param_groups,
                lr=self.config.learning_rate,
                weight_decay=self.config.weight_decay,
            )
        elif self.config.optimizer_type == "adamw":
            self.optimizer = optim.AdamW(
                param_groups,
                lr=self.config.learning_rate,
                weight_decay=self.config.weight_decay,
            )
        else:
            raise ValueError(f"Unsupported optimizer: {self.config.optimizer_type}")
        
        # Create scheduler
        if self.config.scheduler_type == "cosine":
            self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer, T_max=self.config.num_epochs)
        else:
            self.scheduler = None
    
    def _setup_data_loaders(self):
        """Setup data loaders."""
        self.train_loader = DataLoader(
            self.train_dataset, batch_size=self.config.batch_size, shuffle=True, num_workers=4, pin_memory=True, drop_last=True
        )
        
        if self.val_dataset is not None:
            self.val_loader = DataLoader(
                self.val_dataset, batch_size=1, shuffle=False, num_workers=2, pin_memory=True
            )
    
    def _setup_logging(self):
        """Setup logging and checkpointing."""
        os.makedirs(self.config.checkpoint_dir, exist_ok=True)
        os.makedirs(self.config.log_dir, exist_ok=True)
        
        self.writer = SummaryWriter(self.config.log_dir)
    
    def train(self):
        """Main training loop."""
        logger.info("Starting SVRaster training...")
        
        for epoch in range(self.current_epoch, self.config.num_epochs):
            self.current_epoch = epoch
            
            # Training
            train_metrics = self._train_epoch()
            
            # Validation
            if self.val_dataset is not None and epoch % self.config.val_interval == 0:
                val_metrics = self._validate_epoch()
                
                # Save best model
                if val_metrics.get('psnr', 0) > self.best_val_psnr:
                    self.best_val_psnr = val_metrics['psnr']
                    self._save_checkpoint('best_model.pth')
            
            # Adaptive subdivision
            if (self.config.enable_subdivision and 
                epoch >= self.config.subdivision_start_epoch and
                epoch % self.config.subdivision_interval == 0):
                self._perform_subdivision()
            
            # Voxel pruning
            if (self.config.enable_pruning and 
                epoch >= self.config.pruning_start_epoch and
                epoch % self.config.pruning_interval == 0):
                self._perform_pruning()
            
            # Scheduler step
            if self.scheduler is not None:
                self.scheduler.step()
            
            # Save checkpoint
            if epoch % self.config.save_interval == 0:
                self._save_checkpoint(f'checkpoint_epoch_{epoch}.pth')
            
            # Log epoch summary
            logger.info(f"Epoch {epoch}: Train Loss: {train_metrics.get('loss', 0):.4f}")
        
        logger.info("Training completed!")
        self._save_checkpoint('final_model.pth')
    
    def _train_epoch(self) -> dict[str, float]:
        """Train for one epoch."""
        self.model.train()
        
        total_loss = 0.0
        num_batches = 0
        
        pbar = tqdm(self.train_loader, desc=f"Epoch {self.current_epoch}")
        
        for batch_idx, batch in enumerate(pbar):
            # Move to device
            rays_o = batch['rays_o'].to(self.device)
            rays_d = batch['rays_d'].to(self.device)
            target_colors = batch['colors'].to(self.device)
            
            # Forward pass
            outputs = self.model(rays_o, rays_d)
            losses = self.loss_fn(outputs, {'colors': target_colors})
            loss = losses['total_loss']
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.gradient_clip_norm)
            self.optimizer.step()
            
            # Update metrics
            total_loss += loss.item()
            num_batches += 1
            self.global_step += 1
            
            # Logging
            if self.global_step % self.config.log_interval == 0:
                self.writer.add_scalar('train/loss', loss.item(), self.global_step)
                self.writer.add_scalar(
                    'train/lr',
                    self.optimizer.param_groups[0]['lr'],
                    self.global_step,
                )
            
            # Update progress bar
            pbar.set_postfix({'loss': loss.item()})
        
        return {'loss': total_loss / num_batches}
    
    def _validate_epoch(self) -> dict[str, float]:
        """Validate for one epoch."""
        self.model.eval()
        
        total_psnr = 0.0
        num_images = 0
        
        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc="Validation"):
                rays_o = batch['rays_o'].to(self.device)
                rays_d = batch['rays_d'].to(self.device)
                target_colors = batch['colors'].to(self.device)
                
                # Render image
                outputs = self.model(rays_o, rays_d)
                pred_colors = outputs['rgb']
                
                # Compute metrics
                psnr = self._compute_psnr(pred_colors, target_colors)
                total_psnr += psnr
                num_images += 1
        
        avg_psnr = total_psnr / num_images
        
        # Log validation metrics
        self.writer.add_scalar('val/psnr', avg_psnr, self.current_epoch)
        
        return {'psnr': avg_psnr}
    
    def _compute_psnr(self, pred: torch.Tensor, target: torch.Tensor) -> float:
        """Compute PSNR between predicted and target images."""
        mse = torch.mean((pred - target) ** 2)
        psnr = -10 * torch.log10(mse)
        return psnr.item()
    
    def _save_checkpoint(self, filename: str):
        """Save model checkpoint."""
        checkpoint = {
            'epoch': self.current_epoch, 'global_step': self.global_step, 'model_state_dict': self.model.state_dict(
            )
        }
        
        if self.scheduler is not None:
            checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()
        
        checkpoint_path = os.path.join(self.config.checkpoint_dir, filename)
        torch.save(checkpoint, checkpoint_path)
        logger.info(f"Saved checkpoint: {checkpoint_path}")
    
    def load_checkpoint(self, checkpoint_path: str):
        """Load model checkpoint."""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        if self.scheduler is not None and 'scheduler_state_dict' in checkpoint:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        self.current_epoch = checkpoint['epoch']
        self.global_step = checkpoint['global_step']
        self.best_val_psnr = checkpoint['best_val_psnr']
        
        logger.info(f"Loaded checkpoint from epoch {self.current_epoch}")
    
    def _perform_subdivision(self):
        """Perform adaptive voxel subdivision based on training gradients."""
        logger.info(f"Performing voxel subdivision at epoch {self.current_epoch}")
        
        # Compute subdivision criteria based on gradient magnitude
        subdivision_criteria = self._compute_subdivision_criteria()
        
        # Perform adaptive subdivision
        initial_stats = self.model.get_voxel_statistics()
        self.model.adaptive_subdivision(subdivision_criteria)
        final_stats = self.model.get_voxel_statistics()
        
        # Log subdivision results
        total_subdivided = final_stats['total_voxels'] - initial_stats['total_voxels']
        logger.info(f"Subdivided {total_subdivided} voxels")
        
        # Log level-wise statistics
        for level_idx in range(min(len(initial_stats), len(final_stats))):
            level_key = f'level_{level_idx}_voxels'
            if level_key in initial_stats and level_key in final_stats:
                level_change = final_stats[level_key] - initial_stats[level_key]
                if level_change > 0:
                    logger.info(f"Level {level_idx}: +{level_change} voxels")
    
    def _compute_subdivision_criteria(self) -> torch.Tensor:
        """Compute subdivision criteria based on gradient magnitude and reconstruction error."""
        self.model.eval()
        
        # Get a sample batch for gradient computation
        sample_batch = next(iter(self.train_loader))
        rays_o = sample_batch['rays_o'].to(self.device)
        rays_d = sample_batch['rays_d'].to(self.device)
        target_colors = sample_batch['colors'].to(self.device)
        
        # Compute gradients with respect to voxel parameters
        rays_o.requires_grad_(True)
        rays_d.requires_grad_(True)
        
        outputs = self.model(rays_o, rays_d)
        loss = F.mse_loss(outputs['rgb'], target_colors)
        
        # Backward pass to compute gradients
        loss.backward()
        
        # Extract gradient magnitudes for voxel parameters
        criteria = []
        for level_idx in range(len(self.model.sparse_voxels.voxel_positions)):
            if self.model.sparse_voxels.voxel_positions[level_idx].grad is not None:
                # Compute gradient magnitude for each voxel
                grad_magnitude = torch.norm(
                    self.model.sparse_voxels.voxel_positions[level_idx].grad, 
                    dim=1
                )
                criteria.append(grad_magnitude)
            else:
                # If no gradients, use density-based criteria
                densities = self.model.sparse_voxels.voxel_densities[level_idx]
                if self.model_config.density_activation == "exp":
                    density_values = torch.exp(densities)
                else:
                    density_values = F.relu(densities)
                criteria.append(density_values)
        
        # Combine criteria from all levels
        if criteria:
            combined_criteria = torch.cat(criteria)
        else:
            # Fallback: random criteria
            total_voxels = self.model.get_voxel_statistics()['total_voxels']
            combined_criteria = torch.randn(total_voxels, device=self.device)
        
        # Normalize criteria
        if combined_criteria.numel() > 0:
            combined_criteria = (combined_criteria - combined_criteria.mean()) / (combined_criteria.std() + 1e-8)
        
        return combined_criteria
    
    def _perform_pruning(self):
        """Perform voxel pruning based on density threshold."""
        logger.info(f"Performing voxel pruning at epoch {self.current_epoch}")
        
        initial_count = self.model.sparse_voxels.get_total_voxel_count()
        self.model.sparse_voxels.prune_voxels(self.config.pruning_threshold)
        final_count = self.model.sparse_voxels.get_total_voxel_count()
        
        pruned_count = initial_count - final_count
        logger.info(f"Pruned {pruned_count} voxels ({pruned_count/initial_count*100:.1f}%)")
        
        # Log pruning statistics
        if self.writer:
            self.writer.add_scalar('pruning/pruned_voxels', pruned_count, self.current_epoch)
            self.writer.add_scalar('pruning/pruning_ratio', pruned_count/initial_count, self.current_epoch)
            self.writer.add_scalar('pruning/total_voxels', final_count, self.current_epoch)

def create_svraster_trainer(
    model_config: SVRasterConfig,
    trainer_config: SVRasterTrainerConfig,
    train_dataset: SVRasterDataset,
    val_dataset: Optional[SVRasterDataset] = None,
) -> SVRasterTrainer:
    """Create a SVRaster trainer."""
    return SVRasterTrainer(model_config, trainer_config, train_dataset, val_dataset) 