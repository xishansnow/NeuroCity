"""
Training module for DNMP-NeRF.

This module implements the two-stage training pipeline:
1. Geometry optimization using pre-trained mesh autoencoder
2. Radiance field training with fixed geometry
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import os
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path
import time
from tqdm import tqdm
import json

from .core import DNMP, DNMPConfig
from .mesh_autoencoder import MeshAutoEncoder
from .rasterizer import DNMPRasterizer, RasterizationConfig
from .dataset import DNMPDataset
from .utils import evaluation_utils, rendering_utils


class DNMPTrainer:
    """Base trainer class for DNMP."""
    
    def __init__(self,
                 model: DNMP,
                 config: DNMPConfig,
                 device: torch.device = None,
                 log_dir: str = './logs',
                 checkpoint_dir: str = './checkpoints'):
        
        self.model = model
        self.config = config
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.log_dir = Path(log_dir)
        self.checkpoint_dir = Path(checkpoint_dir)
        
        # Create directories
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # Move model to device
        self.model = self.model.to(self.device)
        
        # Initialize logging
        self.writer = SummaryWriter(self.log_dir)
        self.global_step = 0
        self.epoch = 0
        
        # Training metrics
        self.train_losses = []
        self.val_losses = []
        self.best_val_loss = float('inf')
        
        # Initialize rasterizer
        raster_config = RasterizationConfig()
        self.rasterizer = DNMPRasterizer(raster_config).to(self.device)
    
    def setup_optimizers(self, learning_rate: float = None):
        """Setup optimizers for training."""
        lr = learning_rate or self.config.geometry_lr
        
        # Separate optimizers for different components
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr, 
                                   weight_decay=self.config.weight_decay)
        
        # Learning rate scheduler
        self.scheduler = optim.lr_scheduler.StepLR(
            self.optimizer, step_size=1000, gamma=0.9
        )
    
    def train_epoch(self, dataloader: DataLoader) -> Dict[str, float]:
        """Train for one epoch."""
        self.model.train()
        epoch_losses = {}
        num_batches = len(dataloader)
        
        with tqdm(dataloader, desc=f'Epoch {self.epoch}') as pbar:
            for batch_idx, batch in enumerate(pbar):
                # Move batch to device
                batch = self.move_batch_to_device(batch)
                
                # Forward pass
                loss_dict = self.train_step(batch)
                
                # Update metrics
                for key, value in loss_dict.items():
                    if key not in epoch_losses:
                        epoch_losses[key] = 0.0
                    epoch_losses[key] += value
                
                # Update progress bar
                pbar.set_postfix({
                    'loss': f"{loss_dict.get('total_loss', 0.0):.6f}",
                    'lr': f"{self.optimizer.param_groups[0]['lr']:.6f}"
                })
                
                # Log to tensorboard
                if batch_idx % 100 == 0:
                    for key, value in loss_dict.items():
                        self.writer.add_scalar(f'train/{key}', value, self.global_step)
                    
                    self.writer.add_scalar('train/learning_rate', 
                                         self.optimizer.param_groups[0]['lr'], 
                                         self.global_step)
                
                self.global_step += 1
        
        # Average losses over epoch
        for key in epoch_losses:
            epoch_losses[key] /= num_batches
        
        return epoch_losses
    
    def train_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """Single training step."""
        self.optimizer.zero_grad()
        
        # Extract batch data
        rays_o = batch['rays_o'].view(-1, 3)  # [N, 3]
        rays_d = batch['rays_d'].view(-1, 3)  # [N, 3]
        target_rgb = batch['image'].view(-1, 3)  # [N, 3]
        
        # Forward pass
        outputs = self.model(rays_o, rays_d, self.rasterizer)
        
        # Prepare targets
        targets = {'rgb': target_rgb}
        if batch['depth'] is not None:
            targets['depth'] = batch['depth'].view(-1)
        
        # Compute loss
        loss_dict = self.model.compute_loss(outputs, targets)
        
        # Backward pass
        loss_dict['total_loss'].backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        
        self.optimizer.step()
        self.scheduler.step()
        
        # Return scalar losses
        return {key: value.item() for key, value in loss_dict.items()}
    
    def validate(self, dataloader: DataLoader) -> Dict[str, float]:
        """Validate model."""
        self.model.eval()
        val_losses = {}
        num_batches = len(dataloader)
        
        with torch.no_grad():
            for batch in tqdm(dataloader, desc='Validation'):
                batch = self.move_batch_to_device(batch)
                
                # Forward pass
                rays_o = batch['rays_o'].view(-1, 3)
                rays_d = batch['rays_d'].view(-1, 3)
                target_rgb = batch['image'].view(-1, 3)
                
                outputs = self.model(rays_o, rays_d, self.rasterizer)
                
                targets = {'rgb': target_rgb}
                if batch['depth'] is not None:
                    targets['depth'] = batch['depth'].view(-1)
                
                loss_dict = self.model.compute_loss(outputs, targets)
                
                # Accumulate losses
                for key, value in loss_dict.items():
                    if key not in val_losses:
                        val_losses[key] = 0.0
                    val_losses[key] += value.item()
        
        # Average losses
        for key in val_losses:
            val_losses[key] /= num_batches
        
        return val_losses
    
    def move_batch_to_device(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        """Move batch data to device."""
        device_batch = {}
        
        for key, value in batch.items():
            if isinstance(value, torch.Tensor):
                device_batch[key] = value.to(self.device)
            elif isinstance(value, dict):
                device_batch[key] = {}
                for sub_key, sub_value in value.items():
                    if isinstance(sub_value, torch.Tensor):
                        device_batch[key][sub_key] = sub_value.to(self.device)
                    else:
                        device_batch[key][sub_key] = sub_value
            else:
                device_batch[key] = value
        
        return device_batch
    
    def save_checkpoint(self, epoch: int, is_best: bool = False):
        """Save training checkpoint."""
        checkpoint = {
            'epoch': epoch,
            'global_step': self.global_step,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'config': self.config,
            'best_val_loss': self.best_val_loss
        }
        
        checkpoint_path = self.checkpoint_dir / f'checkpoint_epoch_{epoch:04d}.pth'
        torch.save(checkpoint, checkpoint_path)
        
        if is_best:
            best_path = self.checkpoint_dir / 'best_model.pth'
            torch.save(checkpoint, best_path)
        
        # Keep only last 5 checkpoints
        checkpoints = sorted(self.checkpoint_dir.glob('checkpoint_epoch_*.pth'))
        if len(checkpoints) > 5:
            for old_checkpoint in checkpoints[:-5]:
                old_checkpoint.unlink()
    
    def load_checkpoint(self, checkpoint_path: str):
        """Load training checkpoint."""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        self.epoch = checkpoint['epoch']
        self.global_step = checkpoint['global_step']
        self.best_val_loss = checkpoint['best_val_loss']
        
        print(f"Loaded checkpoint from epoch {self.epoch}")


class GeometryTrainer(DNMPTrainer):
    """Trainer for geometry optimization stage."""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.setup_optimizers(self.config.geometry_lr)
    
    def train(self, 
              train_dataset: DNMPDataset,
              val_dataset: Optional[DNMPDataset] = None,
              num_epochs: int = 1000,
              batch_size: int = 1,
              save_freq: int = 100):
        """Train geometry optimization stage."""
        
        # Initialize scene with point cloud
        if train_dataset.point_cloud is not None:
            self.model.initialize_scene(train_dataset.point_cloud)
        else:
            raise ValueError("Point cloud required for geometry optimization")
        
        # Create data loaders
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False) if val_dataset else None
        
        print(f"Starting geometry optimization for {num_epochs} epochs")
        
        for epoch in range(num_epochs):
            self.epoch = epoch
            
            # Train
            train_losses = self.train_epoch(train_loader)
            self.train_losses.append(train_losses)
            
            # Validate
            if val_loader is not None:
                val_losses = self.validate(val_loader)
                self.val_losses.append(val_losses)
                
                # Check if best model
                current_val_loss = val_losses['total_loss']
                is_best = current_val_loss < self.best_val_loss
                if is_best:
                    self.best_val_loss = current_val_loss
                
                # Log validation metrics
                for key, value in val_losses.items():
                    self.writer.add_scalar(f'val/{key}', value, epoch)
            else:
                is_best = False
            
            # Save checkpoint
            if epoch % save_freq == 0 or is_best:
                self.save_checkpoint(epoch, is_best)
            
            # Print progress
            print(f"Epoch {epoch}: Train Loss = {train_losses['total_loss']:.6f}")
            if val_loader is not None:
                print(f"           Val Loss = {val_losses['total_loss']:.6f}")


class RadianceTrainer(DNMPTrainer):
    """Trainer for radiance field training stage."""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.setup_optimizers(self.config.radiance_lr)
    
    def load_geometry_checkpoint(self, geometry_checkpoint_path: str):
        """Load pre-trained geometry."""
        checkpoint = torch.load(geometry_checkpoint_path, map_location=self.device)
        
        # Load only geometry-related parameters
        model_state_dict = checkpoint['model_state_dict']
        
        # Filter out radiance MLP parameters
        geometry_state_dict = {}
        for key, value in model_state_dict.items():
            if not key.startswith('renderer.radiance_mlp'):
                geometry_state_dict[key] = value
        
        # Load geometry parameters
        self.model.load_state_dict(geometry_state_dict, strict=False)
        
        # Freeze geometry parameters
        for name, param in self.model.named_parameters():
            if not name.startswith('renderer.radiance_mlp'):
                param.requires_grad = False
        
        print("Loaded pre-trained geometry and froze parameters")
    
    def train(self,
              train_dataset: DNMPDataset,
              val_dataset: Optional[DNMPDataset] = None,
              num_epochs: int = 2000,
              batch_size: int = 1,
              save_freq: int = 100):
        """Train radiance field stage."""
        
        # Create data loaders
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False) if val_dataset else None
        
        print(f"Starting radiance field training for {num_epochs} epochs")
        
        for epoch in range(num_epochs):
            self.epoch = epoch
            
            # Train
            train_losses = self.train_epoch(train_loader)
            self.train_losses.append(train_losses)
            
            # Validate
            if val_loader is not None:
                val_losses = self.validate(val_loader)
                self.val_losses.append(val_losses)
                
                current_val_loss = val_losses['total_loss']
                is_best = current_val_loss < self.best_val_loss
                if is_best:
                    self.best_val_loss = current_val_loss
                
                for key, value in val_losses.items():
                    self.writer.add_scalar(f'val/{key}', value, epoch)
            else:
                is_best = False
            
            # Save checkpoint
            if epoch % save_freq == 0 or is_best:
                self.save_checkpoint(epoch, is_best)
            
            # Render validation images
            if epoch % (save_freq * 2) == 0 and val_loader is not None:
                self.render_validation_images(val_loader, epoch)
            
            print(f"Epoch {epoch}: Train Loss = {train_losses['total_loss']:.6f}")
            if val_loader is not None:
                print(f"           Val Loss = {val_losses['total_loss']:.6f}")
    
    def render_validation_images(self, val_loader: DataLoader, epoch: int):
        """Render validation images for visualization."""
        self.model.eval()
        
        with torch.no_grad():
            for i, batch in enumerate(val_loader):
                if i >= 3:  # Render only first 3 validation images
                    break
                
                batch = self.move_batch_to_device(batch)
                
                rays_o = batch['rays_o'].view(-1, 3)
                rays_d = batch['rays_d'].view(-1, 3)
                
                # Render in chunks to avoid memory issues
                chunk_size = 1024
                rendered_colors = []
                
                for j in range(0, len(rays_o), chunk_size):
                    rays_o_chunk = rays_o[j:j+chunk_size]
                    rays_d_chunk = rays_d[j:j+chunk_size]
                    
                    outputs = self.model(rays_o_chunk, rays_d_chunk, self.rasterizer)
                    rendered_colors.append(outputs['rgb'])
                
                rendered_image = torch.cat(rendered_colors, dim=0)
                rendered_image = rendered_image.view(*batch['image'].shape)
                
                # Log images to tensorboard
                self.writer.add_image(f'val/rendered_{i}', 
                                    rendered_image.permute(2, 0, 1), epoch)
                self.writer.add_image(f'val/target_{i}', 
                                    batch['image'].squeeze().permute(2, 0, 1), epoch)


class TwoStageTrainer:
    """Complete two-stage training pipeline."""
    
    def __init__(self,
                 model: DNMP,
                 config: DNMPConfig,
                 device: torch.device = None,
                 log_dir: str = './logs',
                 checkpoint_dir: str = './checkpoints'):
        
        self.model = model
        self.config = config
        self.device = device
        self.log_dir = Path(log_dir)
        self.checkpoint_dir = Path(checkpoint_dir)
        
        # Create stage-specific directories
        self.geometry_log_dir = self.log_dir / 'geometry'
        self.radiance_log_dir = self.log_dir / 'radiance'
        self.geometry_checkpoint_dir = self.checkpoint_dir / 'geometry'
        self.radiance_checkpoint_dir = self.checkpoint_dir / 'radiance'
        
        for dir_path in [self.geometry_log_dir, self.radiance_log_dir,
                        self.geometry_checkpoint_dir, self.radiance_checkpoint_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
    
    def train(self,
              train_dataset: DNMPDataset,
              val_dataset: Optional[DNMPDataset] = None,
              geometry_epochs: int = 1000,
              radiance_epochs: int = 2000,
              batch_size: int = 1):
        """Run complete two-stage training."""
        
        print("=" * 60)
        print("STAGE 1: GEOMETRY OPTIMIZATION")
        print("=" * 60)
        
        # Stage 1: Geometry optimization
        geometry_trainer = GeometryTrainer(
            self.model, self.config, self.device,
            str(self.geometry_log_dir), str(self.geometry_checkpoint_dir)
        )
        
        geometry_trainer.train(
            train_dataset, val_dataset,
            num_epochs=geometry_epochs,
            batch_size=batch_size
        )
        
        # Save geometry checkpoint
        geometry_checkpoint_path = self.geometry_checkpoint_dir / 'final_geometry.pth'
        geometry_trainer.save_checkpoint(geometry_epochs, is_best=True)
        
        print("=" * 60)
        print("STAGE 2: RADIANCE FIELD TRAINING")
        print("=" * 60)
        
        # Stage 2: Radiance field training
        radiance_trainer = RadianceTrainer(
            self.model, self.config, self.device,
            str(self.radiance_log_dir), str(self.radiance_checkpoint_dir)
        )
        
        # Load pre-trained geometry
        best_geometry_path = self.geometry_checkpoint_dir / 'best_model.pth'
        if best_geometry_path.exists():
            radiance_trainer.load_geometry_checkpoint(str(best_geometry_path))
        
        radiance_trainer.train(
            train_dataset, val_dataset,
            num_epochs=radiance_epochs,
            batch_size=batch_size
        )
        
        print("=" * 60)
        print("TRAINING COMPLETED")
        print("=" * 60)
        
        return {
            'geometry_trainer': geometry_trainer,
            'radiance_trainer': radiance_trainer
        }


def create_trainer(stage: str, *args, **kwargs):
    """
    Factory function to create trainers.
    
    Args:
        stage: Training stage ('geometry', 'radiance', 'two_stage')
        *args, **kwargs: Trainer-specific arguments
        
    Returns:
        Trainer instance
    """
    if stage.lower() == 'geometry':
        return GeometryTrainer(*args, **kwargs)
    elif stage.lower() == 'radiance':
        return RadianceTrainer(*args, **kwargs)
    elif stage.lower() == 'two_stage':
        return TwoStageTrainer(*args, **kwargs)
    else:
        raise ValueError(f"Unknown training stage: {stage}") 