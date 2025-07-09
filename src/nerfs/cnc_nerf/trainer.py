from __future__ import annotations

from typing import Optional
"""
CNC-NeRF Trainer Module

This module implements training infrastructure for Context-based NeRF Compression.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import time
from pathlib import Path
from dataclasses import dataclass

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False

from .core import CNCNeRF, CNCNeRFConfig
from .dataset import CNCNeRFDataset, CNCNeRFDatasetConfig

@dataclass
class CNCNeRFTrainerConfig:
    """Configuration for CNC-NeRF trainer."""
    
    # Training
    num_epochs: int = 1000
    learning_rate: float = 5e-4
    weight_decay: float = 1e-6
    lr_decay_rate: float = 0.1
    lr_decay_steps: int = 250000
    
    # Loss weights
    rgb_loss_weight: float = 1.0
    compression_loss_weight: float = 0.001
    distortion_loss_weight: float = 0.01
    
    # Validation
    val_every: int = 10
    val_num_rays: int = 1024
    
    # Checkpointing
    save_every: int = 50
    checkpoint_dir: str = "checkpoints"
    
    # Logging
    log_every: int = 100
    use_wandb: bool = False
    project_name: str = "cnc-nerf"
    experiment_name: str = "default"
    
    # Optimization
    grad_clip_norm: float = 1.0

class CNCNeRFTrainer:
    """Trainer for CNC-NeRF model."""
    
    def __init__(
        self,
        model: CNCNeRF,
        train_dataset: CNCNeRFDataset,
        val_dataset: Optional[CNCNeRFDataset] = None,
        config: CNCNeRFTrainerConfig = None,
    ) -> None:
        self.model = model
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.config = config or CNCNeRFTrainerConfig()
        
        # Device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        
        # Optimizer
        self.optimizer = optim.AdamW(
            self.model.parameters(
            )
        )
        
        # Learning rate scheduler
        self.scheduler = optim.lr_scheduler.ExponentialLR(
            self.optimizer, gamma=np.exp(
                np.log,
            )
        )
        
        # Training state
        self.epoch = 0
        self.global_step = 0
        self.best_val_psnr = 0.0
        
        # Checkpointing
        self.checkpoint_dir = Path(self.config.checkpoint_dir)
        self.checkpoint_dir.mkdir(exist_ok=True)
        
        # Setup wandb if available
        if self.config.use_wandb and WANDB_AVAILABLE:
            wandb.init(
                project=self.config.project_name, name=self.config.experiment_name, config=self.config.__dict__
            )
        
        print(f"Trainer initialized on device: {self.device}")
        print(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):, }")
    
    def volume_render(self, rays: torch.Tensor, num_samples: int = 64) -> dict[str, torch.Tensor]:
        """Volume rendering with the CNC model."""
        rays_o, rays_d = rays[..., :3], rays[..., 3:6]
        near, far = rays[..., 6:7], rays[..., 7:8]
        
        # Sample points along rays
        t_vals = torch.linspace(0, 1, num_samples, device=rays.device)
        z_vals = near + (far - near) * t_vals
        
        # Add noise for training
        if self.model.training:
            mids = 0.5 * (z_vals[..., 1:] + z_vals[..., :-1])
            upper = torch.cat([mids, z_vals[..., -1:]], dim=-1)
            lower = torch.cat([z_vals[..., :1], mids], dim=-1)
            t_rand = torch.rand_like(z_vals)
            z_vals = lower + (upper - lower) * t_rand
        
        # Get sample points
        points = rays_o[..., None, :] + rays_d[..., None, :] * z_vals[..., None]
        points_flat = points.reshape(-1, 3)
        
        # Get view directions
        view_dirs = rays_d / torch.norm(rays_d, dim=-1, keepdim=True)
        view_dirs = view_dirs[..., None, :].expand(points.shape)
        view_dirs_flat = view_dirs.reshape(-1, 3)
        
        # Forward pass through model
        output = self.model(points_flat, view_dirs_flat)
        
        # Reshape output
        density = output['density'].reshape(*points.shape[:-1])
        color = output['color'].reshape(*points.shape)
        
        # Volume rendering
        dists = z_vals[..., 1:] - z_vals[..., :-1]
        dists = torch.cat([dists, torch.full_like(dists[..., :1], 1e10)], dim=-1)
        
        alpha = 1.0 - torch.exp(-F.relu(density) * dists)
        transmittance = torch.cumprod(torch.cat([torch.ones_like(alpha[..., :1]), 1.0 - alpha], dim=-1), dim=-1)
        
        weights = alpha * transmittance
        rgb = (weights[..., None] * color).sum(dim=-2)
        depth = (weights * z_vals).sum(dim=-1)
        opacity = weights.sum(dim=-1)
        
        return {
            'rgb': rgb, 'depth': depth, 'opacity': opacity, 'weights': weights, 'z_vals': z_vals
        }
    
    def compute_losses(
        self,
        batch: dict[str,
        torch.Tensor],
        outputs: dict[str,
        torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        """Compute all losses."""
        losses = {}
        
        # RGB reconstruction loss
        rgb_loss = F.mse_loss(outputs['rgb'], batch['colors'])
        losses['rgb_loss'] = rgb_loss
        
        # Compression loss (entropy regularization)
        compression_loss = self.model.compute_compression_loss()
        losses['compression_loss'] = compression_loss
        
        # Distortion loss (optional)
        if self.config.distortion_loss_weight > 0:
            weights = outputs['weights']
            z_vals = outputs['z_vals']
            
            # Compute distortion regularization
            intervals = z_vals[..., 1:] - z_vals[..., :-1]
            intervals = torch.cat([intervals, torch.full_like(intervals[..., :1], 1e10)], dim=-1)
            
            # Distortion loss encourages weights to be localized
            mid_points = 0.5 * (z_vals[..., 1:] + z_vals[..., :-1])
            mid_points = torch.cat([mid_points, z_vals[..., -1:]], dim=-1)
            
            loss_uni = (1/3) * (intervals * weights.pow(2)).sum(dim=-1).mean()
            loss_bi = (weights[..., None, :] * weights[..., :, None] * 
                      torch.abs(
                          mid_points[...,
                          None,
                          :] - mid_points[...,
                          :,
                          None],
                      )).sum(dim=-1).mean() 
            
            distortion_loss = loss_uni + loss_bi
            losses['distortion_loss'] = distortion_loss
        
        # Total loss
        total_loss = (self.config.rgb_loss_weight * rgb_loss + 
                     self.config.compression_loss_weight * compression_loss)
        
        if self.config.distortion_loss_weight > 0:
            total_loss += self.config.distortion_loss_weight * losses['distortion_loss']
        
        losses['total_loss'] = total_loss
        
        return losses
    
    def compute_metrics(self, predictions: torch.Tensor, targets: torch.Tensor) -> dict[str, float]:
        """Compute evaluation metrics."""
        mse = F.mse_loss(predictions, targets).item()
        psnr = -10.0 * np.log10(mse) if mse > 0 else 100.0
        
        return {
            'mse': mse, 'psnr': psnr
        }
    
    def train_step(self, batch: dict[str, torch.Tensor]) -> dict[str, float]:
        """Single training step."""
        self.model.train()
        
        # Move to device
        rays = batch['rays'].to(self.device)
        colors = batch['colors'].to(self.device)
        
        # Forward pass
        outputs = self.volume_render(rays)
        
        # Compute losses
        batch_device = {'colors': colors}
        losses = self.compute_losses(batch_device, outputs)
        
        # Backward pass
        self.optimizer.zero_grad()
        losses['total_loss'].backward()
        
        # Gradient clipping
        if self.config.grad_clip_norm > 0:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.grad_clip_norm)
        
        self.optimizer.step()
        self.scheduler.step()
        
        # Compute metrics
        metrics = self.compute_metrics(outputs['rgb'], colors)
        
        # Combine losses and metrics
        result = {k: v.item() if torch.is_tensor(v) else v for k, v in losses.items()}
        result.update(metrics)
        result['lr'] = self.optimizer.param_groups[0]['lr']
        
        return result
    
    def validate(self) -> dict[str, float]:
        """Run validation."""
        if self.val_dataset is None:
            return {}
        
        self.model.eval()
        val_metrics = []
        
        with torch.no_grad():
            # Sample validation rays
            for i in range(min(10, len(self.val_dataset))):
                batch = self.val_dataset[i]
                
                # Limit number of rays for efficiency
                if len(batch['rays']) > self.config.val_num_rays:
                    indices = torch.randperm(len(batch['rays']))[:self.config.val_num_rays]
                    batch = {
                        'rays': batch['rays'][indices], 'colors': batch['colors'][indices]
                    }
                
                # Move to device
                rays = batch['rays'].to(self.device)
                colors = batch['colors'].to(self.device)
                
                # Forward pass
                outputs = self.volume_render(rays)
                
                # Compute metrics
                metrics = self.compute_metrics(outputs['rgb'], colors)
                val_metrics.append(metrics)
        
        # Average metrics
        avg_metrics = {}
        if val_metrics:
            for key in val_metrics[0].keys():
                avg_metrics[f'val_{key}'] = np.mean([m[key] for m in val_metrics])
        
        return avg_metrics
    
    def save_checkpoint(self, metrics: dict[str, float]):
        """Save model checkpoint."""
        checkpoint = {
            'epoch': self.epoch, 'global_step': self.global_step, 'model_state_dict': self.model.state_dict(
            )
        }
        
        # Save latest checkpoint
        latest_path = self.checkpoint_dir / 'latest.pth'
        torch.save(checkpoint, latest_path)
        
        # Save best checkpoint
        val_psnr = metrics.get('val_psnr', 0)
        if val_psnr > self.best_val_psnr:
            self.best_val_psnr = val_psnr
            best_path = self.checkpoint_dir / 'best.pth'
            torch.save(checkpoint, best_path)
            print(f"New best validation PSNR: {val_psnr:.2f}")
    
    def load_checkpoint(self, checkpoint_path: str):
        """Load model checkpoint."""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        self.epoch = checkpoint['epoch']
        self.global_step = checkpoint['global_step']
        self.best_val_psnr = checkpoint['best_val_psnr']
        
        print(f"Loaded checkpoint from epoch {self.epoch}")
    
    def train(self):
        """Main training loop."""
        print("Starting training...")
        
        for epoch in range(self.epoch, self.config.num_epochs):
            self.epoch = epoch
            epoch_start_time = time.time()
            
            # Training
            train_metrics = []
            for batch_idx, batch in enumerate(self.train_dataset):
                if len(batch['rays']) == 0:
                    continue
                
                step_metrics = self.train_step(batch)
                train_metrics.append(step_metrics)
                
                self.global_step += 1
                
                # Log training metrics
                if self.global_step % self.config.log_every == 0:
                    avg_metrics = {}
                    for key in train_metrics[0].keys():
                        avg_metrics[key] = np.mean([m[key] for m in train_metrics[-self.config.log_every:]])
                    
                    # Log to wandb if available
                    if self.config.use_wandb and WANDB_AVAILABLE:
                        wandb.log(avg_metrics, step=self.global_step)
                    
                    print(f"Epoch {epoch:4d}, Step {self.global_step:6d}: "
                          f"Loss {avg_metrics['total_loss']:.4f}, "
                          f"PSNR {avg_metrics['psnr']:.2f}, "
                          f"LR {avg_metrics['lr']:.2e}")
            
            # Validation
            val_metrics = {}
            if epoch % self.config.val_every == 0:
                val_metrics = self.validate()
                if val_metrics:
                    if self.config.use_wandb and WANDB_AVAILABLE:
                        wandb.log(val_metrics, step=self.global_step)
                    print(f"Validation PSNR: {val_metrics.get('val_psnr', 0):.2f}")
            
            # Checkpointing
            if epoch % self.config.save_every == 0:
                all_metrics = {}
                if train_metrics:
                    for key in train_metrics[0].keys():
                        all_metrics[key] = np.mean([m[key] for m in train_metrics])
                all_metrics.update(val_metrics)
                
                self.save_checkpoint(all_metrics)
            
            epoch_time = time.time() - epoch_start_time
            print(f"Epoch {epoch} completed in {epoch_time:.2f}s")
        
        print("Training completed!")
    
    def compress_and_evaluate(self):
        """Compress the model and evaluate compression metrics."""
        print("Compressing model...")
        
        compression_info = self.model.compress_model()
        stats = self.model.get_compression_stats()
        
        print(f"Compression completed:")
        print(f"  Original size: {stats['original_size_mb']:.2f} MB")
        print(f"  Compressed size: {stats['compressed_size_mb']:.2f} MB")
        print(f"  Compression ratio: {stats['compression_ratio']:.1f}x")
        print(f"  Size reduction: {stats['size_reduction_percent']:.1f}%")
        
        return {
            'compression_info': compression_info, 'compression_stats': stats
        }

def create_cnc_nerf_trainer(
    model_config: CNCNeRFConfig,
    dataset_config: CNCNeRFDatasetConfig,
    trainer_config: CNCNeRFTrainerConfig = None,
) -> CNCNeRFTrainer:
    """Create a CNC-NeRF trainer with datasets."""
    # Create model
    model = CNCNeRF(model_config)
    
    # Create datasets
    train_dataset = CNCNeRFDataset(dataset_config, split='train')
    val_dataset = CNCNeRFDataset(dataset_config, split='val')
    
    # Create trainer
    trainer = CNCNeRFTrainer(model, train_dataset, val_dataset, trainer_config)
    
    return trainer 