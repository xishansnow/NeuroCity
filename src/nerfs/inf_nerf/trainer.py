from typing import Any, Optional
"""
InfNeRF Trainer Module

This module implements the training infrastructure for InfNeRF with support for:
- Multi-scale pyramid supervision
- Distributed training across octree levels  
- Level consistency regularization
- Memory-efficient batch processing
- Dynamic octree construction during training
"""

import os
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
from dataclasses import dataclass
from tqdm import tqdm
import numpy as np
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    
from pathlib import Path
import json
import math

from .core import InfNeRF, InfNeRFConfig, InfNeRFRenderer
from .dataset import InfNeRFDataset, InfNeRFDatasetConfig

@dataclass
class InfNeRFTrainerConfig:
    """Configuration for InfNeRF trainer."""
    
    # Training parameters
    num_epochs: int = 100
    lr_init: float = 1e-2
    lr_final: float = 1e-4
    lr_decay_start: int = 50000
    lr_decay_steps: int = 250000
    weight_decay: float = 1e-6
    
    # Loss weights
    lambda_rgb: float = 1.0
    lambda_depth: float = 0.1
    lambda_distortion: float = 0.01
    lambda_transparency: float = 1e-3
    lambda_regularization: float = 1e-4
    
    # Batch processing
    rays_batch_size: int = 4096
    max_batch_rays: int = 16384
    accumulate_grad_batches: int = 1
    gradient_clip_val: float = 1.0
    
    # Rendering parameters
    num_samples_coarse: int = 64
    num_samples_fine: int = 128
    use_white_background: bool = False
    
    # Octree training
    octree_update_freq: int = 1000
    octree_prune_freq: int = 5000
    adaptive_sampling: bool = True
    
    # Distributed training
    distributed: bool = False
    local_rank: int = 0
    world_size: int = 1
    shared_upper_levels: int = 2
    
    # Logging and checkpointing
    log_dir: str = "logs"
    ckpt_dir: str = "checkpoints" 
    save_freq: int = 5000
    eval_freq: int = 1000
    log_freq: int = 100
    
    # Experiment tracking
    use_wandb: bool = False
    project_name: str = "inf_nerf"
    experiment_name: str = "default"
    
    # Memory management
    mixed_precision: bool = True
    memory_threshold_gb: float = 16.0
    chunk_size: int = 8192

class InfNeRFTrainer:
    """
    Trainer for InfNeRF with distributed training and pyramid supervision.
    """
    
    def __init__(
        self,
        model: InfNeRF,
        train_dataset: InfNeRFDataset,
        config: InfNeRFTrainerConfig,
        val_dataset: Optional[InfNeRFDataset] = None,
        device: Optional[torch.device] = None,
    ) -> None:
        """
        Initialize InfNeRF trainer.
        
        Args:
            model: InfNeRF model to train
            train_dataset: Training dataset
            config: Trainer configuration
            val_dataset: Validation dataset (optional)
            device: Training device
        """
        self.config = config
        self.device = device if device else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Model and dataset
        self.model = model.to(self.device)
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        
        # Initialize renderer
        self.renderer = InfNeRFRenderer(model.config)
        
        # Setup distributed training if enabled
        if config.distributed:
            self._setup_distributed()
        
        # Setup optimizer and scheduler
        self._setup_optimization()
        
        # Setup mixed precision
        if config.mixed_precision:
            self.scaler = torch.cuda.amp.GradScaler()
        
        # Training state
        self.global_step = 0
        self.epoch = 0
        self.best_psnr = 0.0
        
        # Logging
        self.writer = None
        if config.use_wandb and WANDB_AVAILABLE and (not config.distributed or config.local_rank == 0):
            self._setup_wandb()
        
        # Create output directories
        self._create_directories()
        
        print(f"Initialized InfNeRF trainer on device: {self.device}")
        print(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):, }")
        
    def _setup_distributed(self):
        """Setup distributed training."""
        if not dist.is_initialized():
            dist.init_process_group(backend='nccl')
        
        self.config.local_rank = dist.get_rank()
        self.config.world_size = dist.get_world_size()
        
        torch.cuda.set_device(self.config.local_rank)
        self.device = torch.device(f'cuda:{self.config.local_rank}')
        
        # Wrap model with DDP
        self.model = DDP(self.model, device_ids=[self.config.local_rank])
        
        print(f"Distributed training: rank {self.config.local_rank}/{self.config.world_size}")
    
    def _setup_optimization(self):
        """Setup optimizer and learning rate scheduler."""
        # Get model parameters (handle DDP wrapper)
        model_module = self.model.module if hasattr(self.model, 'module') else self.model
        
        # Separate parameters by type for different learning rates
        octree_params = []
        other_params = []
        
        for node in model_module.octree_nodes:
            if node.nerf is not None:
                # Hash encoding parameters get higher learning rate
                hash_params = []
                mlp_params = []
                
                for name, param in node.nerf.named_parameters():
                    if 'hash_encoder' in name or 'position_encoder' in name:
                        hash_params.append(param)
                    else:
                        mlp_params.append(param)
                
                octree_params.extend([
                    {
                        'params': hash_params,
                        'lr': self.config.lr_init * 10,
                        'name': f'hash_level_{node.level}',
                    }
                ])
        
        # Create optimizer
        param_groups = octree_params if octree_params else [{'params': model_module.parameters()}]
        self.optimizer = optim.Adam(param_groups, weight_decay=self.config.weight_decay)
        
        # Learning rate scheduler with exponential decay
        def lr_lambda(step):
            if step < self.config.lr_decay_start:
                return 1.0
            else:
                progress = (step - self.config.lr_decay_start) / self.config.lr_decay_steps
                return max(0.01, (self.config.lr_final / self.config.lr_init) ** progress)
        
        self.scheduler = optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda)
    
    def _setup_wandb(self):
        """Setup Weights & Biases logging."""
        if not WANDB_AVAILABLE:
            print("Warning: wandb not available, skipping wandb setup")
            return
            
        wandb.init(
            project=self.config.project_name, name=self.config.experiment_name, config=self.config.__dict__
        )
    
    def _create_directories(self):
        """Create output directories."""
        if not self.config.distributed or self.config.local_rank == 0:
            os.makedirs(self.config.log_dir, exist_ok=True)
            os.makedirs(self.config.ckpt_dir, exist_ok=True)
            
            # Setup tensorboard
            self.writer = SummaryWriter(self.config.log_dir)
    
    def compute_loss(
        self,
        batch: dict[str,
        torch.Tensor],
        outputs: dict[str,
        torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        """
        Compute InfNeRF training losses.
        
        Args:
            batch: Input batch
            outputs: Model outputs
            
        Returns:
            Dictionary of losses
        """
        losses = {}
        
        # RGB reconstruction loss
        target_rgb = batch['target_rgb']
        pred_rgb = outputs['rgb']
        
        rgb_loss = F.mse_loss(pred_rgb, target_rgb)
        losses['rgb_loss'] = rgb_loss
        
        # Depth loss (if available)
        if 'target_depth' in batch and 'depth' in outputs:
            depth_loss = F.l1_loss(outputs['depth'], batch['target_depth'])
            losses['depth_loss'] = depth_loss
        else:
            losses['depth_loss'] = torch.tensor(0.0, device=self.device)
        
        # Distortion loss (from Mip-NeRF 360)
        if 'weights' in outputs and 'z_vals' in outputs:
            weights = outputs['weights']  # [N, num_samples]
            z_vals = outputs['z_vals']    # [N, num_samples]
            
            # Compute distortion loss
            intervals = z_vals[..., 1:] - z_vals[..., :-1]
            mid_points = 0.5 * (z_vals[..., 1:] + z_vals[..., :-1])
            
            # Weighted sum of intervals
            loss_uni = (weights[..., :-1] * intervals).sum(-1).mean()
            
            # Distance between samples weighted by their weights
            w_normalized = weights / (weights.sum(-1, keepdim=True) + 1e-8)
            loss_bi = 0
            for i in range(weights.shape[-1]):
                for j in range(i+1, weights.shape[-1]):
                    loss_bi += w_normalized[..., i] * w_normalized[..., j] * torch.abs(
                        mid_points[...,
                        i] - mid_points[...,
                        j],).sum(dim=-1).mean()
            
            losses['distortion_loss'] = loss_uni + loss_bi.mean()
        else:
            losses['distortion_loss'] = torch.tensor(0.0, device=self.device)
        
        # Transparency loss for empty space
        if 'density' in outputs and 'positions' in batch:
            # Sample random points in empty space and encourage low density
            empty_space_loss = torch.tensor(0.0, device=self.device)
            losses['transparency_loss'] = empty_space_loss
        else:
            losses['transparency_loss'] = torch.tensor(0.0, device=self.device)
        
        # Level consistency regularization
        if hasattr(self.model, 'module'):
            model_module = self.model.module
        else:
            model_module = self.model
            
        reg_loss = torch.tensor(0.0, device=self.device)
        if len(model_module.octree_nodes) > 1:
            # Encourage consistency between adjacent octree levels
            for node in model_module.octree_nodes:
                if node.parent is not None and not node.is_pruned and not node.parent.is_pruned:
                    # Sample some positions and compare densities
                    positions = batch['positions'][:100]  # Sample subset
                    
                    with torch.no_grad():
                        parent_outputs = node.parent.nerf(positions, batch['directions'][:100])
                        child_outputs = node.nerf(positions, batch['directions'][:100])
                        
                    density_diff = F.mse_loss(child_outputs['density'], parent_outputs['density'])
                    reg_loss += density_diff
        
        losses['regularization_loss'] = reg_loss
        
        # Total loss
        total_loss = (self.config.lambda_rgb * losses['rgb_loss'] +
                     self.config.lambda_depth * losses['depth_loss'] + 
                     self.config.lambda_distortion * losses['distortion_loss'] +
                     self.config.lambda_transparency * losses['transparency_loss'] +
                     self.config.lambda_regularization * losses['regularization_loss'])
        
        losses['total_loss'] = total_loss
        
        # Compute metrics
        with torch.no_grad():
            mse = F.mse_loss(pred_rgb, target_rgb)
            psnr = -10.0 * torch.log10(mse + 1e-8)
            losses['psnr'] = psnr
        
        return losses
    
    def train_step(self, batch: dict[str, torch.Tensor]) -> dict[str, float]:
        """
        Execute single training step.
        
        Args:
            batch: Training batch
            
        Returns:
            Dictionary of loss values
        """
        self.model.train()
        
        # Move batch to device
        for key in batch:
            if torch.is_tensor(batch[key]):
                batch[key] = batch[key].to(self.device)
        
        # Forward pass with mixed precision
        if self.config.mixed_precision:
            with torch.cuda.amp.autocast():
                outputs = self.model(
                    rays_o=batch['rays_o'], rays_d=batch['rays_d'], near=batch.get(
                        'near',
                        0.1,
                    )
                )
                
                losses = self.compute_loss(batch, outputs)
                loss = losses['total_loss'] / self.config.accumulate_grad_batches
        else:
            outputs = self.model(
                rays_o=batch['rays_o'], rays_d=batch['rays_d'], near=batch.get(
                    'near',
                    0.1,
                )
            )
            
            losses = self.compute_loss(batch, outputs)
            loss = losses['total_loss'] / self.config.accumulate_grad_batches
        
        # Backward pass
        if self.config.mixed_precision:
            self.scaler.scale(loss).backward()
        else:
            loss.backward()
        
        # Update parameters
        if (self.global_step + 1) % self.config.accumulate_grad_batches == 0:
            if self.config.mixed_precision:
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters,
                )
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters,
                )
                self.optimizer.step()
            
            self.optimizer.zero_grad()
            self.scheduler.step()
        
        # Convert tensors to float for logging
        return {k: v.item() if torch.is_tensor(v) else v for k, v in losses.items()}
    
    def validate(self) -> dict[str, float]:
        """Run validation."""
        if self.val_dataset is None:
            return {}
        
        self.model.eval()
        val_losses = []
        
        val_loader = DataLoader(
            self.val_dataset, batch_size=self.config.rays_batch_size, shuffle=False, num_workers=2
        )
        
        with torch.no_grad():
            for batch in val_loader:
                # Move to device
                for key in batch:
                    if torch.is_tensor(batch[key]):
                        batch[key] = batch[key].to(self.device)
                
                # Forward pass
                outputs = self.model(
                    rays_o=batch['rays_o'], rays_d=batch['rays_d'], near=batch.get(
                        'near',
                        0.1,
                    )
                )
                
                losses = self.compute_loss(batch, outputs)
                val_losses.append({k: v.item() if torch.is_tensor(v) else v 
                                 for k, v in losses.items()})
        
        # Average validation losses
        avg_losses = {}
        for key in val_losses[0].keys():
            avg_losses[f'val_{key}'] = np.mean([loss[key] for loss in val_losses])
        
        return avg_losses
    
    def update_octree(self):
        """Update octree structure based on training progress."""
        if hasattr(self.model, 'module'):
            model_module = self.model.module
        else:
            model_module = self.model
        
        # Prune nodes with low density
        pruned_count = 0
        for node in model_module.octree_nodes:
            if not node.is_pruned and node.level > model_module.config.min_depth:
                # Sample density at node center
                center = torch.tensor(node.center, device=self.device).unsqueeze(0)
                dummy_dir = torch.zeros_like(center)
                
                with torch.no_grad():
                    outputs = node.nerf(center, dummy_dir)
                    density = outputs['density'].item()
                
                # Prune if density is too low
                if density < model_module.config.pruning_threshold:
                    node.is_pruned = True
                    pruned_count += 1
        
        if pruned_count > 0:
            print(f"Pruned {pruned_count} octree nodes")
    
    def save_checkpoint(self, is_best: bool = False):
        """Save training checkpoint."""
        if self.config.distributed and self.config.local_rank != 0:
            return
        
        # Get model state dict (handle DDP wrapper)
        model_state = self.model.module.state_dict(
        )
        
        checkpoint = {
            'epoch': self.epoch, 'global_step': self.global_step, 'model_state_dict': model_state, 'optimizer_state_dict': self.optimizer.state_dict(
            )
        }
        
        # Save regular checkpoint
        ckpt_path = os.path.join(self.config.ckpt_dir, f'step_{self.global_step:06d}.pth')
        torch.save(checkpoint, ckpt_path)
        
        # Save best checkpoint
        if is_best:
            best_path = os.path.join(self.config.ckpt_dir, 'best.pth')
            torch.save(checkpoint, best_path)
        
        print(f"Checkpoint saved: {ckpt_path}")
    
    def load_checkpoint(self, ckpt_path: str):
        """Load training checkpoint."""
        checkpoint = torch.load(ckpt_path, map_location=self.device)
        
        # Load model state (handle DDP wrapper)
        if hasattr(self.model, 'module'):
            self.model.module.load_state_dict(checkpoint['model_state_dict'])
        else:
            self.model.load_state_dict(checkpoint['model_state_dict'])
        
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        self.epoch = checkpoint['epoch']
        self.global_step = checkpoint['global_step']
        self.best_psnr = checkpoint['best_psnr']
        
        print(f"Checkpoint loaded: {ckpt_path}")
    
    def train(self):
        """Main training loop."""
        print(f"Starting InfNeRF training for {self.config.num_epochs} epochs")
        
        # Build octree from sparse points
        sparse_points = self.train_dataset.get_sparse_points()
        if len(sparse_points) > 0:
            if hasattr(self.model, 'module'):
                self.model.module.build_octree(sparse_points)
            else:
                self.model.build_octree(sparse_points)
        
        # Create data loader
        train_loader = DataLoader(
            self.train_dataset, batch_size=1, # Dataset returns ray batches
            shuffle=True, num_workers=4, pin_memory=True
        )
        
        start_time = time.time()
        
        for epoch in range(self.epoch, self.config.num_epochs):
            self.epoch = epoch
            epoch_losses = []
            
            # Training loop
            pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{self.config.num_epochs}')
            for batch in pbar:
                losses = self.train_step(batch)
                epoch_losses.append(losses)
                self.global_step += 1
                
                # Update progress bar
                pbar.set_postfix({
                    'loss': f"{losses['total_loss']:.4f}",
                })
                
                # Logging
                if self.global_step % self.config.log_freq == 0:
                    self._log_metrics(losses, prefix='train')
                
                # Validation
                if self.global_step % self.config.eval_freq == 0:
                    val_losses = self.validate()
                    if val_losses:
                        self._log_metrics(val_losses, prefix='val')
                        
                        # Check if best model
                        if 'val_psnr' in val_losses and val_losses['val_psnr'] > self.best_psnr:
                            self.best_psnr = val_losses['val_psnr']
                            self.save_checkpoint(is_best=True)
                
                # Update octree structure
                if self.global_step % self.config.octree_update_freq == 0:
                    self.update_octree()
                
                # Save checkpoint
                if self.global_step % self.config.save_freq == 0:
                    self.save_checkpoint()
            
            # Epoch summary
            avg_losses = {}
            for key in epoch_losses[0].keys():
                avg_losses[key] = np.mean([loss[key] for loss in epoch_losses])
            
            epoch_time = time.time() - start_time
            print(f"Epoch {epoch+1} | Time: {epoch_time:.2f}s | "
                  f"Loss: {avg_losses['total_loss']:.4f} | PSNR: {avg_losses['psnr']:.2f}")
        
        # Final checkpoint
        self.save_checkpoint()
        print("Training completed!")
    
    def _log_metrics(self, metrics: dict[str, float], prefix: str = ''):
        """Log metrics to tensorboard and wandb."""
        if not self.config.distributed or self.config.local_rank == 0:
            # Tensorboard
            if self.writer:
                for key, value in metrics.items():
                    tag = f'{prefix}/{key}' if prefix else key
                    self.writer.add_scalar(tag, value, self.global_step)
            
            # Wandb
            if self.config.use_wandb and WANDB_AVAILABLE:
                log_dict = {f'{prefix}_{k}' if prefix else k: v for k, v in metrics.items()}
                log_dict['step'] = self.global_step
                wandb.log(log_dict)

def create_inf_nerf_trainer(
    model_config: InfNeRFConfig, dataset_config: InfNeRFDatasetConfig, trainer_config: InfNeRFTrainerConfig, device: Optional[torch.device] = None
) -> InfNeRFTrainer:
    """
    Factory function to create InfNeRF trainer with all components.
    
    Args:
        model_config: Model configuration
        dataset_config: Dataset configuration  
        trainer_config: Trainer configuration
        device: Training device
        
    Returns:
        Configured InfNeRF trainer
    """
    # Create model
    model = InfNeRF(model_config)
    
    # Create datasets
    train_dataset = InfNeRFDataset(dataset_config, split='train')
    val_dataset = InfNeRFDataset(dataset_config, split='val') if dataset_config else None
    
    # Create trainer
    trainer = InfNeRFTrainer(
        model=model, train_dataset=train_dataset, config=trainer_config, val_dataset=val_dataset, device=device
    )
    
    return trainer 