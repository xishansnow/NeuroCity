from typing import Any, Optional
"""
Nerfacto Trainer Module

This module implements the training pipeline for Nerfacto, including:
- Mixed precision training
- Gradient accumulation  
- Progressive training strategies
- Advanced loss functions
- Evaluation and checkpointing
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import os
import json
import time
from dataclasses import dataclass, field
import wandb
from tqdm import tqdm

from .core import NerfactoModel, NerfactoConfig, NerfactoLoss
from .dataset import NerfactoDataset, NerfactoDatasetConfig

@dataclass
class NerfactoTrainerConfig:
    """Configuration for Nerfacto trainer."""
    
    # Training settings
    max_epochs: int = 30000
    batch_size: int = 1
    learning_rate: float = 5e-4
    weight_decay: float = 1e-6
    
    # Learning rate scheduling
    lr_scheduler: str = "exponential"  # exponential, cosine, step
    lr_decay_rate: float = 0.1
    lr_decay_steps: int = 250000
    lr_final: float = 5e-6
    
    # Mixed precision training
    use_mixed_precision: bool = True
    gradient_clip_val: float = 1.0
    
    # Gradient accumulation
    gradient_accumulation_steps: int = 1
    
    # Progressive training
    use_progressive_training: bool = True
    progressive_levels: list[int] = field(default_factory=lambda: [64, 128, 256, 512])
    progressive_epochs: list[int] = field(default_factory=lambda: [5000, 10000, 20000, 30000])
    
    # Loss weights scheduling
    color_loss_weight: float = 1.0
    depth_loss_weight: float = 0.1
    normal_loss_weight: float = 0.05
    
    # Distillation loss (if using proposal networks)
    distillation_loss_weight: float = 0.01
    
    # Regularization
    proposal_weights_decay: float = 1e-4
    
    # Evaluation settings
    eval_every_n_epochs: int = 1000
    eval_num_rays: int = 4096
    
    # Checkpointing
    save_every_n_epochs: int = 5000
    keep_n_checkpoints: int = 5
    
    # Logging
    log_every_n_steps: int = 100
    use_wandb: bool = False
    wandb_project: str = "nerfacto"
    
    # Output directories
    output_dir: str = "outputs"
    experiment_name: str = "nerfacto_experiment"

class NerfactoTrainer:
    """Main trainer class for Nerfacto."""
    
    def __init__(
        self,
        config: NerfactoTrainerConfig,
        model_config: NerfactoConfig,
        dataset_config: NerfactoDatasetConfig,
        device: str = "cuda",
    ) -> None:
        self.config = config
        self.model_config = model_config
        self.dataset_config = dataset_config
        self.device = device
        
        # Initialize model
        self.model = NerfactoModel(model_config).to(device)
        
        # Initialize loss function
        self.loss_fn = NerfactoLoss(model_config)
        
        # Initialize datasets
        self.train_dataset = NerfactoDataset(dataset_config, split="train")
        self.val_dataset = NerfactoDataset(dataset_config, split="val")
        
        # Initialize data loaders
        self.train_loader = DataLoader(
            self.train_dataset, batch_size=config.batch_size, shuffle=True, num_workers=4, pin_memory=True
        )
        
        if self.val_dataset:
            self.val_loader = DataLoader(
                self.val_dataset, batch_size=1, shuffle=False, num_workers=2, pin_memory=True
            )
        
        # Initialize optimizer
        self.optimizer = self._create_optimizer()
        
        # Initialize scheduler
        self.scheduler = self._create_scheduler()
        
        # Mixed precision training
        self.scaler = GradScaler() if config.use_mixed_precision else None
        
        # Training state
        self.epoch = 0
        self.step = 0
        self.best_psnr = 0.0
        
        # Progressive training state
        self.current_resolution_level = 0
        
        # Logging
        self.setup_logging()
        
        # Create output directories
        self.output_dir = os.path.join(config.output_dir, config.experiment_name)
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(os.path.join(self.output_dir, "checkpoints"), exist_ok=True)
        
        # Save configs
        self.save_configs()
    
    def _create_optimizer(self) -> optim.Optimizer:
        """Create optimizer with different learning rates for different components."""
        param_groups = []
        
        # Encoding parameters (higher learning rate)
        encoding_params = []
        if hasattr(self.model.field, 'encoding'):
            encoding_params.extend(list(self.model.field.encoding.parameters()))
        
        if encoding_params:
            param_groups.append({
                'params': encoding_params, 'lr': self.config.learning_rate, 'name': 'encoding'
            })
        
        # Network parameters (standard learning rate)
        network_params = []
        if hasattr(self.model.field, 'mlp_head'):
            network_params.extend(list(self.model.field.mlp_head.parameters()))
        if hasattr(self.model.field, 'color_net'):
            network_params.extend(list(self.model.field.color_net.parameters()))
        
        if network_params:
            param_groups.append({
                'params': network_params, 'lr': self.config.learning_rate * 0.1, 'name': 'network'
            })
        
        # Proposal network parameters (if any)
        proposal_params = []
        if hasattr(self.model, 'proposal_networks'):
            for proposal_net in self.model.proposal_networks:
                proposal_params.extend(list(proposal_net.parameters()))
        
        if proposal_params:
            param_groups.append({
                'params': proposal_params, 'lr': self.config.learning_rate * 0.01, 'weight_decay': self.config.proposal_weights_decay, 'name': 'proposal'
            })
        
        # Fallback: all parameters
        if not param_groups:
            param_groups = [{'params': self.model.parameters(), 'lr': self.config.learning_rate}]
        
        return optim.AdamW(param_groups, weight_decay=self.config.weight_decay)
    
    def _create_scheduler(self) -> optim.lr_scheduler.LRScheduler:
        """Create learning rate scheduler."""
        if self.config.lr_scheduler == "exponential":
            # Exponential decay
            gamma = (self.config.lr_final / self.config.learning_rate) ** (1.0 / self.config.max_epochs)
            return optim.lr_scheduler.ExponentialLR(self.optimizer, gamma=gamma)
        
        elif self.config.lr_scheduler == "cosine":
            return optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer, T_max=self.config.max_epochs, eta_min=self.config.lr_final
            )
        
        elif self.config.lr_scheduler == "step":
            return optim.lr_scheduler.StepLR(
                self.optimizer, step_size=self.config.lr_decay_steps, gamma=self.config.lr_decay_rate
            )
        
        else:
            return optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda=lambda epoch: 1.0)
    
    def setup_logging(self) -> None:
        """Setup logging with TensorBoard and optionally Weights & Biases."""
        # TensorBoard
        self.writer = SummaryWriter(os.path.join(self.output_dir, "logs"))
        
        # Weights & Biases
        if self.config.use_wandb:
            wandb.init(
                project=self.config.wandb_project, name=self.config.experiment_name, config={
                    "trainer_config": self.config.__dict__, "model_config": self.model_config.__dict__, "dataset_config": self.dataset_config.__dict__
                }
            )
    
    def save_configs(self) -> None:
        """Save configuration files."""
        configs = {
            "trainer_config": self.config.__dict__, "model_config": self.model_config.__dict__, "dataset_config": self.dataset_config.__dict__
        }
        
        with open(os.path.join(self.output_dir, "config.json"), 'w') as f:
            json.dump(configs, f, indent=2, default=str)
    
    def update_progressive_training(self) -> None:
        """Update progressive training parameters."""
        if not self.config.use_progressive_training:
            return
        
        # Check if we need to increase resolution
        for i, epoch_threshold in enumerate(self.config.progressive_epochs):
            if self.epoch < epoch_threshold:
                target_level = i
                break
        else:
            target_level = len(self.config.progressive_levels) - 1
        
        if target_level != self.current_resolution_level:
            self.current_resolution_level = target_level
            target_resolution = self.config.progressive_levels[target_level]
            
            print(f"Progressive training: Switching to resolution {target_resolution}")
            
            # Update model resolution if applicable
            if hasattr(self.model.field, 'set_resolution'):
                self.model.field.set_resolution(target_resolution)
    
    def compute_loss(
        self,
        outputs: dict[str, torch.Tensor],
        targets: dict[str, torch.Tensor]
    ) -> dict[str, torch.Tensor]:
        """Compute training loss."""
        losses = self.loss_fn(outputs, targets)
        
        # Weighted total loss
        total_loss = (
            self.config.color_loss_weight * losses.get('color_loss', 0) +
            self.config.depth_loss_weight * losses.get('depth_loss', 0) +
            self.config.normal_loss_weight * losses.get('normal_loss', 0) +
            self.config.distillation_loss_weight * losses.get('distillation_loss', 0)
        )
        
        losses['total_loss'] = total_loss
        
        return losses
    
    def train_step(self, batch: dict[str, torch.Tensor]) -> dict[str, float]:
        """Single training step."""
        # Move batch to device
        for key in batch:
            if isinstance(batch[key], torch.Tensor):
                batch[key] = batch[key].to(self.device)
        
        # Sample rays
        ray_origins = batch['rays_o']
        ray_directions = batch['rays_d']
        target_colors = batch['colors']
        
        # Forward pass with mixed precision
        if self.scaler is not None:
            with autocast():
                outputs = self.model(ray_origins, ray_directions)
                losses = self.compute_loss(outputs, {'colors': target_colors})
                loss = losses['total_loss']
        else:
            outputs = self.model(ray_origins, ray_directions)
            losses = self.compute_loss(outputs, {'colors': target_colors})
            loss = losses['total_loss']
        
        # Backward pass
        if self.scaler is not None:
            self.scaler.scale(loss).backward()
            
            if (self.step + 1) % self.config.gradient_accumulation_steps == 0:
                # Gradient clipping
                if self.config.gradient_clip_val > 0:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.gradient_clip_val)
                
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad()
        else:
            loss.backward()
            
            if (self.step + 1) % self.config.gradient_accumulation_steps == 0:
                # Gradient clipping
                if self.config.gradient_clip_val > 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.gradient_clip_val)
                
                self.optimizer.step()
                self.optimizer.zero_grad()
        
        # Convert losses to float
        loss_dict = {}
        for key, value in losses.items():
            if isinstance(value, torch.Tensor):
                loss_dict[key] = value.item()
            else:
                loss_dict[key] = value
        
        return loss_dict
    
    def validate(self) -> dict[str, float]:
        """Validation step."""
        if self.val_dataset is None:
            return {}
        
        self.model.eval()
        val_losses = []
        val_psnrs = []
        
        with torch.no_grad():
            for batch in self.val_loader:
                # Move to device
                for key in batch:
                    if isinstance(batch[key], torch.Tensor):
                        batch[key] = batch[key].to(self.device)
                
                # Forward pass
                outputs = self.model(batch['rays_o'], batch['rays_d'])
                
                # Compute metrics
                losses = self.compute_loss(outputs, {'colors': batch['colors']})
                val_losses.append(losses['total_loss'])
                
                # Compute PSNR
                mse = torch.mean((outputs['colors'] - batch['colors']) ** 2)
                psnr = -10 * torch.log10(mse)
                val_psnrs.append(psnr.item())
        
        self.model.train()
        
        return {
            'val_loss': np.mean(val_losses), 'val_psnr': np.mean(val_psnrs)
        }
    
    def save_checkpoint(self, is_best: bool = False) -> None:
        """Save model checkpoint."""
        checkpoint = {
            'epoch': self.epoch, 'step': self.step, 'model_state_dict': self.model.state_dict()
        }
        
        if self.scaler is not None:
            checkpoint['scaler_state_dict'] = self.scaler.state_dict()
        
        # Save regular checkpoint
        checkpoint_path = os.path.join(
            self.output_dir,
            "checkpoints",
            f"checkpoint_epoch_{self.epoch}.pth"
        )
        torch.save(checkpoint, checkpoint_path)
        
        # Save best checkpoint
        if is_best:
            best_path = os.path.join(self.output_dir, "checkpoints", "best_model.pth")
            torch.save(checkpoint, best_path)
        
        # Keep only recent checkpoints
        self.cleanup_checkpoints()
    
    def cleanup_checkpoints(self) -> None:
        """Remove old checkpoints to save disk space."""
        checkpoint_dir = os.path.join(self.output_dir, "checkpoints")
        checkpoints = [f for f in os.listdir(checkpoint_dir) if f.startswith("checkpoint_epoch_")]
        
        if len(checkpoints) > self.config.keep_n_checkpoints:
            # Sort by epoch number
            checkpoints.sort(key=lambda x: int(x.split('_')[-1].split('.')[0]))
            
            # Remove oldest checkpoints
            for checkpoint in checkpoints[:-self.config.keep_n_checkpoints]:
                os.remove(os.path.join(checkpoint_dir, checkpoint))
    
    def log_metrics(self, metrics: dict[str, float], prefix: str = "train") -> None:
        """Log metrics to TensorBoard and wandb."""
        # TensorBoard
        for key, value in metrics.items():
            self.writer.add_scalar(f"{prefix}/{key}", value, self.step)
        
        # Weights & Biases
        if self.config.use_wandb:
            wandb.log({f"{prefix}/{key}": value for key, value in metrics.items()}, step=self.step)
    
    def train(self) -> None:
        """Main training loop."""
        print(f"Starting training for {self.config.max_epochs} epochs")
        print(f"Training dataset: {len(self.train_dataset)} images")
        if self.val_dataset:
            print(f"Validation dataset: {len(self.val_dataset)} images")
        
        self.model.train()
        start_time = time.time()
        
        for epoch in range(self.config.max_epochs):
            self.epoch = epoch
            
            # Update progressive training
            self.update_progressive_training()
            
            # Training epoch
            epoch_losses = []
            progress_bar = tqdm(self.train_loader, desc=f"Epoch {epoch}")
            
            for batch_idx, batch in enumerate(progress_bar):
                # Training step
                losses = self.train_step(batch)
                epoch_losses.append(losses)
                
                # Logging
                if self.step % self.config.log_every_n_steps == 0:
                    self.log_metrics(losses, "train")
                    
                    # Update progress bar
                    progress_bar.set_postfix({
                        'loss': f"{losses['total_loss']:.6f}"
                        })
                
                self.step += 1
            
            # Step scheduler
            self.scheduler.step()
            
            # Validation
            if epoch % self.config.eval_every_n_epochs == 0:
                val_metrics = self.validate()
                if val_metrics:
                    self.log_metrics(val_metrics, "val")
                    
                    # Check for best model
                    current_psnr = val_metrics.get('val_psnr', 0)
                    is_best = current_psnr > self.best_psnr
                    if is_best:
                        self.best_psnr = current_psnr
                    
                    print(f"Validation - Loss: {val_metrics.get('val_loss', 0):.6f}, "
                          f"PSNR: {current_psnr:.2f} (Best: {self.best_psnr:.2f})")
                    
                    # Save checkpoint
                    if epoch % self.config.save_every_n_epochs == 0 or is_best:
                        self.save_checkpoint(is_best)
            
            # Log epoch statistics
            avg_loss = np.mean([l['total_loss'] for l in epoch_losses])
            elapsed_time = time.time() - start_time
            
            print(f"Epoch {epoch} completed - Avg Loss: {avg_loss:.6f}, "
                  f"Time: {elapsed_time:.2f}s, LR: {self.optimizer.param_groups[0]['lr']:.2e}")
        
        # Final checkpoint
        self.save_checkpoint()
        
        print("Training completed!")
        if self.config.use_wandb:
            wandb.finish()
    
    def load_checkpoint(self, checkpoint_path: str) -> None:
        """Load model from checkpoint."""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        self.epoch = checkpoint['epoch']
        self.step = checkpoint['step']
        self.best_psnr = checkpoint['best_psnr']
        
        if self.scaler is not None and 'scaler_state_dict' in checkpoint:
            self.scaler.load_state_dict(checkpoint['scaler_state_dict'])
        
        print(f"Loaded checkpoint from epoch {self.epoch}, step {self.step}")

def create_nerfacto_trainer(
    data_dir: str, output_dir: str = "outputs", experiment_name: str = "nerfacto_experiment", **kwargs: Any
) -> NerfactoTrainer:
    """Factory function to create Nerfacto trainer."""
    
    # Create configs
    trainer_config = NerfactoTrainerConfig(
        output_dir=output_dir, experiment_name=experiment_name, **kwargs
    )
    
    model_config = NerfactoConfig()
    
    dataset_config = NerfactoDatasetConfig(
        data_dir=data_dir
    )
    
    return NerfactoTrainer(trainer_config, model_config, dataset_config) 