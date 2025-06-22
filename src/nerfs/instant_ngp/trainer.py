"""
Trainer module for Instant NGP.

This module contains the trainer for the Instant NGP model.

The trainer is used to:
- Train the Instant NGP model
- Render images
- Save and load checkpoints
- Log training progress
- Compute loss
- Backward pass
"""

import os
import time
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from typing import Dict, Optional
from tqdm import tqdm
import numpy as np

from .core import InstantNGP, InstantNGPConfig, InstantNGPLoss, InstantNGPRenderer


class InstantNGPTrainer:
    """Trainer for Instant NGP."""
    
    def __init__(self, config: InstantNGPConfig, device=None):
        self.config = config
        self.device = device if device else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Create model
        self.model = InstantNGP(config).to(self.device)
        
        # Create renderer and loss
        self.renderer = InstantNGPRenderer(config)
        self.criterion = InstantNGPLoss(config)
        
        # Optimizer with different learning rates for hash encoding
        hash_param_ids = {id(p) for p in self.model.position_encoder.parameters()}
        mlp_params = [p for p in self.model.parameters() if id(p) not in hash_param_ids]
        hash_params = list(self.model.position_encoder.parameters())
        
        self.optimizer = optim.Adam([
            {'params': hash_params, 'lr': config.learning_rate * 10},  # Higher LR for hash grids
            {'params': mlp_params, 'lr': config.learning_rate}
        ], weight_decay=config.weight_decay)
        
        # Scheduler
        self.scheduler = optim.lr_scheduler.StepLR(
            self.optimizer, 
            step_size=config.decay_step,
            gamma=config.learning_rate_decay
        )
        
        self.global_step = 0
        self.writer = None
    
    def setup_logging(self, log_dir: str):
        """Setup tensorboard logging."""
        self.writer = SummaryWriter(log_dir)
    
    def train_step(self, batch):
        """Single training step."""
        self.model.train()
        
        rays_o = batch['rays_o'].to(self.device)
        rays_d = batch['rays_d'].to(self.device) 
        targets = batch['targets'].to(self.device)
        
        # Near/far bounds
        near = torch.full_like(rays_o[..., :1], 0.01)
        far = torch.full_like(rays_o[..., :1], self.config.bound * 1.5)
        
        # Render rays
        outputs = self.renderer.render_rays(self.model, rays_o, rays_d, near, far, num_samples=128)
        
        # Compute loss
        losses = self.criterion(outputs['rgb'], targets)
        
        # Backward pass
        self.optimizer.zero_grad()
        losses['total_loss'].backward()
        self.optimizer.step()
        
        return {k: v.item() if torch.is_tensor(v) else v for k, v in losses.items()}
    
    def train(self, train_loader: DataLoader, num_epochs: int = 20,
              log_dir: str = None, ckpt_dir: str = None, save_interval: int = 10):
        """Train the model."""
        if log_dir:
            self.setup_logging(log_dir)
        
        if ckpt_dir:
            os.makedirs(ckpt_dir, exist_ok=True)
        
        print(f"Training Instant NGP for {num_epochs} epochs")
        print(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        
        for epoch in range(num_epochs):
            epoch_losses = []
            start_time = time.time()
            
            pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs}')
            for batch in pbar:
                losses = self.train_step(batch)
                epoch_losses.append(losses)
                self.global_step += 1
                
                pbar.set_postfix({
                    'loss': f"{losses['total_loss']:.4f}",
                    'psnr': f"{losses['psnr']:.2f}",
                    'lr': f"{self.optimizer.param_groups[0]['lr']:.2e}"
                })
                
                # Log to tensorboard
                if self.writer:
                    for key, value in losses.items():
                        self.writer.add_scalar(f'train/{key}', value, self.global_step)
            
            # Update scheduler
            self.scheduler.step()
            
            # Compute epoch averages
            avg_losses = {}
            for key in epoch_losses[0].keys():
                avg_losses[key] = np.mean([loss[key] for loss in epoch_losses])
            
            epoch_time = time.time() - start_time
            print(f"Epoch {epoch+1} | Time: {epoch_time:.2f}s | "
                  f"Loss: {avg_losses['total_loss']:.4f} | PSNR: {avg_losses['psnr']:.2f}")
            
            # Save checkpoint
            if ckpt_dir and (epoch + 1) % save_interval == 0:
                self.save_checkpoint(ckpt_dir, epoch)
        
        # Save final checkpoint
        if ckpt_dir:
            self.save_checkpoint(ckpt_dir, num_epochs - 1, is_final=True)
    
    def save_checkpoint(self, ckpt_dir: str, epoch: int, is_final: bool = False):
        """Save checkpoint."""
        ckpt_name = 'final.pth' if is_final else f'epoch_{epoch+1:04d}.pth'
        ckpt_path = os.path.join(ckpt_dir, ckpt_name)
        
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'config': self.config
        }, ckpt_path)
        
        print(f"Checkpoint saved: {ckpt_path}")
    
    def load_checkpoint(self, ckpt_path: str):
        """Load checkpoint."""
        checkpoint = torch.load(ckpt_path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        print(f"Checkpoint loaded: {ckpt_path}")
    
    def render_image(self, H: int, W: int, K: np.ndarray, c2w: np.ndarray, chunk: int = 8192):
        """Render full image."""
        self.model.eval()
        
        with torch.no_grad():
            # Generate rays
            i, j = np.meshgrid(np.arange(W), np.arange(H), indexing='xy')
            dirs = np.stack([(i-K[0][2])/K[0][0], -(j-K[1][2])/K[1][1], -np.ones_like(i)], -1)
            rays_d = np.sum(dirs[..., np.newaxis, :] * c2w[:3,:3], -1)
            rays_o = np.broadcast_to(c2w[:3,-1], np.shape(rays_d))
            
            rays_o = torch.from_numpy(rays_o).float().to(self.device).reshape(-1, 3)
            rays_d = torch.from_numpy(rays_d).float().to(self.device).reshape(-1, 3)
            
            # Render in chunks
            all_rgb = []
            for i in range(0, rays_o.shape[0], chunk):
                rays_o_chunk = rays_o[i:i+chunk]
                rays_d_chunk = rays_d[i:i+chunk]
                
                near = torch.full_like(rays_o_chunk[..., :1], 0.01)
                far = torch.full_like(rays_o_chunk[..., :1], self.config.bound * 1.5)
                
                outputs = self.renderer.render_rays(
                    self.model, rays_o_chunk, rays_d_chunk, near, far, num_samples=128
                )
                all_rgb.append(outputs['rgb'])
            
            rgb = torch.cat(all_rgb, 0).reshape(H, W, 3)
            return rgb.cpu().numpy()
