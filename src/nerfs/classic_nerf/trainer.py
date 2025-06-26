"""
Trainer module for Classic NeRF.

Handles training loop, validation, and model management.
"""

from __future__ import annotations

import os
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from typing import Dict, Optional, Tuple, Any, Union 
import imageio
from tqdm import tqdm
from torch import device as torch_device

from .core import NeRF, NeRFConfig, NeRFRenderer, NeRFLoss
from .dataset import create_nerf_dataloader
from .utils import to8b, img2mse, mse2psnr


class NeRFTrainer:
    """Trainer class for Classic NeRF."""
    
    def __init__(
        self, config: NeRFConfig, device: Optional[str | torch_device] = None, **kwargs
    ) -> None:
        """
        Initialize NeRF trainer.
        
        Args:
            config: NeRF configuration
            device: Computing device
        """
        self.config = config
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Create models
        self.model, self.model_fine, self.render_kwargs_train = create_nerf(config)
        self.model = self.model.to(self.device)
        if self.model_fine is not None:
            self.model_fine = self.model_fine.to(self.device)
        
        # Create renderer
        self.renderer = NeRFRenderer(config)
        
        # Create loss function
        self.criterion = NeRFLoss(config)
        
        # Create optimizer
        params = list(self.model.parameters())
        if self.model_fine is not None:
            params += list(self.model_fine.parameters())
        
        self.optimizer = optim.Adam(
            params,
            lr=config.learning_rate,
            betas=(0.9, 0.999)
        )
        
        # Learning rate scheduler
        self.scheduler = optim.lr_scheduler.ExponentialLR(
            self.optimizer, gamma=0.1**(1/(config.lrate_decay*1000))
        )
        
        # Training state
        self.global_step = 0
        self.start_epoch = 0
        
        # Logging
        self.writer = None
        
    def setup_logging(self, log_dir: str):
        """Setup tensorboard logging."""
        self.writer = SummaryWriter(log_dir)
    
    def train_step(self, batch: dict[str, torch.Tensor]) -> dict[str, float]:
        """
        Single training step.
        
        Args:
            batch: Batch of rays and targets
            
        Returns:
            Dictionary of losses and metrics
        """
        self.model.train()
        if self.model_fine is not None:
            self.model_fine.train()
        
        # Move batch to device
        rays_o = batch['rays_o'].to(self.device)
        rays_d = batch['rays_d'].to(self.device)
        targets = batch['targets'].to(self.device)
        
        # Create ray batch for renderer
        ray_batch = {
            'rays_o': rays_o, 'rays_d': rays_d, 'near': torch.full_like(
                rays_o[...,
                :1],
                self.config.near,
            )
        }
        
        # Add view directions if using them
        if self.config.use_viewdirs:
            viewdirs = rays_d / torch.norm(rays_d, dim=-1, keepdim=True)
            ray_batch['viewdirs'] = viewdirs
        
        # Render rays
        outputs = self.renderer.render_rays(
            ray_batch, self.model, self.model_fine
        )
        
        # Compute losses
        losses = self.criterion(outputs, targets)
        
        # Backward pass
        self.optimizer.zero_grad()
        losses['total_loss'].backward()
        self.optimizer.step()
        
        # Update learning rate
        self.scheduler.step()
        
        return {k: v.item() if torch.is_tensor(v) else v for k, v in losses.items()}
    
    def validate(self, val_loader: DataLoader) -> dict[str, float]:
        """
        Run validation.
        
        Args:
            val_loader: Validation data loader
            
        Returns:
            Dictionary of validation metrics
        """
        self.model.eval()
        if self.model_fine is not None:
            self.model_fine.eval()
        
        total_loss = 0.0
        total_psnr = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for batch in val_loader:
                # Move batch to device
                rays_o = batch['rays_o'].to(self.device)
                rays_d = batch['rays_d'].to(self.device)
                targets = batch['targets'].to(self.device)
                
                # Create ray batch
                ray_batch = {
                    'rays_o': rays_o, 'rays_d': rays_d, 'near': torch.full_like(
                        rays_o[...,
                        :1],
                        self.config.near,
                    )
                }
                
                if self.config.use_viewdirs:
                    viewdirs = rays_d / torch.norm(rays_d, dim=-1, keepdim=True)
                    ray_batch['viewdirs'] = viewdirs
                
                # Render rays
                outputs = self.renderer.render_rays(
                    ray_batch, self.model, self.model_fine
                )
                
                # Compute losses  
                losses = self.criterion(outputs, targets)
                
                total_loss += losses['total_loss'].item()
                total_psnr += losses['psnr'].item()
                num_batches += 1
        
        return {
            'val_loss': total_loss / num_batches, 'val_psnr': total_psnr / num_batches
        }
    
    def render_test_image(
        self,
        H: int,
        W: int,
        K: np.ndarray,
        c2w: np.ndarray,
        chunk: int = 1024*32
    ):
        """
        Render a test image.
        
        Args:
            H: Image height
            W: Image width  
            K: Camera intrinsics
            c2w: Camera-to-world transformation
            chunk: Chunk size for rendering
            
        Returns:
            Rendered RGB image
        """
        self.model.eval()
        if self.model_fine is not None:
            self.model_fine.eval()
        
        with torch.no_grad():
            # Generate rays
            i, j = np.meshgrid(
                np.arange(W),
                np.arange(H)
            )
            dirs = np.stack([(i-K[0][2])/K[0][0], -(j-K[1][2])/K[1][1], -np.ones_like(i)], -1)
            rays_d = np.sum(dirs[..., np.newaxis, :] * c2w[:3, :3], -1)
            rays_o = np.broadcast_to(c2w[:3, -1], np.shape(rays_d))
            
            # Convert to tensors
            rays_o = torch.from_numpy(rays_o).float().to(self.device)
            rays_d = torch.from_numpy(rays_d).float().to(self.device)
            
            # Flatten
            rays_o = rays_o.reshape(-1, 3)
            rays_d = rays_d.reshape(-1, 3)
            
            # Render in chunks
            all_ret = []
            for i in range(0, rays_o.shape[0], chunk):
                rays_o_chunk = rays_o[i:i+chunk]
                rays_d_chunk = rays_d[i:i+chunk]
                
                ray_batch = {
                    'rays_o': rays_o_chunk, 'rays_d': rays_d_chunk, 'near': torch.full_like(
                        rays_o_chunk[...,
                        :1],
                        self.config.near,
                    )
                }
                
                if self.config.use_viewdirs:
                    viewdirs = rays_d_chunk / torch.norm(rays_d_chunk, dim=-1, keepdim=True)
                    ray_batch['viewdirs'] = viewdirs
                
                ret = self.renderer.render_rays(
                    ray_batch, self.model, self.model_fine
                )
                all_ret.append(ret['rgb_map'])
            
            # Concatenate results
            rgb_map = torch.cat(all_ret, 0)
            rgb_map = rgb_map.reshape(H, W, 3)
            
            return rgb_map.cpu().numpy()
    
    def train(
        self,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        num_epochs: int = 100,
        log_dir: str = None,
        ckpt_dir: str = None,
        val_interval: int = 10,
        save_interval: int = 50,
        render_test_interval: int = 100,
        test_render_kwargs: Dict = None
    ):
        """
        Train the NeRF model.
        
        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            num_epochs: Number of training epochs
            log_dir: Directory for tensorboard logs
            ckpt_dir: Directory for checkpoints
            val_interval: Validation interval in epochs
            save_interval: Save interval in epochs
            render_test_interval: Test rendering interval in epochs
            test_render_kwargs: Kwargs for test rendering
        """
        if log_dir is not None:
            self.setup_logging(log_dir)
            
        if ckpt_dir is not None:
            os.makedirs(ckpt_dir, exist_ok=True)
        
        print(f"Starting training for {num_epochs} epochs...")
        print(f"Device: {self.device}")
        print(f"Coarse model parameters: {sum(p.numel() for p in self.model.parameters()):, }")
        if self.model_fine is not None:
            print(f"Fine model parameters: {sum(p.numel() for p in self.model_fine.parameters()):,}")
        
        for epoch in range(self.start_epoch, num_epochs):
            epoch_losses = []
            epoch_start_time = time.time()
            
            # Training loop
            train_pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs}')
            for batch in train_pbar:
                losses = self.train_step(batch)
                epoch_losses.append(losses)
                self.global_step += 1
                
                # Update progress bar
                train_pbar.set_postfix({
                    'loss': f"{losses['total_loss']:.4f}"
                })
                
                # Log to tensorboard
                if self.writer is not None:
                    for key, value in losses.items():
                        self.writer.add_scalar(f'train/{key}', value, self.global_step)
                    self.writer.add_scalar(
                        'train/lr',
                        self.optimizer.param_groups[0]['lr'],
                        self.global_step,
                    )
            
            # Compute epoch averages
            avg_losses = {}
            for key in epoch_losses[0].keys():
                avg_losses[key] = np.mean([loss[key] for loss in epoch_losses])
            
            epoch_time = time.time() - epoch_start_time
            print(f"Epoch {epoch+1} completed in {epoch_time:.2f}s | "
                  f"Loss: {avg_losses['total_loss']:.4f} | "
                  f"PSNR: {avg_losses['psnr']:.2f}")
            
            # Validation
            if val_loader is not None and (epoch + 1) % val_interval == 0:
                val_metrics = self.validate(val_loader)
                print(f"Validation | Loss: {val_metrics['val_loss']:.4f} | "
                      f"PSNR: {val_metrics['val_psnr']:.2f}")
                
                if self.writer is not None:
                    for key, value in val_metrics.items():
                        self.writer.add_scalar(key, value, epoch)
            
            # Save checkpoint
            if ckpt_dir is not None and (epoch + 1) % save_interval == 0:
                self.save_checkpoint(ckpt_dir, epoch)
            
            # Render test image
            if (test_render_kwargs is not None and 
                (epoch + 1) % render_test_interval == 0):
                self.render_and_save_test(**test_render_kwargs, epoch=epoch)
        
        print("Training completed!")
        
        # Save final checkpoint
        if ckpt_dir is not None:
            self.save_checkpoint(ckpt_dir, num_epochs - 1, is_final=True)
    
    def save_checkpoint(
        self, save_dir: str, epoch: int, is_final: bool = False, extra_state: Optional[dict[str, Any]] = None
    ) -> None:
        """Save model checkpoint."""
        if save_dir is None:
            save_dir = os.getcwd()
        
        ckpt_name = 'final.pth' if is_final else f'epoch_{epoch:04d}.pth'
        ckpt_path = os.path.join(save_dir, ckpt_name)
        
        checkpoint = {
            'epoch': epoch, 
            'global_step': self.global_step, 
            'model_coarse_state_dict': self.model.state_dict()
        }
        
        if self.model_fine is not None:
            checkpoint['model_fine_state_dict'] = self.model_fine.state_dict()
            checkpoint['optimizer_fine_state_dict'] = self.optimizer.state_dict()
        
        if extra_state:
            checkpoint.update(extra_state)
        
        torch.save(checkpoint, ckpt_path)
        print(f"Checkpoint saved: {ckpt_path}")
    
    def load_checkpoint(self, ckpt_path: str):
        """Load model checkpoint."""
        checkpoint = torch.load(ckpt_path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_coarse_state_dict'])
        if self.model_fine is not None and 'model_fine_state_dict' in checkpoint:
            self.model_fine.load_state_dict(checkpoint['model_fine_state_dict'])
        
        self.optimizer.load_state_dict(checkpoint['optimizer_coarse_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        self.start_epoch = checkpoint['epoch'] + 1
        self.global_step = checkpoint['global_step']
        
        print(f"Checkpoint loaded: {ckpt_path}")
    
    def render_and_save_test(
        self,
        H: int,
        W: int,
        K: np.ndarray,
        pose: np.ndarray,
        save_path: str,
        epoch: int
    ):
        """Render and save test image."""
        rgb = self.render_test_image(H, W, K, pose)
        rgb8 = to8b(rgb)
        
        filename = f'test_epoch_{epoch+1:04d}.png'
        imageio.imwrite(os.path.join(save_path, filename), rgb8)
        print(f"Test image saved: {filename}")
