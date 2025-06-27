from typing import Any, Optional, Union
"""
Training components for Mega-NeRF++

This module implements advanced training strategies for large-scale photogrammetric scenes:
- Multi-scale progressive training
- Memory-efficient batch processing
- Distributed training support
- Adaptive learning rate scheduling
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.cuda.amp import GradScaler, autocast
import numpy as np
import time
import os
from pathlib import Path
import logging
import wandb
from tqdm import tqdm
import json

from .core import MegaNeRFPlus, MegaNeRFPlusConfig
from .memory_manager import MemoryManager
from .multires_renderer import PhotogrammetricVolumetricRenderer

class MegaNeRFPlusTrainer:
    """
    Main trainer for Mega-NeRF++ with photogrammetric optimizations
    """
    
    def __init__(
        self,
        config: MegaNeRFPlusConfig,
        model: MegaNeRFPlus,
        train_dataset,
        val_dataset=None,
        device='cuda',
        log_dir='./logs',
        checkpoint_dir='./checkpoints',
    ) -> None:
        """
        Args:
            config: Training configuration
            model: Mega-NeRF++ model
            train_dataset: Training dataset
            val_dataset: Validation dataset (optional)
            device: Training device
            log_dir: Directory for logging
            checkpoint_dir: Directory for checkpoints
        """
        self.config = config
        self.model = model
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.device = device
        
        # Setup directories
        self.log_dir = Path(log_dir)
        self.checkpoint_dir = Path(checkpoint_dir)
        self.log_dir.mkdir(exist_ok=True)
        self.checkpoint_dir.mkdir(exist_ok=True)
        
        # Setup logging
        self._setup_logging()
        
        # Training state
        self.global_step = 0
        self.epoch = 0
        self.best_val_psnr = 0.0
        
        # Model to device
        self.model = self.model.to(device)
        
        # Setup optimizer
        self.optimizer = self._create_optimizer()
        self.scheduler = self._create_scheduler()
        
        # Mixed precision training
        self.scaler = GradScaler() if config.use_mixed_precision else None
        
        # Memory manager
        self.memory_manager = MemoryManager(max_memory_gb=config.max_memory_gb)
        
        # Renderer for validation
        self.renderer = PhotogrammetricVolumetricRenderer(config)
        
        # Progressive training parameters
        self.current_resolution_level = 0
        self.resolution_schedule = self._create_resolution_schedule()
        
        # Metrics tracking
        self.train_metrics = {}
        self.val_metrics = {}
        
    def _setup_logging(self):
        """Setup logging configuration"""
        logging.basicConfig(
            level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', handlers=[
                logging.FileHandler(self.log_dir / 'training.log'), logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
        
        # Initialize wandb if available
        try:
            wandb.init(
                project="mega_nerf_plus", config=self.config.__dict__, dir=str(self.log_dir)
            )
            self.use_wandb = True
        except Exception as e:
            self.logger.warning(f"Failed to initialize wandb: {e}")
            self.use_wandb = False
    
    def _create_optimizer(self) -> torch.optim.Optimizer:
        """Create optimizer with appropriate settings"""
        
        # Different learning rates for different components
        param_groups = []
        
        # Spatial encoder parameters (typically need higher learning rate)
        spatial_params = []
        if hasattr(self.model.nerf_model, 'spatial_encoder'):
            spatial_params.extend(self.model.nerf_model.spatial_encoder.parameters())
        
        if spatial_params:
            param_groups.append({
                'params': spatial_params, 'lr': self.config.lr_init * 2.0, # Higher lr for spatial encoding
                'name': 'spatial_encoder'
            })
        
        # MLP parameters
        mlp_params = []
        if hasattr(self.model.nerf_model, 'nerf_mlp'):
            mlp_params.extend(self.model.nerf_model.nerf_mlp.parameters())
        
        if mlp_params:
            param_groups.append({
                'params': mlp_params, 'lr': self.config.lr_init, 'name': 'nerf_mlp'
            })
        
        # Fallback: all parameters
        if not param_groups:
            param_groups = [{'params': self.model.parameters(), 'lr': self.config.lr_init}]
        
        return torch.optim.AdamW(param_groups, betas=(0.9, 0.99), eps=1e-15)
    
    def _create_scheduler(self):
        """Create learning rate scheduler"""
        return torch.optim.lr_scheduler.ExponentialLR(
            self.optimizer, gamma=(
                self.config.lr_final / self.config.lr_init,
            )
        )
    
    def _create_resolution_schedule(self) -> list[tuple[int, int]]:
        """Create progressive resolution training schedule"""
        
        if not self.config.progressive_upsampling:
            return [(self.config.max_image_resolution, self.config.lr_decay_steps)]
        
        # Progressive schedule: start low, increase resolution
        base_res = 128
        schedules = []
        steps_per_level = self.config.lr_decay_steps // 4
        
        current_res = base_res
        while current_res <= self.config.max_image_resolution:
            schedules.append((current_res, steps_per_level))
            current_res *= 2
        
        # Final level at max resolution
        remaining_steps = self.config.lr_decay_steps - sum(s[1] for s in schedules)
        if remaining_steps > 0:
            schedules.append((self.config.max_image_resolution, remaining_steps))
        
        return schedules
    
    def train(self, num_epochs: int = 100):
        """Main training loop"""
        
        self.logger.info(f"Starting training for {num_epochs} epochs")
        self.logger.info(f"Training on device: {self.device}")
        self.logger.info(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):, }")
        
        start_time = time.time()
        
        for epoch in range(num_epochs):
            self.epoch = epoch
            
            # Update resolution schedule
            self._update_resolution_schedule()
            
            # Train epoch
            train_metrics = self._train_epoch()
            
            # Validation
            if self.val_dataset is not None and epoch % 5 == 0:
                val_metrics = self._validate()
                self.val_metrics = val_metrics
                
                # Save best model
                if val_metrics.get('psnr', 0) > self.best_val_psnr:
                    self.best_val_psnr = val_metrics['psnr']
                    self._save_checkpoint('best_model.pth')
            
            # Update learning rate
            self.scheduler.step()
            
            # Logging
            self._log_metrics(train_metrics, self.val_metrics, epoch)
            
            # Save checkpoint
            if epoch % 10 == 0:
                self._save_checkpoint(f'checkpoint_epoch_{epoch:04d}.pth')
        
        total_time = time.time() - start_time
        self.logger.info(f"Training completed in {total_time:.2f} seconds")
        
        # Save final model
        self._save_checkpoint('final_model.pth')
    
    def _train_epoch(self) -> dict[str, float]:
        """Train for one epoch"""
        
        self.model.train()
        
        total_loss = 0.0
        rgb_loss = 0.0
        depth_loss = 0.0
        semantic_loss = 0.0
        distortion_loss = 0.0
        num_batches = 0
        
        # Create data loader for current resolution
        dataloader = self._create_dataloader_for_resolution()
        
        progress_bar = tqdm(dataloader, desc=f"Epoch {self.epoch}")
        
        for batch_idx, batch in enumerate(progress_bar):
            # Memory management
            if batch_idx % 100 == 0:
                self.memory_manager.cleanup_cache()
            
            # Move batch to device
            batch = self._batch_to_device(batch)
            
            # Forward pass
            with autocast() if self.config.use_mixed_precision else torch.no_grad(False):
                loss_dict = self._forward_pass(batch)
            
            # Backward pass
            self.optimizer.zero_grad()
            
            total_batch_loss = loss_dict['total_loss']
            
            if self.config.use_mixed_precision:
                self.scaler.scale(total_batch_loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                total_batch_loss.backward()
                
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                
                self.optimizer.step()
            
            # Update metrics
            total_loss += total_batch_loss.item()
            rgb_loss += loss_dict.get('rgb_loss', 0.0)
            depth_loss += loss_dict.get('depth_loss', 0.0)
            semantic_loss += loss_dict.get('semantic_loss', 0.0)
            distortion_loss += loss_dict.get('distortion_loss', 0.0)
            num_batches += 1
            
            self.global_step += 1
            
            # Update progress bar
            progress_bar.set_postfix({
                'loss': f'{total_batch_loss.item():.4f}'
            })
            
            # Log frequently during first epoch
            if self.epoch == 0 and batch_idx % 10 == 0:
                self._log_step_metrics(loss_dict, batch_idx)
        
        return {
            'total_loss': total_loss / num_batches, 'rgb_loss': rgb_loss / num_batches, 'depth_loss': depth_loss / num_batches, 'semantic_loss': semantic_loss / num_batches, 'distortion_loss': distortion_loss / num_batches
        }
    
    def _forward_pass(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """Forward pass and loss computation"""
        
        rays = batch['rays']  # [B, 6] (origin + direction)
        target_rgb = batch['rgbs']  # [B, 3]
        
        rays_o = rays[..., :3]
        rays_d = rays[..., 3:6]
        
        # Render rays
        render_results = self.model.render_rays(
            rays_o, rays_d, near=self.train_dataset.near, far=self.train_dataset.far, lod=self.current_resolution_level, white_bkgd=True
        )
        
        # Compute losses
        loss_dict = self._compute_losses(render_results, target_rgb, batch)
        
        return loss_dict
    
    def _compute_losses(
        self,
        render_results: dict[str,
        torch.Tensor],
        target_rgb: torch.Tensor,
        batch: dict[str,
        torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        """Compute all loss components"""
        
        losses = {}
        
        # RGB loss (coarse)
        if 'coarse' in render_results:
            coarse_rgb = render_results['coarse']['rgb']
            rgb_loss_coarse = F.mse_loss(coarse_rgb, target_rgb)
            losses['rgb_loss_coarse'] = rgb_loss_coarse
        else:
            losses['rgb_loss_coarse'] = 0.0
        
        # RGB loss (fine)
        if 'fine' in render_results:
            fine_rgb = render_results['fine']['rgb']
            rgb_loss_fine = F.mse_loss(fine_rgb, target_rgb)
            losses['rgb_loss_fine'] = rgb_loss_fine
            primary_rgb = fine_rgb
        else:
            losses['rgb_loss_fine'] = 0.0
            primary_rgb = render_results['coarse']['rgb']
        
        # Total RGB loss
        rgb_loss = losses['rgb_loss_coarse'] + losses['rgb_loss_fine']
        losses['rgb_loss'] = rgb_loss
        
        # Depth loss (if available)
        depth_loss = 0.0
        if 'depth_gt' in batch and self.config.lambda_depth > 0:
            if 'fine' in render_results:
                pred_depth = render_results['fine']['depth']
            else:
                pred_depth = render_results['coarse']['depth']
            
            target_depth = batch['depth_gt']
            depth_loss = F.l1_loss(pred_depth, target_depth)
            losses['depth_loss'] = depth_loss
        
        # Semantic loss (if available)
        semantic_loss = 0.0
        if 'semantics_gt' in batch and self.config.lambda_semantic > 0:
            if 'semantics' in render_results.get('fine', {}):
                pred_semantics = render_results['fine']['semantics']
                target_semantics = batch['semantics_gt']
                semantic_loss = F.cross_entropy(pred_semantics, target_semantics)
                losses['semantic_loss'] = semantic_loss
        
        # Distortion loss for better geometry
        distortion_loss = 0.0
        if self.config.lambda_distortion > 0:
            for key in ['coarse', 'fine']:
                if key in render_results and 'weights' in render_results[key]:
                    weights = render_results[key]['weights']
                    # Distortion regularization
                    distortion = self._compute_distortion_loss(weights)
                    distortion_loss += distortion
            
            losses['distortion_loss'] = distortion_loss
        
        # Total loss
        total_loss = (self.config.lambda_rgb * rgb_loss +
                     self.config.lambda_depth * depth_loss +
                     self.config.lambda_semantic * semantic_loss +
                     self.config.lambda_distortion * distortion_loss)
        
        losses['total_loss'] = total_loss
        
        return losses
    
    def _compute_distortion_loss(self, weights: torch.Tensor) -> torch.Tensor:
        """Compute distortion loss for better geometry"""
        
        # Encourage weights to be concentrated
        # This is a simplified distortion loss
        n_samples = weights.shape[-1]
        t_indices = torch.arange(n_samples, device=weights.device, dtype=torch.float32)
        t_indices = t_indices / (n_samples - 1)  # Normalize to [0, 1]
        
        # Compute center of mass
        center_of_mass = torch.sum(weights * t_indices, dim=-1, keepdim=True)
        
        # Compute variance
        variance = torch.sum(weights * (t_indices - center_of_mass)**2, dim=-1)
        
        return torch.mean(variance)
    
    def _validate(self) -> dict[str, float]:
        """Validation loop"""
        
        self.model.eval()
        
        total_psnr = 0.0
        total_ssim = 0.0
        num_images = 0
        
        with torch.no_grad():
            for i in range(min(len(self.val_dataset), 10)):  # Validate on subset
                data = self.val_dataset[i]
                
                if 'image' in data:
                    # Full image validation
                    target_img = data['image'].to(self.device)
                    pose = data['pose'].to(self.device)
                    intrinsic = data['intrinsics'].to(self.device)
                    
                    # Render full image
                    rendered_img = self._render_full_image(pose, intrinsic, target_img.shape[:2])
                    
                    # Compute metrics
                    psnr = self._compute_psnr(rendered_img, target_img)
                    ssim = self._compute_ssim(rendered_img, target_img)
                    
                    total_psnr += psnr
                    total_ssim += ssim
                    num_images += 1
        
        if num_images > 0:
            avg_psnr = total_psnr / num_images
            avg_ssim = total_ssim / num_images
        else:
            avg_psnr = 0.0
            avg_ssim = 0.0
        
        return {
            'psnr': avg_psnr, 'ssim': avg_ssim
        }
    
    def _render_full_image(
        self,
        pose: torch.Tensor,
        intrinsic: torch.Tensor,
        image_size: tuple[int,
        int],
    ) -> torch.Tensor:
        """Render a full image"""
        
        h, w = image_size
        
        # Generate rays for full image
        i_coords, j_coords = torch.meshgrid(
            torch.arange(
                w,
                dtype=torch.float32,
                device=self.device,
            )
        )
        
        fx, fy = intrinsic[0, 0], intrinsic[1, 1]
        cx, cy = intrinsic[0, 2], intrinsic[1, 2]
        
        dirs = torch.stack([
            (i_coords - cx) / fx, -(j_coords - cy) / fy, -torch.ones_like(i_coords)
        ], dim=-1)
        
        dirs = torch.sum(dirs[..., None, :] * pose[:3, :3], dim=-1)
        origins = pose[:3, 3].expand(dirs.shape)
        
        # Render in chunks to manage memory
        chunk_size = self.config.chunk_size
        rendered_pixels = []
        
        rays_o_flat = origins.reshape(-1, 3)
        rays_d_flat = dirs.reshape(-1, 3)
        
        for i in range(0, len(rays_o_flat), chunk_size):
            chunk_rays_o = rays_o_flat[i:i+chunk_size]
            chunk_rays_d = rays_d_flat[i:i+chunk_size]
            
            chunk_results = self.model.render_rays(
                chunk_rays_o, chunk_rays_d, near=self.train_dataset.near, far=self.train_dataset.far, lod=0, # Use highest quality for validation
                white_bkgd=True
            )
            
            if 'fine' in chunk_results:
                chunk_rgb = chunk_results['fine']['rgb']
            else:
                chunk_rgb = chunk_results['coarse']['rgb']
            
            rendered_pixels.append(chunk_rgb)
        
        # Combine chunks and reshape
        rendered_flat = torch.cat(rendered_pixels, dim=0)
        rendered_img = rendered_flat.reshape(h, w, 3)
        
        return rendered_img
    
    def _compute_psnr(self, pred: torch.Tensor, target: torch.Tensor) -> float:
        """Compute PSNR between prediction and target"""
        mse = F.mse_loss(pred, target)
        psnr = -10.0 * torch.log10(mse)
        return psnr.item()
    
    def _compute_ssim(self, pred: torch.Tensor, target: torch.Tensor) -> float:
        """Compute SSIM between prediction and target (simplified)"""
        # This is a simplified SSIM implementation
        # For production, use a proper SSIM implementation
        
        pred_mean = torch.mean(pred)
        target_mean = torch.mean(target)
        
        pred_var = torch.var(pred)
        target_var = torch.var(target)
        covar = torch.mean((pred - pred_mean) * (target - target_mean))
        
        c1 = 0.01 ** 2
        c2 = 0.03 ** 2
        
        ssim = ((2 * pred_mean * target_mean + c1) * (2 * covar + c2)) / \
               ((pred_mean**2 + target_mean**2 + c1) * (pred_var + target_var + c2))
        
        return ssim.item()
    
    def _update_resolution_schedule(self):
        """Update resolution schedule for progressive training"""
        
        if not self.config.progressive_upsampling:
            return
        
        # Determine current resolution level based on global step
        cumulative_steps = 0
        for level, (resolution, steps) in enumerate(self.resolution_schedule):
            if self.global_step < cumulative_steps + steps:
                self.current_resolution_level = level
                break
            cumulative_steps += steps
        else:
            # Use highest resolution
            self.current_resolution_level = len(self.resolution_schedule) - 1
    
    def _create_dataloader_for_resolution(self):
        """Create dataloader for current resolution level"""
        
        if not hasattr(self.train_dataset, 'set_resolution'):
            # Simple dataloader if resolution control not supported
            return torch.utils.data.DataLoader(
                self.train_dataset, batch_size=self.config.batch_size, shuffle=True, num_workers=4, pin_memory=True
            )
        
        # Set dataset resolution
        current_resolution = self.resolution_schedule[self.current_resolution_level][0]
        self.train_dataset.set_resolution(current_resolution)
        
        return torch.utils.data.DataLoader(
            self.train_dataset, batch_size=self.config.batch_size, shuffle=True, num_workers=4, pin_memory=True
        )
    
    def _batch_to_device(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """Move batch to training device"""
        device_batch = {}
        for key, value in batch.items():
            if isinstance(value, torch.Tensor):
                device_batch[key] = value.to(self.device, non_blocking=True)
            else:
                device_batch[key] = value
        return device_batch
    
    def _log_metrics(
        self,
        train_metrics: dict[str,
        float],
        val_metrics: dict[str,
        float],
        epoch: int,
    ) -> None:
        """Log training metrics"""
        
        # Console logging
        log_str = f"Epoch {epoch:04d} | "
        log_str += f"Loss: {train_metrics['total_loss']:.4f} | "
        log_str += f"RGB: {train_metrics['rgb_loss']:.4f} | "
        log_str += f"LR: {self.optimizer.param_groups[0]['lr']:.2e}"
        
        if val_metrics:
            log_str += f" | Val PSNR: {val_metrics['psnr']:.2f}"
        
        self.logger.info(log_str)
        
        # Wandb logging
        if self.use_wandb:
            wandb_dict = {
                'epoch': epoch, 'global_step': self.global_step, 'train_loss': train_metrics['total_loss'], 'train_rgb_loss': train_metrics['rgb_loss'], 'learning_rate': self.optimizer.param_groups[0]['lr'], 'resolution_level': self.current_resolution_level
            }
            
            if val_metrics:
                wandb_dict.update({
                    'val_psnr': val_metrics['psnr'], 'val_ssim': val_metrics['ssim']
                })
            
            wandb.log(wandb_dict)
    
    def _log_step_metrics(self, loss_dict: dict[str, torch.Tensor], step: int):
        """Log step-level metrics"""
        
        if self.use_wandb and step % 100 == 0:
            wandb_dict = {
                'step_loss': loss_dict['total_loss'].item(
                )
            }
            wandb.log(wandb_dict)
    
    def _save_checkpoint(self, filename: str):
        """Save training checkpoint"""
        
        checkpoint = {
            'epoch': self.epoch, 'global_step': self.global_step, 'model_state_dict': self.model.state_dict(
            )
        }
        
        if self.scaler is not None:
            checkpoint['scaler_state_dict'] = self.scaler.state_dict()
        
        checkpoint_path = self.checkpoint_dir / filename
        torch.save(checkpoint, checkpoint_path)
        self.logger.info(f"Saved checkpoint: {checkpoint_path}")
    
    def load_checkpoint(self, checkpoint_path: str) -> bool:
        """Load training checkpoint"""
        
        try:
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            
            self.epoch = checkpoint['epoch']
            self.global_step = checkpoint['global_step']
            self.best_val_psnr = checkpoint.get('best_val_psnr', 0.0)
            self.current_resolution_level = checkpoint.get('current_resolution_level', 0)
            
            if self.scaler is not None and 'scaler_state_dict' in checkpoint:
                self.scaler.load_state_dict(checkpoint['scaler_state_dict'])
            
            self.logger.info(f"Loaded checkpoint from: {checkpoint_path}")
            self.logger.info(f"Resuming from epoch {self.epoch}, step {self.global_step}")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to load checkpoint: {e}")
            return False

class MultiScaleTrainer(MegaNeRFPlusTrainer):
    """
    Multi-scale trainer with enhanced progressive training
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        # Multi-scale specific parameters
        self.scale_factors = [8, 4, 2, 1]  # Downsampling factors
        self.current_scale_idx = 0
        self.steps_per_scale = self.config.lr_decay_steps // len(self.scale_factors)
    
    def _create_resolution_schedule(self) -> list[tuple[int, int]]:
        """Create multi-scale resolution schedule"""
        
        schedules = []
        base_resolution = self.config.max_image_resolution // max(self.scale_factors)
        
        for scale_factor in self.scale_factors:
            resolution = base_resolution * scale_factor
            schedules.append((resolution, self.steps_per_scale))
        
        return schedules
    
    def _update_resolution_schedule(self):
        """Update multi-scale schedule"""
        
        scale_step = self.global_step // self.steps_per_scale
        self.current_scale_idx = min(scale_step, len(self.scale_factors) - 1)
        self.current_resolution_level = self.current_scale_idx

class DistributedTrainer(MegaNeRFPlusTrainer):
    """
    Distributed trainer for multi-GPU training
    """
    
    def __init__(self, *args, **kwargs):
        
        # Initialize distributed training
        if not dist.is_initialized():
            self._init_distributed()
        
        self.local_rank = int(os.environ.get('LOCAL_RANK', 0))
        self.world_size = dist.get_world_size()
        self.rank = dist.get_rank()
        
        super().__init__(*args, **kwargs)
        
        # Wrap model with DDP
        self.model = DDP(
            self.model, device_ids=[self.local_rank], find_unused_parameters=True
        )
    
    def _init_distributed(self):
        """Initialize distributed training"""
        
        if 'RANK' in os.environ:
            dist.init_process_group(backend='nccl')
        else:
            # Single node training
            os.environ['MASTER_ADDR'] = 'localhost'
            os.environ['MASTER_PORT'] = '12345'
            dist.init_process_group(
                backend='nccl', rank=0, world_size=1
            )
    
    def _create_dataloader_for_resolution(self):
        """Create distributed dataloader"""
        
        # Create distributed sampler
        sampler = torch.utils.data.distributed.DistributedSampler(
            self.train_dataset, num_replicas=self.world_size, rank=self.rank, shuffle=True
        )
        
        return torch.utils.data.DataLoader(
            self.train_dataset, batch_size=self.config.batch_size // self.world_size, sampler=sampler, num_workers=4, pin_memory=True
        )
    
    def _log_metrics(
        self,
        train_metrics: dict[str,
        float],
        val_metrics: dict[str,
        float],
        epoch: int,
    ) -> None:
        """Log metrics only on rank 0"""
        
        if self.rank == 0:
            super()._log_metrics(train_metrics, val_metrics, epoch)
    
    def _save_checkpoint(self, filename: str):
        """Save checkpoint only on rank 0"""
        
        if self.rank == 0:
            # Use module.state_dict() for DDP models
            checkpoint = {
                'epoch': self.epoch, 'global_step': self.global_step, 'model_state_dict': self.model.module.state_dict(
                )
            }
            
            checkpoint_path = self.checkpoint_dir / filename
            torch.save(checkpoint, checkpoint_path)
            self.logger.info(f"Saved checkpoint: {checkpoint_path}") 