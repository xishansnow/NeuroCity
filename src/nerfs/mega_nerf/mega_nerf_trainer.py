#!/usr/bin/env python3
"""
Mega-NeRF Trainer Module

This module implements training pipelines for Mega-NeRF models, including parallel training of submodules and sequential training.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
import logging
import os
import time
from tqdm import tqdm
import wandb
from pathlib import Path
import json

from .mega_nerf_model import MegaNeRF, MegaNeRFConfig
from .volumetric_renderer import VolumetricRenderer, BatchRenderer
from .mega_nerf_dataset import MegaNeRFDataset

logger = logging.getLogger(__name__)


class MegaNeRFTrainer:
    """Main trainer for Mega-NeRF models"""
    
    def __init__(
        self,
        config: MegaNeRFConfig,
        model: MegaNeRF,
        dataset: MegaNeRFDataset,
        output_dir: str,
        device: str = 'cuda',
    ):
        """
        Initialize Mega-NeRF trainer
        
        Args:
            config: Mega-NeRF configuration
            model: Mega-NeRF model
            dataset: Training dataset
            output_dir: Output directory for checkpoints and logs
            device: Device for training
        """
        self.config = config
        self.model = model.to(device)
        self.dataset = dataset
        self.output_dir = Path(output_dir)
        self.device = device
        
        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize renderer
        self.renderer = VolumetricRenderer(
            num_coarse_samples=config.num_coarse, num_fine_samples=config.num_fine, near=config.near, far=config.far, use_hierarchical_sampling=True
        )
        
        self.batch_renderer = BatchRenderer(self.renderer)
        
        # Training state
        self.global_step = 0
        self.current_epoch = 0
        self.best_psnr = 0.0
        
        # Optimizers (will be set up per submodule)
        self.optimizers = {}
        self.schedulers = {}
        
        # Loss tracking
        self.loss_history = []
        
        logger.info(f"Initialized trainer with {len(self.model.submodules)} submodules")
    
    def setup_optimizers(self, submodule_indices: Optional[list[int]] = None):
        """Setup optimizers for specified submodules"""
        if submodule_indices is None:
            submodule_indices = list(range(len(self.model.submodules)))
        
        for idx in submodule_indices:
            if idx < len(self.model.submodules):
                submodule = self.model.submodules[idx]
                
                # Optimizer
                optimizer = optim.Adam(
                    submodule.parameters(), lr=self.config.learning_rate, weight_decay=1e-6
                )
                self.optimizers[idx] = optimizer
                
                # Scheduler
                scheduler = optim.lr_scheduler.ExponentialLR(
                    optimizer, gamma=self.config.lr_decay ** (1/10000)
                )
                self.schedulers[idx] = scheduler
        
        logger.info(f"Setup optimizers for {len(submodule_indices)} submodules")
    
    def train_submodule(
        self,
        submodule_idx: int,
        num_iterations: int = 10000,
        log_interval: int = 100,
        val_interval: int = 1000,
    ):
        """
        Train a single submodule
        
        Args:
            submodule_idx: Index of submodule to train
            num_iterations: Number of training iterations
            log_interval: Logging interval
            val_interval: Validation interval
            
        Returns:
            Training statistics
        """
        if submodule_idx >= len(self.model.submodules):
            raise ValueError(f"Submodule {submodule_idx} does not exist")
        
        submodule = self.model.submodules[submodule_idx]
        
        # Setup optimizer if not exists
        if submodule_idx not in self.optimizers:
            self.setup_optimizers([submodule_idx])
        
        optimizer = self.optimizers[submodule_idx]
        scheduler = self.schedulers[submodule_idx]
        
        # Get partition data
        partition_data = self.dataset.get_partition_data(submodule_idx)
        rays_data = partition_data['rays']
        
        if not rays_data:
            logger.warning(f"No data for submodule {submodule_idx}")
            return {}
        
        # Training loop
        submodule.train()
        losses = []
        psnrs = []
        
        pbar = tqdm(range(num_iterations), desc=f"Training submodule {submodule_idx}")
        
        for iteration in pbar:
            # Sample rays
            ray_batch = self._sample_rays_from_partition(rays_data, self.config.batch_size)
            
            # Forward pass
            optimizer.zero_grad()
            
            outputs = self.renderer.render_rays(
                submodule, ray_batch['ray_origins'], ray_batch['ray_directions'], ray_batch.get(
                    'appearance_ids',
                )
            )
            
            # Compute loss
            target_rgb = ray_batch['colors']
            loss = self._compute_loss(outputs, target_rgb)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            scheduler.step()
            
            # Compute metrics
            with torch.no_grad():
                mse = torch.mean((outputs['rgb'] - target_rgb) ** 2)
                psnr = -10 * torch.log10(mse)
            
            losses.append(loss.item())
            psnrs.append(psnr.item())
            
            # Update progress bar
            pbar.set_postfix({
                'loss': f"{loss.item():.4f}"
            })
            
            # Logging
            if iteration % log_interval == 0:
                avg_loss = np.mean(losses[-log_interval:])
                avg_psnr = np.mean(psnrs[-log_interval:])
                
                logger.info(f"Submodule {submodule_idx} - Iter {iteration}: "
                          f"Loss={avg_loss:.4f}, PSNR={avg_psnr:.2f}")
                
                if wandb.run is not None:
                    wandb.log({
                        f'submodule_{submodule_idx}': avg_loss,
                        f'psnr_{submodule_idx}': avg_psnr
                    }, step=self.global_step)
            
            # Validation
            if iteration % val_interval == 0 and iteration > 0:
                val_metrics = self._validate_submodule(submodule_idx)
                
                if wandb.run is not None:
                    for key, value in val_metrics.items():
                        wandb.log({
                            f'submodule_{submodule_idx}': val_metrics['mse'],
                            f'psnr_{submodule_idx}': val_metrics['psnr']
                        }, step=self.global_step)
            
            self.global_step += 1
        
        # Final statistics
        stats = {
            'final_loss': np.mean(
                losses[-100:],
            )
        }
        
        logger.info(f"Completed training submodule {submodule_idx}: "
                   f"Loss={stats['final_loss']:.4f}, PSNR={stats['final_psnr']:.2f}")
        
        return stats
    
    def _sample_rays_from_partition(
        self,
        rays_data: dict[str, np.ndarray],
        batch_size: int
    ):
        """Sample rays from partition data"""
        num_rays = len(rays_data['colors'])
        
        if num_rays == 0:
            # Return empty batch
            return {
                'ray_origins': torch.zeros(
                    0,
                    3,
                    device=self.device,
                )
            }
        
        # Random sampling
        indices = np.random.choice(num_rays, min(batch_size, num_rays), replace=False)
        
        batch = {
            'ray_origins': torch.from_numpy(
                rays_data['ray_origins'][indices],
            )
        }
        
        return batch
    
    def _compute_loss(
        self,
        outputs: dict[str, torch.Tensor],
        target_rgb: torch.Tensor
    ):
        """Compute training loss"""
        # Main RGB loss
        rgb_loss = torch.mean((outputs['rgb'] - target_rgb) ** 2)
        
        # Coarse RGB loss (if available)
        if 'rgb_coarse' in outputs:
            rgb_coarse_loss = torch.mean((outputs['rgb_coarse'] - target_rgb) ** 2)
            total_loss = rgb_loss + 0.5 * rgb_coarse_loss
        else:
            total_loss = rgb_loss
        
        return total_loss
    
    def _validate_submodule(self, submodule_idx: int) -> dict[str, float]:
        """Validate a single submodule"""
        submodule = self.model.submodules[submodule_idx]
        partition_data = self.dataset.get_partition_data(submodule_idx)
        rays_data = partition_data['rays']
        
        if not rays_data:
            return {}
        
        submodule.eval()
        
        # Sample validation rays
        val_batch = self._sample_rays_from_partition(rays_data, 1024)
        
        with torch.no_grad():
            outputs = self.renderer.render_rays(
                submodule, val_batch['ray_origins'], val_batch['ray_directions'], val_batch.get(
                    'appearance_ids',
                )
            )
            
            target_rgb = val_batch['colors']
            mse = torch.mean((outputs['rgb'] - target_rgb) ** 2)
            psnr = -10 * torch.log10(mse)
        
        submodule.train()
        
        return {
            'mse': mse.item(), 'psnr': psnr.item()
        }
    
    def train_sequential(
        self,
        num_iterations_per_submodule: int = 10000,
        log_interval: int = 100,
        val_interval: int = 1000
    ):
        """
        Train all submodules sequentially
        
        Args:
            num_iterations_per_submodule: Number of iterations per submodule
            log_interval: Logging interval
            val_interval: Validation interval
            
        Returns:
            Training statistics for all submodules
        """
        logger.info("Starting sequential training of all submodules")
        
        all_stats = {}
        
        for submodule_idx in range(len(self.model.submodules)):
            logger.info(f"Training submodule {submodule_idx}/{len(self.model.submodules)-1}")
            
            stats = self.train_submodule(
                submodule_idx, num_iterations_per_submodule, log_interval, val_interval
            )
            
            all_stats[f'submodule_{submodule_idx}'] = stats
            
            # Save intermediate checkpoint
            self.save_checkpoint(
                self.output_dir / f'checkpoint_submodule_{submodule_idx}.pth', submodule_idx
            )
        
        logger.info("Completed sequential training of all submodules")
        return all_stats
    
    def save_checkpoint(self, path: str, submodule_idx: Optional[int] = None):
        """Save training checkpoint"""
        checkpoint = {
            'global_step': self.global_step, 'current_epoch': self.current_epoch, 'best_psnr': self.best_psnr, 'config': self.config, 'loss_history': self.loss_history
        }
        
        if submodule_idx is not None:
            # Save single submodule
            if submodule_idx < len(self.model.submodules):
                checkpoint['submodule_state_dict'] = self.model.submodules[submodule_idx].state_dict()
                checkpoint['submodule_idx'] = submodule_idx
                
                if submodule_idx in self.optimizers:
                    checkpoint['optimizer_state_dict'] = self.optimizers[submodule_idx].state_dict()
                    checkpoint['scheduler_state_dict'] = self.schedulers[submodule_idx].state_dict()
        else:
            # Save full model
            checkpoint['model_state_dict'] = self.model.state_dict()
            checkpoint['optimizers'] = {
                idx: opt.state_dict() for idx,
                opt in self.optimizers.items()
            }
            checkpoint['schedulers'] = {
                idx: sch.state_dict() for idx, sch in self.schedulers.items()
            }
        
        torch.save(checkpoint, path)
        logger.info(f"Saved checkpoint to {path}")
    
    def load_checkpoint(self, path: str, submodule_idx: Optional[int] = None):
        """Load training checkpoint"""
        checkpoint = torch.load(path, map_location=self.device)
        
        self.global_step = checkpoint.get('global_step', 0)
        self.current_epoch = checkpoint.get('current_epoch', 0)
        self.best_psnr = checkpoint.get('best_psnr', 0.0)
        self.loss_history = checkpoint.get('loss_history', [])
        
        if submodule_idx is not None:
            # Load single submodule
            if 'submodule_state_dict' in checkpoint:
                self.model.submodules[submodule_idx].load_state_dict(checkpoint['submodule_state_dict'])
                
                if 'optimizer_state_dict' in checkpoint and submodule_idx in self.optimizers:
                    self.optimizers[submodule_idx].load_state_dict(checkpoint['optimizer_state_dict'])
                
                if 'scheduler_state_dict' in checkpoint and submodule_idx in self.schedulers:
                    self.schedulers[submodule_idx].load_state_dict(checkpoint['scheduler_state_dict'])
        else:
            # Load full model
            if 'model_state_dict' in checkpoint:
                self.model.load_state_dict(checkpoint['model_state_dict'])
            
            if 'optimizers' in checkpoint:
                for idx, state_dict in checkpoint['optimizers'].items():
                    if idx in self.optimizers:
                        self.optimizers[idx].load_state_dict(state_dict)
            
            if 'schedulers' in checkpoint:
                for idx, state_dict in checkpoint['schedulers'].items():
                    if idx in self.schedulers:
                        self.schedulers[idx].load_state_dict(state_dict)
        
        logger.info(f"Loaded checkpoint from {path}")
    
    def save_model(self, path: str):
        """Save the trained model"""
        model_data = {
            'model_state_dict': self.model.state_dict(
            )
        }
        
        torch.save(model_data, path)
        logger.info(f"Saved model to {path}")
    
    def load_model(self, path: str):
        """Load a trained model"""
        model_data = torch.load(path, map_location=self.device)
        
        self.model.load_state_dict(model_data['model_state_dict'])
        
        if 'config' in model_data:
            self.config = model_data['config']
        
        logger.info(f"Loaded model from {path}")


class ParallelTrainer:
    """Parallel trainer for training multiple submodules simultaneously"""
    
    def __init__(
        self,
        config: MegaNeRFConfig,
        model: MegaNeRF,
        dataset: MegaNeRFDataset,
        output_dir: str,
        device: str = 'cuda',
        num_parallel_workers: int = 4,
    ):
        """
        Initialize parallel trainer
        
        Args:
            config: Mega-NeRF configuration
            model: Mega-NeRF model
            dataset: Training dataset
            output_dir: Output directory
            device: Device for training
            num_parallel_workers: Number of parallel workers
        """
        self.config = config
        self.model = model
        self.dataset = dataset
        self.output_dir = Path(output_dir)
        self.device = device
        self.num_parallel_workers = num_parallel_workers
        
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Initialized parallel trainer with {num_parallel_workers} workers")
    
    def train_parallel(
        self,
        num_iterations_per_submodule: int = 10000,
        save_interval: int = 5000,
    ):
        """
        Train submodules in parallel
        
        Args:
            num_iterations_per_submodule: Number of iterations per submodule
            save_interval: Interval for saving checkpoints
            
        Returns:
            Training statistics
        """
        from concurrent.futures import ThreadPoolExecutor, as_completed
        
        logger.info("Starting parallel training of submodules")
        
        # Create individual trainers for each submodule
        trainers = []
        for submodule_idx in range(len(self.model.submodules)):
            # Create a copy of the model with only one submodule
            single_submodule_model = self._create_single_submodule_model(submodule_idx)
            
            trainer = MegaNeRFTrainer(
                config=self.config, model=single_submodule_model, dataset=self.dataset, output_dir=self.output_dir / f'submodule_{submodule_idx}', device=self.device
            )
            
            trainers.append((submodule_idx, trainer))
        
        # Train in parallel
        all_stats = {}
        
        with ThreadPoolExecutor(max_workers=self.num_parallel_workers) as executor:
            # Submit training jobs
            future_to_submodule = {
                executor.submit(
                    trainer.train_submodule, submodule_idx, num_iterations_per_submodule
                ): submodule_idx
                for submodule_idx, trainer in trainers
            }
            
            # Collect results
            for future in as_completed(future_to_submodule):
                submodule_idx = future_to_submodule[future]
                try:
                    stats = future.result()
                    all_stats[f'submodule_{submodule_idx}'] = stats
                    logger.info(f"Completed training submodule {submodule_idx}")
                except Exception as e:
                    logger.error(f"Error training submodule {submodule_idx}: {e}")
        
        # Merge trained submodules back to main model
        self._merge_submodules(trainers)
        
        logger.info("Completed parallel training of all submodules")
        return all_stats
    
    def _create_single_submodule_model(self, submodule_idx: int) -> MegaNeRF:
        """Create a model with only one submodule for parallel training"""
        # This is a simplified version - in practice, you might want to
        # create a specialized single-submodule model class
        single_model = MegaNeRF(self.config)
        
        # Copy the specific submodule weights
        single_model.submodules[0].load_state_dict(
            self.model.submodules[submodule_idx].state_dict()
        )
        
        return single_model
    
    def _merge_submodules(self, trainers: list[tuple[int, MegaNeRFTrainer]]):
        """Merge trained submodules back to the main model"""
        for submodule_idx, trainer in trainers:
            # Copy trained weights back to main model
            self.model.submodules[submodule_idx].load_state_dict(
                trainer.model.submodules[0].state_dict()
            )
        
        logger.info("Merged all trained submodules to main model")


def create_sample_camera_path(
center: np.ndarray = np.array,
):
    """
    Create a sample camera path for fly-through rendering
    
    Args:
        center: Center point of the path
        radius: Radius of the circular path
        num_frames: Number of frames in the path
        height_variation: Variation in height
        
    Returns:
        List of camera transform matrices
    """
    camera_path = []
    
    for i in range(num_frames):
        # Angle for circular motion
        angle = i * 2 * np.pi / num_frames
        
        # Camera position with height variation
        height_offset = height_variation * np.sin(angle * 3)  # Vary height
        pos = np.array([
            center[0] + radius * np.cos(
                angle,
            )
        ])
        
        # Look at center with some variation
        target = center + np.array([
            5 * np.sin(angle * 2), # Slight target movement
            5 * np.cos(angle * 2), 0
        ])
        
        # Up vector
        up = np.array([0, 0, 1])
        
        # Build camera matrix
        forward = target - pos
        forward = forward / np.linalg.norm(forward)
        right = np.cross(forward, up)
        right = right / np.linalg.norm(right)
        up = np.cross(right, forward)
        
        transform = np.eye(4)
        transform[:3, 0] = right
        transform[:3, 1] = up
        transform[:3, 2] = -forward
        transform[:3, 3] = pos
        
        camera_path.append(transform)
    
    return camera_path 