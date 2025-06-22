"""
Block-NeRF Trainer

This module implements the training pipeline for Block-NeRF, handling
multiple blocks, appearance embeddings, pose refinement, and visibility prediction.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
import time
import os
from pathlib import Path
import wandb
from tqdm import tqdm

from .block_manager import BlockManager
from .block_compositor import BlockCompositor
from .visibility_network import VisibilityNetwork
from .dataset import BlockNeRFDataset


class PoseRefinement(nn.Module):
    """
    Learned pose refinement for Block-NeRF
    """
    
    def __init__(self, num_images: int, translation_scale: float = 0.1, rotation_scale: float = 0.01):
        super().__init__()
        
        self.num_images = num_images
        self.translation_scale = translation_scale
        self.rotation_scale = rotation_scale
        
        # Learnable pose offsets
        self.translation_offsets = nn.Parameter(torch.zeros(num_images, 3))
        self.rotation_offsets = nn.Parameter(torch.zeros(num_images, 3))  # Axis-angle representation
        
        # Initialize with small random values
        nn.init.normal_(self.translation_offsets, 0, 0.01)
        nn.init.normal_(self.rotation_offsets, 0, 0.001)
    
    def forward(self, image_ids: torch.Tensor, poses: torch.Tensor) -> torch.Tensor:
        """
        Apply pose refinement to input poses
        
        Args:
            image_ids: Image indices (N,)
            poses: Original camera poses (N, 4, 4)
            
        Returns:
            Refined poses (N, 4, 4)
        """
        batch_size = image_ids.shape[0]
        device = poses.device
        
        # Get offsets for this batch
        trans_offsets = self.translation_offsets[image_ids] * self.translation_scale
        rot_offsets = self.rotation_offsets[image_ids] * self.rotation_scale
        
        # Convert axis-angle to rotation matrices
        angle = torch.norm(rot_offsets, dim=1, keepdim=True)
        axis = rot_offsets / (angle + 1e-8)
        
        # Rodrigues' formula
        cos_angle = torch.cos(angle)
        sin_angle = torch.sin(angle)
        
        # Cross product matrix
        K = torch.zeros(batch_size, 3, 3, device=device)
        K[:, 0, 1] = -axis[:, 2]
        K[:, 0, 2] = axis[:, 1]
        K[:, 1, 0] = axis[:, 2]
        K[:, 1, 2] = -axis[:, 0]
        K[:, 2, 0] = -axis[:, 1]
        K[:, 2, 1] = axis[:, 0]
        
        # Rotation matrix
        I = torch.eye(3, device=device).expand(batch_size, 3, 3)
        R_offset = I + sin_angle.unsqueeze(-1) * K + (1 - cos_angle).unsqueeze(-1) * torch.bmm(K, K)
        
        # Apply refinement
        refined_poses = poses.clone()
        
        # Apply rotation offset
        refined_poses[:, :3, :3] = torch.bmm(R_offset, poses[:, :3, :3])
        
        # Apply translation offset
        refined_poses[:, :3, 3] += trans_offsets
        
        return refined_poses


class BlockNeRFTrainer:
    """
    Trainer for Block-NeRF with multiple blocks and components
    """
    
    def __init__(self,
                 block_manager: BlockManager,
                 network_config: Dict,
                 training_config: Dict,
                 device: str = 'cuda'):
        """
        Initialize Block-NeRF trainer
        
        Args:
            block_manager: Block manager instance
            network_config: Network configuration
            training_config: Training configuration
            device: Device for computation
        """
        self.block_manager = block_manager
        self.network_config = network_config
        self.training_config = training_config
        self.device = device
        
        # Initialize compositor
        self.compositor = BlockCompositor(
            interpolation_method=training_config.get('interpolation_method', 'inverse_distance'),
            power=training_config.get('interpolation_power', 2.0)
        )
        
        # Pose refinement (will be initialized with dataset)
        self.pose_refinement: Optional[PoseRefinement] = None
        
        # Training state
        self.current_epoch = 0
        self.global_step = 0
        self.best_psnr = 0.0
        
        # Optimizers (will be set up during training)
        self.optimizers: Dict[str, optim.Optimizer] = {}
        self.schedulers: Dict[str, Any] = {}
        
        # Loss tracking
        self.loss_history = []
        
    def setup_pose_refinement(self, num_images: int):
        """Setup pose refinement module"""
        self.pose_refinement = PoseRefinement(
            num_images=num_images,
            translation_scale=self.training_config.get('pose_translation_scale', 0.1),
            rotation_scale=self.training_config.get('pose_rotation_scale', 0.01)
        ).to(self.device)
    
    def setup_optimizers(self, blocks: List[str]):
        """Setup optimizers for all components"""
        lr = self.training_config.get('learning_rate', 5e-4)
        weight_decay = self.training_config.get('weight_decay', 0.0)
        
        # Block optimizers
        for block_name in blocks:
            if block_name in self.block_manager.blocks:
                block = self.block_manager.blocks[block_name]
                self.optimizers[f'block_{block_name}'] = optim.Adam(
                    block.parameters(), lr=lr, weight_decay=weight_decay
                )
        
        # Visibility network optimizer
        self.optimizers['visibility'] = optim.Adam(
            self.block_manager.visibility_network.parameters(), 
            lr=lr * 0.1, weight_decay=weight_decay
        )
        
        # Pose refinement optimizer
        if self.pose_refinement is not None:
            self.optimizers['pose'] = optim.Adam(
                self.pose_refinement.parameters(),
                lr=lr * 0.01, weight_decay=weight_decay
            )
        
        # Setup schedulers
        scheduler_type = self.training_config.get('scheduler', 'exponential')
        if scheduler_type == 'exponential':
            gamma = self.training_config.get('scheduler_gamma', 0.99)
            for name, optimizer in self.optimizers.items():
                self.schedulers[name] = optim.lr_scheduler.ExponentialLR(optimizer, gamma=gamma)
    
    def compute_losses(self,
                      predicted: Dict[str, torch.Tensor],
                      target: Dict[str, torch.Tensor],
                      block_outputs: List[Dict[str, torch.Tensor]],
                      visibility_outputs: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """
        Compute all training losses
        
        Args:
            predicted: Predicted outputs
            target: Target ground truth
            block_outputs: Individual block outputs
            visibility_outputs: Visibility predictions
            
        Returns:
            Dictionary of losses
        """
        losses = {}
        
        # Main reconstruction loss
        mse_loss = F.mse_loss(predicted['rgb'], target['rgb'])
        losses['mse'] = mse_loss
        losses['psnr'] = -10 * torch.log10(mse_loss)
        
        # Visibility loss
        if visibility_outputs is not None and 'transmittance' in target:
            visibility_loss = F.mse_loss(visibility_outputs, target['transmittance'])
            losses['visibility'] = visibility_loss
        
        # Regularization losses
        if self.pose_refinement is not None:
            # Pose regularization
            trans_reg = torch.mean(self.pose_refinement.translation_offsets ** 2)
            rot_reg = torch.mean(self.pose_refinement.rotation_offsets ** 2)
            losses['pose_reg'] = trans_reg + rot_reg
        
        # Appearance regularization
        appearance_reg = 0.0
        for block_name in self.block_manager.blocks:
            block = self.block_manager.blocks[block_name]
            appearance_weights = block.appearance_embeddings.weight
            appearance_reg += torch.mean(appearance_weights ** 2)
        losses['appearance_reg'] = appearance_reg / len(self.block_manager.blocks)
        
        # Total loss
        total_loss = losses['mse']
        
        # Add weighted auxiliary losses
        if 'visibility' in losses:
            total_loss += self.training_config.get('visibility_weight', 0.1) * losses['visibility']
        
        if 'pose_reg' in losses:
            total_loss += self.training_config.get('pose_reg_weight', 0.01) * losses['pose_reg']
        
        if 'appearance_reg' in losses:
            total_loss += self.training_config.get('appearance_reg_weight', 0.001) * losses['appearance_reg']
        
        losses['total'] = total_loss
        
        return losses
    
    def train_step(self,
                  batch: Dict[str, torch.Tensor],
                  active_blocks: List[str]) -> Dict[str, float]:
        """
        Single training step
        
        Args:
            batch: Training batch
            active_blocks: List of active block names
            
        Returns:
            Dictionary of loss values
        """
        # Extract batch data
        ray_origins = batch['ray_origins'].to(self.device)
        ray_directions = batch['ray_directions'].to(self.device)
        target_rgb = batch['rgb'].to(self.device)
        camera_positions = batch['camera_positions'].to(self.device)
        image_ids = batch['image_ids'].to(self.device)
        appearance_ids = batch['appearance_ids'].to(self.device)
        exposure_values = batch['exposure_values'].to(self.device)
        
        # Apply pose refinement if available
        if self.pose_refinement is not None and 'poses' in batch:
            poses = batch['poses'].to(self.device)
            refined_poses = self.pose_refinement(image_ids, poses)
            # Update ray origins and directions based on refined poses
            # (Implementation depends on how poses are represented)
        
        # Get relevant blocks for this batch
        batch_blocks = []
        batch_block_names = []
        batch_block_centers = []
        
        for block_name in active_blocks:
            if block_name in self.block_manager.blocks:
                batch_blocks.append(self.block_manager.blocks[block_name])
                batch_block_names.append(block_name)
                batch_block_centers.append(self.block_manager.block_centers[block_name])
        
        if not batch_blocks:
            raise ValueError("No active blocks found for training")
        
        # Forward pass through blocks
        block_outputs = self.compositor.render_blocks(
            blocks=batch_blocks,
            block_names=batch_block_names,
            ray_origins=ray_origins,
            ray_directions=ray_directions,
            appearance_ids=appearance_ids,
            exposure_values=exposure_values,
            **self.training_config.get('render_kwargs', {})
        )
        
        # Compute interpolation weights
        interpolation_weights = self.compositor.compute_interpolation_weights(
            camera_positions[0], batch_block_centers  # Assume single camera per batch
        )
        
        # Composite blocks
        predicted = self.compositor.composite_blocks(
            block_outputs, interpolation_weights, camera_positions[0]
        )
        
        # Visibility prediction (optional)
        visibility_outputs = None
        if self.training_config.get('train_visibility', True):
            # Sample points for visibility training
            sample_points = ray_origins + ray_directions * torch.rand_like(ray_origins[:, :1]) * 10.0
            visibility_outputs = self.block_manager.visibility_network(sample_points, ray_directions)
        
        # Compute losses
        target = {'rgb': target_rgb}
        if 'transmittance' in batch:
            target['transmittance'] = batch['transmittance'].to(self.device)
        
        losses = self.compute_losses(predicted, target, block_outputs, visibility_outputs)
        
        # Backward pass
        for optimizer in self.optimizers.values():
            optimizer.zero_grad()
        
        losses['total'].backward()
        
        # Gradient clipping
        if self.training_config.get('grad_clip', 0) > 0:
            for block_name in batch_block_names:
                if f'block_{block_name}' in self.optimizers:
                    nn.utils.clip_grad_norm_(
                        self.block_manager.blocks[block_name].parameters(),
                        self.training_config['grad_clip']
                    )
        
        # Optimizer step
        for optimizer in self.optimizers.values():
            optimizer.step()
        
        # Convert losses to float for logging
        loss_values = {k: v.item() if torch.is_tensor(v) else v for k, v in losses.items()}
        
        return loss_values
    
    def validate(self, val_dataloader: DataLoader) -> Dict[str, float]:
        """
        Validation loop
        
        Args:
            val_dataloader: Validation data loader
            
        Returns:
            Validation metrics
        """
        self.set_eval_mode()
        
        total_loss = 0.0
        total_psnr = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for batch in val_dataloader:
                # Get active blocks for this validation sample
                camera_pos = batch['camera_positions'][0].to(self.device)
                active_blocks = self.block_manager.get_blocks_for_camera(
                    camera_pos, torch.zeros(3, device=self.device)
                )
                
                if not active_blocks:
                    continue
                
                # Forward pass (similar to train_step but without backward)
                ray_origins = batch['ray_origins'].to(self.device)
                ray_directions = batch['ray_directions'].to(self.device)
                target_rgb = batch['rgb'].to(self.device)
                appearance_ids = batch['appearance_ids'].to(self.device)
                exposure_values = batch['exposure_values'].to(self.device)
                
                # Render
                predicted = self.render_batch(
                    ray_origins, ray_directions, camera_pos,
                    appearance_ids, exposure_values, active_blocks
                )
                
                # Compute metrics
                mse = F.mse_loss(predicted['rgb'], target_rgb)
                psnr = -10 * torch.log10(mse)
                
                total_loss += mse.item()
                total_psnr += psnr.item()
                num_batches += 1
        
        self.set_train_mode()
        
        return {
            'val_loss': total_loss / max(num_batches, 1),
            'val_psnr': total_psnr / max(num_batches, 1)
        }
    
    def render_batch(self,
                    ray_origins: torch.Tensor,
                    ray_directions: torch.Tensor,
                    camera_position: torch.Tensor,
                    appearance_ids: torch.Tensor,
                    exposure_values: torch.Tensor,
                    active_blocks: List[str]) -> Dict[str, torch.Tensor]:
        """Render a batch of rays"""
        batch_blocks = [self.block_manager.blocks[name] for name in active_blocks if name in self.block_manager.blocks]
        batch_block_centers = [self.block_manager.block_centers[name] for name in active_blocks if name in self.block_manager.blocks]
        
        return self.compositor.render_with_blocks(
            blocks=batch_blocks,
            block_names=active_blocks,
            block_centers=batch_block_centers,
            camera_position=camera_position,
            ray_origins=ray_origins,
            ray_directions=ray_directions,
            appearance_ids=appearance_ids,
            exposure_values=exposure_values,
            **self.training_config.get('render_kwargs', {})
        )
    
    def set_train_mode(self):
        """Set all components to training mode"""
        for block in self.block_manager.blocks.values():
            block.train()
        self.block_manager.visibility_network.train()
        if self.pose_refinement is not None:
            self.pose_refinement.train()
        self.compositor.training = True
    
    def set_eval_mode(self):
        """Set all components to evaluation mode"""
        for block in self.block_manager.blocks.values():
            block.eval()
        self.block_manager.visibility_network.eval()
        if self.pose_refinement is not None:
            self.pose_refinement.eval()
        self.compositor.training = False
    
    def save_checkpoint(self, save_path: str, epoch: int, is_best: bool = False):
        """Save training checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'global_step': self.global_step,
            'best_psnr': self.best_psnr,
            'training_config': self.training_config,
            'network_config': self.network_config,
            'optimizer_states': {name: opt.state_dict() for name, opt in self.optimizers.items()},
            'scheduler_states': {name: sch.state_dict() for name, sch in self.schedulers.items()},
            'loss_history': self.loss_history
        }
        
        if self.pose_refinement is not None:
            checkpoint['pose_refinement'] = self.pose_refinement.state_dict()
        
        # Save checkpoint
        torch.save(checkpoint, save_path)
        
        # Save best model
        if is_best:
            best_path = save_path.replace('.pth', '_best.pth')
            torch.save(checkpoint, best_path)
    
    def load_checkpoint(self, load_path: str):
        """Load training checkpoint"""
        checkpoint = torch.load(load_path, map_location=self.device)
        
        self.current_epoch = checkpoint['epoch']
        self.global_step = checkpoint['global_step']
        self.best_psnr = checkpoint['best_psnr']
        self.loss_history = checkpoint.get('loss_history', [])
        
        # Load optimizer states
        for name, state_dict in checkpoint['optimizer_states'].items():
            if name in self.optimizers:
                self.optimizers[name].load_state_dict(state_dict)
        
        # Load scheduler states
        for name, state_dict in checkpoint['scheduler_states'].items():
            if name in self.schedulers:
                self.schedulers[name].load_state_dict(state_dict)
        
        # Load pose refinement
        if 'pose_refinement' in checkpoint and self.pose_refinement is not None:
            self.pose_refinement.load_state_dict(checkpoint['pose_refinement'])
    
    def train(self,
             train_dataloader: DataLoader,
             val_dataloader: Optional[DataLoader] = None,
             num_epochs: int = 100,
             save_dir: str = './checkpoints',
             log_interval: int = 100,
             val_interval: int = 1000,
             save_interval: int = 5000):
        """
        Main training loop
        
        Args:
            train_dataloader: Training data loader
            val_dataloader: Validation data loader
            num_epochs: Number of training epochs
            save_dir: Directory to save checkpoints
            log_interval: Logging interval (steps)
            val_interval: Validation interval (steps)
            save_interval: Checkpoint saving interval (steps)
        """
        os.makedirs(save_dir, exist_ok=True)
        
        # Setup pose refinement
        if hasattr(train_dataloader.dataset, 'num_images'):
            self.setup_pose_refinement(train_dataloader.dataset.num_images)
        
        # Get all possible active blocks
        all_blocks = list(self.block_manager.block_centers.keys())
        
        # Setup optimizers
        self.setup_optimizers(all_blocks)
        
        # Training loop
        self.set_train_mode()
        
        for epoch in range(self.current_epoch, num_epochs):
            epoch_losses = []
            
            pbar = tqdm(train_dataloader, desc=f'Epoch {epoch+1}/{num_epochs}')
            
            for batch_idx, batch in enumerate(pbar):
                # Determine active blocks for this batch
                camera_pos = batch['camera_positions'][0].to(self.device)
                active_blocks = self.block_manager.get_training_blocks_for_image(camera_pos)
                
                # Create blocks if they don't exist
                for block_name in active_blocks:
                    if block_name not in self.block_manager.blocks:
                        self.block_manager.create_block(block_name, self.network_config)
                
                # Filter to existing blocks
                active_blocks = [name for name in active_blocks if name in self.block_manager.blocks]
                
                if not active_blocks:
                    continue
                
                # Training step
                step_losses = self.train_step(batch, active_blocks)
                epoch_losses.append(step_losses)
                
                # Update progress bar
                pbar.set_postfix({
                    'loss': f"{step_losses['total']:.4f}",
                    'psnr': f"{step_losses['psnr']:.2f}",
                    'blocks': len(active_blocks)
                })
                
                # Logging
                if self.global_step % log_interval == 0:
                    # Log to wandb if available
                    if wandb.run is not None:
                        wandb.log(step_losses, step=self.global_step)
                
                # Validation
                if val_dataloader is not None and self.global_step % val_interval == 0:
                    val_metrics = self.validate(val_dataloader)
                    
                    if wandb.run is not None:
                        wandb.log(val_metrics, step=self.global_step)
                    
                    # Check if best model
                    if val_metrics['val_psnr'] > self.best_psnr:
                        self.best_psnr = val_metrics['val_psnr']
                        is_best = True
                    else:
                        is_best = False
                    
                    # Save checkpoint
                    if self.global_step % save_interval == 0:
                        checkpoint_path = os.path.join(save_dir, f'checkpoint_{self.global_step}.pth')
                        self.save_checkpoint(checkpoint_path, epoch, is_best)
                
                # Update schedulers
                for scheduler in self.schedulers.values():
                    scheduler.step()
                
                self.global_step += 1
            
            # End of epoch
            self.current_epoch = epoch + 1
            
            # Compute epoch statistics
            if epoch_losses:
                avg_losses = {}
                for key in epoch_losses[0].keys():
                    avg_losses[f'epoch_{key}'] = np.mean([loss[key] for loss in epoch_losses])
                
                print(f"Epoch {epoch+1} - " + " - ".join([f"{k}: {v:.4f}" for k, v in avg_losses.items()]))
                
                if wandb.run is not None:
                    wandb.log(avg_losses, step=self.global_step)
        
        # Save final checkpoint
        final_checkpoint_path = os.path.join(save_dir, 'final_checkpoint.pth')
        self.save_checkpoint(final_checkpoint_path, num_epochs)
        
        # Save all blocks
        blocks_save_dir = os.path.join(save_dir, 'blocks')
        self.block_manager.save_blocks(blocks_save_dir)
        
        print("Training completed!") 