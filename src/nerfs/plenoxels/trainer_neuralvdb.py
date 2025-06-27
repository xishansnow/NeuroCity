from __future__ import annotations

"""
Extended Plenoxel Trainer with NeuralVDB Support

This module extends the basic Plenoxel trainer with NeuralVDB functionality
for efficient external storage of voxel data.
"""

from typing import Any, Optional


import os
import torch
import wandb
import logging
from dataclasses import dataclass
from tqdm import tqdm
import numpy as np
from torch.utils.tensorboard import SummaryWriter

from .trainer import PlenoxelTrainer, PlenoxelTrainerConfig
from .core import PlenoxelConfig, PlenoxelModel
from .dataset import PlenoxelDatasetConfig

try:
    from .neuralvdb_interface import (
        NeuralVDBManager, NeuralVDBConfig, save_plenoxel_as_neuralvdb, load_plenoxel_from_neuralvdb
    )
    NEURALVDB_AVAILABLE = True
except ImportError:
    NEURALVDB_AVAILABLE = False
    NeuralVDBManager = None
    NeuralVDBConfig = None

logger = logging.getLogger(__name__)

@dataclass
class NeuralVDBTrainerConfig(PlenoxelTrainerConfig):
    """Extended trainer configuration with NeuralVDB support."""
    
    # NeuralVDB settings
    save_neuralvdb: bool = True
    neuralvdb_config: Optional[NeuralVDBConfig] = None
    neuralvdb_save_interval: int = 10000
    neuralvdb_compression_level: int = 8
    neuralvdb_half_precision: bool = True
    neuralvdb_tolerance: float = 1e-5
    
    # Hierarchical LOD settings
    create_lod: bool = False
    lod_levels: int = 3
    lod_save_interval: int = 50000
    
    # Storage optimization
    optimize_storage: bool = True
    storage_stats_interval: int = 5000

class NeuralVDBPlenoxelTrainer(PlenoxelTrainer):
    """Extended Plenoxel trainer with NeuralVDB functionality."""
    
    def __init__(
        self,
        model_config: PlenoxelConfig,
        trainer_config: NeuralVDBTrainerConfig,
        dataset_config: PlenoxelDatasetConfig,
    ) -> None:
        """Initialize trainer with NeuralVDB support."""
        super().__init__(model_config, trainer_config, dataset_config)
        
        # Check if NeuralVDB is available
        if trainer_config.save_neuralvdb and not NEURALVDB_AVAILABLE:
            logger.warning("NeuralVDB not available. Disabling NeuralVDB features.")
            trainer_config.save_neuralvdb = False
        
        # Initialize NeuralVDB manager
        if trainer_config.save_neuralvdb:
            self.vdb_config = trainer_config.neuralvdb_config or NeuralVDBConfig(
                compression_level=trainer_config.neuralvdb_compression_level, half_precision=trainer_config.neuralvdb_half_precision, tolerance=trainer_config.neuralvdb_tolerance, include_metadata=True
            )
            self.vdb_manager = NeuralVDBManager(self.vdb_config)
        else:
            self.vdb_config = None
            self.vdb_manager = None
    
    def train(self):
        """Training loop with NeuralVDB checkpointing."""
        logger.info("Starting training with NeuralVDB support...")
        
        # Resume from checkpoint if specified
        if self.trainer_config.resume_from:
            if self.trainer_config.resume_from.endswith('.vdb'):
                self.load_neuralvdb_checkpoint(self.trainer_config.resume_from)
            else:
                self.load_checkpoint(self.trainer_config.resume_from)
        
        for epoch in range(self.epoch, self.trainer_config.max_epochs):
            self.epoch = epoch
            
            # Update resolution if using coarse-to-fine training
            self._update_resolution()
            
            # Training
            train_metrics = self._train_epoch()
            
            # Validation
            val_metrics = None
            if epoch % self.trainer_config.eval_interval == 0:
                val_metrics = self._validate()
                
                # Check for best model
                if val_metrics['psnr'] > self.best_psnr:
                    self.best_psnr = val_metrics['psnr']
                    self._save_checkpoint('best_model.pth')
            
            # Pruning
            if (epoch % self.trainer_config.pruning_interval == 0 and 
                epoch > 0):
                self._prune_voxels()
            
            # Regular checkpointing
            if epoch % self.trainer_config.save_interval == 0:
                self._save_checkpoint(f'checkpoint_epoch_{epoch}.pth')
            
            # NeuralVDB checkpointing
            if (self.trainer_config.save_neuralvdb and 
                epoch % self.trainer_config.neuralvdb_save_interval == 0):
                self._save_neuralvdb_checkpoint(f'checkpoint_epoch_{epoch}.vdb')
            
            # Hierarchical LOD creation
            if (self.trainer_config.create_lod and 
                epoch % self.trainer_config.lod_save_interval == 0 and 
                epoch > 0):
                self._create_hierarchical_lod(epoch)
            
            # Storage statistics
            if (self.trainer_config.save_neuralvdb and 
                epoch % self.trainer_config.storage_stats_interval == 0):
                self._log_storage_stats()
            
            # Logging
            if epoch % self.trainer_config.log_interval == 0:
                self._log_metrics(train_metrics, val_metrics)
        
        # Final save
        self._save_checkpoint('final_model.pth')
        if self.trainer_config.save_neuralvdb:
            self._save_neuralvdb_checkpoint('final_model.vdb')
        
        logger.info("Training completed!")
    
    def _save_neuralvdb_checkpoint(self, filename: str):
        """Save model state as NeuralVDB file."""
        if not self.trainer_config.save_neuralvdb:
            return
        
        try:
            # Get voxel grid from model
            voxel_grid = self.model.voxel_grid
            
            filepath = os.path.join(self.exp_dir, filename)
            success = save_plenoxel_as_neuralvdb(
                voxel_grid=voxel_grid, output_path=filepath, model_config=self.model_config, vdb_config=self.vdb_config
            )
            
            if success:
                # Get file size for logging
                file_size = os.path.getsize(filepath) / (1024 * 1024)
                logger.info(f"Saved NeuralVDB checkpoint: {filepath} ({file_size:.2f} MB)")
                
                # Optimize storage if configured
                if self.trainer_config.optimize_storage:
                    optimized_path = filepath.replace('.vdb', '_optimized.vdb')
                    self.vdb_manager.optimize_vdb_storage(filepath, optimized_path)
                    
                    # Replace original with optimized
                    os.replace(optimized_path, filepath)
                    
                    optimized_size = os.path.getsize(filepath) / (1024 * 1024)
                    compression_ratio = file_size / optimized_size if optimized_size > 0 else 1.0
                    logger.info(f"Optimized storage: {file_size:.2f} MB -> {optimized_size:.2f} MB "
                              f"(compression: {compression_ratio:.2f}x)")
                
                # Log storage statistics
                stats = self.vdb_manager.get_storage_stats(filepath)
                self._log_vdb_stats(stats)
                
            else:
                logger.warning(f"Failed to save NeuralVDB checkpoint: {filepath}")
                
        except Exception as e:
            logger.error(f"Error saving NeuralVDB checkpoint: {str(e)}")
    
    def load_neuralvdb_checkpoint(self, filepath: str):
        """Load model from NeuralVDB checkpoint."""
        try:
            logger.info(f"Loading NeuralVDB checkpoint from {filepath}")
            
            # Load voxel grid and config
            voxel_grid, loaded_model_config = load_plenoxel_from_neuralvdb(filepath, self.device)
            
            # Update model with loaded data
            self.model.voxel_grid = voxel_grid
            
            # Log loaded stats
            density_stats = {
                'min': voxel_grid.density.min(
                )
            }
            
            non_empty_voxels = (torch.exp(voxel_grid.density) > 0.01).sum().item()
            total_voxels = voxel_grid.density.numel()
            sparsity = 1.0 - (non_empty_voxels / total_voxels)
            
            logger.info(f"Loaded voxel grid:")
            logger.info(f"  Resolution: {voxel_grid.resolution}")
            logger.info(f"  Density stats: {density_stats}")
            logger.info(f"  Sparsity: {sparsity:.1%} ({non_empty_voxels}/{total_voxels} non-empty)")
            
        except Exception as e:
            logger.error(f"Failed to load NeuralVDB checkpoint: {str(e)}")
            raise
    
    def _create_hierarchical_lod(self, epoch: int):
        """Create hierarchical levels of detail."""
        if not self.trainer_config.create_lod:
            return
        
        try:
            lod_dir = os.path.join(self.exp_dir, f'lod_epoch_{epoch}')
            
            logger.info(f"Creating hierarchical LOD at epoch {epoch}")
            created_files = self.vdb_manager.create_hierarchical_lod(
                voxel_grid=self.model.voxel_grid, output_dir=lod_dir, levels=self.trainer_config.lod_levels
            )
            
            # Log LOD statistics
            total_size = 0
            for i, filepath in enumerate(created_files):
                file_size = os.path.getsize(filepath) / (1024 * 1024)
                total_size += file_size
                logger.info(f"  LOD level {i}: {file_size:.2f} MB")
                
                # Log to tensorboard
                if self.tb_writer:
                    self.tb_writer.add_scalar(f'lod/level_{i}_size_mb', file_size, epoch)
            
            logger.info(f"Total LOD size: {total_size:.2f} MB ({len(created_files)} levels)")
            
            if self.tb_writer:
                self.tb_writer.add_scalar('lod/total_size_mb', total_size, epoch)
                self.tb_writer.add_scalar('lod/num_levels', len(created_files), epoch)
                
        except Exception as e:
            logger.error(f"Failed to create hierarchical LOD: {str(e)}")
    
    def _log_storage_stats(self):
        """Log storage statistics for monitoring."""
        try:
            # Find latest VDB checkpoint
            vdb_files = [f for f in os.listdir(self.exp_dir) if f.endswith('.vdb')]
            if not vdb_files:
                return
            
            # Get stats for most recent checkpoint
            latest_vdb = max(
                vdb_files,
                key=lambda f: os.path.getmtime,
            )
            filepath = os.path.join(self.exp_dir, latest_vdb)
            
            stats = self.vdb_manager.get_storage_stats(filepath)
            self._log_vdb_stats(stats)
            
        except Exception as e:
            logger.error(f"Failed to log storage stats: {str(e)}")
    
    def _log_vdb_stats(self, stats: dict[str, Any]):
        """Log VDB storage statistics."""
        if not stats:
            return
        
        # TensorBoard logging
        if self.tb_writer:
            self.tb_writer.add_scalar(
                'storage/vdb_file_size_mb',
                stats.get,
            )
            self.tb_writer.add_scalar('storage/num_grids', stats.get('num_grids', 0), self.epoch)
            
            # Log per-grid statistics
            total_active_voxels = 0
            total_memory_mb = 0
            
            for grid_name, grid_stats in stats.get('grids', {}).items():
                active_voxels = grid_stats.get('active_voxel_count', 0)
                memory_mb = grid_stats.get('memory_usage_mb', 0)
                
                total_active_voxels += active_voxels
                total_memory_mb += memory_mb
                
                self.tb_writer.add_scalar(
                    f'storage/grid_{grid_name}_active_voxels',
                    active_voxels,
                    self.epoch,
                )
                self.tb_writer.add_scalar(
                    f'storage/grid_{grid_name}_memory_mb',
                    memory_mb,
                    self.epoch,
                )
            
            self.tb_writer.add_scalar(
                'storage/total_active_voxels',
                total_active_voxels,
                self.epoch,
            )
            self.tb_writer.add_scalar('storage/total_memory_mb', total_memory_mb, self.epoch)
        
        # W&B logging
        if self.trainer_config.use_wandb:
            log_dict = {
                'storage/vdb_file_size_mb': stats.get('file_size_mb', 0),
                'storage/total_memory_mb': sum(grid_stats.get('memory_usage_mb', 0) for grid_stats in stats.get('grids', {}).values()),
            }
            wandb.log(log_dict, step=self.epoch)
    
    def export_final_vdb(self, output_path: str, create_lod: bool = True) -> dict[str, Any]:
        """Export final trained model as optimized VDB with optional LOD."""
        logger.info(f"Exporting final VDB to {output_path}")
        
        # Create high-quality VDB config for final export
        final_vdb_config = NeuralVDBConfig(
            compression_level=9, # Maximum compression
            half_precision=True, tolerance=1e-6, # Higher precision
            include_metadata=True, include_training_info=True
        )
        
        # Save main model
        success = save_plenoxel_as_neuralvdb(
            voxel_grid=self.model.voxel_grid, output_path=output_path, model_config=self.model_config, vdb_config=final_vdb_config
        )
        
        if not success:
            raise RuntimeError(f"Failed to export VDB to {output_path}")
        
        # Optimize storage
        optimized_path = output_path.replace('.vdb', '_optimized.vdb')
        manager = NeuralVDBManager(final_vdb_config)
        manager.optimize_vdb_storage(output_path, optimized_path)
        os.replace(optimized_path, output_path)
        
        # Get final stats
        stats = manager.get_storage_stats(output_path)
        
        # Create LOD if requested
        lod_files = []
        if create_lod:
            lod_dir = output_path.replace('.vdb', '_lod')
            lod_files = manager.create_hierarchical_lod(
                voxel_grid=self.model.voxel_grid, output_dir=lod_dir, levels=4
            )
        
        export_info = {
            'main_file': output_path, 'file_size_mb': stats.get('file_size_mb', 0),
            'total_active_voxels': sum(grid_stats.get('active_voxel_count', 0) for grid_stats in stats.get('grids', {}).values()),
        }
        
        logger.info(f"Export complete:")
        logger.info(f"  Main file: {output_path} ({export_info['file_size_mb']:.2f} MB)")
        logger.info(f"  Active voxels: {export_info['total_active_voxels']:,d}")
        if lod_files:
            logger.info(f"  LOD files: {len(lod_files)} levels")
        
        return export_info

def create_neuralvdb_trainer(
    model_config: PlenoxelConfig,
    trainer_config: NeuralVDBTrainerConfig,
    dataset_config: PlenoxelDatasetConfig,
) -> NeuralVDBPlenoxelTrainer:
    """Create a NeuralVDB-enabled Plenoxel trainer."""
    return NeuralVDBPlenoxelTrainer(model_config, trainer_config, dataset_config) 