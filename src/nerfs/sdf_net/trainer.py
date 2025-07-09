from __future__ import annotations

from typing import Any, Optional, Union
"""
SDF Network Trainer Implementation
SDF网络训练器
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import os
import time
import json

from .core import SDFNetwork, LatentSDFNetwork, MultiScaleSDFNetwork
from .dataset import SDFDataset, create_sdf_dataloader

class SDFTrainer:
    """SDF网络训练器
    
    Args:
        model: SDF网络模型
        train_dataloader: 训练数据加载器
        val_dataloader: 验证数据加载器 (可选)
        optimizer: 优化器 (可选)
        scheduler: 学习率调度器 (可选)
        device: 设备
        log_dir: 日志目录
        checkpoint_dir: 检查点目录
        lambda_gp: Eikonal约束权重
    """
    
    def __init__(
        self, model: SDFNetwork | LatentSDFNetwork | MultiScaleSDFNetwork, train_dataloader: DataLoader, val_dataloader: Optional[DataLoader] = None, optimizer: Optional[optim.Optimizer] = None, scheduler: Optional = None, device: str = 'cuda', log_dir: str = 'logs/sdf_net', checkpoint_dir: str = 'checkpoints/sdf_net', lambda_gp: float = 0.1, loss_type: str = 'l1'
    ):
        self.model = model
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.device = device
        self.log_dir = log_dir
        self.checkpoint_dir = checkpoint_dir
        self.lambda_gp = lambda_gp
        self.loss_type = loss_type
        
        # 设置设备
        self.model = self.model.to(device)
        
        # 设置优化器
        if optimizer is None:
            self.optimizer = optim.Adam(
                self.model.parameters(), lr=1e-4, weight_decay=0
            )
        else:
            self.optimizer = optimizer
        
        # 设置学习率调度器
        self.scheduler = scheduler
        
        # 创建目录
        os.makedirs(log_dir, exist_ok=True)
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        # 设置日志记录器
        self.writer = SummaryWriter(log_dir)
        
        # 训练状态
        self.epoch = 0
        self.global_step = 0
        self.best_val_loss = float('inf')
        self.train_losses = []
        self.val_losses = []
        
        print(f"SDF Trainer initialized with device: {device}")
        print(f"Model parameters: {sum(p.numel() for p in model.parameters()):, }")
        print(f"Eikonal constraint weight: {lambda_gp}")
    
    def train_epoch(self) -> dict[str, float]:
        """训练一个epoch"""
        self.model.train()
        
        epoch_sdf_loss = 0.0
        epoch_eikonal_loss = 0.0
        epoch_total_loss = 0.0
        num_batches = 0
        
        for batch_idx, batch in enumerate(self.train_dataloader):
            # 移动数据到设备
            points = batch['points'].to(self.device)  # [B, N, 3]
            sdf = batch['sdf'].to(self.device)  # [B, N, 1]
            
            # 获取潜在编码
            if 'latent_codes' in batch:
                latent_codes = batch['latent_codes'].to(self.device)
            else:
                # 如果没有预定义的潜在编码，使用随机编码
                batch_size = points.shape[0]
                if isinstance(self.model, SDFNetwork):
                    latent_codes = torch.randn(
                        batch_size,
                        self.model.dim_latent,
                    )
                else:
                    # 对于LatentSDFNetwork，使用shape_ids
                    shape_ids = torch.arange(batch_size).to(self.device)
                    latent_codes = None
            
            # 前向传播
            self.optimizer.zero_grad()
            
            if isinstance(self.model, LatentSDFNetwork) and latent_codes is None:
                pred_sdf = self.model(points, shape_ids=shape_ids)
            elif hasattr(self.model, 'sdf_decoder'):
                pred_sdf = self.model.sdf_decoder(points, latent_codes)
            else:
                pred_sdf = self.model(points, latent_codes)
            
            # 计算SDF损失
            if hasattr(self.model, 'compute_sdf_loss'):
                sdf_loss = self.model.compute_sdf_loss(pred_sdf, sdf, self.loss_type)
            elif hasattr(self.model, 'sdf_decoder'):
                sdf_loss = self.model.sdf_decoder.compute_sdf_loss(pred_sdf, sdf, self.loss_type)
            else:
                # 后备损失计算
                if self.loss_type == 'l1':
                    sdf_loss = torch.nn.functional.l1_loss(pred_sdf, sdf)
                else:
                    sdf_loss = torch.nn.functional.mse_loss(pred_sdf, sdf)
            
            # 计算Eikonal约束
            eikonal_loss = 0.0
            if self.lambda_gp > 0:
                try:
                    if isinstance(self.model, LatentSDFNetwork):
                        if latent_codes is not None:
                            eikonal_loss = self.model.sdf_decoder.compute_gradient_penalty(
                                points, latent_codes, self.lambda_gp
                            )
                        else:
                            # 对于使用shape_ids的情况
                            sample_latent = self.model.latent_codes(shape_ids)
                            eikonal_loss = self.model.sdf_decoder.compute_gradient_penalty(
                                points, sample_latent, self.lambda_gp
                            )
                    elif hasattr(self.model, 'compute_gradient_penalty'):
                        eikonal_loss = self.model.compute_gradient_penalty(
                            points, latent_codes, self.lambda_gp
                        )
                except Exception as e:
                    print(f"Warning: Eikonal loss computation failed: {e}")
                    eikonal_loss = 0.0
            
            # 总损失
            total_loss = sdf_loss + eikonal_loss
            
            # 反向传播
            total_loss.backward()
            
            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            self.optimizer.step()
            
            # 更新统计信息
            epoch_sdf_loss += sdf_loss.item()
            epoch_eikonal_loss += eikonal_loss if isinstance(eikonal_loss, float) else eikonal_loss.item()
            epoch_total_loss += total_loss.item()
            num_batches += 1
            
            # 记录日志
            if batch_idx % 100 == 0:
                self.writer.add_scalar('train/batch_sdf_loss', sdf_loss.item(), self.global_step)
                if eikonal_loss != 0.0:
                    eikonal_val = eikonal_loss if isinstance(eikonal_loss, float) else eikonal_loss.item()
                    self.writer.add_scalar(
                        'train/batch_eikonal_loss',
                        eikonal_val,
                        self.global_step,
                    )
                self.writer.add_scalar(
                    'train/batch_total_loss',
                    total_loss.item,
                )
                
                print(f'Epoch {self.epoch}, Batch {batch_idx}/{len(self.train_dataloader)}, '
                      f'SDF Loss: {sdf_loss.item():.6f}, '
                      f'Eikonal: {eikonal_val if eikonal_loss != 0.0 else 0:.6f}, '
                      f'Total: {total_loss.item():.6f}')
            
            self.global_step += 1
        
        # 计算平均指标
        avg_sdf_loss = epoch_sdf_loss / num_batches
        avg_eikonal_loss = epoch_eikonal_loss / num_batches
        avg_total_loss = epoch_total_loss / num_batches
        
        return {
            'sdf_loss': avg_sdf_loss, 'eikonal_loss': avg_eikonal_loss, 'total_loss': avg_total_loss
        }
    
    def validate(self) -> dict[str, float]:
        """验证模型"""
        if self.val_dataloader is None:
            return {}
        
        self.model.eval()
        
        val_sdf_loss = 0.0
        val_eikonal_loss = 0.0
        val_total_loss = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for batch in self.val_dataloader:
                # 移动数据到设备
                points = batch['points'].to(self.device)
                sdf = batch['sdf'].to(self.device)
                
                # 获取潜在编码
                if 'latent_codes' in batch:
                    latent_codes = batch['latent_codes'].to(self.device)
                else:
                    batch_size = points.shape[0]
                    if isinstance(self.model, SDFNetwork):
                        latent_codes = torch.randn(
                            batch_size,
                            self.model.dim_latent,
                        )
                    else:
                        shape_ids = torch.arange(batch_size).to(self.device)
                        latent_codes = None
                
                # 前向传播
                if isinstance(self.model, LatentSDFNetwork) and latent_codes is None:
                    pred_sdf = self.model(points, shape_ids=shape_ids)
                elif hasattr(self.model, 'sdf_decoder'):
                    pred_sdf = self.model.sdf_decoder(points, latent_codes)
                else:
                    pred_sdf = self.model(points, latent_codes)
                
                # 计算损失
                if hasattr(self.model, 'compute_sdf_loss'):
                    sdf_loss = self.model.compute_sdf_loss(pred_sdf, sdf, self.loss_type)
                elif hasattr(self.model, 'sdf_decoder'):
                    sdf_loss = self.model.sdf_decoder.compute_sdf_loss(
                        pred_sdf,
                        sdf,
                        self.loss_type,
                    )
                else:
                    if self.loss_type == 'l1':
                        sdf_loss = torch.nn.functional.l1_loss(pred_sdf, sdf)
                    else:
                        sdf_loss = torch.nn.functional.mse_loss(pred_sdf, sdf)
                
                val_sdf_loss += sdf_loss.item()
                val_total_loss += sdf_loss.item()
                num_batches += 1
        
        avg_val_sdf_loss = val_sdf_loss / num_batches
        avg_val_total_loss = val_total_loss / num_batches
        
        return {
            'val_sdf_loss': avg_val_sdf_loss, 'val_total_loss': avg_val_total_loss
        }
    
    def train(
        self, num_epochs: int, save_freq: int = 10, eval_freq: int = 5
    ):
        """训练模型
        
        Args:
            num_epochs: 训练轮数
            save_freq: 保存检查点频率
            eval_freq: 验证频率
        """
        print(f"Starting SDF training for {num_epochs} epochs...")
        
        for epoch in range(num_epochs):
            self.epoch = epoch
            start_time = time.time()
            
            # 训练一个epoch
            train_metrics = self.train_epoch()
            
            # 验证
            val_metrics = {}
            if epoch % eval_freq == 0:
                val_metrics = self.validate()
            
            # 更新学习率
            if self.scheduler is not None:
                if val_metrics and 'val_total_loss' in val_metrics:
                    if isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                        self.scheduler.step(val_metrics['val_total_loss'])
                    else:
                        self.scheduler.step()
                else:
                    self.scheduler.step()
            
            # 记录指标
            epoch_time = time.time() - start_time
            self.train_losses.append(train_metrics['total_loss'])
            
            # 记录到tensorboard
            self.writer.add_scalar('train/epoch_sdf_loss', train_metrics['sdf_loss'], epoch)
            self.writer.add_scalar('train/epoch_eikonal_loss', train_metrics['eikonal_loss'], epoch)
            self.writer.add_scalar('train/epoch_total_loss', train_metrics['total_loss'], epoch)
            
            if val_metrics:
                self.val_losses.append(val_metrics['val_total_loss'])
                self.writer.add_scalar('val/sdf_loss', val_metrics['val_sdf_loss'], epoch)
                self.writer.add_scalar('val/total_loss', val_metrics['val_total_loss'], epoch)
            
            self.writer.add_scalar(
                'train/learning_rate',
                self.optimizer.param_groups[0]['lr'],
                epoch,
            )
            
            # 打印进度
            print(f'Epoch {epoch}/{num_epochs} - {epoch_time:.2f}s')
            print(f'Train - SDF: {train_metrics["sdf_loss"]:.6f}, '
                  f'Eikonal: {train_metrics["eikonal_loss"]:.6f}, '
                  f'Total: {train_metrics["total_loss"]:.6f}')
            if val_metrics:
                print(f'Val - SDF: {val_metrics["val_sdf_loss"]:.6f}, '
                      f'Total: {val_metrics["val_total_loss"]:.6f}')
            print('-' * 60)
            
            # 保存最佳模型
            if val_metrics and val_metrics['val_total_loss'] < self.best_val_loss:
                self.best_val_loss = val_metrics['val_total_loss']
                self.save_checkpoint(f'best_model.pth')
                print(f'New best model saved with val_loss: {self.best_val_loss:.6f}')
            
            # 定期保存检查点
            if epoch % save_freq == 0:
                self.save_checkpoint(f'epoch_{epoch}.pth')
        
        print("SDF training completed!")
        self.writer.close()
    
    def save_checkpoint(self, filename: str):
        """保存检查点"""
        checkpoint = {
            'epoch': self.epoch, 'global_step': self.global_step, 'model_state_dict': self.model.state_dict(
            )
        }
        
        if self.scheduler is not None:
            checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()
        
        filepath = os.path.join(self.checkpoint_dir, filename)
        torch.save(checkpoint, filepath)
        print(f'Checkpoint saved: {filepath}')
    
    def load_checkpoint(self, filepath: str):
        """加载检查点"""
        checkpoint = torch.load(filepath, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        self.epoch = checkpoint['epoch']
        self.global_step = checkpoint['global_step']
        self.best_val_loss = checkpoint['best_val_loss']
        self.train_losses = checkpoint['train_losses']
        self.val_losses = checkpoint['val_losses']
        
        if 'lambda_gp' in checkpoint:
            self.lambda_gp = checkpoint['lambda_gp']
        if 'loss_type' in checkpoint:
            self.loss_type = checkpoint['loss_type']
        
        if self.scheduler is not None and 'scheduler_state_dict' in checkpoint:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        print(f'Checkpoint loaded: {filepath}')
        print(f'Resumed from epoch {self.epoch}')
    
    def evaluate_mesh_extraction(
        self, test_dataloader: DataLoader, resolution: int = 128, num_samples: int = 10
    ) -> dict[str, float]:
        """评估网格提取质量"""
        self.model.eval()
        
        mesh_metrics = []
        
        with torch.no_grad():
            for i, batch in enumerate(test_dataloader):
                if i >= num_samples:
                    break
                
                # 获取潜在编码
                if 'latent_codes' in batch:
                    latent_code = batch['latent_codes'][0:1].to(self.device)
                else:
                    if isinstance(self.model, LatentSDFNetwork):
                        shape_id = 0
                        try:
                            latent_code = self.model.get_latent_code(shape_id)
                        except:
                            latent_code = torch.randn(
                                1,
                                self.model.dim_latent,
                            )
                    else:
                        latent_code = torch.randn(1, self.model.dim_latent).to(self.device) * 0.01
                
                # 提取网格
                try:
                    if hasattr(self.model, 'extract_mesh'):
                        mesh_data = self.model.extract_mesh(
                            latent_code=latent_code, resolution=resolution
                        )
                    elif hasattr(self.model, 'sdf_decoder'):
                        mesh_data = self.model.sdf_decoder.extract_mesh(
                            latent_code=latent_code, resolution=resolution
                        )
                    else:
                        continue
                    
                    if 'vertices' in mesh_data and 'faces' in mesh_data:
                        num_vertices = len(mesh_data['vertices'])
                        num_faces = len(mesh_data['faces'])
                        
                        mesh_metrics.append({
                            'num_vertices': num_vertices, 'num_faces': num_faces, 'success': True
                        })
                    else:
                        mesh_metrics.append({'success': False})
                        
                except Exception as e:
                    print(f"Mesh extraction failed: {e}")
                    mesh_metrics.append({'success': False})
        
        # 计算统计信息
        successful_extractions = [m for m in mesh_metrics if m['success']]
        success_rate = len(successful_extractions) / len(mesh_metrics) if mesh_metrics else 0
        
        if successful_extractions:
            avg_vertices = np.mean([m['num_vertices'] for m in successful_extractions])
            avg_faces = np.mean([m['num_faces'] for m in successful_extractions])
        else:
            avg_vertices = 0
            avg_faces = 0
        
        return {
            'mesh_extraction_success_rate': success_rate, 'avg_vertices': avg_vertices, 'avg_faces': avg_faces
        }
    
    def get_training_summary(self) -> dict:
        """获取训练总结"""
        model_info = self.model.get_model_size() if hasattr(self.model, 'get_model_size') else {}
        
        summary = {
            'total_epochs': self.epoch, 'total_steps': self.global_step, 'best_val_loss': self.best_val_loss, 'final_train_loss': self.train_losses[-1] if self.train_losses else None, 'final_val_loss': self.val_losses[-1] if self.val_losses else None, 'lambda_gp': self.lambda_gp, 'loss_type': self.loss_type, 'model_info': model_info
        }
        
        return summary
    
    def save_training_summary(self, filename: str = 'training_summary.json'):
        """保存训练总结"""
        summary = self.get_training_summary()
        filepath = os.path.join(self.log_dir, filename)
        
        with open(filepath, 'w') as f:
            json.dump(summary, f, indent=2, default=str)
        
        print(f'Training summary saved: {filepath}')

def create_sdf_trainer_from_config(config: dict) -> SDFTrainer:
    """从配置创建SDF训练器"""
    
    # 创建模型
    model_config = config.get('model', {})
    model_type = config.get('model_type', 'sdf')
    
    if model_type == 'latent_sdf':
        model = LatentSDFNetwork(**model_config)
    elif model_type == 'multiscale_sdf':
        model = MultiScaleSDFNetwork(**model_config)
    else:
        model = SDFNetwork(**model_config)
    
    # 创建数据集
    dataset_config = config.get('dataset', {})
    train_dataset = SDFDataset(split='train', **dataset_config)
    val_dataset = SDFDataset(
        split='val',
        **dataset_config,
    )
    
    # 创建数据加载器
    dataloader_config = config.get('dataloader', {})
    train_dataloader = create_sdf_dataloader(train_dataset, **dataloader_config)
    val_dataloader = create_sdf_dataloader(
        val_dataset,
        shuffle=False,
        **dataloader_config,
    )
    
    # 创建优化器
    optimizer_config = config.get('optimizer', {})
    optimizer_type = optimizer_config.pop('type', 'adam')
    
    if optimizer_type.lower() == 'adam':
        optimizer = optim.Adam(model.parameters(), **optimizer_config)
    elif optimizer_type.lower() == 'sgd':
        optimizer = optim.SGD(model.parameters(), **optimizer_config)
    else:
        optimizer = optim.Adam(model.parameters(), **optimizer_config)
    
    # 创建学习率调度器
    scheduler = None
    if 'scheduler' in config:
        scheduler_config = config['scheduler']
        scheduler_type = scheduler_config.pop('type', 'step')
        
        if scheduler_type.lower() == 'step':
            scheduler = optim.lr_scheduler.StepLR(optimizer, **scheduler_config)
        elif scheduler_type.lower() == 'cosine':
            scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, **scheduler_config)
        elif scheduler_type.lower() == 'plateau':
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, **scheduler_config)
    
    # 创建训练器
    trainer_config = config.get('trainer', {})
    trainer = SDFTrainer(
        model=model, train_dataloader=train_dataloader, val_dataloader=val_dataloader, optimizer=optimizer, scheduler=scheduler, **trainer_config
    )
    
    return trainer 