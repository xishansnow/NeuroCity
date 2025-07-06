from __future__ import annotations

from typing import Any, Optional, Union
"""
Occupancy Network Trainer Implementation
占用网络训练器
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

from .core import OccupancyNetwork, ConditionalOccupancyNetwork
from .dataset import OccupancyDataset, create_occupancy_dataloader

class OccupancyTrainer:
    """占用网络训练器
    
    Args:
        model: 占用网络模型
        train_dataloader: 训练数据加载器
        val_dataloader: 验证数据加载器 (可选)
        optimizer: 优化器 (可选)
        scheduler: 学习率调度器 (可选)
        device: 设备
        log_dir: 日志目录
        checkpoint_dir: 检查点目录
    """
    
    def __init__(
        self, model: OccupancyNetwork | ConditionalOccupancyNetwork, train_dataloader: DataLoader, val_dataloader: Optional[DataLoader] = None, optimizer: Optional[optim.Optimizer] = None, scheduler: Optional = None, device: str = 'cuda', log_dir: str = 'logs/occupancy_net', checkpoint_dir: str = 'checkpoints/occupancy_net'
    ):
        self.model = model
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.device = device
        self.log_dir = log_dir
        self.checkpoint_dir = checkpoint_dir
        
        # 设置设备
        self.model = self.model.to(device)
        
        # 设置优化器
        if optimizer is None:
            self.optimizer = optim.Adam(
                self.model.parameters(), lr=1e-4, weight_decay=1e-5
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
        
        print(f"Trainer initialized with device: {device}")
        print(f"Model parameters: {sum(p.numel() for p in model.parameters()):, }")
    
    def train_epoch(self) -> Dict[str, float]:
        """训练一个epoch"""
        self.model.train()
        
        epoch_loss = 0.0
        epoch_accuracy = 0.0
        num_batches = 0
        
        for batch_idx, batch in enumerate(self.train_dataloader):
            # 移动数据到设备
            points = batch['points'].to(self.device)  # [B, N, 3]
            occupancy = batch['occupancy'].to(self.device)  # [B, N, 1]
            
            # 前向传播
            self.optimizer.zero_grad()
            
            if isinstance(self.model, ConditionalOccupancyNetwork):
                # 条件占用网络需要形状编码
                shape_code = self.model.encode_shape(points, occupancy)
                pred_occupancy = self.model(points, condition=shape_code)
            else:
                pred_occupancy = self.model(points)
            
            # 计算损失
            loss = self.model.compute_loss(pred_occupancy, occupancy)
            
            # 反向传播
            loss.backward()
            
            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            self.optimizer.step()
            
            # 计算准确率
            with torch.no_grad():
                pred_binary = (pred_occupancy > 0.5).float()
                accuracy = (pred_binary == occupancy).float().mean()
            
            # 更新统计信息
            epoch_loss += loss.item()
            epoch_accuracy += accuracy.item()
            num_batches += 1
            
            # 记录日志
            if batch_idx % 100 == 0:
                self.writer.add_scalar('train/batch_loss', loss.item(), self.global_step)
                self.writer.add_scalar('train/batch_accuracy', accuracy.item(), self.global_step)
                print(f'Epoch {self.epoch}, Batch {batch_idx}/{len(self.train_dataloader)}, '
                      f'Loss: {loss.item():.4f}, Accuracy: {accuracy.item():.4f}')
            
            self.global_step += 1
        
        # 计算平均指标
        avg_loss = epoch_loss / num_batches
        avg_accuracy = epoch_accuracy / num_batches
        
        return {
            'loss': avg_loss, 'accuracy': avg_accuracy
        }
    
    def validate(self) -> Dict[str, float]:
        """验证模型"""
        if self.val_dataloader is None:
            return {}
        
        self.model.eval()
        
        val_loss = 0.0
        val_accuracy = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for batch in self.val_dataloader:
                # 移动数据到设备
                points = batch['points'].to(self.device)
                occupancy = batch['occupancy'].to(self.device)
                
                # 前向传播
                if isinstance(self.model, ConditionalOccupancyNetwork):
                    shape_code = self.model.encode_shape(points, occupancy)
                    pred_occupancy = self.model(points, condition=shape_code)
                else:
                    pred_occupancy = self.model(points)
                
                # 计算损失
                loss = self.model.compute_loss(pred_occupancy, occupancy)
                
                # 计算准确率
                pred_binary = (pred_occupancy > 0.5).float()
                accuracy = (pred_binary == occupancy).float().mean()
                
                val_loss += loss.item()
                val_accuracy += accuracy.item()
                num_batches += 1
        
        avg_val_loss = val_loss / num_batches
        avg_val_accuracy = val_accuracy / num_batches
        
        return {
            'val_loss': avg_val_loss, 'val_accuracy': avg_val_accuracy
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
        print(f"Starting training for {num_epochs} epochs...")
        
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
                if val_metrics and 'val_loss' in val_metrics:
                    if isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                        self.scheduler.step(val_metrics['val_loss'])
                    else:
                        self.scheduler.step()
                else:
                    self.scheduler.step()
            
            # 记录指标
            epoch_time = time.time() - start_time
            self.train_losses.append(train_metrics['loss'])
            
            # 记录到tensorboard
            self.writer.add_scalar('train/epoch_loss', train_metrics['loss'], epoch)
            self.writer.add_scalar('train/epoch_accuracy', train_metrics['accuracy'], epoch)
            
            if val_metrics:
                self.val_losses.append(val_metrics['val_loss'])
                self.writer.add_scalar('val/loss', val_metrics['val_loss'], epoch)
                self.writer.add_scalar('val/accuracy', val_metrics['val_accuracy'], epoch)
            
            self.writer.add_scalar(
                'train/learning_rate',
                self.optimizer.param_groups[0]['lr'],
                epoch,
            )
            
            # 打印进度
            print(f'Epoch {epoch}/{num_epochs} - {epoch_time:.2f}s')
            print(f'Train Loss: {train_metrics["loss"]:.4f}')
            if val_metrics:
                print(f'Val Loss: {val_metrics["val_loss"]:.4f}')
            print('-' * 50)
            
            # 保存最佳模型
            if val_metrics and val_metrics['val_loss'] < self.best_val_loss:
                self.best_val_loss = val_metrics['val_loss']
                self.save_checkpoint(f'best_model.pth')
                print(f'New best model saved with val_loss: {self.best_val_loss:.4f}')
            
            # 定期保存检查点
            if epoch % save_freq == 0:
                self.save_checkpoint(f'epoch_{epoch}.pth')
        
        print("Training completed!")
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
        
        if self.scheduler is not None and 'scheduler_state_dict' in checkpoint:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        print(f'Checkpoint loaded: {filepath}')
        print(f'Resumed from epoch {self.epoch}')
    
    def evaluate_mesh_extraction(
        self, test_dataloader: DataLoader, resolution: int = 64, num_samples: int = 10
    ) -> Dict[str, float]:
        """评估网格提取质量"""
        self.model.eval()
        
        mesh_metrics = []
        
        with torch.no_grad():
            for i, batch in enumerate(test_dataloader):
                if i >= num_samples:
                    break
                
                points = batch['points'].to(self.device)
                occupancy = batch['occupancy'].to(self.device)
                
                # 提取网格
                if isinstance(self.model, ConditionalOccupancyNetwork):
                    shape_code = self.model.encode_shape(points, occupancy)
                    condition = shape_code[0:1]  # 取第一个样本
                else:
                    condition = None
                
                try:
                    mesh_data = self.model.extract_mesh(
                        condition=condition, resolution=resolution
                    )
                    
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
        success_rate = len(successful_extractions) / len(mesh_metrics)
        
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
        summary = {
            'total_epochs': self.epoch, 'total_steps': self.global_step, 'best_val_loss': self.best_val_loss, 'final_train_loss': self.train_losses[-1] if self.train_losses else None, 'final_val_loss': self.val_losses[-1] if self.val_losses else None, 'model_info': self.model.get_model_size(
            )
        }
        
        return summary
    
    def save_training_summary(self, filename: str = 'training_summary.json'):
        """保存训练总结"""
        summary = self.get_training_summary()
        filepath = os.path.join(self.log_dir, filename)
        
        with open(filepath, 'w') as f:
            json.dump(summary, f, indent=2, default=str)
        
        print(f'Training summary saved: {filepath}')

def create_trainer_from_config(config: dict) -> OccupancyTrainer:
    """从配置创建训练器"""
    
    # 创建模型
    model_config = config.get('model', {})
    if config.get('conditional', False):
        model = ConditionalOccupancyNetwork(**model_config)
    else:
        model = OccupancyNetwork(**model_config)
    
    # 创建数据集
    dataset_config = config.get('dataset', {})
    train_dataset = OccupancyDataset(split='train', **dataset_config)
    val_dataset = OccupancyDataset(
        split='val',
        **dataset_config,
    )
    
    # 创建数据加载器
    dataloader_config = config.get('dataloader', {})
    train_dataloader = create_occupancy_dataloader(train_dataset, **dataloader_config)
    val_dataloader = create_occupancy_dataloader(
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
    trainer = OccupancyTrainer(
        model=model, train_dataloader=train_dataloader, val_dataloader=val_dataloader, optimizer=optimizer, scheduler=scheduler, **trainer_config
    )
    
    return trainer 