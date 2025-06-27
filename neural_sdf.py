from typing import Optional, import matplotlib.pyplot as plt
#!/usr/bin/env python3
"""
SDF/Occupancy神经网络训练模块
支持多种网络架构和训练策略
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import json
import os
from tqdm import tqdm
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class VoxelDataset(Dataset):
    """体素数据集"""
    
    def __init__(
        self,
        coords: np.ndarray,
        labels: Optional[np.ndarray] = None,
        sdf_values: Optional[np.ndarray] = None,
        task_type: str = 'occupancy',
    )
        """
        初始化数据集
        
        Args:
            coords: 坐标数据 (N, 3)
            labels: 占用标签 (N, ) 或 None
            sdf_values: SDF值 (N, ) 或 None
            task_type: 任务类型 ('occupancy' 或 'sdf')
        """
        self.coords = torch.FloatTensor(coords)
        self.task_type = task_type
        
        if task_type == 'occupancy':
            if labels is None:
                raise ValueError("Occupancy任务需要labels")
            self.targets = torch.FloatTensor(labels)
        elif task_type == 'sdf':
            if sdf_values is None:
                raise ValueError("SDF任务需要sdf_values")
            self.targets = torch.FloatTensor(sdf_values)
        else:
            raise ValueError(f"未知任务类型: {task_type}")
    
    def __len__(self):
        return len(self.coords)
    
    def __getitem__(self, idx):
        return self.coords[idx], self.targets[idx]

class MLP(nn.Module):
    """多层感知机网络"""
    
    def __init__(
        self,
        input_dim: int = 3,
        hidden_dims: list[int] = [256,
        512,
        512,
        256,
        128],
        output_dim: int = 1,
        activation: str = 'relu',
        dropout: float = 0.1,
    )
        """
        初始化MLP
        
        Args:
            input_dim: 输入维度
            hidden_dims: 隐藏层维度列表
            output_dim: 输出维度
            activation: 激活函数
            dropout: dropout比例
        """
        super(MLP, self).__init__()
        
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(
                    prev_dim,
                    hidden_dim,
                )
            ])
            prev_dim = hidden_dim
        
        layers.append(nn.Linear(prev_dim, output_dim))
        
        self.network = nn.Sequential(*layers)
        
        # 初始化权重
        self.apply(self._init_weights)
    
    def _get_activation(self, activation: str) -> nn.Module:
        """获取激活函数"""
        if activation == 'relu':
            return nn.ReLU()
        elif activation == 'leaky_relu':
            return nn.LeakyReLU()
        elif activation == 'tanh':
            return nn.Tanh()
        elif activation == 'sigmoid':
            return nn.Sigmoid()
        else:
            raise ValueError(f"未知激活函数: {activation}")
    
    def _init_weights(self, module):
        """初始化权重"""
        if isinstance(module, nn.Linear):
            torch.nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
    
    def forward(self, x):
        return self.network(x)

class PositionalEncoding(nn.Module):
    """位置编码"""
    
    def __init__(self, input_dim: int = 3, max_freq_log2: int = 9, num_freqs: int = 10):
        super(PositionalEncoding, self).__init__()
        
        self.input_dim = input_dim
        self.max_freq_log2 = max_freq_log2
        self.num_freqs = num_freqs
        
        # 创建频率
        freq_bands = 2.0 ** torch.linspace(0.0, max_freq_log2, num_freqs)
        
        # 创建编码矩阵
        encodings = []
        for freq in freq_bands:
            for dim in range(input_dim):
                encodings.extend([torch.sin(freq * torch.pi * torch.arange(2))])
        
        self.register_buffer('encodings', torch.stack(encodings))
    
    def forward(self, x):
        # x: (batch_size, input_dim)
        batch_size = x.shape[0]
        
        # 扩展输入
        x_expanded = x.unsqueeze(-1).expand(-1, -1, 2)  # (batch_size, input_dim, 2)
        
        # 应用编码
        encoded = torch.cat([
            torch.sin(self.encodings * x_expanded), torch.cos(self.encodings * x_expanded)
        ], dim=-1)
        
        # 展平
        encoded = encoded.view(batch_size, -1)
        
        return encoded

class NeuralSDFTrainer:
    """SDF/Occupancy网络训练器"""
    
    def __init__(
        self,
        model: nn.Module,
        device: str = 'auto',
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-5,
    )
        """
        初始化训练器
        
        Args:
            model: 神经网络模型
            device: 设备 ('auto', 'cuda', 'cpu')
            learning_rate: 学习率
            weight_decay: 权重衰减
        """
        self.model = model
        
        # 设置设备
        if device == 'auto':
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        self.model.to(self.device)
        
        # 设置优化器
        self.optimizer = optim.Adam(
            self.model.parameters(), lr=learning_rate, weight_decay=weight_decay
        )
        
        # 设置损失函数
        self.criterion = nn.MSELoss()
        
        # 训练历史
        self.train_losses = []
        self.val_losses = []
    
    def train_epoch(self, dataloader: DataLoader) -> float:
        """训练一个epoch"""
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        
        for coords, targets in tqdm(dataloader, desc="Training"):
            coords = coords.to(self.device)
            targets = targets.to(self.device)
            
            # 前向传播
            self.optimizer.zero_grad()
            outputs = self.model(coords)
            loss = self.criterion(outputs.squeeze(), targets)
            
            # 反向传播
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
        
        return total_loss / num_batches
    
    def validate(self, dataloader: DataLoader) -> float:
        """验证"""
        self.model.eval()
        total_loss = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for coords, targets in tqdm(dataloader, desc="Validation"):
                coords = coords.to(self.device)
                targets = targets.to(self.device)
                
                outputs = self.model(coords)
                loss = self.criterion(outputs.squeeze(), targets)
                
                total_loss += loss.item()
                num_batches += 1
        
        return total_loss / num_batches
    
    def train(
        self,
        train_dataloader: DataLoader,
        val_dataloader: Optional[DataLoader] = None,
        num_epochs: int = 100,
        save_path: Optional[str] = None,
        early_stopping_patience: int = 10,
    )
        """
        训练模型
        
        Args:
            train_dataloader: 训练数据加载器
            val_dataloader: 验证数据加载器
            num_epochs: 训练轮数
            save_path: 模型保存路径
            early_stopping_patience: 早停耐心值
        """
        best_val_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(num_epochs):
            logger.info(f"Epoch {epoch+1}/{num_epochs}")
            
            # 训练
            train_loss = self.train_epoch(train_dataloader)
            self.train_losses.append(train_loss)
            
            logger.info(f"Train Loss: {train_loss:.6f}")
            
            # 验证
            if val_dataloader is not None:
                val_loss = self.validate(val_dataloader)
                self.val_losses.append(val_loss)
                
                logger.info(f"Val Loss: {val_loss:.6f}")
                
                # 早停检查
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                    
                    # 保存最佳模型
                    if save_path:
                        self.save_model(save_path)
                else:
                    patience_counter += 1
                
                if patience_counter >= early_stopping_patience:
                    logger.info(f"Early stopping at epoch {epoch+1}")
                    break
            else:
                # 没有验证集，保存最后一个模型
                if save_path:
                    self.save_model(save_path)
        
        logger.info("训练完成！")
    
    def save_model(self, path: str):
        """保存模型"""
        torch.save({
            'model_state_dict': self.model.state_dict(
            )
        }, path)
        logger.info(f"模型已保存到: {path}")
    
    def load_model(self, path: str):
        """加载模型"""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.train_losses = checkpoint.get('train_losses', [])
        self.val_losses = checkpoint.get('val_losses', [])
        logger.info(f"模型已从 {path} 加载")
    
    def predict(self, coords: np.ndarray) -> np.ndarray:
        """预测"""
        self.model.eval()
        coords_tensor = torch.FloatTensor(coords).to(self.device)
        
        with torch.no_grad():
            outputs = self.model(coords_tensor)
        
        return outputs.cpu().numpy()
    
    def plot_training_history(self, save_path: Optional[str] = None):
        """绘制训练历史"""
        plt.figure(figsize=(12, 4))
        
        # 训练损失
        plt.subplot(1, 2, 1)
        plt.plot(self.train_losses, label='Train Loss')
        if self.val_losses:
            plt.plot(self.val_losses, label='Val Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training History')
        plt.legend()
        plt.yscale('log')
        
        # 损失分布
        plt.subplot(1, 2, 2)
        plt.hist(self.train_losses, bins=50, alpha=0.7, label='Train Loss')
        if self.val_losses:
            plt.hist(self.val_losses, bins=50, alpha=0.7, label='Val Loss')
        plt.xlabel('Loss')
        plt.ylabel('Frequency')
        plt.title('Loss Distribution')
        plt.legend()
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path)
            logger.info(f"训练历史图已保存到: {save_path}")
        else:
            plt.show()

def load_training_data(
    samples_dir: str,
    task_type: str = 'occupancy',
    train_ratio: float = 0.8,
)
    """
    加载训练数据
    
    Args:
        samples_dir: 采样数据目录
        task_type: 任务类型
        train_ratio: 训练集比例
        
    Returns:
        (train_dataloader, val_dataloader)
    """
    # 加载所有采样数据
    all_coords = []
    all_targets = []
    
    # 扫描所有坐标文件
    coord_files = [f for f in os.listdir(samples_dir) if f.startswith('coords_') and f.endswith('.npy')]
    
    for coord_file in coord_files:
        tile_name = coord_file.replace('coords_', '').replace('.npy', '')
        
        # 加载坐标
        coords = np.load(os.path.join(samples_dir, coord_file))
        all_coords.append(coords)
        
        # 加载标签或SDF值
        if task_type == 'occupancy':
            label_file = f"labels_{tile_name}.npy"
            if os.path.exists(os.path.join(samples_dir, label_file)):
                labels = np.load(os.path.join(samples_dir, label_file))
                all_targets.append(labels)
            else:
                logger.warning(f"找不到标签文件: {label_file}")
                continue
        elif task_type == 'sdf':
            sdf_file = f"sdf_{tile_name}.npy"
            if os.path.exists(os.path.join(samples_dir, sdf_file)):
                sdf_values = np.load(os.path.join(samples_dir, sdf_file))
                all_targets.append(sdf_values)
            else:
                logger.warning(f"找不到SDF文件: {sdf_file}")
                continue
    
    # 合并所有数据
    all_coords = np.concatenate(all_coords, axis=0)
    all_targets = np.concatenate(all_targets, axis=0)
    
    logger.info(f"加载数据: {all_coords.shape}, 目标: {all_targets.shape}")
    
    # 划分训练集和验证集
    n_samples = len(all_coords)
    n_train = int(n_samples * train_ratio)
    
    indices = np.random.permutation(n_samples)
    train_indices = indices[:n_train]
    val_indices = indices[n_train:]
    
    # 创建数据集
    train_dataset = VoxelDataset(
        all_coords[train_indices], all_targets[train_indices] if task_type == 'occupancy' else None, all_targets[train_indices] if task_type == 'sdf' else None, task_type
    )
    
    val_dataset = VoxelDataset(
        all_coords[val_indices], all_targets[val_indices] if task_type == 'occupancy' else None, all_targets[val_indices] if task_type == 'sdf' else None, task_type
    )
    
    # 创建数据加载器
    train_dataloader = DataLoader(train_dataset, batch_size=1024, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=1024, shuffle=False)
    
    return train_dataloader, val_dataloader

def main():
    """示例用法"""
    # 配置参数
    task_type = 'occupancy'  # 或 'sdf'
    samples_dir = 'samples'
    model_save_path = f'model_{task_type}.pth'
    
    # 加载数据
    logger.info("加载训练数据...")
    train_dataloader, val_dataloader = load_training_data(
        samples_dir, task_type=task_type
    )
    
    # 创建模型
    if task_type == 'occupancy':
        model = MLP(
            input_dim=3, hidden_dims=[256, 512, 512, 256, 128], output_dim=1, activation='relu'
        )
    else:  # SDF
        model = MLP(
            input_dim=3, hidden_dims=[512, 1024, 1024, 512, 256], output_dim=1, activation='relu'
        )
    
    # 创建训练器
    trainer = NeuralSDFTrainer(
        model=model, learning_rate=1e-3, weight_decay=1e-5
    )
    
    # 训练模型
    logger.info("开始训练...")
    trainer.train(
        train_dataloader=train_dataloader, val_dataloader=val_dataloader, num_epochs=50, save_path=model_save_path, early_stopping_patience=10
    )
    
    # 绘制训练历史
    trainer.plot_training_history(f'training_history_{task_type}.png')
    
    logger.info("训练完成！")

if __name__ == "__main__":
    main() 