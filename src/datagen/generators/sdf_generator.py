"""
SDF生成器模块

从neural_sdf.py迁移的SDF网络和训练功能，用于生成SDF数据。
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from typing import Dict, List, Optional, Tuple, import logging

logger = logging.getLogger(__name__)


class SDFDataset(Dataset):
    """SDF数据集"""
    
    def __init__(self, coords: np.ndarray, sdf_values: np.ndarray):
        """
        初始化SDF数据集
        
        Args:
            coords: 坐标数据 (N, 3)
            sdf_values: SDF值 (N, )
        """
        self.coords = torch.FloatTensor(coords)
        self.sdf_values = torch.FloatTensor(sdf_values)
        
    def __len__(self):
        return len(self.coords)
    
    def __getitem__(self, idx):
        return self.coords[idx], self.sdf_values[idx]


class PositionalEncoding(nn.Module):
    """位置编码"""
    
    def __init__(self, input_dim: int = 3, max_freq_log2: int = 9, num_freqs: int = 10):
        super(PositionalEncoding, self).__init__()
        
        self.input_dim = input_dim
        self.max_freq_log2 = max_freq_log2
        self.num_freqs = num_freqs
        
        # 创建频率
        freq_bands = 2.0 ** torch.linspace(0.0, max_freq_log2, num_freqs)
        self.register_buffer('freq_bands', freq_bands)
        
    def forward(self, x):
        # x: (batch_size, input_dim)
        encodings = []
        for freq in self.freq_bands:
            for dim in range(self.input_dim):
                encodings.append(torch.sin(2.0 * torch.pi * freq * x[:, dim]))
                encodings.append(torch.cos(2.0 * torch.pi * freq * x[:, dim]))
        
        # 合并原始输入和编码
        encoded = torch.stack(encodings, dim=1)  # (batch_size, num_features)
        
        return torch.cat([x, encoded], dim=1)


class SDFMLP(nn.Module):
    """SDF多层感知机网络"""
    
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
        use_positional_encoding: bool = True,
        encoding_freqs: int = 10,
    )
        """
        初始化SDF MLP
        """
        super(SDFMLP, self).__init__()
        
        self.use_positional_encoding = use_positional_encoding
        
        # 位置编码
        if use_positional_encoding:
            self.pos_encoding = PositionalEncoding(input_dim, num_freqs=encoding_freqs)
            actual_input_dim = input_dim + 2 * input_dim * encoding_freqs
        else:
            actual_input_dim = input_dim
        
        # MLP层
        layers = []
        prev_dim = actual_input_dim
        
        for i, hidden_dim in enumerate(hidden_dims):
            layers.extend([
                nn.Linear(
                    prev_dim,
                    hidden_dim,
                )
            ])
            prev_dim = hidden_dim
        
        # 输出层
        layers.append(nn.Linear(prev_dim, output_dim))
        
        self.network = nn.Sequential(*layers)
        
        # 权重初始化
        self.apply(self._init_weights)
    
    def _get_activation(self, activation: str) -> nn.Module:
        """获取激活函数"""
        activations = {
            'relu': nn.ReLU(
            )
        }
        
        return activations.get(activation, nn.ReLU())
    
    def _init_weights(self, module):
        """初始化权重"""
        if isinstance(module, nn.Linear):
            torch.nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
    
    def forward(self, x):
        if self.use_positional_encoding:
            x = self.pos_encoding(x)
        return self.network(x)


class SDFGenerator:
    """SDF生成器"""
    
    def __init__(
        self,
        model_config: Optional[Dict] = None,
        device: str = 'auto',
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-5,
    )
        """初始化SDF生成器"""
        # 默认模型配置
        default_config = {
            'input_dim': 3, 'hidden_dims': [256, 512, 512, 256, 128], 'output_dim': 1, 'activation': 'relu', 'dropout': 0.1, 'use_positional_encoding': True, 'encoding_freqs': 10
        }
        
        if model_config:
            default_config.update(model_config)
        
        self.model_config = default_config
        
        # 创建模型
        self.model = SDFMLP(**self.model_config)
        
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
        
        logger.info(f"SDF生成器初始化完成，设备: {self.device}")
    
    def generate_geometric_sdf(
        self,
        coordinates: np.ndarray,
        geometry_type: str = 'sphere',
        geometry_params: Dict = None,
    )
        """生成几何体的SDF值"""
        if geometry_params is None:
            geometry_params = {}
        
        if geometry_type == 'sphere':
            return self._sphere_sdf(coordinates, geometry_params)
        elif geometry_type == 'box':
            return self._box_sdf(coordinates, geometry_params)
        elif geometry_type == 'cylinder':
            return self._cylinder_sdf(coordinates, geometry_params)
        else:
            raise ValueError(f"未知几何体类型: {geometry_type}")
    
    def _sphere_sdf(self, coords: np.ndarray, params: Dict) -> np.ndarray:
        """计算球体SDF"""
        center = np.array(params.get('center', [0, 0, 0]))
        radius = params.get('radius', 1.0)
        
        distances = np.linalg.norm(coords - center, axis=1)
        return distances - radius
    
    def _box_sdf(self, coords: np.ndarray, params: Dict) -> np.ndarray:
        """计算盒子SDF"""
        center = np.array(params.get('center', [0, 0, 0]))
        size = np.array(params.get('size', [1, 1, 1]))
        
        # 相对坐标
        relative_coords = np.abs(coords - center)
        
        # 距离盒子表面的距离
        inside_distance = np.max(relative_coords - size/2, axis=1)
        outside_distance = np.linalg.norm(np.maximum(relative_coords - size/2, 0), axis=1)
        
        # 合并内外距离
        sdf = np.where(inside_distance < 0, inside_distance, outside_distance)
        
        return sdf
    
    def _cylinder_sdf(self, coords: np.ndarray, params: Dict) -> np.ndarray:
        """计算圆柱体SDF"""
        center = np.array(params.get('center', [0, 0, 0]))
        radius = params.get('radius', 1.0)
        height = params.get('height', 2.0)
        axis = params.get('axis', 2)  # 2表示z轴
        
        # 相对坐标
        relative_coords = coords - center
        
        # 根据轴向计算
        if axis == 2:  # z轴
            radial_distance = np.linalg.norm(relative_coords[:, :2], axis=1) - radius
            axial_distance = np.abs(relative_coords[:, 2]) - height/2
        elif axis == 1:  # y轴
            radial_distance = np.linalg.norm(relative_coords[:, [0, 2]], axis=1) - radius
            axial_distance = np.abs(relative_coords[:, 1]) - height/2
        else:  # x轴
            radial_distance = np.linalg.norm(relative_coords[:, [1, 2]], axis=1) - radius
            axial_distance = np.abs(relative_coords[:, 0]) - height/2
        
        # 计算SDF
        inside_sdf = np.maximum(radial_distance, axial_distance)
        outside_sdf = np.linalg.norm(
            np.column_stack([
                np.maximum(radial_distance, 0), np.maximum(axial_distance, 0)
            ]), axis=1
        )
        
        sdf = np.where(inside_sdf < 0, inside_sdf, outside_sdf)
        
        return sdf
    
    def train_sdf_network(
        self,
        train_coords: np.ndarray,
        train_sdf: np.ndarray,
        val_coords: Optional[np.ndarray] = None,
        val_sdf: Optional[np.ndarray] = None,
        num_epochs: int = 100,
        batch_size: int = 1024,
        verbose: bool = True,
    )
        """训练SDF网络"""
        # 创建数据集和数据加载器
        train_dataset = SDFDataset(train_coords, train_sdf)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        
        val_loader = None
        if val_coords is not None and val_sdf is not None:
            val_dataset = SDFDataset(val_coords, val_sdf)
            val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        
        # 训练循环
        for epoch in range(num_epochs):
            # 训练
            train_loss = self._train_epoch(train_loader)
            self.train_losses.append(train_loss)
            
            # 验证
            if val_loader:
                val_loss = self._validate_epoch(val_loader)
                self.val_losses.append(val_loss)
            
            if verbose and (epoch + 1) % 10 == 0:
                if val_loader:
                    logger.info(f"Epoch {epoch+1}/{num_epochs}: "
                              f"Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}")
                else:
                    logger.info(f"Epoch {epoch+1}/{num_epochs}: Train Loss: {train_loss:.6f}")
        
        return {
            'train_losses': self.train_losses, 'val_losses': self.val_losses
        }
    
    def _train_epoch(self, dataloader: DataLoader) -> float:
        """训练一个epoch"""
        self.model.train()
        total_loss = 0.0
        
        for batch in dataloader:
            coords, sdf_values = batch
            coords = coords.to(self.device)
            sdf_values = sdf_values.to(self.device)
            
            # 前向传播
            self.optimizer.zero_grad()
            predictions = self.model(coords).squeeze()
            loss = self.criterion(predictions, sdf_values)
            
            # 反向传播
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
        
        return total_loss / len(dataloader)
    
    def _validate_epoch(self, dataloader: DataLoader) -> float:
        """验证一个epoch"""
        self.model.eval()
        total_loss = 0.0
        
        with torch.no_grad():
            for batch in dataloader:
                coords, sdf_values = batch
                coords = coords.to(self.device)
                sdf_values = sdf_values.to(self.device)
                
                predictions = self.model(coords).squeeze()
                loss = self.criterion(predictions, sdf_values)
                
                total_loss += loss.item()
        
        return total_loss / len(dataloader)
    
    def predict_sdf(self, coordinates: np.ndarray) -> np.ndarray:
        """预测SDF值"""
        self.model.eval()
        
        coords_tensor = torch.FloatTensor(coordinates).to(self.device)
        
        with torch.no_grad():
            predictions = self.model(coords_tensor).squeeze().cpu().numpy()
        
        return predictions
    
    def save_model(self, path: str):
        """保存模型"""
        torch.save({
            'model_state_dict': self.model.state_dict(
            )
        }, path)
        logger.info(f"模型已保存: {path}")
    
    def load_model(self, path: str):
        """加载模型"""
        checkpoint = torch.load(path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        if 'train_losses' in checkpoint:
            self.train_losses = checkpoint['train_losses']
        if 'val_losses' in checkpoint:
            self.val_losses = checkpoint['val_losses']
        
        logger.info(f"模型已加载: {path}")
    
    def generate_training_data(
        self,
        scene_bounds: tuple[float,
        float,
        float,
        float,
        float,
        float],
        geometry_configs: list[Dict],
        n_samples: int = 100000,
        noise_std: float = 0.01,
    )
        """生成SDF训练数据"""
        # 在场景范围内随机采样坐标
        coords = np.random.uniform(
            low=scene_bounds[:3], high=scene_bounds[3:], size=(n_samples, 3)
        )
        
        # 计算每个几何体的SDF
        all_sdf = []
        
        for geom_config in geometry_configs:
            geom_type = geom_config['type']
            geom_params = geom_config.get('params', {})
            
            sdf_values = self.generate_geometric_sdf(coords, geom_type, geom_params)
            all_sdf.append(sdf_values)
        
        # 合并多个几何体（取最小值）
        if len(all_sdf) > 1:
            combined_sdf = np.minimum.reduce(all_sdf)
        else:
            combined_sdf = all_sdf[0]
        
        # 添加噪声
        if noise_std > 0:
            noise = np.random.normal(0, noise_std, coords.shape)
            coords += noise
        
        logger.info(f"生成了 {n_samples} 个SDF训练样本")
        
        return coords, combined_sdf 