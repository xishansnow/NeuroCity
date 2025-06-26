"""
Occupancy Network Core Implementation
基于论文: "Occupancy Networks: Learning 3D Reconstruction in Function Space"

占用网络学习一个连续函数 f: R^3 -> [0, 1]，
将3D空间中的点映射为占用概率。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Union


class ResnetBlockFC(nn.Module):
    """全连接残差块
    
    Args:
        size_in: 输入维度
        size_out: 输出维度
        size_h: 隐藏层维度
    """
    def __init__(self, size_in: int, size_out: int = None, size_h: int = 128):
        super().__init__()
        if size_out is None:
            size_out = size_in
        
        self.size_in = size_in
        self.size_out = size_out
        self.size_h = size_h
        
        # 定义层
        self.fc_0 = nn.Linear(size_in, size_h)
        self.fc_1 = nn.Linear(size_h, size_out)
        self.actvn = nn.ReLU()
        
        # 快捷连接
        if size_in == size_out:
            self.shortcut = None
        else:
            self.shortcut = nn.Linear(size_in, size_out, bias=False)
        
        # 初始化
        nn.init.zeros_(self.fc_1.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        net = self.fc_0(self.actvn(x))
        dx = self.fc_1(self.actvn(net))
        
        if self.shortcut is not None:
            x_s = self.shortcut(x)
        else:
            x_s = x
        
        return x_s + dx


class OccupancyNetwork(nn.Module):
    """占用网络模型
    
    学习一个函数 f: R^3 -> [0, 1] 来预测空间占用概率
    
    Args:
        dim_input: 输入维度（通常为3，表示3D坐标）
        dim_hidden: 隐藏层维度
        num_layers: 网络层数
        use_batch_norm: 是否使用批标准化
        dropout_prob: dropout概率
        use_conditioning: 是否使用条件编码
        dim_condition: 条件编码维度
    """
    
    def __init__(
        self, dim_input: int = 3, dim_hidden: int = 128, num_layers: int = 5, use_batch_norm: bool = True, dropout_prob: float = 0.0, use_conditioning: bool = False, dim_condition: int = 128, **kwargs
    ):
        super().__init__()
        
        self.dim_input = dim_input
        self.dim_hidden = dim_hidden
        self.num_layers = num_layers
        self.use_batch_norm = use_batch_norm
        self.dropout_prob = dropout_prob
        self.use_conditioning = use_conditioning
        self.dim_condition = dim_condition
        
        # 输入层
        if use_conditioning:
            self.fc_in = nn.Linear(dim_input + dim_condition, dim_hidden)
        else:
            self.fc_in = nn.Linear(dim_input, dim_hidden)
        
        # 隐藏层（使用残差块）
        self.blocks = nn.ModuleList([
            ResnetBlockFC(dim_hidden, dim_hidden, dim_hidden)
            for _ in range(num_layers)
        ])
        
        # 输出层
        self.fc_out = nn.Linear(dim_hidden, 1)
        
        # 批标准化
        if use_batch_norm:
            self.bn = nn.BatchNorm1d(dim_hidden)
        
        # Dropout
        if dropout_prob > 0:
            self.dropout = nn.Dropout(dropout_prob)
        
        # 激活函数
        self.actvn = nn.ReLU()
        
        # 初始化权重
        self._initialize_weights()
    
    def _initialize_weights(self):
        """初始化网络权重"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def forward(
        self, points: torch.Tensor, condition: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """前向传播
        
        Args:
            points: 输入点坐标 [B, N, 3] 或 [B*N, 3]
            condition: 条件编码 [B, dim_condition] (可选)
            
        Returns:
            occupancy: 占用概率 [B, N, 1] 或 [B*N, 1]
        """
        batch_size = points.shape[0]
        
        # 展平输入
        if points.dim() == 3:
            B, N, _ = points.shape
            points = points.reshape(B * N, -1)
            reshape_output = True
        else:
            B, N = batch_size, 1
            reshape_output = False
        
        # 条件编码
        if self.use_conditioning and condition is not None:
            # 扩展条件编码
            if condition.dim() == 2:
                condition = condition.unsqueeze(1).expand(-1, N, -1)
                condition = condition.reshape(B * N, -1)
            
            # 连接输入和条件
            x = torch.cat([points, condition], dim=-1)
        else:
            x = points
        
        # 前向传播
        x = self.fc_in(x)
        x = self.actvn(x)
        
        if self.use_batch_norm:
            x = self.bn(x)
        
        if hasattr(self, 'dropout'):
            x = self.dropout(x)
        
        # 残差块
        for block in self.blocks:
            x = block(x)
        
        # 输出层
        occupancy = self.fc_out(x)
        occupancy = torch.sigmoid(occupancy)  # 将输出映射到[0, 1]
        
        # 重塑输出
        if reshape_output:
            occupancy = occupancy.reshape(B, N, 1)
        
        return occupancy
    
    def predict_occupancy(
        self, points: torch.Tensor, condition: Optional[torch.Tensor] = None, chunk_size: int = 100000
    ) -> torch.Tensor:
        """预测占用概率（支持大批量推理）
        
        Args:
            points: 查询点 [N, 3]
            condition: 条件编码 [1, dim_condition]
            chunk_size: 分块大小
            
        Returns:
            occupancy: 占用概率 [N, 1]
        """
        self.eval()
        
        num_points = points.shape[0]
        occupancy_list = []
        
        with torch.no_grad():
            for i in range(0, num_points, chunk_size):
                end_idx = min(i + chunk_size, num_points)
                points_chunk = points[i:end_idx]
                
                if condition is not None:
                    condition_chunk = condition.expand(1, -1)
                else:
                    condition_chunk = None
                
                occupancy_chunk = self.forward(
                    points_chunk.unsqueeze(0), condition_chunk
                ).squeeze(0)
                
                occupancy_list.append(occupancy_chunk)
        
        return torch.cat(occupancy_list, dim=0)
    
    def extract_mesh(
        self, condition: Optional[torch.Tensor] = None, resolution: int = 64, threshold: float = 0.5, bbox: Optional[tuple[float, float]] = None
    ) -> Dict:
        """提取网格表面
        
        Args:
            condition: 条件编码
            resolution: 网格分辨率
            threshold: 占用概率阈值
            bbox: 边界框 (min_val, max_val)
            
        Returns:
            mesh_data: 包含顶点和面的字典
        """
        if bbox is None:
            bbox = (-1.0, 1.0)
        
        # 创建网格点
        x = np.linspace(bbox[0], bbox[1], resolution)
        y = np.linspace(bbox[0], bbox[1], resolution)
        z = np.linspace(bbox[0], bbox[1], resolution)
        
        xx, yy, zz = np.meshgrid(x, y, z, indexing='ij')
        points = np.stack([xx.ravel(), yy.ravel(), zz.ravel()], axis=1)
        points = torch.from_numpy(points).float()
        
        if torch.cuda.is_available():
            points = points.cuda()
        
        # 预测占用概率
        occupancy = self.predict_occupancy(points, condition)
        occupancy = occupancy.reshape(resolution, resolution, resolution)
        
        # 使用marching cubes提取表面（需要scikit-image）
        try:
            from skimage import measure
            
            occupancy_np = occupancy.cpu().numpy()
            vertices, faces, _, _ = measure.marching_cubes(
                occupancy_np, level=threshold, spacing=(
                    (
                        bbox[1] - bbox[0],
                    )
                )
            )
            
            # 调整顶点位置
            vertices = vertices + bbox[0]
            
            return {
                'vertices': vertices, 'faces': faces, 'occupancy_grid': occupancy_np
            }
            
        except ImportError:
            print("Warning: scikit-image not found, returning occupancy grid only")
            return {
                'occupancy_grid': occupancy.cpu().numpy()
            }
    
    def compute_loss(
        self, pred_occupancy: torch.Tensor, gt_occupancy: torch.Tensor, reduction: str = 'mean'
    ) -> torch.Tensor:
        """计算占用预测损失
        
        Args:
            pred_occupancy: 预测占用概率 [B, N, 1]
            gt_occupancy: 真实占用标签 [B, N, 1]
            reduction: 损失聚合方式
            
        Returns:
            loss: 二元交叉熵损失
        """
        pred_occupancy = pred_occupancy.squeeze(-1)
        gt_occupancy = gt_occupancy.squeeze(-1).float()
        
        loss = F.binary_cross_entropy(
            pred_occupancy, gt_occupancy, reduction=reduction
        )
        
        return loss
    
    def get_model_size(self) -> dict[str, int | float]:
        """获取模型大小信息"""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        # 估计模型大小（MB）
        model_size_mb = total_params * 4 / (1024 * 1024)  # 假设float32
        
        return {
            'total_parameters': total_params, 'trainable_parameters': trainable_params, 'model_size_mb': model_size_mb
        }


class ConditionalOccupancyNetwork(OccupancyNetwork):
    """条件占用网络
    
    支持基于特征编码的条件生成
    """
    
    def __init__(
        self, dim_input: int = 3, dim_hidden: int = 128, num_layers: int = 5, dim_condition: int = 128, **kwargs
    ):
        super().__init__(
            dim_input=dim_input, dim_hidden=dim_hidden, num_layers=num_layers, use_conditioning=True, dim_condition=dim_condition, **kwargs
        )
    
    def encode_shape(self, points: torch.Tensor, occupancy: torch.Tensor) -> torch.Tensor:
        """编码形状特征
        
        Args:
            points: 输入点 [B, N, 3]
            occupancy: 占用标签 [B, N, 1]
            
        Returns:
            shape_code: 形状编码 [B, dim_condition]
        """
        # 简单的平均池化编码（实际应用中可使用更复杂的编码器）
        features = torch.cat([points, occupancy], dim=-1)  # [B, N, 4]
        shape_code = features.mean(dim=1)  # [B, 4]
        
        # 通过全连接层映射到目标维度
        if not hasattr(self, 'shape_encoder'):
            self.shape_encoder = nn.Linear(4, self.dim_condition)
            if torch.cuda.is_available():
                self.shape_encoder = self.shape_encoder.cuda()
        
        shape_code = self.shape_encoder(shape_code)
        return shape_code 