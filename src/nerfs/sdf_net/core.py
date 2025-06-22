"""
DeepSDF Network Core Implementation
基于论文: "DeepSDF: Learning Continuous Signed Distance Functions for Shape Representation"

DeepSDF学习一个连续函数 f: (R^3, Z) -> R，
其中Z是潜在编码，输出是有符号距离值。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional, Union


class SDFNetwork(nn.Module):
    """SDF网络模型
    
    学习一个函数 f: (R^3, Z) -> R 来预测有符号距离值
    其中Z是形状的潜在编码
    
    Args:
        dim_input: 输入维度（通常为3，表示3D坐标）
        dim_latent: 潜在编码维度
        dim_hidden: 隐藏层维度
        num_layers: 网络层数
        skip_connections: 跳跃连接层索引
        geometric_init: 是否使用几何初始化
        beta: 几何初始化参数
        bias: 输出层偏置初始值
        weight_norm: 是否使用权重归一化
    """
    
    def __init__(
        self,
        dim_input: int = 3,
        dim_latent: int = 256,
        dim_hidden: int = 512,
        num_layers: int = 8,
        skip_connections: List[int] = [4],
        geometric_init: bool = True,
        beta: float = 100,
        bias: float = 0.5,
        weight_norm: bool = True,
        **kwargs
    ):
        super().__init__()
        
        self.dim_input = dim_input
        self.dim_latent = dim_latent
        self.dim_hidden = dim_hidden
        self.num_layers = num_layers
        self.skip_connections = skip_connections
        self.geometric_init = geometric_init
        self.beta = beta
        self.bias = bias
        self.weight_norm = weight_norm
        
        # 构建网络层
        dims = [dim_input + dim_latent] + [dim_hidden] * (num_layers - 1) + [1]
        
        self.num_layers = len(dims)
        self.skip_in = skip_connections
        
        for layer_idx in range(0, self.num_layers - 1):
            # 检查是否有跳跃连接
            if layer_idx + 1 in skip_connections:
                out_dim = dims[layer_idx + 1] - dims[0]
            else:
                out_dim = dims[layer_idx + 1]
            
            # 创建线性层
            lin = nn.Linear(dims[layer_idx], out_dim)
            
            # 几何初始化
            if geometric_init:
                self._geometric_init(lin, layer_idx == self.num_layers - 2)
            
            # 权重归一化
            if weight_norm:
                lin = nn.utils.weight_norm(lin)
            
            setattr(self, "lin" + str(layer_idx), lin)
        
        # 激活函数
        self.activation = nn.Softplus(beta=100)
    
    def _geometric_init(self, layer: nn.Module, is_output: bool = False):
        """几何初始化"""
        if is_output:
            # 输出层初始化
            torch.nn.init.normal_(layer.weight, mean=np.sqrt(np.pi) / np.sqrt(layer.in_features), std=0.0001)
            torch.nn.init.constant_(layer.bias, -self.bias)
        else:
            # 隐藏层初始化
            torch.nn.init.constant_(layer.bias, 0.0)
            torch.nn.init.normal_(layer.weight, 0.0, np.sqrt(2) / np.sqrt(layer.out_features))
    
    def forward(
        self, 
        points: torch.Tensor,
        latent_code: torch.Tensor
    ) -> torch.Tensor:
        """前向传播
        
        Args:
            points: 输入点坐标 [B, N, 3] 或 [B*N, 3]
            latent_code: 潜在编码 [B, dim_latent] 或 [B*N, dim_latent]
            
        Returns:
            sdf: 有符号距离值 [B, N, 1] 或 [B*N, 1]
        """
        batch_size = points.shape[0]
        
        # 展平输入
        if points.dim() == 3:
            B, N, _ = points.shape
            points = points.reshape(B * N, -1)
            # 扩展潜在编码
            if latent_code.dim() == 2:
                latent_code = latent_code.unsqueeze(1).expand(-1, N, -1)
                latent_code = latent_code.reshape(B * N, -1)
            reshape_output = True
        else:
            B, N = batch_size, 1
            reshape_output = False
        
        # 连接输入和潜在编码
        inputs = torch.cat([points, latent_code], dim=-1)
        
        x = inputs
        for layer_idx in range(0, self.num_layers - 1):
            lin = getattr(self, "lin" + str(layer_idx))
            
            # 跳跃连接
            if layer_idx in self.skip_in:
                x = torch.cat([x, inputs], dim=-1) / np.sqrt(2)
            
            x = lin(x)
            
            # 最后一层不使用激活函数
            if layer_idx < self.num_layers - 2:
                x = self.activation(x)
        
        sdf = x
        
        # 重塑输出
        if reshape_output:
            sdf = sdf.reshape(B, N, 1)
        
        return sdf
    
    def predict_sdf(
        self,
        points: torch.Tensor,
        latent_code: torch.Tensor,
        chunk_size: int = 100000
    ) -> torch.Tensor:
        """预测SDF值（支持大批量推理）
        
        Args:
            points: 查询点 [N, 3]
            latent_code: 潜在编码 [1, dim_latent]
            chunk_size: 分块大小
            
        Returns:
            sdf: 有符号距离值 [N, 1]
        """
        self.eval()
        
        num_points = points.shape[0]
        sdf_list = []
        
        with torch.no_grad():
            for i in range(0, num_points, chunk_size):
                end_idx = min(i + chunk_size, num_points)
                points_chunk = points[i:end_idx]
                
                # 扩展潜在编码
                latent_chunk = latent_code.expand(points_chunk.shape[0], -1)
                
                sdf_chunk = self.forward(points_chunk, latent_chunk)
                sdf_list.append(sdf_chunk)
        
        return torch.cat(sdf_list, dim=0)
    
    def extract_mesh(
        self,
        latent_code: torch.Tensor,
        resolution: int = 256,
        threshold: float = 0.0,
        bbox: Optional[Tuple[float, float]] = None,
        use_marching_cubes: bool = True
    ) -> Dict:
        """提取网格表面
        
        Args:
            latent_code: 潜在编码
            resolution: 网格分辨率
            threshold: SDF阈值（通常为0）
            bbox: 边界框 (min_val, max_val)
            use_marching_cubes: 是否使用marching cubes
            
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
            latent_code = latent_code.cuda()
        
        # 预测SDF值
        sdf = self.predict_sdf(points, latent_code)
        sdf = sdf.reshape(resolution, resolution, resolution)
        
        if use_marching_cubes:
            # 使用marching cubes提取表面
            try:
                from skimage import measure
                
                sdf_np = sdf.cpu().numpy()
                vertices, faces, _, _ = measure.marching_cubes(
                    sdf_np,
                    level=threshold,
                    spacing=(
                        (bbox[1] - bbox[0]) / (resolution - 1),
                        (bbox[1] - bbox[0]) / (resolution - 1),
                        (bbox[1] - bbox[0]) / (resolution - 1)
                    )
                )
                
                # 调整顶点位置
                vertices = vertices + bbox[0]
                
                return {
                    'vertices': vertices,
                    'faces': faces,
                    'sdf_grid': sdf_np
                }
                
            except ImportError:
                print("Warning: scikit-image not found, returning SDF grid only")
        
        return {
            'sdf_grid': sdf.cpu().numpy()
        }
    
    def compute_sdf_loss(
        self,
        pred_sdf: torch.Tensor,
        gt_sdf: torch.Tensor,
        loss_type: str = 'l1',
        reduction: str = 'mean'
    ) -> torch.Tensor:
        """计算SDF预测损失
        
        Args:
            pred_sdf: 预测SDF值 [B, N, 1]
            gt_sdf: 真实SDF值 [B, N, 1]
            loss_type: 损失类型 ('l1', 'l2', 'huber')
            reduction: 损失聚合方式
            
        Returns:
            loss: SDF损失
        """
        pred_sdf = pred_sdf.squeeze(-1)
        gt_sdf = gt_sdf.squeeze(-1)
        
        if loss_type == 'l1':
            loss = F.l1_loss(pred_sdf, gt_sdf, reduction=reduction)
        elif loss_type == 'l2':
            loss = F.mse_loss(pred_sdf, gt_sdf, reduction=reduction)
        elif loss_type == 'huber':
            loss = F.smooth_l1_loss(pred_sdf, gt_sdf, reduction=reduction)
        else:
            raise ValueError(f"Unsupported loss type: {loss_type}")
        
        return loss
    
    def compute_gradient_penalty(
        self,
        points: torch.Tensor,
        latent_code: torch.Tensor,
        lambda_gp: float = 0.1
    ) -> torch.Tensor:
        """计算梯度惩罚项（Eikonal约束）
        
        Args:
            points: 输入点 [B, N, 3]
            latent_code: 潜在编码 [B, dim_latent]
            lambda_gp: 梯度惩罚权重
            
        Returns:
            gradient_penalty: 梯度惩罚损失
        """
        points.requires_grad_(True)
        
        sdf = self.forward(points, latent_code)
        
        # 计算梯度
        gradients = torch.autograd.grad(
            outputs=sdf,
            inputs=points,
            grad_outputs=torch.ones_like(sdf),
            create_graph=True,
            retain_graph=True,
            only_inputs=True
        )[0]
        
        # Eikonal约束：||∇f|| = 1
        gradient_norm = gradients.norm(2, dim=-1)
        gradient_penalty = lambda_gp * ((gradient_norm - 1) ** 2).mean()
        
        return gradient_penalty
    
    def get_model_size(self) -> Dict[str, Union[int, float]]:
        """获取模型大小信息"""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        # 估计模型大小（MB）
        model_size_mb = total_params * 4 / (1024 * 1024)  # 假设float32
        
        return {
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'model_size_mb': model_size_mb
        }


class LatentSDFNetwork(nn.Module):
    """潜在SDF网络
    
    包含形状编码器和SDF解码器的完整模型
    """
    
    def __init__(
        self,
        dim_latent: int = 256,
        num_shapes: Optional[int] = None,
        **sdf_kwargs
    ):
        super().__init__()
        
        self.dim_latent = dim_latent
        self.num_shapes = num_shapes
        
        # SDF解码器
        self.sdf_decoder = SDFNetwork(dim_latent=dim_latent, **sdf_kwargs)
        
        # 潜在编码（如果指定了形状数量）
        if num_shapes is not None:
            self.latent_codes = nn.Embedding(num_shapes, dim_latent)
            # 初始化潜在编码
            nn.init.normal_(self.latent_codes.weight, 0.0, 1e-4)
    
    def forward(
        self,
        points: torch.Tensor,
        shape_ids: Optional[torch.Tensor] = None,
        latent_code: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """前向传播
        
        Args:
            points: 输入点 [B, N, 3]
            shape_ids: 形状ID [B] (可选)
            latent_code: 直接提供的潜在编码 [B, dim_latent] (可选)
            
        Returns:
            sdf: 有符号距离值 [B, N, 1]
        """
        # 获取潜在编码
        if latent_code is None:
            if shape_ids is None:
                raise ValueError("Either shape_ids or latent_code must be provided")
            latent_code = self.latent_codes(shape_ids)
        
        return self.sdf_decoder(points, latent_code)
    
    def encode_shape(self, points: torch.Tensor, sdf: torch.Tensor) -> torch.Tensor:
        """编码形状（简单版本，实际应用中可能需要更复杂的编码器）
        
        Args:
            points: 输入点 [B, N, 3]
            sdf: SDF值 [B, N, 1]
            
        Returns:
            latent_code: 潜在编码 [B, dim_latent]
        """
        # 简单的平均池化编码
        features = torch.cat([points, sdf], dim=-1)  # [B, N, 4]
        shape_code = features.mean(dim=1)  # [B, 4]
        
        # 通过全连接层映射到潜在空间
        if not hasattr(self, 'shape_encoder'):
            self.shape_encoder = nn.Linear(4, self.dim_latent)
            if torch.cuda.is_available():
                self.shape_encoder = self.shape_encoder.cuda()
        
        latent_code = self.shape_encoder(shape_code)
        return latent_code
    
    def get_latent_code(self, shape_id: int) -> torch.Tensor:
        """获取指定形状的潜在编码"""
        if self.num_shapes is None:
            raise ValueError("num_shapes must be specified to use shape IDs")
        
        shape_id_tensor = torch.tensor([shape_id], dtype=torch.long)
        if torch.cuda.is_available():
            shape_id_tensor = shape_id_tensor.cuda()
        
        return self.latent_codes(shape_id_tensor)


class MultiScaleSDFNetwork(SDFNetwork):
    """多尺度SDF网络
    
    在不同分辨率下进行SDF预测
    """
    
    def __init__(
        self,
        scales: List[float] = [1.0, 0.5, 0.25],
        **kwargs
    ):
        super().__init__(**kwargs)
        self.scales = scales
    
    def forward_multiscale(
        self,
        points: torch.Tensor,
        latent_code: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """多尺度前向传播
        
        Args:
            points: 输入点 [B, N, 3]
            latent_code: 潜在编码 [B, dim_latent]
            
        Returns:
            multi_sdf: 不同尺度的SDF预测
        """
        results = {}
        
        for scale in self.scales:
            scaled_points = points * scale
            sdf = self.forward(scaled_points, latent_code)
            results[f'scale_{scale}'] = sdf
        
        return results 