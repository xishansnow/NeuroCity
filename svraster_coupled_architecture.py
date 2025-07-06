"""
SVRaster 架构重构 - 明确的耦合设计

这个文件展示了重构后的 SVRaster 架构，其中：
1. SVRasterTrainer 与 VolumeRenderer 紧密耦合（训练阶段的体积渲染）
2. SVRasterRenderer 与 TrueVoxelRasterizer 紧密耦合（推理阶段的光栅化）

这种设计确保了训练和推理阶段使用不同的渲染策略，符合 SVRaster 论文的设计思想。
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from typing import Dict, List, Optional, Tuple, Union, Any, TYPE_CHECKING
from dataclasses import dataclass
from pathlib import Path
import logging
from tqdm import tqdm

# 解决循环导入问题
if TYPE_CHECKING:
    from src.nerfs.svraster.core import SVRasterModel, SVRasterConfig, SVRasterLoss
    from src.nerfs.svraster.volume_renderer import VolumeRenderer
    from src.nerfs.svraster.true_rasterizer import TrueVoxelRasterizer

logger = logging.getLogger(__name__)

# 尝试导入，如果失败则创建占位符
try:
    import imageio
except ImportError:
    imageio = None
    logger.warning("imageio not available for video rendering")


# ============================================================================
# 1. SVRasterTrainer - 与 VolumeRenderer 紧密耦合
# ============================================================================

class SVRasterTrainer:
    """
    SVRaster 训练器 - 与 VolumeRenderer 紧密耦合
    
    专门负责：
    - 使用体积渲染进行训练
    - 梯度优化和损失计算
    - 模型参数更新
    - 训练监控和日志记录
    """
    
    def __init__(
        self,
        model: SVRasterModel,
        volume_renderer: VolumeRenderer,
        config: SVRasterTrainerConfig
    ):
        self.model = model
        self.volume_renderer = volume_renderer  # 紧密耦合的体积渲染器
        self.config = config
        
        # 确保模型处于训练模式
        self.model.train()
        
        # 设置优化器
        self.optimizer = self._setup_optimizer()
        self.scheduler = self._setup_scheduler()
        
        # 损失函数
        try:
            from src.nerfs.svraster.core import SVRasterLoss
            self.loss_fn = SVRasterLoss()
        except ImportError:
            # 如果导入失败，使用简单的 MSE 损失
            self.loss_fn = nn.MSELoss()
        
        # 训练状态
        self.current_epoch = 0
        self.global_step = 0
        
        logger.info(f"SVRasterTrainer initialized with VolumeRenderer coupling")
    
    def train_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """
        单步训练 - 使用体积渲染
        """
        self.optimizer.zero_grad()
        
        # 从批次中获取数据
        rays_o = batch['rays_o']  # [B, H, W, 3]
        rays_d = batch['rays_d']  # [B, H, W, 3]
        target_rgb = batch['target_rgb']  # [B, H, W, 3]
        
        # 使用体积渲染器进行前向传播
        # 这里体积渲染器与训练器紧密耦合
        with torch.cuda.amp.autocast(enabled=self.config.use_amp):
            render_result = self.volume_renderer.render(
                rays_o=rays_o,
                rays_d=rays_d,
                model=self.model,
                near=self.config.near_plane,
                far=self.config.far_plane,
                num_samples=self.config.num_samples,
                background_color=self.config.background_color
            )
            
            # 计算损失
            losses = self.loss_fn(render_result, target_rgb)
            total_loss = losses['total_loss']
        
        # 反向传播
        if self.config.use_amp:
            self.scaler.scale(total_loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            total_loss.backward()
            self.optimizer.step()
        
        # 更新学习率
        if self.scheduler is not None:
            self.scheduler.step()
        
        self.global_step += 1
        
        # 返回损失信息
        return {k: v.item() if isinstance(v, torch.Tensor) else v 
                for k, v in losses.items()}
    
    def train_epoch(self, dataloader: DataLoader) -> Dict[str, float]:
        """
        训练一个 epoch
        """
        self.model.train()
        epoch_losses = {}
        
        pbar = tqdm(dataloader, desc=f"Epoch {self.current_epoch}")
        for batch in pbar:
            # 将数据移到正确的设备
            batch = {k: v.to(self.model.device) if isinstance(v, torch.Tensor) else v 
                    for k, v in batch.items()}
            
            # 执行训练步骤
            step_losses = self.train_step(batch)
            
            # 累积损失
            for k, v in step_losses.items():
                if k not in epoch_losses:
                    epoch_losses[k] = []
                epoch_losses[k].append(v)
            
            # 更新进度条
            pbar.set_postfix({k: f"{v:.4f}" for k, v in step_losses.items()})
        
        # 计算平均损失
        avg_losses = {k: np.mean(v) for k, v in epoch_losses.items()}
        
        self.current_epoch += 1
        return avg_losses
    
    def _setup_optimizer(self) -> optim.Optimizer:
        """设置优化器"""
        if self.config.optimizer_type.lower() == 'adam':
            return optim.Adam(
                self.model.parameters(),
                lr=self.config.learning_rate,
                weight_decay=self.config.weight_decay
            )
        elif self.config.optimizer_type.lower() == 'adamw':
            return optim.AdamW(
                self.model.parameters(),
                lr=self.config.learning_rate,
                weight_decay=self.config.weight_decay
            )
        else:
            raise ValueError(f"Unsupported optimizer: {self.config.optimizer_type}")
    
    def _setup_scheduler(self) -> Optional[optim.lr_scheduler._LRScheduler]:
        """设置学习率调度器"""
        if self.config.scheduler_type is None:
            return None
        
        if self.config.scheduler_type.lower() == 'cosine':
            return optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=self.config.num_epochs
            )
        elif self.config.scheduler_type.lower() == 'step':
            return optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=self.config.scheduler_step_size,
                gamma=self.config.scheduler_gamma
            )
        else:
            return None


# ============================================================================
# 2. SVRasterRenderer - 与 TrueVoxelRasterizer 紧密耦合
# ============================================================================

class SVRasterRenderer:
    """
    SVRaster 渲染器 - 与 TrueVoxelRasterizer 紧密耦合
    
    专门负责：
    - 加载训练好的模型进行推理
    - 使用光栅化进行快速渲染
    - 生成新视点的图像
    - 支持批量渲染和实时渲染
    """
    
    def __init__(
        self,
        model: SVRasterModel,
        rasterizer: TrueVoxelRasterizer,
        config: SVRasterRendererConfig
    ):
        self.model = model
        self.rasterizer = rasterizer  # 紧密耦合的体素光栅化器
        self.config = config
        
        # 确保模型处于评估模式
        self.model.eval()
        
        logger.info(f"SVRasterRenderer initialized with TrueVoxelRasterizer coupling")
    
    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: str,
        rasterizer: TrueVoxelRasterizer,
        config: SVRasterRendererConfig
    ) -> SVRasterRenderer:
        """
        从检查点加载渲染器
        """
        # 加载检查点
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        
        # 创建模型
        try:
            from src.nerfs.svraster.core import SVRasterModel, SVRasterConfig
            model_config = checkpoint.get('model_config', SVRasterConfig())
            model = SVRasterModel(model_config)
        except ImportError:
            # 如果导入失败，创建占位符
            model = None
            logger.error("Could not import SVRasterModel and SVRasterConfig")
            
        if model is not None:
            model.load_state_dict(checkpoint['model_state_dict'])
        
        # 移到正确的设备
        if torch.cuda.is_available():
            model = model.cuda()
        
        return cls(model, rasterizer, config)
    
    def render_image(
        self,
        camera_pose: torch.Tensor,
        intrinsics: torch.Tensor,
        width: Optional[int] = None,
        height: Optional[int] = None
    ) -> Dict[str, torch.Tensor]:
        """
        渲染单张图像 - 使用光栅化
        """
        width = width or self.config.image_width
        height = height or self.config.image_height
        
        with torch.no_grad():
            # 生成光线
            rays_o, rays_d = self._generate_rays(camera_pose, intrinsics, width, height)
            
            # 使用光栅化器进行渲染
            # 这里光栅化器与渲染器紧密耦合
            render_result = self.rasterizer.render(
                rays_o=rays_o,
                rays_d=rays_d,
                model=self.model,
                background_color=self.config.background_color,
                chunk_size=self.config.render_chunk_size
            )
            
            # 重整形状为图像格式
            rgb = render_result['rgb'].view(height, width, 3)
            depth = render_result['depth'].view(height, width, 1)
            
            return {
                'rgb': rgb,
                'depth': depth,
                'raw_output': render_result
            }
    
    def render_batch(
        self,
        camera_poses: torch.Tensor,
        intrinsics: torch.Tensor,
        width: Optional[int] = None,
        height: Optional[int] = None
    ) -> List[Dict[str, torch.Tensor]]:
        """
        批量渲染多张图像
        """
        results = []
        
        for i in range(camera_poses.shape[0]):
            result = self.render_image(
                camera_poses[i],
                intrinsics[i] if intrinsics.ndim > 2 else intrinsics,
                width,
                height
            )
            results.append(result)
        
        return results
    
    def render_video(
        self,
        camera_trajectory: torch.Tensor,
        intrinsics: torch.Tensor,
        output_path: str,
        fps: int = 30,
        width: Optional[int] = None,
        height: Optional[int] = None
    ) -> None:
        """
        渲染视频序列
        """
        width = width or self.config.image_width
        height = height or self.config.image_height
        
        frames = []
        
        logger.info(f"Rendering {len(camera_trajectory)} frames for video...")
        
        for i, pose in enumerate(tqdm(camera_trajectory, desc="Rendering frames")):
            result = self.render_image(pose, intrinsics, width, height)
            
            # 转换为 numpy 格式 (0-255)
            rgb_np = (result['rgb'].cpu().numpy() * 255).astype(np.uint8)
            frames.append(rgb_np)
        
        # 保存视频
        if imageio is not None:
            imageio.mimsave(output_path, frames, fps=fps)
            logger.info(f"Video saved to {output_path}")
        else:
            logger.error("imageio not available for video saving")
    
    def _generate_rays(
        self,
        camera_pose: torch.Tensor,
        intrinsics: torch.Tensor,
        width: int,
        height: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        根据相机参数生成光线
        """
        device = camera_pose.device
        
        # 生成像素坐标
        i, j = torch.meshgrid(
            torch.linspace(0, width - 1, width, device=device),
            torch.linspace(0, height - 1, height, device=device),
            indexing='ij'
        )
        i = i.t()
        j = j.t()
        
        # 提取内参
        fx, fy = intrinsics[0, 0], intrinsics[1, 1]
        cx, cy = intrinsics[0, 2], intrinsics[1, 2]
        
        # 计算归一化设备坐标
        dirs = torch.stack([
            (i - cx) / fx,
            -(j - cy) / fy,
            -torch.ones_like(i)
        ], -1)
        
        # 转换到世界坐标系
        rotation = camera_pose[:3, :3]
        translation = camera_pose[:3, 3]
        
        rays_d = torch.sum(dirs[..., None, :] * rotation, -1)
        rays_o = translation.expand(rays_d.shape)
        
        return rays_o.reshape(-1, 3), rays_d.reshape(-1, 3)


# ============================================================================
# 3. 配置类
# ============================================================================

@dataclass
class SVRasterTrainerConfig:
    """SVRaster 训练器配置 - 专门为体积渲染训练设计"""
    
    # 训练参数
    num_epochs: int = 100
    batch_size: int = 1
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    
    # 优化器设置
    optimizer_type: str = "adam"
    scheduler_type: str = "cosine"
    scheduler_step_size: int = 30
    scheduler_gamma: float = 0.1
    
    # 体积渲染参数（与 VolumeRenderer 紧密相关）
    num_samples: int = 64
    num_importance_samples: int = 128
    near_plane: float = 0.1
    far_plane: float = 100.0
    background_color: Tuple[float, float, float] = (1.0, 1.0, 1.0)
    
    # 训练设置
    use_amp: bool = True
    grad_clip_norm: Optional[float] = 1.0
    
    # 日志和检查点
    log_every: int = 100
    save_every: int = 1000
    validate_every: int = 500


@dataclass  
class SVRasterRendererConfig:
    """SVRaster 渲染器配置 - 专门为光栅化推理设计"""
    
    # 渲染质量设置
    image_width: int = 800
    image_height: int = 600
    render_batch_size: int = 4096
    render_chunk_size: int = 1024
    
    # 光栅化参数（与 TrueVoxelRasterizer 紧密相关）
    background_color: Tuple[float, float, float] = (1.0, 1.0, 1.0)
    use_alpha_blending: bool = True
    depth_threshold: float = 1e-6
    
    # 输出设置
    output_format: str = "png"
    save_depth: bool = False
    save_alpha: bool = False


# ============================================================================
# 4. 使用示例
# ============================================================================

def example_training_pipeline():
    """
    训练流程示例 - 展示 SVRasterTrainer 与 VolumeRenderer 的耦合使用
    """
    from .core import SVRasterModel, SVRasterConfig
    from .volume_renderer import VolumeRenderer
    from .dataset import SVRasterDataset, SVRasterDatasetConfig
    
    # 1. 创建模型和体积渲染器
    model_config = SVRasterConfig()
    model = SVRasterModel(model_config)
    
    volume_renderer = VolumeRenderer()  # 专门用于训练的体积渲染器
    
    # 2. 创建训练器（与体积渲染器紧密耦合）
    trainer_config = SVRasterTrainerConfig()
    trainer = SVRasterTrainer(model, volume_renderer, trainer_config)
    
    # 3. 创建数据集
    dataset_config = SVRasterDatasetConfig()
    dataset = SVRasterDataset(dataset_config)
    dataloader = DataLoader(dataset, batch_size=trainer_config.batch_size)
    
    # 4. 训练循环
    for epoch in range(trainer_config.num_epochs):
        epoch_losses = trainer.train_epoch(dataloader)
        
        print(f"Epoch {epoch}: {epoch_losses}")
        
        # 保存检查点
        if epoch % 10 == 0:
            checkpoint = {
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': trainer.optimizer.state_dict(),
                'model_config': model_config,
                'epoch': epoch
            }
            torch.save(checkpoint, f"checkpoint_epoch_{epoch}.pth")


def example_inference_pipeline():
    """
    推理流程示例 - 展示 SVRasterRenderer 与 TrueVoxelRasterizer 的耦合使用
    """
    from .true_rasterizer import TrueVoxelRasterizer
    
    # 1. 创建光栅化器
    rasterizer = TrueVoxelRasterizer()  # 专门用于推理的光栅化器
    
    # 2. 从检查点加载渲染器（与光栅化器紧密耦合）
    renderer_config = SVRasterRendererConfig()
    renderer = SVRasterRenderer.from_checkpoint(
        "checkpoint_epoch_100.pth",
        rasterizer,
        renderer_config
    )
    
    # 3. 设置相机参数
    camera_pose = torch.eye(4)
    intrinsics = torch.tensor([
        [800, 0, 400],
        [0, 800, 300],
        [0, 0, 1]
    ], dtype=torch.float32)
    
    # 4. 渲染图像
    result = renderer.render_image(camera_pose, intrinsics)
    
    # 5. 保存结果
    rgb_image = result['rgb'].cpu().numpy()
    depth_image = result['depth'].cpu().numpy()
    
    print(f"Rendered image shape: {rgb_image.shape}")
    print(f"Depth image shape: {depth_image.shape}")


if __name__ == "__main__":
    print("SVRaster 架构重构完成！")
    print("\n耦合设计：")
    print("1. SVRasterTrainer ↔ VolumeRenderer (训练阶段)")
    print("2. SVRasterRenderer ↔ TrueVoxelRasterizer (推理阶段)")
    print("\n运行示例:")
    print("- example_training_pipeline(): 展示训练流程")
    print("- example_inference_pipeline(): 展示推理流程")
