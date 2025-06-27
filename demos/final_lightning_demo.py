from typing import Any
#!/usr/bin/env python3
"""
PyTorch Lightning NeRF项目最终演示

这个脚本展示了项目中PyTorch Lightning的完整集成和功能
"""

import torch
import torch.nn.functional as F
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger
import torchmetrics
from dataclasses import dataclass
import numpy as np
import sys
import os

# 添加项目根目录到Python路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

@dataclass
class NeRFLightningConfig:
    """PyTorch Lightning NeRF配置"""
    hidden_dim: int = 128
    num_layers: int = 4
    learning_rate: float = 5e-4
    pe_freq: int = 10  # 位置编码频率
    scheduler_type: str = "cosine"
    use_ema: bool = True
    ema_decay: float = 0.999

class NeRFLightningModule(pl.LightningModule):
    """PyTorch Lightning NeRF模块"""
    
    def __init__(self, config: NeRFLightningConfig):
        super().__init__()
        self.config = config
        self.save_hyperparameters()
        
        # 构建网络
        layers = []
        input_dim = 3 * 2 * config.pe_freq + 3  # PE后的输入维度
        
        for i in range(config.num_layers):
            if i == 0:
                layers.append(torch.nn.Linear(input_dim, config.hidden_dim))
            else:
                layers.append(torch.nn.Linear(config.hidden_dim, config.hidden_dim))
            layers.append(torch.nn.ReLU())
        
        # 输出层：密度和颜色
        layers.append(torch.nn.Linear(config.hidden_dim, 4))  # 1密度 + 3颜色
        
        self.network = torch.nn.Sequential(*layers)
        
        # 初始化指标
        self.train_psnr = torchmetrics.image.PeakSignalNoiseRatio()
        self.val_psnr = torchmetrics.image.PeakSignalNoiseRatio()
        self.val_ssim = torchmetrics.image.StructuralSimilarityIndexMeasure()
        
        # EMA模型
        if config.use_ema:
            # 创建EMA模型的副本
            self.ema_model = torch.nn.Sequential(*layers)
            self.ema_model.load_state_dict(self.network.state_dict())
            self.ema_model.eval()
            for param in self.ema_model.parameters():
                param.requires_grad_(False)
        else:
            self.ema_model = None
        
    def positional_encoding(self, x: torch.Tensor) -> torch.Tensor:
        """位置编码"""
        encoded = []
        for freq in range(self.config.pe_freq):
            for func in [torch.sin, torch.cos]:
                encoded.append(func(x * (2.0 ** freq)))
        encoded.append(x)  # 原始坐标
        return torch.cat(encoded, dim=-1)
    
    def forward(self, positions: torch.Tensor) -> dict[str, torch.Tensor]:
        """前向传播"""
        # 位置编码
        encoded_pos = self.positional_encoding(positions)
        
        # 通过网络
        output = self.network(encoded_pos)
        
        # 分离密度和颜色
        density = torch.relu(output[..., 0])  # 密度 >= 0
        color = torch.sigmoid(output[..., 1:])  # 颜色 [0, 1]
        
        return {
            'density': density, 'color': color
        }
    
    def training_step(self, batch: dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        """训练步骤"""
        positions = batch['positions']  # [N, 3]
        target_colors = batch['colors']  # [N, 3]
        
        # 前向传播
        outputs = self(positions)
        
        # 计算损失
        color_loss = F.mse_loss(outputs['color'], target_colors)
        
        # 计算PSNR
        psnr = self.train_psnr(outputs['color'], target_colors)
        
        # 记录指标
        self.log('train/loss', color_loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log('train/psnr', psnr, on_step=True, on_epoch=True, prog_bar=True)
        
        return color_loss
    
    def validation_step(
        self,
        batch: dict[str,
        torch.Tensor],
        batch_idx: int,
    )
        """验证步骤"""
        positions = batch['positions']
        target_colors = batch['colors']
        
        # 使用EMA模型进行验证（如果可用）
        if self.ema_model is not None:
            encoded_pos = self.positional_encoding(positions)
            output = self.ema_model(encoded_pos)
            density = torch.relu(output[..., 0])
            color = torch.sigmoid(output[..., 1:])
            outputs = {'density': density, 'color': color}
        else:
            outputs = self(positions)
        
        # 计算损失
        val_loss = F.mse_loss(outputs['color'], target_colors)
        
        # 计算指标
        psnr = self.val_psnr(outputs['color'], target_colors)
        ssim = self.val_ssim(
            outputs['color'].unsqueeze,
        )
        
        # 记录验证指标
        self.log('val/loss', val_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log('val/psnr', psnr, on_step=False, on_epoch=True, prog_bar=True)
        self.log('val/ssim', ssim, on_step=False, on_epoch=True, prog_bar=True)
        
        return {
            'val_loss': val_loss, 'val_psnr': psnr, 'val_ssim': ssim
        }
    
    def on_train_epoch_end(self):
        """训练轮次结束时更新EMA"""
        if self.ema_model is not None:
            self._update_ema()
    
    def _update_ema(self):
        """更新EMA模型"""
        decay = self.config.ema_decay
        with torch.no_grad():
            for ema_param, model_param in zip(
                self.ema_model.parameters,
            )
                ema_param.mul_(decay).add_(model_param, alpha=1 - decay)
    
    def configure_optimizers(self):
        """配置优化器和调度器"""
        optimizer = torch.optim.AdamW(
            self.parameters(), lr=self.config.learning_rate, weight_decay=1e-4
        )
        
        if self.config.scheduler_type == "cosine":
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=100, eta_min=1e-6
            )
            return {
                "optimizer": optimizer, "lr_scheduler": {
                    "scheduler": scheduler, "interval": "epoch"
                }
            }
        else:
            return optimizer

class MockNeRFDataset(torch.utils.data.Dataset):
    """模拟NeRF数据集"""
    
    def __init__(self, size: int = 1000):
        self.size = size
        
    def __len__(self):
        return self.size
    
    def __getitem__(self, idx):
        # 生成随机3D位置和对应的颜色
        position = torch.randn(3) * 2  # [-2, 2] 范围内的位置
        
        # 简单的颜色函数：基于位置生成颜色
        color = torch.sigmoid(position)  # 将位置映射到[0, 1]颜色
        
        return {
            'positions': position, 'colors': color
        }

def demonstrate_complete_lightning():
    """演示完整的PyTorch Lightning功能"""
    
    print("🌟 PyTorch Lightning NeRF 完整功能演示")
    print("=" * 60)
    
    # 检查设备
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"🖥️  使用设备: {device}")
    
    # 1. 创建配置
    config = NeRFLightningConfig(
        hidden_dim=128, num_layers=4, learning_rate=1e-3, pe_freq=8, scheduler_type="cosine", use_ema=True, ema_decay=0.999
    )
    
    print(f"⚙️  模型配置: {config}")
    
    # 2. 创建模型
    model = NeRFLightningModule(config)
    param_count = sum(p.numel() for p in model.parameters())
    print(f"🧠 模型参数数量: {param_count:, }")
    
    # 3. 创建数据集和数据加载器
    train_dataset = MockNeRFDataset(1000)
    val_dataset = MockNeRFDataset(200)
    
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=64, shuffle=True, num_workers=0
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=64, shuffle=False, num_workers=0
    )
    
    print(f"📊 训练数据: {len(train_dataset)} 样本")
    print(f"📊 验证数据: {len(val_dataset)} 样本")
    
    # 4. 创建回调函数
    callbacks = [
        # 模型检查点
        ModelCheckpoint(
            dirpath="checkpoints/final_demo", filename="nerf-{
                epoch:02d,
            }
        ), # 早停
        EarlyStopping(
            monitor="val/psnr", mode="max", patience=20, min_delta=0.001
        ), # 学习率监控
        LearningRateMonitor(
            logging_interval="epoch"
        )
    ]
    
    # 5. 创建日志记录器
    logger = TensorBoardLogger(
        save_dir="logs", name="final_nerf_demo", version="v1.0"
    )
    
    # 6. 创建训练器
    trainer = pl.Trainer(
        max_epochs=50, devices=1, accelerator="auto", precision="16-mixed" if device == "cuda" else "32", callbacks=callbacks, logger=logger, log_every_n_steps=10, val_check_interval=0.5, # 每半个epoch验证一次
        enable_checkpointing=True, enable_progress_bar=True, enable_model_summary=True
    )
    
    print("\n🚀 开始训练...")
    print(f"📅 最大轮次: {trainer.max_epochs}")
    print(f"🎯 精度: {trainer.precision}")
    print(f"📝 日志保存到: logs/final_nerf_demo")
    print(f"💾 检查点保存到: checkpoints/final_demo")
    
    # 7. 开始训练
    trainer.fit(model, train_loader, val_loader)
    
    # 8. 显示结果
    print("\n" + "=" * 60)
    print("✅ 训练完成!")
    
    if trainer.callback_metrics:
        metrics = trainer.callback_metrics
        print(f"📈 最终验证PSNR: {metrics.get('val/psnr', 'N/A'):.3f}")
        print(f"📈 最终验证SSIM: {metrics.get('val/ssim', 'N/A'):.3f}")
        print(f"📉 最终验证损失: {metrics.get('val/loss', 'N/A'):.6f}")
    
    # 9. 保存最终模型
    final_model_path = "checkpoints/final_demo/final_model.ckpt"
    trainer.save_checkpoint(final_model_path)
    print(f"💾 最终模型保存到: {final_model_path}")
    
    print("\n🎉 PyTorch Lightning演示完成!")
    print("\n📋 总结:")
    print("  ✅ 自动混合精度训练")
    print("  ✅ EMA模型更新")
    print("  ✅ 自动检查点保存")
    print("  ✅ 早停机制")
    print("  ✅ 学习率调度")
    print("  ✅ TensorBoard日志记录")
    print("  ✅ 多种评估指标")
    print("  ✅ 进度条和模型摘要")
    
    print(f"\n📊 查看训练日志: tensorboard --logdir logs")
    
    return model, trainer

if __name__ == '__main__':
    print("启动PyTorch Lightning NeRF完整演示...")
    model, trainer = demonstrate_complete_lightning()
    print("演示完成!") 