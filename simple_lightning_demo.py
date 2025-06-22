"""
简化的PyTorch Lightning NeRF演示

展示如何使用PyTorch Lightning训练NeRF模型的基本概念和优势。
"""

import torch
import torch.nn.functional as F
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger
import torchmetrics
from typing import Dict, Any
from dataclasses import dataclass
import numpy as np


@dataclass
class SimpleNeRFConfig:
    """简化的NeRF配置"""
    hidden_dim: int = 128
    num_layers: int = 4
    learning_rate: float = 5e-4
    pe_freq: int = 10  # 位置编码频率


class SimpleNeRF(pl.LightningModule):
    """简化的NeRF模型，用于演示PyTorch Lightning"""
    
    def __init__(self, config: SimpleNeRFConfig):
        super().__init__()
        self.config = config
        self.save_hyperparameters()
        
        # 简单的MLP网络
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
        
    def positional_encoding(self, x: torch.Tensor) -> torch.Tensor:
        """位置编码"""
        encoded = []
        for freq in range(self.config.pe_freq):
            for func in [torch.sin, torch.cos]:
                encoded.append(func(x * (2.0 ** freq)))
        encoded.append(x)  # 原始坐标
        return torch.cat(encoded, dim=-1)
    
    def forward(self, positions: torch.Tensor) -> Dict[str, torch.Tensor]:
        """前向传播"""
        # 位置编码
        encoded_pos = self.positional_encoding(positions)
        
        # 通过网络
        output = self.network(encoded_pos)
        
        # 分离密度和颜色
        density = torch.relu(output[..., 0])  # 密度 >= 0
        color = torch.sigmoid(output[..., 1:])  # 颜色 [0, 1]
        
        return {
            'density': density,
            'color': color
        }
    
    def training_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
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
    
    def validation_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> Dict[str, torch.Tensor]:
        """验证步骤"""
        positions = batch['positions']
        target_colors = batch['colors']
        
        # 前向传播
        outputs = self(positions)
        
        # 计算损失
        val_loss = F.mse_loss(outputs['color'], target_colors)
        
        # 计算PSNR
        psnr = self.val_psnr(outputs['color'], target_colors)
        
        # 记录验证指标
        self.log('val/loss', val_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log('val/psnr', psnr, on_step=False, on_epoch=True, prog_bar=True)
        
        return {
            'val_loss': val_loss,
            'val_psnr': psnr
        }
    
    def on_validation_epoch_end(self):
        """验证轮次结束 - PyTorch Lightning 2.0兼容"""
        # 自动计算和记录累积指标
        pass  # 指标会自动累积和记录
    
    def configure_optimizers(self):
        """配置优化器和调度器"""
        optimizer = torch.optim.Adam(
            self.parameters(),
            lr=self.config.learning_rate
        )
        
        scheduler = torch.optim.lr_scheduler.ExponentialLR(
            optimizer, gamma=0.95
        )
        
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch"
            }
        }


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
        color = torch.sigmoid(position)  # 将位置映射到[0,1]颜色
        
        return {
            'positions': position,
            'colors': color
        }


def demonstrate_lightning_advantages():
    """演示PyTorch Lightning的优势"""
    
    print("🌟 PyTorch Lightning NeRF 训练演示")
    print("=" * 50)
    
    # 1. 创建配置
    config = SimpleNeRFConfig(
        hidden_dim=64,
        num_layers=3,
        learning_rate=1e-3,
        pe_freq=6
    )
    
    # 2. 创建模型
    model = SimpleNeRF(config)
    
    # 3. 创建数据集和数据加载器
    train_dataset = MockNeRFDataset(800)
    val_dataset = MockNeRFDataset(200)
    
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=64, shuffle=True, num_workers=2
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=64, shuffle=False, num_workers=2
    )
    
    # 4. 创建回调函数
    callbacks = [
        ModelCheckpoint(
            dirpath="checkpoints/simple_nerf",
            filename="best-{epoch:02d}-{val/psnr:.2f}",
            monitor="val/psnr",
            mode="max",
            save_top_k=3
        ),
        EarlyStopping(
            monitor="val/psnr",
            mode="max",
            patience=20,
            verbose=True
        ),
        LearningRateMonitor(logging_interval="epoch")
    ]
    
    # 5. 创建日志记录器
    logger = TensorBoardLogger(
        save_dir="logs",
        name="simple_nerf",
        version="demo"
    )
    
    # 6. 创建训练器（这里展示Lightning的强大功能）
    trainer = pl.Trainer(
        max_epochs=50,
        devices=1,  # 使用1个GPU（如果可用）
        accelerator="auto",  # 自动选择硬件
        precision="16-mixed",  # 混合精度训练
        callbacks=callbacks,
        logger=logger,
        gradient_clip_val=1.0,  # 梯度裁剪
        log_every_n_steps=10,
        val_check_interval=0.5,  # 每半个epoch验证一次
        enable_progress_bar=True,
        enable_model_summary=True
    )
    
    # 7. 展示Lightning的功能
    print("\n🚀 PyTorch Lightning 提供的功能:")
    print("✅ 自动混合精度训练 (16-bit)")
    print("✅ 自动梯度裁剪")
    print("✅ 自动检查点保存")
    print("✅ 早停机制")
    print("✅ 学习率监控")
    print("✅ TensorBoard集成")
    print("✅ 进度条和模型摘要")
    print("✅ 自动验证循环")
    print("✅ GPU/CPU自动选择")
    
    print(f"\n📊 模型参数: {sum(p.numel() for p in model.parameters()):,}")
    print(f"📁 检查点保存到: checkpoints/simple_nerf/")
    print(f"📈 日志保存到: logs/simple_nerf/demo/")
    
    # 8. 开始训练
    print("\n🎯 开始训练...")
    try:
        trainer.fit(model, train_loader, val_loader)
        
        # 9. 训练完成后的信息
        print(f"\n✅ 训练完成!")
        print(f"📈 最终验证PSNR: {model.val_psnr.compute():.2f} dB")
        print(f"💾 最佳模型保存在: {trainer.checkpoint_callback.best_model_path}")
        
    except KeyboardInterrupt:
        print("\n⏹️  训练被用户中断")
    except Exception as e:
        print(f"\n❌ 训练出错: {e}")
    
    print("\n🎉 演示完成!")
    print("\n📝 使用 PyTorch Lightning 的主要好处:")
    print("1. 代码更简洁，专注于模型逻辑")
    print("2. 自动化的训练循环和最佳实践")
    print("3. 丰富的回调和日志系统")
    print("4. 轻松的分布式训练支持")
    print("5. 自动化的GPU内存管理")
    print("6. 标准化的检查点和恢复机制")
    
    return model, trainer


def compare_traditional_vs_lightning():
    """对比传统训练方式和Lightning方式的代码量"""
    
    print("\n📊 代码量对比:")
    print("=" * 50)
    
    traditional_lines = '''
传统PyTorch训练代码:
- 训练循环: ~50行
- 验证循环: ~30行 
- 检查点保存/加载: ~20行
- 日志记录: ~15行
- GPU处理: ~10行
- 指标计算: ~15行
- 总计: ~140行样板代码
'''
    
    lightning_lines = '''
PyTorch Lightning代码:
- training_step: ~10行
- validation_step: ~8行
- configure_optimizers: ~5行
- 其他配置: ~5行
- 总计: ~28行核心代码
'''
    
    print(traditional_lines)
    print(lightning_lines)
    print("💡 Lightning减少了约80%的样板代码!")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="PyTorch Lightning NeRF Demo")
    parser.add_argument("--mode", type=str, default="demo", 
                       choices=["demo", "compare"], 
                       help="运行模式")
    
    args = parser.parse_args()
    
    if args.mode == "demo":
        demonstrate_lightning_advantages()
    elif args.mode == "compare":
        compare_traditional_vs_lightning()
    
    print("\n查看训练日志:")
    print("tensorboard --logdir logs") 