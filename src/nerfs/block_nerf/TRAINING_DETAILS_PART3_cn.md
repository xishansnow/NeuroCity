# Block-NeRF 训练机制详解 - 第三部分

## 实际训练脚本

### 主训练脚本

```python
#!/usr/bin/env python3
"""
Block-NeRF 主训练脚本
用法: python train_block_nerf.py --config configs/block_nerf_config.yaml
"""

import argparse
import os
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.utils.tensorboard import SummaryWriter
import yaml
from tqdm import tqdm
import numpy as np

from src.nerfs.block_nerf import (
    BlockNeRF, BlockManager, BlockNeRFDataset,
    AppearanceEmbedding, VisibilityNetwork, PoseRefinementNetwork
)
from src.nerfs.block_nerf.trainer import BlockNeRFTrainer
from src.nerfs.block_nerf.utils import setup_logger, seed_everything

def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='Block-NeRF Training')
    parser.add_argument('--config', type=str, required=True,
                       help='训练配置文件路径')
    parser.add_argument('--data_root', type=str, required=True,
                       help='数据集根目录')
    parser.add_argument('--output_dir', type=str, default='./outputs',
                       help='输出目录')
    parser.add_argument('--resume', type=str, default=None,
                       help='恢复训练的检查点路径')
    parser.add_argument('--distributed', action='store_true',
                       help='启用分布式训练')
    parser.add_argument('--local_rank', type=int, default=0,
                       help='本地rank (分布式训练)')
    parser.add_argument('--debug', action='store_true',
                       help='调试模式')
    
    return parser.parse_args()

def load_config(config_path):
    """加载训练配置"""
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config

def setup_distributed(args):
    """设置分布式训练"""
    if args.distributed:
        # 初始化分布式环境
        dist.init_process_group(
            backend='nccl',
            init_method='env://',
            world_size=int(os.environ['WORLD_SIZE']),
            rank=int(os.environ['RANK'])
        )
        
        # 设置CUDA设备
        torch.cuda.set_device(args.local_rank)
        
        return int(os.environ['RANK']), int(os.environ['WORLD_SIZE'])
    else:
        return 0, 1

def create_models(config, device):
    """创建所有模型"""
    models = {}
    
    # 块管理器
    block_manager = BlockManager(config['block_manager'])
    models['block_manager'] = block_manager
    
    # 主 NeRF 模型
    block_nerf = BlockNeRF(config['block_nerf'])
    models['block_nerf'] = block_nerf.to(device)
    
    # 外观嵌入
    if config['training']['use_appearance_embedding']:
        appearance_embedding = AppearanceEmbedding(config['appearance_embedding'])
        models['appearance_embedding'] = appearance_embedding.to(device)
    
    # 可见性网络
    if config['training']['use_visibility_network']:
        visibility_network = VisibilityNetwork(config['visibility_network'])
        models['visibility_network'] = visibility_network.to(device)
    
    # 姿态细化网络
    if config['training']['use_pose_refinement']:
        pose_refinement = PoseRefinementNetwork(config['pose_refinement'])
        models['pose_refinement'] = pose_refinement.to(device)
    
    return models

def create_optimizers(models, config):
    """创建优化器"""
    optimizers = {}
    
    # NeRF 优化器
    nerf_params = list(models['block_nerf'].parameters())
    optimizers['nerf'] = torch.optim.Adam(
        nerf_params,
        lr=config['training']['nerf_lr'],
        betas=(0.9, 0.999),
        weight_decay=config['training']['weight_decay']
    )
    
    # 外观嵌入优化器
    if 'appearance_embedding' in models:
        appearance_params = list(models['appearance_embedding'].parameters())
        optimizers['appearance'] = torch.optim.Adam(
            appearance_params,
            lr=config['training']['appearance_lr'],
            betas=(0.9, 0.999)
        )
    
    # 可见性网络优化器
    if 'visibility_network' in models:
        visibility_params = list(models['visibility_network'].parameters())
        optimizers['visibility'] = torch.optim.Adam(
            visibility_params,
            lr=config['training']['visibility_lr'],
            betas=(0.9, 0.999)
        )
    
    # 姿态细化优化器
    if 'pose_refinement' in models:
        pose_params = list(models['pose_refinement'].parameters())
        optimizers['pose'] = torch.optim.SGD(
            pose_params,
            lr=config['training']['pose_lr'],
            momentum=0.9
        )
    
    return optimizers

def main():
    """主训练函数"""
    args = parse_args()
    
    # 设置分布式
    rank, world_size = setup_distributed(args)
    is_main_process = rank == 0
    
    # 加载配置
    config = load_config(args.config)
    
    # 设置随机种子
    seed_everything(config['training']['seed'] + rank)
    
    # 设置日志
    if is_main_process:
        logger = setup_logger(os.path.join(args.output_dir, 'train.log'))
        writer = SummaryWriter(os.path.join(args.output_dir, 'tensorboard'))
    else:
        logger = None
        writer = None
    
    # 设置设备
    device = torch.device(f'cuda:{args.local_rank}' if torch.cuda.is_available() else 'cpu')
    
    if is_main_process:
        logger.info(f"开始 Block-NeRF 训练")
        logger.info(f"配置: {args.config}")
        logger.info(f"数据根目录: {args.data_root}")
        logger.info(f"输出目录: {args.output_dir}")
        logger.info(f"设备: {device}")
        logger.info(f"分布式训练: {args.distributed} (rank {rank}/{world_size})")
    
    # 创建数据集
    train_dataset = BlockNeRFDataset(
        data_root=args.data_root,
        split='train',
        config=config['dataset']
    )
    
    val_dataset = BlockNeRFDataset(
        data_root=args.data_root,
        split='val',
        config=config['dataset']
    )
    
    if is_main_process:
        logger.info(f"训练数据: {len(train_dataset)} 样本")
        logger.info(f"验证数据: {len(val_dataset)} 样本")
    
    # 创建模型
    models = create_models(config, device)
    
    # 创建优化器
    optimizers = create_optimizers(models, config)
    
    # 设置分布式数据并行
    if args.distributed:
        for name, model in models.items():
            if isinstance(model, nn.Module):
                models[name] = torch.nn.parallel.DistributedDataParallel(
                    model,
                    device_ids=[args.local_rank],
                    output_device=args.local_rank,
                    find_unused_parameters=True
                )
    
    # 创建训练器
    trainer = BlockNeRFTrainer(
        models=models,
        optimizers=optimizers,
        config=config['training'],
        device=device,
        logger=logger,
        writer=writer,
        is_main_process=is_main_process
    )
    
    # 恢复训练
    start_epoch = 0
    if args.resume:
        start_epoch = trainer.load_checkpoint(args.resume)
        if is_main_process:
            logger.info(f"从epoch {start_epoch}恢复训练")
    
    # 开始训练
    try:
        trainer.train(
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            start_epoch=start_epoch,
            total_epochs=config['training']['num_epochs']
        )
    except KeyboardInterrupt:
        if is_main_process:
            logger.info("训练被用户中断")
    except Exception as e:
        if is_main_process:
            logger.error(f"训练过程中出现错误: {e}")
        raise
    finally:
        # 清理资源
        if args.distributed:
            dist.destroy_process_group()
        
        if is_main_process:
            logger.info("训练完成")
            writer.close()

if __name__ == '__main__':
    main()
```

### 训练器实现

```python
class BlockNeRFTrainer:
    """Block-NeRF 训练器"""
    
    def __init__(self, models, optimizers, config, device, logger=None, writer=None, is_main_process=True):
        self.models = models
        self.optimizers = optimizers
        self.config = config
        self.device = device
        self.logger = logger
        self.writer = writer
        self.is_main_process = is_main_process
        
        # 损失函数
        self.criterion = BlockNeRFLoss(config)
        
        # 学习率调度器
        self.schedulers = self.create_schedulers()
        
        # 混合精度训练
        if config.get('use_mixed_precision', False):
            self.scaler = torch.cuda.amp.GradScaler()
        else:
            self.scaler = None
        
        # 训练状态
        self.global_step = 0
        self.best_psnr = 0.0
        
    def create_schedulers(self):
        """创建学习率调度器"""
        schedulers = {}
        
        for name, optimizer in self.optimizers.items():
            if name == 'nerf':
                schedulers[name] = torch.optim.lr_scheduler.ExponentialLR(
                    optimizer, gamma=0.1 ** (1 / self.config['lr_decay_steps'])
                )
            elif name == 'appearance':
                schedulers[name] = torch.optim.lr_scheduler.CosineAnnealingLR(
                    optimizer, T_max=self.config['num_epochs']
                )
            else:
                schedulers[name] = torch.optim.lr_scheduler.StepLR(
                    optimizer, step_size=50, gamma=0.5
                )
        
        return schedulers
    
    def train_epoch(self, train_dataset, epoch):
        """训练一个epoch"""
        # 设置为训练模式
        for model in self.models.values():
            if isinstance(model, nn.Module):
                model.train()
        
        # 创建数据加载器
        if hasattr(train_dataset, 'distributed_sampler'):
            train_dataset.distributed_sampler.set_epoch(epoch)
        
        dataloader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=self.config['batch_size'],
            shuffle=not hasattr(train_dataset, 'distributed_sampler'),
            sampler=getattr(train_dataset, 'distributed_sampler', None),
            num_workers=self.config.get('num_workers', 4),
            pin_memory=True
        )
        
        epoch_losses = defaultdict(float)
        num_batches = len(dataloader)
        
        # 进度条
        if self.is_main_process:
            pbar = tqdm(dataloader, desc=f'Epoch {epoch}')
        else:
            pbar = dataloader
        
        for batch_idx, batch in enumerate(pbar):
            # 移动数据到设备
            batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                    for k, v in batch.items()}
            
            # 训练步骤
            losses = self.train_step(batch)
            
            # 累积损失
            for loss_name, loss_value in losses.items():
                epoch_losses[loss_name] += loss_value
            
            # 更新进度条
            if self.is_main_process:
                pbar.set_postfix({k: f'{v:.6f}' for k, v in losses.items()})
            
            # 记录到tensorboard
            if self.writer and self.global_step % self.config.get('log_freq', 100) == 0:
                for loss_name, loss_value in losses.items():
                    self.writer.add_scalar(f'train/{loss_name}', loss_value, self.global_step)
            
            self.global_step += 1
        
        # 平均损失
        avg_losses = {k: v / num_batches for k, v in epoch_losses.items()}
        
        return avg_losses
    
    def train_step(self, batch):
        """单步训练"""
        # 清零梯度
        for optimizer in self.optimizers.values():
            optimizer.zero_grad()
        
        # 前向传播
        if self.scaler:
            with torch.cuda.amp.autocast():
                predictions = self.forward_models(batch)
                total_loss, losses = self.criterion(predictions, batch)
        else:
            predictions = self.forward_models(batch)
            total_loss, losses = self.criterion(predictions, batch)
        
        # 反向传播
        if self.scaler:
            self.scaler.scale(total_loss).backward()
            
            # 梯度裁剪
            if self.config.get('gradient_clip_norm', 0) > 0:
                for optimizer in self.optimizers.values():
                    self.scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        self.get_parameters(optimizer), 
                        self.config['gradient_clip_norm']
                    )
            
            # 更新参数
            for optimizer in self.optimizers.values():
                self.scaler.step(optimizer)
            
            self.scaler.update()
        else:
            total_loss.backward()
            
            # 梯度裁剪
            if self.config.get('gradient_clip_norm', 0) > 0:
                for optimizer in self.optimizers.values():
                    torch.nn.utils.clip_grad_norm_(
                        self.get_parameters(optimizer),
                        self.config['gradient_clip_norm']
                    )
            
            # 更新参数
            for optimizer in self.optimizers.values():
                optimizer.step()
        
        return {k: v.item() for k, v in losses.items()}
    
    def forward_models(self, batch):
        """模型前向传播"""
        predictions = {}
        
        # 提取输入
        rays_o = batch['rays_o']
        rays_d = batch['rays_d']
        
        # Block-NeRF 渲染
        if 'appearance_embedding' in self.models:
            appearance_features = self.models['appearance_embedding'](
                batch.get('image_indices', None)
            )
        else:
            appearance_features = None
        
        # 主要渲染
        nerf_output = self.models['block_nerf'](
            rays_o, rays_d, 
            appearance_features=appearance_features
        )
        
        predictions.update(nerf_output)
        
        return predictions
    
    def validate(self, val_dataset):
        """验证"""
        # 设置为评估模式
        for model in self.models.values():
            if isinstance(model, nn.Module):
                model.eval()
        
        val_losses = defaultdict(float)
        val_metrics = defaultdict(list)
        
        dataloader = torch.utils.data.DataLoader(
            val_dataset,
            batch_size=self.config['val_batch_size'],
            shuffle=False,
            num_workers=self.config.get('num_workers', 4)
        )
        
        with torch.no_grad():
            for batch in tqdm(dataloader, desc='Validation', disable=not self.is_main_process):
                # 移动数据到设备
                batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                        for k, v in batch.items()}
                
                # 前向传播
                predictions = self.forward_models(batch)
                
                # 计算损失
                total_loss, losses = self.criterion(predictions, batch)
                
                for loss_name, loss_value in losses.items():
                    val_losses[loss_name] += loss_value.item()
                
                # 计算指标
                metrics = self.compute_metrics(predictions, batch)
                for metric_name, metric_value in metrics.items():
                    val_metrics[metric_name].append(metric_value)
        
        # 平均损失和指标
        avg_losses = {k: v / len(dataloader) for k, v in val_losses.items()}
        avg_metrics = {k: np.mean(v) for k, v in val_metrics.items()}
        
        return avg_losses, avg_metrics
    
    def compute_metrics(self, predictions, targets):
        """计算评估指标"""
        metrics = {}
        
        pred_rgb = predictions['rgb']
        target_rgb = targets['rgb']
        
        # PSNR
        mse = torch.mean((pred_rgb - target_rgb) ** 2)
        psnr = -10 * torch.log10(mse)
        metrics['psnr'] = psnr.item()
        
        # SSIM (简化版本)
        ssim_score = self.compute_ssim(pred_rgb, target_rgb)
        metrics['ssim'] = ssim_score
        
        return metrics
    
    def compute_ssim(self, pred, target, window_size=11):
        """计算SSIM"""
        # 简化的SSIM计算
        mu1 = torch.mean(pred)
        mu2 = torch.mean(target)
        
        sigma1_sq = torch.var(pred)
        sigma2_sq = torch.var(target)
        sigma12 = torch.mean((pred - mu1) * (target - mu2))
        
        c1 = 0.01 ** 2
        c2 = 0.03 ** 2
        
        ssim = ((2 * mu1 * mu2 + c1) * (2 * sigma12 + c2)) / \
               ((mu1 ** 2 + mu2 ** 2 + c1) * (sigma1_sq + sigma2_sq + c2))
        
        return ssim.item()
    
    def save_checkpoint(self, epoch, avg_losses, avg_metrics):
        """保存检查点"""
        checkpoint = {
            'epoch': epoch,
            'global_step': self.global_step,
            'models': {},
            'optimizers': {},
            'schedulers': {},
            'losses': avg_losses,
            'metrics': avg_metrics,
            'config': self.config
        }
        
        # 保存模型状态
        for name, model in self.models.items():
            if isinstance(model, nn.Module):
                if hasattr(model, 'module'):  # DDP
                    checkpoint['models'][name] = model.module.state_dict()
                else:
                    checkpoint['models'][name] = model.state_dict()
        
        # 保存优化器状态
        for name, optimizer in self.optimizers.items():
            checkpoint['optimizers'][name] = optimizer.state_dict()
        
        # 保存调度器状态
        for name, scheduler in self.schedulers.items():
            checkpoint['schedulers'][name] = scheduler.state_dict()
        
        # 保存路径
        checkpoint_dir = self.config.get('checkpoint_dir', './checkpoints')
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        # 最新检查点
        latest_path = os.path.join(checkpoint_dir, 'latest.pth')
        torch.save(checkpoint, latest_path)
        
        # 定期检查点
        if epoch % self.config.get('save_freq', 10) == 0:
            epoch_path = os.path.join(checkpoint_dir, f'epoch_{epoch}.pth')
            torch.save(checkpoint, epoch_path)
        
        # 最佳模型
        current_psnr = avg_metrics.get('psnr', 0)
        if current_psnr > self.best_psnr:
            self.best_psnr = current_psnr
            best_path = os.path.join(checkpoint_dir, 'best.pth')
            torch.save(checkpoint, best_path)
            
            if self.logger:
                self.logger.info(f'新的最佳模型保存: PSNR = {current_psnr:.4f}')
    
    def load_checkpoint(self, checkpoint_path):
        """加载检查点"""
        if self.logger:
            self.logger.info(f'加载检查点: {checkpoint_path}')
        
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        # 加载模型
        for name, model in self.models.items():
            if name in checkpoint['models'] and isinstance(model, nn.Module):
                if hasattr(model, 'module'):  # DDP
                    model.module.load_state_dict(checkpoint['models'][name])
                else:
                    model.load_state_dict(checkpoint['models'][name])
        
        # 加载优化器
        for name, optimizer in self.optimizers.items():
            if name in checkpoint['optimizers']:
                optimizer.load_state_dict(checkpoint['optimizers'][name])
        
        # 加载调度器
        for name, scheduler in self.schedulers.items():
            if name in checkpoint['schedulers']:
                scheduler.load_state_dict(checkpoint['schedulers'][name])
        
        self.global_step = checkpoint.get('global_step', 0)
        self.best_psnr = checkpoint.get('metrics', {}).get('psnr', 0)
        
        return checkpoint['epoch']
    
    def train(self, train_dataset, val_dataset, start_epoch=0, total_epochs=100):
        """主训练循环"""
        if self.logger:
            self.logger.info(f'开始训练: epochs {start_epoch}-{total_epochs}')
        
        for epoch in range(start_epoch, total_epochs):
            # 训练一个epoch
            train_losses = self.train_epoch(train_dataset, epoch)
            
            # 更新学习率
            for scheduler in self.schedulers.values():
                scheduler.step()
            
            # 验证
            if epoch % self.config.get('val_freq', 5) == 0:
                val_losses, val_metrics = self.validate(val_dataset)
                
                if self.is_main_process:
                    # 记录到tensorboard
                    if self.writer:
                        for loss_name, loss_value in val_losses.items():
                            self.writer.add_scalar(f'val/{loss_name}', loss_value, epoch)
                        
                        for metric_name, metric_value in val_metrics.items():
                            self.writer.add_scalar(f'val/{metric_name}', metric_value, epoch)
                    
                    # 打印日志
                    if self.logger:
                        self.logger.info(
                            f'Epoch {epoch}: '
                            f'Train Loss = {train_losses["total"]:.6f}, '
                            f'Val Loss = {val_losses["total"]:.6f}, '
                            f'Val PSNR = {val_metrics.get("psnr", 0):.4f}'
                        )
                    
                    # 保存检查点
                    self.save_checkpoint(epoch, val_losses, val_metrics)
            else:
                val_losses, val_metrics = {}, {}
                
                if self.is_main_process and self.logger:
                    self.logger.info(
                        f'Epoch {epoch}: Train Loss = {train_losses["total"]:.6f}'
                    )
```

---

## 调试技巧和故障排除

### 常见问题诊断

```python
class BlockNeRFDebugger:
    """Block-NeRF 调试工具"""
    
    def __init__(self, models, config):
        self.models = models
        self.config = config
        
    def check_data_quality(self, dataset):
        """检查数据质量"""
        print("检查数据质量...")
        
        # 检查数据集大小
        print(f"数据集大小: {len(dataset)}")
        
        # 采样几个批次检查
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True)
        
        for i, batch in enumerate(dataloader):
            if i >= 5:  # 只检查前5个样本
                break
                
            # 检查数据范围
            rays_o = batch['rays_o']
            rays_d = batch['rays_d']
            rgb = batch['rgb']
            
            print(f"样本 {i}:")
            print(f"  rays_o 范围: [{rays_o.min():.3f}, {rays_o.max():.3f}]")
            print(f"  rays_d 范围: [{rays_d.min():.3f}, {rays_d.max():.3f}]")
            print(f"  rgb 范围: [{rgb.min():.3f}, {rgb.max():.3f}]")
            
            # 检查异常值
            if torch.isnan(rays_o).any():
                print("  警告: rays_o 包含 NaN")
            if torch.isnan(rays_d).any():
                print("  警告: rays_d 包含 NaN")
            if torch.isnan(rgb).any():
                print("  警告: rgb 包含 NaN")
    
    def check_model_gradients(self, model, loss):
        """检查模型梯度"""
        print("检查模型梯度...")
        
        # 反向传播
        loss.backward(retain_graph=True)
        
        # 检查梯度
        grad_stats = {}
        for name, param in model.named_parameters():
            if param.grad is not None:
                grad_norm = param.grad.norm().item()
                grad_max = param.grad.abs().max().item()
                grad_mean = param.grad.mean().item()
                
                grad_stats[name] = {
                    'norm': grad_norm,
                    'max': grad_max, 
                    'mean': grad_mean
                }
                
                # 检查异常梯度
                if grad_norm > 100:
                    print(f"  警告: {name} 梯度过大 (norm={grad_norm:.3f})")
                elif grad_norm < 1e-8:
                    print(f"  警告: {name} 梯度过小 (norm={grad_norm:.3e})")
                
                if torch.isnan(param.grad).any():
                    print(f"  错误: {name} 梯度包含 NaN")
                if torch.isinf(param.grad).any():
                    print(f"  错误: {name} 梯度包含 Inf")
        
        return grad_stats
    
    def visualize_rendering_process(self, batch, output_dir='./debug_renders'):
        """可视化渲染过程"""
        os.makedirs(output_dir, exist_ok=True)
        
        # 设置为评估模式
        for model in self.models.values():
            if isinstance(model, nn.Module):
                model.eval()
        
        with torch.no_grad():
            # 提取一条光线进行详细分析
            ray_o = batch['rays_o'][0:1]  # [1, 3]
            ray_d = batch['rays_d'][0:1]  # [1, 3]
            
            # 生成采样点
            t_vals = torch.linspace(0.1, 10.0, 64, device=ray_o.device)
            pts = ray_o + ray_d * t_vals.unsqueeze(-1)  # [1, 64, 3]
            
            # 查询密度和颜色
            if 'block_nerf' in self.models:
                density, color = self.models['block_nerf'].query_points(pts.squeeze(0))
                
                # 保存采样点
                np.save(
                    os.path.join(output_dir, 'sample_points.npy'),
                    pts.squeeze(0).cpu().numpy()
                )
                
                # 保存密度
                np.save(
                    os.path.join(output_dir, 'densities.npy'),
                    density.cpu().numpy()
                )
                
                # 保存颜色
                np.save(
                    os.path.join(output_dir, 'colors.npy'),
                    color.cpu().numpy()
                )
                
                # 绘制密度曲线
                import matplotlib.pyplot as plt
                
                plt.figure(figsize=(12, 4))
                
                plt.subplot(1, 3, 1)
                plt.plot(t_vals.cpu().numpy(), density.cpu().numpy())
                plt.title('Density along ray')
                plt.xlabel('t')
                plt.ylabel('density')
                
                plt.subplot(1, 3, 2)
                colors_np = color.cpu().numpy()
                plt.plot(t_vals.cpu().numpy(), colors_np[:, 0], 'r-', label='R')
                plt.plot(t_vals.cpu().numpy(), colors_np[:, 1], 'g-', label='G')
                plt.plot(t_vals.cpu().numpy(), colors_np[:, 2], 'b-', label='B')
                plt.title('Color along ray')
                plt.xlabel('t')
                plt.ylabel('color')
                plt.legend()
                
                plt.subplot(1, 3, 3)
                # 计算透明度
                alpha = 1 - torch.exp(-density * 0.1)  # 假设 delta_t = 0.1
                transmittance = torch.cumprod(1 - alpha + 1e-8, dim=0)
                weights = alpha * transmittance
                
                plt.plot(t_vals.cpu().numpy(), weights.cpu().numpy())
                plt.title('Weights along ray')
                plt.xlabel('t')
                plt.ylabel('weight')
                
                plt.tight_layout()
                plt.savefig(os.path.join(output_dir, 'ray_analysis.png'))
                plt.close()
    
    def memory_profiling(self, model, batch):
        """内存分析"""
        print("内存分析...")
        
        # 记录初始内存
        torch.cuda.empty_cache()
        initial_memory = torch.cuda.memory_allocated()
        
        print(f"初始内存使用: {initial_memory / 1e6:.2f} MB")
        
        # 前向传播
        with torch.cuda.amp.autocast():
            output = model(batch['rays_o'], batch['rays_d'])
        
        forward_memory = torch.cuda.memory_allocated()
        print(f"前向传播后内存: {forward_memory / 1e6:.2f} MB")
        print(f"前向传播增加: {(forward_memory - initial_memory) / 1e6:.2f} MB")
        
        # 计算损失
        loss = F.mse_loss(output['rgb'], batch['rgb'])
        
        # 反向传播
        loss.backward()
        
        backward_memory = torch.cuda.memory_allocated()
        print(f"反向传播后内存: {backward_memory / 1e6:.2f} MB")
        print(f"反向传播增加: {(backward_memory - forward_memory) / 1e6:.2f} MB")
        
        # 清理
        del output, loss
        torch.cuda.empty_cache()
        
        final_memory = torch.cuda.memory_allocated()
        print(f"清理后内存: {final_memory / 1e6:.2f} MB")

def debug_training():
    """调试训练过程"""
    # 加载配置和数据
    config = load_config('configs/debug_config.yaml')
    
    # 创建小型数据集用于调试
    debug_dataset = BlockNeRFDataset(
        data_root='./debug_data',
        split='train',
        config=config['dataset']
    )
    
    # 创建模型
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    models = create_models(config, device)
    
    # 创建调试器
    debugger = BlockNeRFDebugger(models, config)
    
    # 检查数据质量
    debugger.check_data_quality(debug_dataset)
    
    # 测试单步训练
    dataloader = torch.utils.data.DataLoader(debug_dataset, batch_size=2)
    batch = next(iter(dataloader))
    batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v 
             for k, v in batch.items()}
    
    # 检查模型输出
    models['block_nerf'].eval()
    with torch.no_grad():
        output = models['block_nerf'](batch['rays_o'], batch['rays_d'])
        print(f"模型输出形状: {output['rgb'].shape}")
        print(f"输出范围: [{output['rgb'].min():.3f}, {output['rgb'].max():.3f}]")
    
    # 检查损失计算
    models['block_nerf'].train()
    output = models['block_nerf'](batch['rays_o'], batch['rays_d'])
    loss = F.mse_loss(output['rgb'], batch['rgb'])
    
    print(f"损失值: {loss.item():.6f}")
    
    # 检查梯度
    grad_stats = debugger.check_model_gradients(models['block_nerf'], loss)
    
    # 可视化渲染过程
    debugger.visualize_rendering_process(batch)
    
    # 内存分析
    debugger.memory_profiling(models['block_nerf'], batch)

if __name__ == '__main__':
    debug_training()
```

---

## 配置文件示例

### 完整的YAML配置

```yaml
# Block-NeRF 训练配置文件
# configs/block_nerf_config.yaml

# 数据集配置
dataset:
  type: "colmap"
  num_workers: 8
  batch_size: 4096
  val_batch_size: 1024
  image_downsample: 4  # 图像下采样倍数
  use_depth: false     # 是否使用深度监督
  white_background: false
  
  # 光线采样
  num_rays_per_image: 1024
  precrop_iters: 500   # 中心裁剪迭代数
  precrop_frac: 0.5    # 中心裁剪比例

# 场景分解配置
block_manager:
  block_size: 50.0     # 块大小(米)
  overlap_ratio: 0.3   # 重叠比例
  min_images_per_block: 50
  max_blocks: 100      # 最大块数
  
  # 块选择策略
  visibility_threshold: 0.1
  distance_threshold: 100.0

# Block-NeRF 模型配置
block_nerf:
  # 网络架构
  num_layers: 8
  hidden_dim: 256
  skip_connections: [4]
  
  # 位置编码
  pos_encoding_levels: 10
  dir_encoding_levels: 4
  use_integrated_pos_enc: true
  
  # 采样配置
  num_samples_coarse: 64
  num_samples_fine: 128
  use_hierarchical_sampling: true
  perturb_samples: true
  
  # 渲染配置
  white_background: false
  density_activation: "relu"
  rgb_activation: "sigmoid"

# 外观嵌入配置
appearance_embedding:
  embedding_dim: 32
  num_images: 10000    # 预计图像总数
  use_global_appearance: true
  use_block_appearance: true

# 可见性网络配置
visibility_network:
  hidden_dims: [512, 256, 128]
  use_camera_features: true
  use_block_features: true
  dropout: 0.1

# 姿态细化配置
pose_refinement:
  feature_extractor: "resnet18"
  hidden_dim: 256
  max_translation: 1.0  # 最大平移(米)
  max_rotation: 0.1     # 最大旋转(弧度)

# 训练配置
training:
  # 基本参数
  num_epochs: 200
  seed: 42
  
  # 学习率
  nerf_lr: 5e-4
  appearance_lr: 1e-3
  visibility_lr: 1e-3
  pose_lr: 1e-5
  
  # 学习率衰减
  lr_decay_steps: [100, 150]
  lr_decay_rate: 0.1
  
  # 优化器参数
  weight_decay: 1e-6
  gradient_clip_norm: 1.0
  
  # 损失权重
  rgb_loss_weight: 1.0
  depth_loss_weight: 0.1
  consistency_loss_weight: 0.5
  regularization_weight: 1e-6
  appearance_loss_weight: 0.1
  
  # 训练策略
  use_mixed_precision: true
  use_gradient_checkpointing: false
  
  # 组件启用
  use_appearance_embedding: true
  use_visibility_network: true
  use_pose_refinement: true
  
  # 阶段训练
  stage_training:
    enable: true
    stages:
      - name: "coarse"
        epochs: 50
        resolution: 256
        components: ["nerf", "appearance"]
      - name: "fine"
        epochs: 100
        resolution: 512
        components: ["nerf", "appearance", "pose"]
      - name: "refinement"
        epochs: 50
        resolution: 1024
        components: ["nerf", "appearance", "pose", "visibility"]

# 验证和保存配置
validation:
  val_freq: 5          # 验证频率
  save_freq: 10        # 保存频率
  num_val_images: 10   # 验证图像数量
  
  # 渲染配置
  render_test_views: true
  render_novel_views: true

# 日志配置
logging:
  log_freq: 100
  image_log_freq: 1000
  use_tensorboard: true
  tensorboard_dir: "./logs/tensorboard"
  
# 输出配置
output:
  checkpoint_dir: "./checkpoints"
  render_dir: "./renders"
  log_dir: "./logs"

# 分布式训练配置
distributed:
  backend: "nccl"
  find_unused_parameters: true
```

这三个部分的文档涵盖了 Block-NeRF 训练的完整细节，从理论机制到实际实现，再到调试和配置。您可以将这些文档组合起来形成完整的训练指南。
