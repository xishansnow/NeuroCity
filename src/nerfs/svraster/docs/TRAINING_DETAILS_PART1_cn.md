# SVRaster 训练机制详解 - 第一部分：基础训练架构与配置

## 概述

SVRaster（Sparse Voxel Rasterization）的训练机制是一个复杂的多阶段优化过程，涉及稀疏体素的自适应细分、渐进式训练策略、多重损失函数优化等关键技术。本文档详细介绍 SVRaster 的训练架构、配置系统、优化器设置等基础训练机制。

## 1. 训练架构设计

### 1.1 核心训练框架

SVRaster 提供了完整的 PyTorch 训练框架，支持灵活的训练配置和优化策略。

```python
class SVRasterTrainer:
    """
    SVRaster 传统训练器
    提供完整的训练流程控制
    """
    
    def __init__(self, 
                 model_config: SVRasterConfig,
                 trainer_config: SVRasterTrainerConfig,
                 train_dataset: SVRasterDataset,
                 val_dataset: Optional[SVRasterDataset] = None):
        """
        初始化训练器
        
        Args:
            model_config: 模型配置
            trainer_config: 训练器配置
            train_dataset: 训练数据集
            val_dataset: 验证数据集
        """
        self.model_config = model_config
        self.config = trainer_config
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        
        # 设备配置
        self.device = torch.device(trainer_config.device)
        
        # 初始化模型和损失函数
        self.model = SVRasterModel(model_config).to(self.device)
        self.loss_fn = SVRasterLoss(model_config)
        
        # 设置优化器和调度器
        self._setup_optimizer()
        
        # 设置数据加载器
        self._setup_data_loaders()
        
        # 设置日志记录
        self._setup_logging()
        
        # 混合精度训练
        if self.config.use_mixed_precision:
            self.scaler = GradScaler()
        
        # 训练状态
        self.current_epoch = 0
        self.global_step = 0
        self.best_val_psnr = 0.0
    
    def train(self):
        """
        主训练循环
        """
        logger.info("开始训练 SVRaster 模型")
        
        for epoch in range(self.current_epoch, self.config.num_epochs):
            self.current_epoch = epoch
            
            # 训练一个 epoch
            self._train_epoch()
            
            # 验证
            if epoch % self.config.val_interval == 0:
                self._validate()
            
            # 自适应细分
            if self._should_subdivide(epoch):
                self._adaptive_subdivision()
            
            # 剪枝
            if self._should_prune(epoch):
                self._prune_voxels()
            
            # 保存检查点
            if epoch % self.config.save_interval == 0:
                self._save_checkpoint()
            
            # 更新学习率
            if self.scheduler:
                self.scheduler.step()
        
        logger.info("训练完成")
    
    def _train_epoch(self):
        """
        训练一个 epoch
        """
        self.model.train()
        epoch_loss = 0.0
        epoch_psnr = 0.0
        
        progress_bar = tqdm(self.train_loader, desc=f"Epoch {self.current_epoch}")
        
        for batch_idx, batch in enumerate(progress_bar):
            loss_dict, metrics = self._train_step(batch)
            
            # 记录损失和指标
            epoch_loss += loss_dict['total_loss'].item()
            epoch_psnr += metrics['psnr']
            
            # 更新进度条
            progress_bar.set_postfix({
                'loss': f"{loss_dict['total_loss'].item():.4f}",
                'psnr': f"{metrics['psnr']:.2f}",
                'lr': f"{self.optimizer.param_groups[0]['lr']:.6f}"
            })
            
            # 日志记录
            if self.global_step % self.config.log_interval == 0:
                self._log_metrics(loss_dict, metrics, 'train')
            
            self.global_step += 1
        
        # 记录 epoch 平均指标
        avg_loss = epoch_loss / len(self.train_loader)
        avg_psnr = epoch_psnr / len(self.train_loader)
        
        logger.info(f"Epoch {self.current_epoch}: Loss={avg_loss:.4f}, PSNR={avg_psnr:.2f}")
    
    def _train_step(self, batch: dict) -> tuple[dict, dict]:
        """
        单步训练
        
        Args:
            batch: 训练批次数据
            
        Returns:
            损失字典和指标字典
        """
        # 数据转移到设备
        rays = batch['rays'].to(self.device)
        target_rgb = batch['rgb'].to(self.device)
        target_depth = batch.get('depth', None)
        if target_depth is not None:
            target_depth = target_depth.to(self.device)
        
        # 前向传播
        if self.config.use_mixed_precision:
            with autocast():
                outputs = self.model(rays)
                loss_dict = self.loss_fn(
                    pred_rgb=outputs['rgb'],
                    target_rgb=target_rgb,
                    pred_depth=outputs.get('depth'),
                    target_depth=target_depth,
                    weights=outputs.get('weights'),
                    extras=outputs
                )
        else:
            outputs = self.model(rays)
            loss_dict = self.loss_fn(
                pred_rgb=outputs['rgb'],
                target_rgb=target_rgb,
                pred_depth=outputs.get('depth'),
                target_depth=target_depth,
                weights=outputs.get('weights'),
                extras=outputs
            )
        
        # 反向传播
        self.optimizer.zero_grad()
        
        if self.config.use_mixed_precision:
            self.scaler.scale(loss_dict['total_loss']).backward()
            if self.config.gradient_clip_norm > 0:
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), 
                    self.config.gradient_clip_norm
                )
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            loss_dict['total_loss'].backward()
            if self.config.gradient_clip_norm > 0:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), 
                    self.config.gradient_clip_norm
                )
            self.optimizer.step()
        
        # 计算指标
        with torch.no_grad():
            metrics = self._compute_metrics(outputs['rgb'], target_rgb)
        
        return loss_dict, metrics
```

## 2. 配置系统设计

### 2.1 训练器配置

```python
@dataclass
class SVRasterTrainerConfig:
    """
    SVRaster 训练器配置
    定义训练过程的所有参数
    """
    
    # 基础训练参数
    num_epochs: int = 100                    # 训练轮数
    batch_size: int = 1                      # 批大小
    learning_rate: float = 1e-3              # 学习率
    weight_decay: float = 1e-4               # 权重衰减
    
    # 优化器设置
    optimizer_type: str = "adamw"            # 优化器类型：adam, adamw, sgd
    scheduler_type: str = "cosine"           # 调度器类型：cosine, step, exponential
    scheduler_params: Optional[dict] = None   # 调度器参数
    warmup_steps: int = 1000                 # 预热步数
    
    # 损失函数权重
    rgb_loss_weight: float = 1.0             # RGB 损失权重
    depth_loss_weight: float = 0.1           # 深度损失权重
    opacity_reg_weight: float = 0.01         # 透明度正则化权重
    ssim_loss_weight: float = 0.1            # SSIM 损失权重
    perceptual_loss_weight: float = 0.1      # 感知损失权重
    
    # 自适应细分配置
    enable_subdivision: bool = True          # 启用自适应细分
    subdivision_start_epoch: int = 10        # 细分开始轮数
    subdivision_interval: int = 5            # 细分间隔
    subdivision_threshold: float = 0.01      # 细分阈值
    max_subdivision_level: int = 12          # 最大细分等级
    
    # 剪枝配置
    enable_pruning: bool = True              # 启用剪枝
    pruning_start_epoch: int = 20            # 剪枝开始轮数
    pruning_interval: int = 10               # 剪枝间隔
    pruning_threshold: float = 0.001         # 剪枝阈值
    
    # 渐进式训练配置
    progressive_training: bool = True        # 启用渐进式训练
    min_resolution: int = 64                 # 最小分辨率
    max_resolution: int = 512                # 最大分辨率
    resolution_schedule: List[int] = None    # 分辨率调度表
    
    # 验证和日志
    val_interval: int = 5                    # 验证间隔
    log_interval: int = 100                  # 日志记录间隔
    save_interval: int = 1000                # 保存间隔
    
    # 文件路径
    checkpoint_dir: str = "checkpoints"      # 检查点目录
    log_dir: str = "logs"                    # 日志目录
    
    # 设备和精度
    device: str = "cuda"                     # 训练设备
    use_mixed_precision: bool = True         # 混合精度训练
    gradient_clip_norm: float = 1.0          # 梯度裁剪
    
    # 渲染设置
    render_batch_size: int = 4096            # 渲染批大小
    render_chunk_size: int = 1024            # 渲染块大小
    
    # 高级训练选项
    use_ema: bool = True                     # 使用 EMA 模型
    ema_decay: float = 0.999                 # EMA 衰减率
    distributed_training: bool = False        # 分布式训练
    
    def __post_init__(self):
        """
        后初始化验证和设置默认值
        """
        if self.scheduler_params is None:
            self.scheduler_params = {}
        
        if self.resolution_schedule is None:
            # 创建默认的分辨率调度表
            self.resolution_schedule = self._create_default_resolution_schedule()
        
        # 验证配置的合理性
        self._validate_config()
    
    def _create_default_resolution_schedule(self) -> List[int]:
        """
        创建默认的分辨率调度表
        """
        schedule = []
        current_res = self.min_resolution
        
        while current_res <= self.max_resolution:
            schedule.append(current_res)
            current_res *= 2
        
        return schedule
    
    def _validate_config(self):
        """
        验证配置参数的合理性
        """
        assert self.num_epochs > 0, "训练轮数必须大于 0"
        assert self.batch_size > 0, "批大小必须大于 0"
        assert self.learning_rate > 0, "学习率必须大于 0"
        assert self.subdivision_start_epoch >= 0, "细分开始轮数不能为负"
        assert self.pruning_start_epoch >= 0, "剪枝开始轮数不能为负"
        
        if self.progressive_training:
            assert self.min_resolution <= self.max_resolution, \
                "最小分辨率不能大于最大分辨率"
```

## 3. 优化器和调度器设置

### 3.1 参数分组策略

SVRaster 的不同组件需要不同的优化策略：

```python
class ParameterGroupManager:
    """
    参数分组管理器
    为不同类型的参数设置不同的学习率和优化策略
    """
    
    def __init__(self, model: SVRasterModel, config: SVRasterTrainerConfig):
        self.model = model
        self.config = config
        
    def create_parameter_groups(self) -> List[dict]:
        """
        创建参数分组
        """
        # 体素相关参数
        voxel_density_params = []
        voxel_color_params = []
        
        # 网络参数
        network_params = []
        
        # 其他参数
        other_params = []
        
        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue
                
            if 'density' in name and 'voxel' in name:
                voxel_density_params.append(param)
            elif 'color' in name and 'voxel' in name:
                voxel_color_params.append(param)
            elif any(net_name in name for net_name in ['mlp', 'network', 'encoder']):
                network_params.append(param)
            else:
                other_params.append(param)
        
        # 创建参数分组
        param_groups = []
        
        # 体素密度参数 - 较高学习率
        if voxel_density_params:
            param_groups.append({
                'params': voxel_density_params,
                'lr': self.config.learning_rate * 2.0,
                'weight_decay': self.config.weight_decay * 0.1,
                'name': 'voxel_density'
            })
        
        # 体素颜色参数 - 中等学习率
        if voxel_color_params:
            param_groups.append({
                'params': voxel_color_params,
                'lr': self.config.learning_rate,
                'weight_decay': self.config.weight_decay,
                'name': 'voxel_color'
            })
        
        # 网络参数 - 较低学习率
        if network_params:
            param_groups.append({
                'params': network_params,
                'lr': self.config.learning_rate * 0.1,
                'weight_decay': self.config.weight_decay * 2.0,
                'name': 'network'
            })
        
        # 其他参数 - 默认学习率
        if other_params:
            param_groups.append({
                'params': other_params,
                'lr': self.config.learning_rate,
                'weight_decay': self.config.weight_decay,
                'name': 'other'
            })
        
        return param_groups
```

### 3.2 高级调度器

```python
class AdvancedSchedulerManager:
    """
    高级学习率调度器管理器
    支持多种复杂的调度策略
    """
    
    def __init__(self, optimizer: torch.optim.Optimizer, config: SVRasterTrainerConfig):
        self.optimizer = optimizer
        self.config = config
        
    def create_scheduler(self):
        """
        创建学习率调度器
        """
        if self.config.scheduler_type == 'cosine':
            return self._create_cosine_scheduler()
        elif self.config.scheduler_type == 'onecycle':
            return self._create_onecycle_scheduler()
        elif self.config.scheduler_type == 'multistep':
            return self._create_multistep_scheduler()
        elif self.config.scheduler_type == 'warmup_cosine':
            return self._create_warmup_cosine_scheduler()
        else:
            return None
    
    def _create_cosine_scheduler(self):
        """
        余弦退火调度器
        """
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=self.config.num_epochs,
            eta_min=self.config.learning_rate * 0.01
        )
    
    def _create_onecycle_scheduler(self):
        """
        OneCycle 调度器
        """
        return torch.optim.lr_scheduler.OneCycleLR(
            self.optimizer,
            max_lr=self.config.learning_rate,
            total_steps=self.config.num_epochs,
            pct_start=0.3,
            anneal_strategy='cos'
        )
    
    def _create_multistep_scheduler(self):
        """
        多步调度器
        """
        milestones = self.config.scheduler_params.get(
            'milestones', 
            [self.config.num_epochs // 2, self.config.num_epochs * 3 // 4]
        )
        gamma = self.config.scheduler_params.get('gamma', 0.1)
        
        return torch.optim.lr_scheduler.MultiStepLR(
            self.optimizer,
            milestones=milestones,
            gamma=gamma
        )
    
    def _create_warmup_cosine_scheduler(self):
        """
        带预热的余弦调度器
        """
        return WarmupCosineAnnealingLR(
            self.optimizer,
            warmup_epochs=self.config.scheduler_params.get('warmup_epochs', 10),
            max_epochs=self.config.num_epochs,
            warmup_start_lr=self.config.learning_rate * 0.1,
            eta_min=self.config.learning_rate * 0.01
        )

class WarmupCosineAnnealingLR(torch.optim.lr_scheduler._LRScheduler):
    """
    带预热的余弦退火学习率调度器
    """
    
    def __init__(self, optimizer, warmup_epochs, max_epochs, 
                 warmup_start_lr=1e-6, eta_min=0, last_epoch=-1):
        self.warmup_epochs = warmup_epochs
        self.max_epochs = max_epochs
        self.warmup_start_lr = warmup_start_lr
        self.eta_min = eta_min
        super().__init__(optimizer, last_epoch)
    
    def get_lr(self):
        if self.last_epoch < self.warmup_epochs:
            # 预热阶段
            lr_mult = (self.last_epoch + 1) / self.warmup_epochs
            return [
                self.warmup_start_lr + (base_lr - self.warmup_start_lr) * lr_mult
                for base_lr in self.base_lrs
            ]
        else:
            # 余弦退火阶段
            progress = (self.last_epoch - self.warmup_epochs) / (self.max_epochs - self.warmup_epochs)
            return [
                self.eta_min + (base_lr - self.eta_min) * 
                (1 + math.cos(math.pi * progress)) / 2
                for base_lr in self.base_lrs
            ]
```

## 4. 数据加载和预处理

### 4.1 训练数据管理

```python
class SVRasterDataManager:
    """
    SVRaster 数据管理器
    处理训练数据的加载、预处理和增强
    """
    
    def __init__(self, config: SVRasterTrainerConfig):
        self.config = config
        self.data_augmenter = DataAugmenter(config)
        
    def create_data_loaders(self, train_dataset, val_dataset=None):
        """
        创建数据加载器
        """
        # 训练数据加载器
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=self.config.num_workers,
            pin_memory=self.config.pin_memory,
            drop_last=True,
            persistent_workers=True,
            prefetch_factor=2
        )
        
        # 验证数据加载器
        val_loader = None
        if val_dataset is not None:
            val_loader = DataLoader(
                val_dataset,
                batch_size=1,  # 验证时批大小为 1
                shuffle=False,
                num_workers=2,
                pin_memory=self.config.pin_memory,
                drop_last=False
            )
        
        return train_loader, val_loader
    
    def apply_data_augmentation(self, batch: dict) -> dict:
        """
        应用数据增强
        """
        if not self.config.data_augmentation:
            return batch
        
        return self.data_augmenter.augment_batch(batch)

class DataAugmenter:
    """
    数据增强器
    为训练数据应用各种增强技术
    """
    
    def __init__(self, config: SVRasterTrainerConfig):
        self.config = config
        
    def augment_batch(self, batch: dict) -> dict:
        """
        批次数据增强
        """
        augmented_batch = batch.copy()
        
        # 颜色抖动
        if self.config.augmentation_config.get('color_jitter', 0) > 0:
            augmented_batch = self._apply_color_jitter(augmented_batch)
        
        # 添加噪声
        if self.config.augmentation_config.get('noise_std', 0) > 0:
            augmented_batch = self._add_noise(augmented_batch)
        
        # 射线抖动
        if self.config.augmentation_config.get('ray_jitter', 0) > 0:
            augmented_batch = self._apply_ray_jitter(augmented_batch)
        
        return augmented_batch
    
    def _apply_color_jitter(self, batch: dict) -> dict:
        """
        应用颜色抖动
        """
        if 'rgb' in batch:
            jitter_strength = self.config.augmentation_config['color_jitter']
            
            # 随机亮度调整
            brightness_factor = 1.0 + torch.randn(1) * jitter_strength
            batch['rgb'] = torch.clamp(batch['rgb'] * brightness_factor, 0, 1)
            
            # 随机对比度调整
            contrast_factor = 1.0 + torch.randn(1) * jitter_strength
            mean_rgb = torch.mean(batch['rgb'])
            batch['rgb'] = torch.clamp(
                mean_rgb + (batch['rgb'] - mean_rgb) * contrast_factor, 
                0, 1
            )
        
        return batch
    
    def _add_noise(self, batch: dict) -> dict:
        """
        添加噪声
        """
        noise_std = self.config.augmentation_config['noise_std']
        
        if 'rgb' in batch:
            noise = torch.randn_like(batch['rgb']) * noise_std
            batch['rgb'] = torch.clamp(batch['rgb'] + noise, 0, 1)
        
        return batch
    
    def _apply_ray_jitter(self, batch: dict) -> dict:
        """
        应用射线抖动
        """
        if 'rays' in batch:
            jitter_strength = self.config.augmentation_config['ray_jitter']
            
            # 对射线方向添加微小扰动
            ray_dirs = batch['rays'][..., 3:6]
            jitter = torch.randn_like(ray_dirs) * jitter_strength
            ray_dirs_jittered = torch.nn.functional.normalize(ray_dirs + jitter, dim=-1)
            
            batch['rays'] = batch['rays'].clone()
            batch['rays'][..., 3:6] = ray_dirs_jittered
        
        return batch
```

## 总结

SVRaster 的基础训练架构包含以下关键组件：

1. **完整训练框架**：基于 PyTorch 的训练器，满足各种训练需求
2. **灵活配置系统**：全面的配置选项，支持各种训练策略
3. **智能参数分组**：针对不同类型参数设置不同的优化策略
4. **高级调度器**：支持多种学习率调度策略，包括预热和余弦退火
5. **数据管理系统**：高效的数据加载和增强机制

这些基础设施为 SVRaster 的高质量训练提供了坚实的基础。

**下一部分预告**：第二部分将详细介绍自适应细分、剪枝策略、渐进式训练等高级训练技术。
