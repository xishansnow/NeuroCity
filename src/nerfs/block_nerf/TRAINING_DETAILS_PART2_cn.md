# Block-NeRF 训练机制详解 - 第二部分

## 损失函数设计

### 主要损失组件

Block-NeRF 的训练涉及多个损失函数的组合，每个组件都有特定的作用：

```python
class BlockNeRFLoss:
    """Block-NeRF 综合损失函数"""
    
    def __init__(self, config):
        self.config = config
        self.rgb_weight = config.rgb_loss_weight
        self.depth_weight = config.depth_loss_weight
        self.consistency_weight = config.consistency_loss_weight
        self.regularization_weight = config.regularization_weight
        self.appearance_weight = config.appearance_loss_weight
        
    def compute_total_loss(self, predictions, targets, block_data):
        """计算总损失"""
        losses = {}
        
        # 1. RGB 重建损失
        losses['rgb'] = self.compute_rgb_loss(
            predictions['rgb'], targets['rgb']
        )
        
        # 2. 深度一致性损失
        if 'depth' in targets:
            losses['depth'] = self.compute_depth_loss(
                predictions['depth'], targets['depth']
            )
        
        # 3. 块间一致性损失
        losses['consistency'] = self.compute_consistency_loss(
            predictions, block_data
        )
        
        # 4. 正则化损失
        losses['regularization'] = self.compute_regularization_loss(
            block_data['model_parameters']
        )
        
        # 5. 外观嵌入损失
        losses['appearance'] = self.compute_appearance_loss(
            predictions['appearance_features'], targets['appearance_targets']
        )
        
        # 总损失
        total_loss = (
            self.rgb_weight * losses['rgb'] +
            self.depth_weight * losses['depth'] +
            self.consistency_weight * losses['consistency'] +
            self.regularization_weight * losses['regularization'] +
            self.appearance_weight * losses['appearance']
        )
        
        return total_loss, losses
```

### 1. RGB 重建损失

```python
def compute_rgb_loss(self, predicted_rgb, target_rgb):
    """RGB 重建损失"""
    # L2 损失 (主要)
    l2_loss = F.mse_loss(predicted_rgb, target_rgb)
    
    # L1 损失 (鲁棒性)
    l1_loss = F.l1_loss(predicted_rgb, target_rgb)
    
    # SSIM 损失 (感知质量)
    ssim_loss = 1 - ssim(predicted_rgb, target_rgb, data_range=1.0)
    
    # 组合损失
    rgb_loss = 0.8 * l2_loss + 0.1 * l1_loss + 0.1 * ssim_loss
    
    return rgb_loss
```

### 2. 深度一致性损失

```python
def compute_depth_loss(self, predicted_depth, target_depth, mask=None):
    """深度一致性损失"""
    if mask is not None:
        predicted_depth = predicted_depth[mask]
        target_depth = target_depth[mask]
    
    # 尺度不变深度损失
    def scale_invariant_loss(pred, target):
        log_diff = torch.log(pred + 1e-6) - torch.log(target + 1e-6)
        return torch.mean(log_diff ** 2) - 0.5 * torch.mean(log_diff) ** 2
    
    # 梯度损失 (边界保持)
    def gradient_loss(pred, target):
        pred_grad_x = torch.abs(pred[:, :, 1:] - pred[:, :, :-1])
        pred_grad_y = torch.abs(pred[:, 1:, :] - pred[:, :-1, :])
        target_grad_x = torch.abs(target[:, :, 1:] - target[:, :, :-1])
        target_grad_y = torch.abs(target[:, 1:, :] - target[:, :-1, :])
        
        grad_loss_x = F.l1_loss(pred_grad_x, target_grad_x)
        grad_loss_y = F.l1_loss(pred_grad_y, target_grad_y)
        
        return grad_loss_x + grad_loss_y
    
    # 组合深度损失
    si_loss = scale_invariant_loss(predicted_depth, target_depth)
    grad_loss = gradient_loss(predicted_depth.view(-1, H, W), 
                             target_depth.view(-1, H, W))
    
    return si_loss + 0.1 * grad_loss
```

### 3. 块间一致性损失

```python
def compute_consistency_loss(self, predictions, block_data):
    """块间一致性损失"""
    consistency_loss = 0
    overlap_regions = block_data['overlap_regions']
    
    for region in overlap_regions:
        block_a_id, block_b_id = region['blocks']
        overlap_rays = region['rays']
        
        # 两个块在重叠区域的预测
        rgb_a = predictions[block_a_id]['rgb'][overlap_rays]
        rgb_b = predictions[block_b_id]['rgb'][overlap_rays]
        depth_a = predictions[block_a_id]['depth'][overlap_rays]
        depth_b = predictions[block_b_id]['depth'][overlap_rays]
        
        # RGB 一致性
        rgb_consistency = F.mse_loss(rgb_a, rgb_b)
        
        # 深度一致性
        depth_consistency = F.l1_loss(depth_a, depth_b)
        
        # 权重衰减 (边界处权重较小)
        weights = region['boundary_weights']
        weighted_rgb = torch.mean(weights * (rgb_a - rgb_b) ** 2)
        weighted_depth = torch.mean(weights * torch.abs(depth_a - depth_b))
        
        consistency_loss += weighted_rgb + 0.1 * weighted_depth
    
    return consistency_loss / len(overlap_regions)
```

### 4. 正则化损失

```python
def compute_regularization_loss(self, model_parameters):
    """模型正则化损失"""
    reg_loss = 0
    
    # L2 权重衰减
    for name, param in model_parameters.items():
        if 'weight' in name:
            reg_loss += torch.sum(param ** 2)
    
    # 密度平滑性正则化
    if 'density_grid' in model_parameters:
        density = model_parameters['density_grid']
        # 三维拉普拉斯算子
        laplacian = compute_3d_laplacian(density)
        smoothness_loss = torch.mean(laplacian ** 2)
        reg_loss += 0.01 * smoothness_loss
    
    # 外观嵌入稀疏性
    if 'appearance_embeddings' in model_parameters:
        appearance_embeddings = model_parameters['appearance_embeddings']
        sparsity_loss = torch.mean(torch.abs(appearance_embeddings))
        reg_loss += 0.001 * sparsity_loss
    
    return reg_loss
```

---

## 优化策略

### 学习率调度

```python
class BlockNeRFScheduler:
    """Block-NeRF 学习率调度器"""
    
    def __init__(self, config):
        self.config = config
        self.schedulers = {}
        
    def setup_schedulers(self, optimizers):
        """设置不同组件的学习率调度"""
        
        # NeRF 网络调度器 (指数衰减)
        self.schedulers['nerf'] = torch.optim.lr_scheduler.ExponentialLR(
            optimizers['nerf'], 
            gamma=self.config.nerf_lr_decay
        )
        
        # 外观嵌入调度器 (余弦退火)
        self.schedulers['appearance'] = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizers['appearance'],
            T_max=self.config.appearance_cosine_cycles,
            eta_min=self.config.appearance_min_lr
        )
        
        # 姿态细化调度器 (阶梯衰减)
        self.schedulers['pose'] = torch.optim.lr_scheduler.MultiStepLR(
            optimizers['pose'],
            milestones=self.config.pose_milestones,
            gamma=self.config.pose_lr_decay
        )
        
        # 可见性网络调度器 (预热 + 衰减)
        self.schedulers['visibility'] = self.create_warmup_scheduler(
            optimizers['visibility']
        )
    
    def step(self, epoch, component=None):
        """更新学习率"""
        if component:
            self.schedulers[component].step()
        else:
            for scheduler in self.schedulers.values():
                scheduler.step()
```

### 渐进式训练策略

```python
class ProgressiveTrainer:
    """渐进式训练器"""
    
    def __init__(self, config):
        self.config = config
        self.current_stage = 0
        self.training_stages = self.define_training_stages()
    
    def define_training_stages(self):
        """定义训练阶段"""
        stages = [
            {
                'name': 'coarse_training',
                'epochs': 50,
                'resolution': 256,
                'num_samples': 64,
                'components': ['nerf', 'appearance']
            },
            {
                'name': 'fine_training', 
                'epochs': 100,
                'resolution': 512,
                'num_samples': 128,
                'components': ['nerf', 'appearance', 'pose']
            },
            {
                'name': 'refinement',
                'epochs': 50,
                'resolution': 1024,
                'num_samples': 256,
                'components': ['nerf', 'appearance', 'pose', 'visibility']
            }
        ]
        return stages
    
    def train_stage(self, stage_config, models, optimizers):
        """训练单个阶段"""
        print(f"开始训练阶段: {stage_config['name']}")
        
        # 调整模型配置
        self.adjust_model_config(stage_config, models)
        
        for epoch in range(stage_config['epochs']):
            for component in stage_config['components']:
                if component in models:
                    self.train_component(
                        models[component], 
                        optimizers[component],
                        stage_config
                    )
```

### 内存优化策略

```python
class MemoryOptimizer:
    """内存优化器"""
    
    def __init__(self, config):
        self.config = config
        self.gradient_checkpointing = config.use_gradient_checkpointing
        self.mixed_precision = config.use_mixed_precision
        
    def setup_memory_optimization(self, models):
        """设置内存优化"""
        
        # 梯度检查点
        if self.gradient_checkpointing:
            for model in models.values():
                if hasattr(model, 'enable_gradient_checkpointing'):
                    model.enable_gradient_checkpointing()
        
        # 混合精度训练
        if self.mixed_precision:
            self.scaler = torch.cuda.amp.GradScaler()
        
        # 动态批处理大小
        self.adaptive_batch_size = AdaptiveBatchSize(
            initial_batch_size=self.config.batch_size,
            memory_limit=self.config.memory_limit_gb
        )
    
    def optimize_forward_pass(self, model, inputs):
        """优化前向传播"""
        if self.mixed_precision:
            with torch.cuda.amp.autocast():
                outputs = model(inputs)
        else:
            outputs = model(inputs)
        
        return outputs
    
    def optimize_backward_pass(self, loss, optimizer):
        """优化反向传播"""
        if self.mixed_precision:
            self.scaler.scale(loss).backward()
            self.scaler.step(optimizer)
            self.scaler.update()
        else:
            loss.backward()
            optimizer.step()
```

---

## 多GPU 分布式训练

### 数据并行策略

```python
class DistributedBlockNeRFTrainer:
    """分布式 Block-NeRF 训练器"""
    
    def __init__(self, config, world_size, rank):
        self.config = config
        self.world_size = world_size
        self.rank = rank
        self.device = torch.device(f'cuda:{rank}')
        
    def setup_distributed_training(self):
        """设置分布式训练"""
        # 初始化进程组
        torch.distributed.init_process_group(
            backend='nccl',
            world_size=self.world_size,
            rank=self.rank
        )
        
        # 设置设备
        torch.cuda.set_device(self.rank)
        
    def distribute_blocks(self, blocks):
        """分配块到不同GPU"""
        blocks_per_gpu = len(blocks) // self.world_size
        start_idx = self.rank * blocks_per_gpu
        end_idx = start_idx + blocks_per_gpu
        
        if self.rank == self.world_size - 1:
            end_idx = len(blocks)  # 最后一个GPU处理剩余块
            
        return blocks[start_idx:end_idx]
    
    def train_distributed(self, models, datasets):
        """分布式训练主循环"""
        # 包装模型为DDP
        ddp_models = {}
        for name, model in models.items():
            ddp_models[name] = torch.nn.parallel.DistributedDataParallel(
                model.to(self.device),
                device_ids=[self.rank],
                find_unused_parameters=True
            )
        
        # 分布式数据加载
        samplers = {}
        for name, dataset in datasets.items():
            samplers[name] = torch.utils.data.distributed.DistributedSampler(
                dataset, num_replicas=self.world_size, rank=self.rank
            )
        
        # 训练循环
        for epoch in range(self.config.num_epochs):
            # 同步随机种子
            for sampler in samplers.values():
                sampler.set_epoch(epoch)
            
            # 训练一个epoch
            self.train_epoch(ddp_models, datasets, samplers)
            
            # 同步模型参数
            self.synchronize_models(ddp_models)
```

### 模型并行策略

```python
class ModelParallelBlockNeRF:
    """模型并行的 Block-NeRF"""
    
    def __init__(self, config, device_map):
        self.config = config
        self.device_map = device_map  # {'encoder': 0, 'decoder': 1, ...}
        
    def setup_model_parallel(self, models):
        """设置模型并行"""
        parallel_models = {}
        
        for component, device_id in self.device_map.items():
            if component in models:
                parallel_models[component] = models[component].to(f'cuda:{device_id}')
        
        return parallel_models
    
    def forward_model_parallel(self, inputs, models):
        """模型并行前向传播"""
        # 输入分布到第一个设备
        current_device = self.device_map['encoder']
        x = inputs.to(f'cuda:{current_device}')
        
        # 编码器
        if 'encoder' in models:
            x = models['encoder'](x)
        
        # 传输到下一个设备
        if 'nerf_mlp' in models:
            next_device = self.device_map['nerf_mlp']
            x = x.to(f'cuda:{next_device}')
            x = models['nerf_mlp'](x)
        
        # 解码器
        if 'decoder' in models:
            final_device = self.device_map['decoder']
            x = x.to(f'cuda:{final_device}')
            x = models['decoder'](x)
        
        return x
```

---

## 训练监控与评估

### 训练指标

```python
class TrainingMetrics:
    """训练指标监控"""
    
    def __init__(self, config):
        self.config = config
        self.metrics_history = defaultdict(list)
        self.best_metrics = {}
        
    def update_metrics(self, epoch, losses, validation_results):
        """更新训练指标"""
        
        # 记录损失
        for loss_name, loss_value in losses.items():
            self.metrics_history[f'loss/{loss_name}'].append(loss_value)
        
        # 记录验证指标
        for metric_name, metric_value in validation_results.items():
            self.metrics_history[f'val/{metric_name}'].append(metric_value)
        
        # 更新最佳指标
        if 'psnr' in validation_results:
            current_psnr = validation_results['psnr']
            if 'best_psnr' not in self.best_metrics or current_psnr > self.best_metrics['best_psnr']:
                self.best_metrics['best_psnr'] = current_psnr
                self.best_metrics['best_epoch'] = epoch
        
        # 记录到tensorboard
        if self.config.use_tensorboard:
            self.log_to_tensorboard(epoch, losses, validation_results)
    
    def compute_validation_metrics(self, model, validation_dataset):
        """计算验证指标"""
        model.eval()
        metrics = {'psnr': [], 'ssim': [], 'lpips': []}
        
        with torch.no_grad():
            for batch in validation_dataset:
                # 渲染
                predicted = model.render(batch['rays_o'], batch['rays_d'])
                target = batch['rgb']
                
                # 计算PSNR
                psnr = compute_psnr(predicted, target)
                metrics['psnr'].append(psnr.item())
                
                # 计算SSIM
                ssim_score = compute_ssim(predicted, target)
                metrics['ssim'].append(ssim_score.item())
                
                # 计算LPIPS (感知损失)
                lpips_score = compute_lpips(predicted, target)
                metrics['lpips'].append(lpips_score.item())
        
        # 平均指标
        avg_metrics = {k: np.mean(v) for k, v in metrics.items()}
        return avg_metrics
```

### 模型保存与加载

```python
class ModelCheckpoint:
    """模型检查点管理"""
    
    def __init__(self, config):
        self.config = config
        self.checkpoint_dir = config.checkpoint_dir
        self.save_frequency = config.save_frequency
        
    def save_checkpoint(self, epoch, models, optimizers, metrics):
        """保存检查点"""
        checkpoint = {
            'epoch': epoch,
            'models': {},
            'optimizers': {},
            'metrics': metrics,
            'config': self.config
        }
        
        # 保存模型状态
        for name, model in models.items():
            if hasattr(model, 'module'):  # DDP包装的模型
                checkpoint['models'][name] = model.module.state_dict()
            else:
                checkpoint['models'][name] = model.state_dict()
        
        # 保存优化器状态
        for name, optimizer in optimizers.items():
            checkpoint['optimizers'][name] = optimizer.state_dict()
        
        # 保存文件
        checkpoint_path = os.path.join(
            self.checkpoint_dir, f'checkpoint_epoch_{epoch}.pth'
        )
        torch.save(checkpoint, checkpoint_path)
        
        # 保存最佳模型
        if metrics.get('is_best', False):
            best_path = os.path.join(self.checkpoint_dir, 'best_model.pth')
            torch.save(checkpoint, best_path)
    
    def load_checkpoint(self, checkpoint_path, models, optimizers=None):
        """加载检查点"""
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        
        # 加载模型参数
        for name, model in models.items():
            if name in checkpoint['models']:
                model.load_state_dict(checkpoint['models'][name])
        
        # 加载优化器参数
        if optimizers:
            for name, optimizer in optimizers.items():
                if name in checkpoint['optimizers']:
                    optimizer.load_state_dict(checkpoint['optimizers'][name])
        
        return checkpoint['epoch'], checkpoint['metrics']
```

---

## 训练配置示例

### 完整训练配置

```python
class BlockNeRFTrainingConfig:
    """Block-NeRF 训练配置"""
    
    def __init__(self):
        # 场景配置
        self.scene_bounds = [-100, -100, -10, 100, 100, 50]  # [x_min, y_min, z_min, x_max, y_max, z_max]
        self.block_size = 50.0  # 块大小(米)
        self.overlap_ratio = 0.3  # 重叠比例
        self.min_images_per_block = 100  # 每块最少图像数
        
        # 网络架构
        self.nerf_layers = 8
        self.nerf_hidden_dim = 256
        self.appearance_embedding_dim = 32
        self.visibility_hidden_dim = 512
        
        # 训练参数
        self.num_epochs = 200
        self.batch_size = 4096
        self.learning_rate = 5e-4
        self.lr_decay_steps = [100, 150]
        self.lr_decay_rate = 0.1
        
        # 损失权重
        self.rgb_loss_weight = 1.0
        self.depth_loss_weight = 0.1
        self.consistency_loss_weight = 0.5
        self.regularization_weight = 1e-6
        self.appearance_loss_weight = 0.1
        
        # 采样配置
        self.num_samples_coarse = 64
        self.num_samples_fine = 128
        self.perturb_samples = True
        self.white_background = False
        
        # 优化配置
        self.use_mixed_precision = True
        self.use_gradient_checkpointing = True
        self.gradient_clip_norm = 1.0
        
        # 分布式训练
        self.distributed = False
        self.world_size = 1
        self.use_model_parallel = False
        
        # 保存配置
        self.checkpoint_dir = './checkpoints'
        self.save_frequency = 10
        self.validation_frequency = 5
        
        # 监控配置
        self.use_tensorboard = True
        self.log_frequency = 100
        self.image_log_frequency = 1000

# 使用示例
config = BlockNeRFTrainingConfig()

# 可以根据需要修改配置
config.batch_size = 8192  # 增加批处理大小
config.num_epochs = 300   # 增加训练轮数
config.distributed = True # 启用分布式训练
```

---

此文档第二部分到此结束。第三部分将包含实际的训练脚本示例、调试技巧和常见问题解决方案。
