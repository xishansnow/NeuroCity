# SVRaster 训练机制详解 - 第三部分：损失函数设计与性能监控

## 概述

本文档是 SVRaster 训练机制详解的第三部分，重点介绍多重损失函数设计、正则化技术、性能监控与调试、模型评估指标以及训练故障诊断等关键技术。这些技术确保了 SVRaster 训练过程的质量保证和性能优化。

## 1. 多重损失函数设计

### 1.1 核心损失函数架构

SVRaster 使用多重损失函数来优化不同方面的渲染质量：

```python
class SVRasterLossFunction:
    """
    SVRaster 多重损失函数
    结合多种损失函数来全面优化渲染质量
    """
    
    def __init__(self, config: SVRasterTrainerConfig):
        self.config = config
        
        # 损失函数权重
        self.rgb_weight = config.loss_config.get('rgb_weight', 1.0)
        self.depth_weight = config.loss_config.get('depth_weight', 0.1)
        self.ssim_weight = config.loss_config.get('ssim_weight', 0.1)
        self.perceptual_weight = config.loss_config.get('perceptual_weight', 0.1)
        self.opacity_reg_weight = config.loss_config.get('opacity_reg_weight', 0.01)
        self.spatial_reg_weight = config.loss_config.get('spatial_reg_weight', 0.1)
        self.temporal_reg_weight = config.loss_config.get('temporal_reg_weight', 0.05)
        self.distortion_weight = config.loss_config.get('distortion_weight', 0.01)
        
        # 损失函数实例
        self.rgb_loss = RGBReconstructionLoss()
        self.depth_loss = DepthConsistencyLoss()
        self.ssim_loss = StructuralSimilarityLoss()
        self.perceptual_loss = PerceptualLoss()
        self.opacity_regularizer = OpacityRegularizer()
        self.spatial_regularizer = SpatialRegularizer()
        self.temporal_regularizer = TemporalRegularizer()
        self.distortion_loss = DistortionLoss()
        
        # 自适应权重调整
        self.adaptive_weights = AdaptiveWeightManager(config)
        
    def compute_loss(self, predictions: dict, targets: dict, 
                    model_outputs: dict, epoch: int) -> dict:
        """
        计算总损失
        
        Args:
            predictions: 模型预测结果
            targets: 目标值
            model_outputs: 模型输出（包含中间结果）
            epoch: 当前训练轮数
            
        Returns:
            损失字典
        """
        losses = {}
        
        # 获取自适应权重
        adaptive_weights = self.adaptive_weights.get_current_weights(epoch)
        
        # 1. RGB重构损失
        rgb_loss = self.rgb_loss(predictions['rgb'], targets['rgb'])
        losses['rgb_loss'] = rgb_loss * self.rgb_weight * adaptive_weights.get('rgb', 1.0)
        
        # 2. 深度一致性损失
        if 'depth' in predictions and 'depth' in targets:
            depth_loss = self.depth_loss(predictions['depth'], targets['depth'])
            losses['depth_loss'] = depth_loss * self.depth_weight * adaptive_weights.get('depth', 1.0)
        
        # 3. 结构相似性损失
        ssim_loss = self.ssim_loss(predictions['rgb'], targets['rgb'])
        losses['ssim_loss'] = ssim_loss * self.ssim_weight * adaptive_weights.get('ssim', 1.0)
        
        # 4. 感知损失
        perceptual_loss = self.perceptual_loss(predictions['rgb'], targets['rgb'])
        losses['perceptual_loss'] = perceptual_loss * self.perceptual_weight * adaptive_weights.get('perceptual', 1.0)
        
        # 5. 透明度正则化
        if 'opacity' in model_outputs:
            opacity_reg = self.opacity_regularizer(model_outputs['opacity'])
            losses['opacity_reg'] = opacity_reg * self.opacity_reg_weight
        
        # 6. 空间正则化
        if 'voxel_features' in model_outputs:
            spatial_reg = self.spatial_regularizer(
                model_outputs['voxel_features'], 
                model_outputs.get('voxel_positions')
            )
            losses['spatial_reg'] = spatial_reg * self.spatial_reg_weight
        
        # 7. 时间正则化（用于动态场景）
        if 'temporal_features' in model_outputs:
            temporal_reg = self.temporal_regularizer(model_outputs['temporal_features'])
            losses['temporal_reg'] = temporal_reg * self.temporal_reg_weight
        
        # 8. 失真损失
        if 'weights' in model_outputs and 'sample_distances' in model_outputs:
            distortion_loss = self.distortion_loss(
                model_outputs['weights'], 
                model_outputs['sample_distances']
            )
            losses['distortion_loss'] = distortion_loss * self.distortion_weight
        
        # 9. 稀疏性损失
        if 'density' in model_outputs:
            sparsity_loss = self._compute_sparsity_loss(model_outputs['density'])
            losses['sparsity_loss'] = sparsity_loss * adaptive_weights.get('sparsity', 0.001)
        
        # 10. 几何一致性损失
        if 'normals' in predictions and 'normals' in targets:
            normal_loss = self._compute_normal_loss(predictions['normals'], targets['normals'])
            losses['normal_loss'] = normal_loss * adaptive_weights.get('normal', 0.01)
        
        # 计算总损失
        total_loss = sum(losses.values())
        losses['total_loss'] = total_loss
        
        return losses
    
    def _compute_sparsity_loss(self, density: torch.Tensor) -> torch.Tensor:
        """
        计算稀疏性损失
        促进稀疏的体素表示
        """
        # L1正则化促进稀疏性
        l1_loss = torch.mean(torch.abs(density))
        
        # 二值化损失
        activated_density = torch.sigmoid(density)
        binary_loss = torch.mean(activated_density * (1 - activated_density))
        
        return l1_loss + 0.1 * binary_loss
    
    def _compute_normal_loss(self, pred_normals: torch.Tensor, 
                           target_normals: torch.Tensor) -> torch.Tensor:
        """
        计算法线一致性损失
        """
        # 余弦相似度损失
        cosine_sim = torch.nn.functional.cosine_similarity(
            pred_normals, target_normals, dim=-1
        )
        
        # 转换为损失（1 - 相似度）
        normal_loss = torch.mean(1.0 - cosine_sim)
        
        return normal_loss

class RGBReconstructionLoss:
    """
    RGB重构损失函数
    支持多种损失类型
    """
    
    def __init__(self, loss_type: str = 'huber'):
        self.loss_type = loss_type
        
    def __call__(self, pred_rgb: torch.Tensor, target_rgb: torch.Tensor) -> torch.Tensor:
        """
        计算RGB重构损失
        """
        if self.loss_type == 'mse':
            return torch.nn.functional.mse_loss(pred_rgb, target_rgb)
        elif self.loss_type == 'l1':
            return torch.nn.functional.l1_loss(pred_rgb, target_rgb)
        elif self.loss_type == 'huber':
            return torch.nn.functional.huber_loss(pred_rgb, target_rgb, delta=0.1)
        elif self.loss_type == 'smooth_l1':
            return torch.nn.functional.smooth_l1_loss(pred_rgb, target_rgb)
        else:
            raise ValueError(f"不支持的损失类型: {self.loss_type}")

class DepthConsistencyLoss:
    """
    深度一致性损失
    """
    
    def __init__(self, loss_type: str = 'l1', ignore_invalid: bool = True):
        self.loss_type = loss_type
        self.ignore_invalid = ignore_invalid
        
    def __call__(self, pred_depth: torch.Tensor, target_depth: torch.Tensor) -> torch.Tensor:
        """
        计算深度一致性损失
        """
        # 处理无效深度值
        if self.ignore_invalid:
            valid_mask = (target_depth > 0) & (target_depth < 100)  # 假设有效深度范围
            if torch.sum(valid_mask) == 0:
                return torch.tensor(0.0, device=pred_depth.device)
            
            pred_depth = pred_depth[valid_mask]
            target_depth = target_depth[valid_mask]
        
        # 深度归一化
        pred_depth_norm = pred_depth / (torch.max(pred_depth) + 1e-8)
        target_depth_norm = target_depth / (torch.max(target_depth) + 1e-8)
        
        if self.loss_type == 'l1':
            return torch.nn.functional.l1_loss(pred_depth_norm, target_depth_norm)
        elif self.loss_type == 'mse':
            return torch.nn.functional.mse_loss(pred_depth_norm, target_depth_norm)
        elif self.loss_type == 'huber':
            return torch.nn.functional.huber_loss(pred_depth_norm, target_depth_norm)
        else:
            raise ValueError(f"不支持的损失类型: {self.loss_type}")

class StructuralSimilarityLoss:
    """
    结构相似性损失 (SSIM)
    """
    
    def __init__(self, window_size: int = 11, reduction: str = 'mean'):
        self.window_size = window_size
        self.reduction = reduction
        
    def __call__(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        计算SSIM损失
        """
        # 如果输入是像素级的，需要重塑为图像
        if pred.dim() == 2:
            # 假设是正方形图像
            size = int(np.sqrt(pred.shape[0]))
            pred = pred.view(size, size, -1).permute(2, 0, 1).unsqueeze(0)
            target = target.view(size, size, -1).permute(2, 0, 1).unsqueeze(0)
        
        # 计算SSIM
        ssim_value = self._ssim(pred, target)
        
        # 转换为损失
        ssim_loss = 1.0 - ssim_value
        
        if self.reduction == 'mean':
            return torch.mean(ssim_loss)
        elif self.reduction == 'sum':
            return torch.sum(ssim_loss)
        else:
            return ssim_loss
    
    def _ssim(self, img1: torch.Tensor, img2: torch.Tensor) -> torch.Tensor:
        """
        计算SSIM值
        """
        C1 = 0.01 ** 2
        C2 = 0.03 ** 2
        
        # 创建高斯窗口
        window = self._create_window(img1.shape[1], self.window_size, img1.device)
        
        mu1 = torch.nn.functional.conv2d(img1, window, padding=self.window_size//2, groups=img1.shape[1])
        mu2 = torch.nn.functional.conv2d(img2, window, padding=self.window_size//2, groups=img2.shape[1])
        
        mu1_sq = mu1 ** 2
        mu2_sq = mu2 ** 2
        mu1_mu2 = mu1 * mu2
        
        sigma1_sq = torch.nn.functional.conv2d(img1**2, window, padding=self.window_size//2, groups=img1.shape[1]) - mu1_sq
        sigma2_sq = torch.nn.functional.conv2d(img2**2, window, padding=self.window_size//2, groups=img2.shape[1]) - mu2_sq
        sigma12 = torch.nn.functional.conv2d(img1*img2, window, padding=self.window_size//2, groups=img1.shape[1]) - mu1_mu2
        
        ssim_map = ((2*mu1_mu2 + C1)*(2*sigma12 + C2))/((mu1_sq + mu2_sq + C1)*(sigma1_sq + sigma2_sq + C2))
        
        return ssim_map
    
    def _create_window(self, channels: int, window_size: int, device: torch.device) -> torch.Tensor:
        """
        创建高斯窗口
        """
        coords = torch.arange(window_size, dtype=torch.float32, device=device)
        coords = coords - window_size // 2
        
        g = torch.exp(-(coords ** 2) / (2.0 * (window_size / 6.0) ** 2))
        g = g / g.sum()
        
        window_2d = g.outer(g)
        window = window_2d.expand(channels, 1, window_size, window_size).contiguous()
        
        return window
```

### 1.2 自适应权重管理

```python
class AdaptiveWeightManager:
    """
    自适应权重管理器
    根据训练进度动态调整损失函数权重
    """
    
    def __init__(self, config: SVRasterTrainerConfig):
        self.config = config
        self.weight_schedule = self._create_weight_schedule()
        
    def _create_weight_schedule(self) -> dict:
        """
        创建权重调度表
        """
        schedule = {
            'rgb': [
                {'epoch': 0, 'weight': 1.0},
                {'epoch': 50, 'weight': 0.8},
                {'epoch': 100, 'weight': 0.6}
            ],
            'ssim': [
                {'epoch': 0, 'weight': 0.05},
                {'epoch': 20, 'weight': 0.1},
                {'epoch': 50, 'weight': 0.15}
            ],
            'perceptual': [
                {'epoch': 0, 'weight': 0.0},
                {'epoch': 30, 'weight': 0.1},
                {'epoch': 70, 'weight': 0.2}
            ],
            'sparsity': [
                {'epoch': 0, 'weight': 0.01},
                {'epoch': 40, 'weight': 0.005},
                {'epoch': 80, 'weight': 0.001}
            ]
        }
        
        return schedule
    
    def get_current_weights(self, epoch: int) -> dict:
        """
        获取当前epoch的权重
        """
        weights = {}
        
        for loss_name, schedule in self.weight_schedule.items():
            # 线性插值计算当前权重
            current_weight = self._interpolate_weight(schedule, epoch)
            weights[loss_name] = current_weight
        
        return weights
    
    def _interpolate_weight(self, schedule: List[dict], epoch: int) -> float:
        """
        插值计算权重
        """
        if epoch <= schedule[0]['epoch']:
            return schedule[0]['weight']
        
        if epoch >= schedule[-1]['epoch']:
            return schedule[-1]['weight']
        
        # 找到当前epoch所在的区间
        for i in range(len(schedule) - 1):
            if schedule[i]['epoch'] <= epoch <= schedule[i+1]['epoch']:
                # 线性插值
                t = (epoch - schedule[i]['epoch']) / (schedule[i+1]['epoch'] - schedule[i]['epoch'])
                weight = schedule[i]['weight'] + t * (schedule[i+1]['weight'] - schedule[i]['weight'])
                return weight
        
        return schedule[-1]['weight']
```

## 2. 正则化技术

### 2.1 空间正则化

```python
class SpatialRegularizer:
    """
    空间正则化器
    促进体素特征的空间连续性和一致性
    """
    
    def __init__(self, config: SVRasterTrainerConfig):
        self.config = config
        self.neighbor_threshold = 1.5  # 邻居判断阈值
        
    def __call__(self, voxel_features: torch.Tensor, 
                voxel_positions: torch.Tensor) -> torch.Tensor:
        """
        计算空间正则化损失
        
        Args:
            voxel_features: 体素特征 [N, F]
            voxel_positions: 体素位置 [N, 3]
            
        Returns:
            空间正则化损失
        """
        if voxel_features is None or voxel_positions is None:
            return torch.tensor(0.0, device=voxel_features.device if voxel_features is not None else 'cuda')
        
        # 计算体素间的距离
        distances = torch.cdist(voxel_positions, voxel_positions)
        
        # 找到邻居关系
        neighbor_mask = (distances > 0) & (distances < self.neighbor_threshold)
        
        # 计算邻居特征差异
        spatial_loss = 0.0
        num_pairs = 0
        
        for i in range(voxel_features.shape[0]):
            neighbors = torch.where(neighbor_mask[i])[0]
            
            if len(neighbors) > 0:
                current_features = voxel_features[i]
                neighbor_features = voxel_features[neighbors]
                
                # 计算特征差异
                feature_diff = torch.norm(
                    current_features.unsqueeze(0) - neighbor_features, 
                    dim=1
                )
                
                # 距离加权
                neighbor_distances = distances[i, neighbors]
                weights = torch.exp(-neighbor_distances)
                
                weighted_diff = torch.sum(weights * feature_diff) / torch.sum(weights)
                spatial_loss += weighted_diff
                num_pairs += 1
        
        if num_pairs > 0:
            spatial_loss = spatial_loss / num_pairs
        
        return spatial_loss

class TemporalRegularizer:
    """
    时间正则化器
    用于动态场景的时间一致性
    """
    
    def __init__(self, config: SVRasterTrainerConfig):
        self.config = config
        self.history_length = 5
        self.feature_history = []
        
    def __call__(self, current_features: torch.Tensor) -> torch.Tensor:
        """
        计算时间正则化损失
        """
        if len(self.feature_history) == 0:
            self.feature_history.append(current_features.detach())
            return torch.tensor(0.0, device=current_features.device)
        
        temporal_loss = 0.0
        
        # 与历史特征比较
        for i, hist_features in enumerate(self.feature_history):
            weight = 0.9 ** (len(self.feature_history) - i)  # 指数衰减权重
            
            feature_diff = torch.norm(current_features - hist_features, dim=1)
            temporal_loss += weight * torch.mean(feature_diff)
        
        # 更新历史
        self.feature_history.append(current_features.detach())
        if len(self.feature_history) > self.history_length:
            self.feature_history.pop(0)
        
        return temporal_loss

class OpacityRegularizer:
    """
    透明度正则化器
    控制体素的透明度分布
    """
    
    def __init__(self, config: SVRasterTrainerConfig):
        self.config = config
        self.sparsity_weight = 0.01
        self.entropy_weight = 0.001
        
    def __call__(self, opacity: torch.Tensor) -> torch.Tensor:
        """
        计算透明度正则化损失
        """
        # 1. 稀疏性正则化
        sparsity_loss = torch.mean(opacity)
        
        # 2. 熵正则化（促进二值化）
        epsilon = 1e-8
        entropy_loss = -torch.mean(
            opacity * torch.log(opacity + epsilon) + 
            (1 - opacity) * torch.log(1 - opacity + epsilon)
        )
        
        # 3. 平滑度正则化
        if opacity.dim() > 1:
            # 假设opacity有空间结构
            smooth_loss = self._compute_smoothness_loss(opacity)
        else:
            smooth_loss = torch.tensor(0.0, device=opacity.device)
        
        total_loss = (self.sparsity_weight * sparsity_loss + 
                     self.entropy_weight * entropy_loss + 
                     0.001 * smooth_loss)
        
        return total_loss
    
    def _compute_smoothness_loss(self, opacity: torch.Tensor) -> torch.Tensor:
        """
        计算平滑度损失
        """
        # 计算梯度
        grad_x = torch.diff(opacity, dim=-1)
        grad_y = torch.diff(opacity, dim=-2) if opacity.dim() > 2 else torch.tensor(0.0)
        
        # 总变差损失
        tv_loss = torch.mean(torch.abs(grad_x)) + torch.mean(torch.abs(grad_y))
        
        return tv_loss
```

### 2.2 高级正则化策略

```python
class AdvancedRegularization:
    """
    高级正则化策略
    包含多种正则化技术
    """
    
    def __init__(self, config: SVRasterTrainerConfig):
        self.config = config
        self.dropout_rate = 0.1
        self.noise_std = 0.01
        
    def apply_feature_dropout(self, features: torch.Tensor, training: bool = True) -> torch.Tensor:
        """
        特征dropout正则化
        """
        if training:
            return torch.nn.functional.dropout(features, p=self.dropout_rate, training=True)
        return features
    
    def apply_feature_noise(self, features: torch.Tensor, training: bool = True) -> torch.Tensor:
        """
        特征噪声正则化
        """
        if training:
            noise = torch.randn_like(features) * self.noise_std
            return features + noise
        return features
    
    def compute_weight_decay_loss(self, model: torch.nn.Module) -> torch.Tensor:
        """
        计算权重衰减损失
        """
        l2_loss = 0.0
        
        for name, param in model.named_parameters():
            if 'weight' in name and param.requires_grad:
                l2_loss += torch.norm(param) ** 2
        
        return l2_loss
    
    def compute_gradient_penalty(self, model: torch.nn.Module, 
                               real_samples: torch.Tensor,
                               fake_samples: torch.Tensor) -> torch.Tensor:
        """
        梯度惩罚正则化
        """
        # 插值样本
        alpha = torch.rand(real_samples.shape[0], 1, device=real_samples.device)
        interpolates = alpha * real_samples + (1 - alpha) * fake_samples
        interpolates.requires_grad_(True)
        
        # 计算梯度
        outputs = model(interpolates)
        gradients = torch.autograd.grad(
            outputs=outputs.sum(),
            inputs=interpolates,
            create_graph=True,
            retain_graph=True
        )[0]
        
        # 梯度惩罚
        gradient_penalty = torch.mean((torch.norm(gradients, dim=1) - 1) ** 2)
        
        return gradient_penalty
```

## 3. 性能监控与调试

### 3.1 训练监控系统

```python
class TrainingMonitor:
    """
    训练监控系统
    全面监控训练过程的各项指标
    """
    
    def __init__(self, config: SVRasterTrainerConfig):
        self.config = config
        self.metrics_history = {
            'loss': [],
            'psnr': [],
            'ssim': [],
            'gpu_memory': [],
            'training_speed': [],
            'learning_rate': []
        }
        
        # 性能统计
        self.performance_stats = {
            'total_training_time': 0.0,
            'average_epoch_time': 0.0,
            'peak_gpu_memory': 0.0,
            'total_iterations': 0
        }
        
        # 警告系统
        self.warning_system = WarningSystem()
        
    def log_epoch_metrics(self, epoch: int, metrics: dict, elapsed_time: float):
        """
        记录epoch指标
        """
        # 记录基础指标
        self.metrics_history['loss'].append(metrics.get('loss', 0.0))
        self.metrics_history['psnr'].append(metrics.get('psnr', 0.0))
        self.metrics_history['ssim'].append(metrics.get('ssim', 0.0))
        
        # 记录系统指标
        gpu_memory = self._get_gpu_memory_usage()
        self.metrics_history['gpu_memory'].append(gpu_memory)
        
        training_speed = metrics.get('samples_per_second', 0.0)
        self.metrics_history['training_speed'].append(training_speed)
        
        lr = metrics.get('learning_rate', 0.0)
        self.metrics_history['learning_rate'].append(lr)
        
        # 更新性能统计
        self.performance_stats['total_training_time'] += elapsed_time
        self.performance_stats['average_epoch_time'] = (
            self.performance_stats['total_training_time'] / (epoch + 1)
        )
        self.performance_stats['peak_gpu_memory'] = max(
            self.performance_stats['peak_gpu_memory'], gpu_memory
        )
        
        # 检查警告条件
        self.warning_system.check_warnings(metrics, self.metrics_history)
        
        # 记录到日志
        logger.info(f"Epoch {epoch}: Loss={metrics.get('loss', 0.0):.4f}, "
                   f"PSNR={metrics.get('psnr', 0.0):.2f}, "
                   f"GPU Memory={gpu_memory:.1f}MB, "
                   f"Speed={training_speed:.1f} samples/sec")
    
    def _get_gpu_memory_usage(self) -> float:
        """
        获取GPU内存使用量（MB）
        """
        if torch.cuda.is_available():
            return torch.cuda.memory_allocated() / 1024 / 1024
        return 0.0
    
    def get_training_summary(self) -> dict:
        """
        获取训练摘要
        """
        if not self.metrics_history['loss']:
            return {}
        
        summary = {
            'total_epochs': len(self.metrics_history['loss']),
            'best_loss': min(self.metrics_history['loss']),
            'best_psnr': max(self.metrics_history['psnr']) if self.metrics_history['psnr'] else 0.0,
            'best_ssim': max(self.metrics_history['ssim']) if self.metrics_history['ssim'] else 0.0,
            'final_loss': self.metrics_history['loss'][-1],
            'final_psnr': self.metrics_history['psnr'][-1] if self.metrics_history['psnr'] else 0.0,
            'average_training_speed': np.mean(self.metrics_history['training_speed']) if self.metrics_history['training_speed'] else 0.0,
            'peak_gpu_memory': self.performance_stats['peak_gpu_memory'],
            'total_training_time': self.performance_stats['total_training_time'],
            'average_epoch_time': self.performance_stats['average_epoch_time']
        }
        
        return summary
    
    def plot_training_curves(self, save_path: str = None):
        """
        绘制训练曲线
        """
        import matplotlib.pyplot as plt
        
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        
        # 损失曲线
        axes[0, 0].plot(self.metrics_history['loss'])
        axes[0, 0].set_title('Training Loss')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        
        # PSNR曲线
        if self.metrics_history['psnr']:
            axes[0, 1].plot(self.metrics_history['psnr'])
            axes[0, 1].set_title('PSNR')
            axes[0, 1].set_xlabel('Epoch')
            axes[0, 1].set_ylabel('PSNR (dB)')
        
        # SSIM曲线
        if self.metrics_history['ssim']:
            axes[0, 2].plot(self.metrics_history['ssim'])
            axes[0, 2].set_title('SSIM')
            axes[0, 2].set_xlabel('Epoch')
            axes[0, 2].set_ylabel('SSIM')
        
        # GPU内存使用
        if self.metrics_history['gpu_memory']:
            axes[1, 0].plot(self.metrics_history['gpu_memory'])
            axes[1, 0].set_title('GPU Memory Usage')
            axes[1, 0].set_xlabel('Epoch')
            axes[1, 0].set_ylabel('Memory (MB)')
        
        # 训练速度
        if self.metrics_history['training_speed']:
            axes[1, 1].plot(self.metrics_history['training_speed'])
            axes[1, 1].set_title('Training Speed')
            axes[1, 1].set_xlabel('Epoch')
            axes[1, 1].set_ylabel('Samples/sec')
        
        # 学习率
        if self.metrics_history['learning_rate']:
            axes[1, 2].plot(self.metrics_history['learning_rate'])
            axes[1, 2].set_title('Learning Rate')
            axes[1, 2].set_xlabel('Epoch')
            axes[1, 2].set_ylabel('Learning Rate')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path)
        else:
            plt.show()

class WarningSystem:
    """
    训练警告系统
    检测训练过程中的异常情况
    """
    
    def __init__(self):
        self.warnings = []
        
    def check_warnings(self, current_metrics: dict, history: dict):
        """
        检查警告条件
        """
        # 检查损失爆炸
        if 'loss' in current_metrics and current_metrics['loss'] > 100:
            self.warnings.append({
                'type': 'loss_explosion',
                'message': f"损失爆炸: {current_metrics['loss']:.2f}",
                'epoch': len(history['loss'])
            })
        
        # 检查损失不下降
        if len(history['loss']) >= 10:
            recent_losses = history['loss'][-10:]
            if all(l >= recent_losses[0] * 0.99 for l in recent_losses[1:]):
                self.warnings.append({
                    'type': 'loss_plateau',
                    'message': "损失停止下降，可能进入平台期",
                    'epoch': len(history['loss'])
                })
        
        # 检查GPU内存过高
        if 'gpu_memory' in history and history['gpu_memory']:
            current_memory = history['gpu_memory'][-1]
            if current_memory > 20000:  # 20GB
                self.warnings.append({
                    'type': 'high_memory_usage',
                    'message': f"GPU内存使用过高: {current_memory:.1f}MB",
                    'epoch': len(history['loss'])
                })
        
        # 检查训练速度下降
        if len(history.get('training_speed', [])) >= 5:
            recent_speeds = history['training_speed'][-5:]
            if all(s < recent_speeds[0] * 0.8 for s in recent_speeds[1:]):
                self.warnings.append({
                    'type': 'speed_degradation',
                    'message': "训练速度显著下降",
                    'epoch': len(history['loss'])
                })
    
    def get_recent_warnings(self, last_n: int = 5) -> List[dict]:
        """
        获取最近的警告
        """
        return self.warnings[-last_n:] if self.warnings else []
```

### 3.2 实时可视化系统

```python
class RealTimeVisualizer:
    """
    实时可视化系统
    为训练过程提供实时的可视化反馈
    """
    
    def __init__(self, config: SVRasterTrainerConfig):
        self.config = config
        self.use_tensorboard = True
        self.use_wandb = getattr(config, 'use_wandb', False)
        
        # 初始化日志器
        if self.use_tensorboard:
            from torch.utils.tensorboard import SummaryWriter
            self.tb_writer = SummaryWriter(config.log_dir)
        
        if self.use_wandb:
            import wandb
            wandb.init(project="svraster_training", config=config)
            self.wandb = wandb
        
    def log_scalars(self, step: int, scalars: dict, tag_prefix: str = ""):
        """
        记录标量值
        """
        for name, value in scalars.items():
            tag = f"{tag_prefix}/{name}" if tag_prefix else name
            
            if self.use_tensorboard:
                self.tb_writer.add_scalar(tag, value, step)
            
            if self.use_wandb:
                self.wandb.log({tag: value}, step=step)
    
    def log_images(self, step: int, images: dict, tag_prefix: str = ""):
        """
        记录图像
        """
        for name, image in images.items():
            tag = f"{tag_prefix}/{name}" if tag_prefix else name
            
            if self.use_tensorboard:
                self.tb_writer.add_image(tag, image, step)
            
            if self.use_wandb:
                self.wandb.log({tag: self.wandb.Image(image)}, step=step)
    
    def log_histograms(self, step: int, tensors: dict, tag_prefix: str = ""):
        """
        记录直方图
        """
        for name, tensor in tensors.items():
            tag = f"{tag_prefix}/{name}" if tag_prefix else name
            
            if self.use_tensorboard:
                self.tb_writer.add_histogram(tag, tensor, step)
    
    def log_model_graph(self, model: torch.nn.Module, sample_input: torch.Tensor):
        """
        记录模型图
        """
        if self.use_tensorboard:
            self.tb_writer.add_graph(model, sample_input)
    
    def log_3d_visualization(self, step: int, voxel_data: dict):
        """
        记录3D可视化
        """
        # 这里可以添加3D体素可视化逻辑
        # 例如使用Open3D或其他3D可视化库
        pass
    
    def close(self):
        """
        关闭可视化器
        """
        if self.use_tensorboard:
            self.tb_writer.close()
        
        if self.use_wandb:
            self.wandb.finish()
```

## 4. 模型评估指标

### 4.1 图像质量评估

```python
class ImageQualityEvaluator:
    """
    图像质量评估器
    提供多种图像质量评估指标
    """
    
    def __init__(self):
        self.metrics = {
            'psnr': self._compute_psnr,
            'ssim': self._compute_ssim,
            'lpips': self._compute_lpips,
            'mse': self._compute_mse,
            'mae': self._compute_mae
        }
    
    def evaluate(self, pred_images: torch.Tensor, 
                target_images: torch.Tensor) -> dict:
        """
        评估图像质量
        
        Args:
            pred_images: 预测图像 [N, C, H, W] 或 [N, H, W, C]
            target_images: 目标图像 [N, C, H, W] 或 [N, H, W, C]
            
        Returns:
            评估指标字典
        """
        results = {}
        
        # 确保图像格式一致
        pred_images = self._normalize_images(pred_images)
        target_images = self._normalize_images(target_images)
        
        # 计算各种指标
        for metric_name, metric_func in self.metrics.items():
            try:
                value = metric_func(pred_images, target_images)
                results[metric_name] = value
            except Exception as e:
                logger.warning(f"计算{metric_name}失败: {e}")
                results[metric_name] = 0.0
        
        return results
    
    def _normalize_images(self, images: torch.Tensor) -> torch.Tensor:
        """
        归一化图像格式
        """
        # 确保值在[0, 1]范围内
        images = torch.clamp(images, 0, 1)
        
        # 确保格式为[N, C, H, W]
        if images.dim() == 4 and images.shape[-1] <= 4:  # 可能是[N, H, W, C]
            images = images.permute(0, 3, 1, 2)
        
        return images
    
    def _compute_psnr(self, pred: torch.Tensor, target: torch.Tensor) -> float:
        """
        计算PSNR
        """
        mse = torch.mean((pred - target) ** 2)
        if mse == 0:
            return float('inf')
        
        psnr = 20 * torch.log10(1.0 / torch.sqrt(mse))
        return psnr.item()
    
    def _compute_ssim(self, pred: torch.Tensor, target: torch.Tensor) -> float:
        """
        计算SSIM
        """
        # 简化的SSIM实现
        from torchmetrics.image import StructuralSimilarityIndexMeasure
        
        ssim_metric = StructuralSimilarityIndexMeasure()
        ssim_value = ssim_metric(pred, target)
        
        return ssim_value.item()
    
    def _compute_lpips(self, pred: torch.Tensor, target: torch.Tensor) -> float:
        """
        计算LPIPS
        """
        try:
            from torchmetrics.image import LearnedPerceptualImagePatchSimilarity
            
            lpips_metric = LearnedPerceptualImagePatchSimilarity(net_type='alex')
            lpips_value = lpips_metric(pred, target)
            
            return lpips_value.item()
        except ImportError:
            # 如果没有安装lpips，使用简化版本
            logger.warning("LPIPS库未安装，使用简化计算")
            return self._simplified_lpips(pred, target)
    
    def _simplified_lpips(self, pred: torch.Tensor, target: torch.Tensor) -> float:
        """
        简化的感知损失计算
        """
        # 使用VGG特征的简化版本
        diff = pred - target
        perceptual_loss = torch.mean(torch.abs(diff))
        
        return perceptual_loss.item()
    
    def _compute_mse(self, pred: torch.Tensor, target: torch.Tensor) -> float:
        """
        计算MSE
        """
        mse = torch.mean((pred - target) ** 2)
        return mse.item()
    
    def _compute_mae(self, pred: torch.Tensor, target: torch.Tensor) -> float:
        """
        计算MAE
        """
        mae = torch.mean(torch.abs(pred - target))
        return mae.item()
```

### 4.2 3D几何评估

```python
class GeometryEvaluator:
    """
    3D几何评估器
    评估重建几何的质量
    """
    
    def __init__(self):
        pass
    
    def evaluate_depth_accuracy(self, pred_depth: torch.Tensor, 
                               target_depth: torch.Tensor,
                               valid_mask: torch.Tensor = None) -> dict:
        """
        评估深度精度
        """
        if valid_mask is not None:
            pred_depth = pred_depth[valid_mask]
            target_depth = target_depth[valid_mask]
        
        # 深度误差指标
        abs_error = torch.abs(pred_depth - target_depth)
        sq_error = (pred_depth - target_depth) ** 2
        
        results = {
            'depth_mae': torch.mean(abs_error).item(),
            'depth_mse': torch.mean(sq_error).item(),
            'depth_rmse': torch.sqrt(torch.mean(sq_error)).item(),
            'depth_abs_rel': torch.mean(abs_error / target_depth).item(),
            'depth_sq_rel': torch.mean(sq_error / target_depth).item()
        }
        
        # 阈值精度
        thresholds = [1.25, 1.25**2, 1.25**3]
        for i, thresh in enumerate(thresholds):
            ratio = torch.max(pred_depth / target_depth, target_depth / pred_depth)
            accuracy = torch.mean((ratio < thresh).float()).item()
            results[f'depth_acc_{i+1}'] = accuracy
        
        return results
    
    def evaluate_normal_accuracy(self, pred_normals: torch.Tensor,
                                target_normals: torch.Tensor) -> dict:
        """
        评估法线精度
        """
        # 归一化法线
        pred_normals = torch.nn.functional.normalize(pred_normals, dim=-1)
        target_normals = torch.nn.functional.normalize(target_normals, dim=-1)
        
        # 角度误差
        cosine_sim = torch.sum(pred_normals * target_normals, dim=-1)
        cosine_sim = torch.clamp(cosine_sim, -1, 1)  # 数值稳定性
        
        angle_error = torch.acos(torch.abs(cosine_sim))  # 使用绝对值处理法线方向
        
        results = {
            'normal_mae': torch.mean(angle_error).item(),
            'normal_median': torch.median(angle_error).item(),
            'normal_mean_cosine': torch.mean(cosine_sim).item()
        }
        
        # 角度阈值精度
        angle_thresholds = [5, 10, 15, 30]  # 度
        for thresh in angle_thresholds:
            thresh_rad = math.radians(thresh)
            accuracy = torch.mean((angle_error < thresh_rad).float()).item()
            results[f'normal_acc_{thresh}deg'] = accuracy
        
        return results
```

## 总结

SVRaster 的损失函数设计与性能监控系统包含以下关键组件：

1. **多重损失函数**：RGB重构、深度一致性、SSIM、感知损失等多种损失函数的有机结合
2. **自适应权重管理**：根据训练进度动态调整不同损失函数的权重
3. **全面正则化**：空间、时间、透明度等多种正则化技术确保训练稳定性
4. **实时监控系统**：全方位监控训练过程，及时发现和处理异常情况
5. **综合评估指标**：图像质量和3D几何质量的全面评估体系

这些技术的综合应用确保了 SVRaster 训练过程的高质量和高效率，为获得优秀的渲染效果提供了坚实的保障。

**系列文档总结**：
- 第一部分：基础训练架构与配置
- 第二部分：自适应细分与渐进式训练
- 第三部分：损失函数设计与性能监控

这三部分文档全面覆盖了 SVRaster 的训练机制，为用户提供了完整的训练技术参考和实现指南。
