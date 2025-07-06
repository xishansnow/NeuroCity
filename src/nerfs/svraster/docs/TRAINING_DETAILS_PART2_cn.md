# SVRaster 训练机制详解 - 第二部分：自适应细分与渐进式训练

## 概述

本文档是 SVRaster 训练机制详解的第二部分，重点介绍自适应体素细分、智能剪枝策略、渐进式训练技术以及动态负载均衡等高级训练技术。这些技术是 SVRaster 实现高效训练和优质渲染效果的关键。

## 1. 自适应体素细分机制

### 1.1 细分策略设计

自适应体素细分是 SVRaster 的核心技术之一，它根据渲染误差和梯度信息动态调整体素分辨率：

```python
class AdaptiveSubdivision:
    """
    自适应体素细分管理器
    根据渲染误差和梯度信息动态细分体素
    """
    
    def __init__(self, config: SVRasterTrainerConfig):
        self.config = config
        self.subdivision_history = []
        self.error_threshold = config.subdivision_threshold
        self.max_level = config.max_subdivision_level
        
        # 细分统计
        self.subdivision_stats = {
            'total_subdivisions': 0,
            'successful_subdivisions': 0,
            'failed_subdivisions': 0,
            'subdivision_epochs': []
        }
    
    def should_subdivide(self, epoch: int, model: SVRasterModel) -> bool:
        """
        判断是否应该进行细分
        
        Args:
            epoch: 当前训练轮数
            model: SVRaster模型
            
        Returns:
            是否应该细分
        """
        # 检查基本条件
        if not self.config.enable_subdivision:
            return False
        
        if epoch < self.config.subdivision_start_epoch:
            return False
        
        if (epoch - self.config.subdivision_start_epoch) % self.config.subdivision_interval != 0:
            return False
        
        # 检查当前体素状态
        current_level = self._get_current_max_level(model)
        if current_level >= self.max_level:
            logger.info(f"已达到最大细分等级 {self.max_level}")
            return False
        
        # 检查细分条件
        subdivision_needed = self._analyze_subdivision_need(model)
        
        return subdivision_needed
    
    def perform_subdivision(self, model: SVRasterModel, 
                          training_data: torch.utils.data.DataLoader) -> bool:
        """
        执行体素细分
        
        Args:
            model: SVRaster模型
            training_data: 训练数据
            
        Returns:
            细分是否成功
        """
        logger.info("开始自适应体素细分")
        
        # 1. 分析需要细分的体素
        voxels_to_subdivide = self._identify_subdivision_candidates(model, training_data)
        
        if len(voxels_to_subdivide) == 0:
            logger.info("没有需要细分的体素")
            return False
        
        # 2. 执行细分操作
        success = self._execute_subdivision(model, voxels_to_subdivide)
        
        # 3. 更新统计信息
        self._update_subdivision_stats(success, len(voxels_to_subdivide))
        
        if success:
            logger.info(f"成功细分 {len(voxels_to_subdivide)} 个体素")
        else:
            logger.warning("体素细分失败")
        
        return success
    
    def _identify_subdivision_candidates(self, model: SVRasterModel, 
                                       training_data: torch.utils.data.DataLoader) -> List[int]:
        """
        识别需要细分的体素候选
        """
        candidates = []
        model.eval()
        
        # 收集渲染误差信息
        error_map = {}
        gradient_map = {}
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(training_data):
                if batch_idx >= 10:  # 只采样部分数据
                    break
                
                # 获取渲染结果和误差
                rays = batch['rays'].to(model.device)
                target_rgb = batch['rgb'].to(model.device)
                
                # 启用梯度计算用于误差分析
                rays.requires_grad_(True)
                
                outputs = model(rays)
                error = torch.nn.functional.mse_loss(
                    outputs['rgb'], target_rgb, reduction='none'
                ).mean(dim=-1)
                
                # 计算梯度
                error.sum().backward()
                gradients = rays.grad.abs().mean(dim=-1)
                
                # 映射到体素
                voxel_intersections = outputs.get('voxel_intersections', [])
                for i, intersection in enumerate(voxel_intersections):
                    voxel_idx = intersection['voxel_id']
                    
                    if voxel_idx not in error_map:
                        error_map[voxel_idx] = []
                        gradient_map[voxel_idx] = []
                    
                    error_map[voxel_idx].append(error[i].item())
                    gradient_map[voxel_idx].append(gradients[i].item())
        
        # 分析候选体素
        for voxel_idx, errors in error_map.items():
            avg_error = np.mean(errors)
            avg_gradient = np.mean(gradient_map[voxel_idx])
            
            # 细分条件：高误差或高梯度
            if avg_error > self.error_threshold or avg_gradient > self.error_threshold:
                candidates.append(voxel_idx)
        
        # 按误差排序，优先细分误差最大的体素
        candidates.sort(key=lambda idx: np.mean(error_map[idx]), reverse=True)
        
        # 限制每次细分的体素数量
        max_subdivisions = min(len(candidates), 100)
        return candidates[:max_subdivisions]
    
    def _execute_subdivision(self, model: SVRasterModel, voxel_indices: List[int]) -> bool:
        """
        执行体素细分操作
        """
        try:
            # 获取当前体素网格
            voxel_grid = model.get_voxel_grid()
            
            # 为每个候选体素创建子体素
            new_voxels = []
            
            for voxel_idx in voxel_indices:
                parent_voxel = voxel_grid.get_voxel(voxel_idx)
                
                # 创建8个子体素
                child_voxels = self._create_child_voxels(parent_voxel)
                
                # 继承父体素的属性
                for child in child_voxels:
                    self._inherit_parent_properties(child, parent_voxel)
                
                new_voxels.extend(child_voxels)
                
                # 标记父体素为非活跃
                parent_voxel.active = False
            
            # 更新体素网格
            voxel_grid.add_voxels(new_voxels)
            
            # 重新构建空间索引
            voxel_grid.rebuild_spatial_index()
            
            return True
            
        except Exception as e:
            logger.error(f"体素细分执行失败: {e}")
            return False
    
    def _create_child_voxels(self, parent_voxel) -> List:
        """
        为父体素创建8个子体素
        """
        child_voxels = []
        parent_center = parent_voxel.center
        parent_size = parent_voxel.size
        child_size = parent_size / 2
        
        # 8个子体素的相对位置
        offsets = [
            (-1, -1, -1), (+1, -1, -1), (-1, +1, -1), (+1, +1, -1),
            (-1, -1, +1), (+1, -1, +1), (-1, +1, +1), (+1, +1, +1)
        ]
        
        for offset in offsets:
            child_center = parent_center + torch.tensor(offset) * child_size / 2
            
            child_voxel = Voxel(
                center=child_center,
                size=child_size,
                level=parent_voxel.level + 1,
                parent_id=parent_voxel.id
            )
            
            child_voxels.append(child_voxel)
        
        return child_voxels
    
    def _inherit_parent_properties(self, child_voxel, parent_voxel):
        """
        子体素继承父体素的属性
        """
        # 密度继承（添加少量噪声）
        child_voxel.density = parent_voxel.density + torch.randn_like(parent_voxel.density) * 0.01
        
        # 颜色继承
        child_voxel.color = parent_voxel.color.clone()
        
        # 球谐系数继承
        if hasattr(parent_voxel, 'sh_coeffs'):
            child_voxel.sh_coeffs = parent_voxel.sh_coeffs.clone()
        
        # 特征继承
        if hasattr(parent_voxel, 'features'):
            child_voxel.features = parent_voxel.features.clone()
    
    def _get_current_max_level(self, model: SVRasterModel) -> int:
        """
        获取当前最大细分等级
        """
        voxel_grid = model.get_voxel_grid()
        max_level = 0
        
        for voxel in voxel_grid.active_voxels:
            max_level = max(max_level, voxel.level)
        
        return max_level
    
    def _analyze_subdivision_need(self, model: SVRasterModel) -> bool:
        """
        分析是否需要细分
        """
        # 检查模型复杂度
        voxel_grid = model.get_voxel_grid()
        active_voxel_count = len(voxel_grid.active_voxels)
        
        # 如果活跃体素太少，可能需要细分
        if active_voxel_count < 1000:
            return True
        
        # 检查渲染质量指标
        # 这里可以添加更复杂的分析逻辑
        
        return False
    
    def _update_subdivision_stats(self, success: bool, num_voxels: int):
        """
        更新细分统计信息
        """
        self.subdivision_stats['total_subdivisions'] += num_voxels
        
        if success:
            self.subdivision_stats['successful_subdivisions'] += num_voxels
        else:
            self.subdivision_stats['failed_subdivisions'] += num_voxels
```

### 1.2 渐进式细分策略

```python
class ProgressiveSubdivision:
    """
    渐进式细分策略
    逐步增加体素分辨率，避免训练不稳定
    """
    
    def __init__(self, config: SVRasterTrainerConfig):
        self.config = config
        self.subdivision_schedule = self._create_subdivision_schedule()
        self.current_stage = 0
    
    def _create_subdivision_schedule(self) -> List[dict]:
        """
        创建细分调度表
        """
        schedule = []
        
        # 定义渐进式细分阶段
        stages = [
            {'epoch': 0, 'max_level': 4, 'threshold': 0.05},
            {'epoch': 20, 'max_level': 6, 'threshold': 0.03},
            {'epoch': 40, 'max_level': 8, 'threshold': 0.02},
            {'epoch': 60, 'max_level': 10, 'threshold': 0.015},
            {'epoch': 80, 'max_level': 12, 'threshold': 0.01}
        ]
        
        return stages
    
    def get_current_subdivision_params(self, epoch: int) -> dict:
        """
        获取当前epoch的细分参数
        """
        # 找到当前阶段
        current_stage = self.subdivision_schedule[0]
        
        for stage in self.subdivision_schedule:
            if epoch >= stage['epoch']:
                current_stage = stage
            else:
                break
        
        return current_stage
    
    def update_subdivision_config(self, epoch: int, subdivision_manager: AdaptiveSubdivision):
        """
        更新细分配置
        """
        current_params = self.get_current_subdivision_params(epoch)
        
        # 更新细分参数
        subdivision_manager.max_level = current_params['max_level']
        subdivision_manager.error_threshold = current_params['threshold']
        
        logger.info(f"更新细分参数: max_level={current_params['max_level']}, "
                   f"threshold={current_params['threshold']}")
```

## 2. 智能剪枝策略

### 2.1 基于重要性的剪枝

```python
class IntelligentPruning:
    """
    智能剪枝管理器
    基于体素重要性和贡献度进行剪枝
    """
    
    def __init__(self, config: SVRasterTrainerConfig):
        self.config = config
        self.pruning_threshold = config.pruning_threshold
        self.pruning_history = []
        
        # 剪枝统计
        self.pruning_stats = {
            'total_pruned': 0,
            'memory_saved': 0,
            'performance_gain': 0
        }
    
    def should_prune(self, epoch: int, model: SVRasterModel) -> bool:
        """
        判断是否应该进行剪枝
        """
        if not self.config.enable_pruning:
            return False
        
        if epoch < self.config.pruning_start_epoch:
            return False
        
        if (epoch - self.config.pruning_start_epoch) % self.config.pruning_interval != 0:
            return False
        
        return True
    
    def perform_pruning(self, model: SVRasterModel, 
                       training_data: torch.utils.data.DataLoader) -> dict:
        """
        执行智能剪枝
        
        Returns:
            剪枝结果统计
        """
        logger.info("开始智能剪枝")
        
        # 1. 分析体素重要性
        importance_scores = self._analyze_voxel_importance(model, training_data)
        
        # 2. 识别需要剪枝的体素
        voxels_to_prune = self._identify_pruning_candidates(importance_scores)
        
        # 3. 执行剪枝操作
        pruning_results = self._execute_pruning(model, voxels_to_prune)
        
        # 4. 更新统计信息
        self._update_pruning_stats(pruning_results)
        
        logger.info(f"剪枝完成: 移除 {pruning_results['pruned_count']} 个体素")
        
        return pruning_results
    
    def _analyze_voxel_importance(self, model: SVRasterModel, 
                                 training_data: torch.utils.data.DataLoader) -> dict:
        """
        分析体素重要性
        """
        importance_scores = {}
        voxel_contributions = {}
        voxel_access_counts = {}
        
        model.eval()
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(training_data):
                if batch_idx >= 20:  # 采样数据
                    break
                
                rays = batch['rays'].to(model.device)
                outputs = model(rays)
                
                # 分析体素贡献
                voxel_intersections = outputs.get('voxel_intersections', [])
                weights = outputs.get('weights', None)
                
                for i, intersection in enumerate(voxel_intersections):
                    voxel_idx = intersection['voxel_id']
                    
                    # 统计访问次数
                    if voxel_idx not in voxel_access_counts:
                        voxel_access_counts[voxel_idx] = 0
                        voxel_contributions[voxel_idx] = []
                    
                    voxel_access_counts[voxel_idx] += 1
                    
                    # 记录体素贡献（基于权重）
                    if weights is not None and i < len(weights):
                        contribution = weights[i].sum().item()
                        voxel_contributions[voxel_idx].append(contribution)
        
        # 计算重要性分数
        for voxel_idx in voxel_access_counts:
            access_count = voxel_access_counts[voxel_idx]
            avg_contribution = np.mean(voxel_contributions[voxel_idx]) if voxel_contributions[voxel_idx] else 0
            
            # 重要性 = 访问频率 × 平均贡献
            importance = access_count * avg_contribution
            importance_scores[voxel_idx] = importance
        
        return importance_scores
    
    def _identify_pruning_candidates(self, importance_scores: dict) -> List[int]:
        """
        识别剪枝候选体素
        """
        # 按重要性排序
        sorted_voxels = sorted(
            importance_scores.items(), 
            key=lambda x: x[1]
        )
        
        # 选择重要性最低的体素进行剪枝
        candidates = []
        for voxel_idx, importance in sorted_voxels:
            if importance < self.pruning_threshold:
                candidates.append(voxel_idx)
        
        # 限制剪枝数量（避免过度剪枝）
        max_prune_count = min(len(candidates), len(importance_scores) // 10)
        
        return candidates[:max_prune_count]
    
    def _execute_pruning(self, model: SVRasterModel, voxel_indices: List[int]) -> dict:
        """
        执行剪枝操作
        """
        voxel_grid = model.get_voxel_grid()
        
        # 记录剪枝前的状态
        before_count = len(voxel_grid.active_voxels)
        before_memory = self._estimate_memory_usage(voxel_grid)
        
        # 执行剪枝
        pruned_count = 0
        for voxel_idx in voxel_indices:
            if voxel_grid.remove_voxel(voxel_idx):
                pruned_count += 1
        
        # 重建空间索引
        voxel_grid.rebuild_spatial_index()
        
        # 记录剪枝后的状态
        after_count = len(voxel_grid.active_voxels)
        after_memory = self._estimate_memory_usage(voxel_grid)
        
        return {
            'pruned_count': pruned_count,
            'before_count': before_count,
            'after_count': after_count,
            'memory_saved': before_memory - after_memory
        }
    
    def _estimate_memory_usage(self, voxel_grid) -> float:
        """
        估算内存使用量（MB）
        """
        # 简化的内存估算
        active_voxels = len(voxel_grid.active_voxels)
        memory_per_voxel = 0.1  # MB per voxel (rough estimate)
        
        return active_voxels * memory_per_voxel
    
    def _update_pruning_stats(self, results: dict):
        """
        更新剪枝统计信息
        """
        self.pruning_stats['total_pruned'] += results['pruned_count']
        self.pruning_stats['memory_saved'] += results['memory_saved']
```

### 2.2 基于梯度的剪枝

```python
class GradientBasedPruning:
    """
    基于梯度的剪枝策略
    利用梯度信息判断体素的重要性
    """
    
    def __init__(self, config: SVRasterTrainerConfig):
        self.config = config
        self.gradient_threshold = 1e-6
        
    def analyze_gradient_importance(self, model: SVRasterModel) -> dict:
        """
        分析基于梯度的体素重要性
        """
        gradient_scores = {}
        
        # 获取体素参数的梯度
        for name, param in model.named_parameters():
            if 'voxel' in name and param.grad is not None:
                # 解析体素索引（假设参数名包含体素ID）
                voxel_id = self._extract_voxel_id(name)
                
                # 计算梯度幅度
                grad_magnitude = torch.norm(param.grad).item()
                gradient_scores[voxel_id] = grad_magnitude
        
        return gradient_scores
    
    def _extract_voxel_id(self, param_name: str) -> int:
        """
        从参数名中提取体素ID
        """
        # 简化的ID提取逻辑
        import re
        match = re.search(r'voxel_(\d+)', param_name)
        if match:
            return int(match.group(1))
        return 0
    
    def prune_by_gradient(self, model: SVRasterModel) -> List[int]:
        """
        基于梯度进行剪枝
        """
        gradient_scores = self.analyze_gradient_importance(model)
        
        # 选择梯度极小的体素进行剪枝
        candidates = []
        for voxel_id, grad_score in gradient_scores.items():
            if grad_score < self.gradient_threshold:
                candidates.append(voxel_id)
        
        return candidates
```

## 3. 渐进式训练策略

### 3.1 多分辨率训练

```python
class MultiResolutionTraining:
    """
    多分辨率渐进式训练
    从低分辨率开始，逐步提高到目标分辨率
    """
    
    def __init__(self, config: SVRasterTrainerConfig):
        self.config = config
        self.resolution_schedule = self._create_resolution_schedule()
        self.current_resolution = config.min_resolution
        
    def _create_resolution_schedule(self) -> List[dict]:
        """
        创建分辨率调度表
        """
        schedule = []
        
        # 从最小分辨率开始
        current_res = self.config.min_resolution
        epoch = 0
        epochs_per_stage = 20
        
        while current_res <= self.config.max_resolution:
            schedule.append({
                'epoch': epoch,
                'resolution': current_res,
                'batch_size_mult': max(1.0, self.config.max_resolution / current_res / 2),
                'learning_rate_mult': 1.0 if current_res >= 256 else 1.5
            })
            
            current_res *= 2
            epoch += epochs_per_stage
            epochs_per_stage = max(10, epochs_per_stage - 2)  # 逐渐减少每阶段的轮数
        
        return schedule
    
    def get_current_training_config(self, epoch: int) -> dict:
        """
        获取当前训练配置
        """
        current_stage = self.resolution_schedule[0]
        
        for stage in self.resolution_schedule:
            if epoch >= stage['epoch']:
                current_stage = stage
            else:
                break
        
        return current_stage
    
    def update_training_resolution(self, epoch: int, model: SVRasterModel, 
                                 dataset: SVRasterDataset) -> bool:
        """
        更新训练分辨率
        
        Returns:
            是否发生了分辨率变化
        """
        stage_config = self.get_current_training_config(epoch)
        new_resolution = stage_config['resolution']
        
        if new_resolution != self.current_resolution:
            logger.info(f"更新训练分辨率: {self.current_resolution} -> {new_resolution}")
            
            # 更新数据集分辨率
            dataset.update_resolution(new_resolution)
            
            # 更新模型分辨率相关参数
            self._update_model_resolution(model, new_resolution)
            
            self.current_resolution = new_resolution
            return True
        
        return False
    
    def _update_model_resolution(self, model: SVRasterModel, resolution: int):
        """
        更新模型的分辨率相关参数
        """
        # 更新体素网格分辨率
        voxel_grid = model.get_voxel_grid()
        voxel_grid.update_resolution(resolution)
        
        # 重新初始化相关组件
        model.reinitialize_for_resolution(resolution)
```

### 3.2 课程学习策略

```python
class CurriculumLearning:
    """
    课程学习策略
    从简单样本开始，逐步增加训练难度
    """
    
    def __init__(self, config: SVRasterTrainerConfig):
        self.config = config
        self.difficulty_schedule = self._create_difficulty_schedule()
        self.current_difficulty = 0.0
        
    def _create_difficulty_schedule(self) -> List[dict]:
        """
        创建难度调度表
        """
        schedule = [
            {'epoch': 0, 'difficulty': 0.0, 'description': '仅简单视角'},
            {'epoch': 10, 'difficulty': 0.2, 'description': '添加中等视角'},
            {'epoch': 30, 'difficulty': 0.5, 'description': '添加复杂视角'},
            {'epoch': 50, 'difficulty': 0.8, 'description': '包含困难样本'},
            {'epoch': 70, 'difficulty': 1.0, 'description': '全部训练数据'}
        ]
        
        return schedule
    
    def get_training_subset(self, epoch: int, dataset: SVRasterDataset) -> torch.utils.data.Subset:
        """
        根据课程学习策略获取训练子集
        """
        # 获取当前难度等级
        difficulty = self._get_current_difficulty(epoch)
        
        # 根据难度过滤训练样本
        filtered_indices = self._filter_samples_by_difficulty(dataset, difficulty)
        
        return torch.utils.data.Subset(dataset, filtered_indices)
    
    def _get_current_difficulty(self, epoch: int) -> float:
        """
        获取当前难度等级
        """
        current_stage = self.difficulty_schedule[0]
        
        for stage in self.difficulty_schedule:
            if epoch >= stage['epoch']:
                current_stage = stage
            else:
                break
        
        return current_stage['difficulty']
    
    def _filter_samples_by_difficulty(self, dataset: SVRasterDataset, difficulty: float) -> List[int]:
        """
        根据难度过滤样本
        """
        # 计算每个样本的难度分数
        difficulty_scores = self._compute_sample_difficulties(dataset)
        
        # 根据难度阈值过滤
        threshold = np.percentile(difficulty_scores, difficulty * 100)
        
        filtered_indices = []
        for i, score in enumerate(difficulty_scores):
            if score <= threshold:
                filtered_indices.append(i)
        
        return filtered_indices
    
    def _compute_sample_difficulties(self, dataset: SVRasterDataset) -> np.ndarray:
        """
        计算样本难度分数
        """
        difficulties = []
        
        for i in range(len(dataset)):
            sample = dataset[i]
            
            # 基于多个因素计算难度
            camera_pose = sample.get('camera_pose', None)
            
            # 视角复杂度
            view_complexity = self._compute_view_complexity(camera_pose) if camera_pose is not None else 0.5
            
            # 场景复杂度（基于图像内容）
            scene_complexity = self._compute_scene_complexity(sample['rgb'])
            
            # 综合难度分数
            difficulty = 0.6 * view_complexity + 0.4 * scene_complexity
            difficulties.append(difficulty)
        
        return np.array(difficulties)
    
    def _compute_view_complexity(self, camera_pose: torch.Tensor) -> float:
        """
        计算视角复杂度
        """
        # 简化的视角复杂度计算
        # 基于相机位置和朝向的变化程度
        
        # 计算相机位置的复杂度（距离场景中心的距离）
        position = camera_pose[:3, 3]
        distance_from_center = torch.norm(position).item()
        
        # 计算视角的复杂度（基于旋转矩阵）
        rotation_matrix = camera_pose[:3, :3]
        rotation_complexity = torch.norm(rotation_matrix - torch.eye(3)).item()
        
        # 综合复杂度
        complexity = 0.5 * min(distance_from_center / 5.0, 1.0) + 0.5 * min(rotation_complexity / 2.0, 1.0)
        
        return complexity
    
    def _compute_scene_complexity(self, image: torch.Tensor) -> float:
        """
        计算场景复杂度
        """
        # 基于图像内容的复杂度计算
        
        # 计算图像梯度（边缘复杂度）
        grad_x = torch.diff(image, dim=1)
        grad_y = torch.diff(image, dim=0)
        gradient_magnitude = torch.sqrt(grad_x[:, :-1] ** 2 + grad_y[:-1, :] ** 2)
        edge_complexity = torch.mean(gradient_magnitude).item()
        
        # 计算颜色复杂度
        color_var = torch.var(image).item()
        
        # 综合复杂度
        complexity = 0.7 * min(edge_complexity * 10, 1.0) + 0.3 * min(color_var * 5, 1.0)
        
        return complexity
```

## 4. 动态负载均衡

### 4.1 自适应批处理

```python
class DynamicBatchSizeManager:
    """
    动态批处理大小管理器
    根据GPU负载和内存使用情况自适应调整批处理大小
    """
    
    def __init__(self, config: SVRasterTrainerConfig):
        self.config = config
        self.initial_batch_size = config.batch_size
        self.current_batch_size = config.batch_size
        self.min_batch_size = max(1, config.batch_size // 4)
        self.max_batch_size = config.batch_size * 2
        
        # 性能监控
        self.performance_history = []
        self.memory_history = []
        
    def update_batch_size(self, gpu_utilization: float, memory_usage: float, 
                         training_speed: float) -> int:
        """
        根据系统状态更新批处理大小
        
        Args:
            gpu_utilization: GPU利用率 (0-1)
            memory_usage: 内存使用率 (0-1)
            training_speed: 训练速度 (samples/sec)
            
        Returns:
            新的批处理大小
        """
        # 记录性能历史
        self.performance_history.append(training_speed)
        self.memory_history.append(memory_usage)
        
        # 保持历史记录在合理范围内
        if len(self.performance_history) > 10:
            self.performance_history.pop(0)
            self.memory_history.pop(0)
        
        # 决策逻辑
        new_batch_size = self.current_batch_size
        
        # 内存不足，减少批处理大小
        if memory_usage > 0.9:
            new_batch_size = max(self.min_batch_size, int(self.current_batch_size * 0.8))
            logger.info(f"内存不足，减少批处理大小: {self.current_batch_size} -> {new_batch_size}")
        
        # GPU利用率低且内存充足，增加批处理大小
        elif gpu_utilization < 0.7 and memory_usage < 0.7:
            new_batch_size = min(self.max_batch_size, int(self.current_batch_size * 1.2))
            logger.info(f"GPU利用率低，增加批处理大小: {self.current_batch_size} -> {new_batch_size}")
        
        # 训练速度下降，可能需要调整
        elif len(self.performance_history) >= 3:
            recent_speed = np.mean(self.performance_history[-3:])
            older_speed = np.mean(self.performance_history[:-3]) if len(self.performance_history) > 3 else recent_speed
            
            if recent_speed < older_speed * 0.9:  # 速度下降超过10%
                new_batch_size = max(self.min_batch_size, int(self.current_batch_size * 0.9))
                logger.info(f"训练速度下降，减少批处理大小: {self.current_batch_size} -> {new_batch_size}")
        
        self.current_batch_size = new_batch_size
        return new_batch_size
    
    def get_optimal_chunk_size(self, batch_size: int) -> int:
        """
        获取最优的块大小用于分块处理
        """
        # 基于批处理大小和GPU内存计算最优块大小
        gpu_memory_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
        
        # 简化的块大小计算
        if gpu_memory_gb >= 24:  # 高端GPU
            chunk_size = min(batch_size, 8192)
        elif gpu_memory_gb >= 12:  # 中端GPU
            chunk_size = min(batch_size, 4096)
        else:  # 低端GPU
            chunk_size = min(batch_size, 2048)
        
        return chunk_size
```

### 4.2 训练进度自适应调整

```python
class AdaptiveTrainingScheduler:
    """
    自适应训练调度器
    根据训练进度动态调整训练策略
    """
    
    def __init__(self, config: SVRasterTrainerConfig):
        self.config = config
        self.loss_history = []
        self.psnr_history = []
        self.plateau_patience = 10
        self.plateau_threshold = 0.001
        
    def should_adjust_strategy(self, epoch: int, current_loss: float, current_psnr: float) -> dict:
        """
        判断是否需要调整训练策略
        """
        self.loss_history.append(current_loss)
        self.psnr_history.append(current_psnr)
        
        adjustments = {}
        
        # 检测收敛平台
        if self._detect_plateau():
            adjustments['learning_rate'] = self._suggest_lr_adjustment()
            adjustments['subdivision'] = self._suggest_subdivision_adjustment()
            adjustments['training_strategy'] = self._suggest_strategy_adjustment()
        
        # 检测过拟合
        if self._detect_overfitting():
            adjustments['regularization'] = self._suggest_regularization_adjustment()
        
        # 检测欠拟合
        if self._detect_underfitting():
            adjustments['model_capacity'] = self._suggest_capacity_adjustment()
        
        return adjustments
    
    def _detect_plateau(self) -> bool:
        """
        检测训练平台
        """
        if len(self.loss_history) < self.plateau_patience:
            return False
        
        recent_losses = self.loss_history[-self.plateau_patience:]
        loss_improvement = recent_losses[0] - recent_losses[-1]
        
        return loss_improvement < self.plateau_threshold
    
    def _detect_overfitting(self) -> bool:
        """
        检测过拟合
        """
        # 简化的过拟合检测
        # 实际实现需要验证集损失
        if len(self.loss_history) < 20:
            return False
        
        # 如果训练损失持续下降但PSNR停止改善
        recent_loss_trend = np.polyfit(range(10), self.loss_history[-10:], 1)[0]
        recent_psnr_trend = np.polyfit(range(10), self.psnr_history[-10:], 1)[0]
        
        return recent_loss_trend < -0.001 and abs(recent_psnr_trend) < 0.1
    
    def _detect_underfitting(self) -> bool:
        """
        检测欠拟合
        """
        # 如果损失下降很慢且PSNR较低
        if len(self.psnr_history) < 10:
            return False
        
        recent_psnr = np.mean(self.psnr_history[-5:])
        return recent_psnr < 25.0  # PSNR阈值
    
    def _suggest_lr_adjustment(self) -> dict:
        """
        建议学习率调整
        """
        return {
            'action': 'reduce',
            'factor': 0.5,
            'reason': '检测到训练平台'
        }
    
    def _suggest_subdivision_adjustment(self) -> dict:
        """
        建议细分调整
        """
        return {
            'action': 'trigger_early',
            'reason': '训练进入平台，可能需要更高分辨率'
        }
    
    def _suggest_strategy_adjustment(self) -> dict:
        """
        建议策略调整
        """
        return {
            'action': 'increase_regularization',
            'reason': '可能需要更强的正则化'
        }
    
    def _suggest_regularization_adjustment(self) -> dict:
        """
        建议正则化调整
        """
        return {
            'action': 'increase_weight',
            'factor': 1.5,
            'reason': '检测到过拟合'
        }
    
    def _suggest_capacity_adjustment(self) -> dict:
        """
        建议容量调整
        """
        return {
            'action': 'increase_subdivision',
            'reason': '模型容量可能不足'
        }
```

## 总结

SVRaster 的高级训练技术包含以下关键组件：

1. **自适应体素细分**：基于渲染误差和梯度信息动态调整体素分辨率
2. **智能剪枝策略**：通过重要性分析和梯度信息移除不必要的体素
3. **渐进式训练**：多分辨率训练和课程学习提高训练效率
4. **动态负载均衡**：自适应批处理和训练策略调整

这些技术的综合应用确保了 SVRaster 训练过程的高效性和稳定性，能够在保证渲染质量的同时显著提高训练效率。

**下一部分预告**：第三部分将详细介绍损失函数设计、正则化技术、性能监控与调试等内容。
