# Inf-NeRF 渲染机制详解 - 第二部分：多尺度网络与自适应采样

**版本**: 1.0  
**日期**: 2025年7月5日  
**依赖**: 第一部分 - 渲染基础

## 概述

本部分深入探讨 Inf-NeRF 的核心技术组件：多尺度神经网络架构、自适应采样策略和抗锯齿渲染技术。这些技术确保了 Inf-NeRF 能够在不同尺度下提供高质量的渲染结果，同时保持计算效率。

## 目录

1. [多尺度神经网络](#多尺度神经网络)
2. [哈希编码优化](#哈希编码优化)
3. [自适应采样策略](#自适应采样策略)
4. [抗锯齿渲染技术](#抗锯齿渲染技术)
5. [体积渲染优化](#体积渲染优化)
6. [级别间一致性](#级别间一致性)

---

## 多尺度神经网络

### 1. LoD感知神经网络

```python
class LoDAwareNeRF(nn.Module):
    """
    细节级别感知的神经辐射场
    每个八叉树节点都有自己的LoDAwareNeRF，复杂度根据级别调整
    """
    
    def __init__(self, config: InfNeRFConfig, level: int):
        super().__init__()
        self.config = config
        self.level = level
        
        # 根据级别调整网络复杂度
        # 更高级别（更精细）使用更复杂的网络
        complexity_factor = min(1.0, (level + 1) / config.max_depth)
        
        self.hidden_dim = max(32, int(config.hidden_dim * complexity_factor))
        self.num_layers = max(1, int(config.num_layers * complexity_factor))
        
        # 哈希编码用于空间特征
        self.spatial_encoder = HashEncoder(
            config, 
            complexity_factor=complexity_factor
        )
        
        # 方向编码（跨级别共享）
        self.direction_encoder = SHEncoder(degree=4)
        
        # 密度网络
        self.density_network = self._build_density_network()
        
        # 颜色网络
        self.color_network = self._build_color_network()
        
        # 级别特定的参数
        self.level_embedding = nn.Parameter(torch.randn(1, 16))
        
    def _build_density_network(self):
        """
        构建密度预测网络
        """
        layers = []
        input_dim = self.spatial_encoder.output_dim + 16  # +16 for level embedding
        
        # 输入层
        layers.append(nn.Linear(input_dim, self.hidden_dim))
        layers.append(nn.ReLU())
        
        # 隐藏层
        for _ in range(self.num_layers - 1):
            layers.append(nn.Linear(self.hidden_dim, self.hidden_dim))
            layers.append(nn.ReLU())
        
        # 输出层：密度 + 几何特征
        layers.append(nn.Linear(self.hidden_dim, 1 + self.config.geo_feat_dim))
        
        return nn.Sequential(*layers)
    
    def _build_color_network(self):
        """
        构建颜色预测网络
        """
        layers = []
        input_dim = (self.config.geo_feat_dim + 
                    self.direction_encoder.output_dim + 16)  # +16 for level embedding
        
        # 输入层
        layers.append(nn.Linear(input_dim, self.hidden_dim))
        layers.append(nn.ReLU())
        
        # 隐藏层
        for _ in range(self.config.num_layers_color - 1):
            layers.append(nn.Linear(self.hidden_dim, self.hidden_dim))
            layers.append(nn.ReLU())
        
        # 输出层：RGB颜色
        layers.append(nn.Linear(self.hidden_dim, 3))
        layers.append(nn.Sigmoid())
        
        return nn.Sequential(*layers)
    
    def forward(self, positions, directions, level_weights=None):
        """
        前向传播
        
        Args:
            positions: 3D位置 [N, 3]
            directions: 视线方向 [N, 3]
            level_weights: 级别权重（用于多级别混合）[N]
            
        Returns:
            density: 密度值 [N]
            color: RGB颜色 [N, 3]
            features: 中间特征（用于调试）
        """
        batch_size = positions.shape[0]
        
        # 空间编码
        spatial_features = self.spatial_encoder(positions)
        
        # 级别嵌入
        level_embed = self.level_embedding.expand(batch_size, -1)
        
        # 密度预测
        density_input = torch.cat([spatial_features, level_embed], dim=-1)
        density_output = self.density_network(density_input)
        
        # 分离密度和几何特征
        density = F.relu(density_output[..., 0])
        geo_features = density_output[..., 1:]
        
        # 方向编码
        direction_features = self.direction_encoder(directions)
        
        # 颜色预测
        color_input = torch.cat([
            geo_features, direction_features, level_embed
        ], dim=-1)
        color = self.color_network(color_input)
        
        # 应用级别权重（如果提供）
        if level_weights is not None:
            density = density * level_weights
            color = color * level_weights.unsqueeze(-1)
        
        return {
            'density': density,
            'color': color,
            'geo_features': geo_features,
            'spatial_features': spatial_features
        }
    
    def get_density(self, positions):
        """
        仅获取密度值（用于快速查询）
        """
        with torch.no_grad():
            spatial_features = self.spatial_encoder(positions)
            level_embed = self.level_embedding.expand(positions.shape[0], -1)
            density_input = torch.cat([spatial_features, level_embed], dim=-1)
            density_output = self.density_network(density_input)
            density = F.relu(density_output[..., 0])
        return density
```

### 2. 自适应网络复杂度

```python
class AdaptiveComplexityController:
    """
    自适应网络复杂度控制器
    根据场景复杂度和渲染需求动态调整网络复杂度
    """
    
    def __init__(self, config: InfNeRFConfig):
        self.config = config
        self.complexity_history = []
        self.performance_monitor = PerformanceMonitor()
        
    def compute_complexity_factor(self, level, scene_statistics):
        """
        计算复杂度因子
        
        Args:
            level: 八叉树级别
            scene_statistics: 场景统计信息
            
        Returns:
            complexity_factor: 复杂度因子 [0, 1]
        """
        # 基础复杂度（基于级别）
        base_complexity = (level + 1) / self.config.max_depth
        
        # 场景复杂度调整
        scene_complexity = scene_statistics.get('complexity', 0.5)
        detail_density = scene_statistics.get('detail_density', 0.5)
        
        # 性能考虑
        current_fps = self.performance_monitor.get_current_fps()
        target_fps = self.config.target_fps if hasattr(self.config, 'target_fps') else 30
        performance_factor = min(1.0, current_fps / target_fps)
        
        # 组合因子
        complexity_factor = (
            base_complexity * 0.4 +
            scene_complexity * 0.3 + 
            detail_density * 0.2 +
            performance_factor * 0.1
        )
        
        # 限制范围
        complexity_factor = np.clip(complexity_factor, 0.1, 1.0)
        
        return complexity_factor
    
    def adaptive_network_pruning(self, network, importance_threshold=0.01):
        """
        自适应网络剪枝
        
        Args:
            network: 要剪枝的网络
            importance_threshold: 重要性阈值
            
        Returns:
            pruned_network: 剪枝后的网络
        """
        # 计算参数重要性
        param_importance = self._compute_parameter_importance(network)
        
        # 基于重要性剪枝
        pruned_params = {}
        for name, param in network.named_parameters():
            importance = param_importance.get(name, 1.0)
            
            if importance > importance_threshold:
                pruned_params[name] = param
            else:
                # 将不重要的参数置零
                param.data.zero_()
        
        return network
    
    def _compute_parameter_importance(self, network):
        """
        计算网络参数重要性
        """
        importance = {}
        
        for name, param in network.named_parameters():
            if param.grad is not None:
                # 基于梯度的重要性
                grad_importance = torch.norm(param.grad).item()
                
                # 基于参数值的重要性
                param_importance = torch.norm(param.data).item()
                
                # 组合重要性
                combined_importance = grad_importance * param_importance
                importance[name] = combined_importance
            else:
                importance[name] = 1.0
        
        # 归一化重要性
        max_importance = max(importance.values()) if importance else 1.0
        for name in importance:
            importance[name] /= max_importance
        
        return importance
```

---

## 哈希编码优化

### 1. 多分辨率哈希编码

```python
class HashEncoder(nn.Module):
    """
    多分辨率哈希编码器
    为不同级别的LoD提供高效的空间特征编码
    """
    
    def __init__(self, config: InfNeRFConfig, complexity_factor=1.0):
        super().__init__()
        self.config = config
        self.complexity_factor = complexity_factor
        
        # 调整参数基于复杂度因子
        self.num_levels = max(4, int(config.num_levels * complexity_factor))
        self.level_dim = config.level_dim
        self.per_level_scale = config.per_level_scale
        
        # 计算哈希表大小
        self.log2_hashmap_size = max(10, int(config.log2_hashmap_size * complexity_factor))
        self.hashmap_size = 2 ** self.log2_hashmap_size
        
        # 计算每个级别的分辨率
        self.resolutions = self._compute_resolutions()
        
        # 初始化哈希表
        self.hash_tables = nn.ModuleList([
            nn.Embedding(self.hashmap_size, self.level_dim)
            for _ in range(self.num_levels)
        ])
        
        # 初始化哈希表权重
        self._initialize_hash_tables()
        
        self.output_dim = self.num_levels * self.level_dim
    
    def _compute_resolutions(self):
        """
        计算每个级别的分辨率
        """
        resolutions = []
        base_resolution = self.config.base_resolution
        
        for level in range(self.num_levels):
            resolution = int(base_resolution * (self.per_level_scale ** level))
            resolutions.append(resolution)
        
        return resolutions
    
    def _initialize_hash_tables(self):
        """
        初始化哈希表
        """
        for i, hash_table in enumerate(self.hash_tables):
            # 使用Xavier初始化
            nn.init.xavier_uniform_(hash_table.weight, gain=1.0)
            
            # 根据级别调整初始化范围
            level_scale = 1.0 / (i + 1)
            hash_table.weight.data *= level_scale
    
    def forward(self, positions):
        """
        哈希编码前向传播
        
        Args:
            positions: 3D位置 [N, 3]，范围 [-1, 1]
            
        Returns:
            features: 编码特征 [N, num_levels * level_dim]
        """
        batch_size = positions.shape[0]
        device = positions.device
        
        # 将位置映射到 [0, 1]
        normalized_pos = (positions + 1.0) / 2.0
        
        level_features = []
        
        for level in range(self.num_levels):
            resolution = self.resolutions[level]
            
            # 计算网格坐标
            grid_coords = normalized_pos * (resolution - 1)
            
            # 获取8个角点的坐标
            corner_coords = self._get_corner_coordinates(grid_coords, resolution)
            
            # 计算哈希索引
            hash_indices = self._compute_hash_indices(corner_coords, resolution)
            
            # 获取特征
            corner_features = self.hash_tables[level](hash_indices)  # [N, 8, level_dim]
            
            # 三线性插值
            interpolated_features = self._trilinear_interpolation(
                corner_features, grid_coords
            )
            
            level_features.append(interpolated_features)
        
        # 连接所有级别的特征
        features = torch.cat(level_features, dim=-1)
        
        return features
    
    def _get_corner_coordinates(self, grid_coords, resolution):
        """
        获取8个角点坐标
        
        Args:
            grid_coords: 网格坐标 [N, 3]
            resolution: 当前级别分辨率
            
        Returns:
            corner_coords: 角点坐标 [N, 8, 3]
        """
        # 获取下界坐标
        lower_coords = torch.floor(grid_coords).long()
        
        # 8个角点的偏移
        offsets = torch.tensor([
            [0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0],
            [0, 0, 1], [1, 0, 1], [0, 1, 1], [1, 1, 1]
        ], device=grid_coords.device, dtype=torch.long)
        
        # 计算所有角点坐标
        corner_coords = lower_coords.unsqueeze(1) + offsets.unsqueeze(0)
        
        # 确保坐标在有效范围内
        corner_coords = torch.clamp(corner_coords, 0, resolution - 1)
        
        return corner_coords
    
    def _compute_hash_indices(self, corner_coords, resolution):
        """
        计算哈希索引
        
        Args:
            corner_coords: 角点坐标 [N, 8, 3]
            resolution: 分辨率
            
        Returns:
            hash_indices: 哈希索引 [N, 8]
        """
        # 空间哈希函数
        # 使用大质数避免哈希冲突
        primes = torch.tensor([1, 2654435761, 805459861], 
                             device=corner_coords.device, dtype=torch.long)
        
        # 计算哈希值
        hash_values = torch.sum(corner_coords * primes.unsqueeze(0).unsqueeze(0), dim=-1)
        
        # 模运算得到哈希索引
        hash_indices = hash_values % self.hashmap_size
        
        return hash_indices
    
    def _trilinear_interpolation(self, corner_features, grid_coords):
        """
        三线性插值
        
        Args:
            corner_features: 角点特征 [N, 8, level_dim]
            grid_coords: 网格坐标 [N, 3]
            
        Returns:
            interpolated: 插值结果 [N, level_dim]
        """
        # 计算插值权重
        lower_coords = torch.floor(grid_coords)
        weights = grid_coords - lower_coords  # [N, 3]
        
        # 计算8个角点的插值权重
        w000 = (1 - weights[:, 0]) * (1 - weights[:, 1]) * (1 - weights[:, 2])
        w001 = (1 - weights[:, 0]) * (1 - weights[:, 1]) * weights[:, 2]
        w010 = (1 - weights[:, 0]) * weights[:, 1] * (1 - weights[:, 2])
        w011 = (1 - weights[:, 0]) * weights[:, 1] * weights[:, 2]
        w100 = weights[:, 0] * (1 - weights[:, 1]) * (1 - weights[:, 2])
        w101 = weights[:, 0] * (1 - weights[:, 1]) * weights[:, 2]
        w110 = weights[:, 0] * weights[:, 1] * (1 - weights[:, 2])
        w111 = weights[:, 0] * weights[:, 1] * weights[:, 2]
        
        # 堆叠权重
        interpolation_weights = torch.stack([
            w000, w100, w010, w110, w001, w101, w011, w111
        ], dim=1)  # [N, 8]
        
        # 加权求和
        interpolated = torch.sum(
            interpolation_weights.unsqueeze(-1) * corner_features, 
            dim=1
        )
        
        return interpolated
```

### 2. 自适应哈希分辨率

```python
class AdaptiveHashResolution:
    """
    自适应哈希分辨率管理器
    根据场景复杂度和级别动态调整哈希分辨率
    """
    
    def __init__(self, config: InfNeRFConfig):
        self.config = config
        self.resolution_cache = {}
        self.complexity_analyzer = SceneComplexityAnalyzer()
        
    def compute_adaptive_resolution(self, level, local_complexity):
        """
        计算自适应分辨率
        
        Args:
            level: 八叉树级别
            local_complexity: 局部场景复杂度
            
        Returns:
            resolution: 自适应分辨率
        """
        # 基础分辨率
        base_resolution = self.config.base_resolution * (2 ** level)
        
        # 复杂度调整
        complexity_factor = local_complexity ** 0.5  # 平方根衰减
        
        # 内存限制调整
        memory_factor = self._compute_memory_factor(level)
        
        # 计算最终分辨率
        adaptive_resolution = int(
            base_resolution * complexity_factor * memory_factor
        )
        
        # 限制范围
        min_resolution = max(16, base_resolution // 4)
        max_resolution = min(2048, base_resolution * 4)
        
        adaptive_resolution = np.clip(
            adaptive_resolution, min_resolution, max_resolution
        )
        
        return adaptive_resolution
    
    def _compute_memory_factor(self, level):
        """
        计算内存限制因子
        """
        # 估计当前内存使用
        current_memory = self._estimate_memory_usage()
        max_memory = self.config.max_memory_gb * 1e9  # 转换为字节
        
        # 内存使用率
        memory_ratio = current_memory / max_memory
        
        # 内存压力调整
        if memory_ratio > 0.8:
            memory_factor = 0.5  # 大幅降低分辨率
        elif memory_ratio > 0.6:
            memory_factor = 0.7
        elif memory_ratio > 0.4:
            memory_factor = 0.9
        else:
            memory_factor = 1.0
        
        return memory_factor
    
    def _estimate_memory_usage(self):
        """
        估计当前内存使用量
        """
        try:
            if torch.cuda.is_available():
                return torch.cuda.memory_allocated()
            else:
                # CPU内存估计（简化）
                import psutil
                return psutil.Process().memory_info().rss
        except:
            return 0
```

---

## 自适应采样策略

### 1. 层次化采样

```python
class HierarchicalSampler:
    """
    层次化采样器
    根据八叉树结构进行高效的分层采样
    """
    
    def __init__(self, config: InfNeRFConfig):
        self.config = config
        self.importance_cache = {}
        
    def sample_hierarchical(self, rays_o, rays_d, octree_nodes, lod_levels):
        """
        层次化采样
        
        Args:
            rays_o: 射线起点 [N, 3]
            rays_d: 射线方向 [N, 3]
            octree_nodes: 相关八叉树节点列表
            lod_levels: 每条射线的LoD级别 [N]
            
        Returns:
            sample_points: 采样点 [N, S, 3]
            sample_distances: 采样距离 [N, S]
            sample_weights: 采样权重 [N, S]
        """
        num_rays = rays_o.shape[0]
        device = rays_o.device
        
        # 初始化采样结果
        all_sample_points = []
        all_sample_distances = []
        all_sample_weights = []
        
        for ray_idx in range(num_rays):
            ray_o = rays_o[ray_idx]
            ray_d = rays_d[ray_idx]
            lod_level = lod_levels[ray_idx].item()
            
            # 找到与该射线相交的节点
            intersecting_nodes = self._find_intersecting_nodes(
                ray_o, ray_d, octree_nodes, lod_level
            )
            
            # 为每个节点分配采样点
            ray_samples = self._allocate_samples_per_node(
                ray_o, ray_d, intersecting_nodes, lod_level
            )
            
            all_sample_points.append(ray_samples['points'])
            all_sample_distances.append(ray_samples['distances'])
            all_sample_weights.append(ray_samples['weights'])
        
        # 转换为张量
        max_samples = max(len(points) for points in all_sample_points)
        
        # 填充到相同长度
        padded_points = torch.zeros(num_rays, max_samples, 3, device=device)
        padded_distances = torch.zeros(num_rays, max_samples, device=device)
        padded_weights = torch.zeros(num_rays, max_samples, device=device)
        
        for i in range(num_rays):
            num_samples = len(all_sample_points[i])
            padded_points[i, :num_samples] = all_sample_points[i]
            padded_distances[i, :num_samples] = all_sample_distances[i]
            padded_weights[i, :num_samples] = all_sample_weights[i]
        
        return padded_points, padded_distances, padded_weights
    
    def _find_intersecting_nodes(self, ray_o, ray_d, octree_nodes, target_level):
        """
        找到与射线相交的节点
        """
        intersecting_nodes = []
        
        for node in octree_nodes:
            # 只考虑目标级别及其父节点
            if node.level <= target_level:
                if self._ray_aabb_intersection(ray_o, ray_d, node.get_aabb()):
                    intersecting_nodes.append(node)
        
        return intersecting_nodes
    
    def _ray_aabb_intersection(self, ray_o, ray_d, aabb):
        """
        射线与AABB相交测试
        """
        aabb_min, aabb_max = aabb
        
        # 计算进入和退出点
        inv_dir = 1.0 / (ray_d + 1e-8)
        t_min = (aabb_min - ray_o) * inv_dir
        t_max = (aabb_max - ray_o) * inv_dir
        
        # 确保t_min < t_max
        t_min, t_max = torch.min(t_min, t_max), torch.max(t_min, t_max)
        
        # 计算交点
        t_near = torch.max(t_min)
        t_far = torch.min(t_max)
        
        return t_near <= t_far and t_far > 0
    
    def _allocate_samples_per_node(self, ray_o, ray_d, nodes, lod_level):
        """
        为每个节点分配采样点
        """
        total_samples = self.config.num_samples
        sample_points = []
        sample_distances = []
        sample_weights = []
        
        if not nodes:
            # 如果没有相交节点，使用均匀采样
            t_vals = torch.linspace(0.1, 10.0, total_samples, device=ray_o.device)
            points = ray_o.unsqueeze(0) + t_vals.unsqueeze(-1) * ray_d.unsqueeze(0)
            weights = torch.ones_like(t_vals) / len(t_vals)
            
            return {
                'points': points,
                'distances': t_vals, 
                'weights': weights
            }
        
        # 根据节点重要性分配采样点
        node_importance = [self._compute_node_importance(node, lod_level) 
                          for node in nodes]
        total_importance = sum(node_importance)
        
        for i, node in enumerate(nodes):
            # 计算该节点的采样点数
            importance_ratio = node_importance[i] / total_importance
            node_samples = max(1, int(total_samples * importance_ratio))
            
            # 在节点内采样
            node_t_vals = self._sample_within_node(ray_o, ray_d, node, node_samples)
            node_points = ray_o.unsqueeze(0) + node_t_vals.unsqueeze(-1) * ray_d.unsqueeze(0)
            node_weights = torch.full_like(node_t_vals, importance_ratio / node_samples)
            
            sample_points.append(node_points)
            sample_distances.append(node_t_vals)
            sample_weights.append(node_weights)
        
        # 合并所有节点的采样
        all_points = torch.cat(sample_points, dim=0)
        all_distances = torch.cat(sample_distances, dim=0)
        all_weights = torch.cat(sample_weights, dim=0)
        
        # 按距离排序
        sorted_indices = torch.argsort(all_distances)
        
        return {
            'points': all_points[sorted_indices],
            'distances': all_distances[sorted_indices],
            'weights': all_weights[sorted_indices]
        }
    
    def _compute_node_importance(self, node, target_level):
        """
        计算节点重要性
        """
        # 基于级别的重要性
        level_importance = 1.0 / (abs(node.level - target_level) + 1)
        
        # 基于节点大小的重要性
        size_importance = 1.0 / (node.size + 1e-8)
        
        # 基于历史密度的重要性（如果可用）
        if hasattr(node, 'avg_density'):
            density_importance = node.avg_density
        else:
            density_importance = 1.0
        
        # 组合重要性
        total_importance = (
            level_importance * 0.5 +
            size_importance * 0.3 +
            density_importance * 0.2
        )
        
        return total_importance
    
    def _sample_within_node(self, ray_o, ray_d, node, num_samples):
        """
        在节点内采样
        """
        # 计算射线与节点的交点
        aabb_min, aabb_max = node.get_aabb()
        
        inv_dir = 1.0 / (ray_d + 1e-8)
        t_min = (aabb_min - ray_o) * inv_dir
        t_max = (aabb_max - ray_o) * inv_dir
        
        t_min, t_max = torch.min(t_min, t_max), torch.max(t_min, t_max)
        t_near = torch.max(t_min)
        t_far = torch.min(t_max)
        
        # 在交点范围内采样
        if t_near < t_far:
            t_vals = torch.linspace(
                max(0.0, t_near.item()), 
                t_far.item(), 
                num_samples, 
                device=ray_o.device
            )
        else:
            # 如果没有交点，返回默认采样
            t_vals = torch.linspace(0.1, 10.0, num_samples, device=ray_o.device)
        
        return t_vals
```

### 2. 重要性引导采样

```python
class ImportanceGuidedSampler:
    """
    重要性引导采样器
    基于密度分布和梯度信息进行智能采样
    """
    
    def __init__(self, config: InfNeRFConfig):
        self.config = config
        self.density_cache = {}
        self.gradient_cache = {}
        
    def importance_sampling(self, rays_o, rays_d, coarse_samples, fine_samples, 
                          density_network):
        """
        重要性采样
        
        Args:
            rays_o: 射线起点 [N, 3]
            rays_d: 射线方向 [N, 3]
            coarse_samples: 粗采样点数
            fine_samples: 精采样点数
            density_network: 密度网络
            
        Returns:
            sample_points: 所有采样点 [N, total_samples, 3]
            sample_distances: 采样距离 [N, total_samples]
        """
        device = rays_o.device
        num_rays = rays_o.shape[0]
        
        # 步骤1: 粗采样
        coarse_t = torch.linspace(0.1, 10.0, coarse_samples, device=device)
        coarse_t = coarse_t.expand(num_rays, coarse_samples)
        
        # 添加随机抖动
        if self.training:
            noise = torch.rand_like(coarse_t) * (10.0 - 0.1) / coarse_samples
            coarse_t = coarse_t + noise
        
        # 计算粗采样点
        coarse_points = rays_o.unsqueeze(1) + coarse_t.unsqueeze(-1) * rays_d.unsqueeze(1)
        
        # 步骤2: 获取粗采样密度
        with torch.no_grad():
            coarse_densities = density_network.get_density(
                coarse_points.reshape(-1, 3)
            ).reshape(num_rays, coarse_samples)
        
        # 步骤3: 计算权重
        weights = self._compute_sampling_weights(coarse_densities, coarse_t)
        
        # 步骤4: 重要性采样
        fine_t = self._sample_from_weights(weights, coarse_t, fine_samples)
        
        # 步骤5: 合并粗细采样
        all_t = torch.cat([coarse_t, fine_t], dim=-1)
        all_t, sort_indices = torch.sort(all_t, dim=-1)
        
        # 计算最终采样点
        all_points = rays_o.unsqueeze(1) + all_t.unsqueeze(-1) * rays_d.unsqueeze(1)
        
        return all_points, all_t
    
    def _compute_sampling_weights(self, densities, distances):
        """
        计算采样权重
        """
        # 计算距离间隔
        dists = distances[..., 1:] - distances[..., :-1]
        dists = torch.cat([
            dists,
            torch.full_like(dists[..., :1], 1e10)
        ], dim=-1)
        
        # 计算透明度
        alpha = 1.0 - torch.exp(-densities * dists)
        
        # 计算累积透明度
        transmittance = torch.cumprod(
            torch.cat([
                torch.ones_like(alpha[..., :1]),
                1.0 - alpha[..., :-1]
            ], dim=-1),
            dim=-1
        )
        
        # 计算权重
        weights = alpha * transmittance
        
        return weights
    
    def _sample_from_weights(self, weights, distances, num_samples):
        """
        从权重分布中采样
        """
        device = weights.device
        num_rays = weights.shape[0]
        
        # 权重归一化
        weights = weights + 1e-5  # 避免数值问题
        pdf = weights / torch.sum(weights, dim=-1, keepdim=True)
        cdf = torch.cumsum(pdf, dim=-1)
        cdf = torch.cat([torch.zeros_like(cdf[..., :1]), cdf], dim=-1)
        
        # 均匀采样CDF
        u = torch.rand(num_rays, num_samples, device=device)
        
        # 逆变换采样
        indices = torch.searchsorted(cdf, u, right=True)
        below = torch.clamp(indices - 1, 0, cdf.shape[-1] - 1)
        above = torch.clamp(indices, 0, cdf.shape[-1] - 1)
        
        # 线性插值
        indices_g = torch.stack([below, above], dim=-1)
        matched_shape = [indices_g.shape[0], indices_g.shape[1], cdf.shape[-1]]
        
        cdf_g = torch.gather(cdf.unsqueeze(1).expand(matched_shape), 2, indices_g)
        bins_g = torch.gather(distances.unsqueeze(1).expand(matched_shape), 2, indices_g)
        
        denom = cdf_g[..., 1] - cdf_g[..., 0]
        denom = torch.where(denom < 1e-5, torch.ones_like(denom), denom)
        
        t = (u - cdf_g[..., 0]) / denom
        samples = bins_g[..., 0] + t * (bins_g[..., 1] - bins_g[..., 0])
        
        return samples
```

---

## 小结

本文档详细介绍了 Inf-NeRF 渲染机制的核心技术，包括：

1. **多尺度神经网络**: LoD感知的网络架构和自适应复杂度控制
2. **哈希编码优化**: 多分辨率哈希编码和自适应分辨率管理
3. **自适应采样策略**: 层次化采样和重要性引导采样
4. **性能优化**: 网络剪枝和内存管理

这些技术的结合使得 Inf-NeRF 能够高效地处理无限尺度场景，在保证渲染质量的同时实现 O(log n) 的空间复杂度。

下一部分将介绍抗锯齿渲染、内存优化和实际实现细节。

---

**说明**: 这是 Inf-NeRF 渲染文档的第二部分。建议与第一部分（渲染基础）和第三部分（优化实现）结合阅读。
