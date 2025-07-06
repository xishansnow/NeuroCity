# Block-NeRF 渲染机制详解 - 第二部分：外观匹配与块间合成

**版本**: 1.0  
**日期**: 2025年7月5日  
**依赖**: 第一部分 - 渲染基础

## 概述

Block-NeRF 的核心挑战之一是如何将多个独立训练的块无缝合成为统一的图像。这需要解决外观不一致、边界伪影和深度冲突等问题。本部分详细介绍外观匹配算法、块间合成策略和深度融合技术。

## 目录

1. [外观匹配机制](#外观匹配机制)
2. [块间合成策略](#块间合成策略)
3. [深度融合技术](#深度融合技术)
4. [边界处理算法](#边界处理算法)
5. [颜色一致性优化](#颜色一致性优化)
6. [实时合成优化](#实时合成优化)

---

## 外观匹配机制

### 1. 外观嵌入对齐

不同块可能有不同的外观嵌入空间，需要对齐：

```python
class AppearanceAligner:
    """
    外观嵌入对齐器，处理不同块间的外观一致性
    """
    
    def __init__(self, embedding_dim=48):
        self.embedding_dim = embedding_dim
        self.alignment_networks = {}
        self.reference_embeddings = {}
        
    def compute_reference_appearance(self, overlap_region, block_a, block_b):
        """
        在重叠区域计算参考外观
        
        Args:
            overlap_region: 重叠区域定义 Dict
            block_a, block_b: 两个相邻块
            
        Returns:
            reference_appearance: 参考外观参数
        """
        # 在重叠区域采样点
        sample_points = self.sample_overlap_region(overlap_region, num_samples=1000)
        
        # 获取两个块在这些点的渲染结果
        with torch.no_grad():
            colors_a = []
            colors_b = []
            
            for point in sample_points:
                # 使用中性外观嵌入渲染
                neutral_embedding = torch.zeros(self.embedding_dim)
                
                color_a = block_a.render_point(point, neutral_embedding)
                color_b = block_b.render_point(point, neutral_embedding)
                
                colors_a.append(color_a)
                colors_b.append(color_b)
            
            colors_a = torch.stack(colors_a)
            colors_b = torch.stack(colors_b)
        
        # 计算颜色偏移和缩放
        color_offset = torch.mean(colors_a - colors_b, dim=0)
        color_scale = torch.std(colors_a, dim=0) / (torch.std(colors_b, dim=0) + 1e-8)
        
        return {
            'offset': color_offset,
            'scale': color_scale,
            'sample_points': sample_points,
            'colors_a': colors_a,
            'colors_b': colors_b
        }
    
    def optimize_appearance_alignment(self, block_a, block_b, overlap_data):
        """
        优化外观对齐参数
        
        Args:
            block_a: 参考块
            block_b: 待对齐块
            overlap_data: 重叠区域数据
            
        Returns:
            alignment_params: 对齐参数
        """
        # 初始化对齐网络
        alignment_net = nn.Sequential(
            nn.Linear(self.embedding_dim, self.embedding_dim * 2),
            nn.ReLU(),
            nn.Linear(self.embedding_dim * 2, self.embedding_dim * 2),
            nn.ReLU(),
            nn.Linear(self.embedding_dim * 2, self.embedding_dim)
        ).to(block_b.device)
        
        optimizer = torch.optim.Adam(alignment_net.parameters(), lr=1e-3)
        
        sample_points = overlap_data['sample_points']
        target_colors = overlap_data['colors_a']
        
        for iteration in range(1000):
            optimizer.zero_grad()
            
            total_loss = 0
            batch_size = 32
            
            for i in range(0, len(sample_points), batch_size):
                batch_points = sample_points[i:i+batch_size]
                batch_targets = target_colors[i:i+batch_size]
                
                # 获取块B的原始外观嵌入
                original_embedding = block_b.get_appearance_embedding(
                    torch.zeros(len(batch_points), dtype=torch.long)
                )
                
                # 对齐变换
                aligned_embedding = alignment_net(original_embedding)
                
                # 渲染
                predicted_colors = block_b.render_points(
                    batch_points, aligned_embedding
                )
                
                # 损失计算
                loss = F.mse_loss(predicted_colors, batch_targets)
                total_loss += loss
            
            total_loss.backward()
            optimizer.step()
            
            if iteration % 100 == 0:
                print(f"Alignment iteration {iteration}, loss: {total_loss.item():.6f}")
        
        return alignment_net
```

### 2. 动态外观匹配

实时渲染时的外观匹配：

```python
class DynamicAppearanceMatcher:
    """
    动态外观匹配器，用于实时渲染中的外观一致性
    """
    
    def __init__(self, config):
        self.config = config
        self.cache = {}
        self.temporal_consistency = config.temporal_consistency
        
    def match_appearance_real_time(self, block_renderings, overlap_masks, camera_pose):
        """
        实时外观匹配
        
        Args:
            block_renderings: 块渲染结果 List[Dict]
            overlap_masks: 重叠掩码 List[Tensor]
            camera_pose: 当前相机姿态
            
        Returns:
            matched_renderings: 外观匹配后的渲染结果
        """
        if len(block_renderings) <= 1:
            return block_renderings
        
        # 计算匹配键（用于缓存）
        cache_key = self.compute_cache_key(block_renderings, camera_pose)
        
        if cache_key in self.cache:
            return self.cache[cache_key]
        
        matched_renderings = []
        
        # 选择参考块（通常是占像素最多的块）
        reference_idx = self.select_reference_block(block_renderings, overlap_masks)
        reference_rendering = block_renderings[reference_idx]
        
        for i, rendering in enumerate(block_renderings):
            if i == reference_idx:
                matched_renderings.append(rendering)
                continue
            
            # 计算与参考块的重叠区域
            overlap_mask = self.compute_overlap_mask(
                overlap_masks[reference_idx], overlap_masks[i]
            )
            
            if torch.sum(overlap_mask) < 100:  # 重叠区域太小
                matched_renderings.append(rendering)
                continue
            
            # 快速外观匹配
            matched_rendering = self.fast_appearance_match(
                rendering, reference_rendering, overlap_mask
            )
            
            matched_renderings.append(matched_rendering)
        
        # 缓存结果
        self.cache[cache_key] = matched_renderings
        
        return matched_renderings
    
    def fast_appearance_match(self, source_rendering, target_rendering, overlap_mask):
        """
        快速外观匹配算法
        
        Args:
            source_rendering: 源渲染结果
            target_rendering: 目标渲染结果  
            overlap_mask: 重叠区域掩码
            
        Returns:
            matched_rendering: 匹配后的渲染结果
        """
        source_rgb = source_rendering['rgb']
        target_rgb = target_rendering['rgb']
        
        # 在重叠区域计算统计信息
        source_overlap = source_rgb[overlap_mask]
        target_overlap = target_rgb[overlap_mask]
        
        if len(source_overlap) == 0:
            return source_rendering
        
        # 计算颜色统计
        source_mean = torch.mean(source_overlap, dim=0)
        source_std = torch.std(source_overlap, dim=0) + 1e-8
        
        target_mean = torch.mean(target_overlap, dim=0)
        target_std = torch.std(target_overlap, dim=0) + 1e-8
        
        # 颜色匹配变换
        matched_rgb = (source_rgb - source_mean) / source_std * target_std + target_mean
        
        # 创建匹配后的渲染结果
        matched_rendering = source_rendering.copy()
        matched_rendering['rgb'] = matched_rgb
        
        return matched_rendering
```

---

## 块间合成策略

### 1. 权重计算与归一化

```python
class BlockWeightCalculator:
    """
    计算块合成权重的各种策略
    """
    
    def __init__(self, config):
        self.config = config
        self.weight_method = config.weight_method
        
    def compute_block_weights(self, block_renderings, visibility_scores, overlap_masks, camera_pose):
        """
        计算块合成权重
        
        Args:
            block_renderings: 块渲染结果 List[Dict]
            visibility_scores: 可见性得分 List[float]
            overlap_masks: 重叠掩码 List[Tensor]
            camera_pose: 相机姿态
            
        Returns:
            weights: 归一化权重 List[Tensor]
        """
        if self.weight_method == 'visibility':
            return self.visibility_based_weights(visibility_scores, overlap_masks)
        elif self.weight_method == 'distance':
            return self.distance_based_weights(block_renderings, camera_pose)
        elif self.weight_method == 'uncertainty':
            return self.uncertainty_based_weights(block_renderings)
        elif self.weight_method == 'adaptive':
            return self.adaptive_weights(block_renderings, visibility_scores, camera_pose)
        else:
            raise ValueError(f"Unknown weight method: {self.weight_method}")
    
    def visibility_based_weights(self, visibility_scores, overlap_masks):
        """
        基于可见性的权重计算
        """
        weights = []
        
        for i, (score, mask) in enumerate(zip(visibility_scores, overlap_masks)):
            # 基础权重
            base_weight = torch.full_like(mask, score, dtype=torch.float32)
            
            # 距离边界的衰减
            distance_to_boundary = self.compute_boundary_distance(mask)
            boundary_weight = torch.sigmoid(distance_to_boundary * 0.1)
            
            # 组合权重
            combined_weight = base_weight * boundary_weight
            weights.append(combined_weight)
        
        return self.normalize_weights(weights, overlap_masks)
    
    def distance_based_weights(self, block_renderings, camera_pose):
        """
        基于距离的权重计算
        """
        camera_pos = camera_pose[:3, 3]
        weights = []
        
        for rendering in block_renderings:
            block_center = rendering['block_center']
            distance = torch.norm(camera_pos - block_center)
            
            # 距离权重（近距离权重更高）
            dist_weight = 1.0 / (1.0 + distance / 100.0)
            
            # 深度一致性权重
            depth_map = rendering['depth']
            depth_variance = torch.var(depth_map)
            depth_weight = torch.exp(-depth_variance)
            
            # 组合权重
            combined_weight = torch.full_like(
                depth_map, dist_weight * depth_weight, dtype=torch.float32
            )
            
            weights.append(combined_weight)
        
        return self.normalize_weights(weights)
    
    def uncertainty_based_weights(self, block_renderings):
        """
        基于不确定性的权重计算
        """
        weights = []
        
        for rendering in block_renderings:
            if 'uncertainty' in rendering:
                uncertainty = rendering['uncertainty']
                # 不确定性越低，权重越高
                weight = torch.exp(-uncertainty)
            else:
                # 使用密度作为置信度代理
                density = rendering.get('density', torch.ones_like(rendering['depth']))
                weight = torch.sigmoid(density - 0.5)
            
            weights.append(weight)
        
        return self.normalize_weights(weights)
    
    def adaptive_weights(self, block_renderings, visibility_scores, camera_pose):
        """
        自适应权重计算（组合多种策略）
        """
        # 计算各种权重
        vis_weights = self.visibility_based_weights(
            visibility_scores, [r.get('mask', torch.ones_like(r['depth'])) for r in block_renderings]
        )
        dist_weights = self.distance_based_weights(block_renderings, camera_pose)
        unc_weights = self.uncertainty_based_weights(block_renderings)
        
        # 动态权重组合
        adaptive_weights = []
        for i in range(len(block_renderings)):
            # 根据场景特性调整权重组合
            alpha_vis = 0.4
            alpha_dist = 0.3
            alpha_unc = 0.3
            
            combined = (alpha_vis * vis_weights[i] + 
                       alpha_dist * dist_weights[i] + 
                       alpha_unc * unc_weights[i])
            
            adaptive_weights.append(combined)
        
        return self.normalize_weights(adaptive_weights)
    
    def normalize_weights(self, weights, masks=None):
        """
        权重归一化
        """
        if masks is None:
            masks = [torch.ones_like(w) for w in weights]
        
        # 计算有效权重和
        valid_masks = []
        for i, (weight, mask) in enumerate(zip(weights, masks)):
            valid_mask = (mask > 0) & (weight > 1e-8)
            valid_masks.append(valid_mask)
        
        # 归一化
        normalized_weights = []
        for i in range(len(weights)):
            # 创建归一化掩码
            normalization_mask = torch.zeros_like(weights[i])
            
            for j, valid_mask in enumerate(valid_masks):
                normalization_mask += valid_mask.float() * weights[j]
            
            # 避免除零
            normalization_mask = torch.clamp(normalization_mask, min=1e-8)
            
            # 归一化权重
            normalized_weight = weights[i] / normalization_mask
            normalized_weight = torch.where(
                valid_masks[i], normalized_weight, torch.zeros_like(normalized_weight)
            )
            
            normalized_weights.append(normalized_weight)
        
        return normalized_weights
```

### 2. 高级合成算法

```python
class AdvancedBlockCompositor:
    """
    高级块合成器，支持多种合成策略
    """
    
    def __init__(self, config):
        self.config = config
        self.blend_method = config.blend_method
        self.feather_size = config.feather_size
        
    def composite_blocks_advanced(self, block_renderings, weights, method='poisson'):
        """
        高级块合成
        
        Args:
            block_renderings: 块渲染结果
            weights: 归一化权重
            method: 合成方法 ('weighted', 'poisson', 'multiband')
            
        Returns:
            composite_result: 合成结果
        """
        if method == 'weighted':
            return self.weighted_composition(block_renderings, weights)
        elif method == 'poisson':
            return self.poisson_composition(block_renderings, weights)
        elif method == 'multiband':
            return self.multiband_composition(block_renderings, weights)
        else:
            raise ValueError(f"Unknown composition method: {method}")
    
    def weighted_composition(self, block_renderings, weights):
        """
        加权合成（基础方法）
        """
        composite_rgb = torch.zeros_like(block_renderings[0]['rgb'])
        composite_depth = torch.zeros_like(block_renderings[0]['depth'])
        
        for rendering, weight in zip(block_renderings, weights):
            composite_rgb += weight.unsqueeze(-1) * rendering['rgb']
            composite_depth += weight * rendering['depth']
        
        return {
            'rgb': composite_rgb,
            'depth': composite_depth,
            'method': 'weighted'
        }
    
    def poisson_composition(self, block_renderings, weights):
        """
        泊松合成（梯度域合成）
        """
        if len(block_renderings) != 2:
            # 对于多于2个块，递归应用
            return self.recursive_poisson_composition(block_renderings, weights)
        
        source = block_renderings[1]['rgb']
        target = block_renderings[0]['rgb']
        mask = weights[1] > 0.5
        
        # 计算梯度场
        source_grad = self.compute_gradient(source)
        target_grad = self.compute_gradient(target)
        
        # 混合梯度
        mixed_grad = torch.where(
            mask.unsqueeze(-1).unsqueeze(-1),
            source_grad,
            target_grad
        )
        
        # 泊松求解
        composite_rgb = self.solve_poisson(mixed_grad, target, mask)
        
        # 深度合成（简单加权）
        composite_depth = torch.where(
            mask,
            block_renderings[1]['depth'],
            block_renderings[0]['depth']
        )
        
        return {
            'rgb': composite_rgb,
            'depth': composite_depth,
            'method': 'poisson'
        }
    
    def multiband_composition(self, block_renderings, weights):
        """
        多频带合成（拉普拉斯金字塔）
        """
        # 构建拉普拉斯金字塔
        pyramids = []
        weight_pyramids = []
        
        for rendering, weight in zip(block_renderings, weights):
            rgb_pyramid = self.build_laplacian_pyramid(rendering['rgb'])
            weight_pyramid = self.build_gaussian_pyramid(weight)
            
            pyramids.append(rgb_pyramid)
            weight_pyramids.append(weight_pyramid)
        
        # 在每个频带合成
        composite_pyramid = []
        num_levels = len(pyramids[0])
        
        for level in range(num_levels):
            level_weights = [wp[level] for wp in weight_pyramids]
            level_images = [p[level] for p in pyramids]
            
            # 归一化权重
            total_weight = sum(level_weights)
            total_weight = torch.clamp(total_weight, min=1e-8)
            
            level_composite = torch.zeros_like(level_images[0])
            for img, weight in zip(level_images, level_weights):
                level_composite += (weight / total_weight).unsqueeze(-1) * img
            
            composite_pyramid.append(level_composite)
        
        # 重建图像
        composite_rgb = self.reconstruct_from_pyramid(composite_pyramid)
        
        # 深度合成（在最低分辨率）
        composite_depth = self.weighted_composition(
            [{'depth': r['depth']} for r in block_renderings], weights
        )['depth']
        
        return {
            'rgb': composite_rgb,
            'depth': composite_depth,
            'method': 'multiband'
        }
```

---

## 深度融合技术

### 1. 深度一致性检查

```python
class DepthConsistencyChecker:
    """
    深度一致性检查和融合
    """
    
    def __init__(self, config):
        self.depth_threshold = config.depth_threshold
        self.consistency_weight = config.consistency_weight
        
    def check_depth_consistency(self, block_renderings, overlap_masks):
        """
        检查块间深度一致性
        
        Args:
            block_renderings: 块渲染结果
            overlap_masks: 重叠掩码
            
        Returns:
            consistency_map: 一致性图
            conflict_regions: 冲突区域
        """
        if len(block_renderings) < 2:
            return None, None
        
        H, W = block_renderings[0]['depth'].shape
        consistency_map = torch.ones(H, W)
        conflict_regions = torch.zeros(H, W, dtype=torch.bool)
        
        # 两两比较深度
        for i in range(len(block_renderings)):
            for j in range(i + 1, len(block_renderings)):
                depth_i = block_renderings[i]['depth']
                depth_j = block_renderings[j]['depth']
                
                # 计算重叠区域
                overlap_mask = overlap_masks[i] & overlap_masks[j]
                
                if torch.sum(overlap_mask) == 0:
                    continue
                
                # 深度差异
                depth_diff = torch.abs(depth_i - depth_j)
                depth_diff_normalized = depth_diff / (torch.max(depth_i, depth_j) + 1e-8)
                
                # 一致性得分
                consistency = torch.exp(-depth_diff_normalized / self.depth_threshold)
                
                # 更新一致性图
                consistency_map = torch.where(
                    overlap_mask,
                    torch.minimum(consistency_map, consistency),
                    consistency_map
                )
                
                # 标记冲突区域
                conflict_mask = overlap_mask & (depth_diff_normalized > self.depth_threshold)
                conflict_regions |= conflict_mask
        
        return consistency_map, conflict_regions
    
    def resolve_depth_conflicts(self, block_renderings, conflict_regions, weights):
        """
        解决深度冲突
        
        Args:
            block_renderings: 块渲染结果
            conflict_regions: 冲突区域
            weights: 块权重
            
        Returns:
            resolved_depth: 解决冲突后的深度图
        """
        if torch.sum(conflict_regions) == 0:
            # 无冲突，直接加权平均
            return self.weighted_depth_fusion(block_renderings, weights)
        
        # 在冲突区域使用特殊策略
        resolved_depth = torch.zeros_like(block_renderings[0]['depth'])
        
        # 非冲突区域：加权平均
        non_conflict_mask = ~conflict_regions
        for rendering, weight in zip(block_renderings, weights):
            resolved_depth += weight * rendering['depth'] * non_conflict_mask.float()
        
        # 冲突区域：使用中位数或最高权重
        conflict_pixels = torch.where(conflict_regions)
        
        for pixel_idx in range(len(conflict_pixels[0])):
            y, x = conflict_pixels[0][pixel_idx], conflict_pixels[1][pixel_idx]
            
            # 收集该像素的所有深度值和权重
            depths = []
            pixel_weights = []
            
            for rendering, weight in zip(block_renderings, weights):
                if weight[y, x] > 1e-8:
                    depths.append(rendering['depth'][y, x])
                    pixel_weights.append(weight[y, x])
            
            if depths:
                depths = torch.stack(depths)
                pixel_weights = torch.stack(pixel_weights)
                
                # 使用加权中位数
                resolved_depth[y, x] = self.weighted_median(depths, pixel_weights)
        
        return resolved_depth
    
    def weighted_median(self, values, weights):
        """
        计算加权中位数
        """
        # 排序
        sorted_indices = torch.argsort(values)
        sorted_values = values[sorted_indices]
        sorted_weights = weights[sorted_indices]
        
        # 累积权重
        cumulative_weights = torch.cumsum(sorted_weights, dim=0)
        total_weight = cumulative_weights[-1]
        
        # 找到中位数位置
        median_position = total_weight / 2
        median_idx = torch.searchsorted(cumulative_weights, median_position)
        median_idx = torch.clamp(median_idx, 0, len(sorted_values) - 1)
        
        return sorted_values[median_idx]
```

### 2. 深度补全与平滑

```python
class DepthCompletion:
    """
    深度补全和平滑处理
    """
    
    def __init__(self, config):
        self.config = config
        self.inpainting_network = self.build_inpainting_network()
        
    def complete_depth_map(self, depth_map, validity_mask):
        """
        补全深度图中的空洞
        
        Args:
            depth_map: 原始深度图 [H, W]
            validity_mask: 有效性掩码 [H, W]
            
        Returns:
            completed_depth: 补全后的深度图
        """
        if torch.sum(~validity_mask) == 0:
            return depth_map
        
        # 使用传统方法快速补全
        completed_depth = self.traditional_inpainting(depth_map, validity_mask)
        
        # 使用神经网络精细化
        if self.config.use_neural_completion:
            completed_depth = self.neural_inpainting(completed_depth, validity_mask)
        
        return completed_depth
    
    def traditional_inpainting(self, depth_map, validity_mask):
        """
        传统深度补全方法
        """
        completed = depth_map.clone()
        
        # 距离变换引导的插值
        invalid_mask = ~validity_mask
        
        if torch.sum(invalid_mask) == 0:
            return completed
        
        # 计算到最近有效像素的距离
        valid_coords = torch.where(validity_mask)
        invalid_coords = torch.where(invalid_mask)
        
        for i in range(len(invalid_coords[0])):
            y, x = invalid_coords[0][i], invalid_coords[1][i]
            
            # 找到最近的有效像素
            distances = torch.sqrt(
                (valid_coords[0].float() - y) ** 2 + 
                (valid_coords[1].float() - x) ** 2
            )
            
            # 使用K个最近邻进行加权平均
            k = min(8, len(distances))
            _, nearest_indices = torch.topk(distances, k, largest=False)
            
            weights = 1.0 / (distances[nearest_indices] + 1e-8)
            weights = weights / torch.sum(weights)
            
            # 加权平均深度
            nearest_depths = depth_map[
                valid_coords[0][nearest_indices],
                valid_coords[1][nearest_indices]
            ]
            
            completed[y, x] = torch.sum(weights * nearest_depths)
        
        return completed
    
    def smooth_depth_map(self, depth_map, rgb_image, edge_threshold=0.1):
        """
        基于RGB图像的深度平滑
        
        Args:
            depth_map: 深度图 [H, W]
            rgb_image: RGB图像 [H, W, 3]
            edge_threshold: 边缘阈值
            
        Returns:
            smoothed_depth: 平滑后的深度图
        """
        # 计算RGB梯度作为边缘指导
        rgb_gray = torch.mean(rgb_image, dim=-1)
        
        # Sobel算子计算梯度
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32)
        sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32)
        
        grad_x = F.conv2d(
            rgb_gray.unsqueeze(0).unsqueeze(0),
            sobel_x.unsqueeze(0).unsqueeze(0),
            padding=1
        ).squeeze()
        
        grad_y = F.conv2d(
            rgb_gray.unsqueeze(0).unsqueeze(0),
            sobel_y.unsqueeze(0).unsqueeze(0),
            padding=1
        ).squeeze()
        
        # 边缘强度
        edge_strength = torch.sqrt(grad_x ** 2 + grad_y ** 2)
        edge_mask = edge_strength > edge_threshold
        
        # 双边滤波权重
        smoothed_depth = depth_map.clone()
        
        kernel_size = 5
        sigma_spatial = 2.0
        sigma_range = 0.1
        
        pad = kernel_size // 2
        padded_depth = F.pad(depth_map, (pad, pad, pad, pad), mode='reflect')
        padded_edge = F.pad(edge_mask.float(), (pad, pad, pad, pad), mode='reflect')
        
        for y in range(depth_map.shape[0]):
            for x in range(depth_map.shape[1]):
                if edge_mask[y, x]:
                    continue  # 保持边缘不变
                
                # 提取邻域
                neighborhood_depth = padded_depth[y:y+kernel_size, x:x+kernel_size]
                neighborhood_edge = padded_edge[y:y+kernel_size, x:x+kernel_size]
                
                center_depth = depth_map[y, x]
                
                # 空间权重
                spatial_weights = torch.zeros(kernel_size, kernel_size)
                for dy in range(kernel_size):
                    for dx in range(kernel_size):
                        dist = (dy - pad) ** 2 + (dx - pad) ** 2
                        spatial_weights[dy, dx] = torch.exp(-dist / (2 * sigma_spatial ** 2))
                
                # 范围权重
                range_weights = torch.exp(
                    -(neighborhood_depth - center_depth) ** 2 / (2 * sigma_range ** 2)
                )
                
                # 边缘权重（避免跨边缘平滑）
                edge_weights = 1.0 - neighborhood_edge
                
                # 组合权重
                total_weights = spatial_weights * range_weights * edge_weights
                total_weights = total_weights / torch.sum(total_weights)
                
                # 加权平均
                smoothed_depth[y, x] = torch.sum(total_weights * neighborhood_depth)
        
        return smoothed_depth
```

---

## 边界处理算法

### 1. 软边界生成

```python
class SoftBoundaryGenerator:
    """
    生成软边界用于平滑块间过渡
    """
    
    def __init__(self, feather_size=20):
        self.feather_size = feather_size
        
    def generate_soft_boundaries(self, block_masks, overlap_regions):
        """
        为块掩码生成软边界
        
        Args:
            block_masks: 块掩码列表 List[Tensor[H, W]]
            overlap_regions: 重叠区域定义
            
        Returns:
            soft_masks: 软边界掩码 List[Tensor[H, W]]
        """
        soft_masks = []
        
        for i, mask in enumerate(block_masks):
            soft_mask = self.create_soft_mask(mask, overlap_regions)
            soft_masks.append(soft_mask)
        
        return soft_masks
    
    def create_soft_mask(self, hard_mask, overlap_regions):
        """
        从硬掩码创建软掩码
        
        Args:
            hard_mask: 硬掩码 [H, W]
            overlap_regions: 重叠区域信息
            
        Returns:
            soft_mask: 软掩码 [H, W]
        """
        # 计算距离变换
        distance_map = self.compute_distance_transform(hard_mask)
        
        # 生成软边界
        soft_mask = torch.sigmoid(
            (distance_map - self.feather_size/2) / (self.feather_size/4)
        )
        
        # 在重叠区域应用特殊处理
        for overlap_region in overlap_regions:
            if self.mask_overlaps_region(hard_mask, overlap_region):
                soft_mask = self.apply_overlap_feathering(
                    soft_mask, overlap_region
                )
        
        return soft_mask
    
    def compute_distance_transform(self, mask):
        """
        计算距离变换
        """
        # 简化的距离变换实现
        H, W = mask.shape
        distance_map = torch.zeros_like(mask, dtype=torch.float32)
        
        # 边界像素
        boundary_mask = self.detect_boundary(mask)
        
        # 对每个像素计算到边界的最小距离
        boundary_coords = torch.where(boundary_mask)
        
        for y in range(H):
            for x in range(W):
                if mask[y, x]:
                    # 内部像素：计算到边界的距离
                    distances = torch.sqrt(
                        (boundary_coords[0].float() - y) ** 2 +
                        (boundary_coords[1].float() - x) ** 2
                    )
                    distance_map[y, x] = torch.min(distances) if len(distances) > 0 else 0
                else:
                    # 外部像素：负距离
                    distances = torch.sqrt(
                        (boundary_coords[0].float() - y) ** 2 +
                        (boundary_coords[1].float() - x) ** 2
                    )
                    distance_map[y, x] = -torch.min(distances) if len(distances) > 0 else 0
        
        return distance_map
    
    def detect_boundary(self, mask):
        """
        检测掩码边界
        """
        # 使用形态学操作检测边界
        kernel = torch.ones(3, 3)
        
        # 腐蚀操作
        eroded = F.conv2d(
            mask.float().unsqueeze(0).unsqueeze(0),
            kernel.unsqueeze(0).unsqueeze(0),
            padding=1
        ).squeeze() == 9  # 所有邻居都为1
        
        # 边界 = 原始 - 腐蚀
        boundary = mask & (~eroded)
        
        return boundary
```

### 2. 自适应羽化

```python
class AdaptiveFeathering:
    """
    基于内容的自适应羽化
    """
    
    def __init__(self, config):
        self.config = config
        self.feature_detector = self.build_feature_detector()
        
    def adaptive_feather(self, mask, rgb_image, depth_map):
        """
        自适应羽化处理
        
        Args:
            mask: 原始掩码 [H, W]
            rgb_image: RGB图像 [H, W, 3]
            depth_map: 深度图 [H, W]
            
        Returns:
            feathered_mask: 羽化后的掩码
        """
        # 检测图像特征
        edge_map = self.detect_edges(rgb_image)
        texture_map = self.detect_texture(rgb_image)
        depth_grad = self.compute_depth_gradient(depth_map)
        
        # 计算自适应羽化大小
        feather_size_map = self.compute_adaptive_feather_size(
            edge_map, texture_map, depth_grad
        )
        
        # 应用变化的羽化
        feathered_mask = self.variable_feathering(mask, feather_size_map)
        
        return feathered_mask
    
    def detect_edges(self, rgb_image):
        """
        边缘检测
        """
        gray = torch.mean(rgb_image, dim=-1)
        
        # Canny边缘检测的简化版本
        # 高斯平滑
        gaussian_kernel = self.create_gaussian_kernel(5, 1.0)
        smoothed = F.conv2d(
            gray.unsqueeze(0).unsqueeze(0),
            gaussian_kernel.unsqueeze(0).unsqueeze(0),
            padding=2
        ).squeeze()
        
        # 梯度计算
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32)
        sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32)
        
        grad_x = F.conv2d(
            smoothed.unsqueeze(0).unsqueeze(0),
            sobel_x.unsqueeze(0).unsqueeze(0),
            padding=1
        ).squeeze()
        
        grad_y = F.conv2d(
            smoothed.unsqueeze(0).unsqueeze(0),
            sobel_y.unsqueeze(0).unsqueeze(0),
            padding=1
        ).squeeze()
        
        edge_magnitude = torch.sqrt(grad_x ** 2 + grad_y ** 2)
        
        return edge_magnitude
    
    def detect_texture(self, rgb_image):
        """
        纹理检测
        """
        gray = torch.mean(rgb_image, dim=-1)
        
        # 使用局部标准差作为纹理度量
        kernel_size = 7
        pad = kernel_size // 2
        
        # 局部均值
        mean_kernel = torch.ones(kernel_size, kernel_size) / (kernel_size ** 2)
        local_mean = F.conv2d(
            gray.unsqueeze(0).unsqueeze(0),
            mean_kernel.unsqueeze(0).unsqueeze(0),
            padding=pad
        ).squeeze()
        
        # 局部方差
        squared_diff = (gray - local_mean) ** 2
        local_var = F.conv2d(
            squared_diff.unsqueeze(0).unsqueeze(0),
            mean_kernel.unsqueeze(0).unsqueeze(0),
            padding=pad
        ).squeeze()
        
        texture_map = torch.sqrt(local_var)
        
        return texture_map
    
    def compute_adaptive_feather_size(self, edge_map, texture_map, depth_grad):
        """
        计算自适应羽化大小
        """
        # 基础羽化大小
        base_feather = self.config.base_feather_size
        
        # 边缘区域减少羽化
        edge_factor = torch.exp(-edge_map / 0.1)
        
        # 纹理区域调整羽化
        texture_factor = 1.0 + texture_map / torch.max(texture_map)
        
        # 深度梯度影响
        depth_factor = torch.exp(-depth_grad / 1.0)
        
        # 组合因子
        adaptive_size = base_feather * edge_factor * texture_factor * depth_factor
        adaptive_size = torch.clamp(adaptive_size, min=2.0, max=50.0)
        
        return adaptive_size
    
    def variable_feathering(self, mask, feather_size_map):
        """
        变化羽化处理
        """
        # 这是一个简化实现，实际中需要更复杂的算法
        distance_map = self.compute_distance_transform(mask)
        
        # 自适应sigmoid
        feathered_mask = torch.sigmoid(
            distance_map / (feather_size_map / 4)
        )
        
        return feathered_mask
```

---

## 小结

本文档详细介绍了 Block-NeRF 渲染机制的高级部分，包括：

1. **外观匹配**: 外观嵌入对齐和动态匹配算法
2. **块间合成**: 多种权重计算策略和高级合成方法
3. **深度融合**: 深度一致性检查、冲突解决和深度补全
4. **边界处理**: 软边界生成和自适应羽化技术

这些技术确保了多个独立训练的块能够无缝合成为高质量的统一图像。下一部分将介绍性能优化和实时渲染技术。

---

**说明**: 这是 Block-NeRF 渲染文档的第二部分。建议与第一部分（渲染基础）和第三部分（性能优化）结合阅读。
