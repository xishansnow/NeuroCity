# SVRaster 基于投影的光栅化设计

## 概述

本文档详细介绍了 SVRaster 渲染器中新增的基于投影的光栅化设计。这个设计遵循 SVRaster 论文的核心思想，通过将稀疏体素投影到图像空间，并使用分块处理来实现高效的推理渲染。

## 设计理念

### 1. 投影式渲染 vs 体积渲染

传统的 NeRF 使用体积渲染，沿着光线进行积分：
```
C(r) = ∫ T(t) σ(r(t)) c(r(t), d) dt
```

而 SVRaster 采用投影式渲染：
1. **体素投影**：将 3D 体素直接投影到 2D 屏幕空间
2. **光栅化**：使用传统图形学管线进行像素级渲染
3. **分块处理**：将图像分割成小块，实现并行处理

### 2. 核心优势

- **高效推理**：避免沿光线积分，直接投影体素
- **GPU 友好**：分块设计更好地利用 GPU 并行性
- **内存局部性**：每个分块独立处理，提高缓存效率
- **负载均衡**：动态分配体素到分块，平衡计算负载

## 架构设计

### 1. 配置系统

```python
@dataclass
class TileConfig:
    """图像分块配置"""
    tile_size: int = 64          # 分块大小
    overlap: int = 8             # 分块重叠像素
    use_adaptive_tiling: bool = True  # 自适应分块
    min_tile_size: int = 32      # 最小分块大小
    max_tile_size: int = 128     # 最大分块大小

@dataclass
class FrustumCullingConfig:
    """视锥剔除配置"""
    enable_frustum_culling: bool = True
    culling_margin: float = 0.1  # 剔除边界余量
    use_octree_culling: bool = True  # 使用八叉树加速
    max_culling_depth: int = 8   # 最大剔除深度

@dataclass
class DepthSortingConfig:
    """深度排序配置"""
    enable_depth_sorting: bool = True
    sort_method: str = "back_to_front"  # 排序方法
    use_bucket_sort: bool = True  # 使用桶排序加速
    bucket_count: int = 100       # 桶数量
```

### 2. 渲染管线

```
输入体素数据
    ↓
投影到屏幕空间
    ↓
视锥剔除
    ↓
深度排序
    ↓
图像分块
    ↓
分块光栅化
    ↓
合并结果
    ↓
输出图像
```

## 核心组件

### 1. 图像分块 (ImageTile)

```python
class ImageTile:
    """图像分块类 - 用于并行处理"""
    
    def __init__(self, x: int, y: int, width: int, height: int, tile_id: int):
        self.x = x
        self.y = y
        self.width = width
        self.height = height
        self.tile_id = tile_id
        self.voxels_in_tile: List[dict] = []
```

**功能**：
- 定义图像分块的边界和属性
- 管理分配到该分块的体素
- 支持重叠分块以处理边界体素

### 2. 增强的投影系统

```python
def _project_voxels_to_screen_enhanced(self, voxels, camera_pose, intrinsics, viewport_size):
    """增强的体素投影到屏幕空间"""
    # 1. 世界坐标到相机坐标变换
    positions_hom = torch.cat([positions, torch.ones(...)], dim=1)
    camera_positions = torch.matmul(positions_hom, camera_pose.T)
    
    # 2. 透视投影到屏幕空间
    screen_positions = torch.matmul(camera_positions, intrinsics.T)
    screen_positions = screen_positions[:, :2] / screen_positions[:, 2:3]
    
    # 3. 计算投影后的边界框
    projected_bounds = self._compute_projected_bounds(screen_positions, screen_sizes)
```

**特点**：
- 支持透视投影
- 计算投影后的边界框用于后续优化
- 考虑体素尺寸的透视缩放

### 3. 视锥剔除优化

```python
def _frustum_culling_enhanced(self, screen_voxels, viewport_size):
    """增强的视锥剔除"""
    for voxel in screen_voxels:
        # 深度剔除
        if depth <= near_plane or depth >= far_plane:
            continue
            
        # 屏幕边界剔除（考虑边界余量）
        margin_pixels = margin * size
        if (min_x - margin_pixels < width and max_x + margin_pixels >= 0 and
            min_y - margin_pixels < height and max_y + margin_pixels >= 0):
            visible_voxels.append(voxel)
```

**优化策略**：
- 深度范围剔除
- 屏幕边界剔除
- 边界余量处理
- 支持八叉树加速（未来扩展）

### 4. 深度排序

```python
def _depth_sort_enhanced(self, visible_voxels):
    """增强的深度排序"""
    if method == "back_to_front":
        return sorted(visible_voxels, key=lambda v: v["depth"].item(), reverse=True)
    elif method == "front_to_back":
        return sorted(visible_voxels, key=lambda v: v["depth"].item(), reverse=False)
```

**排序方法**：
- **后向前 (Back-to-Front)**：适合透明体素的 alpha blending
- **前向后 (Front-to-Back)**：适合早期深度测试
- **无排序**：用于性能测试

### 5. 分块光栅化

```python
def _rasterize_with_tiles(self, sorted_voxels, viewport_size):
    """使用分块进行光栅化渲染"""
    # 1. 创建图像分块
    tiles = self._create_image_tiles(width, height)
    
    # 2. 将体素分配到分块
    self._assign_voxels_to_tiles(sorted_voxels, tiles)
    
    # 3. 初始化帧缓冲
    color_buffer = torch.tensor(...).expand(height, width, -1).clone()
    depth_buffer = torch.full((height, width), far_plane)
    alpha_buffer = torch.zeros(height, width)
    
    # 4. 并行处理每个分块
    for tile in tiles:
        self._rasterize_tile(tile, color_buffer, depth_buffer, alpha_buffer)
```

**分块策略**：
- 固定大小分块
- 重叠分块处理边界
- 自适应分块（根据体素密度）
- 支持并行处理

## 性能优化

### 1. 内存管理

- **分块缓存**：每个分块独立的内存空间
- **体素索引**：使用 morton 码进行空间索引
- **延迟加载**：按需加载体素数据

### 2. 并行处理

- **分块并行**：多个分块可以并行处理
- **体素并行**：分块内的体素并行光栅化
- **像素并行**：像素级并行着色

### 3. 缓存优化

- **空间局部性**：分块设计提高缓存命中率
- **时间局部性**：重用投影和排序结果
- **数据局部性**：体素数据按分块组织

## 性能监控

### 1. 统计指标

```python
render_stats = {
    "total_voxels": 0,           # 总体素数
    "visible_voxels": 0,         # 可见体素数
    "culled_voxels": 0,          # 剔除体素数
    "render_time_ms": 0.0,       # 总渲染时间
    "projection_time_ms": 0.0,   # 投影时间
    "culling_time_ms": 0.0,      # 剔除时间
    "sorting_time_ms": 0.0,      # 排序时间
    "rasterization_time_ms": 0.0, # 光栅化时间
    "tile_count": 0,             # 分块数量
    "voxels_per_tile": [],       # 每分块体素数
}
```

### 2. 性能分析

- **瓶颈识别**：分析各阶段耗时
- **优化建议**：基于统计数据的调优建议
- **负载均衡**：监控分块间的负载分布

## 使用示例

### 1. 基本使用

```python
from nerfs.svraster.renderer import (
    SVRasterRenderer, SVRasterRendererConfig,
    VoxelRasterizerConfig, TileConfig
)

# 创建配置
rasterizer_config = VoxelRasterizerConfig()
renderer_config = SVRasterRendererConfig(
    image_width=800,
    image_height=600,
    tile_config=TileConfig(tile_size=64, overlap=8),
    log_render_stats=True
)

# 创建渲染器
renderer = SVRasterRenderer(model, rasterizer, renderer_config)

# 渲染
result = renderer.render_image(camera_pose, intrinsics)
```

### 2. 高级配置

```python
# 自定义分块配置
tile_config = TileConfig(
    tile_size=128,
    overlap=16,
    use_adaptive_tiling=True,
    min_tile_size=64,
    max_tile_size=256
)

# 自定义视锥剔除
frustum_config = FrustumCullingConfig(
    enable_frustum_culling=True,
    culling_margin=0.2,
    use_octree_culling=True
)

# 自定义深度排序
depth_config = DepthSortingConfig(
    enable_depth_sorting=True,
    sort_method="back_to_front",
    use_bucket_sort=True
)

renderer_config = SVRasterRendererConfig(
    tile_config=tile_config,
    frustum_config=frustum_config,
    depth_config=depth_config
)
```

## 扩展性

### 1. 未来扩展

- **八叉树加速**：实现空间层次结构
- **动态 LOD**：根据距离调整体素精度
- **多级缓存**：GPU 内存和显存的多级缓存
- **异步渲染**：支持异步渲染管线

### 2. 插件系统

- **自定义分块策略**：支持用户定义的分块算法
- **自定义剔除算法**：支持用户定义的剔除策略
- **自定义排序算法**：支持用户定义的排序方法

## 总结

基于投影的光栅化设计为 SVRaster 提供了：

1. **高效的推理渲染**：避免体积渲染的计算开销
2. **GPU 友好的架构**：分块设计充分利用 GPU 并行性
3. **灵活的配置系统**：支持多种优化策略的组合
4. **详细的性能监控**：提供全面的性能分析工具
5. **良好的扩展性**：为未来优化预留接口

这个设计完全符合 SVRaster 论文的核心思想，为稀疏体素的快速渲染提供了高效的解决方案。 