#!/usr/bin/env python3
"""
GFV (Global Feature Vector) Library 使用演示

这个文件演示了如何在NeuroCity项目中使用新的GFV软件包。
GFV包已从原来的global_ngp.py迁移而来，提供了更好的模块化和扩展性。
"""

import sys
import os
import numpy as np
import logging

# 添加项目根目录到Python路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 导入GFV库
from src.gfv import (
    GlobalHashConfig, GlobalFeatureLibrary, GlobalFeatureDataset, MultiScaleDataset, GFVTrainer, SDFDataset
)

# 导入工具函数
from src.gfv.utils import (
    plot_coverage_map, plot_feature_distribution, visualize_global_features, lat_lon_to_tile, calculate_distance, save_feature_cache, load_feature_cache
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def demonstrate_migration_benefits() -> GlobalFeatureLibrary:
    """演示迁移到GFV包后的优势"""
    logger.info("=== GFV包迁移优势演示 ===")
    
    # 1. 更清晰的模块导入
    logger.info("✅ 模块化导入 - 从原来的单文件global_ngp.py拆分为清晰的模块结构")
    
    # 2. 配置更简洁
    config = GlobalHashConfig(
        num_levels=12, max_hash=2**13, feature_dim=4, db_path="gfv_demo.db"
    )
    logger.info("✅ 配置管理 - 使用dataclass提供更好的配置管理")
    
    # 3. 更好的类型提示
    gfv_library: GlobalFeatureLibrary = GlobalFeatureLibrary(config)
    logger.info("✅ 类型安全 - 完整的类型提示支持")
    
    # 4. 丰富的工具函数
    beijing = (39.9042, 116.4074)
    shanghai = (31.2304, 121.4737)
    distance = calculate_distance(beijing[0], beijing[1], shanghai[0], shanghai[1])
    logger.info(f"✅ 工具函数 - 北京到上海距离: {distance:.2f} km")
    
    return gfv_library

def compare_old_vs_new_usage():
    """对比旧用法和新用法"""
    logger.info("=== 新旧用法对比 ===")
    
    logger.info("旧用法 (global_ngp.py):")
    logger.info("  from global_ngp import GlobalHashConfig, GlobalFeatureLibrary")
    logger.info("  # 所有功能混在一个文件中")
    
    logger.info("\n新用法 (GFV包):")
    logger.info("  from src.gfv import GlobalHashConfig, GlobalFeatureLibrary")
    logger.info("  from src.gfv.utils import plot_coverage_map, calculate_distance")
    logger.info("  from src.gfv.trainer import GFVTrainer, GFVLightningModule")
    logger.info("  # 功能按模块分离，更易维护和扩展")

def demonstrate_new_features():
    """演示GFV包的新功能"""
    logger.info("=== GFV包新功能演示 ===")
    
    # 1. 多种数据集支持
    logger.info("1. 多种数据集类型:")
    
    # 全球特征数据集
    coords = [(39.9042, 116.4074), (31.2304, 121.4737)]
    features = [np.random.randn(32), np.random.randn(32)]
    global_dataset = GlobalFeatureDataset(coords, features)
    logger.info(f"   - GlobalFeatureDataset: {len(global_dataset)} 样本")
    
    # 多尺度数据集
    multiscale_dataset = MultiScaleDataset(coords, [8, 10, 12])
    logger.info(f"   - MultiScaleDataset: {len(multiscale_dataset)} 样本")
    
    # 2. 高级训练器
    logger.info("2. 高级训练功能:")
    config = GlobalHashConfig(num_levels=8, feature_dim=4, db_path="demo_train.db")
    gfv_library = GlobalFeatureLibrary(config)
    
    trainer = GFVTrainer(gfv_library, {
        'learning_rate': 1e-3, 'num_epochs': 10, 'batch_size': 16
    })
    logger.info("   - GFVTrainer: 传统PyTorch训练")
    logger.info("   - GFVLightningModule: PyTorch Lightning支持")
    
    # 3. 丰富的可视化工具
    logger.info("3. 可视化工具:")
    logger.info("   - plot_coverage_map: 瓦片覆盖图")
    logger.info("   - plot_feature_distribution: 特征分布图")
    logger.info("   - visualize_global_features: 全球特征可视化")
    logger.info("   - plot_interactive_map: 交互式地图")
    logger.info("   - create_dashboard: 综合仪表板")
    
    # 4. 数据处理工具
    logger.info("4. 数据处理工具:")
    logger.info("   - 坐标转换: lat_lon_to_tile, calculate_distance")
    logger.info("   - 缓存管理: save_feature_cache, load_feature_cache")
    logger.info("   - 数据验证: validate_data_consistency")
    logger.info("   - 特征插值: interpolate_features")

def demonstrate_performance_improvements():
    """演示性能改进"""
    logger.info("=== 性能改进演示 ===")
    
    import time
    
    config = GlobalHashConfig(
        num_levels=10, max_hash=2**12, feature_dim=2, db_path="perf_demo.db", cache_size=5000  # 可配置的缓存大小
    )
    
    gfv_library = GlobalFeatureLibrary(config)
    
    # 测试坐标
    test_coords = []
    for _ in range(50):
        lat = np.random.uniform(-90, 90)
        lon = np.random.uniform(-180, 180)
        test_coords.append((lat, lon))
    
    # 批量查询性能
    start_time = time.time()
    batch_features = gfv_library.database.batch_query_features(test_coords, zoom=10)
    batch_time = time.time() - start_time
    
    logger.info(f"✅ 批量查询优化: {len(test_coords)} 个查询耗时 {batch_time:.3f}s")
    logger.info(f"✅ 平均查询时间: {batch_time/len(test_coords)*1000:.2f}ms")
    
    # 缓存效果
    stats = gfv_library.database.get_database_stats()
    logger.info(f"✅ 缓存状态: {stats['cache_size']} 项缓存")

def demonstrate_extensibility():
    """演示扩展性"""
    logger.info("=== 扩展性演示 ===")
    
    logger.info("GFV包的模块化设计使得扩展变得容易:")
    logger.info("1. 核心模块 (core.py): 可以轻松添加新的编码方法")
    logger.info("2. 数据集模块 (dataset.py): 可以添加新的数据集类型")
    logger.info("3. 训练器模块 (trainer.py): 可以实现自定义训练策略")
    logger.info("4. 工具模块 (utils/): 可以添加新的工具函数")
    
    # 示例：自定义数据集类的扩展点
    class CustomGFVDataset(GlobalFeatureDataset):
        """自定义GFV数据集示例"""
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            logger.info("   - 示例: 自定义数据集类可以轻松继承和扩展")
    
    # 示例：自定义训练器的扩展点
    class CustomGFVTrainer(GFVTrainer):
        """自定义GFV训练器示例"""
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            logger.info("   - 示例: 自定义训练器可以实现特殊的训练逻辑")

def demonstrate_integration_with_neurocity():
    """演示与NeuroCity项目的集成"""
    logger.info("=== 与NeuroCity项目集成演示 ===")
    
    logger.info("GFV包与NeuroCity项目的其他组件无缝集成:")
    
    # 1. 与NeRF模型集成
    logger.info("1. 与NeRF模型集成:")
    logger.info("   - SVRaster: 可以使用GFV进行全球稀疏体素特征管理")
    logger.info("   - Grid-NeRF: 可以使用GFV进行大规模城市场景特征编码")
    logger.info("   - Instant-NGP: GFV基于相同的哈希编码原理，可以共享技术")
    
    # 2. 与数据处理流水线集成
    logger.info("2. 与数据处理集成:")
    logger.info("   - 可以处理来自OSM、卫星图像等地理数据")
    logger.info("   - 支持多种坐标系统转换")
    logger.info("   - 与现有的VDB生成器集成")
    
    # 3. 与可视化系统集成
    logger.info("3. 与可视化集成:")
    logger.info("   - 可以与NeuroCity的3D渲染系统集成")
    logger.info("   - 支持Web端交互式可视化")
    logger.info("   - 与TensorBoard集成进行训练监控")

def main():
    """主演示函数"""
    logger.info("🚀 欢迎使用GFV (Global Feature Vector) Library!")
    logger.info("这是从global_ngp.py迁移而来的全新模块化全球特征向量库")
    
    try:
        # 1. 演示迁移优势
        gfv_library = demonstrate_migration_benefits()
        
        # 2. 对比新旧用法
        compare_old_vs_new_usage()
        
        # 3. 演示新功能
        demonstrate_new_features()
        
        # 4. 演示性能改进
        demonstrate_performance_improvements()
        
        # 5. 演示扩展性
        demonstrate_extensibility()
        
        # 6. 演示项目集成
        demonstrate_integration_with_neurocity()
        
        logger.info("\n" + "="*60)
        logger.info("🎉 GFV包演示完成!")
        logger.info("📚 详细文档请查看: src/gfv/README.md")
        logger.info("💻 使用示例请查看: src/gfv/example_usage.py")
        logger.info("🔧 源码位置: src/gfv/")
        logger.info("="*60)
        
        # 显示包结构
        logger.info("\n📁 GFV包结构:")
        logger.info("src/gfv/")
        logger.info("├── __init__.py          # 包初始化")
        logger.info("├── core.py              # 核心组件")
        logger.info("├── dataset.py           # 数据集类")
        logger.info("├── trainer.py           # 训练器组件")
        logger.info("├── example_usage.py     # 使用示例")
        logger.info("├── README.md            # 详细文档")
        logger.info("└── utils/               # 工具函数包")
        logger.info("    ├── __init__.py")
        logger.info("    ├── coordinate_utils.py")
        logger.info("    ├── visualization_utils.py")
        logger.info("    └── data_utils.py")
        
    except Exception as e:
        logger.error(f"演示过程中发生错误: {e}")
        raise

if __name__ == "__main__":
    main() 