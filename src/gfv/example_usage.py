"""
GFV Usage Examples - 使用示例

This module demonstrates how to use the GFV library for global feature vector processing.
"""

import numpy as np
import logging
from typing import List, Tuple

# GFV imports
from .core import GlobalHashConfig, GlobalFeatureLibrary
from .dataset import GlobalFeatureDataset, MultiScaleDataset
from .trainer import GFVTrainer, GFVLightningModule
from .utils import (
    plot_coverage_map, 
    plot_feature_distribution,
    visualize_global_features,
    lat_lon_to_tile,
    calculate_distance
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def basic_gfv_example():
    """基础GFV使用示例"""
    logger.info("=== 基础GFV使用示例 ===")
    
    # 1. 创建配置
    config = GlobalHashConfig(
        num_levels=16,
        max_hash=2**14,
        base_resolution=16,
        finest_resolution=512,
        feature_dim=2,
        db_path="gfv_example.db"
    )
    
    # 2. 创建全球特征库
    gfv_library = GlobalFeatureLibrary(config)
    
    # 3. 查询全球城市特征
    test_coords = [
        (39.9042, 116.4074),  # 北京
        (31.2304, 121.4737),  # 上海
        (23.1291, 113.2644),  # 广州
        (40.7128, -74.0060),  # 纽约
        (51.5074, -0.1278),   # 伦敦
        (35.6762, 139.6503),  # 东京
        (48.8566, 2.3522),    # 巴黎
        (-33.8688, 151.2093), # 悉尼
    ]
    
    logger.info("查询全球城市特征...")
    for i, (lat, lon) in enumerate(test_coords):
        features = gfv_library.get_feature_vector(lat, lon, zoom=10)
        if features is not None:
            logger.info(f"城市 {i+1}: ({lat:.4f}, {lon:.4f}) -> 特征维度: {features.shape}")
        else:
            logger.warning(f"城市 {i+1}: ({lat:.4f}, {lon:.4f}) -> 未找到特征")
    
    # 4. 获取区域特征
    logger.info("获取区域特征...")
    beijing_bounds = (116.0, 39.5, 117.0, 40.5)  # 北京区域
    region_features = gfv_library.get_region_features(beijing_bounds, zoom=8)
    logger.info(f"北京区域包含 {len(region_features)} 个瓦片特征")
    
    # 5. 显示数据库统计信息
    stats = gfv_library.database.get_database_stats()
    logger.info(f"数据库统计: {stats}")
    
    return gfv_library


def training_example():
    """训练示例"""
    logger.info("=== GFV训练示例 ===")
    
    # 1. 创建配置
    config = GlobalHashConfig(
        num_levels=8,
        max_hash=2**12,
        base_resolution=16,
        finest_resolution=128,
        feature_dim=4,
        db_path="gfv_training.db"
    )
    
    # 2. 创建模拟训练数据
    num_samples = 1000
    coords = []
    features = []
    
    # 生成随机全球坐标
    for _ in range(num_samples):
        lat = np.random.uniform(-90, 90)
        lon = np.random.uniform(-180, 180)
        coords.append((lat, lon))
        
        # 生成模拟特征（基于位置的简单函数）
        feature = np.array([
            np.sin(lat * np.pi / 180),
            np.cos(lon * np.pi / 180),
            (lat + 90) / 180,
            (lon + 180) / 360
        ]) + np.random.normal(0, 0.1, 4)
        features.append(feature)
    
    # 3. 创建数据集
    dataset = GlobalFeatureDataset(coords, features)
    
    # 4. 分割训练和验证集
    train_size = int(0.8 * len(dataset))
    train_coords = coords[:train_size]
    train_features = features[:train_size]
    val_coords = coords[train_size:]
    val_features = features[train_size:]
    
    train_dataset = GlobalFeatureDataset(train_coords, train_features)
    val_dataset = GlobalFeatureDataset(val_coords, val_features)
    
    # 5. 创建训练器
    gfv_library = GlobalFeatureLibrary(config)
    trainer_config = {
        'learning_rate': 1e-3,
        'num_epochs': 50,
        'batch_size': 32
    }
    trainer = GFVTrainer(gfv_library, trainer_config)
    
    # 6. 训练模型
    results = trainer.train(train_dataset, val_dataset, "gfv_model.pth")
    logger.info(f"训练完成，最终验证损失: {results['best_val_loss']:.6f}")
    
    return gfv_library, results


def multiscale_example():
    """多尺度特征示例"""
    logger.info("=== 多尺度特征示例 ===")
    
    # 1. 创建配置
    config = GlobalHashConfig(
        num_levels=12,
        max_hash=2**13,
        feature_dim=3,
        db_path="gfv_multiscale.db"
    )
    
    # 2. 定义感兴趣的区域
    poi_coords = [
        (39.9042, 116.4074),  # 北京
        (31.2304, 121.4737),  # 上海
        (40.7128, -74.0060),  # 纽约
        (51.5074, -0.1278),   # 伦敦
    ]
    
    # 3. 创建多尺度数据集
    zoom_levels = [8, 10, 12, 14]
    multiscale_dataset = MultiScaleDataset(poi_coords, zoom_levels)
    
    logger.info(f"多尺度数据集包含 {len(multiscale_dataset)} 个样本")
    
    # 4. 分析不同尺度的特征
    gfv_library = GlobalFeatureLibrary(config)
    
    for coord in poi_coords:
        lat, lon = coord
        logger.info(f"\n位置: ({lat:.4f}, {lon:.4f})")
        
        for zoom in zoom_levels:
            features = gfv_library.get_feature_vector(lat, lon, zoom)
            tile_x, tile_y = lat_lon_to_tile(lat, lon, zoom)
            
            if features is not None:
                logger.info(f"  缩放级别 {zoom}: 瓦片({tile_x}, {tile_y}), 特征均值: {np.mean(features):.4f}")
    
    return multiscale_dataset


def visualization_example():
    """可视化示例"""
    logger.info("=== 可视化示例 ===")
    
    # 1. 创建示例数据
    coords = [
        (39.9042, 116.4074),  # 北京
        (31.2304, 121.4737),  # 上海
        (23.1291, 113.2644),  # 广州
        (40.7128, -74.0060),  # 纽约
        (51.5074, -0.1278),   # 伦敦
        (35.6762, 139.6503),  # 东京
        (48.8566, 2.3522),    # 巴黎
        (-33.8688, 151.2093), # 悉尼
    ]
    
    # 2. 生成模拟特征
    features = []
    for lat, lon in coords:
        # 基于纬度的简单特征
        feature = np.array([
            abs(lat) / 90,  # 纬度特征
            (lon + 180) / 360,  # 经度特征
            np.random.random(),  # 随机特征
        ])
        features.append(feature)
    
    # 3. 创建特征分布图
    all_features = np.array(features)
    plot_feature_distribution(all_features, 
                            title="全球城市特征分布", 
                            save_path="gfv_feature_distribution.png")
    
    # 4. 创建全球特征可视化
    visualize_global_features(coords, features, 
                            feature_dim=0,
                            title="全球城市纬度特征",
                            save_path="gfv_global_features.png")
    
    logger.info("可视化图表已保存")


def performance_analysis_example():
    """性能分析示例"""
    logger.info("=== 性能分析示例 ===")
    
    import time
    
    # 1. 创建配置
    config = GlobalHashConfig(
        num_levels=16,
        max_hash=2**14,
        feature_dim=2,
        db_path="gfv_performance.db"
    )
    
    gfv_library = GlobalFeatureLibrary(config)
    
    # 2. 性能测试
    test_coords = []
    for _ in range(100):
        lat = np.random.uniform(-90, 90)
        lon = np.random.uniform(-180, 180)
        test_coords.append((lat, lon))
    
    # 3. 单次查询性能
    start_time = time.time()
    for lat, lon in test_coords:
        features = gfv_library.get_feature_vector(lat, lon, zoom=10)
    single_query_time = time.time() - start_time
    
    logger.info(f"单次查询平均时间: {single_query_time/len(test_coords)*1000:.2f} ms")
    
    # 4. 批量查询性能
    start_time = time.time()
    batch_features = gfv_library.database.batch_query_features(test_coords, zoom=10)
    batch_query_time = time.time() - start_time
    
    logger.info(f"批量查询总时间: {batch_query_time:.2f} s")
    logger.info(f"批量查询平均时间: {batch_query_time/len(test_coords)*1000:.2f} ms")
    
    # 5. 距离计算示例
    beijing = (39.9042, 116.4074)
    shanghai = (31.2304, 121.4737)
    distance = calculate_distance(beijing[0], beijing[1], shanghai[0], shanghai[1])
    logger.info(f"北京到上海距离: {distance:.2f} km")
    
    return {
        'single_query_time': single_query_time,
        'batch_query_time': batch_query_time,
        'distance_km': distance
    }


def main():
    """主函数 - 运行所有示例"""
    logger.info("开始GFV库使用示例演示...")
    
    try:
        # 1. 基础使用示例
        gfv_library = basic_gfv_example()
        
        # 2. 训练示例
        trained_library, training_results = training_example()
        
        # 3. 多尺度示例
        multiscale_dataset = multiscale_example()
        
        # 4. 可视化示例
        visualization_example()
        
        # 5. 性能分析示例
        performance_results = performance_analysis_example()
        
        logger.info("=== 示例演示完成 ===")
        logger.info(f"性能结果: {performance_results}")
        
    except Exception as e:
        logger.error(f"示例运行出错: {e}")
        raise


if __name__ == "__main__":
    main() 