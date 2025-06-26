#!/usr/bin/env python3
"""
Basic GFV (Geometric Feature Vector) functionality tests.

This module tests the basic functionality of the GFV package including
core operations, data handling, and basic transformations.
"""

import sys
import os
# Add the src directory to Python path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

import torch
import numpy as np
import pytest
import tempfile
import sqlite3
from pathlib import Path
import logging

from gfv import (
    GFVCore, GFVConfig, GFVDataset, GFVTrainer, create_gfv_dataloader
)
from gfv.utils import (
    coordinate_utils, data_utils, visualization_utils
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_basic_imports():
    """测试基本导入"""
    logger.info("=== 测试基本导入 ===")
    
    try:
        from src.gfv import GlobalHashConfig, GlobalFeatureLibrary
        logger.info("✅ 核心组件导入成功")
        
        from src.gfv.dataset import GlobalFeatureDataset, SDFDataset
        logger.info("✅ 数据集组件导入成功")
        
        from src.gfv.trainer import GFVTrainer
        logger.info("✅ 训练器组件导入成功")
        
        from src.gfv.utils import calculate_distance, lat_lon_to_tile
        logger.info("✅ 工具函数导入成功")
        
        return True
    except Exception as e:
        logger.error(f"❌ 导入失败: {e}")
        return False

def test_config_creation():
    """测试配置创建"""
    logger.info("=== 测试配置创建 ===")
    
    try:
        from src.gfv import GlobalHashConfig
        
        config = GlobalHashConfig(
            num_levels=8, max_hash=2**10, feature_dim=2, db_path="test_gfv.db"
        )
        
        logger.info(f"✅ 配置创建成功: {config.num_levels} 层, {config.feature_dim} 维特征")
        return True, config
    except Exception as e:
        logger.error(f"❌ 配置创建失败: {e}")
        return False, None

def test_library_creation(config):
    """测试库创建"""
    logger.info("=== 测试库创建 ===")
    
    try:
        from src.gfv import GlobalFeatureLibrary
        
        library = GlobalFeatureLibrary(config)
        logger.info("✅ GFV库创建成功")
        return True, library
    except Exception as e:
        logger.error(f"❌ 库创建失败: {e}")
        return False, None

def test_coordinate_utils():
    """测试坐标工具函数"""
    logger.info("=== 测试坐标工具函数 ===")
    
    try:
        from src.gfv.utils import calculate_distance, lat_lon_to_tile
        
        # 测试距离计算
        beijing = (39.9042, 116.4074)
        shanghai = (31.2304, 121.4737)
        distance = calculate_distance(beijing[0], beijing[1], shanghai[0], shanghai[1])
        logger.info(f"✅ 北京到上海距离: {distance:.2f} km")
        
        # 测试坐标转换
        tile_x, tile_y = lat_lon_to_tile(39.9042, 116.4074, 10)
        logger.info(f"✅ 北京瓦片坐标 (zoom=10): ({tile_x}, {tile_y})")
        
        return True
    except Exception as e:
        logger.error(f"❌ 坐标工具测试失败: {e}")
        return False

def test_dataset_creation():
    """测试数据集创建"""
    logger.info("=== 测试数据集创建 ===")
    
    try:
        from src.gfv.dataset import GlobalFeatureDataset
        
        # 创建模拟数据
        coords = [
            (39.9042, 116.4074), # 北京
            (31.2304, 121.4737), # 上海
        ]
        features = [
            np.random.randn(16), np.random.randn(16)
        ]
        
        dataset = GlobalFeatureDataset(coords, features)
        logger.info(f"✅ 数据集创建成功: {len(dataset)} 个样本")
        
        # 测试数据访问
        sample = dataset[0]
        logger.info(f"✅ 数据访问成功: coords shape={
            sample['coords'].shape,
        }
        
        return True, dataset
    except Exception as e:
        logger.error(f"❌ 数据集创建失败: {e}")
        return False, None

def main():
    """主测试函数"""
    logger.info("🧪 开始GFV基本功能测试...")
    
    # 测试导入
    if not test_basic_imports():
        return False
    
    # 测试配置创建
    success, config = test_config_creation()
    if not success:
        return False
    
    # 测试库创建
    success, library = test_library_creation(config)
    if not success:
        return False
    
    # 测试坐标工具
    if not test_coordinate_utils():
        return False
    
    # 测试数据集
    success, dataset = test_dataset_creation()
    if not success:
        return False
    
    logger.info("\n" + "="*50)
    logger.info("🎉 所有基本功能测试通过!")
    logger.info("GFV包迁移成功，核心功能正常工作")
    logger.info("="*50)
    
    return True

if __name__ == "__main__":
    main() 