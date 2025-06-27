from typing import Any, Optional
"""
GFV Core Module - 核心组件

This module contains the core components of the GFV library including:
- GlobalHashConfig: Configuration class
- MultiResolutionHashEncoding: Multi-resolution hash encoding
- GlobalFeatureDatabase: Database for global features
- GlobalFeatureLibrary: Main library interface

基于Instant Neural Graphics Primitives的全球特征向量库

核心特性:
- 多分辨率哈希编码
- 全球地理坐标支持
- 分层特征表示
- 高效查询和更新
- 分布式存储支持
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import json
import os
import hashlib
import pickle
import matplotlib.pyplot as plt
from tqdm import tqdm
import logging
from dataclasses import dataclass
from collections import defaultdict
import sqlite3
from scipy.spatial import cKDTree
import mercantile
import pyproj
from torch.utils.data import DataLoader
from torch.optim import Adam

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class GlobalHashConfig:
    """全球哈希编码配置"""
    # 哈希表参数
    num_levels: int = 16                    # 哈希表层数
    max_hash: int = 2**14                   # 最大哈希值
    base_resolution: int = 16               # 基础分辨率
    finest_resolution: int = 512            # 最细分辨率
    feature_dim: int = 2                    # 每层特征维度
    total_feature_dim: int = 32             # 总特征维度
    
    # 地理参数
    global_bounds: tuple[float, float, float, float] = (-180, -90, 180, 90)  # 全球边界
    tile_size: int = 256                    # 瓦片大小
    max_zoom: int = 18                      # 最大缩放级别
    
    # 存储参数
    db_path: str = "global_features.db"     # 数据库路径
    cache_size: int = 10000                 # 缓存大小
    
    def __post_init__(self):
        self.total_feature_dim = self.num_levels * self.feature_dim

class MultiResolutionHashEncoding(nn.Module):
    """多分辨率哈希编码"""
    
    def __init__(self, config: GlobalHashConfig):
        super(MultiResolutionHashEncoding, self).__init__()
        self.config = config
        
        # 创建哈希表
        self.hash_tables = nn.ModuleList()
        for i in range(config.num_levels):
            # 计算当前层的分辨率
            resolution = config.base_resolution * (2 ** i)
            resolution = min(resolution, config.finest_resolution)
            
            # 创建哈希表
            hash_table = nn.Embedding(config.max_hash, config.feature_dim)
            nn.init.uniform_(hash_table.weight, -0.0001, 0.0001)
            self.hash_tables.append(hash_table)
        
        # 创建输出网络
        self.output_network = nn.Sequential(
            nn.Linear(
                config.total_feature_dim,
                256,
            )
        )
    
    def spatial_hash(self, coords: torch.Tensor, level: int) -> torch.Tensor:
        """空间哈希函数"""
        # 计算当前层的分辨率
        resolution = self.config.base_resolution * (2 ** level)
        resolution = min(resolution, self.config.finest_resolution)
        
        # 归一化坐标到[0, 1]
        coords_normalized = (coords + 1.0) / 2.0
        
        # 量化到网格
        coords_quantized = torch.floor(coords_normalized * resolution).long()
        
        # 计算哈希值
        hash_values = (
            coords_quantized[..., 0] * 73856093 +
            coords_quantized[..., 1] * 19349663 +
            coords_quantized[..., 2] * 83492791
        ) % self.config.max_hash
        
        return hash_values
    
    def forward(self, coords: torch.Tensor) -> torch.Tensor:
        """前向传播"""
        batch_size = coords.shape[0]
        features = []
        
        # 对每一层进行哈希编码
        for level, hash_table in enumerate(self.hash_tables):
            hash_indices = self.spatial_hash(coords, level)
            level_features = hash_table(hash_indices)
            features.append(level_features)
        
        # 连接所有层的特征
        combined_features = torch.cat(features, dim=-1)
        
        # 通过输出网络
        output = self.output_network(combined_features)
        
        return output

class GlobalFeatureDatabase:
    """全球特征数据库"""
    
    def __init__(self, config: GlobalHashConfig):
        self.config = config
        self.db_path = config.db_path
        self.cache = {}
        self.cache_size = config.cache_size
        
        # 初始化数据库
        self._init_database()
        
        # 初始化哈希编码器
        self.hash_encoder = MultiResolutionHashEncoding(config)
        
        # 初始化投影器
        self.proj = pyproj.Proj(proj='merc')
    
    def _init_database(self):
        """初始化数据库"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # 创建特征表
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS features (
                tile_id TEXT PRIMARY KEY, zoom_level INTEGER, x INTEGER, y INTEGER, features BLOB, metadata TEXT, created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # 创建索引
        cursor.execute('''
            CREATE INDEX IF NOT EXISTS idx_tile_coords 
            ON features(zoom_level, x, y)
        ''')
        
        conn.commit()
        conn.close()
        logger.info(f"数据库初始化完成: {self.db_path}")
    
    def lat_lon_to_tile(self, lat: float, lon: float, zoom: int) -> tuple[int, int]:
        """将经纬度转换为瓦片坐标"""
        tile = mercantile.tile(lon, lat, zoom)
        return tile.x, tile.y
    
    def tile_to_lat_lon_bounds(
        self,
        x: int,
        y: int,
        zoom: int,
    )
        """将瓦片坐标转换为经纬度边界"""
        bounds = mercantile.bounds(x, y, zoom)
        return bounds.west, bounds.south, bounds.east, bounds.north
    
    def get_tile_id(self, x: int, y: int, zoom: int) -> str:
        """生成瓦片ID"""
        return f"{zoom}_{x}_{y}"
    
    def store_features(
        self,
        x: int,
        y: int,
        zoom: int,
        features: np.ndarray,
        metadata: dict = None,
    )
        """存储特征"""
        tile_id = self.get_tile_id(x, y, zoom)
        
        # 序列化特征
        features_blob = pickle.dumps(features)
        metadata_json = json.dumps(metadata or {})
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT OR REPLACE INTO features (tile_id, zoom_level, x, y, features, metadata)
            VALUES (?, ?, ?, ?, ?, ?)
        ''', (tile_id, zoom, x, y, features_blob, metadata_json))
        
        conn.commit()
        conn.close()
        
        # 更新缓存
        self.cache[tile_id] = features
        
        # 清理缓存
        if len(self.cache) > self.cache_size:
            oldest_key = next(iter(self.cache))
            del self.cache[oldest_key]
        
        logger.debug(f"特征已存储: {tile_id}")
    
    def load_features(self, x: int, y: int, zoom: int) -> Optional[np.ndarray]:
        """加载特征"""
        tile_id = self.get_tile_id(x, y, zoom)
        
        # 检查缓存
        if tile_id in self.cache:
            return self.cache[tile_id]
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT features FROM features 
            WHERE tile_id = ?
        ''', (tile_id, ))
        
        result = cursor.fetchone()
        conn.close()
        
        if result:
            features = pickle.loads(result[0])
            self.cache[tile_id] = features
            return features
        
        return None
    
    def generate_tile_features(self, x: int, y: int, zoom: int) -> np.ndarray:
        """生成瓦片特征"""
        # 获取瓦片边界
        bounds = self.tile_to_lat_lon_bounds(x, y, zoom)
        west, south, east, north = bounds
        
        # 生成网格点
        num_points = self.config.tile_size ** 2
        lats = np.linspace(south, north, int(np.sqrt(num_points)))
        lons = np.linspace(west, east, int(np.sqrt(num_points)))
        
        # 转换为投影坐标
        coords = []
        for lat in lats:
            for lon in lons:
                x_proj, y_proj = self.proj(lon, lat)
                coords.append([x_proj, y_proj, 0])  # 添加高度维度
        
        coords = np.array(coords)
        
        # 归一化坐标
        coords_normalized = (coords - coords.min(axis=0)) / (coords.max(axis=0) - coords.min(axis=0)) * 2 - 1
        
        # 使用哈希编码器生成特征
        with torch.no_grad():
            coords_tensor = torch.FloatTensor(coords_normalized)
            features = self.hash_encoder(coords_tensor).numpy()
        
        return features
    
    def query_features(self, lat: float, lon: float, zoom: int) -> Optional[np.ndarray]:
        """查询特征"""
        x, y = self.lat_lon_to_tile(lat, lon, zoom)
        
        # 尝试加载现有特征
        features = self.load_features(x, y, zoom)
        
        if features is None:
            # 生成新特征
            features = self.generate_tile_features(x, y, zoom)
            
            # 存储特征
            metadata = {
                'lat': lat, 'lon': lon, 'zoom': zoom, 'generated': True
            }
            self.store_features(x, y, zoom, features, metadata)
        
        return features
    
    def batch_query_features(
        self,
        coords: list[tuple[float,
        float]],
        zoom: int,
    )
        """批量查询特征"""
        features_list = []
        
        for lat, lon in coords:
            features = self.query_features(lat, lon, zoom)
            features_list.append(features)
        
        return features_list
    
    def get_database_stats(self) -> dict[str, Any]:
        """获取数据库统计信息"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # 总瓦片数
        cursor.execute('SELECT COUNT(*) FROM features')
        total_tiles = cursor.fetchone()[0]
        
        # 按缩放级别统计
        cursor.execute('''
            SELECT zoom_level, COUNT(*) 
            FROM features 
            GROUP BY zoom_level 
            ORDER BY zoom_level
        ''')
        zoom_stats = dict(cursor.fetchall())
        
        # 数据库大小
        cursor.execute('SELECT SUM(LENGTH(features)) FROM features')
        total_size = cursor.fetchone()[0] or 0
        
        conn.close()
        
        return {
            'total_tiles': total_tiles, 'zoom_levels': zoom_stats, 'total_size_mb': total_size / (
                1024 * 1024,
            )
        }

class GlobalFeatureLibrary:
    """全球特征向量库主类"""
    
    def __init__(self, config: GlobalHashConfig):
        self.config = config
        self.database = GlobalFeatureDatabase(config)
        
        # 训练状态
        self.is_trained = False
        self.training_history = []
    
    def train_on_global_data(
        self,
        training_data: list[tuple[float,
        float,
        np.ndarray]],
        num_epochs: int = 100,
        learning_rate: float = 1e-3,
    )
        """在全局数据上训练"""
        logger.info("开始训练全球特征库...")
        
        # 准备训练数据
        optimizer = Adam(self.database.hash_encoder.parameters(), lr=learning_rate)
        criterion = nn.MSELoss()
        
        for epoch in range(num_epochs):
            total_loss = 0
            
            for lat, lon, target_features in training_data:
                # 获取当前特征
                current_features = self.database.query_features(lat, lon, 10)  # 使用中等缩放级别
                
                if current_features is not None:
                    # 计算损失
                    current_tensor = torch.FloatTensor(current_features)
                    target_tensor = torch.FloatTensor(target_features)
                    
                    loss = criterion(current_tensor, target_tensor)
                    
                    # 反向传播
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    
                    total_loss += loss.item()
            
            avg_loss = total_loss / len(training_data)
            self.training_history.append(avg_loss)
            
            if epoch % 10 == 0:
                logger.info(f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.6f}")
        
        self.is_trained = True
        logger.info("训练完成")
    
    def get_feature_vector(self, lat: float, lon: float, zoom: int = 10) -> np.ndarray:
        """获取特征向量"""
        features = self.database.query_features(lat, lon, zoom)
        if features is not None:
            # 返回平均特征向量
            return np.mean(features, axis=0)
        return None
    
    def get_region_features(
        self,
        bounds: tuple[float,
        float,
        float,
        float],
        zoom: int = 10,
    )
        """获取区域特征"""
        west, south, east, north = bounds
        
        # 计算覆盖的瓦片
        tiles = mercantile.tiles(west, south, east, north, zoom)
        
        region_features = {}
        for tile in tiles:
            features = self.database.query_features(
                mercantile.lnglat(tile.x, tile.y)[1], # lat
                mercantile.lnglat(tile.x, tile.y)[0], # lon
                zoom
            )
            if features is not None:
                tile_id = self.database.get_tile_id(tile.x, tile.y, zoom)
                region_features[tile_id] = features
        
        return region_features
    
    def save_model(self, path: str):
        """保存模型"""
        torch.save({
            'hash_encoder_state_dict': self.database.hash_encoder.state_dict(
            )
        }, path)
        logger.info(f"模型已保存到: {path}")
    
    def load_model(self, path: str):
        """加载模型"""
        checkpoint = torch.load(path)
        self.database.hash_encoder.load_state_dict(checkpoint['hash_encoder_state_dict'])
        self.training_history = checkpoint.get('training_history', [])
        self.is_trained = checkpoint.get('is_trained', False)
        logger.info(f"模型已从 {path} 加载")
    
    def visualize_coverage(self, save_path: str = None):
        """可视化覆盖范围"""
        stats = self.database.get_database_stats()
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # 缩放级别分布
        zoom_levels = list(stats['zoom_levels'].keys())
        tile_counts = list(stats['zoom_levels'].values())
        
        ax1.bar(zoom_levels, tile_counts)
        ax1.set_xlabel('缩放级别')
        ax1.set_ylabel('瓦片数量')
        ax1.set_title('各缩放级别瓦片分布')
        
        # 训练历史
        if self.training_history:
            ax2.plot(self.training_history)
            ax2.set_xlabel('训练轮数')
            ax2.set_ylabel('损失')
            ax2.set_title('训练历史')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        else:
            plt.show()

