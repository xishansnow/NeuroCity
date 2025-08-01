"""
Visualization Utilities - 可视化工具

This module provides visualization utilities for GFV library.
"""

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from typing import Any
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots


def plot_coverage_map(database_stats: dict[str, Any], save_path: str | None = None) -> None:
    """
    绘制瓦片覆盖图
    
    Args:
        database_stats: 数据库统计信息
        save_path: 保存路径
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # 缩放级别分布
    zoom_levels = list(database_stats['zoom_levels'].keys())
    tile_counts = list(database_stats['zoom_levels'].values())
    
    ax1.bar(zoom_levels, tile_counts, color='skyblue', alpha=0.7)
    ax1.set_xlabel('缩放级别 (Zoom Level)')
    ax1.set_ylabel('瓦片数量 (Tile Count)')
    ax1.set_title('各缩放级别瓦片分布')
    ax1.grid(True, alpha=0.3)
    
    # 数据库大小信息
    total_tiles = database_stats['total_tiles']
    total_size_mb = database_stats['total_size_mb']
    cache_size = database_stats['cache_size']
    
    labels = ['总瓦片数', '数据库大小(MB)', '缓存大小']
    values = [total_tiles, total_size_mb, cache_size]
    
    ax2.pie([1, 1, 1], labels=labels, autopct=lambda pct: f'{
        values[int(pct/100*3)]:.1f,
    }
    ax2.set_title('数据库统计信息')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    else:
        plt.show()


def plot_feature_distribution(
    features: np.ndarray,
    title: str = "特征分布",
    save_path: str | None = None,
)
    """
    绘制特征分布图
    
    Args:
        features: 特征数组 [N, D]
        title: 图标题
        save_path: 保存路径
    """
    if features.ndim == 1:
        features = features.reshape(-1, 1)
    
    num_features = features.shape[1]
    
    if num_features == 1:
        # 单个特征的分布
        plt.figure(figsize=(10, 6))
        plt.hist(features[:, 0], bins=50, alpha=0.7, color='blue')
        plt.xlabel('特征值')
        plt.ylabel('频次')
        plt.title(title)
        plt.grid(True, alpha=0.3)
    
    elif num_features <= 16:
        # 多个特征的分布
        cols = min(4, num_features)
        rows = (num_features + cols - 1) // cols
        
        fig, axes = plt.subplots(rows, cols, figsize=(4*cols, 3*rows))
        if rows == 1 and cols == 1:
            axes = [axes]
        elif rows == 1 or cols == 1:
            axes = axes.flatten()
        else:
            axes = axes.flatten()
        
        for i in range(num_features):
            ax = axes[i] if num_features > 1 else axes
            ax.hist(features[:, i], bins=30, alpha=0.7)
            ax.set_title(f'特征 {i+1}')
            ax.grid(True, alpha=0.3)
        
        # 隐藏多余的子图
        for i in range(num_features, len(axes)):
            axes[i].set_visible(False)
    
    else:
        # 特征过多，绘制特征均值和方差
        feature_means = np.mean(features, axis=0)
        feature_stds = np.std(features, axis=0)
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
        
        ax1.plot(feature_means, 'o-', alpha=0.7)
        ax1.set_title('特征均值')
        ax1.set_xlabel('特征索引')
        ax1.set_ylabel('均值')
        ax1.grid(True, alpha=0.3)
        
        ax2.plot(feature_stds, 'o-', alpha=0.7, color='red')
        ax2.set_title('特征标准差')
        ax2.set_xlabel('特征索引')
        ax2.set_ylabel('标准差')
        ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    else:
        plt.show()


def plot_training_history(
    train_losses: list[float],
    val_losses: list[float] | None = None,
    title: str = "训练历史",
    save_path: str | None = None,
)
    """
    绘制训练历史图
    
    Args:
        train_losses: 训练损失列表
        val_losses: 验证损失列表
        title: 图标题
        save_path: 保存路径
    """
    plt.figure(figsize=(10, 6))
    
    epochs = range(1, len(train_losses) + 1)
    plt.plot(epochs, train_losses, 'b-', label='训练损失', alpha=0.7)
    
    if val_losses:
        plt.plot(epochs, val_losses, 'r-', label='验证损失', alpha=0.7)
    
    plt.xlabel('训练轮数 (Epoch)')
    plt.ylabel('损失 (Loss)')
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    else:
        plt.show()


def visualize_global_features(
    coords: list[tuple[float,
    float]],
    features: list[np.ndarray],
    feature_dim: int = 0,
    title: str = "全球特征可视化",
    save_path: str | None = None,
)
    """
    可视化全球特征分布
    
    Args:
        coords: 坐标列表 [(lat, lon), ...]
        features: 特征列表
        feature_dim: 要可视化的特征维度
        title: 图标题
        save_path: 保存路径
    """
    if not coords or not features:
        print("没有数据可视化")
        return
    
    # 提取坐标和特征值
    lats = [coord[0] for coord in coords]
    lons = [coord[1] for coord in coords]
    
    # 处理特征维度
    if isinstance(features[0], np.ndarray) and features[0].ndim > 1:
        feature_values = [feat.mean() if feat.ndim > 1 else feat[feature_dim] for feat in features]
    else:
        feature_values = [feat if np.isscalar(feat) else feat[feature_dim] for feat in features]
    
    # 创建散点图
    plt.figure(figsize=(12, 8))
    scatter = plt.scatter(lons, lats, c=feature_values, cmap='viridis', alpha=0.6, s=50)
    plt.colorbar(scatter, label=f'特征值 (维度 {feature_dim})')
    plt.xlabel('经度 (Longitude)')
    plt.ylabel('纬度 (Latitude)')
    plt.title(title)
    plt.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    else:
        plt.show()


def plot_interactive_map(
    coords: list[tuple[float,
    float]],
    features: list[np.ndarray],
    feature_dim: int = 0,
    title: str = "交互式全球特征地图",
)
    """
    创建交互式地图
    
    Args:
        coords: 坐标列表 [(lat, lon), ...]
        features: 特征列表
        feature_dim: 要可视化的特征维度
        title: 图标题
        
    Returns:
        fig: plotly图形对象
    """
    if not coords or not features:
        print("没有数据可视化")
        return go.Figure()
    
    # 提取坐标和特征值
    lats = [coord[0] for coord in coords]
    lons = [coord[1] for coord in coords]
    
    # 处理特征维度
    if isinstance(features[0], np.ndarray) and features[0].ndim > 1:
        feature_values = [feat.mean() if feat.ndim > 1 else feat[feature_dim] for feat in features]
    else:
        feature_values = [feat if np.isscalar(feat) else feat[feature_dim] for feat in features]
    
    # 创建地图
    fig = go.Figure(data=go.Scattermapbox(
        lat=lats, lon=lons, mode='markers', marker=dict(
            size=10, color=feature_values, colorscale='Viridis', showscale=True, colorbar=dict(
                title=f'特征值,
            )
        ), text=[f'特征值: {val:.3f}' for val in feature_values], hoverinfo='text'
    ))
    
    # 更新布局
    fig.update_layout(
        title=title, mapbox=dict(
            style='carto-positron', center=dict(lat=np.mean(lats), lon=np.mean(lons)), zoom=2
        ), margin=dict(l=0, r=0, t=30, b=0)
    )
    
    return fig


def plot_feature_correlation_matrix(
    features: np.ndarray,
    title: str = "特征相关性矩阵",
    save_path: str | None = None,
)
    """
    绘制特征相关性矩阵
    
    Args:
        features: 特征数组 [N, D]
        title: 图标题
        save_path: 保存路径
    """
    if features.ndim == 1:
        features = features.reshape(-1, 1)
    
    # 计算相关性矩阵
    corr_matrix = np.corrcoef(features.T)
    
    # 绘制热力图
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, fmt='.2f', square=True)
    plt.title(title)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    else:
        plt.show()


def create_dashboard(
    database_stats: dict[str,
    Any],
    training_history: dict[str,
    list[float]],
    coords: list[tuple[float,
    float]],
    features: list[np.ndarray],
)
    """
    创建仪表板
    
    Args:
        database_stats: 数据库统计信息
        training_history: 训练历史
        coords: 坐标列表
        features: 特征列表
        
    Returns:
        fig: plotly图形对象
    """
    # 创建子图布局
    fig = make_subplots(
        rows=2, cols=2, subplot_titles=(
            '瓦片分布', '训练历史', '特征分布', '全球特征地图'
        ), specs=[
            [{
                'type': 'bar',
            }
        ]
    )
    
    # 1. 瓦片分布
    zoom_levels = list(database_stats['zoom_levels'].keys())
    tile_counts = list(database_stats['zoom_levels'].values())
    fig.add_trace(
        go.Bar(x=zoom_levels, y=tile_counts, name='瓦片数量'), row=1, col=1
    )
    
    # 2. 训练历史
    epochs = list(range(1, len(training_history['train_loss']) + 1))
    fig.add_trace(
        go.Scatter(
            x=epochs,
            y=training_history['train_loss'],
            name='训练损失',
            mode='lines',
        )
    )
    if 'val_loss' in training_history:
        fig.add_trace(
            go.Scatter(
                x=epochs,
                y=training_history['val_loss'],
                name='验证损失',
                mode='lines',
            )
        )
    
    # 3. 特征分布
    if features and isinstance(features[0], np.ndarray):
        feature_values = [f.mean() for f in features]
        fig.add_trace(
            go.Histogram(x=feature_values, name='特征分布'), row=2, col=1
        )
    
    # 4. 全球特征地图
    if coords and features:
        lats = [coord[0] for coord in coords]
        lons = [coord[1] for coord in coords]
        feature_values = [f.mean() if isinstance(f, np.ndarray) else f for f in features]
        
        fig.add_trace(
            go.Scattermapbox(
                lat=lats, lon=lons, mode='markers', marker=dict(
                    size=10, color=feature_values, colorscale='Viridis', showscale=True
                ), text=[f'特征值: {val:.3f}' for val in feature_values], hoverinfo='text'
            ), row=2, col=2
        )
    
    # 更新布局
    fig.update_layout(
        height=800, showlegend=True, title_text='全球特征可视化仪表板', mapbox=dict(
            style='carto-positron', center=dict(lat=0, lon=0), zoom=1
        )
    )
    
    return fig 