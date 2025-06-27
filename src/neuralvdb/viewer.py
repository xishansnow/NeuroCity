from typing import Any, Optional
"""
Visualization Module for NeuralVDB

This module contains visualization tools for viewing and analyzing
VDB data, including 2D/3D viewers and plotting functions.
"""

import numpy as np
import json
import os
import logging

logger = logging.getLogger(__name__)

class VDBViewer:
    """VDB数据查看器"""
    
    def __init__(self, data_path: str):
        """
        初始化VDB查看器
        
        Args:
            data_path: 数据文件路径（.npy或.vdb）
        """
        self.data_path = data_path
        self.grid = None
        self.metadata = None
        self.load_data()
    
    def load_data(self) -> None:
        """加载数据"""
        if self.data_path.endswith('.npy'):
            self.load_numpy_data()
        elif self.data_path.endswith('.vdb'):
            self.load_vdb_data()
        else:
            raise ValueError(f"不支持的文件格式: {self.data_path}")
        
        # 加载元数据
        metadata_path = self.data_path.replace(
            '.npy',
            '_metadata.json',
        )
        if os.path.exists(metadata_path):
            with open(metadata_path, 'r') as f:
                self.metadata = json.load(f)
        
        logger.info(f"VDB数据加载完成: {self.grid.shape if self.grid is not None else 'None'}")
    
    def load_numpy_data(self) -> None:
        """加载numpy数据"""
        self.grid = np.load(self.data_path)
        logger.info(f"加载numpy数据: {self.grid.shape}")
    
    def load_vdb_data(self) -> None:
        """加载VDB数据"""
        try:
            import openvdb as vdb
            vdb_grid = vdb.read(self.data_path)
            self.grid = vdb_grid.copyToArray()
            logger.info(f"加载VDB数据: {self.grid.shape}")
        except ImportError:
            raise ImportError("需要安装openvdb库来读取VDB文件")
    
    def get_statistics(self) -> dict[str, Any]:
        """获取数据统计信息"""
        if self.grid is None:
            return {}
        
        stats = {
            'shape': self.grid.shape, 'total_voxels': self.grid.size, 'non_zero_voxels': np.count_nonzero(
                self.grid,
            )
        }
        
        return stats
    
    def plot_2d_slice(
        self,
        axis: int = 2,
        slice_idx: Optional[int] = None,
        figsize: tuple[int, int] = (10, 10),
    ) -> None:
        """
        绘制2D切片
        
        Args:
            axis: 切片轴 (0=x, 1=y, 2=z)
            slice_idx: 切片索引（None表示中间切片）
            figsize: 图像大小
            save_path: 保存路径
            show_plot: 是否显示图形
        """
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            logger.warning("matplotlib未安装，无法绘制")
            return
        
        if self.grid is None:
            logger.warning("没有数据可显示")
            return
        
        if slice_idx is None:
            slice_idx = self.grid.shape[axis] // 2
        
        # 获取切片
        if axis == 0:
            slice_data = self.grid[slice_idx, :, :]
            title = f"X切片 (x={slice_idx})"
            xlabel, ylabel = 'Y', 'Z'
        elif axis == 1:
            slice_data = self.grid[:, slice_idx, :]
            title = f"Y切片 (y={slice_idx})"
            xlabel, ylabel = 'X', 'Z'
        else:
            slice_data = self.grid[:, :, slice_idx]
            title = f"Z切片 (z={slice_idx})"
            xlabel, ylabel = 'X', 'Y'
        
        # 绘制
        plt.figure(figsize=figsize)
        im = plt.imshow(
            slice_data.T,
            cmap='viridis',
            origin='lower',
            vmin=self.grid.min,
        )
        plt.colorbar(im, label='密度值')
        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"2D切片已保存到: {save_path}")
        
        if show_plot:
            plt.show()
        else:
            plt.close()
    
    def plot_3d_isosurface(
        self,
        threshold: float = 0.5,
        figsize: tuple[int,
        int] =,
    )
        """
        绘制3D等值面
        
        Args:
            threshold: 等值面阈值
            figsize: 图像大小
            max_points: 最大显示点数
            save_path: 保存路径
            show_plot: 是否显示图形
        """
        try:
            import matplotlib.pyplot as plt
            from mpl_toolkits.mplot3d import Axes3D
        except ImportError:
            logger.warning("matplotlib未安装，无法绘制")
            return
        
        if self.grid is None:
            logger.warning("没有数据可显示")
            return
        
        # 找到满足条件的点
        mask = self.grid > threshold
        indices = np.where(mask)
        
        if len(indices[0]) == 0:
            logger.warning(f"没有找到阈值 {threshold} 以上的点")
            return
        
        x_points = indices[0]
        y_points = indices[1]
        z_points = indices[2]
        values = self.grid[mask]
        
        # 随机采样以减少绘制时间
        if len(x_points) > max_points:
            sample_indices = np.random.choice(len(x_points), max_points, replace=False)
            x_points = x_points[sample_indices]
            y_points = y_points[sample_indices]
            z_points = z_points[sample_indices]
            values = values[sample_indices]
        
        # 绘制3D散点图
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111, projection='3d')
        
        scatter = ax.scatter(x_points, y_points, z_points, c=values, cmap='viridis', alpha=0.6, s=1)
        
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title(f'3D等值面 (阈值={threshold}, 点数={len(x_points)})')
        
        plt.colorbar(scatter, label='密度值', shrink=0.8)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"3D等值面已保存到: {save_path}")
        
        if show_plot:
            plt.show()
        else:
            plt.close()
    
    def plot_building_distribution(
        self,
        figsize: tuple[int,
        int] =,
    )
        """绘制建筑分布图"""
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            logger.warning("matplotlib未安装，无法绘制")
            return
        
        if self.metadata is None or 'buildings' not in self.metadata:
            logger.warning("没有建筑元数据")
            return
        
        buildings = self.metadata['buildings']
        
        # 按类型分组
        building_types = {}
        for building in buildings:
            btype = building.get('type', 'unknown')
            if btype not in building_types:
                building_types[btype] = []
            building_types[btype].append(building)
        
        # 绘制
        plt.figure(figsize=figsize)
        
        colors = ['red', 'blue', 'green', 'orange', 'purple', 'brown', 'pink']
        for i, (btype, bldgs) in enumerate(building_types.items()):
            x_coords = [b['center'][0] for b in bldgs]
            y_coords = [b['center'][1] for b in bldgs]
            sizes = [b['size'][2] * 2 for b in bldgs]  # 高度作为大小
            
            plt.scatter(
                x_coords,
                y_coords,
                s=sizes,
                c=colors[i % len,
            )
        
        plt.xlabel('X坐标 (米)')
        plt.ylabel('Y坐标 (米)')
        plt.title('建筑分布图')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.axis('equal')
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"建筑分布图已保存到: {save_path}")
        
        if show_plot:
            plt.show()
        else:
            plt.close()
    
    def plot_statistics_dashboard(
        self,
        figsize: tuple[int,
        int] =,
    )
        """绘制统计信息仪表板"""
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            logger.warning("matplotlib未安装，无法绘制")
            return
        
        if self.grid is None:
            logger.warning("没有数据可显示")
            return
        
        fig, axes = plt.subplots(2, 3, figsize=figsize)
        fig.suptitle('VDB数据统计仪表板', fontsize=16)
        
        # 1. 数值分布直方图
        ax1 = axes[0, 0]
        ax1.hist(self.grid.flatten(), bins=50, alpha=0.7, color='blue')
        ax1.set_xlabel('值')
        ax1.set_ylabel('频次')
        ax1.set_title('数值分布')
        ax1.grid(True, alpha=0.3)
        
        # 2. 稀疏性可视化
        ax2 = axes[0, 1]
        non_zero_ratio = np.count_nonzero(self.grid) / self.grid.size
        zero_ratio = 1 - non_zero_ratio
        
        labels = ['非零', '零']
        sizes = [non_zero_ratio, zero_ratio]
        colors = ['orange', 'lightblue']
        
        ax2.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
        ax2.set_title('稀疏性分布')
        
        # 3. 各轴投影
        ax3 = axes[0, 2]
        z_projection = np.sum(self.grid, axis=2)
        im3 = ax3.imshow(z_projection.T, cmap='viridis', origin='lower')
        ax3.set_xlabel('X')
        ax3.set_ylabel('Y')
        ax3.set_title('Z轴投影')
        plt.colorbar(im3, ax=ax3)
        
        # 4. 中间切片 - XY
        ax4 = axes[1, 0]
        mid_z = self.grid.shape[2] // 2
        slice_xy = self.grid[:, :, mid_z]
        im4 = ax4.imshow(slice_xy.T, cmap='viridis', origin='lower')
        ax4.set_xlabel('X')
        ax4.set_ylabel('Y')
        ax4.set_title(f'XY切片 (z={mid_z})')
        plt.colorbar(im4, ax=ax4)
        
        # 5. 中间切片 - XZ
        ax5 = axes[1, 1]
        mid_y = self.grid.shape[1] // 2
        slice_xz = self.grid[:, mid_y, :]
        im5 = ax5.imshow(slice_xz.T, cmap='viridis', origin='lower')
        ax5.set_xlabel('X')
        ax5.set_ylabel('Z')
        ax5.set_title(f'XZ切片 (y={mid_y})')
        plt.colorbar(im5, ax=ax5)
        
        # 6. 统计信息表
        ax6 = axes[1, 2]
        ax6.axis('off')
        
        stats = self.get_statistics()
        stats_text = []
        stats_text.append(f"形状: {stats['shape']}")
        stats_text.append(f"总体素数: {stats['total_voxels']:, }")
        stats_text.append(f"非零体素: {stats['non_zero_voxels']:, }")
        stats_text.append(f"稀疏率: {stats['sparsity']:.3f}")
        stats_text.append(f"最小值: {stats['min_value']:.4f}")
        stats_text.append(f"最大值: {stats['max_value']:.4f}")
        stats_text.append(f"平均值: {stats['mean_value']:.4f}")
        stats_text.append(f"标准差: {stats['std_value']:.4f}")
        stats_text.append(f"内存: {stats['memory_mb']:.2f} MB")
        
        ax6.text(
            0.1,
            0.9,
            '\n'.join,
        )
        ax6.set_title('统计信息')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"统计仪表板已保存到: {save_path}")
        
        if show_plot:
            plt.show()
        else:
            plt.close()
    
    def interactive_viewer(self):
        """交互式查看器（需要jupyter环境）"""
        try:
            from IPython.widgets import interact, IntSlider, Dropdown
            import matplotlib.pyplot as plt
        except ImportError:
            logger.warning("需要IPython和matplotlib来使用交互式查看器")
            return
        
        if self.grid is None:
            logger.warning("没有数据可显示")
            return
        
        def plot_slice(axis, slice_idx):
            plt.figure(figsize=(10, 8))
            
            if axis == 'X':
                slice_data = self.grid[slice_idx, :, :]
                title = f"X切片 (x={slice_idx})"
            elif axis == 'Y':
                slice_data = self.grid[:, slice_idx, :]
                title = f"Y切片 (y={slice_idx})"
            else:  # Z
                slice_data = self.grid[:, :, slice_idx]
                title = f"Z切片 (z={slice_idx})"
            
            plt.imshow(slice_data.T, cmap='viridis', origin='lower')
            plt.colorbar(label='密度值')
            plt.title(title)
            plt.show()
        
        # 创建交互控件
        axis_dropdown = Dropdown(
            options=['X', 'Y', 'Z'], value='Z', description='切片轴:'
        )
        
        slice_slider = IntSlider(
            min=0, max=max(
                self.grid.shape,
            )
        )
        
        interact(plot_slice, axis=axis_dropdown, slice_idx=slice_slider)
    
    def print_statistics(self):
        """打印统计信息"""
        stats = self.get_statistics()
        
        print("=== VDB数据统计信息 ===")
        for key, value in stats.items():
            if isinstance(value, float):
                if key in ['sparsity', 'mean_value', 'std_value']:
                    print(f"{key}: {value:.6f}")
                elif key == 'memory_mb':
                    print(f"{key}: {value:.2f}")
                else:
                    print(f"{key}: {value:.4f}")
            else:
                print(f"{key}: {value}")
        
        if self.metadata:
            print("\n=== 元数据信息 ===")
            for key, value in self.metadata.items():
                if key != 'buildings':  # 建筑信息太长，单独处理
                    print(f"{key}: {value}")
            
            if 'buildings' in self.metadata:
                print(f"buildings: {len(self.metadata['buildings'])} 个建筑")

def visualize_training_data(
    points: np.ndarray,
    occupancies: np.ndarray,
    save_path: Optional[str] = None,
    show_plot: bool = True,
)
    """
    可视化训练数据
    
    Args:
        points: 3D坐标点
        occupancies: 占用值
        save_path: 保存路径
        show_plot: 是否显示图形
    """
    try:
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D
    except ImportError:
        logger.warning("matplotlib未安装，无法可视化")
        return
    
    logger.info("可视化训练数据...")
    
    fig = plt.figure(figsize=(15, 5))
    
    # 分离占用和空闲点
    occupied_mask = occupancies > 0.5
    occupied_points = points[occupied_mask]
    empty_points = points[~occupied_mask]
    
    # 3D散点图
    ax1 = fig.add_subplot(131, projection='3d')
    if len(occupied_points) > 0:
        # 随机采样以减少绘制时间
        if len(occupied_points) > 5000:
            indices = np.random.choice(len(occupied_points), 5000, replace=False)
            sample_occupied = occupied_points[indices]
        else:
            sample_occupied = occupied_points
        
        ax1.scatter(
            sample_occupied[:,
            0],
            sample_occupied[:,
            1],
            sample_occupied[:,
            2],
            c='red',
            s=1,
            alpha=0.6,
            label='Occupied',
        )
    
    if len(empty_points) > 0:
        # 随机采样空闲点以减少绘制时间
        if len(empty_points) > 2000:
            indices = np.random.choice(len(empty_points), 2000, replace=False)
            sample_empty = empty_points[indices]
        else:
            sample_empty = empty_points
        
        ax1.scatter(
            sample_empty[:,
            0],
            sample_empty[:,
            1],
            sample_empty[:,
            2],
            c='blue',
            s=1,
            alpha=0.1,
            label='Empty',
        )
    
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.set_zlabel('Z')
    ax1.set_title('训练数据 (3D视图)')
    ax1.legend()
    
    # XY平面投影
    ax2 = fig.add_subplot(132)
    if len(occupied_points) > 0:
        ax2.scatter(
            occupied_points[:,
            0],
            occupied_points[:,
            1],
            c='red',
            s=1,
            alpha=0.6,
            label='Occupied',
        )
    if len(empty_points) > 0:
        # 采样显示
        if len(empty_points) > 5000:
            indices = np.random.choice(len(empty_points), 5000, replace=False)
            sample_empty = empty_points[indices]
        else:
            sample_empty = empty_points
        
        ax2.scatter(sample_empty[:, 0], sample_empty[:, 1], c='blue', s=1, alpha=0.1, label='Empty')
    
    ax2.set_xlabel('X')
    ax2.set_ylabel('Y')
    ax2.set_title('XY投影')
    ax2.set_aspect('equal')
    ax2.legend()
    
    # 占用率分布
    ax3 = fig.add_subplot(133)
    ax3.hist(occupancies, bins=50, alpha=0.7, color='green', edgecolor='black')
    ax3.set_xlabel('占用值')
    ax3.set_ylabel('频次')
    ax3.set_title('占用率分布')
    ax3.grid(True, alpha=0.3)
    
    # 添加统计信息
    stats_text = f"总点数: {len(points):, }\n"
    stats_text += f"占用点: {len(occupied_points):, }\n"
    stats_text += f"占用率: {len(occupied_points)/len(points):.3f}\n"
    stats_text += f"平均值: {np.mean(occupancies):.3f}\n"
    stats_text += f"标准差: {np.std(occupancies):.3f}"
    
    ax3.text(
        0.02,
        0.98,
        stats_text,
        transform=ax3.transAxes,
        verticalalignment='top',
        fontsize=9,
        bbox=dict,
    )
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"训练数据可视化已保存到: {save_path}")
    
    if show_plot:
        plt.show()
    else:
        plt.close()

def visualize_predictions(
    points: np.ndarray,
    predictions: np.ndarray,
    ground_truth: Optional[np.ndarray] = None,
    save_path: Optional[str] = None,
    show_plot: bool = True,
)
    """
    可视化预测结果
    
    Args:
        points: 3D坐标点
        predictions: 预测值
        ground_truth: 真实值（可选）
        save_path: 保存路径
        show_plot: 是否显示图形
    """
    try:
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D
    except ImportError:
        logger.warning("matplotlib未安装，无法可视化")
        return
    
    logger.info("可视化预测结果...")
    
    if ground_truth is not None:
        fig = plt.figure(figsize=(20, 5))
        n_subplots = 4
    else:
        fig = plt.figure(figsize=(15, 5))
        n_subplots = 3
    
    # 随机采样以减少绘制时间
    if len(points) > 10000:
        indices = np.random.choice(len(points), 10000, replace=False)
        sample_points = points[indices]
        sample_predictions = predictions[indices]
        if ground_truth is not None:
            sample_gt = ground_truth[indices]
    else:
        sample_points = points
        sample_predictions = predictions
        if ground_truth is not None:
            sample_gt = ground_truth
    
    # 预测值分布
    ax1 = fig.add_subplot(1, n_subplots, 1)
    ax1.hist(predictions, bins=50, alpha=0.7, color='orange', edgecolor='black')
    ax1.set_xlabel('预测占用率')
    ax1.set_ylabel('频次')
    ax1.set_title('预测值分布')
    ax1.grid(True, alpha=0.3)
    
    # 添加统计信息
    stats_text = f"平均值: {np.mean(predictions):.3f}\n"
    stats_text += f"标准差: {np.std(predictions):.3f}\n"
    stats_text += f"最小值: {np.min(predictions):.3f}\n"
    stats_text += f"最大值: {np.max(predictions):.3f}"
    
    ax1.text(
        0.02,
        0.98,
        stats_text,
        transform=ax1.transAxes,
        verticalalignment='top',
        fontsize=9,
        bbox=dict,
    )
    
    # 3D预测结果
    ax2 = fig.add_subplot(1, n_subplots, 2, projection='3d')
    
    scatter = ax2.scatter(
        sample_points[:,
        0],
        sample_points[:,
        1],
        sample_points[:,
        2],
        c=sample_predictions,
        cmap='viridis',
        s=10,
        alpha=0.7,
    )
    
    ax2.set_xlabel('X')
    ax2.set_ylabel('Y')
    ax2.set_zlabel('Z')
    ax2.set_title('预测结果 (3D)')
    
    # XY平面热力图
    ax3 = fig.add_subplot(1, n_subplots, 3)
    scatter_2d = ax3.scatter(
        sample_points[:,
        0],
        sample_points[:,
        1],
        c=sample_predictions,
        cmap='viridis',
        s=10,
        alpha=0.7,
    )
    ax3.set_xlabel('X')
    ax3.set_ylabel('Y')
    ax3.set_title('预测热力图 (XY)')
    ax3.set_aspect('equal')
    plt.colorbar(scatter_2d, ax=ax3, label='预测值')
    
    # 如果有真实值，添加误差分析
    if ground_truth is not None:
        ax4 = fig.add_subplot(1, n_subplots, 4)
        
        errors = np.abs(predictions - ground_truth)
        ax4.hist(errors, bins=50, alpha=0.7, color='red', edgecolor='black')
        ax4.set_xlabel('绝对误差')
        ax4.set_ylabel('频次')
        ax4.set_title(f'误差分布')
        ax4.grid(True, alpha=0.3)
        
        # 添加误差统计
        mae = np.mean(errors)
        rmse = np.sqrt(np.mean((predictions - ground_truth) ** 2))
        
        error_text = f"MAE: {mae:.4f}\n"
        error_text += f"RMSE: {rmse:.4f}\n"
        error_text += f"最大误差: {np.max(errors):.4f}"
        
        ax4.text(
            0.02,
            0.98,
            error_text,
            transform=ax4.transAxes,
            verticalalignment='top',
            fontsize=9,
            bbox=dict,
        )
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"预测可视化结果已保存到: {save_path}")
    
    if show_plot:
        plt.show()
    else:
        plt.close()

def compare_predictions(
    points: np.ndarray,
    predictions1: np.ndarray,
    predictions2: np.ndarray,
    labels: tuple[str,
    str] =,
)
    """
    比较两个模型的预测结果
    
    Args:
        points: 3D坐标点
        predictions1: 第一个模型的预测
        predictions2: 第二个模型的预测
        labels: 模型标签
        ground_truth: 真实值（可选）
        save_path: 保存路径
        show_plot: 是否显示图形
    """
    try:
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D
    except ImportError:
        logger.warning("matplotlib未安装，无法可视化")
        return
    
    logger.info("比较预测结果...")
    
    fig = plt.figure(figsize=(20, 10))
    
    # 随机采样
    if len(points) > 5000:
        indices = np.random.choice(len(points), 5000, replace=False)
        sample_points = points[indices]
        sample_pred1 = predictions1[indices]
        sample_pred2 = predictions2[indices]
        if ground_truth is not None:
            sample_gt = ground_truth[indices]
    else:
        sample_points = points
        sample_pred1 = predictions1
        sample_pred2 = predictions2
        if ground_truth is not None:
            sample_gt = ground_truth
    
    # 预测分布比较
    ax1 = fig.add_subplot(2, 3, 1)
    ax1.hist(predictions1, bins=50, alpha=0.7, label=labels[0], color='blue')
    ax1.hist(predictions2, bins=50, alpha=0.7, label=labels[1], color='red')
    ax1.set_xlabel('预测值')
    ax1.set_ylabel('频次')
    ax1.set_title('预测分布比较')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 散点图比较
    ax2 = fig.add_subplot(2, 3, 2)
    ax2.scatter(predictions1, predictions2, alpha=0.5, s=1)
    ax2.plot([0, 1], [0, 1], 'r--', label='y=x')
    ax2.set_xlabel(f'{labels[0]} 预测值')
    ax2.set_ylabel(f'{labels[1]} 预测值')
    ax2.set_title('预测值相关性')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 差异分布
    ax3 = fig.add_subplot(2, 3, 3)
    diff = predictions1 - predictions2
    ax3.hist(diff, bins=50, alpha=0.7, color='green')
    ax3.set_xlabel('预测差异')
    ax3.set_ylabel('频次')
    ax3.set_title(f'预测差异分布\n(平均差异: {np.mean(diff):.4f})')
    ax3.grid(True, alpha=0.3)
    
    # 3D可视化 - Model 1
    ax4 = fig.add_subplot(2, 3, 4, projection='3d')
    scatter1 = ax4.scatter(
        sample_points[:,
        0],
        sample_points[:,
        1],
        sample_points[:,
        2],
        c=sample_pred1,
        cmap='viridis',
        s=5,
        alpha=0.7,
    )
    ax4.set_title(f'{labels[0]} 预测结果')
    ax4.set_xlabel('X')
    ax4.set_ylabel('Y')
    ax4.set_zlabel('Z')
    
    # 3D可视化 - Model 2
    ax5 = fig.add_subplot(2, 3, 5, projection='3d')
    scatter2 = ax5.scatter(
        sample_points[:,
        0],
        sample_points[:,
        1],
        sample_points[:,
        2],
        c=sample_pred2,
        cmap='viridis',
        s=5,
        alpha=0.7,
    )
    ax5.set_title(f'{labels[1]} 预测结果')
    ax5.set_xlabel('X')
    ax5.set_ylabel('Y')
    ax5.set_zlabel('Z')
    
    # 误差比较（如果有真实值）
    ax6 = fig.add_subplot(2, 3, 6)
    if ground_truth is not None:
        error1 = np.abs(predictions1 - ground_truth)
        error2 = np.abs(predictions2 - ground_truth)
        
        ax6.hist(error1, bins=50, alpha=0.7, label=f'{
            labels[0],
        }
        ax6.hist(error2, bins=50, alpha=0.7, label=f'{
            labels[1],
        }
        ax6.set_xlabel('绝对误差')
        ax6.set_ylabel('频次')
        ax6.set_title('误差分布比较')
        ax6.legend()
    else:
        # 显示差异的3D可视化
        ax6.remove()
        ax6 = fig.add_subplot(2, 3, 6, projection='3d')
        scatter_diff = ax6.scatter(
            sample_points[:,
            0],
            sample_points[:,
            1],
            sample_points[:,
            2],
            c=np.abs,
        )
        ax6.set_title('预测差异 (3D)')
        ax6.set_xlabel('X')
        ax6.set_ylabel('Y')
        ax6.set_zlabel('Z')
        plt.colorbar(scatter_diff, ax=ax6, shrink=0.8, label='|差异|')
    
    ax6.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"预测比较结果已保存到: {save_path}")
    
    if show_plot:
        plt.show()
    else:
        plt.close() 