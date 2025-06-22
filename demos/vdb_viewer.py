#!/usr/bin/env python3
"""
VDB数据查看器
用于查看和分析生成的VDB数据
"""

import numpy as np
import json
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import argparse
import os
from typing import Optional, Tuple

class VDBViewer:
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
    
    def load_data(self):
        """加载数据"""
        if self.data_path.endswith('.npy'):
            self.load_numpy_data()
        elif self.data_path.endswith('.vdb'):
            self.load_vdb_data()
        else:
            raise ValueError("不支持的文件格式")
        
        # 加载元数据
        metadata_path = self.data_path.replace('.npy', '_metadata.json').replace('.vdb', '_metadata.json')
        if os.path.exists(metadata_path):
            with open(metadata_path, 'r') as f:
                self.metadata = json.load(f)
    
    def load_numpy_data(self):
        """加载numpy数据"""
        self.grid = np.load(self.data_path)
        print(f"加载numpy数据: {self.grid.shape}")
    
    def load_vdb_data(self):
        """加载VDB数据"""
        try:
            import openvdb as vdb
            vdb_grid = vdb.read(self.data_path)
            self.grid = vdb_grid.copyToArray()
            print(f"加载VDB数据: {self.grid.shape}")
        except ImportError:
            raise ImportError("需要安装openvdb库来读取VDB文件")
    
    def get_statistics(self) -> dict:
        """获取数据统计信息"""
        if self.grid is None:
            return {}
        
        stats = {
            'shape': self.grid.shape,
            'total_voxels': self.grid.size,
            'non_zero_voxels': np.count_nonzero(self.grid),
            'sparsity': 1.0 - np.count_nonzero(self.grid) / self.grid.size,
            'min_value': float(np.min(self.grid)),
            'max_value': float(np.max(self.grid)),
            'mean_value': float(np.mean(self.grid)),
            'std_value': float(np.std(self.grid))
        }
        
        return stats
    
    def plot_2d_slice(self, axis: int = 2, slice_idx: Optional[int] = None, 
                     figsize: Tuple[int, int] = (12, 8)):
        """
        绘制2D切片
        
        Args:
            axis: 切片轴 (0=x, 1=y, 2=z)
            slice_idx: 切片索引（None表示中间切片）
            figsize: 图像大小
        """
        if self.grid is None:
            print("没有数据可显示")
            return
        
        if slice_idx is None:
            slice_idx = self.grid.shape[axis] // 2
        
        # 获取切片
        if axis == 0:
            slice_data = self.grid[slice_idx, :, :]
            title = f"X切片 (x={slice_idx})"
        elif axis == 1:
            slice_data = self.grid[:, slice_idx, :]
            title = f"Y切片 (y={slice_idx})"
        else:
            slice_data = self.grid[:, :, slice_idx]
            title = f"Z切片 (z={slice_idx})"
        
        # 绘制
        plt.figure(figsize=figsize)
        plt.imshow(slice_data.T, cmap='viridis', origin='lower')
        plt.colorbar(label='密度值')
        plt.title(title)
        plt.xlabel('X' if axis != 0 else 'Y')
        plt.ylabel('Y' if axis != 1 else 'Z')
        plt.show()
    
    def plot_3d_isosurface(self, threshold: float = 0.5, figsize: Tuple[int, int] = (12, 8)):
        """
        绘制3D等值面
        
        Args:
            threshold: 等值面阈值
            figsize: 图像大小
        """
        if self.grid is None:
            print("没有数据可显示")
            return
        
        # 创建坐标网格
        x, y, z = np.meshgrid(
            np.arange(self.grid.shape[0]),
            np.arange(self.grid.shape[1]),
            np.arange(self.grid.shape[2]),
            indexing='ij'
        )
        
        # 找到满足条件的点
        mask = self.grid > threshold
        x_points = x[mask]
        y_points = y[mask]
        z_points = z[mask]
        values = self.grid[mask]
        
        # 绘制3D散点图
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111, projection='3d')
        
        scatter = ax.scatter(x_points, y_points, z_points, 
                           c=values, cmap='viridis', alpha=0.6)
        
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title(f'3D等值面 (阈值={threshold})')
        
        plt.colorbar(scatter, label='密度值')
        plt.show()
    
    def plot_building_distribution(self, figsize: Tuple[int, int] = (12, 8)):
        """绘制建筑分布图"""
        if self.metadata is None or 'buildings' not in self.metadata:
            print("没有建筑元数据")
            return
        
        buildings = self.metadata['buildings']
        
        # 按类型分组
        building_types = {}
        for building in buildings:
            btype = building['type']
            if btype not in building_types:
                building_types[btype] = []
            building_types[btype].append(building)
        
        # 绘制
        plt.figure(figsize=figsize)
        
        colors = ['red', 'blue', 'green', 'orange', 'purple']
        for i, (btype, bldgs) in enumerate(building_types.items()):
            x_coords = [b['center'][0] for b in bldgs]
            y_coords = [b['center'][1] for b in bldgs]
            sizes = [b['size'][2] for b in bldgs]  # 高度作为大小
            
            plt.scatter(x_coords, y_coords, s=sizes, 
                       c=colors[i % len(colors)], alpha=0.7, label=btype)
        
        plt.xlabel('X坐标 (米)')
        plt.ylabel('Y坐标 (米)')
        plt.title('建筑分布图')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.show()
    
    def print_statistics(self):
        """打印统计信息"""
        stats = self.get_statistics()
        
        print("=== 数据统计信息 ===")
        for key, value in stats.items():
            print(f"{key}: {value}")
        
        if self.metadata:
            print("\n=== 元数据信息 ===")
            print(f"城市尺寸: {self.metadata.get('city_size', 'N/A')}")
            print(f"体素大小: {self.metadata.get('voxel_size', 'N/A')} 米")
            print(f"建筑数量: {self.metadata.get('building_count', 'N/A')}")
    
    def interactive_viewer(self):
        """交互式查看器"""
        print("=== VDB数据查看器 ===")
        print("1. 显示统计信息")
        print("2. 显示2D切片 (XY平面)")
        print("3. 显示2D切片 (XZ平面)")
        print("4. 显示2D切片 (YZ平面)")
        print("5. 显示3D等值面")
        print("6. 显示建筑分布")
        print("7. 退出")
        
        while True:
            try:
                choice = input("\n请选择操作 (1-7): ").strip()
                
                if choice == '1':
                    self.print_statistics()
                elif choice == '2':
                    self.plot_2d_slice(axis=2)
                elif choice == '3':
                    self.plot_2d_slice(axis=1)
                elif choice == '4':
                    self.plot_2d_slice(axis=0)
                elif choice == '5':
                    threshold = float(input("请输入等值面阈值 (默认0.5): ") or "0.5")
                    self.plot_3d_isosurface(threshold)
                elif choice == '6':
                    self.plot_building_distribution()
                elif choice == '7':
                    print("退出查看器")
                    break
                else:
                    print("无效选择，请重新输入")
                    
            except KeyboardInterrupt:
                print("\n退出查看器")
                break
            except Exception as e:
                print(f"错误: {e}")

def main():
    parser = argparse.ArgumentParser(description='VDB数据查看器')
    parser.add_argument('data_path', help='数据文件路径 (.npy 或 .vdb)')
    parser.add_argument('--interactive', '-i', action='store_true', 
                       help='启动交互式查看器')
    parser.add_argument('--stats', '-s', action='store_true', 
                       help='显示统计信息')
    parser.add_argument('--slice', type=int, choices=[0, 1, 2], 
                       help='显示2D切片 (0=X, 1=Y, 2=Z)')
    parser.add_argument('--slice-idx', type=int, 
                       help='切片索引')
    parser.add_argument('--isosurface', action='store_true', 
                       help='显示3D等值面')
    parser.add_argument('--threshold', type=float, default=0.5, 
                       help='等值面阈值')
    parser.add_argument('--buildings', action='store_true', 
                       help='显示建筑分布')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.data_path):
        print(f"文件不存在: {args.data_path}")
        return
    
    viewer = VDBViewer(args.data_path)
    
    if args.interactive:
        viewer.interactive_viewer()
    else:
        if args.stats:
            viewer.print_statistics()
        
        if args.slice is not None:
            viewer.plot_2d_slice(axis=args.slice, slice_idx=args.slice_idx)
        
        if args.isosurface:
            viewer.plot_3d_isosurface(threshold=args.threshold)
        
        if args.buildings:
            viewer.plot_building_distribution()
        
        # 如果没有指定任何操作，显示统计信息
        if not any([args.stats, args.slice is not None, args.isosurface, args.buildings]):
            viewer.print_statistics()

if __name__ == "__main__":
    main() 