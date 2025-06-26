#!/usr/bin/env python3
"""
快速测试脚本
验证采样器和训练器的基本功能
"""

import os
import sys
import numpy as np
import torch

# Add the src directory to Python path
current_dir = os.path.dirname(__file__)
src_dir = os.path.join(current_dir, '..', '..', 'src')
sys.path.insert(0, src_dir)

# Add the root directory to Python path for importing sampler and neural_sdf
root_dir = os.path.join(current_dir, '..', '..')
sys.path.insert(0, root_dir)

try:
    from sampler import VoxelSampler
    from neural_sdf import MLP, NeuralSDFTrainer, VoxelDataset
except ImportError as e:
    print(f"Warning: Could not import required modules: {e}")
    print("This test requires sampler.py and neural_sdf.py to be available")

from torch.utils.data import DataLoader

def test_sampler():
    """测试采样器"""
    print("=== 测试采样器 ===")
    
    # 检查是否有tiles数据
    if not os.path.exists("tiles"):
        print("没有找到tiles目录，跳过采样器测试")
        return False
    
    try:
        # 创建采样器
        sampler = VoxelSampler(tiles_dir="tiles", voxel_size=1.0)
        
        # 检查是否有tile数据
        if not sampler.tiles_data:
            print("没有找到tile数据，跳过采样器测试")
            return False
        
        # 对第一个tile进行采样
        tile_keys = list(sampler.tiles_data.keys())
        if tile_keys:
            tile_x, tile_y = tile_keys[0]
            print(f"测试tile ({tile_x}, {tile_y})")
            
            # 分层采样
            samples = sampler.sample_stratified(tile_x, tile_y, n_samples=1000)
            print(f"采样成功: {samples['coordinates'].shape}")
            print(f"标签分布: {np.bincount(samples['labels'].astype(int))}")
            
            return True
        else:
            print("没有可用的tile数据")
            return False
            
    except Exception as e:
        print(f"采样器测试失败: {e}")
        return False

def test_neural_network():
    """测试神经网络"""
    print("=== 测试神经网络 ===")
    
    try:
        # 创建简单的测试数据
        n_samples = 1000
        coords = np.random.rand(n_samples, 3) * 100  # 0-100范围的坐标
        labels = np.random.randint(0, 2, n_samples).astype(np.float32)  # 0或1的标签
        
        # 创建数据集
        dataset = VoxelDataset(coords, labels=labels, task_type='occupancy')
        dataloader = DataLoader(dataset, batch_size=64, shuffle=True)
        
        # 创建模型
        model = MLP(
            input_dim=3, hidden_dims=[64, 128, 64], output_dim=1, activation='relu'
        )
        
        # 创建训练器
        trainer = NeuralSDFTrainer(
            model=model, learning_rate=1e-3, weight_decay=1e-5
        )
        
        # 训练几个epoch
        print("开始训练测试...")
        trainer.train(
            train_dataloader=dataloader, val_dataloader=dataloader, # 用同一个dataloader作为验证
            num_epochs=5, save_path='test_model.pth'
        )
        
        # 测试预测
        test_coords = np.array([[50, 50, 25], [75, 75, 35]])
        predictions = trainer.predict(test_coords)
        print(f"预测结果: {predictions}")
        
        # 清理测试文件
        if os.path.exists('test_model.pth'):
            os.remove('test_model.pth')
        
        return True
        
    except Exception as e:
        print(f"神经网络测试失败: {e}")
        return False

def test_sdf_training():
    """测试SDF训练"""
    print("=== 测试SDF训练 ===")
    
    try:
        # 创建SDF测试数据
        n_samples = 1000
        coords = np.random.rand(n_samples, 3) * 100
        sdf_values = np.random.randn(n_samples).astype(np.float32)  # 正负SDF值
        
        # 创建数据集
        dataset = VoxelDataset(coords, sdf_values=sdf_values, task_type='sdf')
        dataloader = DataLoader(dataset, batch_size=64, shuffle=True)
        
        # 创建SDF模型
        model = MLP(
            input_dim=3, hidden_dims=[128, 256, 128], output_dim=1, activation='relu'
        )
        
        # 创建训练器
        trainer = NeuralSDFTrainer(
            model=model, learning_rate=1e-3, weight_decay=1e-5
        )
        
        # 训练几个epoch
        print("开始SDF训练测试...")
        trainer.train(
            train_dataloader=dataloader, val_dataloader=dataloader, num_epochs=5, save_path='test_sdf_model.pth'
        )
        
        # 测试预测
        test_coords = np.array([[50, 50, 25], [75, 75, 35]])
        predictions = trainer.predict(test_coords)
        print(f"SDF预测结果: {predictions}")
        
        # 清理测试文件
        if os.path.exists('test_sdf_model.pth'):
            os.remove('test_sdf_model.pth')
        
        return True
        
    except Exception as e:
        print(f"SDF训练测试失败: {e}")
        return False

def main():
    """主测试函数"""
    print("开始快速测试...")
    
    # 测试采样器
    sampler_ok = test_sampler()
    
    # 测试神经网络
    nn_ok = test_neural_network()
    
    # 测试SDF训练
    sdf_ok = test_sdf_training()
    
    # 总结
    print("\n=== 测试总结 ===")
    print(f"采样器: {'✓' if sampler_ok else '✗'}")
    print(f"神经网络: {'✓' if nn_ok else '✗'}")
    print(f"SDF训练: {'✓' if sdf_ok else '✗'}")
    
    if sampler_ok and nn_ok and sdf_ok:
        print("\n所有测试通过！系统可以正常使用。")
    else:
        print("\n部分测试失败，请检查相关模块。")

if __name__ == "__main__":
    main() 