#!/usr/bin/env python3
"""
使用示例脚本
展示如何使用采样器和训练器
"""

import os
import numpy as np
from sampler import VoxelSampler
from neural_sdf import MLP, NeuralSDFTrainer, load_training_data
from train_pipeline import TrainingPipeline, get_default_config

def example_sampling():
    """示例：体素采样"""
    print("=== 体素采样示例 ===")
    
    # 创建采样器
    sampler = VoxelSampler(
        tiles_dir="tiles", voxel_size=1.0, sample_ratio=0.1
    )
    
    # 对单个tile进行采样
    print("对tile (0, 0)进行分层采样...")
    samples = sampler.sample_stratified(0, 0, n_samples=5000)
    print(f"采样结果: {samples['coordinates'].shape}")
    print(f"标签分布: {np.bincount(samples['labels'].astype(int))}")
    
    # 对所有tile进行采样
    print("对所有tile进行采样...")
    all_samples = sampler.sample_all_tiles(
        sampling_method='stratified', n_samples_per_tile=5000
    )
    
    # 保存采样数据
    sampler.save_samples(all_samples, "samples")
    print("采样完成！")

def example_training():
    """示例：神经网络训练"""
    print("=== 神经网络训练示例 ===")
    
    # 加载训练数据
    print("加载训练数据...")
    train_dataloader, val_dataloader = load_training_data(
        samples_dir="samples", task_type='occupancy', train_ratio=0.8
    )
    
    # 创建模型
    print("创建神经网络模型...")
    model = MLP(
        input_dim=3, hidden_dims=[256, 512, 512, 256, 128], output_dim=1, activation='relu', dropout=0.1
    )
    
    # 创建训练器
    trainer = NeuralSDFTrainer(
        model=model, learning_rate=1e-3, weight_decay=1e-5
    )
    
    # 训练模型
    print("开始训练...")
    trainer.train(
        train_dataloader=train_dataloader, val_dataloader=val_dataloader, num_epochs=20, # 减少轮数用于演示
        save_path='model_occupancy.pth', early_stopping_patience=5
    )
    
    # 测试预测
    print("测试预测...")
    test_coords = np.array([
        [100, 100, 10], [200, 200, 20], [300, 300, 30], [400, 400, 40]
    ])
    
    predictions = trainer.predict(test_coords)
    print("预测结果:")
    for i, (coord, pred) in enumerate(zip(test_coords, predictions)):
        print(f"  坐标 {coord}: 占用概率 {pred[0]:.4f}")
    
    print("训练完成！")

def example_sdf_training():
    """示例：SDF训练"""
    print("=== SDF训练示例 ===")
    
    # 创建采样器并生成SDF数据
    sampler = VoxelSampler(tiles_dir="tiles", voxel_size=1.0)
    
    # 对单个tile进行表面采样
    print("对tile (0, 0)进行表面采样...")
    sdf_samples = sampler.sample_near_surface(0, 0, n_samples=5000)
    print(f"SDF采样结果: {sdf_samples['coordinates'].shape}")
    
    # 保存SDF采样数据
    os.makedirs("sdf_samples", exist_ok=True)
    np.save("sdf_samples/coords_0_0.npy", sdf_samples['coordinates'])
    np.save("sdf_samples/sdf_0_0.npy", sdf_samples['sdf_values'])
    
    # 创建SDF模型
    model = MLP(
        input_dim=3, hidden_dims=[512, 1024, 1024, 512, 256], output_dim=1, activation='relu'
    )
    
    # 创建SDF训练器
    trainer = NeuralSDFTrainer(
        model=model, learning_rate=1e-3, weight_decay=1e-5
    )
    
    # 创建简单的SDF数据集
    coords = sdf_samples['coordinates']
    sdf_values = sdf_samples['sdf_values']
    
    # 划分训练集和验证集
    n_samples = len(coords)
    n_train = int(n_samples * 0.8)
    
    indices = np.random.permutation(n_samples)
    train_indices = indices[:n_train]
    val_indices = indices[n_train:]
    
    from neural_sdf import VoxelDataset
    from torch.utils.data import DataLoader
    
    train_dataset = VoxelDataset(
        coords[train_indices], sdf_values=sdf_values[train_indices], task_type='sdf'
    )
    val_dataset = VoxelDataset(
        coords[val_indices], sdf_values=sdf_values[val_indices], task_type='sdf'
    )
    
    train_dataloader = DataLoader(train_dataset, batch_size=1024, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=1024, shuffle=False)
    
    # 训练SDF模型
    print("开始SDF训练...")
    trainer.train(
        train_dataloader=train_dataloader, val_dataloader=val_dataloader, num_epochs=20, save_path='model_sdf.pth', early_stopping_patience=5
    )
    
    print("SDF训练完成！")

def example_pipeline():
    """示例：完整流水线"""
    print("=== 完整训练流水线示例 ===")
    
    # 获取默认配置
    config = get_default_config()
    
    # 修改配置
    config['sampling']['n_samples_per_tile'] = 5000  # 减少采样数量
    config['training']['num_epochs'] = 20  # 减少训练轮数
    config['training']['task_type'] = 'occupancy'
    
    # 创建流水线
    pipeline = TrainingPipeline(config)
    
    # 运行完整流水线
    pipeline.run_full_pipeline()
    
    print("完整流水线执行完毕！")

def main():
    """主函数"""
    print("神经网络训练示例")
    print("请选择要运行的示例:")
    print("1. 体素采样")
    print("2. 占用网络训练")
    print("3. SDF网络训练")
    print("4. 完整流水线")
    print("5. 全部运行")
    
    choice = input("请输入选择 (1-5): ").strip()
    
    if choice == '1':
        example_sampling()
    elif choice == '2':
        example_training()
    elif choice == '3':
        example_sdf_training()
    elif choice == '4':
        example_pipeline()
    elif choice == '5':
        print("运行所有示例...")
        example_sampling()
        example_training()
        example_sdf_training()
        example_pipeline()
    else:
        print("无效选择")

if __name__ == "__main__":
    main() 