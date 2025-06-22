"""
Models Package Example Usage
展示如何使用OccupancyNet和SDFNet软件包

包含：
1. 基本模型创建和训练
2. 数据集准备和处理
3. 网格重建和可视化
4. 模型评估和性能分析
"""

import torch
import numpy as np
import os
import matplotlib.pyplot as plt
from typing import Dict, List

from occupancy_net import OccupancyNetwork, ConditionalOccupancyNetwork
from occupancy_net import OccupancyDataset, SyntheticOccupancyDataset
from occupancy_net import OccupancyTrainer, create_occupancy_dataloader

from sdf_net import SDFNetwork, LatentSDFNetwork  
from sdf_net import SDFDataset, SyntheticSDFDataset
from sdf_net import SDFTrainer, create_sdf_dataloader


def example_occupancy_network():
    """占用网络使用示例"""
    print("=" * 50)
    print("Occupancy Network Example")
    print("=" * 50)
    
    # 1. 创建模型
    print("Creating Occupancy Network...")
    model = OccupancyNetwork(
        dim_input=3,
        dim_hidden=256,
        num_layers=8,
        use_batch_norm=True,
        dropout_prob=0.1
    )
    
    print(f"Model created with {sum(p.numel() for p in model.parameters()):,} parameters")
    print(f"Model size: {model.get_model_size()}")
    
    # 2. 创建合成数据集
    print("\nCreating synthetic dataset...")
    dataset = SyntheticOccupancyDataset(
        num_samples=100,
        num_points=10000,
        shape_types=['sphere', 'cube', 'cylinder']
    )
    
    dataloader = create_occupancy_dataloader(
        dataset,
        batch_size=4,
        shuffle=True,
        num_workers=0
    )
    
    # 3. 测试前向传播
    print("\nTesting forward pass...")
    sample_batch = next(iter(dataloader))
    points = sample_batch['points']  # [B, N, 3]
    occupancy = sample_batch['occupancy']  # [B, N, 1]
    
    print(f"Input points shape: {points.shape}")
    print(f"Ground truth occupancy shape: {occupancy.shape}")
    
    # 前向传播
    with torch.no_grad():
        pred_occupancy = model(points)
        loss = model.compute_loss(pred_occupancy, occupancy)
    
    print(f"Predicted occupancy shape: {pred_occupancy.shape}")
    print(f"Loss: {loss.item():.4f}")
    
    # 4. 网格提取示例
    print("\nTesting mesh extraction...")
    try:
        mesh_data = model.extract_mesh(
            resolution=32,  # 使用较低分辨率进行快速测试
            threshold=0.5
        )
        if 'vertices' in mesh_data:
            print(f"Extracted mesh: {len(mesh_data['vertices'])} vertices, {len(mesh_data['faces'])} faces")
        else:
            print("Mesh extraction returned occupancy grid only")
    except Exception as e:
        print(f"Mesh extraction failed: {e}")
    
    return model, dataset


def example_sdf_network():
    """SDF网络使用示例"""
    print("=" * 50)
    print("SDF Network Example")
    print("=" * 50)
    
    # 1. 创建模型
    print("Creating SDF Network...")
    model = SDFNetwork(
        dim_input=3,
        dim_latent=256,
        dim_hidden=512,
        num_layers=8,
        skip_connections=[4],
        geometric_init=True,
        weight_norm=True
    )
    
    print(f"Model created with {sum(p.numel() for p in model.parameters()):,} parameters")
    print(f"Model size: {model.get_model_size()}")
    
    # 2. 创建合成数据集
    print("\nCreating synthetic SDF dataset...")
    dataset = SyntheticSDFDataset(
        num_samples=100,
        num_points=10000,
        shape_types=['sphere', 'cube', 'cylinder']
    )
    
    dataloader = create_sdf_dataloader(
        dataset,
        batch_size=4,
        shuffle=True,
        num_workers=0
    )
    
    # 3. 测试前向传播
    print("\nTesting forward pass...")
    sample_batch = next(iter(dataloader))
    points = sample_batch['points']  # [B, N, 3]
    sdf = sample_batch['sdf']  # [B, N, 1]
    
    print(f"Input points shape: {points.shape}")
    print(f"Ground truth SDF shape: {sdf.shape}")
    
    # 创建随机潜在编码
    batch_size = points.shape[0]
    latent_code = torch.randn(batch_size, 256) * 0.01
    
    # 前向传播
    with torch.no_grad():
        pred_sdf = model(points, latent_code)
        loss = model.compute_sdf_loss(pred_sdf, sdf, loss_type='l1')
    
    print(f"Predicted SDF shape: {pred_sdf.shape}")
    print(f"Loss: {loss.item():.4f}")
    
    # 4. 梯度惩罚测试
    print("\nTesting gradient penalty...")
    try:
        gradient_penalty = model.compute_gradient_penalty(
            points[:1],  # 使用单个样本
            latent_code[:1],
            lambda_gp=0.1
        )
        print(f"Gradient penalty: {gradient_penalty.item():.4f}")
    except Exception as e:
        print(f"Gradient penalty computation failed: {e}")
    
    # 5. 网格提取示例
    print("\nTesting mesh extraction...")
    try:
        mesh_data = model.extract_mesh(
            latent_code=latent_code[:1],
            resolution=32,  # 使用较低分辨率
            threshold=0.0
        )
        if 'vertices' in mesh_data:
            print(f"Extracted mesh: {len(mesh_data['vertices'])} vertices, {len(mesh_data['faces'])} faces")
        else:
            print("Mesh extraction returned SDF grid only")
    except Exception as e:
        print(f"Mesh extraction failed: {e}")
    
    return model, dataset


def example_latent_sdf_network():
    """潜在SDF网络使用示例"""
    print("=" * 50)
    print("Latent SDF Network Example")
    print("=" * 50)
    
    # 创建潜在SDF网络
    latent_model = LatentSDFNetwork(
        dim_latent=256,
        num_shapes=100,  # 预定义100个形状
        dim_hidden=512,
        num_layers=8
    )
    
    print(f"Latent SDF model created with {sum(p.numel() for p in latent_model.parameters()):,} parameters")
    
    # 测试形状编码获取
    shape_id = 42
    try:
        latent_code = latent_model.get_latent_code(shape_id)
        print(f"Retrieved latent code for shape {shape_id}: {latent_code.shape}")
        
        # 测试前向传播
        points = torch.randn(1, 1000, 3)
        pred_sdf = latent_model(points, shape_ids=torch.tensor([shape_id]))
        print(f"Predicted SDF shape: {pred_sdf.shape}")
        
    except Exception as e:
        print(f"Latent SDF forward pass failed: {e}")
    
    return latent_model


def example_conditional_occupancy_network():
    """条件占用网络使用示例"""
    print("=" * 50)
    print("Conditional Occupancy Network Example")
    print("=" * 50)
    
    # 创建条件占用网络
    conditional_model = ConditionalOccupancyNetwork(
        dim_input=3,
        dim_hidden=256,
        num_layers=8,
        dim_condition=128
    )
    
    print(f"Conditional model created with {sum(p.numel() for p in conditional_model.parameters()):,} parameters")
    
    # 测试条件生成
    points = torch.randn(2, 1000, 3)
    occupancy = torch.randint(0, 2, (2, 1000, 1)).float()
    
    try:
        # 编码形状特征
        shape_code = conditional_model.encode_shape(points, occupancy)
        print(f"Encoded shape code: {shape_code.shape}")
        
        # 条件生成
        pred_occupancy = conditional_model(points, condition=shape_code)
        print(f"Conditional prediction shape: {pred_occupancy.shape}")
        
        # 计算损失
        loss = conditional_model.compute_loss(pred_occupancy, occupancy)
        print(f"Conditional loss: {loss.item():.4f}")
        
    except Exception as e:
        print(f"Conditional network failed: {e}")
    
    return conditional_model


def example_training_setup():
    """训练设置示例"""
    print("=" * 50)
    print("Training Setup Example")
    print("=" * 50)
    
    # 占用网络训练设置
    print("Setting up Occupancy Network training...")
    occ_model = OccupancyNetwork(dim_hidden=128, num_layers=5)
    occ_dataset = SyntheticOccupancyDataset(num_samples=50, num_points=5000)
    occ_dataloader = create_occupancy_dataloader(occ_dataset, batch_size=2, num_workers=0)
    
    occ_trainer = OccupancyTrainer(
        model=occ_model,
        train_dataloader=occ_dataloader,
        device='cpu',  # 使用CPU进行演示
        log_dir='logs/demo_occupancy',
        checkpoint_dir='checkpoints/demo_occupancy'
    )
    
    print("Occupancy trainer created successfully")
    
    # SDF网络训练设置
    print("\nSetting up SDF Network training...")
    sdf_model = SDFNetwork(dim_hidden=256, num_layers=6, dim_latent=128)
    sdf_dataset = SyntheticSDFDataset(num_samples=50, num_points=5000)
    sdf_dataloader = create_sdf_dataloader(sdf_dataset, batch_size=2, num_workers=0)
    
    # 注意：SDF训练器还需要实现，这里只是演示创建过程
    print("SDF dataset and dataloader created successfully")
    
    return occ_trainer, sdf_dataloader


def compare_networks():
    """比较不同网络的性能"""
    print("=" * 50)
    print("Network Comparison")
    print("=" * 50)
    
    # 创建相同大小的网络进行比较
    networks = {
        'Occupancy': OccupancyNetwork(dim_hidden=256, num_layers=6),
        'SDF': SDFNetwork(dim_hidden=256, num_layers=6, dim_latent=256),
        'Conditional_Occupancy': ConditionalOccupancyNetwork(dim_hidden=256, num_layers=6, dim_condition=128)
    }
    
    print("Network Parameter Comparison:")
    print("-" * 30)
    
    for name, model in networks.items():
        model_info = model.get_model_size()
        print(f"{name}:")
        print(f"  Parameters: {model_info['total_parameters']:,}")
        print(f"  Size (MB): {model_info['model_size_mb']:.2f}")
        print()
    
    # 测试推理速度
    print("Inference Speed Test:")
    print("-" * 20)
    
    test_points = torch.randn(1, 10000, 3)
    
    for name, model in networks.items():
        model.eval()
        with torch.no_grad():
            start_time = torch.cuda.Event(enable_timing=True) if torch.cuda.is_available() else None
            end_time = torch.cuda.Event(enable_timing=True) if torch.cuda.is_available() else None
            
            import time
            start = time.time()
            
            if name == 'SDF':
                latent_code = torch.randn(1, 256)
                _ = model(test_points, latent_code)
            elif name == 'Conditional_Occupancy':
                condition = torch.randn(1, 128)
                _ = model(test_points, condition=condition)
            else:
                _ = model(test_points)
            
            end = time.time()
            
            print(f"{name}: {(end - start) * 1000:.2f} ms")


def main():
    """主函数 - 运行所有示例"""
    print("NeuroCity Models Package - Example Usage")
    print("=" * 60)
    
    try:
        # 1. 占用网络示例
        occ_model, occ_dataset = example_occupancy_network()
        
        # 2. SDF网络示例  
        sdf_model, sdf_dataset = example_sdf_network()
        
        # 3. 潜在SDF网络示例
        latent_sdf_model = example_latent_sdf_network()
        
        # 4. 条件占用网络示例
        cond_occ_model = example_conditional_occupancy_network()
        
        # 5. 训练设置示例
        occ_trainer, sdf_dataloader = example_training_setup()
        
        # 6. 网络比较
        compare_networks()
        
        print("=" * 60)
        print("All examples completed successfully!")
        print("=" * 60)
        
        # 输出总结信息
        print("\nSummary:")
        print(f"- Occupancy Network: {sum(p.numel() for p in occ_model.parameters()):,} parameters")
        print(f"- SDF Network: {sum(p.numel() for p in sdf_model.parameters()):,} parameters")
        print(f"- Latent SDF Network: {sum(p.numel() for p in latent_sdf_model.parameters()):,} parameters")
        print(f"- Conditional Occupancy Network: {sum(p.numel() for p in cond_occ_model.parameters()):,} parameters")
        
    except Exception as e:
        print(f"Example execution failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main() 