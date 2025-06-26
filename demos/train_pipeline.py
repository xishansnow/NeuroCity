#!/usr/bin/env python3
"""
完整的训练流水线
整合体素采样和神经网络训练
"""

import os
import json
import argparse
from typing import Dict, Any
import logging

from sampler import VoxelSampler
from neural_sdf import MLP, NeuralSDFTrainer, load_training_data

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TrainingPipeline:
    def __init__(self, config: dict[str, Any]):
        """
        初始化训练流水线
        
        Args:
            config: 配置字典
        """
        self.config = config
        self.sampler = None
        self.trainer = None
        
    def run_sampling(self):
        """运行采样阶段"""
        logger.info("=== 开始采样阶段 ===")
        
        # 创建采样器
        self.sampler = VoxelSampler(
            tiles_dir=self.config['sampling']['tiles_dir'], voxel_size=self.config['sampling']['voxel_size'], sample_ratio=self.config['sampling']['sample_ratio']
        )
        
        # 执行采样
        samples = self.sampler.sample_all_tiles(
            sampling_method=self.config['sampling']['method'], n_samples_per_tile=self.config['sampling']['n_samples_per_tile'], **self.config['sampling'].get(
                'method_params',
                {},
            )
        )
        
        # 保存采样结果
        self.sampler.save_samples(
            samples, self.config['sampling']['output_dir']
        )
        
        logger.info("采样阶段完成！")
        return samples
    
    def run_training(self):
        """运行训练阶段"""
        logger.info("=== 开始训练阶段 ===")
        
        # 加载训练数据
        train_dataloader, val_dataloader = load_training_data(
            samples_dir=self.config['sampling']['output_dir'], task_type=self.config['training']['task_type'], train_ratio=self.config['training']['train_ratio']
        )
        
        # 创建模型
        model_config = self.config['model']
        model = MLP(
            input_dim=model_config['input_dim'], hidden_dims=model_config['hidden_dims'], output_dim=model_config['output_dim'], activation=model_config['activation'], dropout=model_config.get(
                'dropout',
                0.1,
            )
        )
        
        # 创建训练器
        training_config = self.config['training']
        self.trainer = NeuralSDFTrainer(
            model=model, device=training_config['device'], learning_rate=training_config['learning_rate'], weight_decay=training_config['weight_decay']
        )
        
        # 训练模型
        self.trainer.train(
            train_dataloader=train_dataloader, val_dataloader=val_dataloader, num_epochs=training_config['num_epochs'], save_path=training_config['model_save_path'], early_stopping_patience=training_config['early_stopping_patience']
        )
        
        # 保存训练历史
        self.trainer.plot_training_history(
            training_config['history_plot_path']
        )
        
        logger.info("训练阶段完成！")
    
    def run_evaluation(self, test_coords: list):
        """运行评估阶段"""
        logger.info("=== 开始评估阶段 ===")
        
        if self.trainer is None:
            raise ValueError("训练器未初始化，请先运行训练阶段")
        
        # 预测测试点
        predictions = self.trainer.predict(test_coords)
        
        # 保存预测结果
        results = {
            'test_coords': test_coords, 'predictions': predictions.tolist(
            )
        }
        
        with open(self.config['evaluation']['results_path'], 'w') as f:
            json.dump(results, f, indent=2)
        
        logger.info(f"评估结果已保存到: {self.config['evaluation']['results_path']}")
        logger.info("评估阶段完成！")
    
    def run_full_pipeline(self):
        """运行完整流水线"""
        logger.info("开始完整训练流水线...")
        
        # 1. 采样阶段
        self.run_sampling()
        
        # 2. 训练阶段
        self.run_training()
        
        # 3. 评估阶段（可选）
        if 'evaluation' in self.config:
            test_coords = self.config['evaluation'].get('test_coords', [])
            if test_coords:
                self.run_evaluation(test_coords)
        
        logger.info("完整流水线执行完毕！")

def get_default_config():
    """获取默认配置"""
    return {
        'sampling': {
            'tiles_dir': 'tiles', 'voxel_size': 1.0, 'sample_ratio': 0.1, 'method': 'stratified', 'n_samples_per_tile': 10000, 'output_dir': 'samples', 'method_params': {
                'occupied_ratio': 0.3
            }
        }, 'model': {
            'input_dim': 3, 'hidden_dims': [256, 512, 512, 256, 128], 'output_dim': 1, 'activation': 'relu', 'dropout': 0.1
        }, 'training': {
            'task_type': 'occupancy', # 'occupancy' 或 'sdf'
            'device': 'auto', 'learning_rate': 1e-3, 'weight_decay': 1e-5, 'num_epochs': 50, 'train_ratio': 0.8, 'early_stopping_patience': 10, 'model_save_path': 'model.pth', 'history_plot_path': 'training_history.png'
        }, 'evaluation': {
            'results_path': 'evaluation_results.json', 'test_coords': [
                [100, 100, 10], [200, 200, 20], [300, 300, 30]
            ]
        }
    }

def save_config(config: dict[str, Any], path: str):
    """保存配置"""
    with open(path, 'w') as f:
        json.dump(config, f, indent=2)
    logger.info(f"配置已保存到: {path}")

def load_config(path: str) -> dict[str, Any]:
    """加载配置"""
    with open(path, 'r') as f:
        config = json.load(f)
    logger.info(f"配置已从 {path} 加载")
    return config

def main():
    parser = argparse.ArgumentParser(description='训练流水线')
    parser.add_argument('--config', type=str, help='配置文件路径')
    parser.add_argument('--task', choices=['occupancy', 'sdf'], default='occupancy', help='任务类型')
    parser.add_argument('--tiles-dir', type=str, default='tiles', help='tiles目录')
    parser.add_argument('--samples-dir', type=str, default='samples', help='采样输出目录')
    parser.add_argument('--model-path', type=str, default='model.pth', help='模型保存路径')
    parser.add_argument('--epochs', type=int, default=50, help='训练轮数')
    parser.add_argument('--skip-sampling', action='store_true', help='跳过采样阶段')
    
    args = parser.parse_args()
    
    # 加载或创建配置
    if args.config and os.path.exists(args.config):
        config = load_config(args.config)
    else:
        config = get_default_config()
        
        # 更新配置
        config['sampling']['tiles_dir'] = args.tiles_dir
        config['sampling']['output_dir'] = args.samples_dir
        config['training']['task_type'] = args.task
        config['training']['num_epochs'] = args.epochs
        config['training']['model_save_path'] = args.model_path
        
        # 保存配置
        save_config(config, 'training_config.json')
    
    # 创建流水线
    pipeline = TrainingPipeline(config)
    
    if args.skip_sampling:
        # 跳过采样，直接训练
        logger.info("跳过采样阶段，直接开始训练...")
        pipeline.run_training()
    else:
        # 运行完整流水线
        pipeline.run_full_pipeline()

if __name__ == "__main__":
    main() 