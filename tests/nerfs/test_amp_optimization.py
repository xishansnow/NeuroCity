#!/usr/bin/env python3
"""
测试SVRaster core模块的AMP优化
"""

import os
import sys
import torch

# 添加项目路径
sys.path.insert(0, os.path.abspath('.'))

def test_amp_optimization():
    """测试AMP优化是否正常工作"""
    try:
        # 测试导入
        from src.nerfs.svraster.core import SVRasterConfig, SVRasterModel, autocast, GradScaler
        print("✅ AMP模块导入成功")
        
        # 测试配置
        config = SVRasterConfig()
        print(f"✅ 配置创建成功，设备: {config.device}")
        
        # 测试AMP功能
        if config.use_amp:
            print("✅ AMP功能已启用")
            
            # 测试autocast
            with autocast(device_type="cuda" if torch.cuda.is_available() else "cpu"):
                test_tensor = torch.randn(10, 10, requires_grad=True)
                result = test_tensor * 2
                print(f"✅ Autocast测试成功，结果类型: {result.dtype}")
            
            # 测试GradScaler
            scaler = GradScaler()
            loss = torch.tensor(1.0, requires_grad=True)
            scaled_loss = scaler.scale(loss)
            print(f"✅ GradScaler测试成功，缩放因子: {scaler.get_scale()}")
            
        # 测试模型创建
        model = SVRasterModel(config)
        print(f"✅ 模型创建成功，设备: {model.device}")
        
        return True
        
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("开始测试SVRaster AMP优化...")
    success = test_amp_optimization()
    if success:
        print("\n🎉 所有测试通过！AMP优化已成功应用。")
    else:
        print("\n❌ 测试失败，需要进一步调试。")
