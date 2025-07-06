#!/usr/bin/env python3
"""
Final refactor verification test

Tests that all components can be imported and instantiated correctly after refactoring.
"""

import torch
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))


def test_import_and_instantiation():
    """Test that all refactored components can be imported and instantiated."""
    print("=" * 70)
    print("SVRaster Final Refactor Verification Test")
    print("=" * 70)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    
    # 1. Test core imports
    try:
        from src.nerfs.svraster.core import SVRasterConfig, SVRasterModel, SVRasterLoss
        print("✅ 1. 核心组件导入成功")
    except Exception as e:
        print(f"❌ 1. 核心组件导入失败: {e}")
        return False
    
    # 2. Test volume renderer import
    try:
        from src.nerfs.svraster.volume_renderer import VolumeRenderer
        print("✅ 2. VolumeRenderer 导入成功")
    except Exception as e:
        print(f"❌ 2. VolumeRenderer 导入失败: {e}")
        return False
    
    # 3. Test spherical harmonics import
    try:
        from src.nerfs.svraster.spherical_harmonics import eval_sh_basis
        print("✅ 3. 球谐函数导入成功")
    except Exception as e:
        print(f"❌ 3. 球谐函数导入失败: {e}")
        return False
    
    # 4. Test true rasterizer import
    try:
        from src.nerfs.svraster.true_rasterizer import TrueVoxelRasterizer
        print("✅ 4. TrueVoxelRasterizer 导入成功")
    except Exception as e:
        print(f"❌ 4. TrueVoxelRasterizer 导入失败: {e}")
        return False
    
    # 5. Test trainer import
    try:
        from src.nerfs.svraster.trainer import SVRasterTrainer, SVRasterTrainerConfig
        print("✅ 5. SVRasterTrainer 导入成功")
    except Exception as e:
        print(f"❌ 5. SVRasterTrainer 导入失败: {e}")
        return False
    
    # 6. Test renderer import
    try:
        from src.nerfs.svraster.renderer import SVRasterRenderer, SVRasterRendererConfig
        print("✅ 6. SVRasterRenderer 导入成功")
    except Exception as e:
        print(f"❌ 6. SVRasterRenderer 导入失败: {e}")
        return False
    
    # 7. Test package-level imports
    try:
        from src.nerfs.svraster import (
            SVRasterConfig, SVRasterModel, VolumeRenderer, 
            TrueVoxelRasterizer, SVRasterTrainer, SVRasterRenderer
        )
        print("✅ 7. 包级别导入成功")
    except Exception as e:
        print(f"❌ 7. 包级别导入失败: {e}")
        return False
    
    # 8. Test instantiation
    try:
        config = SVRasterConfig()
        model = SVRasterModel(config)
        volume_renderer = VolumeRenderer(config)
        print("✅ 8. 组件实例化成功")
        print(f"   - 模型参数数量: {sum(p.numel() for p in model.parameters()):,}")
        print(f"   - 模型设备: {next(model.parameters()).device}")
        print(f"   - 体积渲染器配置: {type(volume_renderer.config).__name__}")
    except Exception as e:
        print(f"❌ 8. 组件实例化失败: {e}")
        return False
    
    # 9. Test spherical harmonics function
    try:
        directions = torch.randn(100, 3)
        directions = directions / directions.norm(dim=-1, keepdim=True)
        sh_basis = eval_sh_basis(2, directions)
        print("✅ 9. 球谐函数计算成功")
        print(f"   - 输入方向形状: {directions.shape}")
        print(f"   - SH基函数形状: {sh_basis.shape}")
    except Exception as e:
        print(f"❌ 9. 球谐函数计算失败: {e}")
        return False
    
    print("\n" + "=" * 70)
    print("🎉 所有测试通过！重构成功完成！")
    print("=" * 70)
    return True


if __name__ == "__main__":
    success = test_import_and_instantiation()
    sys.exit(0 if success else 1)
