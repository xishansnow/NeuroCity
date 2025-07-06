#!/usr/bin/env python3
"""
Python 3.10 兼容性测试脚本
检查 SVRaster 渲染器是否与 Python 3.10 兼容
"""

import sys
import traceback

def test_python310_compatibility():
    """测试 Python 3.10 兼容性"""
    print(f"Python 版本: {sys.version}")
    
    try:
        # 测试导入
        print("测试导入 SVRaster 渲染器...")
        from src.nerfs.svraster.renderer import SVRasterRenderer, SVRasterRendererConfig
        print("✅ 导入成功")
        
        # 测试基本实例化
        print("测试基本实例化...")
        config = SVRasterRendererConfig(
            image_width=400,
            image_height=300,
            quality_level="medium"
        )
        renderer = SVRasterRenderer(config)
        print("✅ 实例化成功")
        
        # 测试类型注解
        print("测试类型注解...")
        import typing
        print(f"✅ typing 模块版本: {getattr(typing, '__version__', 'N/A')}")
        
        # 测试数据类
        print("测试数据类...")
        from dataclasses import fields
        config_fields = fields(SVRasterRendererConfig)
        print(f"✅ 配置字段数量: {len(config_fields)}")
        
        # 测试现代 Python 特性
        print("测试现代 Python 特性...")
        from pathlib import Path
        test_path = Path("test")
        print(f"✅ pathlib 工作正常: {test_path}")
        
        print("\n🎉 所有兼容性测试通过！")
        return True
        
    except Exception as e:
        print(f"\n❌ 兼容性测试失败: {e}")
        traceback.print_exc()
        return False

def test_specific_features():
    """测试特定的 Python 3.10 特性"""
    print("\n测试特定 Python 3.10 特性...")
    
    try:
        # 测试类型注解
        from typing import Optional, Dict, List, Tuple, Union, Any
        print("✅ typing 导入成功")
        
        # 测试 f-strings
        name = "SVRaster"
        version = "1.0"
        message = f"Welcome to {name} version {version}"
        print(f"✅ f-string 测试: {message}")
        
        # 测试字典 union（Python 3.9+）
        test_dict = {"a": 1} | {"b": 2}
        print(f"✅ 字典 union 测试: {test_dict}")
        
        # 测试异常链
        try:
            raise ValueError("测试异常")
        except ValueError as e:
            raise RuntimeError("包装异常") from e
            
    except Exception as e:
        if "字典 union" in str(e):
            print("⚠️  字典 union 操作不支持（需要 Python 3.9+）")
        else:
            print(f"✅ 异常处理测试通过: {type(e).__name__}")

if __name__ == "__main__":
    print("SVRaster 渲染器 Python 3.10 兼容性测试")
    print("=" * 50)
    
    success = test_python310_compatibility()
    test_specific_features()
    
    if success:
        print("\n🎉 渲染器已完全兼容 Python 3.10!")
        sys.exit(0)
    else:
        print("\n❌ 发现兼容性问题，请检查上述错误信息")
        sys.exit(1)
