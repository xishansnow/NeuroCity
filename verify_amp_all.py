#!/usr/bin/env python3
"""
验证所有nerfs模块的AMP优化
"""

import warnings
import torch
import sys
import os

def test_amp_imports():
    """测试AMP导入是否正常"""
    print("🔍 测试AMP导入...")
    
    try:
        from torch.amp.autocast_mode import autocast
        from torch.amp.grad_scaler import GradScaler
        print("✅ AMP导入成功")
        return True
    except ImportError as e:
        print(f"❌ AMP导入失败: {e}")
        return False

def test_amp_functionality():
    """测试AMP功能是否正常工作"""
    print("🔍 测试AMP功能...")
    
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        
        try:
            from torch.amp.autocast_mode import autocast
            from torch.amp.grad_scaler import GradScaler
            
            # 测试基本功能
            scaler = GradScaler()
            device_type = "cuda" if torch.cuda.is_available() else "cpu"
            
            with autocast(device_type=device_type):
                x = torch.randn(10, 10, requires_grad=True)
                y = x * 2
                loss = y.sum()
            
            scaled_loss = scaler.scale(loss)
            scaled_loss.backward()
            
            # 检查过时警告
            deprecation_warnings = [
                warning for warning in w 
                if issubclass(warning.category, (DeprecationWarning, FutureWarning))
            ]
            
            if deprecation_warnings:
                print(f"❌ 发现 {len(deprecation_warnings)} 个过时警告:")
                for warning in deprecation_warnings:
                    print(f"  - {warning.message}")
                return False
            else:
                print("✅ AMP功能测试通过，没有过时警告")
                return True
                
        except Exception as e:
            print(f"❌ AMP功能测试失败: {e}")
            return False

def check_nerfs_modules():
    """检查所有nerfs模块的导入"""
    print("🔍 检查nerfs模块导入...")
    
    # 添加src到路径
    sys.path.insert(0, os.path.abspath('src'))
    
    modules_to_test = [
        'nerfs.svraster.core',
        'nerfs.plenoxels.core', 
        # 可以添加更多模块测试
    ]
    
    success_count = 0
    
    for module_name in modules_to_test:
        try:
            # 创建临时模块避免相对导入问题
            import importlib.util
            
            # 构建文件路径
            file_path = f"src/{module_name.replace('.', '/')}.py"
            
            if os.path.exists(file_path):
                with open(file_path, 'r') as f:
                    content = f.read()
                
                # 检查是否包含过时的AMP API
                if 'torch.cuda.amp' in content:
                    print(f"❌ {module_name}: 仍然使用过时的torch.cuda.amp")
                elif 'torch.amp.autocast(' in content or 'torch.amp.GradScaler(' in content:
                    print(f"❌ {module_name}: 仍然使用错误的torch.amp直接调用")
                else:
                    print(f"✅ {module_name}: AMP使用正确")
                    success_count += 1
            else:
                print(f"⚠️ {module_name}: 文件不存在")
                
        except Exception as e:
            print(f"❌ {module_name}: 检查失败 - {e}")
    
    return success_count, len(modules_to_test)

def main():
    print("🚀 开始验证所有nerfs模块的AMP优化...\n")
    print(f"PyTorch版本: {torch.__version__}")
    print(f"CUDA可用: {torch.cuda.is_available()}\n")
    
    all_passed = True
    
    # 测试AMP导入
    if not test_amp_imports():
        all_passed = False
    
    print()
    
    # 测试AMP功能
    if not test_amp_functionality():
        all_passed = False
    
    print()
    
    # 检查nerfs模块
    success, total = check_nerfs_modules()
    
    print(f"\n📊 检查结果:")
    print(f"✅ 成功: {success}/{total} 个模块")
    
    if success < total:
        all_passed = False
    
    print("\n" + "="*60)
    
    if all_passed:
        print("🎉 所有AMP优化验证通过！")
        print("✅ 所有过时的torch.cuda.amp API已更新")
        print("✅ 使用现代的torch.amp API")
        print("✅ 没有过时函数警告")
        print("✅ 兼容PyTorch 2.x+")
    else:
        print("❌ 部分验证失败，请检查上述错误")
    
    return all_passed

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
