#!/usr/bin/env python3
"""
编译 NeuroCity 项目中的所有 CUDA 扩展
"""

import os
import sys
import subprocess
import shutil
import argparse
from pathlib import Path

def print_header(title):
    """打印标题"""
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")

def print_status(message, success=True):
    """打印状态信息"""
    status = "✅" if success else "❌"
    print(f"{status} {message}")

def check_cuda_environment():
    """检查 CUDA 环境"""
    print_header("检查 CUDA 环境")
    
    # 检查 nvcc
    try:
        result = subprocess.run(['nvcc', '--version'], 
                              capture_output=True, text=True)
        if result.returncode == 0:
            print_status("nvcc 可用")
            print(f"   版本: {result.stdout.split('release')[1].split(',')[0].strip()}")
        else:
            print_status("nvcc 不可用", False)
            return False
    except FileNotFoundError:
        print_status("nvcc 未找到", False)
        return False
    
    # 检查 PyTorch CUDA
    try:
        import torch
        if torch.cuda.is_available():
            print_status("PyTorch CUDA 支持可用")
            print(f"   CUDA 版本: {torch.version.cuda}")
            print(f"   GPU 数量: {torch.cuda.device_count()}")
            print(f"   当前设备: {torch.cuda.get_device_name()}")
        else:
            print_status("PyTorch CUDA 支持不可用", False)
            return False
    except ImportError:
        print_status("PyTorch 未安装", False)
        return False
    
    return True

def clean_build_files(module_path):
    """清理构建文件"""
    build_dirs = ['build', 'dist', '*.egg-info']
    
    for pattern in build_dirs:
        if '*' in pattern:
            # 使用 glob 匹配
            for path in module_path.glob(pattern):
                if path.is_dir():
                    shutil.rmtree(path)
                    print(f"   清理: {path}")
        else:
            path = module_path / pattern
            if path.exists():
                shutil.rmtree(path)
                print(f"   清理: {path}")
    
    # 清理 .so 文件
    for so_file in module_path.glob('**/*.so'):
        so_file.unlink()
        print(f"   清理: {so_file}")

def compile_cuda_extension(module_path, module_name, force_rebuild=False):
    """编译单个 CUDA 扩展"""
    print_header(f"编译 {module_name} CUDA 扩展")
    
    if not module_path.exists():
        print_status(f"模块路径不存在: {module_path}", False)
        return False
    
    # 检查是否有 setup.py
    setup_py = module_path / 'setup.py'
    if not setup_py.exists():
        print_status(f"setup.py 不存在于 {module_path}", False)
        return False
    
    # 如果强制重建，清理构建文件
    if force_rebuild:
        print("清理旧的构建文件...")
        clean_build_files(module_path)
    
    # 设置环境变量
    env = os.environ.copy()
    env['TORCH_CUDA_ARCH_LIST'] = "6.0;6.1;7.0;7.5;8.0;8.6"
    env['CUDA_HOME'] = env.get('CUDA_HOME', '/usr/local/cuda')
    
    # 编译扩展
    try:
        print("开始编译...")
        result = subprocess.run([
            sys.executable, 'setup.py', 'build_ext', '--inplace'
        ], cwd=module_path, capture_output=True, text=True, env=env)
        
        if result.returncode == 0:
            print_status(f"{module_name} 编译成功")
            return True
        else:
            print_status(f"{module_name} 编译失败", False)
            print("错误输出:")
            print(result.stderr)
            return False
            
    except Exception as e:
        print_status(f"{module_name} 编译异常: {e}", False)
        return False

def compile_svraster():
    """编译 SVRaster CUDA 扩展"""
    module_path = Path('src/nerfs/svraster')
    return compile_cuda_extension(module_path, "SVRaster")

def compile_plenoxels():
    """编译 Plenoxels CUDA 扩展"""
    module_path = Path('src/nerfs/plenoxels')
    
    # 检查 CUDA 目录
    cuda_dir = module_path / 'cuda'
    if cuda_dir.exists():
        # 如果有专门的 CUDA 构建脚本，使用它
        build_script = cuda_dir / 'build_cuda.sh'
        if build_script.exists():
            print_header("编译 Plenoxels CUDA 扩展")
            try:
                result = subprocess.run(['bash', str(build_script)], 
                                      cwd=cuda_dir, capture_output=True, text=True)
                if result.returncode == 0:
                    print_status("Plenoxels CUDA 编译成功")
                    return True
                else:
                    print_status("Plenoxels CUDA 编译失败", False)
                    print("错误输出:")
                    print(result.stderr)
                    return False
            except Exception as e:
                print_status(f"Plenoxels CUDA 编译异常: {e}", False)
                return False
    
    # 回退到标准 setup.py 方法
    return compile_cuda_extension(module_path, "Plenoxels")

def compile_infnerf():
    """编译 InfNeRF CUDA 扩展"""
    module_path = Path('src/nerfs/inf_nerf')
    return compile_cuda_extension(module_path, "InfNeRF")

def verify_compilation():
    """验证编译结果"""
    print_header("验证编译结果")
    
    modules_to_test = [
        ("SVRaster", "src.nerfs.svraster"),
        ("Plenoxels", "src.nerfs.plenoxels"),
        ("InfNeRF", "src.nerfs.inf_nerf"),
    ]
    
    success_count = 0
    
    for module_name, import_path in modules_to_test:
        try:
            __import__(import_path)
            print_status(f"{module_name} 模块导入成功")
            success_count += 1
        except ImportError as e:
            print_status(f"{module_name} 模块导入失败: {e}", False)
        except Exception as e:
            print_status(f"{module_name} 模块测试异常: {e}", False)
    
    print(f"\n编译验证完成: {success_count}/{len(modules_to_test)} 模块成功")
    return success_count == len(modules_to_test)

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='编译 NeuroCity CUDA 扩展')
    parser.add_argument('--force', action='store_true', 
                       help='强制重建（清理旧文件）')
    parser.add_argument('--module', choices=['svraster', 'plenoxels', 'infnerf', 'all'],
                       default='all', help='指定要编译的模块')
    parser.add_argument('--skip-verify', action='store_true',
                       help='跳过编译验证')
    
    args = parser.parse_args()
    
    print("NeuroCity CUDA 扩展编译器")
    print("=" * 60)
    
    # 检查 CUDA 环境
    if not check_cuda_environment():
        print("\n❌ CUDA 环境检查失败，无法编译 CUDA 扩展")
        sys.exit(1)
    
    # 编译指定模块
    success_modules = []
    failed_modules = []
    
    if args.module in ['svraster', 'all']:
        if compile_svraster():
            success_modules.append('SVRaster')
        else:
            failed_modules.append('SVRaster')
    
    if args.module in ['plenoxels', 'all']:
        if compile_plenoxels():
            success_modules.append('Plenoxels')
        else:
            failed_modules.append('Plenoxels')
    
    if args.module in ['infnerf', 'all']:
        if compile_infnerf():
            success_modules.append('InfNeRF')
        else:
            failed_modules.append('InfNeRF')
    
    # 验证编译结果
    if not args.skip_verify:
        verification_passed = verify_compilation()
    else:
        verification_passed = True
    
    # 输出结果
    print_header("编译结果总结")
    print(f"成功编译: {', '.join(success_modules) if success_modules else '无'}")
    print(f"编译失败: {', '.join(failed_modules) if failed_modules else '无'}")
    
    if failed_modules:
        print("\n编译失败的可能原因:")
        print("1. CUDA 工具包版本不兼容")
        print("2. PyTorch 版本与 CUDA 版本不匹配")
        print("3. 缺少必要的依赖库")
        print("4. 编译环境配置问题")
        print("\n请检查错误输出并参考文档进行排错")
        sys.exit(1)
    
    if verification_passed:
        print("\n🎉 所有 CUDA 扩展编译并验证成功！")
    else:
        print("\n⚠️  编译完成但验证失败，请检查模块导入")
        sys.exit(1)

if __name__ == "__main__":
    main()
