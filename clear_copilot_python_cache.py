#!/usr/bin/env python3
"""
清理 Python 和 Copilot 相关缓存
"""

import os
import shutil
from pathlib import Path

def clear_python_cache():
    """清理 Python 缓存"""
    print("🧹 清理 Python 缓存...")
    
    # 清理 __pycache__ 目录
    for root, dirs, files in os.walk("."):
        for dir_name in dirs:
            if dir_name == "__pycache__":
                cache_path = Path(root) / dir_name
                print(f"删除: {cache_path}")
                shutil.rmtree(cache_path, ignore_errors=True)
    
    # 清理 .pyc 文件
    for root, dirs, files in os.walk("."):
        for file_name in files:
            if file_name.endswith(".pyc"):
                pyc_path = Path(root) / file_name
                print(f"删除: {pyc_path}")
                pyc_path.unlink(missing_ok=True)
    
    print("✅ Python 缓存清理完成")

def clear_copilot_cache():
    """清理 Copilot 缓存"""
    print("🤖 清理 Copilot 缓存...")
    
    home = Path.home()
    
    # VS Code 扩展缓存
    vscode_extensions = home / ".vscode" / "extensions"
    if vscode_extensions.exists():
        for ext_dir in vscode_extensions.glob("github.copilot-*"):
            cache_dirs = [
                ext_dir / "dist" / "cache",
                ext_dir / "dist" / "language_server_*"
            ]
            for cache_dir in cache_dirs:
                if cache_dir.exists():
                    print(f"删除: {cache_dir}")
                    shutil.rmtree(cache_dir, ignore_errors=True)
    
    # Copilot 配置缓存
    copilot_config = home / ".config" / "github-copilot"
    if copilot_config.exists():
        print(f"删除: {copilot_config}")
        shutil.rmtree(copilot_config, ignore_errors=True)
    
    print("✅ Copilot 缓存清理完成")

def main():
    print("🧹 开始清理缓存...")
    clear_python_cache()
    clear_copilot_cache()
    print("\n🎉 所有缓存清理完成！")
    print("📝 建议重启 VS Code 以使更改生效")

if __name__ == "__main__":
    main()