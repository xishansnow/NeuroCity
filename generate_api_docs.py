#!/usr/bin/env python3
"""
API 文档自动生成器

这个脚本会自动扫描 SVRaster 模块，提取所有公共类、方法和函数的文档字符串，
并生成格式化的 API 参考文档。

使用方法:
    python generate_api_docs.py
"""

from __future__ import annotations

import ast
import inspect
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass
import importlib.util


@dataclass
class ClassInfo:
    """类信息"""
    name: str
    docstring: Optional[str]
    methods: List[MethodInfo]
    module_path: str


@dataclass 
class MethodInfo:
    """方法信息"""
    name: str
    docstring: Optional[str]
    signature: str
    is_property: bool = False
    is_static: bool = False
    is_class_method: bool = False


@dataclass
class FunctionInfo:
    """函数信息"""
    name: str
    docstring: Optional[str]
    signature: str
    module_path: str


class APIDocGenerator:
    """API 文档生成器"""
    
    def __init__(self, module_path: str, output_path: str):
        self.module_path = Path(module_path)
        self.output_path = Path(output_path)
        self.classes: List[ClassInfo] = []
        self.functions: List[FunctionInfo] = []
        
    def extract_module_info(self, file_path: Path) -> None:
        """提取模块信息"""
        try:
            # 读取源码
            with open(file_path, 'r', encoding='utf-8') as f:
                source = f.read()
            
            # 解析 AST
            tree = ast.parse(source)
            
            # 提取类和函数
            for node in ast.walk(tree):
                if isinstance(node, ast.ClassDef):
                    self._extract_class_info(node, str(file_path))
                elif isinstance(node, ast.FunctionDef) and not self._is_in_class(node, tree):
                    self._extract_function_info(node, str(file_path))
                    
        except Exception as e:
            print(f"解析文件失败 {file_path}: {e}")
    
    def _extract_class_info(self, node: ast.ClassDef, module_path: str) -> None:
        """提取类信息"""
        methods = []
        
        for item in node.body:
            if isinstance(item, ast.FunctionDef):
                method_info = MethodInfo(
                    name=item.name,
                    docstring=ast.get_docstring(item),
                    signature=self._get_function_signature(item),
                    is_static='staticmethod' in [d.id for d in item.decorator_list if isinstance(d, ast.Name)],
                    is_class_method='classmethod' in [d.id for d in item.decorator_list if isinstance(d, ast.Name)]
                )
                methods.append(method_info)
        
        class_info = ClassInfo(
            name=node.name,
            docstring=ast.get_docstring(node),
            methods=methods,
            module_path=module_path
        )
        
        self.classes.append(class_info)
    
    def _extract_function_info(self, node: ast.FunctionDef, module_path: str) -> None:
        """提取函数信息"""
        if not node.name.startswith('_'):  # 只处理公共函数
            function_info = FunctionInfo(
                name=node.name,
                docstring=ast.get_docstring(node),
                signature=self._get_function_signature(node),
                module_path=module_path
            )
            self.functions.append(function_info)
    
    def _get_function_signature(self, node: ast.FunctionDef) -> str:
        """获取函数签名"""
        args = []
        
        # 处理普通参数
        for arg in node.args.args:
            arg_str = arg.arg
            if arg.annotation:
                arg_str += f": {ast.unparse(arg.annotation)}"
            args.append(arg_str)
        
        # 处理默认参数
        defaults_offset = len(node.args.args) - len(node.args.defaults)
        for i, default in enumerate(node.args.defaults):
            args[defaults_offset + i] += f" = {ast.unparse(default)}"
        
        # 处理 *args
        if node.args.vararg:
            args.append(f"*{node.args.vararg.arg}")
        
        # 处理 **kwargs
        if node.args.kwarg:
            args.append(f"**{node.args.kwarg.arg}")
        
        # 处理返回类型
        return_type = ""
        if node.returns:
            return_type = f" -> {ast.unparse(node.returns)}"
        
        return f"{node.name}({', '.join(args)}){return_type}"
    
    def _is_in_class(self, func_node: ast.FunctionDef, tree: ast.AST) -> bool:
        """检查函数是否在类内部"""
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                if func_node in node.body:
                    return True
        return False
    
    def scan_directory(self) -> None:
        """扫描目录中的所有 Python 文件"""
        for file_path in self.module_path.rglob("*.py"):
            if file_path.name.startswith('_') and file_path.name != '__init__.py':
                continue
            print(f"扫描文件: {file_path}")
            self.extract_module_info(file_path)
    
    def generate_markdown(self) -> str:
        """生成 Markdown 格式的 API 文档"""
        content = []
        
        content.append("# SVRaster API 参考文档\n")
        content.append("本文档自动生成，包含 SVRaster 模块的所有公共 API。\n")
        
        # 生成目录
        content.append("## 📚 目录\n")
        
        if self.classes:
            content.append("### 🏗️ 类 (Classes)\n")
            for cls in sorted(self.classes, key=lambda x: x.name):
                content.append(f"- [{cls.name}](#{cls.name.lower()})")
            content.append("")
        
        if self.functions:
            content.append("### 🔧 函数 (Functions)\n")
            for func in sorted(self.functions, key=lambda x: x.name):
                content.append(f"- [{func.name}](#{func.name.lower()})")
            content.append("")
        
        # 生成类文档
        if self.classes:
            content.append("## 🏗️ 类 (Classes)\n")
            
            for cls in sorted(self.classes, key=lambda x: x.name):
                content.append(f"### {cls.name}\n")
                
                if cls.docstring:
                    content.append(f"{cls.docstring}\n")
                else:
                    content.append("*暂无文档*\n")
                
                content.append(f"**模块路径**: `{os.path.relpath(cls.module_path)}`\n")
                
                # 生成方法文档
                if cls.methods:
                    content.append("#### 方法 (Methods)\n")
                    
                    for method in sorted(cls.methods, key=lambda x: x.name):
                        if method.name.startswith('_') and method.name not in ['__init__', '__call__']:
                            continue
                        
                        # 方法标题
                        method_title = f"##### `{method.signature}`\n"
                        content.append(method_title)
                        
                        # 方法类型标签
                        tags = []
                        if method.is_static:
                            tags.append("🔧 静态方法")
                        elif method.is_class_method:
                            tags.append("🏭 类方法")
                        elif method.is_property:
                            tags.append("🏷️ 属性")
                        
                        if tags:
                            content.append(f"*{' | '.join(tags)}*\n")
                        
                        # 方法文档
                        if method.docstring:
                            content.append(f"{method.docstring}\n")
                        else:
                            content.append("*暂无文档*\n")
                        
                        content.append("---\n")
                
                content.append("\n")
        
        # 生成函数文档
        if self.functions:
            content.append("## 🔧 函数 (Functions)\n")
            
            for func in sorted(self.functions, key=lambda x: x.name):
                content.append(f"### `{func.signature}`\n")
                
                if func.docstring:
                    content.append(f"{func.docstring}\n")
                else:
                    content.append("*暂无文档*\n")
                
                content.append(f"**模块路径**: `{os.path.relpath(func.module_path)}`\n")
                content.append("---\n")
        
        # 生成时间戳
        from datetime import datetime
        content.append(f"\n---\n*本文档生成于 {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*\n")
        
        return '\n'.join(content)
    
    def generate_documentation(self) -> None:
        """生成完整的 API 文档"""
        print("开始扫描模块...")
        self.scan_directory()
        
        print(f"发现 {len(self.classes)} 个类, {len(self.functions)} 个函数")
        
        print("生成 Markdown 文档...")
        markdown_content = self.generate_markdown()
        
        # 确保输出目录存在
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # 写入文件
        with open(self.output_path, 'w', encoding='utf-8') as f:
            f.write(markdown_content)
        
        print(f"API 文档已生成: {self.output_path}")


def main():
    """主函数"""
    # 设置路径
    current_dir = Path(__file__).parent
    svraster_path = current_dir / "src" / "nerfs" / "svraster"
    output_path = current_dir / "src" / "nerfs" / "svraster" / "API_REFERENCE_cn.md"
    
    if not svraster_path.exists():
        print(f"错误: SVRaster 模块路径不存在: {svraster_path}")
        return
    
    # 生成文档
    generator = APIDocGenerator(str(svraster_path), str(output_path))
    generator.generate_documentation()


if __name__ == "__main__":
    main()
