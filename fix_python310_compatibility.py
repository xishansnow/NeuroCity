#!/usr/bin/env python3
"""
自动修复 Python 3.10 兼容性问题

这个脚本会自动将新式类型注解转换为兼容 Python 3.10 的格式
"""

from __future__ import annotations

import re
import os
from pathlib import Path
from typing import Dict, List, Set


def fix_type_annotations(file_path: Path) -> bool:
    """修复单个文件的类型注解"""

    try:
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()

        original_content = content

        # 检查是否已经有 future import
        has_future_import = "from __future__ import annotations" in content

        # 替换新式类型注解
        replacements = {
            r"\blist\[([^\]]+)\]": r"List[\1]",
            r"\bdict\[([^\]]+)\]": r"Dict[\1]",
            r"\btuple\[([^\]]+)\]": r"Tuple[\1]",
            r"\bset\[([^\]]+)\]": r"Set[\1]",
        }

        for pattern, replacement in replacements.items():
            content = re.sub(pattern, replacement, content)

        # 检查是否需要添加 typing 导入
        needs_typing_imports = set()
        if "List[" in content:
            needs_typing_imports.add("List")
        if "Dict[" in content:
            needs_typing_imports.add("Dict")
        if "Tuple[" in content:
            needs_typing_imports.add("Tuple")
        if "Set[" in content:
            needs_typing_imports.add("Set")
        if "Union[" in content:
            needs_typing_imports.add("Union")
        if "Optional[" in content:
            needs_typing_imports.add("Optional")
        if "Any" in content and not content.startswith('"""'):
            needs_typing_imports.add("Any")

        # 添加必要的导入
        if needs_typing_imports and "from typing import" not in content:
            imports = ", ".join(sorted(needs_typing_imports))
            import_line = f"from typing import {imports}\n"

            # 找到合适的位置插入导入
            lines = content.split("\n")
            insert_index = 0

            # 寻找导入语句的位置
            for i, line in enumerate(lines):
                if line.startswith("from __future__"):
                    insert_index = i + 1
                elif line.startswith("import ") or line.startswith("from "):
                    insert_index = i + 1
                elif line.strip() == "" and insert_index > 0:
                    break

            lines.insert(insert_index, import_line)
            content = "\n".join(lines)

        # 添加 future import 如果需要且不存在
        if needs_typing_imports and not has_future_import:
            lines = content.split("\n")

            # 找到第一个非注释、非空行
            insert_index = 0
            for i, line in enumerate(lines):
                if (
                    line.strip()
                    and not line.strip().startswith('"""')
                    and not line.strip().startswith("#")
                ):
                    insert_index = i
                    break

            lines.insert(insert_index, "from __future__ import annotations\n")
            content = "\n".join(lines)

        # 只有内容变化时才写入文件
        if content != original_content:
            with open(file_path, "w", encoding="utf-8") as f:
                f.write(content)
            return True

        return False

    except Exception as e:
        print(f"处理文件 {file_path} 时出错: {e}")
        return False


def main():
    """主函数"""
    svraster_dir = Path("src/nerfs/")

    if not svraster_dir.exists():
        print(f"SVRaster 目录不存在: {svraster_dir}")
        return

    print("开始修复 Python 3.10 兼容性问题...")

    modified_files = []

    # 遍历所有 Python 文件
    for py_file in svraster_dir.rglob("*.py"):
        print(f"处理文件: {py_file}")
        if fix_type_annotations(py_file):
            modified_files.append(str(py_file))

    if modified_files:
        print(f"\n✅ 已修复 {len(modified_files)} 个文件:")
        for file_path in modified_files:
            print(f"  - {file_path}")
    else:
        print("\n✅ 所有文件已经兼容 Python 3.10")

    print("\n🎉 Python 3.10 兼容性修复完成!")


if __name__ == "__main__":
    main()
