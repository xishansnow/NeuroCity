#!/bin/bash
# 文件名: clear_copilot_cache.sh

echo "🧹 正在清理 GitHub Copilot 缓存..."

# 1. 清理 VS Code 扩展缓存
echo "清理 VS Code 扩展缓存..."
rm -rf ~/.vscode/extensions/github.copilot-*/dist/cache/ 2>/dev/null
rm -rf ~/.vscode/extensions/github.copilot-*/dist/language_server_* 2>/dev/null

# 2. 清理 Copilot 配置缓存
echo "清理 Copilot 配置缓存..."
rm -rf ~/.config/github-copilot/ 2>/dev/null

# 3. 清理 Python 缓存
echo "清理 Python 缓存..."
find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null
find . -type f -name "*.pyc" -delete 2>/dev/null

# 4. 清理 VS Code 工作区缓存
echo "清理 VS Code 工作区缓存..."
rm -rf .vscode/settings.json.bak 2>/dev/null

echo "✅ 缓存清理完成！"
echo "📝 建议重启 VS Code 以使更改生效"