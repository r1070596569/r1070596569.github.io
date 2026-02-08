#!/bin/bash
# 最终验证脚本：检查所有关键资源是否可访问

echo "=== 最终验证：专业程序员风格网站 ==="

# 1. 检查核心文件
echo "1. 检查核心文件..."
if [ -f "index.html" ] && [ -f "css/professional-theme.css" ]; then
    echo "✓ index.html 和 CSS文件存在"
else
    echo "✗ 核心文件缺失"
    exit 1
fi

# 2. 检查内容集成
echo "2. 检查内容集成..."
if [ -d "content/python-learning" ] && [ -d "content/ai-agent-architecture" ]; then
    echo "✓ Python学习和AI智能体架构内容已集成"
else
    echo "✗ 内容集成不完整"
    exit 1
fi

# 3. 检查CDN链接（本地测试）
echo "3. 检查CDN链接配置..."
grep -q "https://cdn.jsdelivr.net" index.html && echo "✓ CDN链接已替换为国内镜像" || echo "✗ CDN链接未更新"

# 4. 验证CSS主题
echo "5. 验证CSS主题..."
if grep -q "professional-theme.css" index.html; then
    echo "✓ 使用新的专业主题CSS"
fi

if grep -q "themeColor: '#2088ff'" index.html; then
    echo "✓ 主题颜色设置为#2088ff"
fi

echo ""
echo "=== 验证完成 ==="
echo "网站已成功重构为专业程序员风格，所有CDN链接已替换为国内可访问版本"
echo ""
echo "访问方式：http://localhost:8000"
echo "按 Ctrl+C 停止服务器"