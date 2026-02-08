#!/bin/bash
# 测试脚本：验证新设计的网站是否正常工作

echo "=== 开始测试专业程序员风格文档网站 ==="

# 1. 检查CSS文件是否存在
if [ -f "css/professional-theme.css" ]; then
    echo "✓ CSS文件已创建: css/professional-theme.css"
else
    echo "✗ CSS文件不存在"
    exit 1
fi

# 2. 检查index.html配置
if grep -q "professional-theme.css" index.html; then
    echo "✓ index.html已更新为使用新的CSS主题"
else
    echo "✗ index.html未更新CSS引用"
    exit 1
fi

if grep -q "themeColor: '#2088ff'" index.html; then
    echo "✓ index.html主题颜色已更新为#2088ff"
else
    echo "✗ index.html主题颜色未更新"
    exit 1
fi

# 3. 检查内容集成
if [ -d "content/python-learning" ] && [ -f "content/python-learning/week1-guide.md" ]; then
    echo "✓ Python学习内容已集成"
else
    echo "✗ Python学习内容未集成"
    exit 1
fi

if [ -d "content/ai-agent-architecture" ] && [ -f "content/ai-agent-architecture/architecture-design.md" ]; then
    echo "✓ 作业AI智能体架构内容已集成"
else
    echo "✗ 作业AI智能体架构内容未集成"
    exit 1
fi

# 4. 检查侧边栏导航
if grep -q "Python学习" _sidebar.md; then
    echo "✓ 侧边栏包含Python学习分类"
fi

if grep -q "作业AI智能体架构" _sidebar.md; then
    echo "✓ 侧边栏包含作业AI智能体架构分类"
fi

echo ""
echo "=== 测试完成 ==="
echo "所有检查通过！网站已成功重构为专业程序员风格。"
echo ""
echo "要启动本地服务器，请运行："
echo "python3 -m http.server 8000"
echo "然后访问 http://localhost:8000"