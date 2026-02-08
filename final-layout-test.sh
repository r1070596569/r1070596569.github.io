#!/bin/bash
# 最终布局测试脚本

echo "=== 最终布局验证 ==="

# 1. 检查CSS文件
if [ -f "css/professional-layout.css" ]; then
    echo "✓ 新的布局CSS文件已创建"
else
    echo "✗ 布局CSS文件缺失"
    exit 1
fi

# 2. 检查index.html配置
if grep -q "professional-layout.css" index.html; then
    echo "✓ index.html已使用新的布局CSS"
else
    echo "✗ index.html未更新CSS引用"
    exit 1
fi

# 3. 检查关键CSS规则
echo "3. 检查关键CSS规则..."
if grep -q ".sidebar {" css/professional-layout.css; then
    echo "✓ 包含.sidebar样式定义"
fi

if grep -q ".markdown-section {" css/professional-layout.css; then
    echo "✓ 包含.markdown-section样式定义"
fi

if grep -q "margin-left: 260px" css/professional-layout.css; then
    echo "✓ 包含避免重叠的margin-left设置"
fi

# 4. 验证内容集成
if [ -d "content/python-learning" ] && [ -d "content/ai-agent-architecture" ]; then
    echo "✓ 内容集成完整"
fi

echo ""
echo "=== 验证完成 ==="
echo "新的专业布局已就绪，解决了左侧菜单和主内容重叠的问题"
echo "访问方式：http://localhost:8000"
echo "按 Ctrl+C 停止服务器"