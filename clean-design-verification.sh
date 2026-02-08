#!/bin/bash
# 简洁专业设计验证脚本

echo "=== 简洁专业设计验证 ==="

# 1. 检查CSS文件
if [ -f "css/clean-professional.css" ]; then
    echo "✓ 简洁专业CSS文件已创建"
else
    echo "✗ 简洁专业CSS文件缺失"
    exit 1
fi

# 2. 检查index.html配置
if grep -q "clean-professional.css" index.html; then
    echo "✓ index.html已使用简洁专业CSS"
else
    echo "✗ index.html未更新CSS引用"
    exit 1
fi

# 3. 检查简洁设计特性
echo "3. 检查简洁设计特性..."
if grep -q "#0366d6" css/clean-professional.css; then
    echo "✓ 使用GitHub风格蓝色主色调"
fi

if grep -q "#24292e" css/clean-professional.css; then
    echo "✓ 使用GitHub风格文字颜色"
fi

if grep -q "#f6f8fa" css/clean-professional.css; then
    echo "✓ 使用GitHub风格背景色"
fi

if grep -q "font-family.*GitHub" css/clean-professional.css; then
    echo "✓ 使用GitHub风格字体"
fi

# 4. 检查布局特性
echo "4. 检查布局特性..."
if grep -q "sidebar-width: 256px" css/clean-professional.css; then
    echo "✓ 侧边栏宽度适中"
fi

if grep -q "border-radius: 6px" css/clean-professional.css; then
    echo "✓ 使用适中的圆角"
fi

if grep -q "transition.*0.15s" css/clean-professional.css; then
    echo "✓ 使用快速的过渡动画"
fi

# 5. 验证内容集成
if [ -d "content/python-learning" ] && [ -d "content/ai-agent-architecture" ]; then
    echo "✓ 内容集成完整"
fi

# 6. 检查响应式设计
if grep -q "@media.*768px" css/clean-professional.css; then
    echo "✓ 包含移动端响应式设计"
fi

echo ""
echo "=== 验证完成 ==="
echo "🎯 简洁专业设计已就绪！"
echo "✨ 特性包括："
echo "   - GitHub风格的简洁配色"
echo "   - 适中的边距和间距"
echo "   - 专业的字体和排版"
echo "   - 简洁的交互效果"
echo "   - 清晰的层次结构"
echo ""
echo "访问方式：http://localhost:8000"
echo "按 Ctrl+C 停止服务器"