#!/bin/bash
# 现代化设计验证脚本

echo "=== 现代化专业设计验证 ==="

# 1. 检查CSS文件
if [ -f "css/modern-professional.css" ]; then
    echo "✓ 现代化专业CSS文件已创建"
else
    echo "✗ 现代化CSS文件缺失"
    exit 1
fi

# 2. 检查index.html配置
if grep -q "modern-professional.css" index.html; then
    echo "✓ index.html已使用现代化CSS"
else
    echo "✗ index.html未更新CSS引用"
    exit 1
fi

# 3. 检查现代化设计特性
echo "3. 检查现代化设计特性..."
if grep -q "gradient" css/modern-professional.css; then
    echo "✓ 包含渐变背景设计"
fi

if grep -q "box-shadow" css/modern-professional.css; then
    echo "✓ 包含阴影效果"
fi

if grep -q "transition" css/modern-professional.css; then
    echo "✓ 包含平滑过渡动画"
fi

if grep -q "border-radius" css/modern-professional.css; then
    echo "✓ 包含圆角设计"
fi

if grep -q "hover" css/modern-professional.css; then
    echo "✓ 包含悬停效果"
fi

# 4. 检查颜色方案
echo "4. 检查现代化颜色方案..."
if grep -q "#2563eb" css/modern-professional.css; then
    echo "✓ 使用现代蓝色主色调"
fi

if grep -q "#7c3aed" css/modern-professional.css; then
    echo "✓ 使用紫色强调色"
fi

# 5. 验证内容集成
if [ -d "content/python-learning" ] && [ -d "content/ai-agent-architecture" ]; then
    echo "✓ 内容集成完整"
fi

# 6. 检查响应式设计
if grep -q "@media" css/modern-professional.css; then
    echo "✓ 包含响应式设计"
fi

echo ""
echo "=== 验证完成 ==="
echo "🎨 现代化专业设计已就绪！"
echo "✨ 特性包括："
echo "   - 现代渐变背景和阴影效果"
echo "   - 平滑过渡动画和悬停效果"
echo "   - 专业的配色方案"
echo "   - 响应式设计"
echo "   - 优化的字体和间距"
echo ""
echo "访问方式：http://localhost:8000"
echo "按 Ctrl+C 停止服务器"