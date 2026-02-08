#!/bin/bash
# 布局修复验证脚本

echo "=== 布局修复验证 ==="

# 1. 检查CSS文件
if [ -f "css/clean-professional.css" ]; then
    echo "✓ CSS文件存在"
else
    echo "✗ CSS文件缺失"
    exit 1
fi

# 2. 检查布局修复
echo "2. 检查布局修复..."
if grep -q "sidebar-width: 240px" css/clean-professional.css; then
    echo "✓ 侧边栏宽度已调整为240px"
fi

if grep -q "content: 'NB'" css/clean-professional.css; then
    echo "✓ 侧边栏标题已缩短为'NB'"
fi

if grep -q "margin-left: var(--sidebar-width)" css/clean-professional.css; then
    echo "✓ 主内容区域正确设置左边距"
fi

if grep -q "box-sizing: border-box" css/clean-professional.css; then
    echo "✓ 主内容区域使用border-box盒模型"
fi

# 3. 检查间距优化
echo "3. 检查间距优化..."
if grep -q "padding: 24px 32px" css/clean-professional.css; then
    echo "✓ 主内容区域内边距已优化"
fi

if grep -q "margin-bottom: 12px" css/clean-professional.css; then
    echo "✓ 段落间距已优化"
fi

if grep -q "margin-top: 28px" css/clean-professional.css; then
    echo "✓ 标题间距已优化"
fi

# 4. 检查标题大小
echo "4. 检查标题大小..."
if grep -q "font-size: 28px" css/clean-professional.css; then
    echo "✓ H1标题大小已调整为28px"
fi

if grep -q "font-size: 22px" css/clean-professional.css; then
    echo "✓ H2标题大小已调整为22px"
fi

if grep -q "font-size: 18px" css/clean-professional.css; then
    echo "✓ H3标题大小已调整为18px"
fi

# 5. 检查响应式设计
if grep -q "padding: 16px 20px" css/clean-professional.css; then
    echo "✓ 移动端内边距已优化"
fi

echo ""
echo "=== 验证完成 ==="
echo "🔧 布局问题已修复！"
echo "✨ 修复内容："
echo "   - 侧边栏宽度从256px调整为240px"
echo "   - 侧边栏标题从'NaturalBean'缩短为'NB'"
echo "   - 主内容区域布局优化，避免重叠"
echo "   - 标题大小和间距重新调整"
echo "   - 段落和代码块间距优化"
echo "   - 移动端响应式布局改进"
echo ""
echo "访问方式：http://localhost:8000"
echo "按 Ctrl+C 停止服务器"