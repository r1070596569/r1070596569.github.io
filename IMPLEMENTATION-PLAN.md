# 实施计划：程序员风格文档网站重构

## 1. 概述
本计划详细说明如何将当前的docsify文档网站重构为专业的程序员风格网站。

## 2. 实施步骤

### 2.1 文件结构准备
```bash
# 创建必要的目录
mkdir -p css
mkdir -p content/python-learning
mkdir -p content/ai-agent-architecture
```

### 2.2 CSS主题创建
创建 `css/programmer-theme.css` 文件，内容如下：

```css
/* Programmer Theme - Dark */
:root {
  --bg-color: #1e1e1e;
  --sidebar-bg: #252526;
  --text-color: #d4d4d4;
  --link-color: #4ec9b0;
  --border-color: #3c3c3c;
  --code-bg: #1e1e1e;
  --code-text: #d4d4d4;
  --search-bg: #2d2d30;
}

body {
  background-color: var(--bg-color);
  color: var(--text-color);
  font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
  line-height: 1.6;
}

.sidebar {
  background-color: var(--sidebar-bg);
  border-right: 1px solid var(--border-color);
  padding: 20px 0;
}

.sidebar li a {
  color: var(--text-color);
  transition: color 0.2s ease;
  padding: 8px 20px;
  display: block;
  text-decoration: none;
}

.sidebar li a:hover {
  color: var(--link-color);
  background-color: rgba(78, 201, 176, 0.1);
}

.sidebar ul {
  list-style: none;
  padding: 0;
  margin: 0;
}

.sidebar > ul {
  padding: 0 10px;
}

.markdown-section {
  max-width: 800px;
  margin: 0 auto;
  padding: 20px;
  background-color: #1e1e1e;
  min-height: calc(100vh - 60px);
}

.markdown-section h1,
.markdown-section h2,
.markdown-section h3,
.markdown-section h4,
.markdown-section h5,
.markdown-section h6 {
  color: var(--link-color);
  border-bottom: 1px solid var(--border-color);
  padding-bottom: 10px;
  margin-top: 30px;
  margin-bottom: 20px;
}

a {
  color: var(--link-color);
  text-decoration: none;
}

a:hover {
  text-decoration: underline;
}

code {
  font-family: 'Fira Code', 'Cascadia Code', 'Consolas', 'Monaco', monospace;
  background-color: #2d2d30;
  padding: 2px 4px;
  border-radius: 3px;
  color: #ce9178;
}

pre {
  background-color: var(--code-bg);
  border: 1px solid var(--border-color);
  border-radius: 4px;
  padding: 16px;
  overflow-x: auto;
}

.token.comment {
  color: #6a9955;
}

.token.string {
  color: #ce9178;
}

.token.keyword {
  color: #569cd6;
}

.token.function {
  color: #dcdcaa;
}

.token.number {
  color: #b5cea8;
}

.token.operator {
  color: #d4d4d4;
}

table {
  background-color: #2d2d30;
  border-collapse: collapse;
  width: 100%;
  margin: 20px 0;
}

th, td {
  border: 1px solid var(--border-color);
  padding: 8px;
  text-align: left;
}

th {
  background-color: #3c3c3c;
  color: var(--link-color);
}

.search input {
  background-color: var(--search-bg);
  border: 1px solid var(--border-color);
  color: var(--text-color);
  padding: 8px 12px;
  border-radius: 4px;
  width: 100%;
  font-size: 14px;
}

.search .search-keyword {
  color: var(--link-color);
  font-weight: bold;
}

/* 响应式设计 */
@media (max-width: 768px) {
  .sidebar {
    position: fixed;
    top: 0;
    left: 0;
    height: 100%;
    width: 250px;
    z-index: 1000;
    transform: translateX(-100%);
    transition: transform 0.3s ease;
  }
  
  .sidebar.open {
    transform: translateX(0);
  }
  
  .markdown-section {
    padding: 15px;
  }
}

/* 页脚 */
footer {
  text-align: center;
  padding: 20px;
  color: #808080;
  font-size: 14px;
  border-top: 1px solid var(--border-color);
}
```

### 2.3 更新index.html配置
修改 `index.html` 文件：

1. **移除封面页配置**：删除或注释掉 `coverpage: true`
2. **添加自定义CSS**：在 `<head>` 部分添加自定义CSS链接
3. **更新主题颜色**：设置 `themeColor: '#4ec9b0'`

具体修改如下：

```html
<!-- 在<head>中添加自定义CSS -->
<link rel="stylesheet" href="css/programmer-theme.css">

<!-- 在window.$docsify配置中 -->
<script>
  window.$docsify = {
    name: 'NaturalBean的学习笔记',
    loadSidebar: true,
    loadNavbar: true,
    // coverpage: true, // 移除此行或注释掉
    maxLevel: 5,
    subMaxLevel: 4,
    mergeNavbar: true,
    autoHeader: true,
    topMargin: 15,
    themeColor: '#4ec9b0',
    // ... 其他配置保持不变
  }
</script>
```

### 2.4 内容文件重命名和移动

#### Python学习内容
```bash
# 重命名文件夹
mv "content/=== python学习 ===" content/python-learning

# 重命名文件
mv content/python-learning/python_ai_agent_week1_guide.md content/python-learning/week1-guide.md
mv content/python-learning/python_ai_agent_week2_guide.md content/python-learning/week2-guide.md
mv content/python-learning/python_ai_agent_week3_guide.md content/python-learning/week3-guide.md
mv content/python-learning/python_ai_agent_week4_guide.md content/python-learning/week4-guide.md
```

#### 作业AI智能体架构内容
```bash
# 重命名文件夹
mv "content/=== 作业AI智能体架构 ===" content/ai-agent-architecture

# 重命名文件
mv content/ai-agent-architecture/1-AI-Platform-Architecture-Design.md content/ai-agent-architecture/architecture-design.md
mv content/ai-agent-architecture/2-Core-Function-Modules-and-API-Specification.md content/ai-agent-architecture/core-modules.md
mv content/ai-agent-architecture/3-Agent-Architecture-and-Homework-Agent-Design.md content/ai-agent-architecture/agent-implementation.md
mv content/ai-agent-architecture/AI-Management-Platform-Technical-Specification.md content/ai-agent-architecture/technical-specification.md
mv content/ai-agent-architecture/6-Implementation-Roadmap-and-Technical-Specification.md content/ai-agent-architecture/roadmap.md
```

### 2.5 应用新的侧边栏
将 `_sidebar-new.md` 重命名为 `_sidebar.md`：

```bash
mv _sidebar-new.md _sidebar.md
```

### 2.6 清理不需要的文件
```bash
# 删除封面页文件（如果不需要）
rm _coverpage.md
```

## 3. 验证清单

- [ ] 自定义CSS文件已创建并正确链接
- [ ] index.html配置已更新（移除封面页，添加主题色）
- [ ] 所有内容文件已正确重命名和移动
- [ ] 侧边栏导航结构已更新
- [ ] 网站在本地测试正常显示
- [ ] 深色主题正确应用
- [ ] 代码高亮正常工作
- [ ] 所有链接都能正常访问

## 4. 测试步骤

1. **本地启动**：
   ```bash
   # 使用Python简单服务器
   python -m http.server 8000
   
   # 或使用docsify CLI
   npx docsify serve .
   ```

2. **验证功能**：
   - 访问 `http://localhost:8000`
   - 检查深色主题是否正确应用
   - 测试所有导航链接
   - 验证代码块显示效果
   - 检查移动端响应式效果

3. **浏览器兼容性**：
   - Chrome/Edge最新版
   - Firefox最新版
   - Safari（如果适用）

## 5. 故障排除

### 常见问题及解决方案

**问题1：CSS样式未生效**
- 检查CSS文件路径是否正确
- 确认index.html中CSS链接是否正确
- 清除浏览器缓存

**问题2：侧边栏链接404**
- 检查文件路径是否正确
- 确认文件名大小写是否匹配
- 验证文件是否已正确移动

**问题3：代码高亮异常**
- 确认Prism.js相关脚本已加载
- 检查CSS中token样式是否正确

**问题4：移动端显示异常**
- 检查响应式CSS媒体查询
- 验证viewport meta标签

## 6. 后续优化建议

1. **性能优化**：
   - 添加图片懒加载
   - 优化CSS文件大小
   - 考虑使用CDN加速

2. **功能增强**：
   - 添加暗色/亮色主题切换
   - 集成更多代码语言支持
   - 添加文章阅读进度条

3. **SEO优化**：
   - 添加meta描述
   - 优化页面标题
   - 添加sitemap

## 7. 完成标准

- ✅ 无封面页直接进入首页
- ✅ 深色程序员风格主题
- ✅ 所有原有内容可正常访问
- ✅ 新增的Python学习和AI架构内容已集成
- ✅ 导航结构清晰合理
- ✅ 代码显示专业美观
- ✅ 响应式设计适配移动端