# Python AI智能体开发 - 第2周学习教材

## 目录
1. [环境设置与工具](#环境设置与工具)
2. [Day 1-2: NumPy数据处理](#day-1-2-numpy数据处理)
3. [Day 3-4: Pandas数据分析](#day-3-4-pandas数据分析)
4. [Day 5-7: 数据可视化与Jupyter](#day-5-7-数据可视化与jupyter)
5. [综合项目](#综合项目)
6. [学习资源](#学习资源)

---

## 环境设置与工具

### 1.1 安装数据科学库
```bash
# 激活虚拟环境
source ai_agent_env/bin/activate  # Linux/Mac
# ai_agent_env\Scripts\activate   # Windows

# 安装核心数据科学库
pip install numpy pandas matplotlib seaborn jupyter scikit-learn

# 验证安装
python -c "import numpy as np; print('NumPy version:', np.__version__)"
python -c "import pandas as pd; print('Pandas version:', pd.__version__)"
```

### 1.2 Jupyter Notebook设置
```bash
# 启动Jupyter Notebook
jupyter notebook

# 或使用Jupyter Lab（推荐）
pip install jupyterlab
jupyter lab
```

### 1.3 开发环境配置
- **VS Code扩展**: Python, Jupyter, Pylance
- **Jupyter内核**: 确保使用正确的Python环境
- **代码格式化**: 安装black或autopep8

---

## Day 1-2: NumPy数据处理

### 2.1 NumPy基础概念

#### 为什么需要NumPy？
- **性能**: C语言实现，比Python列表快100倍
- **内存效率**: 连续内存存储，减少内存开销
- **功能丰富**: 数学运算、线性代数、随机数生成
- **AI基础**: 所有深度学习框架的基础

#### 核心概念对比
| Python原生 | NumPy | 优势 |
|------------|-------|------|
| `list` | `ndarray` | 向量化操作、内存连续 |
| 循环计算 | 向量化运算 | 无需显式循环、性能提升 |
| 基本数学 | 丰富数学函数 | 专为科学计算优化 |

### 2.2 NumPy数组创建和操作

#### 数组创建
```python
import numpy as np

# 从Python列表创建
arr1 = np.array([1, 2, 3, 4, 5])
arr2 = np.array([[1, 2, 3], [4, 5, 6]])

# 特殊数组
zeros = np.zeros((3, 4))        # 全零数组
ones = np.ones((2, 3))          # 全一数组
full = np.full((2, 2), 7)       # 填充指定值
eye = np.eye(3)                 # 单位矩阵
arange = np.arange(0, 10, 2)    # 类似range()
linspace = np.linspace(0, 1, 5) # 等间距数组

print("Array 1:", arr1)
print("Array 2:\n", arr2)
print("Zeros:\n", zeros)
print("Arange:", arange)
print("Linspace:", linspace)
```

#### 数组属性和形状操作
```python
# 数组属性
arr = np.array([[1, 2, 3], [4, 5, 6]])
print("Shape:", arr.shape)      # (2, 3)
print("Size:", arr.size)        # 6
print("Dimensions:", arr.ndim)  # 2
print("Data type:", arr.dtype)  # int64

# 形状操作
reshaped = arr.reshape(3, 2)
flattened = arr.flatten()
transposed = arr.T

print("Original:\n", arr)
print("Reshaped:\n", reshaped)
print("Flattened:", flattened)
print("Transposed:\n", transposed)
```

#### 实践练习1：NumPy基础操作
```python
# 练习1.1: 创建智能体观察空间
def create_observation_space(env_size=(84, 84, 4)):
    """创建智能体观察空间（类似Atari游戏）"""
    # 初始化观察空间
    observation_space = np.zeros(env_size, dtype=np.float32)
    return observation_space

# 练习1.2: 数据预处理
def preprocess_observation(observation, normalize=True):
    """预处理观察数据"""
    # 转换为float32
    obs = np.array(observation, dtype=np.float32)
    
    # 归一化到[0, 1]
    if normalize and obs.max() > 1.0:
        obs = obs / 255.0
    
    return obs

# 练习1.3: 批量处理
def batch_process_observations(observations):
    """批量处理观察数据"""
    # 转换为NumPy数组
    batch = np.array(observations, dtype=np.float32)
    
    # 计算批次统计信息
    batch_mean = np.mean(batch, axis=0)
    batch_std = np.std(batch, axis=0)
    
    return {
        "batch": batch,
        "mean": batch_mean,
        "std": batch_std,
        "shape": batch.shape
    }

# 测试
obs_space = create_observation_space()
print("Observation space shape:", obs_space.shape)

# 模拟观察数据
sample_obs = np.random.randint(0, 256, (84, 84, 4))
processed_obs = preprocess_observation(sample_obs)
print("Processed observation range:", processed_obs.min(), "to", processed_obs.max())

# 批量处理
batch_obs = [np.random.randint(0, 256, (84, 84, 4)) for _ in range(32)]
batch_result = batch_process_observations(batch_obs)
print("Batch shape:", batch_result["batch"].shape)
print("Batch mean shape:", batch_result["mean"].shape)
```

### 2.3 NumPy向量化运算

#### 基本运算
```python
# 元素级运算
a = np.array([1, 2, 3, 4])
b = np.array([5, 6, 7, 8])

print("Addition:", a + b)        # [6 8 10 12]
print("Multiplication:", a * b)  # [5 12 21 32]
print("Power:", a ** 2)          # [1 4 9 16]

# 广播机制
matrix = np.array([[1, 2, 3], [4, 5, 6]])
vector = np.array([10, 20, 30])

# 向量加到矩阵的每一行
result = matrix + vector
print("Broadcasting result:\n", result)

# 条件运算
arr = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
even_numbers = arr[arr % 2 == 0]
squared_even = even_numbers ** 2

print("Even numbers:", even_numbers)
print("Squared even:", squared_even)
```

#### 数学和统计函数
```python
arr = np.random.randn(1000)  # 标准正态分布

# 基本统计
print("Mean:", np.mean(arr))
print("Median:", np.median(arr))
print("Standard deviation:", np.std(arr))
print("Variance:", np.var(arr))
print("Min:", np.min(arr))
print("Max:", np.max(arr))

# 聚合函数
matrix = np.random.rand(5, 4)
print("Sum of all elements:", np.sum(matrix))
print("Sum along axis 0:", np.sum(matrix, axis=0))  # 列和
print("Sum along axis 1:", np.sum(matrix, axis=1))  # 行和

# 线性代数
A = np.array([[1, 2], [3, 4]])
B = np.array([[5, 6], [7, 8]])

print("Matrix multiplication:\n", np.dot(A, B))
print("Matrix transpose:\n", A.T)
print("Matrix inverse:\n", np.linalg.inv(A))
```

#### 实践练习2：智能体奖励处理
```python
# 练习2.1: 奖励标准化
def standardize_rewards(rewards, epsilon=1e-8):
    """标准化奖励（类似PPO中的奖励标准化）"""
    rewards = np.array(rewards, dtype=np.float32)
    mean = np.mean(rewards)
    std = np.std(rewards)
    return (rewards - mean) / (std + epsilon)

# 练习2.2: 折扣奖励计算
def compute_discounted_returns(rewards, gamma=0.99):
    """计算折扣回报（用于强化学习）"""
    rewards = np.array(rewards, dtype=np.float32)
    discounted_returns = np.zeros_like(rewards)
    running_return = 0
    
    # 从后往前计算
    for i in reversed(range(len(rewards))):
        running_return = rewards[i] + gamma * running_return
        discounted_returns[i] = running_return
    
    return discounted_returns

# 练习2.3: 优势函数计算
def compute_advantages(rewards, values, gamma=0.99, lam=0.95):
    """计算GAE优势函数"""
    rewards = np.array(rewards, dtype=np.float32)
    values = np.array(values, dtype=np.float32)
    
    # 计算TD误差
    deltas = rewards[:-1] + gamma * values[1:] - values[:-1]
    
    # 计算GAE
    advantages = np.zeros_like(deltas)
    gae = 0
    for i in reversed(range(len(deltas))):
        gae = deltas[i] + gamma * lam * gae
        advantages[i] = gae
    
    return advantages

# 测试
sample_rewards = [1, 0, 0, 0, 10, 0, 0, 0, 5]
standardized = standardize_rewards(sample_rewards)
discounted = compute_discounted_returns(sample_rewards, gamma=0.95)
print("Original rewards:", sample_rewards)
print("Standardized rewards:", standardized)
print("Discounted returns:", discounted)

# 测试优势函数
sample_values = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
advantages = compute_advantages(sample_rewards, sample_values[:-1], gamma=0.95, lam=0.9)
print("Advantages:", advantages)
```

---

## Day 3-4: Pandas数据分析

### 3.1 Pandas核心概念

#### 为什么需要Pandas？
- **数据结构**: Series（一维）和DataFrame（二维表格）
- **数据处理**: 缺失值处理、数据清洗、转换
- **分析功能**: 分组、聚合、时间序列
- **输入输出**: 支持多种格式（CSV, JSON, Excel, SQL）

#### 核心数据结构
```python
import pandas as pd
import numpy as np

# Series - 一维带标签的数组
series = pd.Series([1, 3, 5, np.nan, 6, 8])
print("Series:")
print(series)

# DataFrame - 二维表格
data = {
    'name': ['Alice', 'Bob', 'Charlie', 'Diana'],
    'age': [25, 30, 35, 28],
    'score': [85.5, 92.0, 78.5, 96.0],
    'city': ['Beijing', 'Shanghai', 'Guangzhou', 'Shenzhen']
}
df = pd.DataFrame(data)
print("\nDataFrame:")
print(df)
```

### 3.2 DataFrame基本操作

#### 数据查看和基本信息
```python
# 查看数据
print("First 3 rows:")
print(df.head(3))
print("\nLast 2 rows:")
print(df.tail(2))
print("\nBasic info:")
print(df.info())
print("\nStatistical summary:")
print(df.describe())
print("\nColumn names:")
print(df.columns.tolist())
print("\nData types:")
print(df.dtypes)
```

#### 数据选择和索引
```python
# 列选择
names = df['name']              # Series
subset = df[['name', 'age']]    # DataFrame

# 行选择
first_row = df.iloc[0]          # 按位置
alice_row = df.loc[df['name'] == 'Alice']  # 按条件

# 行列同时选择
specific_data = df.loc[df['age'] > 25, ['name', 'score']]

print("Names:", names.tolist())
print("Subset:\n", subset)
print("First row:\n", first_row)
print("Alice's data:\n", alice_row)
print("Specific data:\n", specific_data)
```

#### 数据修改
```python
# 添加新列
df['grade'] = df['score'].apply(lambda x: 'A' if x >= 90 else 'B' if x >= 80 else 'C')

# 修改现有值
df.loc[df['city'] == 'Beijing', 'city'] = 'BJ'

# 删除列
df_copy = df.drop('grade', axis=1)

print("DataFrame with grade:\n", df)
```

#### 实践练习3：智能体训练日志分析
```python
# 练习3.1: 创建训练日志DataFrame
def create_training_log(episodes, seed=42):
    """创建模拟的智能体训练日志"""
    np.random.seed(seed)
    
    data = {
        'episode': list(range(1, episodes + 1)),
        'total_reward': np.random.normal(100, 20, episodes).cumsum(),
        'steps': np.random.randint(50, 200, episodes),
        'loss': np.random.exponential(1, episodes) * np.exp(-np.arange(episodes) / 50),
        'epsilon': np.maximum(0.01, 1.0 * 0.995 ** np.arange(episodes)),
        'learning_rate': np.full(episodes, 0.001)
    }
    
    return pd.DataFrame(data)

# 练习3.2: 日志分析
def analyze_training_log(df):
    """分析训练日志"""
    analysis = {
        'total_episodes': len(df),
        'final_reward': df['total_reward'].iloc[-1],
        'avg_reward_last_10': df['total_reward'].tail(10).mean(),
        'best_reward': df['total_reward'].max(),
        'best_episode': df['total_reward'].idxmax() + 1,
        'final_loss': df['loss'].iloc[-1],
        'final_epsilon': df['epsilon'].iloc[-1]
    }
    
    return analysis

# 练习3.3: 性能指标计算
def calculate_performance_metrics(df, window_size=10):
    """计算性能指标"""
    # 移动平均
    df['reward_ma'] = df['total_reward'].rolling(window=window_size, min_periods=1).mean()
    df['loss_ma'] = df['loss'].rolling(window=window_size, min_periods=1).mean()
    
    # 收敛检测
    last_rewards = df['total_reward'].tail(window_size)
    convergence = abs(last_rewards.diff().mean()) < 1.0
    
    return df, convergence

# 测试
training_log = create_training_log(100)
print("Training log shape:", training_log.shape)
print("First 5 episodes:\n", training_log.head())

analysis = analyze_training_log(training_log)
print("\nTraining analysis:")
for key, value in analysis.items():
    print(f"{key}: {value:.4f}" if isinstance(value, float) else f"{key}: {value}")

# 计算性能指标
enhanced_log, converged = calculate_performance_metrics(training_log)
print(f"\nConverged: {converged}")
print("Last 5 episodes with moving average:\n", enhanced_log.tail())
```

### 3.3 数据清洗和转换

#### 处理缺失值
```python
# 创建包含缺失值的数据
df_with_nan = pd.DataFrame({
    'A': [1, 2, np.nan, 4],
    'B': [5, np.nan, np.nan, 8],
    'C': [9, 10, 11, 12]
})

print("Original with NaN:\n", df_with_nan)
print("Missing values:\n", df_with_nan.isnull().sum())

# 处理缺失值
df_filled = df_with_nan.fillna(0)           # 填充0
df_dropped = df_with_nan.dropna()           # 删除含NaN的行
df_forward = df_with_nan.fillna(method='ffill')  # 前向填充

print("Filled with 0:\n", df_filled)
print("Dropped NaN:\n", df_dropped)
print("Forward filled:\n", df_forward)
```

#### 数据类型转换和分组
```python
# 数据类型转换
df['age'] = df['age'].astype('int32')
df['score'] = df['score'].astype('float32')

# 分组操作
grouped = df.groupby('grade')['score'].agg(['mean', 'std', 'count'])
print("Grouped by grade:\n", grouped)

# 条件分组
df['age_group'] = pd.cut(df['age'], bins=[0, 25, 30, 100], labels=['Young', 'Middle', 'Senior'])
age_group_stats = df.groupby('age_group')['score'].mean()
print("Age group stats:\n", age_group_stats)
```

#### 实践练习4：多智能体日志分析
```python
# 练习4.1: 创建多智能体日志
def create_multi_agent_logs(agents, episodes_per_agent=50):
    """创建多智能体训练日志"""
    all_logs = []
    
    for agent_id in range(agents):
        np.random.seed(agent_id)
        agent_data = {
            'agent_id': f'Agent_{agent_id}',
            'episode': list(range(1, episodes_per_agent + 1)),
            'reward': np.random.normal(100 + agent_id * 10, 15, episodes_per_agent).cumsum(),
            'algorithm': np.random.choice(['DQN', 'PPO', 'A2C']),
            'env_complexity': np.random.choice(['Easy', 'Medium', 'Hard'])
        }
        agent_df = pd.DataFrame(agent_data)
        all_logs.append(agent_df)
    
    return pd.concat(all_logs, ignore_index=True)

# 练习4.2: 多智能体分析
def analyze_multi_agent_performance(df):
    """分析多智能体性能"""
    # 按智能体分组
    agent_stats = df.groupby('agent_id').agg({
        'reward': ['max', 'mean', 'std'],
        'episode': 'count'
    }).round(2)
    
    # 按算法分组
    algo_stats = df.groupby('algorithm').agg({
        'reward': ['max', 'mean', 'std'],
        'agent_id': 'count'
    }).round(2)
    
    # 最佳智能体
    best_agent = df.loc[df['reward'].idxmax()]
    
    return {
        'agent_stats': agent_stats,
        'algo_stats': algo_stats,
        'best_agent': best_agent
    }

# 测试
multi_logs = create_multi_agent_logs(5)
print("Multi-agent logs shape:", multi_logs.shape)
print("Sample logs:\n", multi_logs.head(10))

analysis = analyze_multi_agent_performance(multi_logs)
print("\nAgent statistics:\n", analysis['agent_stats'])
print("\nAlgorithm statistics:\n", analysis['algo_stats'])
print("\nBest agent:")
print(analysis['best_agent'])
```

---

## Day 5-7: 数据可视化与Jupyter

### 5.1 Matplotlib基础

#### 基本图表类型
```python
import matplotlib.pyplot as plt
import numpy as np

# 设置中文字体（如果需要）
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False    # 用来正常显示负号

# 线图
x = np.linspace(0, 10, 100)
y1 = np.sin(x)
y2 = np.cos(x)

plt.figure(figsize=(10, 6))
plt.plot(x, y1, label='sin(x)', linewidth=2)
plt.plot(x, y2, label='cos(x)', linewidth=2)
plt.xlabel('X')
plt.ylabel('Y')
plt.title('三角函数')
plt.legend()
plt.grid(True)
plt.show()
```

#### 子图和布局
```python
# 创建子图
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# 散点图
axes[0, 0].scatter(np.random.randn(100), np.random.randn(100), alpha=0.6)
axes[0, 0].set_title('散点图')

# 直方图
axes[0, 1].hist(np.random.randn(1000), bins=30, alpha=0.7)
axes[0, 1].set_title('直方图')

# 条形图
categories = ['A', 'B', 'C', 'D']
values = [23, 45, 56, 78]
axes[1, 0].bar(categories, values)
axes[1, 0].set_title('条形图')

# 箱线图
axes[1, 1].boxplot([np.random.randn(100), np.random.randn(100) + 1])
axes[1, 1].set_title('箱线图')

plt.tight_layout()
plt.show()
```

#### 实践练习5：智能体训练可视化
```python
# 练习5.1: 训练曲线可视化
def plot_training_curves(df, title="智能体训练曲线"):
    """绘制训练曲线"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # 奖励曲线
    axes[0, 0].plot(df['episode'], df['total_reward'], 'b-', linewidth=1, alpha=0.7)
    axes[0, 0].plot(df['episode'], df['reward_ma'], 'r-', linewidth=2, label='移动平均')
    axes[0, 0].set_title('总奖励')
    axes[0, 0].set_xlabel('Episode')
    axes[0, 0].set_ylabel('Reward')
    axes[0, 0].legend()
    axes[0, 0].grid(True)
    
    # 损失曲线
    axes[0, 1].plot(df['episode'], df['loss'], 'g-', linewidth=1, alpha=0.7)
    axes[0, 1].plot(df['episode'], df['loss_ma'], 'r-', linewidth=2, label='移动平均')
    axes[0, 1].set_title('损失')
    axes[0, 1].set_xlabel('Episode')
    axes[0, 1].set_ylabel('Loss')
    axes[0, 1].legend()
    axes[0, 1].grid(True)
    
    # Epsilon衰减
    axes[1, 0].plot(df['episode'], df['epsilon'], 'm-', linewidth=2)
    axes[1, 0].set_title('探索率 (Epsilon)')
    axes[1, 0].set_xlabel('Episode')
    axes[1, 0].set_ylabel('Epsilon')
    axes[1, 0].grid(True)
    
    # 步数
    axes[1, 1].plot(df['episode'], df['steps'], 'c-', linewidth=1)
    axes[1, 1].set_title('每轮步数')
    axes[1, 1].set_xlabel('Episode')
    axes[1, 1].set_ylabel('Steps')
    axes[1, 1].grid(True)
    
    plt.suptitle(title, fontsize=16)
    plt.tight_layout()
    plt.show()

# 练习5.2: 多智能体对比
def plot_multi_agent_comparison(df):
    """多智能体性能对比"""
    plt.figure(figsize=(12, 8))
    
    # 按智能体分组绘制
    for agent_id in df['agent_id'].unique():
        agent_data = df[df['agent_id'] == agent_id]
        plt.plot(agent_data['episode'], agent_data['reward'], 
                label=agent_id, linewidth=2, alpha=0.8)
    
    plt.title('多智能体性能对比')
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.legend()
    plt.grid(True)
    plt.show()

# 测试
training_log, _ = calculate_performance_metrics(create_training_log(100))
plot_training_curves(training_log, "DQN智能体训练曲线")

multi_logs = create_multi_agent_logs(3, 50)
plot_multi_agent_comparison(multi_logs)
```

### 5.2 Seaborn高级可视化

#### 统计图表
```python
import seaborn as sns

# 设置样式
sns.set_style("whitegrid")
sns.set_palette("husl")

# 加载示例数据
tips = sns.load_dataset("tips")

# 分布图
plt.figure(figsize=(12, 4))
plt.subplot(1, 3, 1)
sns.histplot(tips['total_bill'], kde=True)
plt.title('账单分布')

plt.subplot(1, 3, 2)
sns.boxplot(x='day', y='total_bill', data=tips)
plt.title('每日账单分布')

plt.subplot(1, 3, 3)
sns.scatterplot(x='total_bill', y='tip', hue='time', data=tips)
plt.title('账单vs小费')
plt.tight_layout()
plt.show()
```

#### 热力图和相关性
```python
# 相关性热力图
correlation_matrix = tips[['total_bill', 'tip', 'size']].corr()
plt.figure(figsize=(8, 6))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0)
plt.title('特征相关性热力图')
plt.show()
```

#### 实践练习6：智能体特征分析
```python
# 练习6.1: 智能体特征相关性分析
def analyze_agent_features(df):
    """分析智能体特征相关性"""
    # 选择数值列
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    correlation_matrix = df[numeric_cols].corr()
    
    # 绘制热力图
    plt.figure(figsize=(10, 8))
    sns.heatmap(correlation_matrix, annot=True, cmap='RdYlBu', center=0, 
                square=True, fmt='.2f')
    plt.title('智能体特征相关性分析')
    plt.show()
    
    return correlation_matrix

# 练习6.2: 算法性能对比
def compare_algorithms(df):
    """算法性能对比"""
    plt.figure(figsize=(12, 6))
    
    # 箱线图比较
    plt.subplot(1, 2, 1)
    sns.boxplot(x='algorithm', y='reward', data=df)
    plt.title('算法奖励分布')
    plt.xticks(rotation=45)
    
    # 小提琴图
    plt.subplot(1, 2, 2)
    sns.violinplot(x='algorithm', y='reward', data=df)
    plt.title('算法奖励分布（小提琴图）')
    plt.xticks(rotation=45)
    
    plt.tight_layout()
    plt.show()

# 测试
multi_logs = create_multi_agent_logs(10, 30)
correlation = analyze_agent_features(multi_logs)
compare_algorithms(multi_logs)
```

### 5.3 Jupyter Notebook最佳实践

#### Jupyter魔法命令
```python
# 在Jupyter中使用
%matplotlib inline    # 内联显示图表
%load_ext autoreload  # 自动重载模块
%autoreload 2

# 性能分析
%timeit np.random.randn(1000)  # 计时
%prun some_function()          # 性能分析

# 文件操作
%ls     # 列出文件
%pwd    # 显示当前目录
%cd     # 切换目录
```

#### 交互式可视化
```python
# 安装交互式库
# pip install ipywidgets plotly

import plotly.express as px
import plotly.graph_objects as go

# 交互式图表
def create_interactive_training_plot(df):
    """创建交互式训练图表"""
    fig = px.line(df, x='episode', y='total_reward', 
                  title='交互式训练曲线',
                  hover_data=['loss', 'epsilon', 'steps'])
    fig.show()

# 如果在Jupyter中运行
# create_interactive_training_plot(training_log)
```

#### 实践练习7：Jupyter智能体分析报告
```python
# 在Jupyter Notebook中创建完整分析报告
def create_agent_analysis_report():
    """
    在Jupyter中创建完整的智能体分析报告
    这个函数展示了如何组织Jupyter Notebook
    """
    print("# 智能体训练分析报告")
    print("## 1. 数据加载和预处理")
    print("```python")
    print("training_log = create_training_log(100)")
    print("training_log, converged = calculate_performance_metrics(training_log)")
    print("```")
    
    print("\n## 2. 基本统计信息")
    print("```python")
    print("print(training_log.describe())")
    print("```")
    
    print("\n## 3. 训练曲线可视化")
    print("```python")
    print("plot_training_curves(training_log)")
    print("```")
    
    print("\n## 4. 收敛性分析")
    print("```python")
    print(f"print('训练是否收敛: {converged}')")
    print("```")
    
    print("\n## 5. 结论和建议")
    print("- 训练曲线显示稳定提升")
    print("- 建议调整学习率以加速收敛")
    print("- 可以尝试不同的探索策略")

# 这个函数主要用于展示Jupyter Notebook的结构
# create_agent_analysis_report()
```

---

## 综合项目

### 智能体训练日志分析系统

创建一个完整的智能体训练日志分析系统，包含以下功能：

1. **日志生成**: 模拟不同智能体的训练日志
2. **数据处理**: 使用Pandas进行数据清洗和转换
3. **统计分析**: 计算关键性能指标
4. **可视化**: 生成训练曲线和对比图表
5. **报告生成**: 自动生成分析报告

#### 项目结构
```
agent_log_analyzer/
├── main.py              # 主程序
├── data_generator.py    # 日志生成器
├── log_analyzer.py      # 日志分析器
├── visualizer.py        # 可视化模块
├── report_generator.py  # 报告生成器
└── notebooks/          # Jupyter笔记本
    └── analysis_demo.ipynb
```

#### 实现步骤
1. 创建`data_generator.py`实现各种日志生成函数
2. 创建`log_analyzer.py`实现数据分析功能
3. 创建`visualizer.py`实现可视化功能
4. 创建`report_generator.py`实现报告生成功能
5. 在Jupyter Notebook中整合所有功能

---

## 学习资源

### 官方文档
- [NumPy官方文档](https://numpy.org/doc/)
- [Pandas官方文档](https://pandas.pydata.org/docs/)
- [Matplotlib官方文档](https://matplotlib.org/stable/contents.html)
- [Seaborn官方文档](https://seaborn.pydata.org/)

### 在线教程
- **免费**:
  - [Kaggle Python课程](https://www.kaggle.com/learn/python)
  - [Kaggle Pandas课程](https://www.kaggle.com/learn/pandas)
  - [Real Python数据科学](https://realpython.com/tutorials/data-science/)
- **付费**:
  - Coursera: Applied Data Science with Python
  - Udemy: Python for Data Science and Machine Learning

### 书籍推荐
- 《Python数据科学手册》- Jake VanderPlas
- 《利用Python进行数据分析》- Wes McKinney (Pandas作者)
- 《Python数据科学实战》- Practical examples

### 实践平台
- [Kaggle](https://www.kaggle.com/) - 数据科学竞赛和数据集
- [Google Colab](https://colab.research.google.com/) - 免费GPU资源
- [Binder](https://mybinder.org/) - 在线Jupyter环境

### 社区资源
- [Stack Overflow](https://stackoverflow.com/questions/tagged/pandas)
- [Reddit r/datascience](https://www.reddit.com/r/datascience/)
- [Python数据科学Discord](https://discord.gg/python)

---

## 学习建议

1. **动手实践**: 每个概念都要在Jupyter Notebook中实践
2. **数据驱动**: 使用真实数据集进行练习
3. **渐进式学习**: 先掌握基础操作，再学习高级功能
4. **项目导向**: 用小项目巩固所学知识
5. **可视化思维**: 学会用图表表达数据洞察

### 每日学习计划
- **上午**: 学习新概念和函数（1-2小时）
- **下午**: 动手练习和项目开发（2-3小时）
- **晚上**: 复习和整理笔记（30分钟）

记住：数据科学的核心是理解数据，而不仅仅是工具的使用。通过第2周的学习，你将掌握Python数据科学的基础，为后续的机器学习和AI智能体开发打下坚实的基础。