# Python AI智能体开发 - 第1周学习教材

## 目录
1. [环境设置](#环境设置)
2. [Day 1-2: Python基础语法](#day-1-2-python基础语法)
3. [Day 3-4: 函数与模块](#day-3-4-函数与模块)
4. [Day 5-7: 面向对象与文件操作](#day-5-7-面向对象与文件操作)
5. [综合项目](#综合项目)
6. [学习资源](#学习资源)

---

## 环境设置

### 1.1 安装Python
- **推荐版本**: Python 3.8+ (建议3.9或3.10)
- **下载地址**: https://www.python.org/downloads/
- **验证安装**: 
  ```bash
  python --version
  # 或
  python3 --version
  ```

### 1.2 设置开发环境
```bash
# 创建项目目录
mkdir python-ai-agent
cd python-ai-agent

# 创建虚拟环境
python -m venv ai_agent_env

# 激活虚拟环境
# Linux/Mac:
source ai_agent_env/bin/activate
# Windows:
ai_agent_env\Scripts\activate

# 升级pip
pip install --upgrade pip

# 安装基础包
pip install jupyter notebook
```

### 1.3 IDE选择
- **推荐**: VS Code + Python插件
- **替代**: PyCharm, Jupyter Notebook
- **在线**: Google Colab (无需本地安装)

---

## Day 1-2: Python基础语法

### 2.1 变量与数据类型

#### 核心数据类型
| Python类型 | Java等效 | 特点 |
|------------|----------|------|
| `int` | `int`/`Integer` | 任意精度整数 |
| `float` | `double` | 64位浮点数 |
| `str` | `String` | 不可变字符串 |
| `bool` | `boolean` | True/False |
| `list` | `ArrayList` | 动态数组 |
| `dict` | `HashMap` | 键值对映射 |
| `tuple` | 不可变数组 | 固定大小，不可修改 |
| `set` | `HashSet` | 无序唯一元素集合 |

#### 实践练习1：基础数据操作
```python
# 练习1.1: 创建学生信息管理系统
def create_student_record(name, age, grades):
    """创建学生记录"""
    return {
        "name": name,
        "age": age,
        "grades": grades,
        "average": sum(grades) / len(grades) if grades else 0
    }

# 测试
student1 = create_student_record("张三", 20, [85, 92, 78, 96])
student2 = create_student_record("李四", 19, [76, 88, 91, 83])

print(f"{student1['name']} 平均分: {student1['average']:.2f}")
print(f"{student2['name']} 平均分: {student2['average']:.2f}")

# 练习1.2: 数据类型转换
numbers_str = "1,2,3,4,5"
numbers_list = [int(x) for x in numbers_str.split(",")]
print("转换后的列表:", numbers_list)
```

### 2.2 控制流

#### 条件语句
```python
# Python的if-elif-else
def get_grade(score):
    if score >= 90:
        return "A"
    elif score >= 80:
        return "B"
    elif score >= 70:
        return "C"
    elif score >= 60:
        return "D"
    else:
        return "F"

# 三元运算符
result = "及格" if score >= 60 else "不及格"
```

#### 循环语句
```python
# for循环 - 遍历任何可迭代对象
fruits = ["apple", "banana", "orange"]
for fruit in fruits:
    print(fruit)

# while循环
count = 0
while count < 5:
    print(f"Count: {count}")
    count += 1

# 列表推导式（Python特有）
squares = [x**2 for x in range(10)]
even_squares = [x**2 for x in range(10) if x % 2 == 0]
```

#### 实践练习2：控制流应用
```python
# 练习2.1: 数字分类器
def classify_numbers(numbers):
    """将数字分类为正数、负数、零"""
    positive = []
    negative = []
    zero = []
    
    for num in numbers:
        if num > 0:
            positive.append(num)
        elif num < 0:
            negative.append(num)
        else:
            zero.append(num)
    
    return {
        "positive": positive,
        "negative": negative,
        "zero": zero,
        "summary": {
            "positive_count": len(positive),
            "negative_count": len(negative),
            "zero_count": len(zero)
        }
    }

# 使用列表推导式重写
def classify_numbers_compact(numbers):
    return {
        "positive": [n for n in numbers if n > 0],
        "negative": [n for n in numbers if n < 0],
        "zero": [n for n in numbers if n == 0],
        "summary": {
            "positive_count": len([n for n in numbers if n > 0]),
            "negative_count": len([n for n in numbers if n < 0]),
            "zero_count": len([n for n in numbers if n == 0])
        }
    }

# 测试
test_data = [-5, -2, 0, 3, 7, -1, 0, 4, 8, -3]
result1 = classify_numbers(test_data)
result2 = classify_numbers_compact(test_data)
print("结果1:", result1)
print("结果2:", result2)
```

---

## Day 3-4: 函数与模块

### 3.1 函数定义

#### 基本函数
```python
def greet(name, greeting="Hello"):
    """基本函数示例"""
    return f"{greeting}, {name}!"

# 调用
print(greet("Alice"))           # Hello, Alice!
print(greet("Bob", "Hi"))       # Hi, Bob!
```

#### 参数类型
```python
# 位置参数
def add(a, b):
    return a + b

# 关键字参数
def create_user(name, age, email=None, active=True):
    return {"name": name, "age": age, "email": email, "active": active}

# 可变参数
def sum_all(*args):
    """接受任意数量的位置参数"""
    return sum(args)

def print_info(**kwargs):
    """接受任意数量的关键字参数"""
    for key, value in kwargs.items():
        print(f"{key}: {value}")

# 混合使用
def flexible_function(required, *args, **kwargs):
    print(f"Required: {required}")
    print(f"Args: {args}")
    print(f"Kwargs: {kwargs}")
```

#### 实践练习3：智能体配置函数
```python
# 练习3.1: 智能体配置生成器
def create_agent_config(name, agent_type="DQN", **custom_params):
    """
    创建智能体配置
    
    Args:
        name: 智能体名称
        agent_type: 智能体类型 ("DQN", "PPO", "A2C")
        **custom_params: 自定义参数
    
    Returns:
        dict: 完整的智能体配置
    """
    # 默认配置
    base_config = {
        "name": name,
        "type": agent_type,
        "learning_rate": 0.001,
        "gamma": 0.99,
        "epsilon": 1.0,
        "epsilon_decay": 0.995,
        "epsilon_min": 0.01,
        "batch_size": 32,
        "memory_size": 10000
    }
    
    # 根据智能体类型调整默认值
    if agent_type == "PPO":
        base_config.update({
            "learning_rate": 0.0003,
            "clip_range": 0.2,
            "n_epochs": 10
        })
    elif agent_type == "A2C":
        base_config.update({
            "learning_rate": 0.0007,
            "n_steps": 5
        })
    
    # 合并自定义参数
    base_config.update(custom_params)
    
    return base_config

# 测试
dqn_agent = create_agent_config("CartPole_DQN", learning_rate=0.0005, memory_size=20000)
ppo_agent = create_agent_config("MountainCar_PPO", agent_type="PPO", clip_range=0.1)

print("DQN Agent Config:")
print(dqn_agent)
print("\nPPO Agent Config:")
print(ppo_agent)
```

### 3.2 模块和包管理

#### 创建模块
```python
# utils.py
"""通用工具函数模块"""

def normalize_data(data):
    """标准化数据到0-1范围"""
    if not data:
        return []
    min_val = min(data)
    max_val = max(data)
    if min_val == max_val:
        return [0.0] * len(data)
    return [(x - min_val) / (max_val - min_val) for x in data]

def calculate_moving_average(data, window_size=3):
    """计算移动平均"""
    if len(data) < window_size:
        return []
    return [sum(data[i:i+window_size]) / window_size 
            for i in range(len(data) - window_size + 1)]

# __init__.py (可选，用于包)
# 如果目录包含__init__.py，Python会将其视为包
```

#### 导入和使用
```python
# main.py
from utils import normalize_data, calculate_moving_average
import utils as ut

# 使用示例
raw_data = [10, 20, 30, 40, 50, 60, 70]
normalized = normalize_data(raw_data)
moving_avg = calculate_moving_average(raw_data, window_size=3)

print("原始数据:", raw_data)
print("标准化数据:", normalized)
print("移动平均:", moving_avg)
```

#### 包管理命令
```bash
# 查看已安装的包
pip list

# 安装包
pip install package_name

# 从requirements.txt安装
pip install -r requirements.txt

# 导出当前环境
pip freeze > requirements.txt

# 创建requirements.txt示例
numpy==1.21.0
pandas==1.3.0
matplotlib==3.4.2
```

---

## Day 5-7: 面向对象与文件操作

### 4.1 面向对象编程

#### 基本类定义
```python
class Agent:
    """智能体基类"""
    
    # 类变量（所有实例共享）
    total_agents = 0
    
    def __init__(self, name, learning_rate=0.001):
        """构造函数"""
        # 实例变量
        self.name = name
        self.learning_rate = learning_rate
        self.experience = []
        self.total_reward = 0
        
        # 更新类变量
        Agent.total_agents += 1
    
    def learn(self, observation, action, reward):
        """学习方法"""
        self.experience.append({
            "observation": observation,
            "action": action,
            "reward": reward
        })
        self.total_reward += reward
    
    def get_stats(self):
        """获取统计信息"""
        return {
            "name": self.name,
            "total_episodes": len(self.experience),
            "total_reward": self.total_reward,
            "avg_reward": self.total_reward / len(self.experience) if self.experience else 0
        }
    
    # 属性装饰器
    @property
    def is_experienced(self):
        """判断是否是有经验的智能体"""
        return len(self.experience) > 100
    
    # 类方法
    @classmethod
    def get_total_count(cls):
        """获取总智能体数量"""
        return cls.total_agents
    
    # 静态方法
    @staticmethod
    def validate_learning_rate(lr):
        """验证学习率是否有效"""
        return 0 < lr <= 1

# 继承示例
class DQNAgent(Agent):
    """DQN智能体"""
    
    def __init__(self, name, learning_rate=0.001, gamma=0.99):
        super().__init__(name, learning_rate)
        self.gamma = gamma
        self.q_table = {}
    
    def update_q_value(self, state, action, reward, next_state):
        """更新Q值"""
        current_q = self.q_table.get((state, action), 0)
        max_next_q = max([self.q_table.get((next_state, a), 0) 
                         for a in range(4)], default=0)
        new_q = current_q + self.learning_rate * (
            reward + self.gamma * max_next_q - current_q
        )
        self.q_table[(state, action)] = new_q
```

#### 实践练习4：OOP应用
```python
# 练习4.1: 数据处理器类
class DataProcessor:
    """数据处理器类"""
    
    def __init__(self, data_source=None):
        self.data_source = data_source
        self.raw_data = None
        self.processed_data = None
        self.processing_history = []
    
    def load_data(self):
        """加载数据"""
        if self.data_source is None:
            raise ValueError("No data source provided")
        
        if isinstance(self.data_source, str):
            # 假设是文件路径
            print(f"Loading data from: {self.data_source}")
            # 这里应该是实际的文件读取逻辑
            self.raw_data = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        elif isinstance(self.data_source, (list, tuple)):
            self.raw_data = list(self.data_source)
        else:
            raise TypeError("Unsupported data source type")
    
    def process(self, operation="normalize", **kwargs):
        """处理数据"""
        if self.raw_data is None:
            self.load_data()
        
        original_data = self.raw_data.copy()
        
        if operation == "normalize":
            processed = self._normalize(original_data)
        elif operation == "scale":
            factor = kwargs.get("factor", 1.0)
            processed = self._scale(original_data, factor)
        elif operation == "filter":
            threshold = kwargs.get("threshold", 0)
            processed = self._filter_above(original_data, threshold)
        else:
            raise ValueError(f"Unsupported operation: {operation}")
        
        # 记录处理历史
        self.processing_history.append({
            "operation": operation,
            "input_size": len(original_data),
            "output_size": len(processed),
            "params": kwargs
        })
        
        self.processed_data = processed
        return processed
    
    def _normalize(self, data):
        """内部方法：标准化"""
        if not data:
            return []
        min_val, max_val = min(data), max(data)
        if min_val == max_val:
            return [0.0] * len(data)
        return [(x - min_val) / (max_val - min_val) for x in data]
    
    def _scale(self, data, factor):
        """内部方法：缩放"""
        return [x * factor for x in data]
    
    def _filter_above(self, data, threshold):
        """内部方法：过滤"""
        return [x for x in data if x > threshold]
    
    def get_processing_summary(self):
        """获取处理摘要"""
        return {
            "total_operations": len(self.processing_history),
            "operations": self.processing_history,
            "final_data_size": len(self.processed_data) if self.processed_data else 0
        }

# 测试
processor = DataProcessor([10, 20, 30, 40, 50])
normalized = processor.process("normalize")
scaled = processor.process("scale", factor=2.0)
filtered = processor.process("filter", threshold=25)

print("处理摘要:", processor.get_processing_summary())
```

### 4.2 文件操作

#### 基本文件操作
```python
# 写入文件
def save_to_file(data, filename, mode='w'):
    """保存数据到文件"""
    with open(filename, mode, encoding='utf-8') as f:
        if isinstance(data, str):
            f.write(data)
        elif isinstance(data, (list, dict)):
            import json
            json.dump(data, f, indent=2, ensure_ascii=False)
        else:
            f.write(str(data))

# 读取文件
def load_from_file(filename):
    """从文件加载数据"""
    with open(filename, 'r', encoding='utf-8') as f:
        content = f.read()
        # 尝试解析为JSON
        try:
            import json
            return json.loads(content)
        except json.JSONDecodeError:
            return content

# 异常处理
def safe_file_operation(filename, operation, data=None):
    """安全的文件操作"""
    try:
        if operation == "read":
            return load_from_file(filename)
        elif operation == "write":
            save_to_file(data, filename)
            return True
    except FileNotFoundError:
        print(f"文件未找到: {filename}")
        return None
    except PermissionError:
        print(f"权限不足: {filename}")
        return None
    except Exception as e:
        print(f"文件操作错误: {e}")
        return None
```

#### 实践练习5：配置文件管理
```python
# config_manager.py
import json
import os
from datetime import datetime

class ConfigManager:
    """配置管理器"""
    
    def __init__(self, config_dir="configs"):
        self.config_dir = config_dir
        os.makedirs(config_dir, exist_ok=True)
    
    def save_config(self, config, name):
        """保存配置"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{name}_{timestamp}.json"
        filepath = os.path.join(self.config_dir, filename)
        
        # 添加元数据
        full_config = {
            "metadata": {
                "name": name,
                "created_at": timestamp,
                "version": "1.0"
            },
            "config": config
        }
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(full_config, f, indent=2, ensure_ascii=False)
        
        print(f"配置已保存: {filepath}")
        return filepath
    
    def load_latest_config(self, name):
        """加载最新配置"""
        config_files = []
        for file in os.listdir(self.config_dir):
            if file.startswith(name) and file.endswith('.json'):
                config_files.append(file)
        
        if not config_files:
            return None
        
        # 按时间戳排序，获取最新的
        latest_file = max(config_files)
        filepath = os.path.join(self.config_dir, latest_file)
        
        with open(filepath, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    def list_configs(self, name=None):
        """列出所有配置"""
        configs = []
        for file in os.listdir(self.config_dir):
            if file.endswith('.json'):
                if name is None or file.startswith(name):
                    configs.append(file)
        return sorted(configs)

# 使用示例
if __name__ == "__main__":
    # 创建配置管理器
    cm = ConfigManager()
    
    # 保存智能体配置
    agent_config = {
        "agent_type": "DQN",
        "learning_rate": 0.001,
        "gamma": 0.99,
        "epsilon": 1.0,
        "memory_size": 10000
    }
    
    cm.save_config(agent_config, "cartpole_agent")
    
    # 加载最新配置
    latest_config = cm.load_latest_config("cartpole_agent")
    if latest_config:
        print("最新配置:", latest_config["config"])
    
    # 列出所有配置
    all_configs = cm.list_configs()
    print("所有配置文件:", all_configs)
```

---

## 综合项目

### 智能体配置管理系统

创建一个完整的智能体配置管理系统，包含以下功能：

1. **智能体配置创建**
   - 支持多种智能体类型（DQN, PPO, A2C）
   - 可自定义参数
   - 参数验证

2. **配置持久化**
   - 保存到JSON文件
   - 支持版本管理
   - 加载历史配置

3. **配置管理**
   - 列出所有配置
   - 比较配置差异
   - 备份和恢复

#### 项目结构
```
agent_config_manager/
├── main.py              # 主程序
├── agent_config.py      # 智能体配置类
├── config_manager.py    # 配置管理器
├── utils.py            # 工具函数
└── configs/            # 配置文件目录
```

#### 实现步骤
1. 创建`agent_config.py`实现`AgentConfig`类
2. 创建`config_manager.py`实现配置管理功能
3. 在`main.py`中整合所有功能
4. 添加命令行界面或简单菜单

---

## 学习资源

### 官方文档
- [Python官方教程](https://docs.python.org/zh-cn/3/tutorial/)
- [Python标准库](https://docs.python.org/zh-cn/3/library/)

### 在线学习平台
- **免费**: 
  - [Python官方文档](https://docs.python.org/3/tutorial/)
  - [Real Python](https://realpython.com/)
  - [W3Schools Python](https://www.w3schools.com/python/)
- **付费**:
  - Coursera: Python for Everybody
  - Udemy: Complete Python Bootcamp

### 书籍推荐
- 《Python Crash Course》- 适合初学者
- 《Effective Python》- 适合有经验的程序员
- 《Fluent Python》- 深入理解Python特性

### 实践平台
- [LeetCode](https://leetcode.com/) - 算法练习
- [Kaggle](https://www.kaggle.com/) - 数据科学项目
- [Google Colab](https://colab.research.google.com/) - 在线Jupyter环境

### 社区资源
- [Stack Overflow](https://stackoverflow.com/questions/tagged/python)
- [Reddit r/Python](https://www.reddit.com/r/Python/)
- [Python Discord](https://discord.gg/python)

---

## 学习建议

1. **动手实践**: 每个概念都要写代码验证
2. **对比学习**: 时刻思考"这在Java中是怎么做的"
3. **渐进式学习**: 先掌握基础，再深入高级特性
4. **项目驱动**: 用小项目巩固所学知识
5. **代码规范**: 遵循PEP 8编码规范

### 每日学习计划
- **上午**: 学习新概念（1-2小时）
- **下午**: 动手练习（2-3小时）
- **晚上**: 复习和总结（30分钟）

记住：作为Java程序员，你已经掌握了编程的核心思想，现在只是学习一种新的表达方式。Python的简洁性会让你的开发效率大大提升！