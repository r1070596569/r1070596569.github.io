# Python AI智能体开发 - 第4周学习教材

## 目录
1. [环境设置与深度学习基础](#环境设置与深度学习基础)
2. [Day 1-2: TensorFlow/PyTorch基础](#day-1-2-tensorflowpytorch基础)
3. [Day 3-4: 神经网络与自然语言处理](#day-3-4-神经网络与自然语言处理)
4. [Day 5-7: 强化学习与AI智能体](#day-5-7-强化学习与ai智能体)
5. [综合项目](#综合项目)
6. [学习资源](#学习资源)

---

## 环境设置与深度学习基础

### 1.1 安装深度学习框架

#### TensorFlow安装
```bash
# 激活虚拟环境
source ai_agent_env/bin/activate  # Linux/Mac
# ai_agent_env\Scripts\activate   # Windows

# 安装TensorFlow（CPU版本）
pip install tensorflow

# 如果有GPU支持（需要CUDA）
# pip install tensorflow-gpu

# 验证安装
python -c "import tensorflow as tf; print('TensorFlow version:', tf.__version__)"
```

#### PyTorch安装
```bash
# 安装PyTorch（CPU版本）
pip install torch torchvision torchaudio

# GPU版本（需要CUDA）
# pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# 验证安装
python -c "import torch; print('PyTorch version:', torch.__version__)"
```

### 1.2 深度学习基础概念

#### 神经网络基础
- **神经元**: 基本计算单元，接收输入、加权求和、激活函数
- **层**: 神经元的集合，全连接层、卷积层、循环层
- **激活函数**: 引入非线性（ReLU, Sigmoid, Tanh）
- **损失函数**: 衡量预测误差（MSE, Cross-Entropy）
- **优化器**: 更新权重（SGD, Adam, RMSprop）

#### 框架选择对比
| 特性 | TensorFlow | PyTorch |
|------|------------|---------|
| **易用性** | 较复杂 | 更直观 |
| **调试** | 静态图较难 | 动态图易调试 |
| **部署** | 生产部署优秀 | 部署在改进 |
| **研究** | 广泛使用 | 学术界首选 |
| **社区** | Google支持 | Facebook支持 |

**建议**: 初学者从PyTorch开始，更接近Python原生体验

### 1.3 张量操作基础

#### NumPy vs PyTorch vs TensorFlow
```python
import numpy as np
import torch
import tensorflow as tf

# 创建数组/张量
np_array = np.array([1, 2, 3, 4])
torch_tensor = torch.tensor([1, 2, 3, 4])
tf_tensor = tf.constant([1, 2, 3, 4])

print("NumPy:", np_array)
print("PyTorch:", torch_tensor)
print("TensorFlow:", tf_tensor)

# 基本运算
print("\nAddition:")
print("NumPy:", np_array + 2)
print("PyTorch:", torch_tensor + 2)
print("TensorFlow:", tf_tensor + 2)

# 形状操作
print("\nReshape:")
np_reshaped = np_array.reshape(2, 2)
torch_reshaped = torch_tensor.reshape(2, 2)
tf_reshaped = tf.reshape(tf_tensor, (2, 2))

print("NumPy reshaped:\n", np_reshaped)
print("PyTorch reshaped:\n", torch_reshaped)
print("TensorFlow reshaped:\n", tf_reshaped)
```

---

## Day 1-2: TensorFlow/PyTorch基础

### 2.1 PyTorch基础

#### 张量操作
```python
import torch
import torch.nn as nn
import torch.optim as optim

# 创建张量
x = torch.tensor([1.0, 2.0, 3.0, 4.0], requires_grad=True)
y = torch.tensor([2.0, 4.0, 6.0, 8.0])

# 基本运算
z = x * y + torch.sin(x)
print("Result:", z)

# 自动微分
z.sum().backward()
print("Gradients:", x.grad)

# 张量属性
print(f"Shape: {x.shape}")
print(f"Data type: {x.dtype}")
print(f"Device: {x.device}")

# GPU支持（如果有）
if torch.cuda.is_available():
    x_gpu = x.cuda()
    print(f"GPU tensor device: {x_gpu.device}")
```

#### 简单神经网络
```python
class SimpleNet(nn.Module):
    """简单神经网络"""
    
    def __init__(self, input_size, hidden_size, output_size):
        super(SimpleNet, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

# 创建网络
net = SimpleNet(input_size=4, hidden_size=10, output_size=2)
print("Network architecture:")
print(net)

# 前向传播
input_data = torch.randn(1, 4)
output = net(input_data)
print(f"Input shape: {input_data.shape}")
print(f"Output shape: {output.shape}")
print(f"Output: {output}")

# 训练循环
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(net.parameters(), lr=0.01)

# 模拟训练数据
for epoch in range(100):
    optimizer.zero_grad()
    
    # 前向传播
    outputs = net(input_data)
    loss = criterion(outputs, torch.tensor([1]))  # 假设真实标签是1
    
    # 反向传播
    loss.backward()
    optimizer.step()
    
    if epoch % 20 == 0:
        print(f"Epoch {epoch}, Loss: {loss.item():.4f}")
```

#### 实践练习1：PyTorch回归任务
```python
# 练习1.1: 使用PyTorch进行回归
import matplotlib.pyplot as plt

def pytorch_regression_example():
    """PyTorch回归示例"""
    # 生成合成数据
    torch.manual_seed(42)
    X = torch.linspace(0, 10, 100).reshape(-1, 1)
    y_true = 2 * X + 1 + torch.randn(100, 1) * 0.5  # y = 2x + 1 + noise
    
    # 定义模型
    class LinearRegressionModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = nn.Linear(1, 1)
        
        def forward(self, x):
            return self.linear(x)
    
    model = LinearRegressionModel()
    criterion = nn.MSELoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01)
    
    # 训练
    losses = []
    for epoch in range(1000):
        optimizer.zero_grad()
        y_pred = model(X)
        loss = criterion(y_pred, y_true)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
        
        if epoch % 200 == 0:
            print(f"Epoch {epoch}, Loss: {loss.item():.4f}")
    
    # 评估
    with torch.no_grad():
        final_pred = model(X)
        final_loss = criterion(final_pred, y_true)
        print(f"Final loss: {final_loss.item():.4f}")
        
        # 获取权重和偏置
        weight = model.linear.weight.item()
        bias = model.linear.bias.item()
        print(f"Learned parameters: weight={weight:.2f}, bias={bias:.2f}")
        print(f"True parameters: weight=2.00, bias=1.00")
    
    # 可视化
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(losses)
    plt.title('Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    
    plt.subplot(1, 2, 2)
    plt.scatter(X.numpy(), y_true.numpy(), alpha=0.6, label='True data')
    plt.plot(X.numpy(), final_pred.numpy(), 'r-', label='Predicted', linewidth=2)
    plt.title('Regression Results')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.legend()
    
    plt.tight_layout()
    plt.show()
    
    return model, losses

# 运行回归示例
regression_model, loss_history = pytorch_regression_example()
```

### 2.2 TensorFlow基础

#### 张量和自动微分
```python
import tensorflow as tf
import numpy as np

# 创建张量
x = tf.Variable([1.0, 2.0, 3.0, 4.0])
y = tf.constant([2.0, 4.0, 6.0, 8.0])

# 自动微分
with tf.GradientTape() as tape:
    z = x * y + tf.sin(x)

gradients = tape.gradient(z, x)
print("Result:", z)
print("Gradients:", gradients)

# 张量操作
print(f"Shape: {x.shape}")
print(f"Data type: {x.dtype}")
print(f"Device: {x.device}")

# GPU支持（如果有）
if tf.config.list_physical_devices('GPU'):
    with tf.device('/GPU:0'):
        x_gpu = tf.Variable([1.0, 2.0, 3.0, 4.0])
        print(f"GPU tensor device: {x_gpu.device}")
```

#### Keras高级API
```python
from tensorflow import keras
from tensorflow.keras import layers

# 使用Sequential API
model = keras.Sequential([
    layers.Dense(10, activation='relu', input_shape=(4,)),
    layers.Dense(2, activation='softmax')
])

print("Model summary:")
model.summary()

# 编译模型
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# 模拟训练数据
X_train = np.random.randn(100, 4)
y_train = np.random.randint(0, 2, 100)

# 训练
history = model.fit(X_train, y_train, epochs=10, verbose=1)

# 预测
X_test = np.random.randn(10, 4)
predictions = model.predict(X_test)
print(f"Predictions shape: {predictions.shape}")
print(f"First prediction: {predictions[0]}")
```

#### 实践练习2：TensorFlow分类任务
```python
# 练习2.1: 使用TensorFlow进行分类
def tensorflow_classification_example():
    """TensorFlow分类示例"""
    # 加载鸢尾花数据集
    from sklearn.datasets import load_iris
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    
    iris = load_iris()
    X, y = iris.data, iris.target
    
    # 划分数据集
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # 标准化
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # 创建模型
    model = keras.Sequential([
        layers.Dense(16, activation='relu', input_shape=(4,)),
        layers.Dropout(0.2),
        layers.Dense(8, activation='relu'),
        layers.Dropout(0.2),
        layers.Dense(3, activation='softmax')
    ])
    
    # 编译模型
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    # 训练
    history = model.fit(
        X_train_scaled, y_train,
        epochs=100,
        batch_size=16,
        validation_split=0.2,
        verbose=0
    )
    
    # 评估
    test_loss, test_accuracy = model.evaluate(X_test_scaled, y_test, verbose=0)
    print(f"Test accuracy: {test_accuracy:.4f}")
    
    # 预测
    predictions = model.predict(X_test_scaled)
    predicted_classes = np.argmax(predictions, axis=1)
    
    print("\nSample predictions:")
    for i in range(5):
        true_class = iris.target_names[y_test[i]]
        pred_class = iris.target_names[predicted_classes[i]]
        confidence = predictions[i][predicted_classes[i]]
        print(f"True: {true_class}, Predicted: {pred_class}, Confidence: {confidence:.4f}")
    
    # 可视化训练历史
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    
    plt.tight_layout()
    plt.show()
    
    return model, history

# 运行分类示例
classification_model, training_history = tensorflow_classification_example()
```

### 2.3 框架对比实践

#### 相同任务的不同实现
```python
# 比较PyTorch和TensorFlow实现相同任务
def compare_frameworks():
    """比较两个框架"""
    # 生成相同的数据
    np.random.seed(42)
    torch.manual_seed(42)
    tf.random.set_seed(42)
    
    X = np.random.randn(100, 4)
    y = np.random.randint(0, 2, 100)
    
    # PyTorch实现
    class TorchModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(4, 8),
                nn.ReLU(),
                nn.Linear(8, 2)
            )
        
        def forward(self, x):
            return self.net(x)
    
    torch_model = TorchModel()
    torch_criterion = nn.CrossEntropyLoss()
    torch_optimizer = optim.Adam(torch_model.parameters(), lr=0.01)
    
    torch_X = torch.FloatTensor(X)
    torch_y = torch.LongTensor(y)
    
    torch_losses = []
    for epoch in range(50):
        torch_optimizer.zero_grad()
        outputs = torch_model(torch_X)
        loss = torch_criterion(outputs, torch_y)
        loss.backward()
        torch_optimizer.step()
        torch_losses.append(loss.item())
    
    # TensorFlow实现
    tf_model = keras.Sequential([
        layers.Dense(8, activation='relu', input_shape=(4,)),
        layers.Dense(2)
    ])
    
    tf_model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    tf_history = tf_model.fit(X, y, epochs=50, verbose=0)
    
    # 比较结果
    print("Framework Comparison:")
    print(f"PyTorch final loss: {torch_losses[-1]:.4f}")
    print(f"TensorFlow final loss: {tf_history.history['loss'][-1]:.4f}")
    
    # 可视化
    plt.figure(figsize=(10, 6))
    plt.plot(torch_losses, label='PyTorch')
    plt.plot(tf_history.history['loss'], label='TensorFlow')
    plt.title('Framework Comparison - Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.show()

# 运行框架比较
compare_frameworks()
```

---

## Day 3-4: 神经网络与自然语言处理

### 3.1 神经网络架构

#### 卷积神经网络（CNN）
```python
# PyTorch CNN示例
class SimpleCNN(nn.Module):
    """简单CNN用于图像分类"""
    
    def __init__(self, num_classes=10):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 8 * 8, 128)
        self.fc2 = nn.Linear(128, num_classes)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)
    
    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = x.view(-1, 64 * 8 * 8)
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

# 创建CNN模型
cnn_model = SimpleCNN(num_classes=10)
print("CNN Architecture:")
print(cnn_model)
```

#### 循环神经网络（RNN）
```python
# PyTorch RNN示例
class SimpleRNN(nn.Module):
    """简单RNN用于序列数据"""
    
    def __init__(self, input_size, hidden_size, output_size, num_layers=1):
        super(SimpleRNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.rnn = nn.RNN(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        # 初始化隐藏状态
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)
        
        # 前向传播RNN
        out, _ = self.rnn(x, h0)
        
        # 取最后一个时间步的输出
        out = self.fc(out[:, -1, :])
        return out

# 创建RNN模型
rnn_model = SimpleRNN(input_size=10, hidden_size=20, output_size=2)
print("RNN Architecture:")
print(rnn_model)
```

### 3.2 自然语言处理基础

#### 文本预处理
```python
import re
import nltk
from collections import Counter

# 下载必要的NLTK数据
# nltk.download('punkt')
# nltk.download('stopwords')

def preprocess_text(text):
    """文本预处理"""
    # 转换为小写
    text = text.lower()
    
    # 移除特殊字符
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    
    # 分词
    tokens = nltk.word_tokenize(text)
    
    # 移除停用词
    stop_words = set(nltk.corpus.stopwords.words('english'))
    tokens = [token for token in tokens if token not in stop_words and len(token) > 2]
    
    return tokens

# 示例文本
sample_text = "Hello! This is a sample text for NLP preprocessing. It contains punctuation, stopwords, and other noise."
processed_tokens = preprocess_text(sample_text)
print("Original text:", sample_text)
print("Processed tokens:", processed_tokens)
```

#### 词嵌入
```python
# 简单的词袋模型
def create_bag_of_words(documents):
    """创建词袋模型"""
    # 收集所有词汇
    all_words = []
    for doc in documents:
        tokens = preprocess_text(doc)
        all_words.extend(tokens)
    
    # 创建词汇表
    word_freq = Counter(all_words)
    vocab = sorted(word_freq.keys())
    word_to_idx = {word: idx for idx, word in enumerate(vocab)}
    
    # 创建词袋矩阵
    bow_matrix = []
    for doc in documents:
        tokens = preprocess_text(doc)
        doc_vector = [0] * len(vocab)
        for token in tokens:
            if token in word_to_idx:
                doc_vector[word_to_idx[token]] += 1
        bow_matrix.append(doc_vector)
    
    return np.array(bow_matrix), vocab, word_to_idx

# 示例文档
documents = [
    "I love machine learning and artificial intelligence",
    "Deep learning is a subset of machine learning",
    "Natural language processing is fascinating",
    "I enjoy working with neural networks"
]

bow_matrix, vocabulary, word_index = create_bag_of_words(documents)
print("Vocabulary size:", len(vocabulary))
print("Bag of words matrix shape:", bow_matrix.shape)
print("First document vector:", bow_matrix[0])
```

#### 实践练习3：情感分析
```python
# 练习3.1: 简单情感分析
def sentiment_analysis_example():
    """情感分析示例"""
    # 创建模拟数据
    positive_texts = [
        "I love this movie! It's amazing and wonderful.",
        "Great product, highly recommended!",
        "Excellent service and quality.",
        "Fantastic experience, will come again.",
        "Outstanding performance and results."
    ]
    
    negative_texts = [
        "This movie is terrible and boring.",
        "Poor quality product, not recommended.",
        "Bad service and disappointing experience.",
        "Worst experience ever, avoid this place.",
        "Terrible performance and waste of money."
    ]
    
    texts = positive_texts + negative_texts
    labels = [1] * len(positive_texts) + [0] * len(negative_texts)
    
    # 创建词袋特征
    bow_matrix, vocab, word_to_idx = create_bag_of_words(texts)
    
    # 划分训练测试集
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(
        bow_matrix, labels, test_size=0.3, random_state=42
    )
    
    # 训练分类器
    from sklearn.linear_model import LogisticRegression
    classifier = LogisticRegression(random_state=42)
    classifier.fit(X_train, y_train)
    
    # 评估
    y_pred = classifier.predict(X_test)
    accuracy = np.mean(y_pred == y_test)
    
    print("Sentiment Analysis Results:")
    print(f"Vocabulary size: {len(vocab)}")
    print(f"Accuracy: {accuracy:.4f}")
    
    # 测试新文本
    test_texts = [
        "This is an awesome product!",
        "I hate this terrible service."
    ]
    
    for text in test_texts:
        tokens = preprocess_text(text)
        test_vector = [0] * len(vocab)
        for token in tokens:
            if token in word_to_idx:
                test_vector[word_to_idx[token]] += 1
        
        prediction = classifier.predict([test_vector])[0]
        probability = classifier.predict_proba([test_vector])[0]
        
        sentiment = "Positive" if prediction == 1 else "Negative"
        confidence = probability[prediction]
        
        print(f"\nText: '{text}'")
        print(f"Prediction: {sentiment} (confidence: {confidence:.4f})")
    
    return classifier, vocab, word_to_idx

# 运行情感分析
sentiment_model, vocab, word_idx = sentiment_analysis_example()
```

### 3.3 使用预训练模型

#### Hugging Face Transformers
```python
# 安装transformers库
# pip install transformers

from transformers import pipeline

# 情感分析管道
sentiment_pipeline = pipeline("sentiment-analysis")

# 测试
texts = [
    "I love this new AI technology!",
    "This is the worst product I've ever used.",
    "The weather is okay today."
]

results = sentiment_pipeline(texts)
for text, result in zip(texts, results):
    print(f"Text: {text}")
    print(f"Sentiment: {result['label']}, Score: {result['score']:.4f}\n")
```

#### 实践练习4：文本生成
```python
# 练习4.1: 使用预训练模型进行文本生成
def text_generation_example():
    """文本生成示例"""
    try:
        from transformers import pipeline
        
        # 创建文本生成管道
        generator = pipeline("text-generation", model="gpt2")
        
        # 生成文本
        prompt = "Artificial intelligence is"
        generated = generator(prompt, max_length=50, num_return_sequences=2)
        
        print("Text Generation Results:")
        print(f"Prompt: {prompt}")
        for i, result in enumerate(generated, 1):
            print(f"Generated {i}: {result['generated_text']}\n")
            
    except ImportError:
        print("Transformers library not installed. Install with: pip install transformers")
    except Exception as e:
        print(f"Error loading model: {e}")
        print("This might be due to limited memory or internet connectivity.")

# 运行文本生成（注意：可能需要较多内存）
# text_generation_example()
```

---

## Day 5-7: 强化学习与AI智能体

### 5.1 强化学习基础

#### 强化学习核心概念
- **智能体（Agent）**: 学习者和决策者
- **环境（Environment）**: 智能体交互的世界
- **状态（State）**: 环境的当前情况
- **动作（Action）**: 智能体可以执行的操作
- **奖励（Reward）**: 执行动作后获得的反馈
- **策略（Policy）**: 状态到动作的映射
- **价值函数（Value Function）**: 状态或动作的长期价值

#### 马尔可夫决策过程（MDP）
```python
class SimpleMDP:
    """简单马尔可夫决策过程环境"""
    
    def __init__(self):
        self.states = ['A', 'B', 'C', 'D', 'E']
        self.actions = ['left', 'right']
        self.current_state = 'C'
        self.terminal_states = ['A', 'E']
    
    def reset(self):
        """重置环境"""
        self.current_state = 'C'
        return self.current_state
    
    def step(self, action):
        """执行动作"""
        if self.current_state in self.terminal_states:
            return self.current_state, 0, True, {}
        
        # 状态转移
        if action == 'left':
            if self.current_state == 'B':
                self.current_state = 'A'
            elif self.current_state == 'C':
                self.current_state = 'B'
            elif self.current_state == 'D':
                self.current_state = 'C'
        elif action == 'right':
            if self.current_state == 'B':
                self.current_state = 'C'
            elif self.current_state == 'C':
                self.current_state = 'D'
            elif self.current_state == 'D':
                self.current_state = 'E'
        
        # 奖励
        reward = 0
        done = False
        if self.current_state == 'A':
            reward = -1
            done = True
        elif self.current_state == 'E':
            reward = 1
            done = True
        
        return self.current_state, reward, done, {}

# 测试MDP环境
env = SimpleMDP()
print("Initial state:", env.reset())

actions = ['right', 'right', 'left', 'right', 'right']
for action in actions:
    state, reward, done, info = env.step(action)
    print(f"Action: {action}, State: {state}, Reward: {reward}, Done: {done}")
    if done:
        break
```

### 5.2 Q-Learning算法

#### 简单Q-Learning实现
```python
import random

class QLearningAgent:
    """Q-Learning智能体"""
    
    def __init__(self, actions, learning_rate=0.1, discount_factor=0.9, epsilon=0.1):
        self.actions = actions
        self.lr = learning_rate
        self.gamma = discount_factor
        self.epsilon = epsilon
        self.q_table = {}
    
    def get_q_value(self, state, action):
        """获取Q值"""
        return self.q_table.get((state, action), 0.0)
    
    def update_q_value(self, state, action, reward, next_state):
        """更新Q值"""
        current_q = self.get_q_value(state, action)
        max_next_q = max([self.get_q_value(next_state, a) for a in self.actions])
        new_q = current_q + self.lr * (reward + self.gamma * max_next_q - current_q)
        self.q_table[(state, action)] = new_q
    
    def choose_action(self, state):
        """选择动作"""
        if random.random() < self.epsilon:
            # 探索：随机选择动作
            return random.choice(self.actions)
        else:
            # 利用：选择最佳动作
            q_values = [self.get_q_value(state, a) for a in self.actions]
            max_q = max(q_values)
            best_actions = [a for a, q in zip(self.actions, q_values) if q == max_q]
            return random.choice(best_actions)

# 训练Q-Learning智能体
def train_q_learning_agent():
    """训练Q-Learning智能体"""
    env = SimpleMDP()
    agent = QLearningAgent(actions=['left', 'right'])
    
    episodes = 1000
    rewards_per_episode = []
    
    for episode in range(episodes):
        state = env.reset()
        total_reward = 0
        done = False
        
        while not done:
            action = agent.choose_action(state)
            next_state, reward, done, _ = env.step(action)
            agent.update_q_value(state, action, reward, next_state)
            state = next_state
            total_reward += reward
        
        rewards_per_episode.append(total_reward)
        
        if episode % 100 == 0:
            avg_reward = np.mean(rewards_per_episode[-100:])
            print(f"Episode {episode}, Average Reward: {avg_reward:.2f}")
    
    return agent, rewards_per_episode

# 训练智能体
q_agent, q_rewards = train_q_learning_agent()

# 测试训练后的智能体
print("\nTesting trained agent:")
env = SimpleMDP()
state = env.reset()
total_reward = 0
steps = 0

while steps < 10:  # 限制最大步数
    action = q_agent.choose_action(state)
    state, reward, done, _ = env.step(action)
    total_reward += reward
    steps += 1
    print(f"Step {steps}: Action={action}, State={state}, Reward={reward}")
    if done:
        break

print(f"Total reward: {total_reward}")
```

#### 实践练习5：Grid World智能体
```python
# 练习5.1: Grid World环境
class GridWorld:
    """网格世界环境"""
    
    def __init__(self, width=5, height=5):
        self.width = width
        self.height = height
        self.start_pos = (0, 0)
        self.goal_pos = (width-1, height-1)
        self.obstacles = [(2, 2), (3, 3)]
        self.reset()
    
    def reset(self):
        """重置环境"""
        self.agent_pos = self.start_pos
        self.steps = 0
        return self._get_state()
    
    def _get_state(self):
        """获取当前状态"""
        return f"({self.agent_pos[0]},{self.agent_pos[1]})"
    
    def _is_valid_position(self, pos):
        """检查位置是否有效"""
        x, y = pos
        if x < 0 or x >= self.width or y < 0 or y >= self.height:
            return False
        if pos in self.obstacles:
            return False
        return True
    
    def step(self, action):
        """执行动作"""
        self.steps += 1
        
        # 动作映射
        action_map = {
            'up': (0, -1),
            'down': (0, 1),
            'left': (-1, 0),
            'right': (1, 0)
        }
        
        dx, dy = action_map.get(action, (0, 0))
        new_pos = (self.agent_pos[0] + dx, self.agent_pos[1] + dy)
        
        # 检查新位置是否有效
        if self._is_valid_position(new_pos):
            self.agent_pos = new_pos
        
        # 计算奖励
        reward = -0.1  # 每步小惩罚
        done = False
        
        if self.agent_pos == self.goal_pos:
            reward = 10.0  # 到达目标
            done = True
        elif self.steps >= 100:  # 最大步数限制
            done = True
        
        return self._get_state(), reward, done, {}

# 训练Grid World智能体
def train_grid_world_agent():
    """训练Grid World智能体"""
    env = GridWorld()
    agent = QLearningAgent(
        actions=['up', 'down', 'left', 'right'],
        learning_rate=0.1,
        discount_factor=0.9,
        epsilon=0.1
    )
    
    episodes = 2000
    rewards_history = []
    steps_history = []
    
    for episode in range(episodes):
        state = env.reset()
        total_reward = 0
        done = False
        
        while not done:
            action = agent.choose_action(state)
            next_state, reward, done, _ = env.step(action)
            agent.update_q_value(state, action, reward, next_state)
            state = next_state
            total_reward += reward
        
        rewards_history.append(total_reward)
        steps_history.append(env.steps)
        
        # 衰减epsilon
        if episode % 100 == 0:
            agent.epsilon = max(0.01, agent.epsilon * 0.95)
            avg_reward = np.mean(rewards_history[-100:])
            avg_steps = np.mean(steps_history[-100:])
            print(f"Episode {episode}: Avg Reward={avg_reward:.2f}, Avg Steps={avg_steps:.1f}, Epsilon={agent.epsilon:.3f}")
    
    return agent, rewards_history, steps_history

# 训练Grid World智能体
grid_agent, grid_rewards, grid_steps = train_grid_world_agent()

# 可视化训练过程
plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(grid_rewards)
plt.title('Training Rewards')
plt.xlabel('Episode')
plt.ylabel('Total Reward')

plt.subplot(1, 2, 2)
plt.plot(grid_steps)
plt.title('Training Steps')
plt.xlabel('Episode')
plt.ylabel('Steps to Complete')

plt.tight_layout()
plt.show()
```

### 5.3 深度Q网络（DQN）

#### DQN实现
```python
# DQN智能体
class DQNAgent:
    """深度Q网络智能体"""
    
    def __init__(self, state_size, action_size, learning_rate=0.001):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = []  # 经验回放
        self.batch_size = 32
        self.epsilon = 1.0  # 探索率
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = learning_rate
        
        # 创建Q网络
        self.model = self._build_model()
    
    def _build_model(self):
        """构建Q网络"""
        model = keras.Sequential([
            layers.Dense(24, input_dim=self.state_size, activation='relu'),
            layers.Dense(24, activation='relu'),
            layers.Dense(self.action_size, activation='linear')
        ])
        model.compile(loss='mse', optimizer=keras.optimizers.Adam(lr=self.learning_rate))
        return model
    
    def remember(self, state, action, reward, next_state, done):
        """存储经验"""
        self.memory.append((state, action, reward, next_state, done))
    
    def act(self, state):
        """选择动作"""
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        q_values = self.model.predict(state)
        return np.argmax(q_values[0])
    
    def replay(self):
        """经验回放"""
        if len(self.memory) < self.batch_size:
            return
        
        batch = random.sample(self.memory, self.batch_size)
        states = np.array([e[0] for e in batch])
        actions = np.array([e[1] for e in batch])
        rewards = np.array([e[2] for e in batch])
        next_states = np.array([e[3] for e in batch])
        dones = np.array([e[4] for e in batch])
        
        # 预测Q值
        current_q_values = self.model.predict(states)
        next_q_values = self.model.predict(next_states)
        
        # 更新目标Q值
        for i in range(self.batch_size):
            if dones[i]:
                current_q_values[i][actions[i]] = rewards[i]
            else:
                current_q_values[i][actions[i]] = rewards[i] + 0.95 * np.max(next_q_values[i])
        
        # 训练模型
        self.model.fit(states, current_q_values, epochs=1, verbose=0)
        
        # 衰减探索率
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

# 注意：完整的DQN实现需要更多的工程工作，这里展示核心概念
```

#### 实践练习6：CartPole智能体
```python
# 练习6.1: 使用Gym环境训练智能体
def cartpole_dqn_example():
    """CartPole DQN示例"""
    try:
        import gym
        
        # 创建环境
        env = gym.make('CartPole-v1')
        state_size = env.observation_space.shape[0]
        action_size = env.action_space.n
        
        # 创建DQN智能体
        agent = DQNAgent(state_size, action_size)
        
        episodes = 1000
        scores = []
        
        for e in range(episodes):
            state = env.reset()
            state = np.reshape(state, [1, state_size])
            score = 0
            
            for time in range(500):
                # env.render()  # 可视化（可选）
                action = agent.act(state)
                next_state, reward, done, _ = env.step(action)
                reward = reward if not done else -10
                next_state = np.reshape(next_state, [1, state_size])
                
                agent.remember(state, action, reward, next_state, done)
                state = next_state
                score += reward
                
                if done:
                    print(f"Episode {e}: Score={score}, Epsilon={agent.epsilon:.2f}")
                    break
                
                if len(agent.memory) > agent.batch_size:
                    agent.replay()
            
            scores.append(score)
            
            # 检查是否解决
            if len(scores) >= 100 and np.mean(scores[-100:]) >= 195:
                print(f"Environment solved in {e} episodes!")
                break
        
        env.close()
        return scores
        
    except ImportError:
        print("Gym library not installed. Install with: pip install gym")
        return []

# 运行CartPole示例（需要安装gym）
# cartpole_scores = cartpole_dqn_example()
```

---

## 综合项目

### 完整的AI智能体系统

创建一个完整的AI智能体系统，包含以下功能：

1. **多种智能体类型**: Q-Learning、DQN、基于规则的智能体
2. **标准化环境接口**: 支持不同类型的环境
3. **训练和评估模块**: 自动化训练和性能评估
4. **可视化和监控**: 实时训练监控和结果可视化
5. **模型保存和加载**: 持久化训练好的智能体

#### 项目结构
```
ai_agent_system/
├── main.py                  # 主程序
├── agents/                 # 智能体实现
│   ├── base_agent.py       # 基础智能体类
│   ├── q_learning_agent.py
│   ├── dqn_agent.py
│   └── rule_based_agent.py
├── environments/           # 环境实现
│   ├── base_environment.py
│   ├── grid_world.py
│   └── custom_env.py
├── trainers/               # 训练器
│   ├── base_trainer.py
│   └── dqn_trainer.py
├── evaluators/             # 评估器
│   └── performance_evaluator.py
├── visualizers/            # 可视化
│   └── training_visualizer.py
└── utils/                  # 工具函数
    ├── logger.py
    └── config_manager.py
```

#### 实现步骤
1. 创建基础智能体和环境接口
2. 实现具体的智能体类型
3. 开发训练和评估模块
4. 添加可视化和监控功能
5. 实现模型持久化

---

## 学习资源

### 官方文档
- [PyTorch官方文档](https://pytorch.org/docs/stable/index.html)
- [TensorFlow官方文档](https://www.tensorflow.org/api_docs)
- [Hugging Face Transformers文档](https://huggingface.co/docs/transformers/index)
- [OpenAI Gym文档](https://www.gymlibrary.dev/)

### 在线教程
- **免费**:
  - [PyTorch官方教程](https://pytorch.org/tutorials/)
  - [TensorFlow官方教程](https://www.tensorflow.org/tutorials)
  - [Deep Learning with PyTorch: A 60 Minute Blitz](https://pytorch.org/tutorials/beginner/deep_learning_60min_blitz.html)
- **付费**:
  - Coursera: Deep Learning Specialization by Andrew Ng
  - Udacity: Deep Reinforcement Learning Nanodegree

### 书籍推荐
- 《深度学习》- Ian Goodfellow, Yoshua Bengio, Aaron Courville
- 《强化学习：原理与实践》- Richard Sutton, Andrew Barto
- 《Hands-On Machine Learning with Scikit-Learn, Keras & TensorFlow》- Aurélien Géron

### 实践平台
- [Kaggle](https://www.kaggle.com/) - 深度学习竞赛
- [Google Colab](https://colab.research.google.com/) - 免费GPU资源
- [Papers with Code](https://paperswithcode.com/) - 最新研究和代码实现
- [Hugging Face Spaces](https://huggingface.co/spaces) - 模型演示

### 社区资源
- [Stack Overflow](https://stackoverflow.com/questions/tagged/pytorch)
- [Reddit r/MachineLearning](https://www.reddit.com/r/MachineLearning/)
- [PyTorch论坛](https://discuss.pytorch.org/)
- [TensorFlow论坛](https://discuss.tensorflow.org/)

---

## 学习建议

1. **循序渐进**: 从简单概念开始，逐步深入复杂主题
2. **动手实践**: 每个概念都要通过代码实现来验证
3. **理解数学**: 深度学习需要一定的数学基础（线性代数、微积分、概率）
4. **阅读论文**: 关注最新的研究成果和最佳实践
5. **参与社区**: 加入相关社区，分享经验和解决问题

### 每日学习计划
- **上午**: 学习理论概念和算法原理（1-2小时）
- **下午**: 动手实现和调试代码（2-3小时）
- **晚上**: 阅读相关论文或博客文章（1小时）

记住：深度学习和强化学习是复杂的领域，需要时间和实践来掌握。通过这4周的学习，你已经建立了坚实的基础，可以继续深入学习特定领域的高级主题。保持好奇心和持续学习的态度，你将在AI智能体开发领域取得成功！