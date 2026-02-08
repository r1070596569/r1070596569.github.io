# Python AI智能体开发 - 第3周学习教材

## 目录
1. [环境设置与机器学习基础](#环境设置与机器学习基础)
2. [Day 1-2: Scikit-learn基础](#day-1-2-scikit-learn基础)
3. [Day 3-4: 模型评估与特征工程](#day-3-4-模型评估与特征工程)
4. [Day 5-7: 简单智能体实现](#day-5-7-简单智能体实现)
5. [综合项目](#综合项目)
6. [学习资源](#学习资源)

---

## 环境设置与机器学习基础

### 1.1 安装机器学习库
```bash
# 激活虚拟环境
source ai_agent_env/bin/activate  # Linux/Mac
# ai_agent_env\Scripts\activate   # Windows

# 安装机器学习库
pip install scikit-learn xgboost lightgbm

# 验证安装
python -c "import sklearn; print('Scikit-learn version:', sklearn.__version__)"
```

### 1.2 机器学习基础概念

#### 机器学习类型
- **监督学习**: 有标签数据，预测目标变量
  - 分类：预测类别（如垃圾邮件检测）
  - 回归：预测连续值（如房价预测）
- **无监督学习**: 无标签数据，发现模式
  - 聚类：分组相似数据（如客户细分）
  - 降维：减少特征数量（如PCA）
- **强化学习**: 通过试错学习最优策略（智能体核心）

#### Scikit-learn API设计哲学
- **一致性**: 所有算法都有`fit()`, `predict()`, `score()`方法
- **可组合性**: 可以轻松组合多个步骤
- **合理性**: 默认参数通常工作良好

### 1.3 数据集和问题定义

#### 内置数据集
```python
from sklearn.datasets import load_iris, load_boston, make_classification

# 分类数据集
iris = load_iris()
print("Iris dataset shape:", iris.data.shape)
print("Feature names:", iris.feature_names)
print("Target names:", iris.target_names)

# 回归数据集
# boston = load_boston()  # 已弃用，使用其他数据集
from sklearn.datasets import fetch_california_housing
housing = fetch_california_housing()
print("Housing dataset shape:", housing.data.shape)

# 生成合成数据集
X, y = make_classification(n_samples=1000, n_features=20, n_classes=2, random_state=42)
print("Synthetic dataset shape:", X.shape)
```

---

## Day 1-2: Scikit-learn基础

### 2.1 基本工作流程

#### 标准机器学习流程
```python
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# 1. 加载数据
from sklearn.datasets import load_iris
iris = load_iris()
X, y = iris.data, iris.target

# 2. 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 3. 特征标准化
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 4. 训练模型
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train_scaled, y_train)

# 5. 预测和评估
y_pred = model.predict(X_test_scaled)
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=iris.target_names))
```

#### 实践练习1：基础分类器
```python
# 练习1.1: 多种分类器比较
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier

def compare_classifiers(X, y, test_size=0.2, random_state=42):
    """比较多种分类器的性能"""
    # 划分数据集
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    # 标准化
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # 定义分类器
    classifiers = {
        'Logistic Regression': LogisticRegression(random_state=random_state),
        'SVM': SVC(random_state=random_state),
        'KNN': KNeighborsClassifier(),
        'Decision Tree': DecisionTreeClassifier(random_state=random_state),
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=random_state)
    }
    
    results = {}
    for name, clf in classifiers.items():
        # 训练
        clf.fit(X_train_scaled, y_train)
        # 预测
        y_pred = clf.predict(X_test_scaled)
        # 评估
        accuracy = accuracy_score(y_test, y_pred)
        results[name] = accuracy
        print(f"{name}: {accuracy:.4f}")
    
    return results

# 测试
iris = load_iris()
results = compare_classifiers(iris.data, iris.target)
best_classifier = max(results, key=results.get)
print(f"\nBest classifier: {best_classifier} ({results[best_classifier]:.4f})")
```

### 2.2 回归任务

#### 基本回归流程
```python
from sklearn.datasets import fetch_california_housing
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

# 加载数据
housing = fetch_california_housing()
X, y = housing.data, housing.target

# 划分数据集
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 标准化
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 训练多个回归模型
regressors = {
    'Linear Regression': LinearRegression(),
    'Ridge': Ridge(alpha=1.0),
    'Lasso': Lasso(alpha=0.1),
    'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42)
}

for name, reg in regressors.items():
    reg.fit(X_train_scaled, y_train)
    y_pred = reg.predict(X_test_scaled)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    print(f"{name}: MSE={mse:.4f}, R²={r2:.4f}")
```

#### 实践练习2：回归模型调优
```python
# 练习2.1: 房价预测
def house_price_prediction():
    """房价预测完整流程"""
    # 1. 数据加载
    housing = fetch_california_housing()
    X, y = housing.data, housing.target
    
    # 2. 数据探索
    print("Dataset shape:", X.shape)
    print("Feature names:", housing.feature_names)
    print("Target description:", housing.DESCR.split('\n')[5])
    
    # 3. 数据划分
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # 4. 特征工程 - 添加交互特征
    from sklearn.preprocessing import PolynomialFeatures
    poly = PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)
    X_train_poly = poly.fit_transform(X_train)
    X_test_poly = poly.transform(X_test)
    
    # 5. 标准化
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_poly)
    X_test_scaled = scaler.transform(X_test_poly)
    
    # 6. 模型训练和比较
    models = {
        'Linear': LinearRegression(),
        'Ridge': Ridge(alpha=10.0),
        'Random Forest': RandomForestRegressor(n_estimators=200, random_state=42)
    }
    
    results = {}
    for name, model in models.items():
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        results[name] = {'MSE': mse, 'R2': r2}
        print(f"{name}: MSE={mse:.4f}, R²={r2:.4f}")
    
    return results

# 运行房价预测
house_results = house_price_prediction()
```

### 2.3 无监督学习

#### 聚类分析
```python
from sklearn.cluster import KMeans, DBSCAN
from sklearn.decomposition import PCA
from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt

# 生成聚类数据
X, y_true = make_blobs(n_samples=300, centers=4, cluster_std=0.60, random_state=0)

# K-Means聚类
kmeans = KMeans(n_clusters=4, random_state=0)
y_kmeans = kmeans.fit_predict(X)

# 可视化
plt.figure(figsize=(12, 4))

plt.subplot(1, 3, 1)
plt.scatter(X[:, 0], X[:, 1], c=y_true, cmap='viridis')
plt.title('True Labels')

plt.subplot(1, 3, 2)
plt.scatter(X[:, 0], X[:, 1], c=y_kmeans, cmap='viridis')
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], 
           c='red', marker='x', s=200, linewidths=3)
plt.title('K-Means Clustering')

# DBSCAN聚类
dbscan = DBSCAN(eps=0.5, min_samples=5)
y_dbscan = dbscan.fit_predict(X)

plt.subplot(1, 3, 3)
plt.scatter(X[:, 0], X[:, 1], c=y_dbscan, cmap='viridis')
plt.title('DBSCAN Clustering')

plt.tight_layout()
plt.show()
```

#### 降维技术
```python
# PCA降维
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)

print("Original shape:", X.shape)
print("PCA shape:", X_pca.shape)
print("Explained variance ratio:", pca.explained_variance_ratio_)
print("Total explained variance:", sum(pca.explained_variance_ratio_))

# 可视化PCA结果
plt.figure(figsize=(8, 6))
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y_true, cmap='viridis')
plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%} variance)')
plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%} variance)')
plt.title('PCA Results')
plt.show()
```

#### 实践练习3：客户细分
```python
# 练习3.1: 客户细分分析
def customer_segmentation():
    """客户细分分析"""
    # 生成模拟客户数据
    np.random.seed(42)
    n_customers = 1000
    
    data = {
        'age': np.random.normal(35, 10, n_customers),
        'income': np.random.normal(50000, 15000, n_customers),
        'spending_score': np.random.normal(50, 15, n_customers),
        'loyalty_years': np.random.exponential(3, n_customers)
    }
    
    df = pd.DataFrame(data)
    
    # 数据预处理
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df)
    
    # 确定最佳聚类数量（肘部法则）
    inertias = []
    K_range = range(1, 11)
    for k in K_range:
        kmeans = KMeans(n_clusters=k, random_state=42)
        kmeans.fit(X_scaled)
        inertias.append(kmeans.inertia_)
    
    # 绘制肘部图
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(K_range, inertias, 'bo-')
    plt.xlabel('Number of clusters (k)')
    plt.ylabel('Inertia')
    plt.title('Elbow Method')
    plt.grid(True)
    
    # 选择k=4进行聚类
    kmeans = KMeans(n_clusters=4, random_state=42)
    df['cluster'] = kmeans.fit_predict(X_scaled)
    
    # 分析聚类结果
    cluster_analysis = df.groupby('cluster').mean()
    print("Cluster Analysis:")
    print(cluster_analysis)
    
    # 可视化（使用PCA降维到2D）
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)
    
    plt.subplot(1, 2, 2)
    scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=df['cluster'], cmap='viridis')
    plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%} variance)')
    plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%} variance)')
    plt.title('Customer Segmentation (PCA)')
    plt.colorbar(scatter)
    
    plt.tight_layout()
    plt.show()
    
    return df, cluster_analysis

# 运行客户细分
customer_df, analysis = customer_segmentation()
```

---

## Day 3-4: 模型评估与特征工程

### 3.1 模型评估指标

#### 分类评估指标
```python
from sklearn.metrics import confusion_matrix, precision_recall_curve, roc_curve, auc
from sklearn.model_selection import cross_val_score

# 加载数据
from sklearn.datasets import make_classification
X, y = make_classification(n_samples=1000, n_features=20, n_classes=2, random_state=42)

# 划分数据
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 训练模型
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
y_proba = model.predict_proba(X_test)[:, 1]

# 混淆矩阵
cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print(cm)

# 详细分类报告
print("\nDetailed Classification Report:")
print(classification_report(y_test, y_pred))

# 交叉验证
cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy')
print(f"\nCross-validation scores: {cv_scores}")
print(f"Mean CV score: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")

# ROC曲线
fpr, tpr, _ = roc_curve(y_test, y_proba)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc="lower right")

# Precision-Recall曲线
precision, recall, _ = precision_recall_curve(y_test, y_proba)
plt.subplot(1, 2, 2)
plt.plot(recall, precision, color='blue', lw=2)
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve')
plt.grid(True)

plt.tight_layout()
plt.show()
```

#### 回归评估指标
```python
# 回归评估指标
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# 使用之前的房价预测结果
y_true = y_test  # 假设这是真实值
y_pred_reg = y_pred  # 假设这是预测值

mae = mean_absolute_error(y_true, y_pred_reg)
mse = mean_squared_error(y_true, y_pred_reg)
rmse = np.sqrt(mse)
r2 = r2_score(y_true, y_pred_reg)

print(f"MAE: {mae:.4f}")
print(f"MSE: {mse:.4f}")
print(f"RMSE: {rmse:.4f}")
print(f"R²: {r2:.4f}")

# 残差图
residuals = y_true - y_pred_reg
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.scatter(y_pred_reg, residuals)
plt.xlabel('Predicted Values')
plt.ylabel('Residuals')
plt.title('Residual Plot')
plt.axhline(y=0, color='r', linestyle='--')

plt.subplot(1, 2, 2)
plt.hist(residuals, bins=30, alpha=0.7)
plt.xlabel('Residuals')
plt.ylabel('Frequency')
plt.title('Residual Distribution')

plt.tight_layout()
plt.show()
```

### 3.2 特征工程

#### 特征选择
```python
from sklearn.feature_selection import SelectKBest, f_classif, RFE
from sklearn.ensemble import ExtraTreesClassifier

# 加载数据
iris = load_iris()
X, y = iris.data, iris.target

# 单变量特征选择
selector = SelectKBest(score_func=f_classif, k=2)
X_selected = selector.fit_transform(X, y)
selected_features = selector.get_support(indices=True)
print("Selected features:", [iris.feature_names[i] for i in selected_features])

# 递归特征消除
estimator = RandomForestClassifier(n_estimators=100, random_state=42)
rfe = RFE(estimator, n_features_to_select=2)
X_rfe = rfe.fit_transform(X, y)
rfe_features = rfe.get_support(indices=True)
print("RFE selected features:", [iris.feature_names[i] for i in rfe_features])

# 基于树的特征重要性
tree_clf = ExtraTreesClassifier(n_estimators=100, random_state=42)
tree_clf.fit(X, y)
feature_importance = tree_clf.feature_importances_
feature_names = iris.feature_names

# 可视化特征重要性
plt.figure(figsize=(10, 6))
plt.barh(feature_names, feature_importance)
plt.xlabel('Feature Importance')
plt.title('Feature Importance from Extra Trees')
plt.show()
```

#### 特征编码和转换
```python
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer

# 创建包含分类和数值特征的数据
data = {
    'age': [25, 30, 35, 40, 45],
    'city': ['Beijing', 'Shanghai', 'Guangzhou', 'Beijing', 'Shenzhen'],
    'income': [50000, 60000, 70000, 80000, 90000],
    'education': ['Bachelor', 'Master', 'PhD', 'Bachelor', 'Master']
}
df = pd.DataFrame(data)

print("Original data:")
print(df)

# 标签编码（适用于有序分类）
label_encoder = LabelEncoder()
df['education_encoded'] = label_encoder.fit_transform(df['education'])
print("\nAfter label encoding education:")
print(df[['education', 'education_encoded']])

# 独热编码（适用于无序分类）
categorical_features = ['city', 'education']
numerical_features = ['age', 'income']

preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_features),
        ('cat', OneHotEncoder(drop='first'), categorical_features)
    ])

X_processed = preprocessor.fit_transform(df)
print(f"\nProcessed data shape: {X_processed.shape}")
print("Processed data:")
print(X_processed)

# 获取特征名称
feature_names = (numerical_features + 
                list(preprocessor.named_transformers_['cat']
                     .get_feature_names_out(categorical_features)))
print("\nFeature names:")
print(feature_names)
```

#### 实践练习4：特征工程管道
```python
# 练习4.1: 完整的特征工程管道
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

def create_ml_pipeline():
    """创建完整的机器学习管道"""
    # 创建包含缺失值的数据
    np.random.seed(42)
    n_samples = 1000
    
    data = {
        'age': np.random.normal(35, 10, n_samples),
        'income': np.random.normal(50000, 15000, n_samples),
        'city': np.random.choice(['Beijing', 'Shanghai', 'Guangzhou', 'Shenzhen'], n_samples),
        'education': np.random.choice(['High School', 'Bachelor', 'Master', 'PhD'], n_samples)
    }
    
    # 添加一些缺失值
    data['age'][np.random.choice(n_samples, 50, replace=False)] = np.nan
    data['income'][np.random.choice(n_samples, 30, replace=False)] = np.nan
    
    df = pd.DataFrame(data)
    y = (df['income'] > df['income'].median()).astype(int)  # 二分类目标
    
    # 定义数值和分类特征
    numerical_features = ['age', 'income']
    categorical_features = ['city', 'education']
    
    # 创建预处理器
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', Pipeline([
                ('imputer', SimpleImputer(strategy='median')),
                ('scaler', StandardScaler())
            ]), numerical_features),
            ('cat', Pipeline([
                ('imputer', SimpleImputer(strategy='most_frequent')),
                ('encoder', OneHotEncoder(drop='first', handle_unknown='ignore'))
            ]), categorical_features)
        ]
    )
    
    # 创建完整管道
    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))
    ])
    
    # 划分数据
    X_train, X_test, y_train, y_test = train_test_split(
        df, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # 训练和评估
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    print(f"Pipeline accuracy: {accuracy:.4f}")
    print(f"Pipeline steps: {list(pipeline.named_steps.keys())}")
    
    return pipeline, X_test, y_test, y_pred

# 运行特征工程管道
ml_pipeline, X_test, y_test, y_pred = create_ml_pipeline()
```

---

## Day 5-7: 简单智能体实现

### 5.1 基于规则的智能体

#### 简单决策系统
```python
class RuleBasedAgent:
    """基于规则的智能体"""
    
    def __init__(self, rules=None):
        self.rules = rules or self._default_rules()
        self.memory = []
    
    def _default_rules(self):
        """默认规则集"""
        return [
            {'condition': lambda obs: obs['temperature'] > 30, 'action': 'turn_on_ac'},
            {'condition': lambda obs: obs['temperature'] < 15, 'action': 'turn_on_heater'},
            {'condition': lambda obs: obs['humidity'] > 80, 'action': 'turn_on_dehumidifier'},
            {'condition': lambda obs: obs['light'] < 100, 'action': 'turn_on_lights'},
            {'condition': lambda obs: True, 'action': 'do_nothing'}  # 默认规则
        ]
    
    def observe(self, observation):
        """接收观察"""
        self.memory.append(observation)
        if len(self.memory) > 100:  # 限制内存大小
            self.memory.pop(0)
    
    def decide(self, observation):
        """基于规则做决策"""
        for rule in self.rules:
            if rule['condition'](observation):
                return rule['action']
        return 'do_nothing'
    
    def act(self, observation):
        """完整的行为循环"""
        self.observe(observation)
        return self.decide(observation)

# 测试基于规则的智能体
agent = RuleBasedAgent()

test_observations = [
    {'temperature': 35, 'humidity': 60, 'light': 200},
    {'temperature': 10, 'humidity': 40, 'light': 50},
    {'temperature': 25, 'humidity': 85, 'light': 150},
    {'temperature': 20, 'humidity': 50, 'light': 80}
]

for obs in test_observations:
    action = agent.act(obs)
    print(f"Observation: {obs}")
    print(f"Action: {action}\n")
```

#### 实践练习5：智能家居智能体
```python
# 练习5.1: 智能家居智能体
class SmartHomeAgent(RuleBasedAgent):
    """智能家居智能体"""
    
    def __init__(self):
        super().__init__()
        self.device_states = {
            'ac': False,
            'heater': False,
            'dehumidifier': False,
            'lights': False
        }
        self.energy_consumption = 0
    
    def _default_rules(self):
        return [
            # 温度控制
            {'condition': lambda obs: obs['temperature'] > 28 and not self.device_states['ac'], 
             'action': 'turn_on_ac'},
            {'condition': lambda obs: obs['temperature'] < 22 and self.device_states['ac'], 
             'action': 'turn_off_ac'},
            {'condition': lambda obs: obs['temperature'] < 18 and not self.device_states['heater'], 
             'action': 'turn_on_heater'},
            {'condition': lambda obs: obs['temperature'] > 20 and self.device_states['heater'], 
             'action': 'turn_off_heater'},
            
            # 湿度控制
            {'condition': lambda obs: obs['humidity'] > 70 and not self.device_states['dehumidifier'], 
             'action': 'turn_on_dehumidifier'},
            {'condition': lambda obs: obs['humidity'] < 50 and self.device_states['dehumidifier'], 
             'action': 'turn_off_dehumidifier'},
            
            # 照明控制
            {'condition': lambda obs: obs['light'] < 100 and obs['time_of_day'] == 'night' and not self.device_states['lights'], 
             'action': 'turn_on_lights'},
            {'condition': lambda obs: obs['light'] > 200 and self.device_states['lights'], 
             'action': 'turn_off_lights'},
            
            # 节能模式
            {'condition': lambda obs: obs['occupancy'] == False and any(self.device_states.values()),
             'action': 'energy_saving_mode'},
            
            # 默认
            {'condition': lambda obs: True, 'action': 'monitor'}
        ]
    
    def execute_action(self, action, observation):
        """执行动作并更新状态"""
        energy_cost = 0
        
        if action == 'turn_on_ac':
            self.device_states['ac'] = True
            energy_cost = 2.0
        elif action == 'turn_off_ac':
            self.device_states['ac'] = False
            energy_cost = 0.1
        elif action == 'turn_on_heater':
            self.device_states['heater'] = True
            energy_cost = 1.5
        elif action == 'turn_off_heater':
            self.device_states['heater'] = False
            energy_cost = 0.1
        elif action == 'turn_on_dehumidifier':
            self.device_states['dehumidifier'] = True
            energy_cost = 0.8
        elif action == 'turn_off_dehumidifier':
            self.device_states['dehumidifier'] = False
            energy_cost = 0.1
        elif action == 'turn_on_lights':
            self.device_states['lights'] = True
            energy_cost = 0.3
        elif action == 'turn_off_lights':
            self.device_states['lights'] = False
            energy_cost = 0.05
        elif action == 'energy_saving_mode':
            # 关闭所有设备
            for device in self.device_states:
                self.device_states[device] = False
            energy_cost = 0.2
        else:
            energy_cost = 0.05  # 监控模式
        
        self.energy_consumption += energy_cost
        return energy_cost
    
    def step(self, observation):
        """完整的智能体步骤"""
        action = self.act(observation)
        energy_cost = self.execute_action(action, observation)
        
        return {
            'action': action,
            'device_states': self.device_states.copy(),
            'energy_cost': energy_cost,
            'total_energy': self.energy_consumption
        }

# 测试智能家居智能体
smart_agent = SmartHomeAgent()

test_scenarios = [
    {'temperature': 30, 'humidity': 60, 'light': 50, 'time_of_day': 'night', 'occupancy': True},
    {'temperature': 25, 'humidity': 80, 'light': 300, 'time_of_day': 'day', 'occupancy': False},
    {'temperature': 15, 'humidity': 40, 'light': 80, 'time_of_day': 'night', 'occupancy': True},
    {'temperature': 22, 'humidity': 50, 'light': 150, 'time_of_day': 'day', 'occupancy': True}
]

print("Smart Home Agent Simulation:")
print("=" * 50)

for i, obs in enumerate(test_scenarios, 1):
    result = smart_agent.step(obs)
    print(f"Step {i}:")
    print(f"  Observation: {obs}")
    print(f"  Action: {result['action']}")
    print(f"  Device States: {result['device_states']}")
    print(f"  Energy Cost: {result['energy_cost']:.2f} kWh")
    print(f"  Total Energy: {result['total_energy']:.2f} kWh")
    print("-" * 30)
```

### 5.2 基于机器学习的智能体

#### 简单预测智能体
```python
class MLBasedAgent:
    """基于机器学习的智能体"""
    
    def __init__(self, model=None):
        self.model = model or RandomForestClassifier(n_estimators=100, random_state=42)
        self.scaler = StandardScaler()
        self.is_trained = False
        self.memory = []
        self.max_memory = 1000
    
    def remember(self, observation, action, reward):
        """记忆经验"""
        self.memory.append({
            'observation': observation,
            'action': action,
            'reward': reward
        })
        
        if len(self.memory) > self.max_memory:
            self.memory.pop(0)
    
    def train_from_memory(self):
        """从记忆中训练模型"""
        if len(self.memory) < 10:  # 需要最少10个样本
            return False
        
        # 准备训练数据
        observations = [exp['observation'] for exp in self.memory]
        actions = [exp['action'] for exp in self.memory]
        
        # 转换为数组
        X = np.array(observations)
        y = np.array(actions)
        
        # 标准化
        if not self.is_trained:
            X_scaled = self.scaler.fit_transform(X)
            self.is_trained = True
        else:
            X_scaled = self.scaler.transform(X)
        
        # 训练模型
        self.model.fit(X_scaled, y)
        return True
    
    def predict_action(self, observation):
        """预测最佳动作"""
        if not self.is_trained:
            # 随机选择动作
            return np.random.choice(['action_0', 'action_1', 'action_2'])
        
        # 预测
        obs_array = np.array(observation).reshape(1, -1)
        obs_scaled = self.scaler.transform(obs_array)
        predicted_action = self.model.predict(obs_scaled)[0]
        
        return predicted_action
    
    def step(self, observation):
        """智能体步骤"""
        action = self.predict_action(observation)
        return action

# 测试ML智能体
ml_agent = MLBasedAgent()

# 生成一些训练数据
np.random.seed(42)
training_data = []
for _ in range(100):
    obs = np.random.randn(4).tolist()
    # 简单的规则：如果第一个特征大于0，选择action_0，否则选择action_1
    action = 'action_0' if obs[0] > 0 else 'action_1'
    reward = 1.0 if (obs[0] > 0 and action == 'action_0') or (obs[0] <= 0 and action == 'action_1') else 0.0
    ml_agent.remember(obs, action, reward)

# 训练智能体
trained = ml_agent.train_from_memory()
print(f"Agent trained: {trained}")

# 测试预测
test_observations = [
    [1.0, 0.5, -0.3, 0.8],   # 应该预测 action_0
    [-1.0, 0.2, 0.4, -0.6],  # 应该预测 action_1
    [0.5, -0.1, 0.9, 0.3],   # 应该预测 action_0
    [-0.8, 0.7, -0.2, 0.1]   # 应该预测 action_1
]

print("\nML Agent Predictions:")
for i, obs in enumerate(test_observations, 1):
    action = ml_agent.step(obs)
    expected = 'action_0' if obs[0] > 0 else 'action_1'
    print(f"Test {i}: Observation={obs}, Predicted={action}, Expected={expected}")
```

#### 实践练习6：股票交易智能体
```python
# 练习6.1: 简单股票交易智能体
class StockTradingAgent:
    """简单股票交易智能体"""
    
    def __init__(self, initial_capital=10000):
        self.initial_capital = initial_capital
        self.capital = initial_capital
        self.shares = 0
        self.transaction_history = []
        self.model = RandomForestClassifier(n_estimators=100, random_state=42)
        self.scaler = StandardScaler()
        self.is_trained = False
    
    def calculate_technical_indicators(self, prices, window=10):
        """计算技术指标"""
        prices = np.array(prices)
        indicators = {}
        
        # 移动平均
        indicators['ma'] = np.mean(prices[-window:]) if len(prices) >= window else np.mean(prices)
        
        # 相对强弱指数 (简化版)
        if len(prices) > 1:
            price_changes = np.diff(prices)
            gains = price_changes[price_changes > 0].sum() if len(price_changes[price_changes > 0]) > 0 else 0
            losses = -price_changes[price_changes < 0].sum() if len(price_changes[price_changes < 0]) > 0 else 0
            rs = gains / (losses + 1e-8)
            indicators['rsi'] = 100 - (100 / (1 + rs))
        else:
            indicators['rsi'] = 50
        
        # 波动率
        indicators['volatility'] = np.std(prices[-window:]) if len(prices) >= window else np.std(prices)
        
        # 当前价格
        indicators['current_price'] = prices[-1]
        
        return indicators
    
    def get_features(self, price_history):
        """获取特征向量"""
        indicators = self.calculate_technical_indicators(price_history)
        return [
            indicators['ma'],
            indicators['rsi'],
            indicators['volatility'],
            indicators['current_price']
        ]
    
    def decide_action(self, features):
        """决定交易动作"""
        if not self.is_trained:
            # 随机决策
            return np.random.choice(['buy', 'sell', 'hold'])
        
        # 使用模型预测
        features_scaled = self.scaler.transform([features])
        prediction = self.model.predict(features_scaled)[0]
        return prediction
    
    def execute_trade(self, action, current_price):
        """执行交易"""
        transaction = {
            'action': action,
            'price': current_price,
            'capital_before': self.capital,
            'shares_before': self.shares
        }
        
        if action == 'buy' and self.capital >= current_price:
            shares_to_buy = int(self.capital // current_price)
            if shares_to_buy > 0:
                self.shares += shares_to_buy
                self.capital -= shares_to_buy * current_price
                transaction['shares_bought'] = shares_to_buy
        elif action == 'sell' and self.shares > 0:
            self.capital += self.shares * current_price
            transaction['shares_sold'] = self.shares
            self.shares = 0
        
        transaction['capital_after'] = self.capital
        transaction['shares_after'] = self.shares
        transaction['portfolio_value'] = self.capital + self.shares * current_price
        
        self.transaction_history.append(transaction)
        return transaction
    
    def train_on_historical_data(self, price_data, labels):
        """在历史数据上训练"""
        X = np.array(price_data)
        y = np.array(labels)
        
        # 标准化
        X_scaled = self.scaler.fit_transform(X)
        
        # 训练模型
        self.model.fit(X_scaled, y)
        self.is_trained = True
    
    def simulate_trading(self, price_series):
        """模拟交易"""
        price_history = []
        portfolio_values = []
        
        for i, price in enumerate(price_series):
            price_history.append(price)
            
            if len(price_history) < 5:  # 需要最少5个价格点
                continue
            
            # 获取特征
            features = self.get_features(price_history)
            
            # 决策
            action = self.decide_action(features)
            
            # 执行交易
            transaction = self.execute_trade(action, price)
            
            # 记录投资组合价值
            portfolio_value = self.capital + self.shares * price
            portfolio_values.append(portfolio_value)
            
            if i % 50 == 0:  # 每50步打印一次
                print(f"Step {i}: Price={price:.2f}, Action={action}, Portfolio={portfolio_value:.2f}")
        
        return portfolio_values

# 生成模拟股价数据
np.random.seed(42)
def generate_stock_prices(days=1000, initial_price=100, volatility=0.02):
    """生成模拟股价"""
    prices = [initial_price]
    for _ in range(days - 1):
        # 随机游走
        change = np.random.normal(0, volatility)
        new_price = prices[-1] * (1 + change)
        prices.append(max(new_price, 0.01))  # 价格不能为负
    return prices

# 创建训练数据
def create_training_data(price_series, window=10):
    """创建训练数据"""
    X, y = [], []
    
    for i in range(window, len(price_series) - 1):
        # 获取过去的价格
        past_prices = price_series[i-window:i]
        
        # 计算特征
        agent = StockTradingAgent()
        features = agent.get_features(past_prices)
        X.append(features)
        
        # 确定标签（如果明天价格上涨则买入，否则卖出）
        tomorrow_price = price_series[i+1]
        today_price = price_series[i]
        if tomorrow_price > today_price * 1.01:  # 上涨超过1%
            y.append('buy')
        elif tomorrow_price < today_price * 0.99:  # 下跌超过1%
            y.append('sell')
        else:
            y.append('hold')
    
    return X, y

# 测试股票交易智能体
print("Stock Trading Agent Simulation")
print("=" * 40)

# 生成股价数据
stock_prices = generate_stock_prices(days=1000, initial_price=100, volatility=0.015)

# 创建训练数据
X_train, y_train = create_training_data(stock_prices[:800])

# 创建和训练智能体
trading_agent = StockTradingAgent(initial_capital=10000)
trading_agent.train_on_historical_data(X_train, y_train)

# 模拟交易
portfolio_values = trading_agent.simulate_trading(stock_prices[800:])

# 最终结果
final_portfolio = portfolio_values[-1]
initial_investment = 10000
buy_and_hold_value = 10000 / stock_prices[800] * stock_prices[-1]

print(f"\nFinal Results:")
print(f"Initial Capital: ${initial_investment:.2f}")
print(f"Final Portfolio Value: ${final_portfolio:.2f}")
print(f"Buy and Hold Strategy: ${buy_and_hold_value:.2f}")
print(f"Agent Return: {(final_portfolio - initial_investment) / initial_investment * 100:.2f}%")
print(f"Buy & Hold Return: {(buy_and_hold_value - initial_investment) / initial_investment * 100:.2f}%")
```

---

## 综合项目

### 智能体性能评估系统

创建一个完整的智能体性能评估系统，包含以下功能：

1. **多种智能体实现**: 基于规则、基于ML、随机智能体
2. **环境模拟**: 提供标准化的测试环境
3. **性能评估**: 多维度评估指标
4. **结果可视化**: 生成性能对比图表
5. **报告生成**: 自动生成评估报告

#### 项目结构
```
agent_evaluation_system/
├── main.py                  # 主程序
├── agents/                 # 智能体实现
│   ├── rule_based_agent.py
│   ├── ml_agent.py
│   └── random_agent.py
├── environments/           # 环境实现
│   ├── grid_world.py
│   └── trading_env.py
├── evaluators/             # 评估器
│   ├── performance_evaluator.py
│   └── comparison_evaluator.py
├── visualizers/            # 可视化
│   └── results_visualizer.py
└── reports/                # 报告生成
    └── evaluation_report.py
```

#### 实现步骤
1. 创建不同类型的智能体
2. 实现标准化的测试环境
3. 开发性能评估模块
4. 实现结果可视化功能
5. 创建自动报告生成器

---

## 学习资源

### 官方文档
- [Scikit-learn官方文档](https://scikit-learn.org/stable/)
- [Scikit-learn用户指南](https://scikit-learn.org/stable/user_guide.html)
- [Scikit-learn API参考](https://scikit-learn.org/stable/modules/classes.html)

### 在线教程
- **免费**:
  - [Scikit-learn官方教程](https://scikit-learn.org/stable/tutorial/index.html)
  - [Kaggle机器学习课程](https://www.kaggle.com/learn/machine-learning)
  - [Google机器学习速成课程](https://developers.google.com/machine-learning/crash-course)
- **付费**:
  - Coursera: Machine Learning by Andrew Ng
  - Udemy: Machine Learning A-Z

### 书籍推荐
- 《Python机器学习》- Sebastian Raschka
- 《机器学习实战》- Peter Harrington
- 《Hands-On Machine Learning with Scikit-Learn, Keras & TensorFlow》- Aurélien Géron

### 实践平台
- [Kaggle](https://www.kaggle.com/) - 机器学习竞赛
- [Google Colab](https://colab.research.google.com/) - 免费机器学习环境
- [Papers with Code](https://paperswithcode.com/) - 最新研究和代码

### 社区资源
- [Stack Overflow](https://stackoverflow.com/questions/tagged/scikit-learn)
- [Reddit r/MachineLearning](https://www.reddit.com/r/MachineLearning/)
- [Kaggle讨论区](https://www.kaggle.com/discussion)

---

## 学习建议

1. **理论结合实践**: 每学一个算法都要动手实现
2. **理解原理**: 不只是调用API，要理解算法原理
3. **项目驱动**: 用实际项目巩固所学知识
4. **持续学习**: 机器学习领域发展迅速，保持学习
5. **数学基础**: 补充必要的数学知识（线性代数、概率统计）

### 每日学习计划
- **上午**: 学习算法原理和API使用（1-2小时）
- **下午**: 动手实现和项目开发（2-3小时）
- **晚上**: 复习和阅读相关论文（30分钟-1小时）

记住：机器学习的核心是理解问题、选择合适的算法、调优参数。通过第3周的学习，你将掌握Scikit-learn的基础，能够构建简单的智能体，并为后续的深度学习和强化学习打下坚实的基础。