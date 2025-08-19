from collections import Counter
import pandas as pd # 数据处理
import numpy as np # 数学运算
from sklearn.model_selection import train_test_split, cross_validate # 划分数据集函数
from sklearn.metrics import accuracy_score # 准确率函数
RANDOM_SEED = 2020 # 固定随机种子


# ### 读入数据
csv_data = './data/high_diamond_ranked_10min.csv' # 数据路径
data_df = pd.read_csv(csv_data, sep=',') # 读入csv文件为pandas的DataFrame
data_df = data_df.drop(columns='gameId') # 舍去对局标号列


# ###  数据概览
print(data_df.iloc[0]) # 输出第一行数据
data_df.describe() # 每列特征的简单统计信息


# ### 增删特征
drop_features = ['blueGoldDiff', 'redGoldDiff', 
                 'blueExperienceDiff', 'redExperienceDiff', 
                 'blueCSPerMin', 'redCSPerMin', 
                 'blueGoldPerMin', 'redGoldPerMin'] # 需要舍去的特征列
df = data_df.drop(columns=drop_features) # 舍去特征列
info_names = [c[3:] for c in df.columns if c.startswith('red')] # 取出要作差值的特征名字（除去red前缀）
for info in info_names: # 对于每个特征名字
    df['br' + info] = df['blue' + info] - df['red' + info] # 构造一个新的特征，由蓝色特征减去红色特征，前缀为br
# 其中FirstBlood为首次击杀最多有一只队伍能获得，brFirstBlood=1为蓝，0为没有产生，-1为红
df = df.drop(columns=['blueFirstBlood', 'redFirstBlood']) # 原有的FirstBlood可删除

# ### 特征离散化
discrete_df = df.copy() # 先复制一份数据
for c in df.columns[1:]: # 遍历每一列特征，跳过标签列
    if len(df[c].unique()) < 10:
        continue
    discrete_df[c] = pd.qcut(df[c], q=5, labels=False, duplicates='drop')

# ### 数据集准备
all_y = discrete_df['blueWins'].values # 所有标签数据
feature_names = discrete_df.columns[1:] # 所有特征的名称
all_x = discrete_df[feature_names].values # 所有原始特征值，pandas的DataFrame.values取出为numpy的array矩阵

# 划分训练集和测试集
x_train, x_test, y_train, y_test = train_test_split(all_x, all_y, test_size=0.2, random_state=RANDOM_SEED)
print(all_y.shape, all_x.shape, x_train.shape, x_test.shape, y_train.shape, y_test.shape) # 输出数据行列信息


# ###  决策树模型的实现
class Node:
    def __init__(self, feature=None, children=None, prediction=None):
        self.feature = feature
        self.children = children or {}
        self.prediction = prediction
# 定义决策树类
class DecisionTree(object):
    def __init__(self, classes, features, 
                 max_depth=10, min_samples_split=10,
                 impurity_t='entropy'):
        self.classes = classes
        self.features = features
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.impurity_t = impurity_t
        self.root = None # 定义根节点，未训练时为空

    def impurity(self , y):
        counts = np.bincount(y)
        probs = counts / len(y)
        if self.impurity_t == 'entropy':
            return -np.sum([p * np.log2(p) for p in probs if p > 0])
        elif self.impurity_t == 'gini':
            return 1 - np.sum(probs ** 2)

    def gain(self , x , y , feature_idx):
        base_imp = self.impurity(y)
        values = np.unique(x[:,feature_idx])
        new_imp = 0
        for v in values:
            y_sub = y[x[:,feature_idx] == v]
            new_imp += len(y_sub) / len(y) * self.impurity(y_sub)
        return base_imp - new_imp

    def expand_node(self, x, y, depth):
        majority = Counter(y).most_common(1)[0][0]#选择多数的标签
        if depth >= self.max_depth or len(y) < self.min_samples_split or len(set(y)) == 1:
            return Node(prediction=majority)

        gains = [self.gain(x, y, i) for i in range(x.shape[1])]#选择增益最多的
        best = np.argmax(gains)
        if gains[best] <= 1e-6:
            return Node(prediction=majority)

        node = Node(feature=best, prediction=majority)
        for v in np.unique(x[:, best]):
            mask = (x[:, best] == v)#拿出一列
            node.children[v] = self.expand_node(x[mask], y[mask], depth + 1)#x[mask],y[mask]为二维数组，(选中的样本数 * 特征数)
        return node

    def traverse_node(self, node, f):
        if node.prediction is not None and node.feature is None:
            return node.prediction
        v = f[node.feature]
        if v not in node.children:
            return node.prediction  # 兜底
        return self.traverse_node(node.children[v], f)
        
    def fit(self, feature, label):
        assert len(self.features) == len(feature[0]) # 输入数据的特征数目应该和模型定义时的特征数目相同
        self.root = self.expand_node(feature, label, 1)
        
    #单样本其实就是一次比赛结果的预测，多样本是多次比赛结果的预测，一定是这两种情况
    def predict(self, feature):
        assert len(feature.shape) == 1 or len(feature.shape) == 2  # 只能是1维或2维
        if len(feature.shape) == 1:  # 单样本
            return self.traverse_node(self.root, feature)
        return np.array([self.traverse_node(self.root, f) for f in feature])


# 定义决策树模型，传入算法参数
DT = DecisionTree(classes=[0,1], features=feature_names, max_depth=5, min_samples_split=10, impurity_t='gini')

DT.fit(x_train, y_train) # 在训练集上训练
p_test = DT.predict(x_test) # 在测试集上预测，获得预测值
print(p_test) # 输出预测值
test_acc = accuracy_score(p_test, y_test) # 将测试预测值与测试集标签对比获得准确率
print('accuracy: {:.4f}'.format(test_acc)) # 输出准确率
