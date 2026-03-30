# 导包

import pandas as pd     # 数据读取和处理
import pickle           # 用于模型和向量化器的序列化保存
from sklearn.feature_extraction.text import TfidfVectorizer # 将文本转成数值特征(可以理解为: 词向量)
from sklearn.model_selection import train_test_split        # 训练集和测试集的划分
from sklearn.ensemble import RandomForestClassifier         # 随机森林分类器
# 模型评估指标: 准确率, 精准率, 召回率, F1-score, 分类报告, 混淆矩阵
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix
from gz_config import Config        # 配置文件
from tqdm import tqdm               # 进度条
# 导入三种集成学习分类器: 随机森林, AdaBoost, GBDT
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier


# todo 1. Pandas的基础设置.
pd.set_option('display.expand_frame_repr', False)  # 避免宽表格换行
pd.set_option('display.max_columns', None)         # 确保所有列可见
# 实例化配置对象, 用于获取: 文件路径等配置.
conf = Config()

# todo 2.读取训练数据.
# 2.1 pandas读取训练数据, 仅取前2W条数据(控制数据量, 加快训练速度)
df = pd.read_csv(conf.process_train_path)[:20000]

# 2.2 提取特征列'words' -> 预处理后的文本数据.
words = df['words']
# print(f'words: {words[:5]}')

# 2.3 提取标签列'label' -> 训练标签.
labels = df['label']
# print(f'labels: {labels[:5]}')

# 2.4 打印前5行.
print(f'df: {df.head()}')


# todo 3. 文本特征提取(TF-IDF向量化)
# 3.1 读取停用词文件(停用词: 对分类无意义的词, 如: 的, 是等这些词), 按行分割为列表.
stop_words = open(conf.stop_words_path, encoding='utf-8').read().split()
# print(f'stop_words: {stop_words[:5]}', stop_words[-5:])

# 3.2 初始化 TF-IDF向量化器, 指定停用词列表(过滤停用词的)
tfidf = TfidfVectorizer(stop_words=stop_words)

# 3.3 对文本特征进行拟合(学习词汇表) 并转换 TF-IDF特征矩阵
features = tfidf.fit_transform(words)
print(f'features: {features.shape}')        # (20000, 34930), 行索引, 列索引

# 3.4 查看生成的词汇表(特征名称)列表
print(list(tfidf.get_feature_names_out()))      # [...'龚方雄', '龚炜任', '龚蓓'...], 大小是: 34930

# 3.5 查看词汇表中 词和索引的映射关系(字典形式: 词 -> 列索引)
print(tfidf.vocabulary_)        # {'中华': 3987, '女子': 12910, '学院': 13359,...}
# 3.6 再次确认词汇表的大小.
print(len(tfidf.vocabulary_))   # 34930


# todo 4. 模型的训练和评估
# 4.1 划分训练集和测试集.
x_train, x_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)

# 4.2 初始化随机森林分类器
model = RandomForestClassifier()

# 4.3 训练模型, 并使用tqdm 显示进度条
for _ in tqdm(range(1)):
    model.fit(x_train, y_train)

# 4.4 模型预测和评估.
y_pred = model.predict(x_test)
# 4.5 打印预测结果
print(f'预测结果: {y_pred}')
print(f'准确率: {accuracy_score(y_test, y_pred):.4f}')
# 打印微平均精确率(所有类别合集计算的精确率, 适用于 多类别不平衡的场景)
print(f'精准率(micro): {precision_score(y_test, y_pred, average="micro"):.4f}')
print(f'召回率(micor): {recall_score(y_test, y_pred, average="micro"):.4f}')
print(f'F1-score(micro): {f1_score(y_test, y_pred, average="micro"):.4f}')


# todo 5. 模型和向量化器保存.
# 5.1 保存训练好的 随机森林模型.
with open(conf.rf_model_save_path, 'wb') as f:
    pickle.dump(model, f)

# 5.2 保存训练好的 TF-IDF向量化器(后续预测时, 需要使用同一个向量化器转换新文本)
with open(conf.tfidf_model_save_path, 'wb') as f:
    pickle.dump(tfidf, f)

# 5.3 提示保存成功.
print(f'模型和向量化器, 保存成功!')
