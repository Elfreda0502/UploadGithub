"""
案例: 演示 随机森林版的 模型预测.
"""

# 导包
import pandas as pd
import pickle
from gz_config import Config
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# import warnings
# warnings.filterwarnings('ignore')

# todo 1. 加载配置文件.
# 设置pandas显示选项
pd.set_option('display.max_columns', None)
# 加载配置
conf = Config()

# todo 2. 加载模型和向量化器.
# 1. 加载随机森林模型(rf)
with open(conf.rf_model_save_path, 'rb') as f:
    rf = pickle.load(f)

# 2. 加载向量化器(tfidf)
with open(conf.tfidf_model_save_path, 'rb') as f:
    tfidf = pickle.load(f)

# todo 3. 读取dev数据集, 分隔符为: ,(默认的, 可以不处理, 因为是csv文件)
df_data = pd.read_csv(conf.process_dev_path)
words = df_data['words']
# print(f'words: {words}')

# todo 4. 对dev数据集进行向量化.
features = tfidf.transform(words)
# print(f'features.shape: {features.shape}')  # (10000, 34930)

# todo 5. 模型预测.
y_pred = rf.predict(features)
# 计算准确率(accuracy): 正确预测的样本 占 总样本的比例
print(f'准确率: {accuracy_score(df_data["label"], y_pred)}')
# 计算精确率(precision): 预测为正类的样本中, 实际为正类的样本所占的比例
# macro: 宏平均 对所有类别平等加权, micro: 微观平均
print(f'精确率: {precision_score(df_data["label"], y_pred, average="macro")}')
# 计算召回率(recall): 实际为正类的样本中, 预测为正类的样本所占的比例
print(f'召回率: {recall_score(df_data["label"], y_pred, average="macro")}')
# 计算F1-score: 精确率和召回率的调和平均数
print(f'F1-score: {f1_score(df_data["label"], y_pred, average="macro")}')

# todo 6.保存结果.
df_data['pred_label'] = y_pred
# print(f'df_data: {df_data}')

# 参1: 结果路径, 参2: 分隔符, 参3: 是否保存索引
df_data.to_csv(conf.model_predict_result, sep='\t', index=False)