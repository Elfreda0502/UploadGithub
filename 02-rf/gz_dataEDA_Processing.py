
"""
EDA:
    探索性数据分析(Exploratory Data Analysis), 是一种分析数据集以总结其主要特征, 一般和图形表示法结合使用.
    大白话: 就是分析数据的, 可以帮我们发现模式, 检测异常, 测试假设, 从而对数据集有一个直观的了解.
"""


# 导包
import pandas as pd
import jieba
from gz_config import Config

# 1. 初始化配置文件.
conf = Config()

# 2. 设置处理的及分析的文件路径, 默认为: train.txt
current_path = conf.train_datapath
current_path = conf.test_datapath
current_path = conf.dev_datapath

# 3. 读取数据.
df = pd.read_csv(current_path, sep='\t', names=['text', 'label'])
print(f'df: {df.head()}')


# 4.进行分词预处理, 对输入文本进行分词, 并限制前30个词.
def cut_sentence(s):
    # 对输入文本进行分词, 并限制前30个词.
    return ' '.join(list(jieba.cut(s))[:30])      # ['词1', '词2', '词3'...]

# 5. 对每行文本进行分词并存储到words类.
df['words'] = df['text'].apply(cut_sentence)
print(f'df: {df.head()}')


# 6. 保存处理以后的数据.
if 'train' in  current_path :
    df.to_csv(conf.process_train_path, index=False)
    print(f'train数据保存成功, 存储路径为: {conf.process_train_path}!')

elif 'test' in  current_path :
    df.to_csv(conf.process_test_path, index=False)
    print(f'test数据保存成功, 存储路径为: {conf.process_test_path}!')

elif 'dev' in  current_path :
    df.to_csv(conf.process_dev_path, index=False)
    print(f'dev数据保存成功, 存储路径为: {conf.process_dev_path}!')

