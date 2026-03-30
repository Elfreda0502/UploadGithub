# 该py文件用于: 字符级别 默认参数训练

# 导包
import fasttext
from config import Config
import datetime
import os

# todo 1. 加载项目配置
config = Config()


# todo 2. 模型训练: 使用 fasttext训练 字符级 文本分类模型
# fasttext.train_supervised(): fasttext的核心训练函数, 用于训练: 有监督学习分类模型
model = fasttext.train_supervised(
    input = config.process_train_datapath_char, # 输入的训练数据文件
    dim = 10,   # 词向量维度, 维度越小, 计算越快.
    minn = 1,   # 子句最小长度
    maxn = 4    # 子句最大长度, 控制子词特征最大的粒度(例如: 日本地震 -> 可拆出 4字符子词)
)

# todo 3. 打印模型训练后的关键信息.
# 获取字符'日'的向量表示(长度: 10维), 查看数字的特征, 例如:
# [ 0.06251448  0.21214555 -0.5582269  -0.2776528  -0.40804806 -0.2712767, -0.03183828 -0.10957453  0.31537786  0.19355372]
print(model.get_word_vector('日'))
# 获取模型训练到的 所有类别标签
print(model.labels)         # ['__label__stocks', '__label__science'...]
# 获取模型训练到的 类别数量
print(len(model.labels))    # 10

# 获取模型训练到的 所有单词及对应的词频
print(model.get_words(include_freq=True))

# 用zip()函数, 配对: 字符和频率, 转为列表, 方便查看每个字符对应的出现次数.
print(list(zip(*model.get_words(include_freq=True))))       # [('</s>', 180000), ('0', 60319),  ('：', 28269), ('大', 26024)...]
print('-' * 40)


# todo 4. 模型保存.
model_path = os.path.join(config.ft_model_save_path, 'model_char_1_default.bin')
model.save_model(model_path)
print('模型保存成功!')


# todo 5. 模型预测.
print(model.predict('日 本 地 震 海 啸'))

# todo 6. 模型词表查看, 查看模型学习到的词汇表
print(model.words[:10])     # 取前10个字符, 方便我们快速了解词汇表内容.

# todo 7. 模型子词查看.
print(model.get_subwords('日本地震海啸!'))

# todo 8. 查看模型的向量维度.
print(model.get_dimension())

# todo 9. 模型评估.  (样本数, 精确率, 召回率)
print(model.test(config.process_test_datapath_char))