# 该py文件用于: 单词级别 默认参数训练


# 导包
import fasttext
from config import Config

# 1. 加载配置文件
config = Config()

# 2. 模型训练.
model = fasttext.train_supervised(input=config.process_train_datapath_word, seed=42)    # seed: 随机种子

# 3. 模型保存.
model_path = config.ft_model_save_path + '/model_word_1_default.bin'
model.save_model(model_path)

# 4. 模型预测
print(model.predict('日本 地震 海啸'))

# 5. 模型词表查看.
print(model.words[:10])

# 6. 模型子词查看.
print(model.get_subwords('日本'))

# 7. 模型的词向量维度.
print(model.get_dimension())    # 默认: 100维度

# 8. 模型评估       (样本数, 精确率, 召回率)
print(model.test(config.process_test_datapath_word))