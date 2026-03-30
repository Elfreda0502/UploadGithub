# 该py文件用于: 字符级别 自动调参训练

# 导包
import fasttext
from config import Config
import datetime

# todo 1. 加载项目配置
config = Config()

# 获取当前时间, 并格式化为: YYYYMMDD的形式, 例如: 20250914
# current_time = datetime.datetime.now().date().today().strftime('%Y%m%d')
current_time = datetime.datetime.now().strftime('%Y%m%d')

# todo 2. 模型训练: 使用 fasttext训练 字符级 文本分类模型
# fasttext.train_supervised(): fasttext的核心训练函数, 用于训练: 有监督学习分类模型
model = fasttext.train_supervised(
    input=config.process_train_datapath_char,  # 输入的训练数据文件
    autotuneValidationFile=config.process_dev_datapath_char,  # 自动调参的验证集.
    autotuneDuration=120,  # 120秒(自动调参的时长)
    thread=1,       # 训练线程数, 设置为1(单线程) 避免多线程的随机波动, 确保实验可复现.
    verbose=3,      # 日志详细程度(3表示: 输出最详细信息)
    seed=40         # 随机数种子, 确保模型可复现.
)


# todo 3. 模型保存.
model_path = config.ft_model_save_path + f'/model_char_2_auto_{current_time}.bin'
model.save_model(model_path)
print('模型保存成功!')

# todo 5. 模型预测.
print(model.predict('日 本 地 震 海 啸'))

# todo 6. 模型词表查看, 查看模型学习到的词汇表
print(model.words[:10])  # 取前10个字符, 方便我们快速了解词汇表内容.

# todo 7. 模型子词查看.
print(model.get_subwords('日本地震海啸!'))

# todo 8. 查看模型的向量维度.
print(model.get_dimension())

# todo 9. 模型评估.  (样本数, 精确率, 召回率)
print(model.test(config.process_test_datapath_char))
