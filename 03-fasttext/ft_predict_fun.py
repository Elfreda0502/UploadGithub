"""
案例: 演示 随机森林版的 模型预测.
"""

# 导包
import fasttext
import jieba
import pickle
from config import Config


# todo 1. 加载配置文件.
# 加载配置
conf = Config()

# todo 2. 加载模型.
model = fasttext.load_model(conf.ft_model_save_path + '/model_word_2_auto_20250914.bin')

# todo 3. 定义预测函数.
def predict_fun(data):  # data就是待预测的数据, 例如: {'text': '2011年全国各地高考各科考试时间汇总'}
    # 1. 使用jieba对文本做分词处理.
    text = " ".join(jieba.lcut(data['text']))
    # 2. 使用加载的fasttext模型 对处理后的文本进行 预测, 获取预测结果.
    re = model.predict(text)        # re的结果格式为: (('__label__stocks',), array([0.99470019]))

    # 3. 处理预测结果, 获取类别标签.
    # data['pred_class'] = re[0][0][9:]       # 切片方式
    data['pred_class'] = re[0][0].replace('__label__', '')   # 效果同上, 替换方式

    # 4. 返回包含 原始文本 和 预测类别的结果字典
    return data     # 格式为: {'text': '预测文本', 'pred_class': '预测类别'}

# todo 4. 测试
if __name__ == '__main__':
    data = {'text': '2011年全国各地高考各科考试时间汇总'}
    result = predict_fun(data)
    print(result)

