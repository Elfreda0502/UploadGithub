"""
案例: 演示 随机森林版的 模型预测.
"""

# 导包
import jieba
import pickle
from gz_config import Config


# todo 1. 加载配置文件.
# 加载配置
conf = Config()

# todo 2. 加载模型和向量化器.
# 1. 加载随机森林模型(rf)
with open(conf.rf_model_save_path, 'rb') as f:
    rf = pickle.load(f)

# 2. 加载向量化器(tfidf)
with open(conf.tfidf_model_save_path, 'rb') as f:
    tfidf = pickle.load(f)


# todo 3. 定义预测函数.
def predict_fun(data):      # data: 就是待预测的数据, 例如:  {'text': '2011年全国各地高考各科考试时间汇总'}
    # 1. jieba分词
    words = " ".join(jieba.lcut(data['text'])[:30])
    # 2. 向量化.
    features = tfidf.transform([words])
    # 3. 预测.
    y_pred_id = rf.predict(features)[0]
    # 4. 设置标签索引 和 名称对应关系.       # {0:'finance', 1: 'realty', ...}
    id2class = {i:line.strip() for i, line in enumerate(open(conf.class_datapath, encoding='utf-8'))}
    # 5. 获取预测类别的名称.
    y_pred = id2class[y_pred_id]        # education
    # 6. 给原始数据新增1列(预测的类别), 并返回.
    data['pred_class'] = y_pred
    # 7. 返回结果, 例如: {'text': '2011年全国各地高考各科考试时间汇总', 'pred_class': 'education'}
    return data


# todo 4. 测试
if __name__ == '__main__':
    data = {'text': '2011年全国各地高考各科考试时间汇总'}
    result = predict_fun(data)
    print(result)

