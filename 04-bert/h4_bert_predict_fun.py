# 该 .py文件的作用是 -> 预测分类的 函数版, 对接后续的 API 和 APP版.


# 导包.
from config import Config
import torch
from h2_bert_classifier_model import BertClassifier

# 压制警告.
import warnings
warnings.filterwarnings("ignore")


# todo 1.加载全局配置.
conf = Config()

# todo 2. 准备BERT预测模型.
model = BertClassifier()
# 加载预训练模型的全虫.
model.load_state_dict(torch.load(conf.model_save_path, weights_only=True))
model.to(conf.device)
model.eval()        # 设置模型为评估模式.

# todo 3. 定义预测函数, 接收文本数据, 返回分类结果.
def predict_fun(data_dict):
    """
    接收包含文本的字典, 通过BERT模型预测文本类别, 返回带预测结果的字典.
    :param data_dict: 输入字典, 格式为: {'text': '待预测文本内容'}
    :return:  {'text': '待预测文本内容', 'pred_class': '文本类别'}
    """
    # 1. 提取输入文本, 获取待预测的字符串.
    text = data_dict['text']
    # 2. 文本编码, 将原始文本 -> BERT模型可识别的token id
    text_tokens = conf.tokenizer.batch_encode_plus(
        [text],                       # 将传入的文本转成列表
        padding='max_length',         # 填充策略
        max_length=conf.pad_size,     # 最大长度
        pad_to_max_length=True,       # 是否(强制)填充到最大长度
    )
    # 3. 提取模型所需要的特征.
    input_ids =text_tokens['input_ids']
    attention_mask = text_tokens['attention_mask']
    # 4. 转换数据并指定设备.
    input_ids = torch.tensor(input_ids).to(conf.device)
    attention_mask = torch.tensor(attention_mask).to(conf.device)

    # 5. 模型预测动作.
    with torch.no_grad():
        # 5.1 前向传播
        logits = model(input_ids, attention_mask=attention_mask)
        # 5.2 获取预测类别索引.
        preds = torch.argmax(logits, dim=-1)
        # 5.3 转换索引格式, 从PyTorch张量  -> Python的标量.
        pred_idx = preds.item()
        # 5.4 获取预测类别. 根据索引 -> 类别名
        pred_class = conf.class_list[pred_idx]
        # 5.5 打印结果
        # print(f'pred_class: {pred_class}')

        # 5.6 添加预测结果到字典, 并返回.
        data_dict['pred_class'] = pred_class

    # 6. 返回结果
    return data_dict


# todo 4. 测试代码.
if __name__ == '__main__':
    # 1. 创建测试数据集.
    data_dict = {'text': '体验2D巅峰 倚天屠龙记十大创新概览'}
    # 2. 调用预测接口, 并打印结果.
    print(predict_fun(data_dict))
