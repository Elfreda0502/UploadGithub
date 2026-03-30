 # 该.py文件用于搭建 BERT分类模型的 -> 充当: 教师模型类.

# 导包
import torch                        # 深度学习框架
import torch.nn as nn               # 神经网络模块
from transformers import BertModel, BertTokenizer  # Bert模型, 分词器
from config import Config           # 配置文件类

# todo 1.加载配置文件信息.
conf = Config()     # 后续可以通过 conf. 的形式, 获取配置信息.


# todo 2. 定义BERT分类模型框架
class BertClassifier(nn.Module):
    # todo 2.1 初始化模型.
    def __init__(self):
        # 1. 继承父类初始化方法
        super().__init__()

        # 2. 加载BERT模型
        self.bert = BertModel.from_pretrained(conf.bert_path)

        # 3. 定义全连接分类层, 输入维度: 768(BERT的隐藏层维度), 输出维度: conf.num_classes(10个类别)
        self.fc = nn.Linear(conf.bert_config.hidden_size, conf.num_classes)

    # todo 2.2 定义前向传播方法.
    def forward(self, input_ids, attention_mask):
        # 1. 将Token ID 和 注意力掩码输入BERT模型, 获取模型输出(包含: last_hidden_state, pooler_output)
        # input_ids: 输入的Token ID张量, 形状为: [batch_size, 序列长度max_length]
        # attention_mask: 输入的注意力掩码张量, 形状为: [batch_size, 序列长度max_length]
        outputs = self.bert(input_ids, attention_mask)
        # print(f'outputs: {outputs}')

        # 2. 获取分类结果, 形状为: [batch_size, num_classes]
        # outputs[1]: 池化层输出, 形状为: [batch_size, 768] -> torch.Size([2, 768])
        logits = self.fc(outputs[1])
        # print(f'logits: {logits}')

        # 3. 返回分类结果.
        return logits



# todo 3.测试代码
if __name__ == '__main__':
    # 1. 加载BERT分词器, 将文本 -> 模型可识别的 Token ID
    tokenizer = BertTokenizer.from_pretrained(conf.bert_path)

    # 2. 准备示例文本, 用于测试 模型的输入数据.
    texts = ['王者荣耀', '今天天气很好']

    # 3. 编码文本 -> 将原始文本转成模型所需要的 的输入数据(Token ID, Attention Mask)
    encode_inputs = tokenizer(
        texts,                      # 待编码的文本列表
        max_length=9,               # 最大长度, 目标序列长度, 超过就截断, 不足就填充
        padding='max_length',       # 填充方式
        truncation=True,            # 开启截断
        return_tensors='pt'         # 返回(PyTorch)张量
    )

    # 4. 提取模型输入张量: 从编码结果中拿出 Token ID 和 Attention Mask张量.
    input_ids = encode_inputs['input_ids']
    attention_mask = encode_inputs['attention_mask']
    print(f'input_ids: {input_ids}')
    print(f'attention_mask: {attention_mask}')
    print('-' * 40)

    # 5. 创建自定义的BERT分类模型
    model = BertClassifier()
    # 6. 模型前向传播, 获取模型输出.
    logits = model(input_ids, attention_mask)
    print(f'logits: {logits}')      # 未归一化的分类得分(每行对应1个样本, 每列对应1个类别)
    print('-' * 40)

    # 7. 计算类别概率, 对logits做softmax()归一化, 得到每个类别在[0, 1]区间的概率
    probs = torch.softmax(logits, dim=-1)
    print(f'probs: {probs}')
    print('-' * 40)

    # 8. 获取预测分类: 即概率最大的类别索引.
    preds = torch.argmax(probs, dim=-1)
    print(f'preds: {preds}')        # 最终结果: 每个样本的预测类别索引.
