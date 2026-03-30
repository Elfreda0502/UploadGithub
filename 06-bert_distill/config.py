# 导包

import torch            # 用于深度学习模型的构建和训练
import datetime         # 时间处理包

# 导入Bert相关组件, BertModel(BERT模型的主体), BertTokenizer(BERT的分词器), BertConfig(BERT的配置)
from transformers import BertModel, BertTokenizer, BertConfig


# todo 1.定义变量, 记录当前时间(年月日格式)
current_date = datetime.datetime.now().strftime('%Y%m%d')   # 例如: 20250914


# todo 2. 定义配置文件类, 集中管理 模型和训练所需的参数.
class Config(object):
    def __init__(self):
        """
        配置类，包含模型和训练所需的各种参数。
        """

        # 1. 基础的模型信息, 例如: 模型名称
        self.model_name = "bert"  # 模型名称

        # 2. 路径配置.
        # 根目录
        self.root_path = 'D:/PycharmProjects/TMF_Project/'
        # 原始数据路径
        self.train_datapath = self.root_path + '01-data/data/train.txt'
        self.test_datapath = self.root_path + '01-data/data/test.txt'
        self.dev_datapath = self.root_path + '01-data/data/dev.txt'
        # 类别文档
        self.class_path = self.root_path + "01-data/data/class.txt"

        # 从类别文件中读取所有类别的名称.
        self.class_list = [line.strip() for line in open(self.class_path, encoding="utf-8")]  # 类别名单

        # 模型训练保存路径
        self.model_save_path = self.root_path + "06-bert_distill/save_models/bert_classifier_model.pt"  # 模型训练结果保存路径

        # 模型训练+预测的时候, 指定设备
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # 训练设备，如果GPU可用，则为cuda，否则为cpu


        # 3. BERT模型的相关配置.
        self.bert_path = self.root_path + "/04-bert/bert-base-chinese"  # 预训练BERT模型的路径
        self.bert_model = BertModel.from_pretrained(self.bert_path) # 加载预训练BERT模型
        self.tokenizer = BertTokenizer.from_pretrained(self.bert_path)  # BERT模型的分词器
        self.bert_config = BertConfig.from_pretrained(self.bert_path)  # BERT模型的配置


        # 4. 训练参数配置.
        self.num_classes = len(self.class_list)  # 类别数
        self.num_epochs = 2  # epoch数
        self.batch_size = 64  # mini-batch大小
        self.pad_size = 32    # 每句话处理成的长度(短填长切)
        self.learning_rate = 5e-5  # 学习率

        # 5. 蒸馏模型存放地址
        self.bert_model_distill_model_path = self.root_path + "06-bert_distill/save_models/bert_classifier_bilstm_model.pt"  # 模型训练结果保存路径

        # 6. bert模型蒸馏 -> BiLSTM模型的参数配置.
        self.embed_size = 128
        self.hidden_size_lstm = 256
        self.lstm_learning_rate = 1e-3
        self.dropout = 0.3
        self.num_layers = 3


# todo 3.主函数
if __name__ == '__main__':
    # 1. 创建Config类的对象, 加载所有配置参数
    conf = Config()

    # 2. 打印设备信息(GPU/CPU)
    print(conf.device)

    # 3. 打印类别列表
    print(conf.class_list)

    # 4. 查看保存模型的路径.
    print(conf.bert_model_quantization_model_path)

    # 5. 打印分词器对象(验证分词器是否成功加载)
    print(conf.tokenizer)

    # 需求: 测试分词器的 token转id功能, 将: ['你', '好', '文', '俊'] 转换为对应的id
    my_input_ids = conf.tokenizer.convert_tokens_to_ids(['你', '好', '文', '俊'])
    print(my_input_ids)     # [872, 1962, 3152, 916]