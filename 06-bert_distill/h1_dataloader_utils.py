# 该.py文件的作用是 -> 获取到 数据集加载器(DataLoader)

# 导包
from tqdm import tqdm  # 进度条
import torch  # 深度学习框架
from torch.utils.data import Dataset, DataLoader  # 数据集对象, 数据加载器对象.
from transformers import BertTokenizer            # BERT分词器
import time  # 时间处理
from config import Config  # 配置文件类
from transformers.utils import PaddingStrategy  # 控制填充策略

# 创建配置文件对象
conf = Config()

# todo 1.定义函数, 加载并处理原始数据集.
def load_raw_data(file_path):
    """
    从指定文件中加载数据, 处理为: '文件内容-标签索引'的元组列表, 供后续封装使用.
    :param file_path: 原始数据文件的路径
    :return: 列表嵌套元组, 例如: [('文本字符串', 标签整数索引), ('文本字符串', 标签整数索引), (...)]
    """
    # 1. 初始化结果列表, 存储处理后的数据.
    result = []
    # 2. 打开源文件.
    with open(file_path, 'r', encoding='utf-8') as f:
        # 3. 一次性读取所有行, 并遍历, 获取到每行数据.
        for line in tqdm(f.readlines(), desc=f'加载原始数据: {file_path}'):
            # 4. 去除首尾空白
            line = line.strip()
            # 5. 如果行数据为空, 则跳过.
            if not line:
                continue
            # 6. 切割, 获取到: 文本 和 标签.
            text, label = line.split('\t')
            # 7. 将标签从字符串转成整数(类别索引), 例如: '3' -> 3
            label = int(label)
            # 8. 封装为元组, 例如: ('文本字符串', 3) 添加到列表中.
            result.append((text, label))
        # 9. 返回处理结果.
        return result


# todo 2. 自定义数据集类(继承PyTorch中的DataSet)
class TextDataset(Dataset):
    # 1. 初始化函数
    def __init__(self, data_list):
        """
        初始化数据集, 接收原始数据列表, 将其转换为 DataLoader可以识别的格式.
        :param data_list: 列表嵌套元组, 例如: [('文本字符串', 3), ('文本字符串', 3), (...)]
        """
        # 1. 初始化父类成员.
        super().__init__()
        # 2. 创建数据集对象.
        self.data_list = data_list


    # 2. 获取数据集大小
    def __len__(self):
        return len(self.data_list)

    # 3. 获取指定索引的数据
    def __getitem__(self, index):
        # text: 文本字符串, label: 标签索引(整数形式)
        text, label = self.data_list[index]
        # 返回结果
        return text, label


# todo 3. 定义整理函数 -> 批量处理某一批次数据
def collate_fn(batch):
    """
    给DataLoader的一个批次(Batch)原始数据进行预处理: 分词, 填充, 转张量
    :param batch: 某一批次的数据, 例如: [(text1, label1), (text2, label2), ...]
    :return: 元组形式, 三个值分别是: input_ids , attention_mask, labels
        input_ids: 分词后token的ID, 形状为: (batch_size, max_length)
        attention_mask: 注意力掩码, 标记有效token和填充token, 形状和 input_ids一致.
        labels: 批次标签, 形状为: (batch_size,)
    """
    # 1. 用zip()函数, 将批次中的文本 和 标签分别加压为两个元组.
    texts, labels = zip(*batch)     # 等价于: texts, labels = [item[0] for ....]
    # 2. 调用BERT分词器的batch_encoder_plus()方法, 对批量文本进行编码.
    text_tokens = conf.tokenizer.batch_encode_plus(
        texts,                              # 要编码的文本列表
        add_special_tokens=True,            # 是否添加特殊标记(CLS, SEP, PAD, ...)
        padding='max_length',               # 填充策略
        max_length=conf.pad_size,           # 最大长度
        truncation=True,                    # 是否截断
        return_attention_mask=True          # 是否返回注意力掩码
    )

    # 3. 获取分词结果
    input_ids = text_tokens['input_ids']
    attention_mask = text_tokens['attention_mask']
    # print(f'input_ids: {input_ids}')
    # print(f'attention_mask: {attention_mask}')

    # 4. 把列表转换为PyTorch张量(模型计算需要张量格式)
    input_ids = torch.tensor(input_ids)
    attention_mask = torch.tensor(attention_mask)
    labels = torch.tensor(labels)

    # 5. 返回结果, 即: 模型可以直接使用的张量.
    return input_ids, attention_mask, labels


# todo 4. 构建数据加载器函数
def build_dataloader():
    """
    构建训练集, 验证集, 测试集的数据加载器(DataLoader)
    :return: 包含单个DataLoader的元素, 顺序为: (train_dataloader, dev_dataloader, test_dataloader)
    """
    # 1. 调用load_raw_data()函数, 加载原始数据 -> 训练集, 验证集, 测试集
    train_data_list = load_raw_data(conf.train_datapath)
    dev_data_list = load_raw_data(conf.dev_datapath)
    test_data_list = load_raw_data(conf.test_datapath)

    # 2. 将上述的数据(列表嵌套元组) -> 自定义数据集对象
    train_dataset = TextDataset(train_data_list)
    dev_dataset = TextDataset(dev_data_list)
    test_dataset = TextDataset(test_data_list)

    # 3. 构建DataLoader: 指定批次大小, 是否打乱数据, 批量处理函数.
    train_dataloader = DataLoader(
        train_dataset,                  # 训练集
        batch_size=conf.batch_size,     # 批次大小
        shuffle=True,                   # 是否打乱数据
        collate_fn=collate_fn           # 批量处理函数(即: 每批次的数据都会被该函数处理一次)
    )

    dev_dataloader = DataLoader(
        dev_dataset,
        batch_size=conf.batch_size,
        shuffle=False,                  # 不打乱数据
        collate_fn=collate_fn
    )

    test_dataloader = DataLoader(
        test_dataset,
        batch_size=conf.batch_size,
        shuffle=False,                  # 不打乱数据
        collate_fn=collate_fn
    )

    # 4. 返回结果
    return train_dataloader, dev_dataloader, test_dataloader



# todo 程序的主入口
if __name__ == '__main__':
    # # 测试1: load_raw_data()函数
    # data_list = load_raw_data(conf.dev_datapath)
    # print(data_list[0])     # 打印验证集第1条数据, 例如: ('体验2D巅峰 倚天屠龙记十大创新概览', 8)
    # print(data_list[1])     # 打印验证集第2条数据, 例如: ('60年铁树开花形状似玉米芯(组图)', 5)
    # print(data_list[:5])    # 打印验证集前5条数据, 例如: [('体验2D巅峰 倚天屠龙记十大创新概览', 8), ('60年铁树开花形状似玉米芯(组图)', 5), ('同步A股首秀：港股缩量回调', 2), ('中青宝sg现场抓拍 兔子舞热辣表演', 8), ('锌价难续去年辉煌', 0)]
    # print('-' * 40)
    #
    # # 测试2: TextDataset()类
    # dataset = TextDataset(data_list)
    # print(dataset[0])      # 获取数据集第1条数据, 例如: ('体验2D巅峰 倚天屠龙记十大创新概览', 8)
    # print(dataset[1])
    #
    # # <class 'list'> <class '__main__.TextDataset'>
    # print(type(data_list), type(dataset))
    # print('-' * 40)

    # 测试3: build_dataloader()函数
    train_dataloader, dev_dataloader, test_dataloader = build_dataloader()
    # print(len(train_  dataloader))    # 训练集总样本数 / 批次数 = train_dataloader / batch_size = 18W / 64 = 2813
    # print(len(dev_da  taloader))      # 验证集总样本数 / 批次数 = dev_dataloader / batch_size = 1W / 64 = 157
    # print(len(test_d  ataloader))     # 测试集总样本数 / 批次数 = test_dataloader / batch_size = 1W / 64 = 157


    # 测试4: 获取数据集加载器对象.
    for i, batch, in enumerate(train_dataloader):
        # 1. 解包批次数据.
        input_ids, attention_mask, labels = batch
        print(f'打印input_ids的具体值: {input_ids.tolist()}')
        print(f'打印input_ids的形状: {input_ids.shape}')

        print(f'attention_mask: {attention_mask.tolist()}') # 打印注意力掩码的具体指.
        print(f'打印attention_mask的形状: {attention_mask.shape}')

        # 打印标签的具体值
        print(f'labels: {labels.tolist()}')
        print(f'labels的形状: {labels.shape}')
        print('\n' * 4)