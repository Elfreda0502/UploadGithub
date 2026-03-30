"""
BERT 全局非结构化剪枝：对所有 encoder 层注意力权重剪枝 30%，L1 范数。
"""
import torch                            # Pytorch框架, 封装了张量计算等相关函数.
import torch.nn.utils.prune as prune    # PyTorch官方剪枝工具库, 提供了多种剪枝的方法.
from h2_bert_classifier_model import BertClassifier # 自定义的BERT分类框架
from h1_dataloader_utils import build_dataloader    # 导入数据集加载器
from model2dev_utils import model2dev               # 导入模型评估函数
from config import Config                           # 导入配置类

import warnings
warnings.filterwarnings("ignore")


# todo 1.加载配置.
conf = Config()

# todo 2.工具函数 -> 计算 Bert模型的 Encoder层 query权重的稀疏度(大白话: 零值参数占比)
def compute_sparsity(model):
    """
    计算BERT模型中Encoder层query权重的稀疏度, 即:  零值参数数量 / 总参数数量
    :param model: 待评估的BERT分类模型
    :return: 稀疏度 -> float类型, 范围: [0, 1]
    """
    # 1. 定义变量, 分别记录: 总参数数量(所有encoder层 query权重的总元素数)
    total_params = 0
    # 零值参数数量(所有encoder层 query权重的值为0的元素数)
    zero_params = 0
    # 定义变量, 记录 BERT分类模型的 encoder的层数(默认12层)
    layer_num = len(model.bert.encoder.layer)       # 简单写法: layer_num = 12

    # 2. 遍历所有encoder层, 计算 query权重的稀疏度
    for i in range(layer_num):      # i就代表具体的每一层.
        # 2.1 获取当前encoder层 attention模块中 query权重张量
        weight = model.bert.encoder.layer[i].attention.self.query.weight
        # 2.2 统计权重中'值为0'的元素数量.
        zero_params += (weight == 0).sum().item()
        # 2.3 统计权重中 总的元素数量.
        total_params += weight.numel()

    # 3. 返回结果.
    return zero_params / total_params if total_params > 0 else 0  # 避免除零


# todo 3. 工具函数 -> 打印权重张量的局部内容(方便观察剪枝前后参数分布变化)
def print_weights(weight, name, rows=5, cols=5):
    """
    打印权重张量的前rows行, 前cols列 局部区域, 直观展示参数数值分布.
    :param weight: 待打印的权重张量
    :param name: 权重张量的名称, 例如: layer0 query权重
    :param rows: 要打印的行数
    :param cols: 要打印的列数
    :return: 无
    """
    print(f'\n{name} (前 {rows} × {cols} 区域): ')     # 例如: layer0 query (前 5 × 5 区域):
    print(weight[:rows, :cols])     # 切片获取即可.


# todo 4. 具体的剪枝过程.
if __name__ == '__main__':
    # 1. 加载数据.
    train_loader, dev_loader, test_loader = build_dataloader()
    # 2. 加载预训练的BERT分类模型.
    model = BertClassifier()
    # 3. 加载模型参数.
    # strict=False 兼容部分键不匹配.
    model.load_state_dict(torch.load(conf.model_save_path, weights_only=False), strict=False)
    # 4. 移动模型到GPU环境.
    model = model.to(conf.device)

    # 5.剪枝前: 打印模型局部参数, 观察原始权重分布.
    print('================================= 剪枝前模型 ================================= ')
    print_weights(
        model.bert.encoder.layer[0].attention.self.query.weight,
        'layer[0].attention.self.query.weight 剪枝前'
    )

    # 6. 剪枝前: 计算并打印权重稀疏度.
    sparsity = compute_sparsity(model)
    print(f'剪枝前的稀疏度: {sparsity:.4f}')

    # 7. 剪枝前: 在验证集上评估模型性能.
    report, f1score, accuracy, precision, recall = model2dev(model, dev_loader, conf.device)
    print(f'剪枝前: 准确率: {accuracy:.4f}, F1: {f1score:.4f}')

    print('================================= 剪枝开始 ================================= ')
    # 8. 定义全局非结构化剪枝的目标参数: 所有encoder层的 query权重, 格式为: [(第1层权重参数), (第2层的权重参数)...]
    parameters_to_prune = [
        # (model.bert.encoder.layer[i].attention.self.query, 'weight') for i in range(len(model.bert.encoder.layer))
        (model.bert.encoder.layer[i].attention.self.query, 'weight') for i in range(12)     # 默认: 一共12层
    ]
    # 9. 执行全局非结构化剪枝.
    # 参1: 待剪枝的参数组, 参2: 剪枝方法(按L1范数剪枝, 即: 参考绝对值的大小), 参3: 剪枝比例(0-1之间), 这里是移除30%的参数, 保留70%的参数.
    prune.global_unstructured(parameters_to_prune, pruning_method=prune.L1Unstructured, amount=0.3)

    # 10. 固话剪枝: 将剪枝后的稀疏权重永久保存到模型中(移除剪枝的临时掩码结构)
    for module, param in parameters_to_prune:
        prune.remove(module, param)     # 调用后, 模型仅保留剪枝后的权重, 不再含 剪枝相关的临时结构.

    print('================================= 剪枝结束 ================================= ')

    # 11. 剪枝后: 打印模型局部参数, 观察剪枝后的权重分布(会出现大量零值)
    print('================================= 剪枝后模型 ================================= ')
    print_weights(
        model.bert.encoder.layer[0].attention.self.query.weight,
        'layer[0].attention.self.query.weight 剪枝后'
    )

    # 12. 剪枝后: 计算并打印权重稀疏度(应该: 明显高于剪枝前)
    sparsity = compute_sparsity(model)
    print(f'剪枝后的稀疏度: {sparsity:.4f}')

    # 13. 剪枝后: 在验证集上评估模型性能
    report, f1score, accuracy, precision, recall = model2dev(model, dev_loader, conf.device)
    print(f'剪枝后: 准确率: {accuracy:.4f}, F1: {f1score:.4f}')

    # 14. 保存剪枝后的模型
    torch.save(model.state_dict(), conf.bert_model_pruning_model_path)
    print('已保存剪枝后的模型!')
