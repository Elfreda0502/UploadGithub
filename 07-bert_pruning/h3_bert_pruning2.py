"""
BERT 全局非结构化剪枝：对所有 encoder 层注意力权重剪枝 30%，L1 范数。
这个是纯剪枝的动作, 只保留了核心代码, 你要是觉得之前那个代码多, 就看这个.
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


    # 14. 保存剪枝后的模型
    torch.save(model.state_dict(), conf.bert_model_pruning_model_path)
    print('已保存剪枝后的模型!')
