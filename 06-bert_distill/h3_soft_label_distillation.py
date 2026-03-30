# 该.py文件的作用 -> 模型蒸馏之 -> 软标签

import torch                            # 深度学习框架, 提供张量计算, 神经网络构建等...
import torch.nn as nn                   # 神经网络模块, 损失函数, 网络层
from torch.optim import AdamW           # 优化器, 适用于Transformer类模型的优化器, 缓解梯度消失问题.

# 用于评估模型性能的库
from sklearn.metrics import classification_report, f1_score, accuracy_score, precision_score, recall_score
from tqdm import tqdm                   # 进度条
from config import Config               # 配置文件类
from model2dev_utils import model2dev   # 导入自定义验证函数(例如: 精确率, 召回率...)
from h1_dataloader_utils import build_dataloader    # 获取数据集加载器


from h1_bert_classifier_model import BertClassifier     # 教师模型, 导入BERT分类模型
from h2_bilstm_classifier_model import BiLSTMClassifier # 学生模型, 导入Bilstm分类模型
import torch.nn.functional as F         # 函数模块, 激活函数, 损失函数等.


# 忽略的警告信息
import warnings
warnings.filterwarnings("ignore")


# todo 1. 加载配置对象，包含模型参数、路径等
conf = Config()

# todo 2. 定义模型训练函数, 封装完整的训练流程(数据加载, 模型训练, 验证, 保存)
def model2train():
    # 1. 准备训练/验证/测试数据, 获取其对应的 数据集加载器.
    train_loader, dev_loader, test_loader = build_dataloader()

    # 2. 初始化并配置模型.
    # 2.1 实例化教师模型.
    # 创建教师对象.
    teacher_model = BertClassifier().to(conf.device)
    # 加载教师模型的预训练参数.
    teacher_model.load_state_dict(torch.load(conf.model_save_path))

    # 2.2 创建学生模型.
    student_model = BiLSTMClassifier().to(conf.device)

    # 3. 定义损失函数, 使用: 交叉熵损失.
    criterion = nn.CrossEntropyLoss()

    # 4. 定义优化器, 使用: AdamW 优化器.
    optimizer = AdamW(student_model.parameters(), lr=conf.lstm_learning_rate)

    # 5. 初始化最优F1分数, 用于筛选性能最好的模型(即: 初始值为0, 后续更新)
    best_f1 = 0.0
    T = 2.0         # 温度参数, T越大概率分布越平滑.
    alpha = 0.7     # 软标签的权重参数(就是我们讲解的 α)

    # 6. 具体的训练过程, 外层循环表示训练的轮数. 每轮都要遍历所有训练数据.
    for epoch in range(conf.num_epochs):
        # 6.1 设置模型为训练模式.
        teacher_model.eval()
        student_model.train()

        # 6.2 初始化训练过程中的统计变量.
        total_loss = 0.0        # 累计当前批次的损失值.
        train_preds, train_labels = [], []      # 存储当前批次的预测结果和真实标签(用于计算指标)

        # 6.3 内层循环, 遍历训练集, 获取到每个批次, 逐批次更新模型.
        for i, batch in enumerate(tqdm(train_loader, desc="(训练集)蒸馏中...")):
            # 6.3.1 提取批次数据并移动到指定设备
            input_ids, attention_mask, labels = batch
            input_ids = input_ids.to(conf.device)            # token id序列(模型输入)
            attention_mask = attention_mask.to(conf.device)  # 表示哪些token有效, 哪些无效
            labels = labels.to(conf.device)                  # 标签(模型输出)

            # ----------------------------- 学生模型前向传播 -----------------------------
            # 6.3.2 前向传播：模型预测
            student_logits = student_model(input_ids, attention_mask=attention_mask)        # logits: 未归一化的分类得分, 形状: [batch_size, num_classes]
            # 6.3.3 获取当前批次的预测标签.
            student_labels = torch.argmax(student_logits, dim=1)  # 概率最大的类别的索引

            # ----------------------------- 老师模型前向传播 -----------------------------
            with torch.no_grad():
                teacher_logits = teacher_model(input_ids, attention_mask=attention_mask)
                teacher_labels = torch.argmax(teacher_logits, dim=1)

            # 6.3.4 计算软标签损失: KL散度损失, 衡量学生和教师的平滑概率分布的差异.
            teacher_log_soft_labels = F.log_softmax(teacher_logits / T, dim=1)
            student_log_soft_labels = F.log_softmax(student_logits / T, dim=1)

            # 6.3.5 具体的计算KL散度的动作.
            soft_loss = F.kl_div(
                student_log_soft_labels,    # 学生的平滑对数概率(预测分布)
                teacher_log_soft_labels,    # 教师的平滑对数概率(真实分布)
                reduction='batchmean',      # 批次平均
                log_target=True             # 是否使用对数概率作为目标
            )

            # 6.3.6 计算硬损失标签.
            hard_loss1 = criterion(student_logits, teacher_labels)
            # 6.3.7 计算总蒸馏损失
            loss = (1 - alpha) * hard_loss1 + alpha * (T * T) * soft_loss

            # 6.3.8 累计统计变量: 更新损失和标签列表.
            # 6.3.4 累计损失值.
            total_loss += loss.item()

            # 6.3.6 存储当前批次的预测标签和真实标签.
            train_preds.extend(student_labels.cpu().tolist())  # 需要先转回CPU(避免GPU内存占用), 在转回列表
            train_labels.extend(labels.cpu().tolist())
            # 6.3.7 梯度清零, 保证参数更新准确
            optimizer.zero_grad()
            # 6.3.8 反向传播：计算梯度, 链式求导.
            loss.backward()
            # 6.3.9 参数更新(梯度更新), 基于梯度和优化器规则(AdamW)更新模型权重
            optimizer.step()

            # 6.4 每10个批次或者轮次末尾, 计算并打印训练集指标(监控训练效果)
            # 条件: 当前批次是10的倍数, 或者当前轮次的最后1批.
            if (i + 1) % 10 == 0 or i == len(train_loader) - 1:
                # 6.4.1 计算训练集指标: 准确率
                acc = accuracy_score(train_labels, train_preds)
                # 6.4.2 Macro-F1值(多分类任务常用)
                f1 = f1_score(train_labels, train_preds, average='macro')
                # 6.4.3 计算当前累积的平均损失: 总损失 / 累计批次数量.
                batch_count = i % 10 + 1        # 累计批次数量(1 ~ 10)
                avg_loss = total_loss / batch_count
                # 6.4.4 打印训练日志.
                print(f'轮次:{epoch + 1}, 批次: {i + 1}, 损失: {avg_loss:.4f}, 准确率: {acc:.4f}, Macro-F1: {f1:.4f}')
                # 6.4.5 重置统计变量.
                total_loss = 0.0
                train_preds, train_labels = [], []

            # 6.5 每100个批次或者轮次末尾, 验证模型效果(在验证集评估模型并保存最优模型)
            if (i + 1) % 100 == 0 or i == len(train_loader) - 1:
                # 6.5.1 调用验证函数, 计算验证集的评估报告.
                report, f1score, accuracy, precision, recall = model2dev(student_model, dev_loader, conf.device)
                # 6.5.2 打印验证集日志, 查看详细评估报告.
                print(f'验证集的f1: {f1score:.4f}, 准确率: {accuracy:.4f}, 精准率: {precision:.4f}, 召回率: {recall:.4f}')

                # 6.5.3 重置模型为训练模式.
                student_model.train()

                # 6.5.4 保存模型.
                if f1score > best_f1:
                    best_f1 = f1score       # 更新历史最佳F1分数
                    torch.save(student_model.state_dict(), conf.bert_model_distill_model_path)
                    print(f'保存最优学生模型成功, 当前f1分数: {best_f1}')


# todo 3. 主程序入口.
if __name__ == '__main__':
    model2train()