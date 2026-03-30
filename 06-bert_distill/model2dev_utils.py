import torch
from sklearn.metrics import classification_report, f1_score, accuracy_score, precision_score, recall_score
from tqdm import tqdm

def model2dev(model, data_loader, device):
    """
    在验证或测试集上评估 BERT 分类模型的性能。
    参数：
        model (nn.Module): BERT 分类模型。
        data_loader (DataLoader): 数据加载器（验证或测试集）。
        device (str): 设备（"cuda" 或 "cpu"）。
    返回：
        tuple: (分类报告, F1 分数, 准确度, 精确度，召回率)
            - report: 分类报告（包含每个类别的精确度、召回率、F1 分数等）。
            - f1score: 微平均 F1 分数。
            - accuracy: 准确度。
            - precision: 微平均精确度
            - recall: 微平均召回率
    """


    # todo 1. 设置模型为评估模式（禁用 dropout,并改变batch_norm行为）
    model.eval()
    # 2. 初始化列表，存储预测结果和真实标签
    all_preds, all_labels = [], []
    # 3. todo torch.no_grad()禁用梯度计算以提高效率并减少内存占用
    with torch.no_grad():
        # 4. 遍历数据加载器，逐批次进行预测
        for i, batch in enumerate(tqdm(data_loader, desc="验证集评估中...")):
            # 4.1 提取批次数据并移动到设备
            input_ids, attention_mask, labels = batch
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            labels = labels.to(device)

            # 4.2 前向传播：模型预测
            outputs = model(input_ids, attention_mask=attention_mask)

            # 4.3 获取预测结果（最大 logits分数 对应的类别）
            y_pred_list = torch.argmax(outputs, dim=1)

            # 4.4 存储预测和真实标签
            all_preds.extend(y_pred_list.cpu().tolist())
            all_labels.extend(labels.cpu().tolist())

    # 5. 计算分类报告、F1 分数、准确率，精确率，召回率
    report = classification_report(all_labels, all_preds)
    f1score = f1_score(all_labels, all_preds, average='macro')
    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, average='macro')
    recall = recall_score(all_labels, all_preds, average='macro')

    # 6. 返回评估结果
    return report, f1score, accuracy, precision, recall
