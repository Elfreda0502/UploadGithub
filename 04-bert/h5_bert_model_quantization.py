# 该.py文件用于: 演示BERT模型的量化, 使用的是: 动态量化(DQ, Dynamic Quantization)
# 细节: 切换到 CPU环境 运行.

# 禁用oneDNN优化, 或者使用 warning 压制警告.
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'


# 导包
from config import Config                   # 导入配置类
import torch                                # 深度学习框架
from h2_bert_classifier_model import BertClassifier # 导入自定义的BERT分类器模型
from h1_dataloader_utils import build_dataloader    # 导入数据集加载器
from h3_train_predict import model2dev              # 导入自定义的验证函数

# todo 0.(可选) 查看并配置量化引擎.
# 打印当前环境支持的 量化计算 后端引擎, 了解可用的量化加入方式.
# 常见的引擎包括: ['none', 'onednn', 'x86', 'fbgemm'] -> ['无加速', '英特尔深度学习加速', 'x86架构优化', 'FaceBook的量化计算库']
print(torch.backends.quantized.supported_engines)

# 显示设置引擎为: fbgemm(适用于X86架构的CPU, 可以优化量化计算速度)
torch.backends.quantized.engine = 'fbgemm'      # 非必须操作(可选), PyTorch会根据硬件自动选择最合适的引擎.

# todo 1.加载配置参数.
conf = Config()

# todo 2. 构建数据集加载器.
train_dataloader, dev_dataloader, test_dataloader = build_dataloader()

# todo 3. 初始化模型, 并加载参数.
model = BertClassifier()
# model.load_state_dict(torch.load(conf.model_save_path, map_location=conf.device))       # GPU环境
model.load_state_dict(torch.load(conf.model_save_path, map_location='cpu'))         # CPU环境
# 将模型设置为: 评估模式(会关闭训练时的dropout等随机层, 确保推理结果稳定)
model.eval()
# print(f'量化前的模型: {model}')

# todo 4. 量化前模型性能验证.
report, f1score, accuracy, precision, recall = model2dev(model, dev_dataloader, conf.device)
print(f'量化前 f1-socre: {f1score}')
print('-' * 40)

# todo 5. 执行模型动态量化.
# 参1 model: 待量化的模型.
# 参2 {torch.nn.Linear}: 待量化的层, 这里是仅对 Linear层进行量化.
# 参3: dtype: 量化(后)的数据类型.
quantized_model = torch.quantization.quantize_dynamic(model, {torch.nn.Linear}, dtype=torch.qint8)
# print(f'量化后的模型: {quantized_model}')

# todo 6. 量化后的模型性能验证.
report2, f1score2, accuracy2, precision2, recall2 = model2dev(quantized_model, dev_dataloader, conf.device)
print(f'量化后的 f1-socre: {f1score2}')

# todo 7. 保存量化后的模型.
torch.save(quantized_model.state_dict(), conf.bert_model_quantization_model_path)
print(f'量化后的模型保存成功, 路径为: {conf.bert_model_quantization_model_path}')