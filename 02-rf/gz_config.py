

# todo 1. 定义配置类
class Config():
    # 1. 初始化函数.
    def __init__(self):
        # 原始文件的路径.
        # 1.1 训练数据路径
        self.train_datapath = "../01-data/data/train.txt"
        # 1.2 测试数据路径
        self.test_datapath = "../01-data/data/test.txt"
        # 1.3 验证数据路径
        self.dev_datapath = "../01-data/data/dev.txt"
        # 1.4 类别数据路径
        self.class_datapath = "../01-data/data/class.txt"
        # 1.5 停用词数据路径
        self.stop_words_path = "../01-data/data/stopwords.txt"

        # 2. 保存处理后的文件信息.
        self.process_train_path = "./data/process_train_data.csv"
        self.process_test_path = "./data/process_test_data.csv"
        self.process_dev_path = "./data/process_dev_data.csv"

        # 3. 模型保存的路径.
        self.rf_model_save_path = "./save_model/rf_model.pkl"
        self.tfidf_model_save_path = "./save_model/tfidf_model.pkl"

        # 保存模型预测的结果.
        self.model_predict_result = './result/predict_result.txt'



# todo 2.测试
if __name__ == '__main__':
    # import pandas as pd
    # df = pd.read_csv('../01-data/data/train.txt', sep='\t', names=['text', 'label'])
    # print(df.head())

    # 1. 测试配置类的功能, 创建配置类的对象.
    conf = Config()
    # 2. 打印配置路径进行验证.
    print(f'训练集路径: {conf.train_datapath}')
    print(f'测试集路径: {conf.test_datapath}')

















