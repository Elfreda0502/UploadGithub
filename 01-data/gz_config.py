
# todo 1.定义配置类, 用来集中管理项目中所有数据文件的路径
# 采用类的形式类管理, 可以提高代码的可维护性, 以便统一的修改和调用.
class Config():
    # 1. 初始化函数.
    def __init__(self):
        # 1.1 配置训练集的路径.
        self.train_datapath = "./data/train.txt"
        # 1.2 配置测试集的路径.
        self.test_datapath = "./data/test.txt"
        # 1.3 配置验证集的路径.
        self.dev_datapath = "./data/dev.txt"
        # 1.4 配置 类别定义文件 路径
        self.class_datapath = "./data/class.txt"


# todo 2. 测试配置类功能.
if __name__ == '__main__':
    # 1. 创建对象.
    config = Config()

    # 2. 打印路径以便测试.
    print(f'训练集路径: {config.train_datapath}')
    print(f'测试集路径: {config.test_datapath}')
    print(f'验证集路径: {config.dev_datapath}')