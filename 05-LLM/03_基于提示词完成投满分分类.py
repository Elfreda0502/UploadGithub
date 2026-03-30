# 该.py文件的作用 -> 结合提示词 和 标题问题, 对 文本做分类.
# 扩展: 你可以去注册下 阿里百炼 或者 智普AI等大模型的 Api Key, 然后用这些大模型再把程序跑一遍看结果.

# 导包
from dotenv import load_dotenv
load_dotenv()       # 加载环境配置, 执行后, 会默认读取当前目录下的 .env文件

import os
from openai import OpenAI

# todo 1.定义系统提示常量(提示词), 告知模型任务要求, 分类规则和示例, 指导模型生成符合预期的输出.
SYSTEM_PROMPT = """
你是一个中文新闻文本分类系统。你的任务是根据用户输入的中文新闻标题，将其准确分类到以下10个类别中的一种：finance（财经）、realty（房产）、stocks（股票）、education（教育）、science（科技）、society（社会）、politics（时政）、sports（体育）、game（游戏）、entertainment（娱乐）。

请严格遵循以下规则：
1.只输出最终的英文类别名称，不要输出任何其他内容、解释或中文。
2.你的判断必须基于标题的内容和关键词。
3.你的输出必须是且只能是上述10个类别之一。
现在，请对用户输入的新闻标题进行分类。
"""


# todo 2.定义获取DeepSeek模型响应的函数.
def get_deepseek_res(prompt):
    """
    接收用户录入的提示词, 调用DeepSeek模型获取结果, 并返回.
    :param prompt: 提示词
    :return: 模型响应结果
    """

    # 1. 创建API客户端实例.
    # 参2: base_url, 指定API请求的基础地址.
    client = OpenAI(
        api_key=os.getenv('API_KEY'),
        base_url=os.getenv('BASE_URL')
    )

    # todo 2.构建 并发送聊天请求.
    # model: 指定要使用的模型名称
    # messages: 对话消息列表, role:角色, content: 内容, stream: 是否流式返回结果
    response = client.chat.completions.create(
        model="deepseek-chat",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        stream=False
    )

    # todo 3.处理并输出响应结果.
    # print(response.choices[0].message.content)
    return response.choices[0].message.content


# todo 2. 测试函数.
if __name__ == '__main__':
    # 1. 测试1: 获取分类
    res1 = get_deepseek_res('考研英语语法学习两大主体原则')
    print(res1)
