# 该.py文件的作用 -> 演示DeepSeek的api key的 封装后的函数版, 用于对接后续的Flask等...


# 导包
from openai import OpenAI


# todo 1.定义获取DeepSeek模型响应的函数.
def get_deepseek_res(prompt):
    """
    接收用户录入的提示词, 调用DeepSeek模型获取结果, 并返回.
    :param prompt: 提示词
    :return: 模型响应结果
    """

    # 1. 创建API客户端实例.
    # 参2: base_url, 指定API请求的基础地址.
    client = OpenAI(api_key="sk-822c08e0a0b2406d919d6d049495ba03",
                    base_url="https://api.deepseek.com")

    # todo 2.构建 并发送聊天请求.
    # model: 指定要使用的模型名称
    # messages: 对话消息列表, role:角色, content: 内容, stream: 是否流式返回结果
    response = client.chat.completions.create(
        model="deepseek-chat",
        messages=[
            {"role": "system", "content": "你是一个AI助手"},
            {"role": "user", "content": prompt},
        ],
        stream=False
    )

    # todo 3.处理并输出响应结果.
    # print(response.choices[0].message.content)
    return response.choices[0].message.content


# todo 2. 测试函数.
if __name__ == '__main__':
    # 1. 测试1: 获取诗词.
    res1 = get_deepseek_res('我叫展哥, 我是一名AI教师, 请帮我写一篇赞扬我的诗词')
    print(res1)
    print('-' * 40)

    # 2. 测试2: 获取睡前故事.
    res2 = get_deepseek_res('我叫展哥, 你刚才生成的诗词令我激动的睡不着觉, 请给我讲一个睡前故事')
    print(res2)