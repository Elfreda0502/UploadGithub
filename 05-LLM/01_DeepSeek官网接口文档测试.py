
# 导包
from openai import OpenAI       # 需要你安装下 openai这个包, 即: pip3 install openai

# todo 1. 初始化API的客户端, 后续发送API请求.
# 参1: api_key = 用于做身份验证的密钥, 需要去 deepseek注册, 然后获取真实的key
# 参2: base_url, 指定API请求的基础地址.
client = OpenAI(api_key="sk-822c08e0a0b2406d919d6d049495ba03", base_url="https://api.deepseek.com")

# todo 2.构建 并发送聊天请求.
# model: 指定要使用的模型名称
# messages: 对话消息列表, role:角色, content: 内容, stream: 是否流式返回结果
response = client.chat.completions.create(
    model="deepseek-chat",
    messages=[
        {"role": "system", "content": "You are a helpful assistant"},
        {"role": "user", "content": "给我讲一个关于笑话的笑话"},
    ],
    stream=False
)

# todo 3.处理并输出响应结果.
print(response.choices[0].message.content)