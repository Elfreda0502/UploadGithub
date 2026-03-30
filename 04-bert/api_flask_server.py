# 该py文件的主要任务: 通过Flask组件, 构建 路由 + 预测函数的 应用.


# 导包
from flask import Flask, request, jsonify
from h4_bert_predict_fun import predict_fun

# todo 1. 创建App应用(对象)
app = Flask(__name__)

# todo 2. 创建预测接口(路由 + 预测函数)
@app.route('/predict', methods=['POST'])
def predict():
    # 获取用户请求中的数据
    data = request.get_json()
    print(f'data: {data}, {type(data)}')        # 字典
    # 调用预测函数
    result = predict_fun(data)
    # 返回结果
    return jsonify(result)


# todo 3. 启动App应用
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=10086, debug=True)
    # app.run(host='192.168.109.56', port=10086, debug=True)