from flask import Flask
app = Flask(__name__)

# 定义跟路由处理函数
@app.route('/')
def index():
    return "Hello, World! This is a Flask app running in a Docker container."

if __name__ == '__main__':
    # 运行Flask应用
    app.run(host='0.0.0.0', port=5002)