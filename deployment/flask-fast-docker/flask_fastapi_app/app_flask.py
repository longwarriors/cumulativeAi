from flask import Flask
flask_app = Flask(__name__)

# 定义Flask应用的路由
@flask_app.route('/flask')
def hello_flask():
    return "Hello from Flask!"