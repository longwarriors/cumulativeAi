import threading
from app_flask import flask_app
from app_fastapi import fastapi_app
import uvicorn  # FastAPI的ASGI服务器


def run_flask():
    flask_app.run(host='0.0.0.0', port=5004)


def run_fastapi():
    uvicorn.run(fastapi_app, host='0.0.0.0', port=5005)


if __name__ == '__main__':
    t1 = threading.Thread(target=run_flask)  # 将Flask放在后台线程运行
    t1.start()
    run_fastapi()  # 在主线程运行FastAPI
