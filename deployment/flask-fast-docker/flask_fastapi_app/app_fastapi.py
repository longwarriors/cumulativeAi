from fastapi import FastAPI
fastapi_app = FastAPI()

# 定义FastAPI应用的路由
@fastapi_app.get("/fastapi")
def hello_fastapi():
    return {"message": "Hello from FastAPI!"}