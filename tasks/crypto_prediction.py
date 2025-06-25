# https://grok.com/chat/a43defd2-bfcb-4169-91fa-28bcbebb1b24
# https://github.com/copilot/c/dddb5e24-7c5b-43ca-92de-789b7b6b7e7f

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import xgboost as xgb
import lightgbm as lgb
from torch import nn, optim
from torch.utils.data import DataLoader, Dataset
from scipy.stats import pearsonr
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
import warnings, os
warnings.filterwarnings("ignore")

# 设置随机种子
def set_seed(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

set_seed(42)
DVEICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"使用设备: {DVEICE}")

# 数据加载分析
def load_data(data_dir):
    print("加载数据...")
    train_file = os.path.join(data_dir, "train.parquet")
    test_file = os.path.join(data_dir, "test.parquet")
    submission_file = os.path.join(data_dir, "sample_submission.csv")
    train_df = pd.read_parquet(train_file)
    test_df = pd.read_parquet(test_file)
    sample_submission_df = pd.read_csv(submission_file)
    print(f"训练数据形状: {train_df.shape}")
    print(f"测试数据形状: {test_df.shape}")
    print(f"样例提交文件形状: {sample_submission_df.shape}")

    # 检查缺失值
    print("\n训练数据缺失值:")
    print(train_df.isnull().sum().sum())

    # 查看数据基本信息
    print("\n训练数据基本信息:")
    print(train_df.info())

    # 数据统计摘要
    print("\n数据统计摘要:")
    print(train_df.describe())

    return train_df, test_df, sample_submission_df

DATA_DIR = r"../data/kaggle-drw-crypto-market-prediction"
train_df, test_df, sample_submission_df = load_data(DATA_DIR)