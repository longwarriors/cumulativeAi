# https://www.kaggle.com/code/nekokiku/simple-resnet-baseline
# https://zhuanlan.zhihu.com/p/1889819407540278719
# 微调resnet34模型
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import os
import matplotlib.pyplot as plt
import torchvision.models as models
from tqdm import tqdm
import seaborn as sns

# 1.查看label文件
BASE_DIR = r'../data/kaggle-classify-leaves'
train_file_path = os.path.join(BASE_DIR, 'train.csv')
test_file_path = os.path.join(BASE_DIR, 'test.csv')

test_df = pd.read_csv(test_file_path)
train_df = pd.read_csv(train_file_path)
print(train_df.head(10))
print('-' * 100)
print(train_df.describe())


def barw(ax):
    for p in ax.patches:
        val = p.get_width()  # height of the bar
        x = p.get_x() + p.get_width()  # x- position
        y = p.get_y() + p.get_height() / 2  # y-position
        ax.annotate(round(val, 2), (x, y))

plt.figure(figsize=(15, 30))
ax0 = sns.countplot(y=train_df['label'],
                    order=train_df['label'].value_counts().index)
barw(ax0)
plt.title('train label distribution')
plt.show()