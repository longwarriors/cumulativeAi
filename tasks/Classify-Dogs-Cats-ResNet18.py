# https://huggingface.co/NewBreaker/classify-cat_vs_dog/blob/main/1.ResNet18(98.43%25).py
# https://zhuanlan.zhihu.com/p/629746685
import os
import torchvision
import torch
import torch.nn as nn
from torchvision.models import ResNet18_Weights
from tqdm import tqdm
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader



def make_dir(path):
    dir = os.path.exists(path)
    if not dir:
        os.makedirs(path)
        print(f"Directory {path} created")
    else:
        print(f"Directory {path} already exists")

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 8
PRETRAIN = models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1).to(DEVICE)

train_set = datasets.ImageFolder(
    root='../data/dogs-vs-cats/train',
)
train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
valid_set = datasets.ImageFolder(
    root=
)
valid_loader = DataLoader(valid_set, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)