# https://huggingface.co/NewBreaker/classify-cat_vs_dog/blob/main/1.ResNet18(98.43%25).py
# https://zhuanlan.zhihu.com/p/629746685
import os
import copy
import torchvision
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.models import ResNet18_Weights
from tqdm import tqdm
from PIL import Image
from torchvision import datasets, transforms, models
from torch.utils.data import Dataset, DataLoader, random_split


class DogCatDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.img_paths = []
        self.labels = []
        for filename in os.listdir(root_dir):
            if filename.endswith(".jpg"):
                # 从文件名提取标签：cat 为 0，dog 为 1
                label = 0 if filename.startswith('cat') else 1
                self.img_paths.append(os.path.join(self.root_dir, filename))
                self.labels.append(label)

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        """根据索引获取图像和标签"""
        img_path = self.img_paths[idx]
        label = self.labels[idx]
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, label


# hyperparameters
TRAIN_RATIO = 0.8
BATCH_SIZE = 32

# 创建数据集和加载器
transform = transforms.Compose([
    transforms.Resize((512, 512)),
    transforms.ToTensor(),
])
# dataset = DogCatDataset(root_dir='../data/kaggle-dogs-vs-cats-redux-kernels-edition/train', transform=transform)
# train_size = int(TRAIN_RATIO * len(dataset))
# valid_size = len(dataset) - train_size
# train_set, valid_set = random_split(dataset, [train_size, valid_size])
# train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
# valid_loader = DataLoader(valid_set, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

# 创建模型
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
PRETRAIN = models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1).to(DEVICE)

# 修改最后一层适应二分类
num_features = PRETRAIN.fc.in_features # 512
PRETRAIN.fc = nn.Linear(num_features, 2).to(DEVICE)

# 冻结预训练层
for param in PRETRAIN.parameters():
    param.requires_grad = False # 冻结所有层
PRETRAIN.fc.requires_grad = True # 只训练最后一层

# 定义损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(PRETRAIN.fc.parameters(), lr=0.001, momentum=0.9)

# 训练模型
# https://grok.com/chat/7bcee617-38d4-4925-b851-3b35fe8abc90
def train_model(model, train_loader, valid_loader, criterion, optimizer, num_epochs=10):
    best_acc = 0.0
    best_model_wts = copy.deepcopy(PRETRAIN.state_dict())