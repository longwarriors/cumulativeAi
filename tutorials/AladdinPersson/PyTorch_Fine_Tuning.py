# 迁移学习微调模型
# https://www.youtube.com/watch?v=qaDe0qQZ5AQ

import torch
from scipy.cluster.hierarchy import weighted
from torch import nn, optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torchvision import models
import sys
from tqdm import tqdm

# Set device
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Hyperparameters
IN_CHANNEL = 3
NUM_CLASSES = 10
LEARNING_RATE = 1e-4
BATCH_SIZE = 64
NUM_EPOCHS = 15

# Load pretrained model and modify it
model = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1)
for param in model.parameters():
    param.requires_grad = False  # freeze all layers

model.avgpool = nn.Identity()  # passes the input tensor through without any changes
model.classifier = nn.Sequential(nn.Linear(512, 100),
                                 nn.ReLU(),
                                 nn.Linear(100, NUM_CLASSES))
# print(model)
# sys.exit() # 调试代码立即终止
model.to(DEVICE)

# Load dataset
train_set = datasets.CIFAR10(root='../../data', train=True, download=False, transform=transforms.ToTensor())
test_set = datasets.CIFAR10(root='../../data', train=False, download=False, transform=transforms.ToTensor())
train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=True)
# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

# Train the model
for epoch in range(NUM_EPOCHS):
    losses = []
    for batch_idx, (data, target) in enumerate(tqdm(train_loader)):
        data, target = data.to(DEVICE), target.to(DEVICE)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        losses.append(loss.item())
        loss.backward()
        optimizer.step()
    print(f"Epoch {epoch + 1}/{NUM_EPOCHS}, Loss: {sum(losses) / len(losses):.5f}")

def check_accuracy(loader, model):
    if loader.dataset.train:
        print("Checking accuracy on training set")
    else:
        print("Checking accuracy on test set")

    num_correct = 0
    num_samples = 0
    model.eval()
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            scores = model(x)
            _, predictions = scores.max(1)
            num_correct += (predictions == y).sum()
            num_samples += predictions.size(0)
            print(f"Got {num_correct} / {num_samples} with accuracy {float(num_correct) / float(num_samples) * 100:.2f}")
    model.train()

check_accuracy(train_loader, model)