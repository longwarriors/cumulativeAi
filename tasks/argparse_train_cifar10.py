# https://zhuanlan.zhihu.com/p/2593970497
# 使用argparse解析命令行参数

import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, random_split
from tqdm.auto import tqdm
from torchinfo import summary  # from torchsummary import summary


class SimpleCNN(nn.Module):
    def __init__(self, num_classes=10):
        super(SimpleCNN, self).__init__()
        # 输入通道为3（RGB图像），输出通道为32，卷积核大小为3x3，padding为1保证尺寸不变
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(64 * 8 * 8, 512)
        self.fc2 = nn.Linear(512, num_classes)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.5)

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = x.view(-1, 64 * 8 * 8)  # 展平
        x = self.dropout(self.relu(self.fc1(x)))
        x = self.fc2(x)
        return x


def train_model(net,
                train_loader,
                valid_loader,
                criterion,
                optimizer,
                device,
                num_epochs=10):
    train_losses, train_acc = [], []
    valid_losses, valid_acc = [], []
    num_train_batches = len(train_loader)
    num_valid_batches = len(valid_loader)

    for epoch in range(num_epochs):
        # 1.训练阶段
        net.train()
        running_loss = 0.0
        num_correct = 0
        num_samples = 0
        with tqdm(train_loader, desc=f"Epoch {epoch + 1}/{num_epochs} [Train]") as pbar_train:
            for images, labels in pbar_train:
                batch_size = images.size(0)
                images, labels = images.to(device), labels.to(device)

                # 前向传播
                optimizer.zero_grad()
                outputs = net(images)
                loss = criterion(outputs, labels)

                # 反向传播和优化
                loss.backward()
                optimizer.step()

                # 计算损失和准确率
                running_loss += loss.item()
                predicted = torch.argmax(outputs, dim=-1)  # _, predicted = torch.max(outputs.data, 1)
                num_samples += batch_size
                num_correct += (predicted == labels).sum().item()

                # 更新进度条信息
                pbar_train.set_postfix({
                    'loss': f'{running_loss / num_samples:.4f}',
                    'accuracy': f'{num_correct / num_samples:.4f}'
                })

        # 记录训练集的批次平均损失和准确率
        train_losses.append(running_loss / num_train_batches)
        train_acc.append(100 * num_correct / num_samples)

        # 2.验证阶段
        net.eval()
        val_loss = 0.0
        val_correct = 0
        val_samples = 0
        with torch.no_grad():
            with tqdm(valid_loader, desc=f"Epoch {epoch + 1}/{num_epochs} [Valid]") as pbar_valid:
                for images, labels in pbar_valid:
                    batch_size = images.size(0)
                    images, labels = images.to(device), labels.to(device)

                    # 前向传播
                    outputs = net(images)
                    loss = criterion(outputs, labels)

                    # 计算损失和准确率
                    val_loss += loss.item()
                    predicted = torch.argmax(outputs, dim=-1)
                    val_samples += batch_size
                    val_correct += (predicted == labels).sum().item()

                    # 更新进度条信息
                    pbar_valid.set_postfix({
                        'loss': f'{val_loss / val_samples:.4f}',
                        'accuracy': f'{val_correct / val_samples:.4f}'
                    })

        # 记录验证集的批次平均损失和准确率
        valid_losses.append(val_loss / num_valid_batches)
        valid_acc.append(val_correct / val_samples * 100)

        # 打印当前 epoch 的训练和验证结果
        tqdm.write(f"Epoch {epoch + 1}/{num_epochs} - "
                   f"Train Loss: {train_losses[-1]:.4f}, "
                   f"Train Accuracy: {train_acc[-1]:.2f}%, "
                   f"Valid Loss: {valid_losses[-1]:.4f}, "
                   f"Valid Accuracy: {valid_acc[-1]:.2f}%")

    return train_losses, valid_losses, train_acc, valid_acc


def inference_model(net, dataloader, device):
    net.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad(), tqdm(dataloader, desc="Inference") as pbar:
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)

            # 前向传播
            outputs = net(images)
            predicted = torch.argmax(outputs, dim=-1)

            # 收集结果
            all_preds.append(predicted.cpu())
            all_labels.append(labels.cpu())

            # 更新进度条信息，显示当前已处理的样本数
            pbar.set_postfix({
                'processed samples': f'{len(all_preds) * dataloader.batch_size}/{len(dataloader.dataset)}'
            })

    all_preds = torch.cat(all_preds)
    all_labels = torch.cat(all_labels)
    return all_preds, all_labels


def plot_metrics(train_losses, valid_losses, train_acc, valid_acc):
    epochs = range(1, len(train_losses) + 1)

    # 绘制损失曲线
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_losses, label='Train Loss')
    plt.plot(epochs, valid_losses, label='Valid Loss')
    plt.title('Loss Curve')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    # 绘制准确率曲线
    plt.subplot(1, 2, 2)
    plt.plot(epochs, train_acc, label='Train Accuracy')
    plt.plot(epochs, valid_acc, label='Valid Accuracy')
    plt.title('Accuracy Curve')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy (%)')
    plt.legend()

    plt.tight_layout()
    plt.show()


def main():
    # 1.解析命令行参数
    parser = argparse.ArgumentParser(description='Train a CNN on CIFAR-10')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size for training and validation')
    parser.add_argument('--num_epochs', type=int, default=10, help='Number of epochs to train')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate for the optimizer')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of workers for DataLoader')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                        help='Device to use for training (cuda or cpu)')
    args = parser.parse_args()

    # 2.数据预处理和加载
    transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32, padding=4),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    cifar10_train = torchvision.datasets.CIFAR10(root='../data', train=True,
                                                 download=True, transform=transform)
    cifar10_test = torchvision.datasets.CIFAR10(root='../data', train=False,
                                                download=True, transform=transform)

    # 划分训练集和验证集
    train_size = int(0.8 * len(cifar10_train))
    valid_size = len(cifar10_train) - train_size
    cifar10_train, cifar10_valid = random_split(cifar10_train, [train_size, valid_size])

    train_loader = DataLoader(cifar10_train, batch_size=args.batch_size,
                              shuffle=True, num_workers=args.num_workers)
    valid_loader = DataLoader(cifar10_valid, batch_size=args.batch_size,
                              shuffle=False, num_workers=args.num_workers)
    test_loader = DataLoader(cifar10_test, batch_size=args.batch_size,
                             shuffle=False, num_workers=args.num_workers)

    # 3.模型、损失函数和优化器
    device = torch.device(args.device)
    net = SimpleCNN(num_classes=10).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=args.learning_rate)
