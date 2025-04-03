import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import transforms
import os
import pandas as pd


# 模型评估函数
def evaluate(model,
             dataloader: DataLoader,
             criterion,
             device='cpu'):
    model.eval()  # 关闭 dropout 和 batch norm 的训练行为
    sum_loss: float = 0.0
    sum_correct: float = 0.0
    num_samples: int = 0

    with torch.no_grad():  # 不计算梯度
        for X, y in dataloader:
            batch_size = X.size(0)  # 或 y.size(0)
            X, y = X.to(device), y.to(device)
            logits = model(X)  # shape:  (batch_size, num_classes)
            # probabilities = torch.softmax(logits, dim=1)
            # y_hat = probabilities.argmax(dim=1)  # 结果与直接对logits使用argmax相同
            y_hat = logits.argmax(dim=1)
            batch_loss = criterion(logits, y)
            sum_loss += batch_loss.item() * batch_size  # 按样本数加权
            sum_correct += (y_hat == y).sum().item()
            num_samples += batch_size

    epoch_loss = sum_loss / num_samples
    epoch_accuracy = sum_correct / num_samples * 100  # 百分比形式
    return epoch_loss, epoch_accuracy


# 残差网络
class ResNet(nn.Module):
    def __init__(self, input_channels, num_channels, use_1x1conv=False, strides=1):
        super().__init__()
        self.conv1 = nn.Conv2d(input_channels, num_channels, kernel_size=3, padding=1, stride=strides)
        self.conv2 = nn.Conv2d(num_channels, num_channels, kernel_size=3, padding=1)
        if use_1x1conv:
            self.conv3 = nn.Conv2d(input_channels, num_channels, kernel_size=1, stride=strides)
        else:
            self.conv3 = None
        self.bn1 = nn.BatchNorm2d(num_channels)
        self.bn2 = nn.BatchNorm2d(num_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        y = self.relu(self.bn1(self.conv1(x)))
        y = self.bn2(self.conv2(y))
        if self.conv3:
            x = self.conv3(x)
        y += x  # 残差
        return self.relu(y)


# 残差块
def resnet_block(input_channels, num_channels, num_residuals, first_block=False):
    blk = []
    for i in range(num_residuals):
        if i == 0 and not first_block:
            blk.append(ResNet(input_channels, num_channels, use_1x1conv=True, strides=2))
        else:
            blk.append(ResNet(num_channels, num_channels))
    return blk


# 主网络
class ClassifyLeaves(nn.Module):
    def __init__(self, num_classes, set_device='cpu'):
        super().__init__()
        b1 = nn.Sequential(nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3),
                           nn.BatchNorm2d(64),
                           nn.ReLU(),
                           nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
        b2 = nn.Sequential(*resnet_block(64, 64, 2, first_block=True))
        b3 = nn.Sequential(*resnet_block(64, 128, 2))
        b4 = nn.Sequential(*resnet_block(128, 256, 2))
        b5 = nn.Sequential(*resnet_block(256, 512, 2))
        self.net = nn.Sequential(b1, b2, b3, b4, b5,
                                 nn.AdaptiveAvgPool2d((1, 1)),
                                 nn.Flatten(),
                                 nn.Linear(512, num_classes))
        self.unique_labels = None
        self.device = torch.device(set_device if torch.cuda.is_available() else 'cpu')
        self.to(self.device)  # 把神经网络实例移到设备上
        print(f"Model initialized on {self.device}")

    def forward(self, x):
        return self.net(x)

    def train_model(self,
                    train_dl: DataLoader,
                    val_dl: DataLoader,
                    max_epoch: int = 10,
                    learning_rate: float = 0.01,
                    weight_decay: float = 1e-4,
                    checkpoint_file_path: str = r'../checkpoints/kaggle_leaves.pth.tar',
                    resume: bool = False,  # 是否从断点恢复训练
                    ):
        self.train()  # 设置模型为训练模式
        self.unique_labels = train_dl.dataset.dataset.unique_labels  # 获取训练集的唯一标签
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.AdamW(self.parameters(), lr=learning_rate, weight_decay=weight_decay)
        best_val_accuracy: float = 0.0
        start_epoch: int = 0
        if resume and os.path.exists(checkpoint_file_path):
            checkpoint = torch.load(checkpoint_file_path,
                                    map_location=self.device,
                                    weights_only=True)
            self.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            start_epoch = checkpoint['epoch'] + 1
            best_val_accuracy = checkpoint['best_val_accuracy']
            print(f"Model loaded from epoch {start_epoch - 1} with val_accuracy: {best_val_accuracy:.2f}%")

        for epoch in range(start_epoch, max_epoch):
            self.train()  # 验证集评估evaluate步骤会将模型设置为eval模式
            sum_loss: float = 0.0
            sum_correct: float = 0.0
            num_samples: int = 0

            for X, y in train_dl:
                batch_size = X.size(0)  # 或 y.size(0)
                X, y = X.to(self.device), y.to(self.device)
                logits = self(X)
                probabilities = torch.softmax(logits, dim=1)
                y_hat = probabilities.argmax(dim=1)  # 获取预测类别索引
                batch_loss = criterion(logits, y)
                optimizer.zero_grad()
                batch_loss.backward()
                optimizer.step()  # 更新参数

                sum_loss += batch_loss.item() * batch_size  # 按样本数加权
                sum_correct += (y_hat == y).sum().item()
                num_samples += batch_size

            # 计算当前 epoch 训练集的平均损失和准确率
            train_epoch_loss = sum_loss / num_samples
            train_epoch_accuracy = sum_correct / num_samples * 100  # 百分比形式

            # 评估训练过的模型在验证集的表现
            val_epoch_loss, val_epoch_accuracy = evaluate(self, val_dl, criterion, self.device)

            print('-' * 100)
            print(f"Epoch {epoch + 1}/{max_epoch}: "
                  f"Train Loss: {train_epoch_loss:.4f}, Train Accuracy: {train_epoch_accuracy:.2f}%, "
                  f"Val Loss: {val_epoch_loss:.4f}, Val Accuracy: {val_epoch_accuracy:.2f}%")

            # 保存检查点
            if val_epoch_accuracy > best_val_accuracy:
                best_val_accuracy = val_epoch_accuracy
                checkpoint = {
                    'epoch': epoch,
                    'model_state_dict': self.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'best_val_accuracy': best_val_accuracy
                }
                torch.save(checkpoint, checkpoint_file_path, _use_new_zipfile_serialization=True)
                print(f"Checkpoint saved at epoch {epoch + 1} with Val Accuracy: {best_val_accuracy:.2f}%")

    def create_submission(self,
                          test_dl: DataLoader,
                          checkpoint_file_path: str = r'../checkpoints/kaggle_leaves.pth.tar',
                          submission_file_path: str = r'../output/submission_kaggle_leaves.csv'):
        """生成Kaggle提交文件"""
        if os.path.exists(checkpoint_file_path):
            checkpoint = torch.load(checkpoint_file_path,
                                    map_location=self.device,
                                    weights_only=True)
            self.load_state_dict(checkpoint['model_state_dict'])
            print(f"Model loaded from {checkpoint_file_path} with val_accuracy: {checkpoint['best_val_accuracy']:.2f}%")
        else:
            raise FileNotFoundError(f"Checkpoint not found at {checkpoint_file_path}")
        self.eval()
        preds = []
        with torch.no_grad():
            for X in test_dl:
                X = X.to(self.device)
                logits = model(X)
                probabilities = torch.softmax(logits, dim=1)
                y_hat = probabilities.argmax(dim=1)  # 获取预测类别索引
                preds.extend(y_hat.cpu().numpy())

        # 索引转换为类别名
        predicted_labels = [self.unique_labels[idx] for idx in preds]

        # 获取图片路径
        relative_paths = test_dl.dataset.relative_image_paths  # images/18353.jpg

        # 创建提交 DataFrame
        submission_df = pd.DataFrame({'image': relative_paths, 'label': predicted_labels})
        submission_df.to_csv(submission_file_path, index=False)  # 保存提交文件
        print(f"Submission file saved to {submission_file_path}")


if __name__ == '__main__':
    # 1. 创建数据集和数据加载器
    from datahubs.dl_leaves import make_leaves_dl

    DATA_DIR = r'../data/kaggle-classify-leaves'

    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),  # 随机水平翻转
        transforms.RandomRotation(degrees=15),  # 随机旋转
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    test_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    train_dl, val_dl, test_dl = make_leaves_dl(data_dir=DATA_DIR,
                                               batch_size=64,
                                               train_ratio=0.85,
                                               train_transform=train_transform,
                                               test_transform=test_transform)

    # 2. 创建模型实例
    NUM_CLASSES = train_dl.dataset.dataset.num_classes
    print(f"Creating model with {NUM_CLASSES} classes")
    model = ClassifyLeaves(num_classes=NUM_CLASSES, set_device='cuda')
    print("Model structure:")  # 打印模型结构
    print(model)

    # 3. 训练模型
    model.train_model(train_dl,
                      val_dl,
                      max_epoch=300,
                      learning_rate=0.0055,
                      weight_decay=0.025,
                      checkpoint_file_path=r'../checkpoints/kaggle_leaves-2.pth.tar',
                      resume=True)

    # 4. 创建提交文件
    model.create_submission(test_dl,
                            checkpoint_file_path=r'../checkpoints/kaggle_leaves-2.pth.tar',
                            submission_file_path=r'../output/submission_kaggle_leaves-2.csv')
