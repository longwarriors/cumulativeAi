# https://huggingface.co/NewBreaker/classify-cat_vs_dog/blob/main/1.ResNet18(98.43%25).py
# https://zhuanlan.zhihu.com/p/629746685
# https://claude.ai/chat/a16f2a9e-d7f8-4db1-be87-cd36c54f24ca
import os, copy, gc, torch
import torch.nn as nn
import torch.optim as optim
from torchvision.models import resnet18, ResNet18_Weights
from tqdm import tqdm
from PIL import Image
from torchvision import transforms, models
from torch.utils.data import Dataset, DataLoader, random_split


class DogCatDataset(Dataset):
    def __init__(self, root_dir: str, transform=None, is_train: bool = True):
        self.root_dir = root_dir
        self.transform = transform
        self.img_paths = []
        self.labels = []

        if is_train:
            for filename in os.listdir(root_dir):
                if filename.endswith(('.jpg', '.jpeg', '.png')):
                    # 从文件名提取标签：cat 为 0，dog 为 1
                    label = 0 if filename.lower().startswith('cat') else 1
                    self.img_paths.append(os.path.join(self.root_dir, filename))
                    self.labels.append(label)
        else:
            for filename in os.listdir(root_dir):
                if filename.endswith(('.jpg', '.jpeg', '.png')):
                    # 测试集没有标签，所以使用 -1 作为占位符
                    self.img_paths.append(os.path.join(self.root_dir, filename))
                    self.labels.append(-1)

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
transform = {
    'train': transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ]),
    'inference': transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ]),
}

labelled_set = DogCatDataset(root_dir='../data/kaggle-dogs-vs-cats-redux-kernels-edition/train', transform=transform['train'], is_train=True)
inference_set = DogCatDataset(root_dir='../data/kaggle-dogs-vs-cats-redux-kernels-edition/test', transform=transform['inference'], is_train=False)
train_size = int(TRAIN_RATIO * len(labelled_set))
valid_size = len(labelled_set) - train_size
train_set, valid_set = random_split(labelled_set, [train_size, valid_size])
train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
valid_loader = DataLoader(valid_set, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
inference_loader = DataLoader(inference_set, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

# 清理缓存
gc.collect()
torch.cuda.empty_cache()

# 创建模型
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
PRETRAIN = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1).to(DEVICE)

# 检查点保存路径
CHECKPOINT_DIR = os.path.join('../checkpoints', 'resnet18_dogcat')
if not os.path.exists(CHECKPOINT_DIR):
    os.makedirs(CHECKPOINT_DIR)

# 修改最后一层适应二分类
num_features = PRETRAIN.fc.in_features  # 512
PRETRAIN.fc = nn.Linear(num_features, 2).to(DEVICE)

# 冻结预训练层
for param in PRETRAIN.parameters():
    param.requires_grad = False  # 冻结所有层

# 设置最后一层的参数为可训练
for param in PRETRAIN.fc.parameters():
    param.requires_grad = True

# 定义损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(filter(lambda p: p.requires_grad, PRETRAIN.parameters()), lr=0.001, momentum=0.9)


# 训练模型
def train_model(model, train_loader, valid_loader, criterion, optimizer, checkpoint_dir, num_epochs=10):
    best_acc = 0.0
    best_model_wts = copy.deepcopy(model.state_dict())
    epoch_bar = tqdm(range(num_epochs), desc='Epochs', unit='epoch')
    for epoch in epoch_bar:
        # 训练阶段
        model.train()
        running_loss = 0.0
        running_corrects = 0

        # 训练进度条
        train_bar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{num_epochs} [Train]", unit='batch')
        for images, labels in train_bar:
            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            _, preds = torch.max(outputs, 1)
            running_loss += loss.item() * images.size(0)
            running_corrects += torch.sum(preds == labels.data)

            # 训练进度条更新loss和accuracy
            batch_loss = loss.item()
            batch_acc = torch.sum(preds == labels.data).double() / images.size(0)
            train_bar.set_postfix(loss=batch_loss, acc=batch_acc.item())

        epoch_loss = running_loss / len(train_loader.dataset)
        epoch_acc = running_corrects.double() / len(train_loader.dataset)
        print(f'Epoch {epoch + 1}/{num_epochs} Train Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

        # 验证阶段
        model.eval()
        valid_loss = 0.0
        valid_corrects = 0

        # 验证进度条
        valid_bar = tqdm(valid_loader, desc=f"Epoch {epoch + 1}/{num_epochs} [Valid]", unit='batch')
        with torch.no_grad():
            for images, labels in valid_bar:
                images = images.to(DEVICE)
                labels = labels.to(DEVICE)

                outputs = model(images)
                loss = criterion(outputs, labels)

                _, preds = torch.max(outputs, 1)
                valid_loss += loss.item() * images.size(0)
                valid_corrects += torch.sum(preds == labels.data)

                # 验证进度条更新loss和accuracy
                batch_loss = loss.item()
                batch_acc = torch.sum(preds == labels.data).double() / images.size(0)
                valid_bar.set_postfix(loss=batch_loss, acc=batch_acc.item())

        epoch_valid_loss = valid_loss / len(valid_loader.dataset)
        epoch_valid_acc = valid_corrects.double() / len(valid_loader.dataset)
        print(f'Epoch {epoch + 1}/{num_epochs} Val Loss: {epoch_valid_loss:.4f} Acc: {epoch_valid_acc:.4f}')

        # 保存epoch_ckpt
        epoch_ckpt = {
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'valid_loss': epoch_valid_loss,
            'valid_accuracy': epoch_valid_acc,
        }
        torch.save(epoch_ckpt, os.path.join(checkpoint_dir, f'checkpoint_epoch_{epoch + 1}.pth'))

        # 保存最佳模型
        if epoch_valid_acc > best_acc:
            best_acc = epoch_valid_acc
            best_model_wts = copy.deepcopy(model.state_dict())
            torch.save(epoch_ckpt, os.path.join(checkpoint_dir, 'best_model.pth'))
            print('Best model updated!')

    # 加载最佳模型权重
    model.load_state_dict(best_model_wts)
    return model


fine_tune_model = train_model(PRETRAIN, train_loader, valid_loader, criterion, optimizer, CHECKPOINT_DIR, num_epochs=5)


# 加载检查点函数
def load_checkpoint(model, optimizer, checkpoint_path):
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        epoch = checkpoint['epoch']
        valid_loss = checkpoint['valid_loss']
        valid_accuracy = checkpoint['valid_accuracy']
        print(
            f'Loaded checkpoint from epoch {epoch} with valid loss {valid_loss:.4f} and accuracy {valid_accuracy:.4f}')
        return model, optimizer, epoch
    else:
        print(f'No checkpoint found at {checkpoint_path}, starting from scratch.')
        return model, optimizer, 0


# 推理函数
def inference(model, dataloader, device):
    model.eval()
    predictions = []
    with torch.no_grad():
        for images, _ in tqdm(dataloader, desc="Inference", unit="batch"):
            images = images.to(device)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            predictions.extend(preds.cpu().numpy())
    return predictions

# 加载最佳模型权重
best_model_path = os.path.join(CHECKPOINT_DIR, 'best_model.pth')
if os.path.exists(best_model_path):
    checkpoint = torch.load(best_model_path)
    PRETRAIN.load_state_dict(checkpoint['model_state_dict'])
    print("Best model loaded for inference.")
else:
    print("No best model found. Please train the model first.")

# 对测试集进行推理
test_predictions = inference(PRETRAIN, inference_loader, DEVICE)

# 保存推理结果
output_file = os.path.join('../results', 'test_predictions.txt')
if not os.path.exists('../results'):
    os.makedirs('../results')

with open(output_file, 'w') as f:
    for idx, pred in enumerate(test_predictions):
        f.write(f"Image {idx + 1}: {'Dog' if pred == 1 else 'Cat'}\n")

print(f"Inference completed. Results saved to {output_file}.")

