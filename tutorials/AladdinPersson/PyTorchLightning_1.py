# PyTorch Lightning #1 - Why Lightning?
# https://www.youtube.com/watch?v=XbIN9LaQycQ&list=PLhhyoLH6IjfyL740PTuXef4TstxAK6nGP
import torch
from torch import nn, optim
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
from tqdm import tqdm


class Net(nn.Module):
    def __init__(self, input_size, num_classes):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(input_size, 50)
        self.fc2 = nn.Linear(50, num_classes)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x


DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Hyperparameters
INPUT_SIZE = 28 * 28
NUM_CLASSES = 10
BATCH_SIZE = 64
NUM_EPOCHS = 5
TRAIN_RATIO = 0.8
learning_rate = 0.001

# Load MNIST dataset
entire_dataset = datasets.MNIST(root='../../data', train=True, download=False, transform=transforms.ToTensor())
test_dataset = datasets.MNIST(root='../../data', train=False, download=False, transform=transforms.ToTensor())
train_size = int(TRAIN_RATIO * len(entire_dataset))
val_size = len(entire_dataset) - train_size
train_dataset, val_dataset = random_split(entire_dataset, [train_size, val_size])

# DataLoader用来分批次、打乱、多线程等操作
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

# Initialize model, loss function, and optimizer
model = Net(INPUT_SIZE, NUM_CLASSES).to(DEVICE)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Training loop
for epoch in range(NUM_EPOCHS):
    model.train()
    running_loss = 0.0
    for batch_idx, (data, targets) in enumerate(tqdm(train_loader)):
        data, targets = data.view(data.size(0), -1).to(DEVICE), targets.to(DEVICE)
        optimizer.zero_grad()  # 清空梯度
        outputs = model(data)  # 前向传播
        loss = criterion(outputs, targets)  # 计算损失
        loss.backward()  # 反向传播
        optimizer.step()  # 更新参数

        running_loss += loss.item()

    print(f"Epoch [{epoch + 1}/{NUM_EPOCHS}], Loss: {running_loss / len(train_loader):.4f}")


# Check accuracy on validation set
def check_accuracy(loader, model):
    model.eval()
    num_correct = 0
    num_samples = 0
    with torch.no_grad():
        for data, targets in loader:
            data, targets = data.view(data.size(0), -1).to(DEVICE), targets.to(DEVICE)
            outputs = model(data)
            _, predicted = outputs.max(1)
            num_samples += targets.size(0)
            num_correct += (predicted == targets).sum().item()
    model.train()
    return num_correct / num_samples


# Validation loop
model.to(DEVICE)
print(f"Training accuracy: {check_accuracy(train_loader, model) * 100:.2f}%")
print(f"Validation accuracy: {check_accuracy(val_loader, model) * 100:.2f}%")
print(f"Test accuracy: {check_accuracy(test_loader, model) * 100:.2f}%")
