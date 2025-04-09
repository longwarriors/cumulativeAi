# https://www.youtube.com/watch?v=RLqsxWaQdHE

import torch
import torchvision
from torch import nn, optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter
from torchvision import datasets, transforms


class SimpleCNN(nn.Module):
    def __init__(self, in_channels=3, num_classes=10):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, 8, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))
        self.conv2 = nn.Conv2d(8, 16, kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(16 * 7 * 7, num_classes)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = x.reshape(x.size(0), -1)
        x = self.fc1(x)
        return x


DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Hyperparameters
BATCH_SIZE_LIST = [2, 16, 64, 256]
LEARNING_RATE_LIST = [1e-1, 1e-2, 1e-3, 1e-4]
# LEARNING_RATE = 1e-3
IN_CHANNEL = 1
NUM_CLASSES = 10
# BATCH_SIZE = 64
NUM_EPOCHS = 2

# Load MNIST dataset
mnist_full = datasets.MNIST(root=r'../../data', train=True, download=False, transform=transforms.ToTensor())
train_ratio = 0.8
train_size = int(train_ratio * len(mnist_full))
val_size = len(mnist_full) - train_size
mnist_train, mnist_val = random_split(mnist_full, [train_size, val_size])
mnist_test = datasets.MNIST(root=r'../../data', train=False, download=False, transform=transforms.ToTensor())

# test_loader = DataLoader(dataset=mnist_test, batch_size=BATCH_SIZE, shuffle=False)


for batch_size in BATCH_SIZE_LIST:
    for learning_rate in LEARNING_RATE_LIST:
        step = 0
        # Initialize model, loss function, optimizer
        model = SimpleCNN(in_channels=IN_CHANNEL, num_classes=NUM_CLASSES)
        model.to(DEVICE)
        model.train()
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        train_loader = DataLoader(mnist_train, batch_size, shuffle=True)
        # Initialize TensorBoard
        writer = SummaryWriter(
            log_dir=r"..\..\logs\tb_logs\hyperparams\MinBatchSize_" + f"{batch_size}_LR_{learning_rate}"
        )

        # Training loop
        for epoch in range(NUM_EPOCHS):
            losses = []
            accuracies = []
            for batch_idx, (images, labels) in enumerate(train_loader):
                batch_size = labels.size(0)
                images, labels = images.to(DEVICE), labels.to(DEVICE)
                optimizer.zero_grad()
                outputs = model(images)
                loss = criterion(outputs, labels)
                losses.append(loss.item())
                loss.backward()
                optimizer.step()

                # Calculate training accuracy
                predicted = torch.argmax(outputs.data, dim=1)
                accuracy = (predicted == labels).sum().item() / batch_size
                accuracies.append(accuracy)

                # Image grid
                img_grid = torchvision.utils.make_grid(images)
                writer.add_image('MNIST_images', img_grid)
                writer.add_histogram('fc1_weights', model.fc1.weight)
                # Log to TensorBoard
                writer.add_scalar('Training Loss', loss.item(), global_step=step)
                writer.add_scalar('Training Accuracy', accuracy, global_step=step)
                step += 1
            writer.add_hparams(hparam_dict={'Bsize': batch_size, 'LR': learning_rate},
                               metric_dict={'loss': sum(losses) / len(losses),
                                            'accuracy': sum(accuracies) / len(accuracies)})
