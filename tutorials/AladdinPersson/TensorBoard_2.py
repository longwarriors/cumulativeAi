# https://github.com/aladdinpersson/Machine-Learning-Collection/blob/master/ML/Pytorch/Basics/pytorch_tensorboard_.py

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


# Set device
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Hyperparameters
IN_CHANNEL = 1
NUM_CLASSES = 10
NUM_EPOCHS = 2
BATCH_SIZES = [16, 64, 256]
LEARNING_RATES = [1e-1, 1e-2, 1e-3, 1e-4]
CLASSES = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']

# Load MNIST dataset
mnist_full = datasets.MNIST(root=r'../../data', train=True, download=False, transform=transforms.ToTensor())
mnist_test = datasets.MNIST(root=r'../../data', train=False, download=False, transform=transforms.ToTensor())
train_ratio = 0.8
train_size = int(train_ratio * len(mnist_full))
val_size = len(mnist_full) - train_size
mnist_train, mnist_val = random_split(mnist_full, [train_size, val_size])

# Train loop
for bsize in BATCH_SIZES:
    for lr in LEARNING_RATES:
        step = 0
        # Initialize model, loss function, optimizer
        model = SimpleCNN(in_channels=IN_CHANNEL, num_classes=NUM_CLASSES)
        model.to(DEVICE)
        model.train()
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=0.0)
        train_loader = DataLoader(mnist_train, bsize, shuffle=True)
        # Create a TensorBoard writer
        writer = SummaryWriter(
            # 调用指令>tensorboard --logdir E:\CodeProjects\cumulativeAi\logs\tb_logs\runs\hyperparams
            log_dir=f"../../logs/tb_logs/runs/hyperparams/mnist_{bsize}_{lr}",
            comment=f"mnist_{bsize}_{lr}",
            flush_secs=2
        )
        # writer = SummaryWriter(
        #     log_dir=r"..\..\logs\tb_logs\runs\hyperparams\MinBatchSize_" + f"{bsize}_LR_{lr}"
        # )

        # Visualize the model graph
        example_data, _ = next(iter(train_loader))
        writer.add_graph(model, example_data.to(DEVICE))
        writer.close()

        # Training loop
        for epoch in range(NUM_EPOCHS):
            epoch_losses = []
            epoch_accuracies = []

            for batch_idx, (data, target) in enumerate(train_loader):
                batch_size = target.size(0)
                data, target = data.to(DEVICE), target.to(DEVICE)
                optimizer.zero_grad()
                scores = model(data)
                loss = criterion(scores, target)
                epoch_losses.append(loss.item())
                loss.backward()
                optimizer.step()

                # Calculate 'running' train accuracy

                preds = scores.argmax(dim=1)
                num_correct = (preds == target).sum().item()
                accuracy = num_correct / batch_size
                epoch_accuracies.append(accuracy)

                # Plotting on TensorBoard
                img_grid = torchvision.utils.make_grid(data)
                features = data.reshape(batch_size, -1)
                labels = [CLASSES[i] for i in preds.tolist()]
                writer.add_image('MNIST_images', img_grid)
                writer.add_histogram('fc1', model.fc1.weight)
                writer.add_scalar('Training Loss', loss.item(), global_step=step)
                writer.add_scalar('Training Accuracy', accuracy, global_step=step)

                if batch_idx % 100 == 230:
                    writer.add_embedding(features,
                                         metadata=labels,
                                         label_img=data,
                                         global_step=batch_idx)
                step += 1
            writer.add_hparams(hparam_dict={'batch_size': bsize, 'learning_rate': lr},
                               metric_dict={'loss': sum(epoch_losses) / len(epoch_losses),
                                            'accuracy': sum(epoch_accuracies) / len(epoch_accuracies)})
