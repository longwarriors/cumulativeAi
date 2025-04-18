# https://www.youtube.com/watch?v=qaDe0qQZ5AQ

import torch
from scipy.cluster.hierarchy import weighted
from torch import nn, optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torchvision import models

# Set device
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Hyperparameters
IN_CHANNEL = 3
NUM_CLASSES = 10
LEARNING_RATE = 1e-3
BATCH_SIZE = 1024
NUM_EPOCHS = 5

# Load pretrained model and modify it
model = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1)
print(model)
for name, param in model.named_parameters():
    print(f"------ {name} ------ {param.shape}")
# model.avgpool = nn.Identity()
# model.classifier = nn.Linear(in_features=512, out_features=NUM_CLASSES)



# Load dataset
train_set = datasets.CIFAR10(root='../../data', train=True, download=False, transform=transforms.ToTensor())
test_set = datasets.CIFAR10(root='../../data', train=False, download=False, transform=transforms.ToTensor())
