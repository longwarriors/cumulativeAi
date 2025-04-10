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

# Model
model = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1)
print(model)
for name, param in model.named_parameters():
    print(f"------ {name} ------ {param.shape}")
# model.avgpool = nn.Identity()
# model.classifier = nn.Linear(in_features=512, out_features=NUM_CLASSES)
def pretrained_resnet18():
    NUM_CLASSES = 10
    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    num_features = model.fc.in_features

    # Freeze all layers
    for param in model.parameters():
        param.requires_grad = False

    # Unfreeze the last layer
    for param in model.fc.parameters():
        param.requires_grad = True

    # Change the last layer
    model.fc = nn.Linear(num_features, NUM_CLASSES)
    print(model.fc)  # Check the new last layer

    # 查看层的名称和是否需要梯度
    for name, param in model.named_parameters():
        print(name, param.requires_grad)


def pretrained_bert():
    from transformers import BertForSequenceClassification
    NUM_CLASSES = 2
    model = BertForSequenceClassification.from_pretrained(
        pretrained_model_name_or_path="bert-base-uncased",
        num_labels=NUM_CLASSES)

    # 冻结嵌入层和前 6 层
    for name, param in model.named_parameters():
        # and 的优先级高于 or
        if "embeddings" in name or "layer" in name and int(name.split("layer.")[1].split(".")[0]) < 6:
            param.requires_grad = False

    # 查看层的名称和是否需要梯度
    for name, param in model.named_parameters():
        print(name, param.requires_grad)


# if __name__ == "__main__":
    # pretrained_resnet18()
    # pretrained_bert()
