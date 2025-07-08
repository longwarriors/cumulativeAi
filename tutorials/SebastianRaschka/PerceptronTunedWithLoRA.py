# https://github.com/copilot/c/ab0c6e0e-7d75-4cf6-a3fb-a171c0d4f362
import os, torch, copy
import torch.nn.functional as F
from torch import nn, optim
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
from models import add_lora_to_linear, train_loop_with_resume_lora
from utils import EarlyStopping

###########################
#### mnist dataset
###########################
BATCH_SIZE = 128
TRAIN_RATIO = 0.8
labelled_set = datasets.MNIST(root='../../data', train=True, transform=transforms.ToTensor(), download=False)
test_set = datasets.MNIST(root='../../data', train=False, transform=transforms.ToTensor(), download=False)

# 划分训练集和验证集
train_size = int(TRAIN_RATIO * len(labelled_set))
valid_size = len(labelled_set) - train_size
train_set, valid_set = random_split(labelled_set, [train_size, valid_size])
train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
valid_loader = DataLoader(valid_set, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=False)

# 检查数据集张量维度
check_batch = next(iter(train_loader))
print(f"批次数据维度: {check_batch[0].shape}, 标签维度: {check_batch[1].shape}")


#################################
######## Perceptron Model
#################################
class MultiLayerPerceptron(nn.Module):
    def __init__(self, in_features, hidden1, hidden2, out_features, rank=4, alpha=1.0):
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden1)
        self.fc2 = nn.Linear(hidden1, hidden2)
        self.fc3 = nn.Linear(hidden2, out_features)

    def forward(self, x):
        x = x.view(x.size(0), -1)  # 展平输入
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.fc3(x)
        return x


#################################
######## Architecture
#################################
n_features = 28 * 28  # MNIST images are 28x28 pixels
n_classes = 10
n_hidden1 = 256
n_hidden2 = 64
model_pretrained = MultiLayerPerceptron(n_features, n_hidden1, n_hidden2, n_classes)
lora_rank = 4
lora_alpha = 1.5
print("改造前的模型结构:")
print(model_pretrained)

#############################
#### Hyperparameters
#############################
SEED = 123
torch.manual_seed(SEED)
LEARNING_RATE = 1e-4
NUM_EPOCHS = 50

#################################
######## 模型装配
#################################
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
PRETRAINED_CHECKPOINT_PATH = "best_pretrained_20250707.pth"
BEST_LORA_CHECKPOINT_PATH = "best_lora_parameters.pth"

# 基础模型参数加载
base_model_ckpt = torch.load(PRETRAINED_CHECKPOINT_PATH, map_location=DEVICE)
model_pretrained.load_state_dict(base_model_ckpt['model_state_dict'])
print(f"成功加载预训练基础模型权重: {PRETRAINED_CHECKPOINT_PATH}")

# 微调模型LoRA参数加载
model_lora = copy.deepcopy(model_pretrained)
add_lora_to_linear(model_lora, rank=lora_rank, alpha=lora_alpha)
print("\n改造后的模型结构:")
print(model_lora)
print("\n验证LoRA模型的可训练参数:")
for name, param in model_lora.named_parameters():
    if param.requires_grad:
        print(f"可训练 (Trainable): {name}: 形状: {param.shape}")
    else:
        print(f"已冻结 (Frozen): {name}: 形状: {param.shape}")
optimizer = optim.Adam(filter(lambda p: p.requires_grad, model_lora.parameters()), lr=LEARNING_RATE)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS, eta_min=1e-6)
criterion = nn.CrossEntropyLoss()
early_stopping = EarlyStopping(patience=5, delta=1e-3, mode="max")
best_acc, record = train_loop_with_resume_lora(model_lora,
                                               train_loader,
                                               valid_loader,
                                               criterion,
                                               optimizer,
                                               scheduler,
                                               early_stopping,
                                               BEST_LORA_CHECKPOINT_PATH,
                                               NUM_EPOCHS,
                                               DEVICE, )
print(f"训练过程损失: {record['train_loss']}")
print(f"训练过程准确率: {record['train_acc']}")
print(f"验证过程损失: {record['valid_loss']}")
print(f"验证过程准确率: {record['valid_acc']}")
