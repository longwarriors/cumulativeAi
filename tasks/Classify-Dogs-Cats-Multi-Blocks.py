import os, copy, torch
import torch.nn.functional as F
import numpy as np
from sklearn.utils.class_weight import compute_class_weight
from torch import optim, nn
from torch.utils.data import DataLoader

torch.manual_seed(42)


# https://github.com/copilot/c/056c785d-6708-42f4-bedf-b7d03dca73d9
# https://claude.ai/chat/51496ea8-ae14-44ba-8524-e76e8da24e17
class Trainer:
    def __init__(self,
                 model: nn.Module,
                 optimizer: optim.Optimizer,
                 criterion: nn.Module,
                 train_loader: DataLoader,
                 valid_loader: DataLoader,
                 device: str,
                 checkpoint_dir: str):
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.device = device
        self.checkpoint_dir = checkpoint_dir
        self.start_epoch: int = 0
        self.best_valid_acc: float = 0.0

        self.best_model_wts: dict = copy.deepcopy(model.state_dict())

    def save_checkpoint(self):
        pass


# 计算类别权重
class_weights = compute_class_weight(class_weight='balanced',
                                     classes=np.unique(y_train),
                                     y=y_train)
# 转换为 PyTorch 张量
class_weights = torch.as_tensor(class_weights, dtype=torch.float)



