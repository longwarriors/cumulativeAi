import os, copy, torch
from torch import optim, nn
from torch.utils.data import DataLoader

# https://github.com/copilot/c/056c785d-6708-42f4-bedf-b7d03dca73d9
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
