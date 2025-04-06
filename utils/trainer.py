# 自己的炼丹炉
import os, time, logging
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import Optimizer
from torch.optim.lr_scheduler import CosineAnnealingLR, StepLR, ReduceLROnPlateau, _LRScheduler
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from pathlib import Path
from typing import Dict, Any, Optional, Callable, Tuple, List, Union


class Trainer:
    """
    PyTorch Trainer class for training and evaluating models.
    PyTorch模型训练器，封装了训练循环、验证、模型保存等功能。
    支持TensorBoard日志、微调、梯度裁剪和增强的检查点功能。
    """

    def __init__(self, model: nn.Module,
                 loss_fn: Callable,
                 optimizer_type: str = 'adam',
                 scheduler_type: str = 'cosine',
                 lr: float = 1e-3,
                 lr_min: float = 1e-5,
                 weight_decay: float = 1e-4,
                 clip_grad_norm: float = 5.0,
                 device: Optional[Union[torch.device, str]] = None,
                 checkpoint_dir: Optional[Union[str, Path]] = None,
                 log_dir: Optional[Union[str, Path]] = None,
                 init_weights: bool = True,
                 init_method: str = 'xavier', ):
        """
        初始化训练器 Initialize the Trainer.
        Args:
            model: 要训练的模型
            loss_fn: 损失函数
            optimizer_type: 优化器类型 ('sgd', 'adam', 'adamw', 'ranger')
            scheduler_type: 学习率调度器类型 ('cosine', 'step', 'plateau', 'none')
            lr: 初始学习率
            lr_min: 最小学习率（用于学习率调度）
            weight_decay: 权重衰减系数
            clip_grad_norm: 梯度裁剪的最大范数
            device: 训练设备 ('cuda', 'cpu', None自动选择)
            checkpoint_dir: 检查点保存目录
            log_dir: TensorBoard日志目录，None表示不使用TensorBoard
            init_weights: 是否初始化模型权重
            init_method: 权重初始化方法 ('xavier', 'kaiming', 'normal')
        """
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = model.to(self.device)
        self.loss_fn = loss_fn

        # 训练参数
        self.lr = lr
        self.lr_min = lr_min
        self.weight_decay = weight_decay
        self.clip_grad_norm = clip_grad_norm

        # 优化器
        self.optimizer_type = optimizer_type.lower()
        self.optimizer = self._create_optimizer()
        self.scheduler_type = scheduler_type.lower()
        self.scheduler = None  # 后续在train方法中初始化

        # 日志和检查点
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir = log_dir
        self.writer = SummaryWriter(log_dir=self.log_dir) if log_dir else None
        self.global_step = 0

        # 指标记录
        self.train_losses = []
        self.train_accuracies = []
        self.val_losses = []
        self.val_accuracies = []
        self.best_val_accuracy = 0.0
        self.current_epoch = 0

        # 初始化权重
        if init_weights:
            self._initialize_weights(init_method)

        # 设置日志
        self.logger = self._setup_logging()
        self.logger.info(f"训练器初始化完成，使用设备: {self.device}")

    def _setup_logging(self) -> logging.Logger:
        logger = logging.getLogger(f"{__name__}.{id(self)}")
        logger.setLevel(logging.INFO)
        if not logger.handlers:
            console_handler = logging.StreamHandler()
            console_formatter = logging.Formatter(fmt="%(asctime)s - %(levelname)s - %(message)s",
                                                  datefmt="%Y-%m-%d %H:%M:%S")
            console_handler.setFormatter(console_formatter)
            logger.addHandler(console_handler)
            file_handler = logging.FileHandler("training.log")
            file_formatter = logging.Formatter(fmt="%(asctime)s - %(levelname)s - %(message)s",
                                               datefmt="%Y-%m-%d %H:%M:%S")
            file_handler.setFormatter(file_formatter)
            logger.addHandler(file_handler)
        return logger

    def _initialize_weights(self, method: str = 'xavier') -> None:
        def init_fn(m):
            if isinstance(m, (nn.Linear, nn.Conv2d)):
                if method == 'xavier':
                    nn.init.xavier_uniform_(m.weight)
                elif method == 'kaiming':
                    nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
                elif method == 'normal':
                    nn.init.normal_(m.weight, mean=0.0, std=0.02)
                else:
                    raise ValueError(f"不支持的初始化方法: {method}. 支持方法：'xavier', 'kaiming', 'normal'.")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

        self.model.apply(init_fn)
        self.logger.info(f"模型权重初始化完成，方法: {method}")

    def _create_optimizer(self) -> Optimizer:
        # parameters = filter(lambda p: p.requires_grad, self.model.parameters())
        parameters = [p for p in self.model.parameters() if p.requires_grad]
        if self.optimizer_type == 'sgd':
            return optim.SGD(parameters, lr=self.lr, momentum=0.9, weight_decay=self.weight_decay)
        elif self.optimizer_type == 'adam':
            return optim.Adam(parameters, lr=self.lr, weight_decay=self.weight_decay)
        elif self.optimizer_type == 'adamw':
            return optim.AdamW(parameters, lr=self.lr, weight_decay=self.weight_decay)
        elif self.optimizer_type == 'ranger':
            try:
                from ranger import Ranger
                return Ranger(parameters, lr=self.lr, weight_decay=self.weight_decay)
            except ImportError:
                self.logger.warning("Ranger优化器未安装，回退到Adam")
                return optim.Adam(parameters, lr=self.lr, weight_decay=self.weight_decay)
        else:
            raise ValueError(f"不支持的优化器类型:{self.optimizer_type}. 支持类型：'sgd', 'adam', 'adamw', 'ranger'.")

    def _create_scheduler(self, num_epochs: int):
        if self.scheduler_type == 'cosine':
            return CosineAnnealingLR(self.optimizer, T_max=num_epochs, eta_min=self.lr_min)
        elif self.scheduler_type == 'step':
            return StepLR(self.optimizer, step_size=num_epochs // 3, gamma=0.1)
        elif self.scheduler_type == 'plateau':
            return ReduceLROnPlateau(self.optimizer, mode='max', factor=0.1, patience=5, min_lr=self.lr_min)
        elif self.scheduler_type == 'none':
            return None
        else:
            self.logger.warning(f"不支持的学习率调度器类型: {self.scheduler_type}. 使用默认的余弦退火调度器.")
            return CosineAnnealingLR(self.optimizer, T_max=num_epochs, eta_min=self.lr_min)

    def load_pretrained(self, pretrained_path: str, modules_to_train: List[str] = None) -> None:
        """
        加载预训练模型并选择性冻结层
        Args:
            pretrained_path: 预训练模型路径
            modules_to_train: 需要训练的模块名称列表，None表示全部训练
        """
        if os.path.isfile(pretrained_path):
            pretrained_dict = torch.load(pretrained_path, map_location=self.device, weights_only=True)
            model_dict = self.model.state_dict()
            pretrained_dict = {k: v for k, v in pretrained_dict.items() if
                               k in model_dict and model_dict[k].size() == v.size()}
            model_dict.update(pretrained_dict)
            self.model.load_state_dict(model_dict, strict=False)
            self.logger.info(f"加载预训练模型成功: {pretrained_path}")
        else:
            self.logger.warning(f"预训练模型文件不存在: {pretrained_path}")

        if modules_to_train is not None:
            for name, param in self.model.named_parameters():
                param.requires_grad = any(module_name in name for module_name in modules_to_train)
                if param.requires_grad:
                    self.logger.info(f"模块 {name} 将被训练")
                else:
                    self.logger.info(f"模块 {name} 将被冻结")
            # trainable_params = filter(lambda p: p.requires_grad, self.model.parameters())
            trainable_params = [p for p in self.model.parameters() if p.requires_grad]
            self.optimizer = self._create_optimizer()

    def train_epoch(self, train_dataloader: DataLoader,
                    val_dataloader: DataLoader,
                    clip_grad_norm: float = 1.0)-> Tuple[float, float]:
        self.model.train()
        epoch_loss = 0.0
        epoch_accuracy = 0.0
        batch_count = len(train_dataloader)