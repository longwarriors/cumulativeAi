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


def save_checkpoint(model, optimizer, scheduler, epoch, best_acc, path):
    checkpoint = {"model_state_dict": model.state_dict(),
                  "optimizer_state_dict": optimizer.state_dict(),
                  "scheduler_state_dict": scheduler.state_dict() if scheduler else None,
                  "epoch": epoch,
                  "best_acc": best_acc, }
    torch.save(checkpoint, path)
    print(f"Checkpoint saved to {path}")


def load_checkpoint(path, model, optimizer, scheduler=None, device="cpu"):
    checkpoint = torch.load(path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    if scheduler and checkpoint.get("scheduler_state_dict") is not None:
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
    start_epoch = checkpoint.get("epoch", 0) + 1
    best_acc = checkpoint.get("best_acc", 0.0)
    print(f"Checkpoint loaded from {path}, resume from epoch {start_epoch}")
    return start_epoch, best_acc


class EarlyStopping:
    """早停模块，用于防止过拟合"""

    def __init__(self, patience: int = 7, delta: float = 1e-4, mode: str = "min", ):
        self.patience = patience
        self.delta = delta
        self.mode = mode
        self.counter = 0
        self.best_score = None
        self.early_stop = False

    def __call__(self, metric_score: float):
        if self.mode == "min":  # 分数越低越好（如损失）
            is_improved = (self.best_score is None or metric_score < self.best_score - self.delta)
        elif self.mode == "max":  # 分数越高越好（如准确率）
            is_improved = (self.best_score is None or metric_score > self.best_score + self.delta)
        else:
            raise ValueError(f"不支持的模式: {self.mode}，应为 'min' 或 'max'")

        if is_improved:
            self.best_score = metric_score
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True


def train_epoch(model, dl_train: DataLoader, optimizer, loss_fn, device) -> Tuple[float, float]:
    model.train()
    batch_count = len(dl_train)
    total_loss: float = 0.0
    total_correct: int = 0
    total_samples: int = 0

    # 使用 with 语句确保 tqdm 正确关闭
    with tqdm(dl_train, desc="训练中", leave=False) as pbar_train:
        for batch_idx, (X, y) in enumerate(pbar_train):
            X, y = X.to(device), y.to(device)
            batch_size = X.size(0)

            # 前向传播
            optimizer.zero_grad()
            logits = model(X)
            loss = loss_fn(logits, y)

            # 反向传播
            loss.backward()
            optimizer.step()

            # 计算准确率和累积损失
            with torch.no_grad():
                preds = torch.argmax(logits, dim=-1)
                batch_correct = (preds == y).sum().item()
                total_correct += batch_correct
                total_loss += loss.item() * batch_size  # 累积加权损失
                total_samples += batch_size
                current_accuracy = (total_correct / total_samples)  # 迭代到当前batch的平均准确率
                pbar_train.set_postfix({
                    "batch_loss": f"{loss.item():.4f}",
                    "epoch_accuracy": f"{current_accuracy:.4f}",
                })
    # 计算整个epoch的平均损失和准确率
    avg_epoch_loss = total_loss / total_samples
    avg_epoch_accuracy = total_correct / total_samples
    return avg_epoch_loss, avg_epoch_accuracy


def validate_epoch(model, dl_valid: DataLoader, loss_fn, device) -> Tuple[float, float]:
    model.eval()
    batch_count = len(dl_valid)
    total_loss: float = 0.0
    total_correct: int = 0
    total_samples: int = 0
    with torch.no_grad():
        with tqdm(dl_valid, desc="验证中", leave=False) as pbar_valid:
            for batch_idx, (X, y) in enumerate(pbar_valid):
                X, y = X.to(device), y.to(device)
                batch_size = X.size(0)

                # 前向传播
                logits = model(X)
                loss = loss_fn(logits, y)

                # 计算预测结果和准确率
                preds = torch.argmax(logits, dim=-1)
                batch_correct = (preds == y).sum().item()

                # 累积统计
                total_correct += batch_correct
                total_loss += loss.item() * batch_size  # 累积加权损失
                total_samples += batch_size

                # 更新 tqdm 进度条
                current_accuracy = (total_correct / total_samples)  # 迭代到当前batch的平均准确率
                pbar_valid.set_postfix({
                    "batch_loss": f"{loss.item():.4f}",
                    "epoch_accuracy": f"{current_accuracy:.4f}",
                })
    avg_epoch_loss = total_loss / total_samples
    avg_epoch_accuracy = total_correct / total_samples
    return avg_epoch_loss, avg_epoch_accuracy


def train_loop_with_resume(pre_model,
                           train_loader,
                           valid_loader,
                           criterion,
                           optimizer,
                           scheduler,
                           early_stopping,
                           best_ckpt_file_path,
                           num_epochs,
                           device, ):
    # 训练表征曲线记录
    history = {"train_loss": [], "train_acc": [],
               "valid_loss": [], "valid_acc": [], }

    # 断点续训练
    if os.path.exists(best_ckpt_file_path):
        start_epoch, best_acc = load_checkpoint(best_ckpt_file_path, pre_model, optimizer, scheduler, device)
        print(f"加载训练点模型成功，当前准确率为{best_acc:.4f}，从第{start_epoch}个epoch开始训练...")
    else:
        start_epoch = 0
        best_acc = 0.0
        print("未找到检查点，从头开始训练...")

    for epoch in range(start_epoch, num_epochs):
        print(f"\nEpoch {epoch + 1}/{num_epochs} - {'-' * 50}")
        train_loss, train_acc = train_epoch(pre_model, train_loader, optimizer, criterion, device)
        valid_loss, valid_acc = validate_epoch(pre_model, valid_loader, criterion, device)
        print(f"当前验证准确率: {valid_acc:.4f}")
        # 记录历史
        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["valid_loss"].append(valid_loss)
        history["valid_acc"].append(valid_acc)

        # 早停
        early_stopping(valid_acc)
        if early_stopping.early_stop:
            print("早停触发!")
            break

        # 学习率调度
        scheduler.step()
        print(f"当前学习率: {scheduler.get_last_lr()[0]:.2e}")

        # 保存最佳模型
        if valid_acc > best_acc:
            best_acc = valid_acc
            print(f"更新最佳验证准确率: {best_acc:.4f}")
            save_checkpoint(pre_model, optimizer, scheduler, epoch, valid_acc, best_ckpt_file_path)
    return best_acc, history


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
                 learning_rate: float = 1e-3,
                 min_learning_rate: float = 1e-5,
                 weight_decay: float = 1e-4,
                 clip_grad_norm: float = 5.0,
                 device: Optional[Union[torch.device, str]] = None,
                 checkpoint_dir: Optional[Union[str, Path]] = 'checkpoints',
                 log_dir: Optional[Union[str, Path]] = None,
                 init_weights: bool = True,
                 init_method: str = 'xavier'):
        """
        初始化训练器 Initialize the Trainer.
        Args:
            model: 要训练的模型
            loss_fn: 损失函数
            optimizer_type: 优化器类型 ('sgd', 'adam', 'adamw', 'ranger')
            scheduler_type: 学习率调度器类型 ('cosine', 'step', 'plateau', 'none')
            learning_rate: 初始学习率
            min_learning_rate: 最小学习率（用于学习率调度）
            weight_decay: 权重衰减系数， L2正则化在优化器中的实现
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
        self.lr = learning_rate
        self.lr_min = min_learning_rate
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

    def _create_optimizer(self, parameters: Optional[List] = None) -> Optimizer:
        # parameters = filter(lambda p: p.requires_grad, self.model.parameters())
        params = parameters if parameters is not None else [p for p in self.model.parameters() if p.requires_grad]
        if self.optimizer_type == 'sgd':
            return optim.SGD(params, lr=self.lr, momentum=0.9, weight_decay=self.weight_decay)
        elif self.optimizer_type == 'adam':
            return optim.Adam(params, lr=self.lr, weight_decay=self.weight_decay)
        elif self.optimizer_type == 'adamw':
            return optim.AdamW(params, lr=self.lr, weight_decay=self.weight_decay)
        elif self.optimizer_type == 'ranger':
            try:
                from ranger import Ranger
                return Ranger(params, lr=self.lr, weight_decay=self.weight_decay)
            except ImportError:
                self.logger.warning("Ranger优化器未安装，回退到Adam")
                return optim.Adam(params, lr=self.lr, weight_decay=self.weight_decay)
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
            self.optimizer = self._create_optimizer(trainable_params)

    def train_epoch(self, train_dataloader: DataLoader, clip_grad_norm: float = 1.0) -> Tuple[float, float]:
        self.model.train()
        epoch_loss = 0.0
        epoch_accuracy = 0.0
        batch_count = len(train_dataloader)
        try:
            pbar = tqdm(train_dataloader, desc="训练中")
            for batch_idx, (inputs, targets) in enumerate(pbar):
                batch_size = inputs.size(0)  # 或 y.size(0)
                self.model.zero_grad()
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                self.optimizer.zero_grad()
                logits = self.model(inputs)
                # 获取预测类别索引
                # probabilities = torch.softmax(logits, dim=1)
                # y_hat = probabilities.argmax(dim=1)
                preds = torch.argmax(logits, dim=1)
                loss = self.loss_fn(logits, targets)
                loss.backward()
                nn.utils.clip_grad_norm_(parameters=self.model.parameters(), max_norm=clip_grad_norm)
                self.optimizer.step()
                correct = (preds == targets).sum().item()
                accuracy = correct / batch_size
                epoch_loss += loss.item() * batch_size  # 按样本数加权
                epoch_accuracy += accuracy
                if self.writer:
                    self.global_step += 1
                    self.writer.add_scalar("Train/Loss", loss.item(), self.global_step)
                    self.writer.add_scalar("Train/Accuracy", accuracy, self.global_step)
                pbar.set_postfix(loss=loss.item(), accuracy=accuracy)
        except Exception as e:
            self.logger.error(f"训练过程中发生错误: {e}")
            raise

        avg_loss = epoch_loss / batch_count
        avg_accuracy = epoch_accuracy / batch_count
        return avg_loss, avg_accuracy

    def validate(self, val_dataloader: DataLoader) -> Tuple[float, float]:
        """在验证集上评估模型"""
        self.model.eval()
        val_loss = 0.0
        val_accuracy = 0.0
        batch_count = len(val_dataloader)
        try:
            with torch.no_grad():
                pbar = tqdm(val_dataloader, desc="验证中")
                for inputs, targets in val_dataloader:
                    inputs, targets = inputs.to(self.device), targets.to(self.device)
                    logits = self.model(inputs)
                    preds = torch.argmax(logits, dim=1)
                    loss = self.loss_fn(logits, targets)
                    correct = (preds == targets).sum().item()
                    accuracy = correct / inputs.size(0)
                    epoch_loss += loss.item() * inputs.size(0)
        except Exception as e:
            self.logger.error(f"验证过程中发生错误: {e}")
            raise
        avg_loss = epoch_loss / batch_count
        avg_accuracy = epoch_accuracy / batch_count
        return avg_loss, avg_accuracy
