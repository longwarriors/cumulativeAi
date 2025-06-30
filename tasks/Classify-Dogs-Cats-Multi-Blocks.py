# https://www.zhihu.com/question/523869554/answer/2560312612
# https://claude.ai/chat/51496ea8-ae14-44ba-8524-e76e8da24e17
# https://github.com/copilot/c/086de222-f492-4864-a9b2-f571b7870000
import warnings, os, copy, logging, torch
import torch.nn.functional as F
import numpy as np
from sklearn.utils.class_weight import compute_class_weight
from torch import optim, nn
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import LRScheduler, CosineAnnealingLR, StepLR, ReduceLROnPlateau
from typing import Callable, Optional, Union, List, Tuple, Dict, Any
from pathlib import Path
from datetime import datetime
from tqdm import tqdm

warnings.filterwarnings("ignore")


# 设置随机种子
def set_seed(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


set_seed(42)


class Trainer:
    """PyTorch模型训练器，封装了训练循环、验证、模型保存等功能。"""

    def __init__(self,
                 model: nn.Module,
                 loss_fn: Optional[Callable] = None,
                 optimizer_type: str = 'sgd',
                 scheduler_type: str = 'cosine',
                 learning_rate: float = 1e-3,
                 min_learning_rate: float = 1e-6,
                 weight_decay: float = 0.0,
                 device: Optional[str] = None,
                 checkpoint_dir: str = 'checkpoints',
                 init_weights: bool = True,
                 init_weights_method: str = 'xavier',
                 class_weight_mode: Optional[Union[str, List[float], torch.Tensor]] = None,
                 num_classes: Optional[int] = None
                 ):
        """
        初始化Trainer实例。
        :param model: 要训练的模型
        :param loss_fn: 损失函数，如果为None且指定了class_weights，会自动创建带权重的CrossEntropyLoss
        :param optimizer_type: 优化器类型， ('sgd', 'adam', 'adamw', 'ranger')
        :param scheduler_type: 学习率调度器类型， ('cosine', 'step', 'plateau', 'none')
        :param learning_rate: 初始学习率
        :param min_learning_rate: 最小学习率（用于调度器）
        :param weight_decay: 优化器的权重衰减
        :param device: 设备类型，('cuda', 'cpu', None自动选择)
        :param checkpoint_dir: 模型检查点保存目录
        :param init_weights: 是否初始化模型权重
        :param init_weights_method: 权重初始化方法，('xavier', 'kaiming', 'normal')
        :param class_weight_mode: 类权重设置
            - 'balanced'：自动计算类别权重
            - 'balanced_subsample': 使用子采样计算平衡权重
            - torch.Tensor或List[float]: 手动指定权重
            - None: 不使用类别权重
        :param num_classes: 类别数量，用于验证权重长度
        """
        # 设置设备
        self.device = device or 'cuda' if torch.cuda.is_available() else 'cpu'

        # 模型
        self.model = model.to(self.device)

        # 训练参数
        self.lr = learning_rate
        self.lr_min = min_learning_rate
        self.weight_decay = weight_decay

        # 类权重和损失函数设置
        self.class_weight_mode = class_weight_mode
        self.class_weights = None  # 实际使用的权重tensor，在需要时自定义
        self.num_classes = num_classes
        if isinstance(self.class_weight_mode, (list, torch.Tensor)):
            self._set_class_weights(self.class_weight_mode)
            if self.num_classes is None:
                self.num_classes = len(self.class_weights)

        # 损失函数 - 延迟初始化，在第一次训练时根据数据自动计算权重
        self.loss_fn = self._init_loss_fn(loss_fn)

        # 优化器和调度器
        self.optimizer_type = optimizer_type.lower()
        self.optimizer = self._create_optimizer()
        self.scheduler = None  # 后续在train方法中初始化，因为需要知道epoch数
        self.scheduler_type = scheduler_type.lower()

        # 检查点目录
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(exist_ok=True, parents=True)

        # 指标记录
        self.train_losses = []
        self.train_accuracies = []
        self.valid_losses = []
        self.valid_accuracies = []
        self.best_valid_acc: float = 0.0
        self.best_model_wts: dict = copy.deepcopy(model.state_dict())
        self.current_epoch = 0

        # 初始化权重
        if init_weights:
            self._initialize_model_weights(init_weights_method)

        # 设置日志
        self.logger = self._setup_logging()
        self.logger.info(f"训练器初始化完成，使用设备: {self.device}")

    def _setup_logging(self) -> logging.Logger:
        logger = logging.getLogger(f"{__name__}.{id(self)}")  # 使用唯一ID避免冲突
        logger.setLevel(logging.INFO)
        if not logger.hasHandlers():  # 避免重复添加处理器
            console_handler = logging.StreamHandler()
            console_formatter = logging.Formatter(
                "%(asctime)s - %(levelname)s - %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S"
            )
            console_handler.setFormatter(console_formatter)
            logger.addHandler(console_handler)

            # 使用时间戳生成日志文件名
            log_file = self.checkpoint_dir / f"training_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
            file_handler = logging.FileHandler(log_file, encoding='utf-8')
            file_formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                datefmt="%Y-%m-%d %H:%M:%S"
            )
            file_handler.setFormatter(file_formatter)
            logger.addHandler(file_handler)
        return logger

    def _initialize_model_weights(self, method: str = 'xavier') -> None:
        def init_func(m):
            if isinstance(m, (nn.Linear, nn.Conv2d)):
                if method == 'xavier':
                    nn.init.xavier_normal_(m.weight)
                elif method == 'kaiming':
                    nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
                elif method == 'normal':
                    nn.init.normal_(m.weight, mean=0.0, std=0.02)
                else:
                    raise ValueError(f"未知的初始化方法: {method}")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

        self.model.apply(init_func)
        self.logger.info(f"模型权重已使用{method}方法初始化")

    def _create_optimizer(self) -> optim.Optimizer:
        params = filter(lambda p: p.requires_grad, self.model.parameters())
        if self.optimizer_type == 'sgd':
            return optim.SGD(params, lr=self.lr, momentum=0.9, weight_decay=self.weight_decay)
        elif self.optimizer_type == 'adam':
            return optim.Adam(params, lr=self.lr, weight_decay=self.weight_decay)
        elif self.optimizer_type == 'adamw':
            return optim.AdamW(params, lr=self.lr, weight_decay=self.weight_decay)
        elif self.optimizer_type == 'ranger':
            try:  # 需要安装ranger优化器
                from ranger import Ranger
                return Ranger(params, lr=self.lr, weight_decay=self.weight_decay)
            except ImportError:
                self.logger.warning("Ranger优化器未安装，回退到Adam")
                return optim.Adam(params, lr=self.lr, weight_decay=self.weight_decay)
        else:
            raise ValueError(f"不支持的优化器类型: {self.optimizer_type}")

    def _create_scheduler(self, num_epochs: int) -> Optional[LRScheduler]:
        if self.scheduler_type == 'cosine':
            return CosineAnnealingLR(self.optimizer, T_max=num_epochs, eta_min=self.lr_min)
        elif self.scheduler_type == 'step':
            return StepLR(self.optimizer, step_size=num_epochs // 3, gamma=0.1)
        elif self.scheduler_type == 'plateau':
            return ReduceLROnPlateau(self.optimizer, mode='max', factor=0.1, patience=num_epochs // 5,
                                     min_lr=self.lr_min)
        elif self.scheduler_type == 'none':
            return None
        else:
            self.logger.warning(f"不支持的调度器类型: {self.scheduler_type}，使用Cosine")
            return CosineAnnealingLR(self.optimizer, T_max=num_epochs, eta_min=self.lr_min)

    def calculate_class_weights(self, dl_train: DataLoader) -> torch.Tensor:
        """
        计算类别权重并设置到损失函数中。
        :param dl_train: 数据加载器
        """
        self.logger.info("正在计算类权重...")
        all_labels = []
        with tqdm(dl_train, desc="收集类别标签") as pbar:
            for _, labels in pbar:
                if isinstance(labels, torch.Tensor):
                    all_labels.extend(labels.cpu().numpy())
                else:
                    all_labels.extend(labels)
        all_labels = np.array(all_labels)
        unique_labels = np.unique(all_labels)

        # 更新类别数量
        if self.num_classes is None:
            self.num_classes = len(unique_labels)
            self.logger.info(f"检测到类别数量: {self.num_classes}")

        # 计算权重策略
        if self.class_weight_mode == 'balanced':
            class_weights = compute_class_weight(class_weight='balanced',
                                                 classes=unique_labels,
                                                 y=all_labels)
        elif self.class_weight_mode == 'balanced_subsample':
            subsample_size = min(10000, len(all_labels))  # 使用子采样进行平衡权重计算
            subsample_indices = np.random.choice(len(all_labels), size=subsample_size, replace=False)
            subsample_labels = all_labels[subsample_indices]
            class_weights = compute_class_weight(class_weight='balanced',
                                                 classes=unique_labels,
                                                 y=subsample_labels)
        else:
            raise ValueError(f"不支持的权重计算方式: {self.class_weight_mode}")
        class_weights_tensor = torch.as_tensor(class_weights, dtype=torch.float32, device=self.device)
        self.logger.info(f"类别数量: {len(unique_labels)}")
        self.logger.info(f"类别分布: {np.bincount(all_labels)}")
        self.logger.info(f"计算得到的类权重: {class_weights}")
        return class_weights_tensor

    def _set_class_weights(self, weights: Union[torch.Tensor, List[float]]) -> None:
        if isinstance(weights, list):
            weights = torch.as_tensor(weights, dtype=torch.float32)
        self.class_weights_tensor = weights.to(self.device)

        # 验证权重长度
        if self.num_classes is not None and len(self.class_weights_tensor) != self.num_classes:
            raise ValueError(f"类权重长度 {len(self.class_weights_tensor)} 与类别数量 {self.num_classes} 不匹配")

        # 更新损失函数
        if self._loss_fn_template is None:
            self.loss_fn = nn.CrossEntropyLoss(weight=self.class_weights_tensor)
        else:  # 提供了损失函数模板
            if hasattr(self._loss_fn_template, 'weight'):
                self.loss_fn = self._loss_fn_template.__class__(weight=self.class_weights_tensor)
            else:
                self.logger.warning("自定义损失函数不支持权重参数，将使用原始损失函数")
                self.loss_fn = self._loss_fn_template
        self.logger.info(f"类权重已设置: {self.class_weights_tensor}")

    def train_epoch(self, dl_train: DataLoader) -> Tuple[float, float]:
        """
        训练一个epoch
        :param dl_train: 训练数据加载器
        """
        self.model.train()
        epoch_loss = 0.0
        epoch_accuracy = 0.0
        batch_count = len(dl_train)
        with tqdm(dl_train, desc="训练中") as pbar:
            for batch_idx, (inputs, targets) in enumerate(pbar):
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)

                # 前向传播
                self.optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = self.loss_fn(outputs, targets)

                # 反向传播和优化
                loss.backward()
                self.optimizer.step()

                # 计算准确率
                preds = torch.argmax(outputs, dim=1)
                num_corrects = (preds == targets).sum().item()
                accuracy = num_corrects / inputs.size(0)

                # 更新统计信息
                epoch_loss += loss.item()
                epoch_accuracy += accuracy

                # 更新进度条
                pbar.set_postfix({
                    'loss': f"{loss.item():.4f}",
                    'accuracy': f"{accuracy:.4f}"
                })
        # 计算平均损失和准确率
        avg_loss = epoch_loss / batch_count
        avg_accuracy = epoch_accuracy / batch_count
        return avg_loss, avg_accuracy

    def validate_epoch(self, dl_valid: DataLoader) -> Tuple[float, float]:
        pass

    def save_checkpoint(self):
        pass
