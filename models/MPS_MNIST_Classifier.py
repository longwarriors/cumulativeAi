import torch
from torch import nn, optim
from torch.utils.data import DataLoader, TensorDataset, random_split
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
from typing import Optional, Dict, Tuple, List
import matplotlib.pyplot as plt
from dataclasses import dataclass
import logging
from torchvision import datasets, transforms
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns


@dataclass
class MNISTConfig:
    """MNIST训练配置类"""
    num_sites: int = 784  # 28*28 = 784 pixels
    phys_dim: int = 2  # 二值化
    bond_dim: int = 16  # 增加bond维度以处理更复杂的数据
    max_bond_dim: int = 32
    learning_rate: float = 0.005  # 稍微降低学习率
    num_sweeps: int = 30
    num_local_epochs: int = 8
    batch_size: int = 128
    early_stopping_patience: int = 5
    convergence_threshold: float = 1e-6
    device: str = 'auto'
    num_classes: int = 10
    binary_threshold: float = 0.5  # 二值化阈值


class ImprovedMPS(nn.Module):
    """改进的MPS模型，适配MNIST分类"""

    def __init__(self, num_sites: int, phys_dim: int, bond_dim: int, num_classes: int = 10):
        super().__init__()
        self.num_sites = num_sites
        self.phys_dim = phys_dim
        self.bond_dim = bond_dim
        self.num_classes = num_classes
        self.tensor_sites = nn.ParameterList()

        # 初始化MPS张量
        self._initialize_tensors()
        self._canonical_form()

        # 分类层：将最后的bond维度映射到类别数
        final_bond_dim = self.tensor_sites[-1].shape[2]
        self.classifier = nn.Linear(final_bond_dim, num_classes)

    def _initialize_tensors(self):
        """智能初始化策略，适应长序列"""
        # 使用更智能的bond维度分配
        bond_dims = self._compute_optimal_bond_dims()

        for i in range(self.num_sites):
            left_bond = bond_dims[i]
            right_bond = bond_dims[i + 1]
            shape = (left_bond, self.phys_dim, right_bond)

            # 使用Xavier初始化
            fan_in = left_bond * self.phys_dim
            fan_out = self.phys_dim * right_bond
            std = np.sqrt(2.0 / (fan_in + fan_out))

            tensor = torch.randn(shape) * std

            # 对中间张量进行正交化
            if i > 0 and i < self.num_sites - 1:
                tensor = self._orthogonalize_tensor(tensor)

            self.tensor_sites.append(nn.Parameter(tensor))

    def _compute_optimal_bond_dims(self) -> List[int]:
        """计算最优的bond维度分布"""
        bond_dims = [1]  # 左边界

        for i in range(self.num_sites):
            if i < self.num_sites // 2:
                # 前半部分：指数增长但受限于bond_dim
                next_bond = min(self.bond_dim, bond_dims[-1] * self.phys_dim)
            else:
                # 后半部分：对称减少
                mirror_pos = self.num_sites - 1 - i
                if mirror_pos < len(bond_dims):
                    next_bond = bond_dims[mirror_pos]
                else:
                    next_bond = max(1, bond_dims[-1] // self.phys_dim)

            next_bond = max(1, min(next_bond, self.bond_dim))
            bond_dims.append(next_bond)

        bond_dims[-1] = 1  # 右边界
        return bond_dims

    def _orthogonalize_tensor(self, tensor: torch.Tensor) -> torch.Tensor:
        """正交化张量"""
        left_dim, phys_dim, right_dim = tensor.shape
        matrix = tensor.reshape(left_dim * phys_dim, right_dim)

        if matrix.shape[0] >= matrix.shape[1]:
            q, r = torch.linalg.qr(matrix)
            return q.reshape(left_dim, phys_dim, -1)
        else:
            q, r = torch.linalg.qr(matrix.T)
            return q.T.reshape(left_dim, phys_dim, -1)

    def _canonical_form(self):
        """将MPS转换为左正交形式"""
        with torch.no_grad():
            for i in range(self.num_sites - 1):
                tensor = self.tensor_sites[i]
                left_dim, phys_dim, right_dim = tensor.shape

                matrix = tensor.reshape(left_dim * phys_dim, right_dim)

                try:
                    q, r = torch.linalg.qr(matrix)

                    # 更新当前张量
                    new_right_dim = q.shape[1]
                    self.tensor_sites[i].data = q.reshape(left_dim, phys_dim, new_right_dim)

                    # 将R矩阵吸收到下一个张量
                    next_tensor = self.tensor_sites[i + 1]
                    next_left, next_phys, next_right = next_tensor.shape

                    # 调整维度匹配
                    if r.shape[0] != next_left:
                        # 如果维度不匹配，截断或填充
                        min_dim = min(r.shape[0], next_left)
                        r_adjusted = torch.zeros(next_left, r.shape[1], device=r.device, dtype=r.dtype)
                        r_adjusted[:min_dim, :] = r[:min_dim, :]
                        r = r_adjusted

                    next_matrix = next_tensor.reshape(next_left, next_phys * next_right)
                    new_next = torch.mm(r, next_matrix).reshape(r.shape[0], next_phys, next_right)
                    self.tensor_sites[i + 1].data = new_next

                except Exception as e:
                    print(f"QR decomposition failed at site {i}: {e}")
                    continue

    def forward(self, x_batch: torch.Tensor,
                tensors_override: Optional[Dict[int, torch.Tensor]] = None) -> torch.Tensor:
        """前向传播"""
        batch_size = x_batch.shape[0]
        device = x_batch.device

        # 初始化左边界向量
        left_vector = torch.ones(batch_size, 1, device=device, dtype=x_batch.dtype)

        for i in range(self.num_sites):
            A_i = (tensors_override or {}).get(i, self.tensor_sites[i])
            x_i = x_batch[:, i, :]  # (batch_size, phys_dim)

            # 张量收缩
            try:
                # 收缩物理维度: x_i @ A_i
                contracted = torch.einsum('bp,lpd->bld', x_i, A_i)
                # 收缩左键: left_vector @ contracted
                left_vector = torch.einsum('bl,bld->bd', left_vector, contracted)
            except Exception as e:
                print(f"Contraction failed at site {i}: {e}")
                print(f"Shapes: x_i={x_i.shape}, A_i={A_i.shape}, left_vector={left_vector.shape}")
                raise e

        # 通过分类器得到最终输出
        logits = self.classifier(left_vector)  # (batch_size, num_classes)
        return logits

    def get_entanglement_entropy(self, site: int) -> float:
        """计算指定位置的纠缠熵"""
        if site >= self.num_sites - 1:
            return 0.0

        with torch.no_grad():
            try:
                # 简化的纠缠熵计算，避免内存问题
                if site < 10:  # 只计算前几个位置的纠缠熵
                    tensor = self.tensor_sites[site]
                    left_dim, phys_dim, right_dim = tensor.shape

                    # 重塑并进行SVD
                    matrix = tensor.sum(dim=1)  # 对物理维度求和
                    if matrix.numel() > 0:
                        try:
                            _, s, _ = torch.linalg.svd(matrix)
                            s = s[s > 1e-12]
                            if len(s) > 0:
                                s_normalized = s / s.sum()
                                entropy = -torch.sum(s_normalized * torch.log(s_normalized + 1e-12))
                                return entropy.item()
                        except:
                            return 0.0
                return 0.0
            except:
                return 0.0


class AdaptiveDMRGOptimizer:
    """适配MNIST的DMRG优化器"""

    def __init__(self, mps_model: ImprovedMPS, config: MNISTConfig):
        self.mps_model = mps_model
        self.config = config
        self.convergence_history = []
        self.loss_history = []
        self.best_loss = float('inf')
        self.patience_counter = 0

    def train(self, train_loader: DataLoader, val_loader: DataLoader,
              loss_fn: nn.Module) -> Dict:
        """训练流程"""
        training_history = {
            'train_loss': [],
            'val_loss': [],
            'train_acc': [],
            'val_acc': [],
            'entanglement': []
        }

        pbar = tqdm(range(self.config.num_sweeps), desc="DMRG Training on MNIST")

        for sweep in pbar:
            direction = 'left_to_right' if sweep % 2 == 0 else 'right_to_left'

            # 训练一个sweep，但每隔几个sweep才做一次完整的优化
            if sweep % 3 == 0:  # 每3个sweep做一次完整优化
                train_metrics = self._sweep(train_loader, loss_fn, direction)
            else:
                train_metrics = self._light_sweep(train_loader, loss_fn)

            # 验证
            val_metrics = self._evaluate(val_loader, loss_fn)

            # 记录历史
            training_history['train_loss'].append(train_metrics['loss'])
            training_history['train_acc'].append(train_metrics['accuracy'])
            training_history['val_loss'].append(val_metrics['loss'])
            training_history['val_acc'].append(val_metrics['accuracy'])

            # 计算平均纠缠熵（简化版）
            entanglement = self._compute_average_entanglement()
            training_history['entanglement'].append(entanglement)

            # 更新loss历史
            self.loss_history.append(val_metrics['loss'])

            # 更新进度条
            pbar.set_postfix({
                'train_loss': f"{train_metrics['loss']:.4f}",
                'train_acc': f"{train_metrics['accuracy']:.4f}",
                'val_loss': f"{val_metrics['loss']:.4f}",
                'val_acc': f"{val_metrics['accuracy']:.4f}",
                'entanglement': f"{entanglement:.3f}"
            })

            # 早停检查
            if self._early_stopping_check(val_metrics['loss']):
                print(f"Early stopping at sweep {sweep}")
                break

        return training_history

    def _light_sweep(self, train_loader: DataLoader, loss_fn: nn.Module) -> Dict:
        """轻量级sweep，只优化分类器"""
        total_loss = 0.0
        total_correct = 0
        total_samples = 0

        # 创建分类器优化器
        classifier_optimizer = optim.Adam(self.mps_model.classifier.parameters(),
                                          lr=self.config.learning_rate)

        for x_batch, y_batch in train_loader:
            device = next(self.mps_model.parameters()).device
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)

            classifier_optimizer.zero_grad()

            # 前向传播
            logits = self.mps_model(x_batch)
            loss = loss_fn(logits, y_batch)

            # 反向传播
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.mps_model.classifier.parameters(), max_norm=1.0)
            classifier_optimizer.step()

            # 统计
            total_loss += loss.item() * x_batch.size(0)
            preds = torch.argmax(logits, dim=1)
            total_correct += (preds == y_batch).sum().item()
            total_samples += x_batch.size(0)

        return {
            'loss': total_loss / total_samples,
            'accuracy': total_correct / total_samples
        }

    def _sweep(self, train_loader: DataLoader, loss_fn: nn.Module,
               direction: str) -> Dict:
        """完整sweep，优化MPS张量"""
        total_loss = 0.0
        total_correct = 0
        total_samples = 0

        # 选择要优化的位置（不优化所有位置以节省时间）
        if direction == 'left_to_right':
            positions = list(range(0, min(50, self.mps_model.num_sites - 1), 5))  # 每5个位置选一个
        else:
            positions = list(reversed(range(5, min(50, self.mps_model.num_sites), 5)))

        for x_batch, y_batch in train_loader:
            device = next(self.mps_model.parameters()).device
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)

            for pos in positions:
                site1_idx = pos
                site2_idx = min(pos + 1, self.mps_model.num_sites - 1)

                if site1_idx != site2_idx:
                    loss = self._optimize_adjacent_sites(
                        site1_idx, site2_idx, x_batch, y_batch, loss_fn
                    )
                    total_loss += loss * x_batch.size(0)

            # 计算准确率
            with torch.no_grad():
                logits = self.mps_model(x_batch)
                preds = torch.argmax(logits, dim=1)
                total_correct += (preds == y_batch).sum().item()
                total_samples += x_batch.size(0)

        return {
            'loss': total_loss / total_samples if total_samples > 0 else 0,
            'accuracy': total_correct / total_samples if total_samples > 0 else 0
        }

    def _optimize_adjacent_sites(self, site1_idx: int, site2_idx: int,
                                 x_batch: torch.Tensor, y_batch: torch.Tensor,
                                 loss_fn: nn.Module) -> float:
        """优化相邻张量对"""

        with torch.no_grad():
            orig_A1 = self.mps_model.tensor_sites[site1_idx].clone()
            orig_A2 = self.mps_model.tensor_sites[site2_idx].clone()
            orig_shape1 = orig_A1.shape
            orig_shape2 = orig_A2.shape

            theta_init = torch.einsum('ijk,klm->ijlm', orig_A1, orig_A2)

        theta_param = nn.Parameter(theta_init.contiguous())
        optimizer = optim.Adam([theta_param], lr=self.config.learning_rate * 0.1)  # 更小的学习率

        best_loss = float('inf')
        best_A1, best_A2 = orig_A1.clone(), orig_A2.clone()

        # 减少局部优化次数
        for epoch in range(min(3, self.config.num_local_epochs)):
            optimizer.zero_grad()

            A1_new, A2_new = self._controlled_decompose_theta(
                theta_param, orig_shape1, orig_shape2
            )

            overrides = {site1_idx: A1_new, site2_idx: A2_new}
            logits = self.mps_model(x_batch, tensors_override=overrides)
            loss = loss_fn(logits, y_batch)

            if loss.item() < best_loss:
                best_loss = loss.item()
                with torch.no_grad():
                    best_A1 = A1_new.detach().clone()
                    best_A2 = A2_new.detach().clone()

            loss.backward()
            torch.nn.utils.clip_grad_norm_([theta_param], max_norm=0.5)
            optimizer.step()

        # 更新模型参数
        with torch.no_grad():
            self.mps_model.tensor_sites[site1_idx].data.copy_(best_A1)
            self.mps_model.tensor_sites[site2_idx].data.copy_(best_A2)

        return best_loss

    def _controlled_decompose_theta(self, theta: torch.Tensor,
                                    orig_shape1: Tuple[int, ...],
                                    orig_shape2: Tuple[int, ...]) -> Tuple[torch.Tensor, torch.Tensor]:
        """受控SVD分解"""
        chi_L, d1, d2, chi_R = theta.shape
        matrix = theta.reshape(chi_L * d1, d2 * chi_R)

        try:
            U, S, Vh = torch.linalg.svd(matrix, full_matrices=False)

            original_bond = orig_shape1[2]
            k = min(len(S), original_bond, self.config.max_bond_dim)
            k = max(k, 1)

            U_trunc = U[:, :k]
            S_trunc = S[:k]
            Vh_trunc = Vh[:k, :]

            sqrt_S = torch.sqrt(S_trunc + 1e-12)

            A1_new = (U_trunc * sqrt_S.unsqueeze(0)).reshape(chi_L, d1, k)
            A2_new = (sqrt_S.unsqueeze(1) * Vh_trunc).reshape(k, d2, chi_R)

            A1_new = self._adjust_tensor_shape(A1_new, orig_shape1)
            A2_new = self._adjust_tensor_shape(A2_new, orig_shape2)

            return A1_new, A2_new

        except Exception as e:
            print(f"SVD failed: {e}")
            device = theta.device
            dtype = theta.dtype
            A1_fallback = torch.randn(orig_shape1, device=device, dtype=dtype) * 0.01
            A2_fallback = torch.randn(orig_shape2, device=device, dtype=dtype) * 0.01
            return A1_fallback, A2_fallback

    def _adjust_tensor_shape(self, tensor: torch.Tensor, target_shape: Tuple[int, ...]) -> torch.Tensor:
        """调整张量形状"""
        current_shape = tensor.shape

        if current_shape == target_shape:
            return tensor

        adjusted = torch.zeros(target_shape, device=tensor.device, dtype=tensor.dtype)
        min_dims = [min(c, t) for c, t in zip(current_shape, target_shape)]

        if len(min_dims) == 3:
            adjusted[:min_dims[0], :min_dims[1], :min_dims[2]] = \
                tensor[:min_dims[0], :min_dims[1], :min_dims[2]]

        return adjusted

    def _evaluate(self, data_loader: DataLoader, loss_fn: nn.Module) -> Dict:
        """评估模型"""
        self.mps_model.eval()
        total_loss = 0.0
        total_correct = 0
        total_samples = 0

        with torch.no_grad():
            for x_batch, y_batch in data_loader:
                device = next(self.mps_model.parameters()).device
                x_batch, y_batch = x_batch.to(device), y_batch.to(device)

                logits = self.mps_model(x_batch)
                loss = loss_fn(logits, y_batch)

                preds = torch.argmax(logits, dim=1)

                total_loss += loss.item() * x_batch.size(0)
                total_correct += (preds == y_batch).sum().item()
                total_samples += x_batch.size(0)

        self.mps_model.train()
        return {
            'loss': total_loss / total_samples,
            'accuracy': total_correct / total_samples
        }

    def _compute_average_entanglement(self) -> float:
        """计算平均纠缠熵（简化版）"""
        entropies = []
        for i in range(min(10, self.mps_model.num_sites - 1)):  # 只计算前10个
            try:
                entropy = self.mps_model.get_entanglement_entropy(i)
                if not np.isnan(entropy) and not np.isinf(entropy):
                    entropies.append(entropy)
            except:
                continue

        return np.mean(entropies) if entropies else 0.0

    def _early_stopping_check(self, val_loss: float) -> bool:
        """早停检查"""
        if val_loss < self.best_loss - self.config.convergence_threshold:
            self.best_loss = val_loss
            self.patience_counter = 0
            return False
        else:
            self.patience_counter += 1
            return self.patience_counter >= self.config.early_stopping_patience


def load_and_preprocess_mnist(config: MNISTConfig) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """加载和预处理MNIST数据"""

    # 定义变换：归一化并二值化
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),  # MNIST标准化
        transforms.Lambda(lambda x: (x > config.binary_threshold).float())  # 二值化
    ])

    # 加载数据
    train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST('./data', train=False, transform=transform)

    # 数据预处理：展平并转换为one-hot
    def preprocess_dataset(dataset):
        data, targets = [], []
        for img, label in dataset:
            # 展平图像
            flat_img = img.view(-1)  # (784,)
            # 转换为one-hot编码
            one_hot = F.one_hot(flat_img.long(), num_classes=config.phys_dim).float()
            data.append(one_hot)
            targets.append(label)

        return torch.stack(data), torch.tensor(targets)

    print("Preprocessing MNIST data...")
    train_X, train_y = preprocess_dataset(train_dataset)
    test_X, test_y = preprocess_dataset(test_dataset)

    # 创建训练/验证分割
    train_size = int(0.9 * len(train_X))
    val_size = len(train_X) - train_size

    indices = torch.randperm(len(train_X))
    train_indices, val_indices = indices[:train_size], indices[train_size:]

    train_X_split, train_y_split = train_X[train_indices], train_y[train_indices]
    val_X, val_y = train_X[val_indices], train_y[val_indices]

    # 创建数据加载器
    train_dataset_processed = TensorDataset(train_X_split, train_y_split)
    val_dataset_processed = TensorDataset(val_X, val_y)
    test_dataset_processed = TensorDataset(test_X, test_y)

    train_loader = DataLoader(train_dataset_processed, batch_size=config.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset_processed, batch_size=config.batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset_processed, batch_size=config.batch_size, shuffle=False)

    print(f"Data shapes: Train {train_X_split.shape}, Val {val_X.shape}, Test {test_X.shape}")
    print(f"Label range: {train_y.min()}-{train_y.max()}")

    return train_loader, val_loader, test_loader


def evaluate_model(model: ImprovedMPS, test_loader: DataLoader, device: torch.device):
    """详细评估模型"""
    model.eval()
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for x_batch, y_batch in tqdm(test_loader, desc="Evaluating"):
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)

            logits = model(x_batch)
            preds = torch.argmax(logits, dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(y_batch.cpu().numpy())

    # 计算指标
    accuracy = np.mean(np.array(all_preds) == np.array(all_targets))

    print(f"\nTest Accuracy: {accuracy:.4f}")
    print("\nClassification Report:")
    print(classification_report(all_targets, all_preds))

    # 绘制混淆矩阵
    plt.figure(figsize=(10, 8))
    cm = confusion_matrix(all_targets, all_preds)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.show()

    return accuracy


def plot_training_history(history: Dict):
    """绘制训练历史"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))

    # Loss
    axes[0, 0].plot(history['train_loss'], label='Train Loss', alpha=0.7)
    axes[0, 0].plot(history['val_loss'], label='Val Loss', alpha=0.7)
    axes[0, 0].set_title('Loss vs Sweeps')
    axes[0, 0].set_xlabel('Sweep')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True)

    # Accuracy
    axes[0, 1].plot(history['train_acc'], label='Train Acc', alpha=0.7)
    axes[0, 1].plot(history['val_acc'], label='Val Acc', alpha=0.7)
    axes[0, 1].set_title('Accuracy vs Sweeps')
    axes[0, 1].set_xlabel('Sweep')
    axes[0, 1].set_ylabel('Accuracy')
    axes[0, 1].legend()
    axes[0, 1].grid(True)

    # Entanglement
    axes[1, 0].plot(history['entanglement'], alpha=0.7)
    axes[1, 0].set_title('Average Entanglement Entropy')
    axes[1, 0].set_xlabel('Sweep')
    axes[1, 0].set_ylabel('Entropy')
    axes[1, 0].grid(True)

    # Validation accuracy zoom
    if len(history['val_acc']) > 5:
        axes[1, 1].plot(history['val_acc'][-10:], 'ro-', alpha=0.7)
        axes[1, 1].set_title('Recent Validation Accuracy')
        axes[1, 1].set_xlabel('Recent Sweeps')
        axes[1, 1].set_ylabel('Accuracy')
        axes[1, 1].grid(True)

    plt.tight_layout()
    plt.show()


def main():
    """MNIST分类主函数"""
    # 配置
    config = MNISTConfig(
        num_sites=784,  # 28*28
        phys_dim=2,  # 二值化
        bond_dim=16,
        max_bond_dim=32,
        learning_rate=0.005,
        num_sweeps=30,
        num_local_epochs=8,
        batch_size=128,
        early_stopping_patience=5,
        device='cuda' if torch.cuda.is_available() else 'cpu'
    )

    device = torch.device(config.device)
    print(f"Using device: {device}")
    print(f"Configuration: {config}")

    # 加载数据
    print("Loading MNIST dataset...")
    train_loader, val_loader, test_loader = load_and_preprocess_mnist(config)

    # 创建模型
    print("Creating MPS model...")
    model = ImprovedMPS(
        num_sites=config.num_sites,
        phys_dim=config.phys_dim,
        bond_dim=config.bond_dim,
        num_classes=config.num_classes
    ).to(device)

    # 打印模型信息
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")

    # 创建优化器
    optimizer = AdaptiveDMRGOptimizer(model, config)

    # 损失函数
    criterion = nn.CrossEntropyLoss()

    print("Starting training...")
    history = optimizer.train(train_loader, val_loader, criterion)

    # 最终评估
    print("\nFinal evaluation on test set:")
    test_accuracy = evaluate_model(model, test_loader, device)

    # 绘制训练历史
    plot_training_history(history)

    # 保存模型
    torch.save({
        'model_state_dict': model.state_dict(),
        'config': config,
        'history': history,
        'test_accuracy': test_accuracy
    }, 'mps_mnist_model.pth')

    print(f"\nModel saved! Final test accuracy: {test_accuracy:.4f}")


if __name__ == "__main__":
    main()