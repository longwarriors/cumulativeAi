# https://claude.ai/chat/76a13d3b-f455-4a60-82f0-6bb8a5e809f3 未完成对话
# https://github.com/copilot/c/ebbfb30b-a03f-4500-abff-249bb02e2e76 继续上文

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


@dataclass
class TrainingConfig:
    """训练配置类"""
    num_sites: int = 10
    phys_dim: int = 2
    bond_dim: int = 8
    max_bond_dim: int = 16
    learning_rate: float = 0.01
    num_sweeps: int = 10
    num_local_epochs: int = 5
    batch_size: int = 32
    early_stopping_patience: int = 3
    convergence_threshold: float = 1e-6
    device: str = 'auto'


class ImprovedMPS(nn.Module):
    def __init__(self, num_sites: int, phys_dim: int, bond_dim: int):
        super().__init__()
        self.num_sites = num_sites
        self.phys_dim = phys_dim
        self.bond_dim = bond_dim
        self.tensor_sites = nn.ParameterList()

        # 初始化策略
        self._initialize_tensors()
        self._canonical_form()

    def _initialize_tensors(self):
        """使用随机矩阵乘积态初始化，保证正交性"""
        current_bond = 1

        for i in range(self.num_sites):
            if i == 0:
                # 第一个张量
                next_bond = min(self.bond_dim, self.phys_dim)
                shape = (1, self.phys_dim, next_bond)
                current_bond = next_bond
            elif i == self.num_sites - 1:
                # 最后一个张量
                shape = (current_bond, self.phys_dim, 1)
            else:
                # 中间张量
                next_bond = min(self.bond_dim, current_bond * self.phys_dim)
                shape = (current_bond, self.phys_dim, next_bond)
                current_bond = next_bond

            # 使用正交随机初始化
            tensor = torch.empty(shape)
            # 重塑为矩阵进行QR分解
            left_dim = shape[0] * shape[1]
            right_dim = shape[2]

            if left_dim >= right_dim:
                # 左正交
                matrix = torch.randn(left_dim, right_dim) * 0.1
                q, r = torch.linalg.qr(matrix)
                tensor = q.reshape(shape)
            else:
                # 右正交
                matrix = torch.randn(left_dim, right_dim) * 0.1
                q, r = torch.linalg.qr(matrix.T)
                tensor = q.T.reshape(shape)

            self.tensor_sites.append(nn.Parameter(tensor))

    def _canonical_form(self):
        """将MPS转换为左正交形式"""
        with torch.no_grad():
            for i in range(self.num_sites - 1):
                tensor = self.tensor_sites[i]
                left_dim, phys_dim, right_dim = tensor.shape

                # 重塑为矩阵
                matrix = tensor.reshape(left_dim * phys_dim, right_dim)
                q, r = torch.linalg.qr(matrix)

                # 更新当前张量为左正交
                self.tensor_sites[i].data = q.reshape(left_dim, phys_dim, q.shape[1])

                # 将R矩阵吸收到下一个张量
                next_tensor = self.tensor_sites[i + 1]
                next_left, next_phys, next_right = next_tensor.shape
                next_matrix = next_tensor.reshape(next_left, next_phys * next_right)

                # R @ next_matrix
                new_next = torch.mm(r, next_matrix).reshape(r.shape[0], next_phys, next_right)
                self.tensor_sites[i + 1].data = new_next

    def forward(self, x_batch: torch.Tensor,
                tensors_override: Optional[Dict[int, torch.Tensor]] = None) -> torch.Tensor:
        """前向传播，增加数值稳定性"""
        batch_size = x_batch.shape[0]
        device = x_batch.device

        # 初始化左边界向量
        left_vector = torch.ones(batch_size, 1, device=device, dtype=x_batch.dtype)

        for i in range(self.num_sites):
            A_i = (tensors_override or {}).get(i, self.tensor_sites[i])
            x_i = x_batch[:, i, :]  # (batch_size, phys_dim)

            # 收缩操作，使用更高效的实现
            # left_vector: (batch, left_bond)
            # A_i: (left_bond, phys_dim, right_bond)
            # x_i: (batch, phys_dim)

            # 先收缩物理维度
            contracted = torch.einsum('bp,lpd->bld', x_i, A_i)  # (batch, left_bond, right_bond)
            # 再收缩左键
            left_vector = torch.einsum('bl,bld->bd', left_vector, contracted)  # (batch, right_bond)

        return left_vector  # (batch_size, 1)

    def get_entanglement_entropy(self, site: int) -> float:
        """计算指定位置的纠缠熵，修复维度问题"""
        if site >= self.num_sites - 1:
            return 0.0

        with torch.no_grad():
            try:
                # 构建转移矩阵
                transfer_matrix = None

                for i in range(site + 1):
                    tensor = self.tensor_sites[i]  # (left, phys, right)
                    # 对物理维度求迹：sum over physical dimension
                    traced = tensor.sum(dim=1)  # (left, right)

                    if transfer_matrix is None:
                        transfer_matrix = traced
                    else:
                        # 确保矩阵乘法的维度正确
                        if transfer_matrix.dim() == 1:
                            transfer_matrix = transfer_matrix.unsqueeze(0)
                        if traced.dim() == 1:
                            traced = traced.unsqueeze(1)

                        # 矩阵乘法
                        transfer_matrix = torch.mm(transfer_matrix, traced)

                # 确保是2D矩阵
                if transfer_matrix.dim() == 1:
                    transfer_matrix = transfer_matrix.unsqueeze(0)
                if transfer_matrix.dim() == 0:
                    return 0.0

                # SVD分解计算纠缠熵
                if transfer_matrix.numel() > 0:
                    U, s, Vh = torch.linalg.svd(transfer_matrix)

                    # 过滤小奇异值并归一化
                    s = s[s > 1e-12]
                    if len(s) == 0:
                        return 0.0

                    s_normalized = s / s.sum()
                    entropy = -torch.sum(s_normalized * torch.log(s_normalized + 1e-12))
                    return entropy.item()
                else:
                    return 0.0

            except Exception as e:
                print(f"Entanglement calculation failed at site {site}: {e}")
                return 0.0


class AdaptiveDMRGOptimizer:
    """自适应DMRG优化器，包含多项改进"""

    def __init__(self, mps_model: ImprovedMPS, config: TrainingConfig):
        self.mps_model = mps_model
        self.config = config
        self.convergence_history = []
        self.loss_history = []
        self.best_loss = float('inf')
        self.patience_counter = 0

    def train(self, train_loader: DataLoader, val_loader: DataLoader,
              loss_fn: nn.Module) -> Dict:
        """完整的训练流程"""
        training_history = {
            'train_loss': [],
            'val_loss': [],
            'train_acc': [],
            'val_acc': [],
            'entanglement': []
        }

        pbar = tqdm(range(self.config.num_sweeps), desc="DMRG Training")

        for sweep in pbar:
            direction = 'left_to_right' if sweep % 2 == 0 else 'right_to_left'

            # 训练一个sweep
            train_metrics = self._sweep(train_loader, loss_fn, direction)

            # 验证
            val_metrics = self._evaluate(val_loader, loss_fn)

            # 记录历史
            training_history['train_loss'].append(train_metrics['loss'])
            training_history['train_acc'].append(train_metrics['accuracy'])
            training_history['val_loss'].append(val_metrics['loss'])
            training_history['val_acc'].append(val_metrics['accuracy'])

            # 计算纠缠熵（添加异常处理）
            try:
                entanglement = self._compute_average_entanglement()
            except Exception as e:
                print(f"Entanglement calculation failed: {e}")
                entanglement = 0.0
            training_history['entanglement'].append(entanglement)

            # 更新loss历史
            self.loss_history.append(val_metrics['loss'])

            # 更新进度条
            pbar.set_postfix({
                'train_loss': f"{train_metrics['loss']:.4f}",
                'val_acc': f"{val_metrics['accuracy']:.4f}",
                'entanglement': f"{entanglement:.3f}"
            })

            # 早停检查
            if self._early_stopping_check(val_metrics['loss']):
                print(f"Early stopping at sweep {sweep}")
                break

            # 收敛检查
            if self._convergence_check():
                print(f"Converged at sweep {sweep}")
                break

        return training_history

    def _sweep(self, train_loader: DataLoader, loss_fn: nn.Module,
               direction: str) -> Dict:
        """执行一次sweep"""
        total_loss = 0.0
        total_correct = 0
        total_samples = 0

        for x_batch, y_batch in train_loader:
            device = next(self.mps_model.parameters()).device
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)

            if direction == 'left_to_right':
                indices = range(self.mps_model.num_sites - 1)
            else:
                indices = reversed(range(1, self.mps_model.num_sites))

            for i in indices:
                site1_idx = i if direction == 'left_to_right' else i - 1
                site2_idx = i + 1 if direction == 'left_to_right' else i

                loss = self._optimize_adjacent_sites(
                    site1_idx, site2_idx, x_batch, y_batch, loss_fn
                )
                total_loss += loss * x_batch.size(0)

            # 计算准确率
            with torch.no_grad():
                logits = self.mps_model(x_batch)
                preds = (torch.sigmoid(logits.squeeze()) > 0.5).float()
                total_correct += (preds == y_batch).sum().item()
                total_samples += x_batch.size(0)

        return {
            'loss': total_loss / total_samples,
            'accuracy': total_correct / total_samples
        }

    def _optimize_adjacent_sites(self, site1_idx: int, site2_idx: int,
                                 x_batch: torch.Tensor, y_batch: torch.Tensor,
                                 loss_fn: nn.Module) -> float:
        """优化相邻张量对，修复维度不匹配和梯度问题"""

        # 获取原始张量并分离梯度
        with torch.no_grad():
            orig_A1 = self.mps_model.tensor_sites[site1_idx].clone()
            orig_A2 = self.mps_model.tensor_sites[site2_idx].clone()
            orig_shape1 = orig_A1.shape
            orig_shape2 = orig_A2.shape

            # 构造初始theta
            theta_init = torch.einsum('ijk,klm->ijlm', orig_A1, orig_A2)

        # 创建可训练的theta参数
        theta_param = nn.Parameter(theta_init.contiguous())
        optimizer = optim.Adam([theta_param], lr=self.config.learning_rate)

        best_loss = float('inf')
        best_A1, best_A2 = orig_A1.clone(), orig_A2.clone()
        prev_loss = float('inf')

        for epoch in range(self.config.num_local_epochs):
            optimizer.zero_grad()

            # 分解theta为两个张量
            A1_new, A2_new = self._controlled_decompose_theta(
                theta_param, orig_shape1, orig_shape2
            )

            # 前向传播
            overrides = {site1_idx: A1_new, site2_idx: A2_new}
            y_pred = self.mps_model(x_batch, tensors_override=overrides)
            loss = loss_fn(y_pred.squeeze(), y_batch.float())

            # 保存最佳结果
            if loss.item() < best_loss:
                best_loss = loss.item()
                with torch.no_grad():
                    best_A1 = A1_new.detach().clone()
                    best_A2 = A2_new.detach().clone()

            # 反向传播
            loss.backward()

            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_([theta_param], max_norm=1.0)
            optimizer.step()

            # 早停条件
            if epoch > 0 and abs(prev_loss - loss.item()) < self.config.convergence_threshold:
                break
            prev_loss = loss.item()

        # 更新模型参数
        with torch.no_grad():
            self.mps_model.tensor_sites[site1_idx].data.copy_(best_A1)
            self.mps_model.tensor_sites[site2_idx].data.copy_(best_A2)

        return best_loss

    def _controlled_decompose_theta(self, theta: torch.Tensor,
                                    orig_shape1: Tuple[int, ...],
                                    orig_shape2: Tuple[int, ...]) -> Tuple[torch.Tensor, torch.Tensor]:
        """受控的SVD分解，确保输出尺寸与原始张量兼容"""
        chi_L, d1, d2, chi_R = theta.shape
        matrix = theta.reshape(chi_L * d1, d2 * chi_R)

        try:
            U, S, Vh = torch.linalg.svd(matrix, full_matrices=False)

            # 确定保留的奇异值数量
            original_bond = orig_shape1[2]  # 原始A1的右bond维度
            k = min(len(S), original_bond, self.config.max_bond_dim)
            k = max(k, 1)  # 至少保留一个

            # 截断奇异值
            U_trunc = U[:, :k]
            S_trunc = S[:k]
            Vh_trunc = Vh[:k, :]

            # 分配奇异值以保持数值稳定性
            sqrt_S = torch.sqrt(S_trunc + 1e-12)

            # 重塑回张量形式
            A1_new = (U_trunc * sqrt_S.unsqueeze(0)).reshape(chi_L, d1, k)
            A2_new = (sqrt_S.unsqueeze(1) * Vh_trunc).reshape(k, d2, chi_R)

            # 调整维度以匹配原始形状
            A1_new = self._adjust_tensor_shape(A1_new, orig_shape1)
            A2_new = self._adjust_tensor_shape(A2_new, orig_shape2)

            return A1_new, A2_new

        except Exception as e:
            print(f"SVD failed: {e}")
            # 返回原始形状的随机小扰动
            device = theta.device
            dtype = theta.dtype
            A1_fallback = torch.randn(orig_shape1, device=device, dtype=dtype) * 0.01
            A2_fallback = torch.randn(orig_shape2, device=device, dtype=dtype) * 0.01
            return A1_fallback, A2_fallback

    def _adjust_tensor_shape(self, tensor: torch.Tensor, target_shape: Tuple[int, ...]) -> torch.Tensor:
        """调整张量形状以匹配目标形状"""
        current_shape = tensor.shape

        # 如果形状已经匹配，直接返回
        if current_shape == target_shape:
            return tensor

        # 创建目标形状的零张量
        adjusted = torch.zeros(target_shape, device=tensor.device, dtype=tensor.dtype)

        # 计算可以复制的最小维度
        min_dims = [min(c, t) for c, t in zip(current_shape, target_shape)]

        # 复制数据到新张量中
        if len(min_dims) == 3:
            adjusted[:min_dims[0], :min_dims[1], :min_dims[2]] = \
                tensor[:min_dims[0], :min_dims[1], :min_dims[2]]

        return adjusted

    def _evaluate(self, data_loader: DataLoader, loss_fn: nn.Module) -> Dict:
        """评估模型性能"""
        self.mps_model.eval()
        total_loss = 0.0
        total_correct = 0
        total_samples = 0

        with torch.no_grad():
            for x_batch, y_batch in data_loader:
                device = next(self.mps_model.parameters()).device
                x_batch, y_batch = x_batch.to(device), y_batch.to(device)

                logits = self.mps_model(x_batch)
                loss = loss_fn(logits.squeeze(), y_batch.float())

                preds = (torch.sigmoid(logits.squeeze()) > 0.5).float()

                total_loss += loss.item() * x_batch.size(0)
                total_correct += (preds == y_batch).sum().item()
                total_samples += x_batch.size(0)

        self.mps_model.train()
        return {
            'loss': total_loss / total_samples,
            'accuracy': total_correct / total_samples
        }

    def _compute_average_entanglement(self) -> float:
        """计算平均纠缠熵"""
        entropies = []
        for i in range(self.mps_model.num_sites - 1):
            try:
                entropy = self.mps_model.get_entanglement_entropy(i)
                if not np.isnan(entropy) and not np.isinf(entropy):
                    entropies.append(entropy)
            except Exception as e:
                print(f"Failed to compute entanglement at site {i}: {e}")
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

    def _convergence_check(self) -> bool:
        """收敛检查"""
        if len(self.loss_history) < 2:
            return False

        recent_losses = self.loss_history[-3:]
        if len(recent_losses) >= 2:
            loss_change = abs(recent_losses[-1] - recent_losses[-2])
            return loss_change < self.config.convergence_threshold
        return False


def create_complex_dataset(num_samples: int, num_sites: int, phys_dim: int,
                           device: str = 'cpu') -> Tuple[torch.Tensor, torch.Tensor]:
    """创建更复杂的合成数据集"""
    X = torch.randint(0, phys_dim, (num_samples, num_sites), device=device)
    X_encoded = F.one_hot(X, num_classes=phys_dim).float()

    # 更复杂的标签生成：基于局部模式和长程相关性
    y = torch.zeros(num_samples, device=device)

    for i in range(num_samples):
        sample = X[i]
        # 局部模式：连续的相同值
        local_pattern = sum([1 for j in range(num_sites - 1)
                             if sample[j] == sample[j + 1]])

        # 长程相关性：首尾相关
        long_range = (sample[0] == sample[-1])

        # 奇偶性
        parity = (sample.sum() % 2) == 1

        # 复合规则 - 修复布尔类型转换问题
        result = (local_pattern >= 2) or (long_range and parity)
        y[i] = float(result)  # 将Python bool转换为float

    return X_encoded, y


def plot_training_history(history: Dict):
    """绘制训练历史"""
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    # Loss
    axes[0, 0].plot(history['train_loss'], label='Train Loss')
    axes[0, 0].plot(history['val_loss'], label='Val Loss')
    axes[0, 0].set_title('Loss vs Sweeps')
    axes[0, 0].set_xlabel('Sweep')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True)

    # Accuracy
    axes[0, 1].plot(history['train_acc'], label='Train Acc')
    axes[0, 1].plot(history['val_acc'], label='Val Acc')
    axes[0, 1].set_title('Accuracy vs Sweeps')
    axes[0, 1].set_xlabel('Sweep')
    axes[0, 1].set_ylabel('Accuracy')
    axes[0, 1].legend()
    axes[0, 1].grid(True)

    # Entanglement
    axes[1, 0].plot(history['entanglement'])
    axes[1, 0].set_title('Average Entanglement Entropy')
    axes[1, 0].set_xlabel('Sweep')
    axes[1, 0].set_ylabel('Entropy')
    axes[1, 0].grid(True)

    # Loss difference
    if len(history['train_loss']) > 1:
        loss_diff = np.diff(history['val_loss'])
        axes[1, 1].plot(loss_diff)
        axes[1, 1].set_title('Validation Loss Change')
        axes[1, 1].set_xlabel('Sweep')
        axes[1, 1].set_ylabel('Loss Difference')
        axes[1, 1].grid(True)

    plt.tight_layout()
    plt.show()


def main():
    """主函数示例"""
    # 配置
    config = TrainingConfig(
        num_sites=12,
        phys_dim=2,
        bond_dim=8,
        max_bond_dim=16,
        learning_rate=0.01,
        num_sweeps=20,
        num_local_epochs=10,
        batch_size=64,
        early_stopping_patience=5,
        device='cuda' if torch.cuda.is_available() else 'cpu'
    )

    device = torch.device(config.device)
    print(f"Using device: {device}")

    # 创建数据
    X, y = create_complex_dataset(2000, config.num_sites, config.phys_dim, device)
    dataset = TensorDataset(X, y)

    # 划分训练集和验证集
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False)

    # 创建模型
    model = ImprovedMPS(config.num_sites, config.phys_dim, config.bond_dim).to(device)

    # 创建优化器
    optimizer = AdaptiveDMRGOptimizer(model, config)

    # 使用更适合分类的损失函数
    criterion = nn.BCEWithLogitsLoss()

    print("Starting training...")
    history = optimizer.train(train_loader, val_loader, criterion)

    # 最终测试
    final_metrics = optimizer._evaluate(val_loader, criterion)
    print(f"\nFinal Results:")
    print(f"Validation Loss: {final_metrics['loss']:.4f}")
    print(f"Validation Accuracy: {final_metrics['accuracy']:.4f}")

    # 绘制训练历史
    plot_training_history(history)

    # 打印纠缠熵分析
    print("\nEntanglement Analysis:")
    for i in range(model.num_sites - 1):
        try:
            entropy = model.get_entanglement_entropy(i)
            print(f"Site {i}-{i + 1}: {entropy:.4f}")
        except Exception as e:
            print(f"Site {i}-{i + 1}: calculation failed ({e})")


if __name__ == "__main__":
    main()