# https://grok.com/chat/1216570a-4c06-4a03-8ba7-53a26889c56e
import torch
from torch import nn, optim
from torch.utils.data import DataLoader, TensorDataset
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
from typing import Optional, Dict, Tuple
import logging

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- 1. MPS 模型定义 ---
class MPS(nn.Module):
    def __init__(self, num_sites: int, phys_dim: int, bond_dim: int, num_classes: int = 1):
        super().__init__()
        self.num_sites = num_sites
        self.phys_dim = phys_dim
        self.bond_dim = bond_dim
        self.num_classes = num_classes
        self.tensor_sites = nn.ParameterList()
        for i in range(self.num_sites):
            if i == 0:
                shape = (1, phys_dim, bond_dim)
            elif i == self.num_sites - 1:
                shape = (bond_dim, phys_dim, num_classes)
            else:
                shape = (bond_dim, phys_dim, bond_dim)
            # 正交初始化
            tensor_i = torch.randn(shape)
            if i < self.num_sites - 1:  # 最后一维不需正交化
                tensor_i = tensor_i.reshape(-1, shape[-1])
                tensor_i, _ = torch.linalg.qr(tensor_i)
                tensor_i = tensor_i.reshape(shape)
            self.tensor_sites.append(nn.Parameter(tensor_i))

    def forward(self, x_batch: torch.Tensor,
                tensors_override: Optional[Dict[int, torch.Tensor]] = None) -> torch.Tensor:
        """
        前向传播，计算MPS的输出
        :param x_batch: 输入张量，形状为 (batch_size, num_sites, phys_dim)
        :param tensors_override: 可选的字典，用于覆盖特定点的张量
        :return: logits: 形状为 (batch_size, num_classes)
        """
        batch_size = x_batch.shape[0]
        device = x_batch.device
        left_vector = torch.ones(batch_size, 1, device=device)
        for i in range(self.num_sites):
            A_i = (tensors_override or {}).get(i, self.tensor_sites[i])
            left_vector = torch.einsum("bl, lpr -> bpr", left_vector, A_i)
            left_vector = torch.einsum("bpr, bp -> br", left_vector, x_batch[:, i, :])
        return left_vector.squeeze(-1) if self.num_classes == 1 else left_vector

    def normalize(self):
        """对MPS张量进行左规范"""
        with torch.no_grad():
            for i in range(self.num_sites - 1):
                tensor = self.tensor_sites[i]
                reshaped = tensor.reshape(-1, tensor.shape[-1])
                q, r = torch.linalg.qr(reshaped)
                self.tensor_sites[i] = nn.Parameter(q.reshape(tensor.shape))
                next_tensor = self.tensor_sites[i + 1]
                self.tensor_sites[i + 1] = nn.Parameter(torch.einsum("ij, jkl -> ikl", r, next_tensor))

# --- 2. DMRG 风格的优化器 ---
class DMRGOptimizer:
    def __init__(self, mps_model: MPS, local_optimizer_class, local_optimizer_params: dict,
                 max_bond_dim_truncate: int, density_matrix_truncation: bool = True):
        self.mps_model = mps_model
        self.local_optimizer_class = local_optimizer_class
        self.local_optimizer_params = local_optimizer_params
        self.max_bond_dim_truncate = max_bond_dim_truncate
        self.density_matrix_truncation = density_matrix_truncation
        self.local_optimizer = None  # 延迟初始化优化器

    def sweep(self, train_loader: DataLoader, valid_loader: Optional[DataLoader],
              loss_fn, num_local_epochs: int, direction: str = 'left_to_right') -> Tuple[float, float]:
        """
        执行一次DMRG优化的sweep
        :return: 平均训练损失, 验证准确率（若提供valid_loader）
        """
        train_loss = 0.0
        num_batches = 0
        self.mps_model.train()
        for x_batch, y_batch in tqdm(train_loader, desc=f'DMRG Sweep [{direction}]'):
            x_batch, y_batch = x_batch.to(self.mps_model.tensor_sites[0].device), y_batch.to(self.mps_model.tensor_sites[0].device)
            try:
                if direction == 'left_to_right':
                    for i in range(self.mps_model.num_sites - 1):
                        self._optimize_adjacent_sites(i, i + 1, x_batch, y_batch, loss_fn, num_local_epochs)
                else:
                    for i in reversed(range(1, self.mps_model.num_sites)):
                        self._optimize_adjacent_sites(i - 1, i, x_batch, y_batch, loss_fn, num_local_epochs)
                with torch.no_grad():
                    y_pred = self.mps_model(x_batch)
                    train_loss += loss_fn(y_pred, y_batch).item()
                    num_batches += 1
            except RuntimeError as e:
                logger.error(f"Error in sweep: {e}")
                continue
        train_loss = train_loss / num_batches if num_batches > 0 else float('inf')

        valid_acc = 0.0
        if valid_loader:
            valid_acc = self.evaluate(valid_loader, loss_fn)
        logger.info(f"Sweep [{direction}]: Train Loss = {train_loss:.4f}, Valid Acc = {valid_acc:.4f}")
        return train_loss, valid_acc

    def _optimize_adjacent_sites(self, site1_idx: int, site2_idx: int,
                                 x_batch: torch.Tensor, y_batch: torch.Tensor,
                                 loss_fn, num_local_epochs: int):
        """优化相邻的两个MPS张量"""
        try:
            A1 = self.mps_model.tensor_sites[site1_idx].detach()
            A2 = self.mps_model.tensor_sites[site2_idx].detach()
            theta_init = torch.tensordot(A1, A2, dims=([2], [0]))
            theta_param = nn.Parameter(theta_init.clone())

            if self.local_optimizer is None:
                self.local_optimizer = self.local_optimizer_class([theta_param], **self.local_optimizer_params)
            else:
                self.local_optimizer.param_groups[0]['params'] = [theta_param]

            for _ in range(num_local_epochs):
                self.local_optimizer.zero_grad()
                A1_new, A2_new = self._decompose_theta(theta_param, self.max_bond_dim_truncate)
                overrides = {site1_idx: A1_new, site2_idx: A2_new}
                y_pred = self.mps_model(x_batch, tensors_override=overrides)
                loss = loss_fn(y_pred, y_batch)
                loss.backward()
                self.local_optimizer.step()

            with torch.no_grad():
                A1_final, A2_final = self._decompose_theta(theta_param, self.max_bond_dim_truncate)
                self.mps_model.tensor_sites[site1_idx] = nn.Parameter(A1_final)
                self.mps_model.tensor_sites[site2_idx] = nn.Parameter(A2_final)
        except RuntimeError as e:
            logger.error(f"Error optimizing sites {site1_idx}, {site2_idx}: {e}")

    def _decompose_theta(self, theta: torch.Tensor, truncated_dim: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """从theta分解A1_new和A2_new，支持密度矩阵截断"""
        chi_L, d1, d2, chi_R = theta.shape
        unfold_theta = theta.reshape(chi_L * d1, d2 * chi_R)
        try:
            if self.density_matrix_truncation:
                rho = torch.matmul(unfold_theta, unfold_theta.t())
                eigvals, eigvecs = torch.linalg.eigh(rho)
                k = min(len(eigvals), truncated_dim)
                idx = torch.argsort(eigvals, descending=True)[:k]
                U = eigvecs[:, idx]
                unfold_theta = torch.matmul(U, torch.matmul(U.t(), unfold_theta))
            U, S, Vh = torch.linalg.svd(unfold_theta, full_matrices=False)
            k = min(S.shape[0], truncated_dim)
            A1_new = U[:, :k].reshape(chi_L, d1, k)
            A2_new = (torch.diag(S[:k]) @ Vh[:k, :]).reshape(k, d2, chi_R)
            return A1_new, A2_new
        except RuntimeError as e:
            logger.error(f"SVD decomposition failed: {e}")
            return theta.reshape(chi_L, d1, d2, chi_R).split([d1], dim=1)

    def evaluate(self, data_loader: DataLoader, loss_fn) -> float:
        """评估模型准确率"""
        self.mps_model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for x_batch, y_batch in data_loader:
                x_batch, y_batch = x_batch.to(self.mps_model.tensor_sites[0].device), y_batch.to(self.mps_model.tensor_sites[0].device)
                y_pred = self.mps_model(x_batch)
                if self.mps_model.num_classes == 1:
                    preds = (torch.sigmoid(y_pred) > 0.5).float()
                else:
                    preds = y_pred.argmax(dim=-1)
                correct += (preds == y_batch).float().sum().item()
                total += y_batch.size(0)
        return correct / total if total > 0 else 0.0

# --- 3. 数据生成与训练 ---
def synthetic_data(num_samples: int, num_sites: int, phys_dim: int, device: str = 'cpu') -> Tuple[torch.Tensor, torch.Tensor]:
    """生成合成数据"""
    X = torch.randint(0, phys_dim, (num_samples, num_sites), device=device)
    X_encoded = F.one_hot(X, num_classes=phys_dim).float()
    y = (X.sum(dim=1) % 2).float()
    return X_encoded, y

def train_mps(num_sweeps: int, num_sites: int, phys_dim: int, bond_dim: int,
              batch_size: int, num_samples: int, device: str = 'cpu') -> dict:
    """训练MPS模型并返回最佳模型状态"""
    # 数据准备
    X, y = synthetic_data(num_samples, num_sites, phys_dim, device)
    train_size = int(0.8 * num_samples)
    train_dataset = TensorDataset(X[:train_size], y[:train_size])
    valid_dataset = TensorDataset(X[train_size:], y[train_size:])
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size)

    # 模型与优化器
    model = MPS(num_sites, phys_dim, bond_dim).to(device)
    model.normalize()
    optimizer = DMRGOptimizer(model, optim.Adam, {"lr": 0.05}, bond_dim, density_matrix_truncation=True)
    criterion = nn.BCEWithLogitsLoss()

    # 训练循环
    best_acc = 0.0
    best_model_state = None
    for sweep in range(num_sweeps):
        direction = 'left_to_right' if sweep % 2 == 0 else 'right_to_left'
        try:
            train_loss, valid_acc = optimizer.sweep(train_loader, valid_loader, criterion, num_local_epochs=3, direction=direction)
            if valid_acc > best_acc:
                best_acc = valid_acc
                best_model_state = model.state_dict()
                torch.save(best_model_state, "best_mps_model.pth")
                logger.info(f"Saved best model at sweep {sweep+1} with valid acc {valid_acc:.4f}")
            if valid_acc > 0.95:  # 早停
                logger.info(f"Early stopping at sweep {sweep+1} with valid acc {valid_acc:.4f}")
                break
        except Exception as e:
            logger.error(f"Training failed at sweep {sweep+1}: {e}")
            break
    logger.info(f"Final Valid Accuracy: {best_acc:.4f}")
    return {"model": model, "best_acc": best_acc, "best_model_state": best_model_state}

if __name__ == "__main__":
    result = train_mps(
        num_sweeps=10,
        num_sites=6,
        phys_dim=2,
        bond_dim=4,
        batch_size=16,
        num_samples=1000,
        device='cuda' if torch.cuda.is_available() else 'cpu'
    )