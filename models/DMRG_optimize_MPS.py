import torch
from torch import nn, optim
from torch.utils.data import DataLoader, TensorDataset
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
from typing import Optional, Dict


# https://github.com/copilot/c/80cde471-49fd-4281-9012-b35ec59e7f57
# https://chatgpt.com/c/6847a9ca-f5a4-800f-9924-2d1bc76b2808

# --- 1. MPS 模型定义 ---
class MPS(nn.Module):
    def __init__(self, num_sites: int, phys_dim: int, bond_dim: int):
        super().__init__()
        self.num_sites = num_sites
        self.phys_dim = phys_dim
        self.bond_dim = bond_dim
        self.tensor_sites = nn.ParameterList()
        for i in range(self.num_sites):
            if i == 0:
                shape = (1, phys_dim, bond_dim)
            elif i == self.num_sites - 1:
                shape = (bond_dim, phys_dim, 1)
            else:
                shape = (bond_dim, phys_dim, bond_dim)
            # 初始权重过大或过小可能导致梯度消失或爆炸
            tensor_i = nn.init.xavier_uniform_(torch.empty(shape))
            self.tensor_sites.append(nn.Parameter(tensor_i))

    def forward(self, x_batch: torch.Tensor,
                tensors_override: Optional[Dict[int, torch.Tensor]] = None) -> torch.Tensor:
        """
        前向传播，计算MPS的输出
        :param x_batch: 输入张量，形状为 (batch_size, num_sites, phys_dim)
        :param tensors_override: 可选的字典，用于覆盖特定点的张量
        :return: logits: 形状为 (batch_size, 1)
        """
        batch_size = x_batch.shape[0]
        # 初始化左边界向量(batch_size, left_bond_dim_of_first_tensor=1)
        left_vector = torch.ones(batch_size, self.tensor_sites[0].shape[0], device=x_batch.device)
        for i in range(self.num_sites):
            # 支持动态替换 tensor_sites（用于 DMRG 局部优化）
            A_i = (tensors_override or {}).get(i, self.tensor_sites[i])  # 形状为 (chi_left, phys_dim, chi_right)
            x_i = x_batch[:, i, :]  # 形状为 (batch_size, phys_dim)

            # step 1: left_vector 和 A_i 的左键收缩
            # left_vector (batch, chi_L) @ A_i (chi_L, phys, chi_R) -> (batch, phys, chi_R)
            left_vector = torch.einsum("bl, lpr -> bpr", left_vector, A_i)

            # step 2: left_vector 和 x_site 的物理维度收缩
            # left_vector (batch, phys, chi_R) @ x_site (batch, phys) -> (batch, chi_R)
            left_vector = torch.einsum("bpr, bq -> br", left_vector, x_i)

        # 最终 left_vector 形状为 (batch_size, right_bond_dim_of_last_tensor=1)
        return left_vector


# --- 2. DMRG 风格的优化器 ---
class DMRGOptimizer:
    def __init__(self, mps_model: MPS,
                 local_optimizer_class,
                 local_optimizer_params: dict,
                 max_bond_dim_truncate):
        self.mps_model = mps_model  # 要优化的MPS模型
        self.local_optimizer_class = local_optimizer_class  # 用于局部优化的PyTorch优化器类 (例如 optim.Adam)
        self.local_optimizer_params = local_optimizer_params  # 局部优化器的参数 (例如 {'lr': LEARNING_RATE})
        self.max_bond_dim_truncate = max_bond_dim_truncate  # SVD截断时使用的最大键长

    def sweep(self, train_loader, loss_fn, num_local_epochs, direction='left_to_right'):
        """
        执行一次DMRG优化的sweep，遍历所有张量对进行局部优化
        :param train_loader: 训练数据加载器
        :param loss_fn: 损失函数
        :param num_local_epochs: 每对张量的局部优化迭代次数
        :param direction: 优化方向，'left_to_right' 或 'right_to_left'
        """
        for x_batch, y_batch in tqdm(train_loader, desc=f'DMRG Sweep [{direction}]'):
            device = next(self.mps_model.parameters()).device
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            if direction == 'left_to_right':
                for i in range(self.mps_model.num_sites - 1):
                    self._optimize_adjacent_sites(i, i + 1, x_batch, y_batch, loss_fn, num_local_epochs)
            else:
                for i in reversed(range(1, self.mps_model.num_sites)):
                    self._optimize_adjacent_sites(i - 1, i, x_batch, y_batch, loss_fn, num_local_epochs)

    def _optimize_adjacent_sites(self,
                                 site1_idx: int,
                                 site2_idx: int,
                                 x_batch: torch.Tensor,
                                 y_batch: torch.Tensor,
                                 loss_fn,
                                 num_local_epochs: int = 1):
        """优化相邻的两个MPS张量 (A_i, A_{i+1})
        A1: (chi_L, d, chi_mid)
        A2: (chi_mid, d, chi_R)
        Theta: (chi_L, d, d, chi_R) via tensordot on chi_mid
        """
        # 1. 缩并相邻的两个张量形成一个更大的张量 Theta
        A1 = self.mps_model.tensor_sites[site1_idx].detach()
        A2 = self.mps_model.tensor_sites[site2_idx].detach()

        with torch.no_grad():
            theta_init = torch.tensordot(A1, A2, dims=([2], [0]))

        # 将theta设为可学习参数，用于局部优化
        theta_param = nn.Parameter(theta_init.detach().clone())
        local_optimizer = self.local_optimizer_class([theta_param], **self.local_optimizer_params)

        # 2. 局部优化循环
        for epoch in range(num_local_epochs):
            local_optimizer.zero_grad()
            # 从当前的theta_param分解出A1_current, A2_current用于前向传播。
            # 这个分解步骤必须是计算图的一部分，这样theta_param才能接收到梯度。
            A1_new, A2_new = self._decompose_theta(theta_param, self.max_bond_dim_truncate)
            overrides = {site1_idx: A1_new, site2_idx: A2_new}
            y_pred = self.mps_model(x_batch, tensors_override=overrides)
            loss = loss_fn(y_pred, y_batch.unsqueeze(1).float())
            loss.backward()
            local_optimizer.step()

        with torch.no_grad():
            # 临时替换模型中的张量，避免该操作被外层梯度追踪
            A1_final, A2_final = self._decompose_theta(theta_param, self.max_bond_dim_truncate)
            self.mps_model.tensor_sites[site1_idx] = nn.Parameter(A1_final)
            self.mps_model.tensor_sites[site2_idx] = nn.Parameter(A2_final)

    @staticmethod
    def _decompose_theta(theta, truncated_dim):
        """从两点张量theta通过SVD分解和截断变回A1_new和A2_new（用于前向传播）
        Theta (chi_L, d1, d2, chi_R) -> M (chi_L*d1, d2*chi_R)
        :param theta: 当前的theta张量，形状为 (chi_L, d1, d2, chi_R)
        :param truncated_dim: 截断后的最大键长
        :return: A1_new, A2_new
        """
        chi_L, d1, d2, chi_R = theta.shape
        unfold_theta = theta.reshape(chi_L * d1, d2 * chi_R)
        U, S, Vh = torch.linalg.svd(unfold_theta, full_matrices=False)  # 仅返回与非零奇异值对应的列向量

        # 重构新的 A1_new 和 A2_new - 将奇异值S吸收到 A2 (或者平均分配，或只给一个张量)
        k = min(S.shape[0], truncated_dim)  # 新键长不能超过SVD结果的秩和设定的最大键长
        A1_new = U[:, :k].reshape(chi_L, d1, k)  # (chi_L, d1, k)
        A2_new = (torch.diag(S[:k]) @ Vh[:k, :]).reshape(k, d2, chi_R)  # (k, d2, chi_R)
        return A1_new, A2_new


# -------------------- 示例使用 --------------------
def synthetic_data(num_samples: int, num_sites: int, phys_dim: int, device: str = 'cpu'):
    """生成合成数据用于测试"""
    X = torch.randint(0, phys_dim, (num_samples, num_sites), device=device)
    if phys_dim > 1:
        X_encoded = F.one_hot(X, num_classes=phys_dim).float()
    else:
        X_encoded = X.unsqueeze(-1).float()
    y = (X.sum(dim=1) % 2).float()
    return X_encoded, y


def example():
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    num_sites, phys_dim, bond_dim = 6, 2, 4
    model = MPS(num_sites, phys_dim, bond_dim).to(DEVICE)
    X, y = synthetic_data(100, num_sites, phys_dim, DEVICE)
    dataset = TensorDataset(X, y)
    train_loader = DataLoader(dataset, batch_size=16, shuffle=True)
    optimizer = DMRGOptimizer(model, optim.Adam, {"lr": 0.05}, bond_dim)
    criterion = nn.MSELoss()

    for sweep in range(5):
        direction = 'left_to_right' if sweep % 2 == 0 else 'right_to_left'
        optimizer.sweep(train_loader, criterion, num_local_epochs=3, direction=direction)

    with torch.no_grad():
        logits = model(X)
        preds = (logits.squeeze() > 0.5).float()
        acc = (preds == y).float().mean()
        print(f"Accuracy: {acc.item():.4f}")


if __name__ == "__main__":
    example()
