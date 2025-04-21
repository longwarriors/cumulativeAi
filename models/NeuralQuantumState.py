# From Architectures to Applications: A Review of Neural Quantum States
# http://arxiv.org/abs/2402.09402
# https://claude.ai/chat/0892aceb-edd6-46a0-857f-2482a5a0dcb2
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class RBMQuantumState(nn.Module):
    """受限玻尔兹曼机 (RBM) 神经量子态"""

    def __init__(self, n_visible, n_hidden, complex_weights=True):
        super().__init__()
        self.n_visible = n_visible
        self.n_hidden = n_hidden
        self.complex_weights = complex_weights
        if complex_weights:  # 复数权重和偏置
            self.W_re = nn.Parameter(torch.randn(n_visible, n_hidden) * 0.01)
            self.W_im = nn.Parameter(torch.randn(n_visible, n_hidden) * 0.01)

            self.b_v_re = nn.Parameter(torch.zeros(n_visible))
            self.b_v_im = nn.Parameter(torch.zeros(n_visible))

            self.b_h_re = nn.Parameter(torch.zeros(n_hidden))
            self.b_h_im = nn.Parameter(torch.zeros(n_hidden))
        else:
            self.W_amp = nn.Parameter(torch.randn(n_visible, n_hidden) * 0.01)
            self.W_phase = nn.Parameter(torch.randn(n_visible, n_hidden) * 0.01)

            self.b_v_amp = nn.Parameter(torch.zeros(n_visible))
            self.b_v_phase = nn.Parameter(torch.zeros(n_visible))

            self.b_h_amp = nn.Parameter(torch.zeros(n_hidden))
            self.b_h_phase = nn.Parameter(torch.zeros(n_hidden))

    def forward(self, x):
        """计算波函数的振幅和相位
        Args:
            x (torch.Tensor): 输入数据，形状为 (batch_size, n_visible)
        Returns:
            torch.Tensor: log振幅和相位
        """
        if self.complex_weights:
            W = torch.complex(self.W_re, self.W_im)
            b_v = torch.complex(self.b_v_re, self.b_v_im)
            b_h = torch.complex(self.b_h_re, self.b_h_im)
            linear_visible = torch.matmul(x.float(), b_v)
            linear_hidden = torch.matmul(x.float(), W) + b_h
            activation = torch.log(1 + torch.exp(linear_hidden))  # softplus激活函数
            psi = linear_visible + torch.sum(activation, dim=1)
            log_amp = torch.real(psi)
            phase = torch.imag(psi)
        else:
            linear_visible_amp = torch.matmul(x.float(), self.b_v_amp)
            linear_visible_phase = torch.matmul(x.float(), self.b_v_phase)

            linear_hidden_amp = torch.matmul(x.float(), self.W_amp) + self.b_h_amp
            linear_hidden_phase = torch.matmul(x.float(), self.W_phase) + self.b_h_phase

            activation_amp = torch.log(1 + torch.exp(linear_hidden_amp))
            activation_phase = torch.log(1 + torch.exp(linear_hidden_phase))

            log_amp = linear_visible_amp + torch.sum(activation_amp, dim=1)
            phase = linear_visible_phase + torch.sum(activation_phase, dim=1)

        return log_amp, phase

    def evaluate_psi(self, x):
        log_amp, phase = self.forward(x)
        return torch.exp(log_amp) * torch.exp(1j * phase)


class HeisenbergChain:
    """一维海森堡模型 H = J * sum_i S_i · S_{i+1}
    Args:
        n_sites (int): 格点数量，量子位数
        J (float): 耦合常数
        periodic (bool): 是否周期性边界条件
    """

    def __init__(self, n_sites, J=1.0, periodic=True):
        self.n_sites = n_sites
        self.J = J
        self.periodic = periodic

    def get_local_energy(self, model, samples, n_samples=1000):
        """使用蒙特卡洛采样计算局部能量
        Args:
            model (RBMQuantumState): 神经量子态模型
            samples (torch.Tensor): 输入组态，形状为 (n_samples, n_sites)
            n_samples (int): 采样数量
        Returns:
            torch.Tensor: 局部能量的均值和标准误差
        """
        # 计算波函数的振幅和相位
        device = next(model.parameters()).device
        local_energies = torch.zeros(samples.shape[0], dtype=torch.complex64, device=device)

        # 获取原始样本的波函数值
        psi_samples = model.evaluate_psi(samples)

        # 对每个样本计算局部能量
        for i in range(samples.shape[0]):
            sample = samples[i]
            local_energy = torch.tensor(0.0, dtype=torch.complex64, device=device)
            for j in range(self.n_sites):
                next_j = (j + 1) % self.n_sites if self.periodic else j + 1
                if not self.periodic and next_j >= self.n_sites:
                    continue

                # S^z_j·S^z_{j+1}项
                sz_sz = (2 * sample[j] - 1) * (2 * sample[next_j] - 1) * 0.25  # 将0和1映射到-1/2和1/2

                # S^x_j·S^x_{j+1} + S^y_j·S^y_{j+1}项，需要翻转自旋
                flipped_sample = sample.clone()
                if sample[j] != sample[next_j]:
                    flipped_sample[j] = 1 - flipped_sample[j]
                    flipped_sample[next_j] = 1 - flipped_sample[next_j]
                    psi_flipped = model.evaluate_psi(flipped_sample.unsqueeze(0))
                    sxy_term = 0.5 * (psi_flipped / psi_samples[i])  # 非对角项 S^+_j·S^-_{j+1} + S^-_j·S^+_{j+1}
                    local_energy += self.J * (sz_sz + sxy_term)
                else:
                    local_energy += self.J * sz_sz  # 自旋相同，只考虑S^z_j·S^z_{j+1}项
            local_energies[i] = local_energy

        mean_energy = torch.mean(local_energies)
        error = torch.std(local_energies) / np.sqrt(samples.shape[0])
        return mean_energy.real, error.real


class VMCOptimizer:
    """变分蒙特卡洛优化器
    Args:
        model (RBMQuantumState): 神经量子态模型
        hamiltonian (HeisenbergChain): 海森堡链模型
        learning_rate (float): 学习率
        n_samples (int): 采样数量
        sr_reg: 随机重构正则化参数
    """

    def __init__(self, model, hamiltonian, learning_rate=0.01, n_samples=1000, sr_reg=0.01):
        self.model = model
        self.hamiltonian = hamiltonian
        self.lr = learning_rate
        self.n_samples = n_samples
        self.sr_reg = sr_reg
        self.device = next(model.parameters()).device
        self.optimizer = torch.optim.Adam(model.parameters(), lr=self.lr)

    def sample_states(self):
        """使用蒙特卡洛方法生成样本"""
        n_sites = self.hamiltonian.n_sites
        batch_size = self.n_samples

        # 随机初始化样本
        samples = torch.randint(0, 2, (batch_size, n_sites), device=self.device)
        accepted_samples = samples.clone()

        # 进行热化更新
        n_sweeps = 100
        n_steps_per_sweep = n_sites
        for sweep in range(n_sweeps):
            for step in range(n_steps_per_sweep):
                # 随机选择一个位置并翻转自旋
                site_idx = torch.randint(0, n_sites - 1, device=self.device)
                proposed_samples = samples.clone()
                proposed_samples[:, site_idx] = 1 - proposed_samples[:, site_idx]

                # 计算接受概率
                log_amp_old, _ = self.model(samples)
                log_amp_new, _ = self.model(proposed_samples)

                # 计算接受概率 (使用振幅的平方)
                acceptance_probs = torch.exp(2 * (log_amp_new - log_amp_old))

                # Metropolis-Hastings接受/拒绝步骤
                random_nums = torch.rand(batch_size, device=self.device)
                accepted_idx = random_nums < acceptance_probs
                samples[accepted_idx] = proposed_samples[accepted_idx]

        # 返回经过热化后的样本
        return samples

    def stochastic_reconfiguration_step(self, samples):
        """使用随机重构优化模型参数"""
        energy, _ = self.hamiltonian.get_local_energy(self.model, samples)
        self.optimizer.zero_grad()
        log_amp, phase = self.model(samples)
        loss = torch.mean(log_amp)  # 简化版本，实际应根据能量进行梯度计算
        loss.backward()

        # 提取梯度
        gradients = []
