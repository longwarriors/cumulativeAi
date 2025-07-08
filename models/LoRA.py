import torch, math, os
import torch.nn as nn
from utils import train_epoch, validate_epoch


class LoRALayer(nn.Module):
    def __init__(self, in_features, out_features, rank=1, alpha=1.0):
        super().__init__()
        self.rank = rank
        self.alpha = alpha
        self.in_features = in_features
        self.out_features = out_features

        # 低秩矩阵初始化
        std_deviation = math.sqrt(1 / self.rank)
        self.A = nn.Parameter(torch.randn(in_features, rank) * std_deviation)
        self.B = nn.Parameter(torch.zeros(rank, out_features))

    def forward(self, x):
        return x @ self.A @ self.B * self.alpha


class LinearWithLoRA(nn.Module):
    def __init__(self, linear_layer: nn.Linear, lora_layer: LoRALayer):
        super().__init__()
        self.linear = linear_layer
        # 冻结原线性层的参数
        for param in self.linear.parameters():
            param.requires_grad = False
        self.lora = lora_layer
        self._merged = False  # 标志位，表示是否已合并LoRA层的权重

    def forward(self, x):
        # 如果权重已合并，只使用修改后的线性层进行前向传播
        if self._merged:
            return self.linear(x)
        # 如果未合并，返回线性层和LoRA层的输出之和
        else:
            return self.linear(x) + self.lora(x)

    def merge_weights(self):
        """合并LoRA层的权重到原线性层"""
        if self._merged:
            return  # 不执行任何操作
        with torch.no_grad():
            delta_W = self.lora.alpha * (self.lora.A @ self.lora.B).t()
            self.linear.weight += delta_W
            self._merged = True  # 防止重复合并的标志位

    def unmerge_weights(self):
        """从原线性层中减去LoRA层的权重，恢复原状"""
        if not self._merged:
            return  # 不执行任何操作
        with torch.no_grad():
            delta_W = self.alpha * (self.lora.A @ self.lora.B).t()
            self.linear.weight -= delta_W
            self._merged = False  # 设置标志位，表示未合并


def add_lora_to_linear(net: nn.Module, rank=4, alpha=1.0) -> None:
    """递归替换模型中的线性层为带LoRA的线性层"""
    for name, module in net.named_children():
        if isinstance(module, nn.Linear):
            print(f"替换层: {name} -> 带LoRA的线性层")
            lora_layer = LoRALayer(module.in_features, module.out_features, rank, alpha)
            setattr(net, name, LinearWithLoRA(module, lora_layer))
        elif len(list(module.children())) > 0:
            # 如果模块还有子模块，递归替换
            add_lora_to_linear(module, rank, alpha)


def merge_all_lora_weights(lora_model):
    """递归合并所有LoRA层的权重"""
    for module in lora_model.modules():
        if isinstance(module, LinearWithLoRA):
            module.merge_weights()


def unmerge_all_lora_weights(lora_model):
    """递归取消合并所有LoRA层的权重"""
    for module in lora_model.modules():
        if isinstance(module, LinearWithLoRA):
            module.unmerge_weights()


def get_lora_state_dict(net):
    """获取模型中的LoRA层的状态字典"""
    return {k: v for k, v in net.named_parameters() if 'lora' in k and v.requires_grad}


def get_lora_state_dict_explicitly(net):
    """显式地遍历模块"""
    lora_state_dict = {}
    for name, module in net.named_modules():
        # 我们要找的是 LoRALayer, 而不是 LinearWithLoRA
        if isinstance(module, LoRALayer):
            # 'name' 已经是正确的层级名称，例如 'fc1.lora'
            lora_state_dict[name + ".A"] = module.A
            lora_state_dict[name + ".B"] = module.B
    return lora_state_dict


def save_lora_checkpoint(lora_model, optimizer, scheduler, epoch, best_acc, ckpt_path):
    """保存LoRA模型的checkpoint，注意，不是保存完整的 model.state_dict()"""
    lora_state_dict = get_lora_state_dict(lora_model)
    checkpoint = {
        'epoch': epoch,
        'best_acc': best_acc,
        'lora_state_dict': lora_state_dict,
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
    }
    torch.save(checkpoint, ckpt_path)
    print(f"LoRA模型参数已保存到 {ckpt_path}")


def load_lora_checkpoint(ckpt_path, lora_model, optimizer, scheduler=None, device="cpu"):
    """
    加载LoRA检查点以恢复训练或进行推理

    Args:
        ckpt_path (str): 检查点文件路径。
        lora_model (nn.Module): LoRA模型实例 (预训练模型 + LoRA结构)。
        optimizer (torch.optim.Optimizer, optional): 优化器实例。
        scheduler (torch.optim.lr_scheduler._LRScheduler, optional): 调度器实例。
        device (str, optional): 设备类型，默认为 "cpu"。

    Returns:
        tuple: (start_epoch, best_acc) 重开的轮次和最佳准确率。
    """
    checkpoint = torch.load(ckpt_path, map_location=device)
    lora_model.load_state_dict(checkpoint['lora_state_dict'], strict=False)  # strict=False 是必须的，因为只加载模型参数的一个子集
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    if scheduler:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    start_epoch = checkpoint['epoch']
    best_acc = checkpoint['best_acc']
    print(f"LoRA模型参数已从 {ckpt_path} 加载")
    return start_epoch, best_acc


def train_loop_with_resume_lora(lora_model,
                                train_loader,
                                valid_loader,
                                criterion,
                                optimizer,
                                scheduler,
                                early_stopping,
                                best_lora_params_path,
                                num_epochs,
                                device, ):
    lora_model.to(device)
    # 训练表征曲线记录
    history = {
        "train_loss": [],
        "train_acc": [],
        "valid_loss": [],
        "valid_acc": [],
    }

    # 断点续训练
    if os.path.exists(best_lora_params_path):
        start_epoch, best_acc = load_lora_checkpoint(best_lora_params_path, lora_model, optimizer, scheduler, device)
        print(f"加载训练点LoRA参数成功，当前准确率为{best_acc:.4f}，从第{start_epoch}个epoch开始训练...")
    else:
        start_epoch, best_acc = 0, 0.0
        print("未找到LoRA参数文件，开始从头训练LoRA层...")

    for epoch in range(start_epoch, num_epochs):
        print(f"\nEpoch {epoch + 1}/{num_epochs}")
        print("-" * 50)
        train_loss, train_acc = train_epoch(lora_model, train_loader, optimizer, criterion, device)
        valid_loss, valid_acc = validate_epoch(lora_model, valid_loader, criterion, device)
        print(f"当前验证准确率: {valid_acc:.4f}")

        # 记录历史
        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["valid_loss"].append(valid_loss)
        history["valid_acc"].append(valid_acc)

        # 早停
        early_stopping(valid_acc)
        if early_stopping.early_stop:
            print("验证准确率未提升，提前停止训练")
            break

        # 学习率调度
        scheduler.step()
        print(f"当前学习率: {scheduler.get_last_lr()[0]:.3e}")

        # 保存最佳LoRA参数
        if valid_acc > best_acc:
            best_acc = valid_acc
            save_lora_checkpoint(lora_model, optimizer, scheduler, epoch, best_acc, best_lora_params_path)
            print(f"保存新的最佳LoRA参数，当前最佳验证准确率: {best_acc:.4f}")

    return best_acc, history


if __name__ == '__main__':
    base_ln = nn.Linear(10, 5)
    lora_ln = LoRALayer(10, 5, rank=2, alpha=1.0)
    model = LinearWithLoRA(base_ln, lora_ln)
    original_weight = model.linear.weight.clone()
    x_in = torch.randn(3, 10)
    output = model(x_in)
    print(f"LoRA线性层输出:\n{output}")

    # 测试LoRA参数参与梯度计算
    output.sum().backward()
    print("\nGradients of LoRA parameters:")
    print("LoRA A grad:\n", model.lora.A.grad)
    print("LoRA B grad:\n", model.lora.B.grad)
    model.merge_weights()
    print('\nMerged status:', model._merged)
