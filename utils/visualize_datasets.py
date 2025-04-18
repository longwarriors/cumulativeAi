import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
import numpy as np
import torch


def display_batch_gray_images(dataloader: DataLoader,
                              rows_count: int = 10,
                              cols_count: int = 10):
    """
    display a batch of images from the dataloader.

    Args:
        dataloader (DataLoader): The DataLoader containing the images.
        rows_count (int): Number of rows in the grid.
        cols_count (int): Number of columns in the grid.
    """
    # Get a batch of images and labels
    images, labels = next(iter(dataloader))
    # images.shape = (batch_size, channels, height, width)
    batch_size = images.shape[0]

    # 确保图形不会太小
    min_size = 6
    width = max(2.0 * cols_count, min_size)  # 每列分配2.0英寸
    height = max(2.0 * rows_count, min_size)  # 每行分配2.0英寸
    fig, axes = plt.subplots(rows_count, cols_count, figsize=(width, height))

    # 确保不超出可用图像数量
    max_images = min(rows_count * cols_count, batch_size)
    for i in range(rows_count):
        for j in range(cols_count):
            idx = i * cols_count + j
            if idx < max_images:
                ax = axes[i, j]
                ax.imshow(images[idx].squeeze().numpy(),
                          cmap='gray',  # 更常用于MNIST等数据集
                          interpolation='nearest')
                ax.set_title(f'Label: {labels[idx].item()}')
                ax.axis('off')
            else:
                axes[i, j].axis('off')

    plt.tight_layout()
    plt.show()


def display_batch_color_images(dataloader: DataLoader,
                               rows_count: int = 10,
                               cols_count: int = 10):
    """
    display a batch of images from the dataloader.

    Args:
        dataloader (DataLoader): The DataLoader containing the images.
        rows_count (int): Number of rows in the grid.
        cols_count (int): Number of columns in the grid.
    """
    # Get a batch of images and labels
    images, labels = next(iter(dataloader))
    # images.shape = (batch_size, channels, height, width)
    batch_size = images.shape[0]

    # 确保图形不会太小
    min_size = 6
    width = max(1.8 * cols_count, min_size)
    height = max(1.8 * rows_count, min_size)
    fig, axes = plt.subplots(rows_count, cols_count, figsize=(width, height))

    # 确保不超出可用图像数量
    max_images = min(rows_count * cols_count, batch_size)
    for i in range(rows_count):
        for j in range(cols_count):
            idx = i * cols_count + j
            if idx < max_images:
                # PyTorch加载的图像格式为[C, H, W]，matplotlib需要[H, W, C]
                image_np = images[idx].permute(1, 2, 0).cpu().detach().numpy()
                if image_np.max() > 1.0:
                    image_np = image_np / 255.0

                # 设置标题显示的类别名称
                label = labels[idx].item()
                ax = axes[i, j]
                ax.imshow(image_np, interpolation='nearest') # 不需要设置cmap参数，彩色图像会自动使用RGB通道
                ax.set_title(f'Label: {label}')
                ax.axis('off')
            else:
                axes[i, j].axis('off')

    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    from torchvision import datasets, transforms
    from torch.utils.data import Dataset, DataLoader

    mnist_train = datasets.MNIST(root='../data', train=True, transform=transforms.ToTensor(), download=False)
    train_loader = DataLoader(mnist_train, batch_size=64, shuffle=True)
    display_batch_gray_images(train_loader, rows_count=5, cols_count=10)
