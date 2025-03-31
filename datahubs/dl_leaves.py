import torch
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
import pandas as pd
import os
from PIL import Image


# def get_image_size(img_path: str) -> tuple:
#     with Image.open(img_path) as img:
#         return img.size  # (width, height)


class LeavesDataset(Dataset):
    def __init__(self, df: pd.DataFrame, transform=None):
        """
        Args:
            df (DataFrame): 包含图片路径和标签的数据框 (train_df/test_df)
            transform (callable, optional): 对图片的变换操作
        """
        self.df = df.reset_index(drop=True)  # 重置索引，原索引丢弃
        self.transform = transform

        if 'label' in df.columns:  # 训练集
            labels, self.unique_labels = pd.factorize(df['label'])
            self.labels = torch.tensor(labels, dtype=torch.long)
            self.num_classes = len(self.unique_labels)
        else:  # 如果 'label' 列不存在则为测试集
            self.relative_image_paths = df['image'].values
            self.labels = None

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        """
        Returns:
            (image, label) 如果是训练集，
            image 如果是测试集
        """
        img_path = self.df.loc[idx, 'path']
        image = Image.open(img_path).convert('RGB')

        if self.transform:
            image = self.transform(image)

        if self.labels is not None:  # 训练集
            label = self.labels[idx]
            return image, label
        else:
            return image


def make_leaves_dl(data_dir: str = r"../data/kaggle-classify-leaves",
                   train_ratio: float = 0.9,
                   batch_size: int = 32,
                   train_transform=None,
                   test_transform=None):
    """
    Args:
        train_ratio (float): 训练集占总数据集的比例
        batch_size (int): 批量大小
        train_transform (callable, optional): 对训练图片的变换操作
        test_transform (callable, optional): 对测试图片的变换操作
    """
    # 数据文件路径
    train_file_path = os.path.join(data_dir, 'train.csv')
    test_file_path = os.path.join(data_dir, 'test.csv')

    # 读取标注数据
    train_df = pd.read_csv(train_file_path)
    test_df = pd.read_csv(test_file_path)

    # 将图片文件路径加入DataFrame
    train_df['path'] = train_df['image'].apply(lambda x: os.path.join(data_dir, x))
    test_df['path'] = test_df['image'].apply(lambda x: os.path.join(data_dir, x))

    # 将图片分辨率加入DataFrame
    # train_df['resolution'] = train_df['path'].apply(get_image_size)
    # test_df['resolution'] = test_df['path'].apply(get_image_size)

    # 创建数据集
    train_set = LeavesDataset(train_df, train_transform)
    test_set = LeavesDataset(test_df, test_transform)

    # 划分训练集和验证集
    train_size = int(len(train_set) * train_ratio)
    val_size = len(train_set) - train_size
    train_subset, val_subset = random_split(train_set, [train_size, val_size])

    # 创建数据加载器
    train_loader = DataLoader(train_subset, batch_size, shuffle=True, num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_subset, batch_size, shuffle=False, num_workers=0, pin_memory=True)
    test_loader = DataLoader(test_set, batch_size, shuffle=False, num_workers=0, pin_memory=True)

    return train_loader, val_loader, test_loader


if __name__ == '__main__':
    DATA_DIR = r"../data/kaggle-classify-leaves"
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),  # 随机水平翻转
        transforms.RandomRotation(degrees=15),  # 随机旋转
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    test_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    train_dl, val_dl, test_dl = make_leaves_dl(DATA_DIR, 0.8, 64, train_transform, test_transform)
    # 检查数据加载器
    for batch_idx, (X, y) in enumerate(train_dl, start=1):
        print(f'batch {batch_idx}: X.shape={X.shape}, y.shape={y.shape}')
    print("-" * 100)
    for batch_idx, X in enumerate(test_dl, start=1):
        print(f'batch {batch_idx}: X.shape={X.shape}')