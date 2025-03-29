# PyTorch Lightning #7 - Callbacks
# https://www.youtube.com/watch?v=Wcze6oGch1g&list=PLhhyoLH6IjfyL740PTuXef4TstxAK6nGP&index=7
import torch
import torchvision
from torch import nn, optim
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
import pytorch_lightning as pl
import torchmetrics
from torchmetrics import Metric
from pytorch_lightning.callbacks import Callback, EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger

torch.set_float32_matmul_precision('high')


class MyPrintCallback(Callback):
    def __init__(self):
        super().__init__()

    def on_train_start(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        print("Training started.")

    def on_train_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        print("Training ended.")


class MyMetric(Metric):
    def __init__(self):
        super().__init__()
        self.add_state("num_correct", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("num_samples", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, preds, target):
        preds = torch.argmax(preds, dim=1)
        assert preds.shape == target.shape
        self.num_correct += (preds == target).sum().item()
        self.num_samples += target.numel()

    def compute(self):
        return self.num_correct / self.num_samples


class Net(pl.LightningModule):
    def __init__(self, input_size, num_classes, learning_rate=0.001):
        super().__init__()
        self.fc1 = nn.Linear(input_size, 50)
        self.fc2 = nn.Linear(50, num_classes)
        self.relu = nn.ReLU()
        self.loss_fn = nn.CrossEntropyLoss()
        self.lr = learning_rate
        self.my_accuracy = MyMetric()
        self.accuracy = torchmetrics.Accuracy(task="multiclass", num_classes=num_classes)
        self.f1_score = torchmetrics.F1Score(task="multiclass", num_classes=num_classes)

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x

    def training_step(self, batch, batch_idx):
        X, y = batch
        loss, scores, y = self._common_step(batch, batch_idx)
        self.my_accuracy.update(scores, y)
        accuracy = self.my_accuracy.compute()  # 修改为正确的调用方式
        self.accuracy.update(scores, y)
        self.f1_score.update(scores, y)
        f1 = self.f1_score.compute()
        self.log_dict({
            'train_loss': loss,
            'train_accuracy': accuracy,
            'train_f1_score': f1},
            on_step=False,
            on_epoch=True,
            prog_bar=True
        )
        if batch_idx % 100 == 0:
            grid = torchvision.utils.make_grid(X[:8].view(-1, 1, 28, 28), nrow=8)
            self.logger.experiment.add_image('mnist_images', grid, self.global_step)
        return {'loss': loss}

    def validation_step(self, batch, batch_idx):
        loss, scores, y = self._common_step(batch, batch_idx)
        self.accuracy.update(scores, y)
        val_acc = self.accuracy.compute()
        self.log_dict({'val_loss': loss, 'val_accuracy': val_acc},
                      on_step=False, on_epoch=True, prog_bar=True)

    def test_step(self, batch, batch_idx):
        loss, scores, y = self._common_step(batch, batch_idx)
        self.accuracy.update(scores, y)
        test_acc = self.accuracy.compute()
        self.log_dict({'test_loss': loss, 'test_accuracy': test_acc},
                      on_step=False, on_epoch=True)

    def _common_step(self, batch, batch_idx):
        data, targets = batch
        data = data.view(data.size(0), -1)
        outputs = self(data)
        loss = self.loss_fn(outputs, targets)
        return loss, outputs, targets

    def predict_step(self, batch, batch_idx):
        data, targets = batch
        data = data.view(data.size(0), -1)
        outputs = self(data)
        preds = torch.argmax(outputs, dim=1)
        return preds

    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=self.lr)

    def on_epoch_end(self):
        # 重置指标，避免累积
        self.my_accuracy.reset()
        self.accuracy.reset()
        self.f1_score.reset()


class MnistDataModule(pl.LightningDataModule):
    def __init__(self, data_dir, batch_size=64, num_workers=4):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.mnist_test = None
        self.mnist_val = None
        self.mnist_train = None
        self.transform = transforms.Compose([
            transforms.RandomVerticalFlip(),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))  # MNIST标准化参数
        ])

    def prepare_data(self):
        # datasets.MNIST(root='../../data', train=True, download=True)
        # datasets.MNIST(root='../../data', train=False, download=True)
        pass

    def setup(self, stage=None):
        mnist_full = datasets.MNIST(root=self.data_dir, train=True, download=False, transform=self.transform)
        train_ratio = 0.8
        train_size = int(train_ratio * len(mnist_full))
        val_size = len(mnist_full) - train_size
        self.mnist_train, self.mnist_val = random_split(mnist_full, [train_size, val_size])
        if stage == 'test' or stage is None:
            self.mnist_test = datasets.MNIST(root=self.data_dir, train=False, download=False, transform=self.transform)

    def train_dataloader(self):
        return DataLoader(
            self.mnist_train,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            persistent_workers=True if self.num_workers > 0 else False  # 只有当有worker时才启用
        )

    def val_dataloader(self):
        return DataLoader(
            self.mnist_val,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            persistent_workers=True if self.num_workers > 0 else False
        )

    def test_dataloader(self):
        return DataLoader(
            self.mnist_test,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            persistent_workers=True if self.num_workers > 0 else False
        )


def main():
    # --------------------Compute related---------------------------
    ACCELERATOR = 'gpu' if torch.cuda.is_available() else 'cpu'
    DEVICES = [0] if torch.cuda.is_available() else None
    PRECISION = "16-mixed" if torch.cuda.is_available() else 32

    # Hyperparameters
    INPUT_SIZE = 28 * 28
    NUM_CLASSES = 10
    BATCH_SIZE = 32
    NUM_EPOCHS = 30
    LEARNING_RATE = 0.001

    # Dataset
    DATA_DIR = r'../../data/'
    NUM_WORKERS = 8 if torch.cuda.is_available() else 0

    # Initialize model
    model = Net(INPUT_SIZE, NUM_CLASSES, LEARNING_RATE)
    dm = MnistDataModule(data_dir=DATA_DIR, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS)

    # Logger
    tb_logger = TensorBoardLogger(save_dir=r"..\logs\tb_logs", name="mnist_pl_model_v1")
    # Lightning trainer
    trainer = pl.Trainer(min_epochs=1,
                         max_epochs=NUM_EPOCHS,
                         accelerator=ACCELERATOR,
                         devices=DEVICES,
                         precision=PRECISION,
                         callbacks=[MyPrintCallback(), EarlyStopping(monitor='val_loss', patience=5)],
                         logger=tb_logger)
    trainer.fit(model, dm)
    trainer.validate(model, dm)
    trainer.test(model, dm)


if __name__ == '__main__':
    main()
