# https://www.kaggle.com/datasets/gauravduttakiit/covid-19-tweet-classification
# https://zindi.africa/competitions/covid-19-tweet-classification/data

import pandas as pd
import numpy as np
import torch
from torch import optim, nn
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils import clip_grad_norm_
from transformers import AutoTokenizer, AutoConfig, BertForSequenceClassification
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from tqdm import tqdm
from typing import Union, Optional
import re, os, warnings

warnings.filterwarnings("ignore")

# 加载数据集
data_train = pd.read_csv('../data/COVID-19-Tweet/updated_train.csv')
data_test = pd.read_csv('../data/COVID-19-Tweet/updated_test.csv')
submission_example = pd.read_csv('../data/COVID-19-Tweet/updated_ss.csv')

# 检查数据分布
print(f"训练数据标签分布：\n{data_train['target'].value_counts()}")
print(f"不平衡标签比例：{data_train['target'].value_counts()[0] / data_train['target'].value_counts()[1]:.2f} : 1\n")


# 数据预处理
def clean_text(text: Union[str, float, int, None, np.nan]) -> str:
    if pd.isna(text):
        return ""
    text = str(text)
    text = re.sub(r'[^\w\s@#.,!?-]', ' ', text)  # 移除特殊字符但保留基本标点
    text = re.sub(r'\s+', ' ', text).strip()  # 替换多个空格为一个空格
    return text


data_train['text'] = data_train['text'].apply(clean_text)
data_test['text'] = data_test['text'].apply(clean_text)

# 分层采样确保训练集和验证集的标签分布一致
X_train, X_valid, y_train, y_valid = train_test_split(
    data_train['text'], data_train['target'],
    test_size=0.2, shuffle=True, random_state=42, stratify=data_train['target']
)
print(f"训练集大小: {len(X_train)}, 验证集大小: {len(X_valid)}")
print(f"训练集标签分布：{y_train.value_counts().to_dict()}")
print(f"验证集标签分布：{y_valid.value_counts().to_dict()}")

# 对社交媒体文本更友好的模型
MODEL_NAME = 'bert-base-uncased'
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

# 分析文本长度并设置最大长度
train_lengths = [len(tokenizer.encode(text, add_special_tokens=True)) for text in X_train]
print(f"平均token长度: {np.mean(train_lengths):.2f}")
print(f"最大token长度: {np.max(train_lengths)}")
print(f"95%分位数token长度: {np.percentile(train_lengths, 95):.2f}\n")
MAX_LENGTH = min(128, int(np.percentile(train_lengths, 95)))  # 设置最大长度为95%分位数或128，取小者

# 对数据进行tokenization
train_tokens = tokenizer(
    list(X_train),
    padding=True,
    truncation=True,
    max_length=MAX_LENGTH,
    return_tensors='pt',
)
valid_tokens = tokenizer(
    list(X_valid),
    padding=True,
    truncation=True,
    max_length=MAX_LENGTH,
    return_tensors='pt',
)
test_tokens = tokenizer(
    list(data_test['text']),
    padding=True,
    truncation=True,
    max_length=MAX_LENGTH,
    return_tensors='pt',
)


# 自定义数据集类
class TweetDataset(Dataset):
    def __init__(self, texts, tokens, labels):
        self.texts = texts
        self.tokens = tokens
        self.labels = labels

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        return {
            'input_ids': self.tokens['input_ids'][idx],
            'attention_mask': self.tokens['attention_mask'][idx],
            'labels': torch.tensor(self.labels.iloc[idx], dtype=torch.long)
        }


class TestDataset(Dataset):
    def __init__(self, tokens):
        self.tokens = tokens

    def __len__(self):
        return len(self.tokens['input_ids'])

    def __getitem__(self, idx):
        item_ = {
            'input_ids': self.tokens['input_ids'][idx],
            'attention_mask': self.tokens['attention_mask'][idx],
        }
        return item_


# 创建数据集和数据加载器
BATCH_SIZE = 16
train_dataset = TweetDataset(X_train, train_tokens, y_train)
valid_dataset = TweetDataset(X_valid, valid_tokens, y_valid)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
valid_loader = DataLoader(valid_dataset, batch_size=BATCH_SIZE, shuffle=False)

test_dataset = TestDataset(test_tokens)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

sample_batch = next(iter(train_loader))
input_memory = sample_batch['input_ids'].element_size() * sample_batch['input_ids'].nelement()
mask_memory = sample_batch['attention_mask'].element_size() * sample_batch['attention_mask'].nelement()
labels_memory = sample_batch['labels'].element_size() * sample_batch['labels'].nelement()
total_memory = input_memory + mask_memory + labels_memory
print("=" * 60)
print("数据加载器特征分析")
print(f"实际batch中的样本数: {sample_batch['input_ids'].shape[0]}")
print(f"序列最大长度: {sample_batch['input_ids'].shape[1]}")

print("\n--- Input IDs 信息 ---")
print(f"Input IDs shape: {sample_batch['input_ids'].shape}")
print(f"Input IDs dtype: {sample_batch['input_ids'].dtype}")
print(f"Input IDs 范围: [{sample_batch['input_ids'].min().item()}, {sample_batch['input_ids'].max().item()}]")

print("\n--- Attention Mask 信息 ---")
print(f"Attention Mask shape: {sample_batch['attention_mask'].shape}")
print(f"Attention Mask dtype: {sample_batch['attention_mask'].dtype}")
print(f"Attention Mask unique values: {torch.unique(sample_batch['attention_mask']).tolist()}")

print(f"\n--- 内存使用情况 ---")
print(f"{'Single batch':>15}: {total_memory / 1024 / 1024:.2f} MB")
print(f"{'Input IDs':>15}: {input_memory / 1024 / 1024:.2f} MB")
print(f"{'Attention Mask':>15}: {mask_memory / 1024 / 1024:.2f} MB")
print(f"{'Labels':>15}: {labels_memory / 1024 / 1024:.2f} MB")
print("=" * 60)

# 模型配置和初始化
config = AutoConfig.from_pretrained(MODEL_NAME)
config.num_labels = 2  # 二分类任务
config.hidden_dropout_prob = 0.2  # 增加dropout以防止过拟合
config.attention_probs_dropout_prob = 0.2
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = BertForSequenceClassification.from_pretrained(MODEL_NAME, config=config)
model.to(DEVICE)
print(f"使用的设备: {DEVICE}")

# 冻结预训练BERT模型的前6层以进行正则化
for name, param in model.bert.named_parameters():
    if 'encoder.layer' in name:
        layer_num = int(name.split('encoder.layer.')[1].split('.')[0])
        if layer_num < 6:
            param.requires_grad = False
            print(f"冻结层: {name}")
print("模型参数冻结状态检查 (第5层应为False, 第6层应为True):")
for name, param in model.bert.named_parameters():
    if 'encoder.layer.5.output.dense.weight' in name:
        print(f"第5层{name}输出权重 requires_grad={param.requires_grad}")
    if 'encoder.layer.6.output.dense.weight' in name:
        print(f"第6层{name}输出权重 requires_grad={param.requires_grad}")

# 处理类别不平衡
class_counts = y_train.value_counts()
print(f"类别计数: {class_counts.to_dict()}")
total_samples = len(y_train)
class_weights = total_samples / (2 * class_counts)
class_weights = torch.FloatTensor(class_weights).to(DEVICE)
print(f"类别权重: {class_weights}")

# 损失函数和优化器
optimizer = optim.AdamW(model.parameters(), lr=2e-6, weight_decay=0.001)
criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.1)  # 添加标签平滑来防止模型过于自信

# 学习率调度器
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=180, eta_min=1e-6)


# 早停机制
class EarlyStopping:
    def __init__(self, patience: int = 3, delta: float = 1e-4, mode: str = 'min'):
        """
        Args:
            patience (int, optional): 容忍的epochs数
            delta (float, optional): 最小改善幅度
            mode (str, optional): 'max' 表示分数越高越好（如准确率），'min' 表示越小越好（如损失）
        """
        self.patience = patience
        self.delta = delta
        self.mode = mode
        self.counter = 0
        self.best_score = None
        self.early_stop = False

    def __call__(self, score: float):
        if self.mode == 'max':  # 分数越高越好（如准确率）
            improved = self.best_score is None or score > self.best_score + self.delta
        else:  # 分数越低越好（如损失）
            improved = self.best_score is None or score < self.best_score - self.delta

        if improved:
            self.best_score = score
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True


early_stopping = EarlyStopping(patience=8, delta=1e-5, mode='max')  # 准确率越高越好


# 训练和验证函数
def train_model(model, device, train_loader, optimizer, loss_fn=None):
    model.train()
    total_loss = 0.0
    correct_predictions = 0
    total_predictions = 0

    # 使用 with 语句确保 tqdm 正确关闭
    with tqdm(train_loader, desc="Training", unit="batch") as pbar:
        for batch in pbar:
            batch = {k: v.to(device) for k, v in batch.items()}
            batch_size = batch['input_ids'].size(0)
            optimizer.zero_grad()

            if loss_fn is None:
                # 使用模型内置损失
                outputs = model(
                    input_ids=batch['input_ids'],
                    attention_mask=batch['attention_mask'],
                    labels=batch['labels']
                )
                logits = outputs.logits
                loss = outputs.loss
            else:
                # 不传入labels，手动计算损失
                outputs = model(
                    input_ids=batch['input_ids'],
                    attention_mask=batch['attention_mask']
                )
                logits = outputs.logits
                loss = loss_fn(logits, batch['labels'])

            loss.backward()
            clip_grad_norm_(model.parameters(), max_norm=1.0)  # 梯度裁剪，防止梯度爆炸
            optimizer.step()

            total_loss += loss.item() * batch_size
            predictions = torch.argmax(logits, dim=-1)
            correct_predictions += (predictions == batch['labels']).sum().item()
            total_predictions += batch_size

            # 更新进度条显示当前损失
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})

    avg_loss = total_loss / len(train_loader)
    accuracy = correct_predictions / total_predictions
    return avg_loss, accuracy


def validate_model(model, device, valid_loader, loss_fn=None):
    model.eval()
    total_loss = 0.0
    all_predictions = []
    all_labels = []

    with torch.no_grad():
        # 使用 with 语句确保 tqdm 正确关闭
        with tqdm(valid_loader, desc="Validating", unit="batch") as pbar:
            for batch in pbar:
                batch = {k: v.to(device) for k, v in batch.items()}
                batch_size = batch['input_ids'].size(0)

                if loss_fn is None:
                    # 使用模型内置损失
                    outputs = model(
                        input_ids=batch['input_ids'],
                        attention_mask=batch['attention_mask'],
                        labels=batch['labels']
                    )
                    logits = outputs.logits
                    loss = outputs.loss
                else:
                    # 不传入labels，手动计算损失
                    outputs = model(batch['input_ids'], batch['attention_mask'])
                    logits = outputs.logits
                    loss = loss_fn(logits, batch['labels'])

                total_loss += loss.item() * batch_size
                predictions = torch.argmax(logits, dim=-1)
                all_predictions.extend(predictions.cpu().numpy())
                all_labels.extend(batch['labels'].cpu().numpy())

                # 更新进度条显示当前准确率
                pbar.set_postfix({'loss': f'{loss.item():.4f}'})

    avg_loss = total_loss / len(valid_loader)
    accuracy = accuracy_score(all_labels, all_predictions)
    return avg_loss, accuracy, all_predictions, all_labels


def predict_model(model, device, test_loader):
    """对测试数据进行推理"""
    model.eval()
    all_predictions = []
    all_probabilities = []

    print("\n开始推理...")
    with torch.no_grad():
        # 使用 with 语句确保 tqdm 正确关闭
        with tqdm(test_loader, desc="Predicting", unit="batch") as pbar:
            for batch in pbar:
                batch = {k: v.to(device) for k, v in batch.items()}
                outputs = model(
                    input_ids=batch['input_ids'],
                    attention_mask=batch['attention_mask']
                )
                logits = outputs.logits
                probabilities = torch.softmax(logits, dim=-1)
                predictions = torch.argmax(logits, dim=-1)
                all_predictions.extend(predictions.cpu().numpy())
                all_probabilities.extend(probabilities.cpu().numpy())

    return all_predictions, all_probabilities


# 训练和验证循环
NUM_EPOCHS = 20
best_valid_accuracy = 0.0
ckpt_path = r'..\checkpoints\best_model_covid-19_tweet.pth'

# 断点续训练
if os.path.exists(ckpt_path):
    print(f"加载断点续训模型: {ckpt_path}")
    try:
        model.load_state_dict(torch.load(ckpt_path, map_location=DEVICE))
        print("模型加载成功!")
        print("测试加载模型的性能...")
        _, val_acc, _, _ = validate_model(model, DEVICE, valid_loader, criterion)
        best_valid_accuracy = val_acc
        print(f"加载模型的验证准确率: {val_acc:.4f}")
    except Exception as e:
        print(f"加载模型失败: {e}")
        best_valid_accuracy = 0.0
else:
    print("未检测到之前的模型，将使用新初始化的预训练模型.")


def train_step():
    print("开始训练...")
    for epoch in range(NUM_EPOCHS):
        print(f"\nEpoch {epoch + 1}/{NUM_EPOCHS}")
        print('-' * 50)

        train_loss, train_accuracy = train_model(model, DEVICE, train_loader, optimizer, loss_fn=criterion)
        valid_loss, valid_accuracy, valid_predictions, valid_labels = validate_model(model, DEVICE, valid_loader,
                                                                                     loss_fn=criterion)
        scheduler.step()
        print(f"训练阶段损失: {train_loss:.4f}, 训练阶段准确率: {train_accuracy:.4f}")
        print(f"验证阶段损失: {valid_loss:.4f}, 验证阶段准确率: {valid_accuracy:.4f}")
        print(f"当前学习率: {scheduler.get_last_lr()[0]:.2e}")

        # 保存最佳模型
        if valid_accuracy > best_valid_accuracy:
            best_valid_accuracy = valid_accuracy
            torch.save(model.state_dict(), r'..\checkpoints\best_model_covid-19_tweet.pth')
            print(f'保存最佳模型，验证准确率: {valid_accuracy:.4f}')

        # 早停检查
        early_stopping(valid_accuracy)
        if early_stopping.early_stop:
            print("早停触发!")
            break

    # 打印分类报告
    print(f'\n最佳验证准确率: {best_valid_accuracy:.4f}')
    print('=' * 60)
    print("分类报告:")
    print(classification_report(valid_labels, valid_predictions, target_names=['Non-COVID', 'COVID']))


def predict_step():
    print("开始推理...")
    test_predictions, test_probabilities = predict_model(model, DEVICE, test_loader)
    print(f"预测结果数量: {len(test_predictions)}")
    print(f"预测分布: {np.bincount(test_predictions)}")

    # 保存预测结果
    submission = pd.DataFrame({
        'ID': data_test['ID'],
        'target': test_predictions
    })
    output_path = r'..\output\submission_covid-19_tweet.csv'
    submission.to_csv(output_path, index=False)
    print(f"推理完成，预测结果已保存到: {output_path}")


if __name__ == "__main__":
    predict_step()
