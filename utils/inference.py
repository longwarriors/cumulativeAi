import torch, os
from torch.utils.data import DataLoader
from tqdm import tqdm
from typing import List, Tuple


def predict(model, dl_unlabelled: DataLoader, device) -> List[int]:
    """原始推理函数，只返回预测结果"""
    model.eval()
    all_preds = []
    with torch.no_grad():
        with tqdm(dl_unlabelled, desc="推理中", leave=False) as pbar_predict:
            for batch_idx, (X, *_) in enumerate(pbar_predict):
                X = X.to(device)
                logits = model(X)
                preds = torch.argmax(logits, dim=-1)
                all_preds.extend(preds.cpu().tolist())
                pbar_predict.set_postfix({"batch": f"{batch_idx + 1}"})
    return all_preds


def inference_with_ids(model, dl_unlabelled: DataLoader, device) -> Tuple[List[str], List[int]]:
    """带ID的推理函数，返回ID列表和对应的预测结果"""
    model.eval()
    all_ids = []
    all_preds = []
    with torch.no_grad():
        with tqdm(dl_unlabelled, desc="推理中", leave=False) as pbar_predict:
            for batch_idx, (X, ids) in enumerate(pbar_predict):
                all_ids.extend(ids)  # 直接来自 Dataset
                X = X.to(device)
                logits = model(X)
                preds = torch.argmax(logits, dim=-1)
                all_preds.extend(preds.cpu().tolist())
                pbar_predict.set_postfix({"batch": f"{batch_idx + 1}"})
    return all_ids, all_preds
