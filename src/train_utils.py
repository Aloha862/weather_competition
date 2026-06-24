"""Training helpers for weather classification."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import torch
from torch import nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

from src.metrics import calculate_accuracy, calculate_f1


@dataclass
class EarlyStopping:
    """Simple early stopping by validation score."""
    patience: int = 6
    best_score: float = -1.0
    counter: int = 0
    should_stop: bool = False

    def step(self, score: float) -> bool:
        if score > self.best_score:
            self.best_score = score
            self.counter = 0
            return True
        self.counter += 1
        if self.counter >= self.patience:
            self.should_stop = True
        return False


def build_optimizer(model: nn.Module, lr: float, weight_decay: float):
    """Build AdamW optimizer."""
    return AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)


def build_scheduler(optimizer, epochs: int, min_lr: float):
    """Build cosine annealing scheduler."""
    return CosineAnnealingLR(optimizer, T_max=max(1, epochs), eta_min=min_lr)


def compute_class_weights(labels: Iterable[int], num_classes: int, device: torch.device) -> torch.Tensor:
    """Compute inverse-frequency class weights."""
    labels = np.asarray(list(labels), dtype=np.int64)
    counts = np.bincount(labels, minlength=num_classes).astype(np.float32)
    counts[counts == 0] = 1.0
    weights = counts.sum() / (num_classes * counts)
    return torch.tensor(weights, dtype=torch.float32, device=device)


def build_loss_function(num_classes: int, labels: Optional[Iterable[int]], use_class_weight: bool,
                        label_smoothing: float, device: torch.device):
    """Build cross entropy loss with optional class weights."""
    weight = None
    if use_class_weight and labels is not None:
        weight = compute_class_weights(labels, num_classes, device)
        print(f"Class weights: {weight.detach().cpu().numpy().round(4).tolist()}")
    return nn.CrossEntropyLoss(weight=weight, label_smoothing=label_smoothing)


def _autocast_context(device: torch.device, use_amp: bool):
    """Return a safe autocast context for different torch versions."""
    enabled = bool(use_amp and device.type == "cuda")
    try:
        return torch.amp.autocast(device_type="cuda", enabled=enabled)
    except Exception:
        return torch.cuda.amp.autocast(enabled=enabled)


def train_one_epoch(model: nn.Module, loader, criterion, optimizer, device: torch.device,
                    scaler, use_amp: bool, epoch: int) -> Dict[str, float]:
    """Run one training epoch."""
    model.train()
    total_loss = 0.0
    all_true: List[int] = []
    all_pred: List[int] = []
    progress = tqdm(loader, desc=f"Train {epoch}", leave=False)
    for images, labels in progress:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with _autocast_context(device, use_amp):
            logits = model(images)
            loss = criterion(logits, labels)
        if scaler is not None and use_amp and device.type == "cuda":
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()
        total_loss += float(loss.item()) * images.size(0)
        preds = logits.argmax(dim=1)
        all_true.extend(labels.detach().cpu().tolist())
        all_pred.extend(preds.detach().cpu().tolist())
        progress.set_postfix(loss=f"{loss.item():.4f}")
    avg_loss = total_loss / max(1, len(loader.dataset))
    acc = calculate_accuracy(all_true, all_pred)
    macro_f1, weighted_f1 = calculate_f1(all_true, all_pred)
    return {"loss": avg_loss, "accuracy": acc, "macro_f1": macro_f1, "weighted_f1": weighted_f1}


def validate(model: nn.Module, loader, criterion, device: torch.device) -> Tuple[Dict[str, float], List[int], List[int]]:
    """Validate model without random augmentation."""
    model.eval()
    total_loss = 0.0
    all_true: List[int] = []
    all_pred: List[int] = []
    with torch.inference_mode():
        for images, labels in tqdm(loader, desc="Validate", leave=False):
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            logits = model(images)
            loss = criterion(logits, labels)
            total_loss += float(loss.item()) * images.size(0)
            preds = logits.argmax(dim=1)
            all_true.extend(labels.detach().cpu().tolist())
            all_pred.extend(preds.detach().cpu().tolist())
    avg_loss = total_loss / max(1, len(loader.dataset))
    acc = calculate_accuracy(all_true, all_pred)
    macro_f1, weighted_f1 = calculate_f1(all_true, all_pred)
    metrics = {"loss": avg_loss, "accuracy": acc, "macro_f1": macro_f1, "weighted_f1": weighted_f1}
    return metrics, all_true, all_pred
