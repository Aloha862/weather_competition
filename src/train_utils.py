"""Training helpers for weather classification."""
from __future__ import annotations

import copy
import math
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import torch
from torch import nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LambdaLR, SequentialLR
from tqdm.auto import tqdm

from src.metrics import calculate_accuracy, calculate_f1

PROGRESS_LEAVE = True


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


class FocalLoss(nn.Module):
    """Multi-class focal loss for hard examples and minority classes."""

    def __init__(self, gamma: float = 2.0, weight: Optional[torch.Tensor] = None,
                 label_smoothing: float = 0.0):
        super().__init__()
        self.gamma = gamma
        self.weight = weight
        self.label_smoothing = label_smoothing

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce = nn.functional.cross_entropy(
            logits,
            targets,
            weight=self.weight,
            label_smoothing=self.label_smoothing,
            reduction="none",
        )
        pt = torch.exp(-ce)
        return (((1.0 - pt) ** self.gamma) * ce).mean()


class ModelEMA:
    """Exponential moving average of model weights for stabler validation."""

    def __init__(self, model: nn.Module, decay: float = 0.999):
        self.ema = copy.deepcopy(model).eval()
        self.decay = decay
        for param in self.ema.parameters():
            param.requires_grad_(False)

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        ema_state = self.ema.state_dict()
        model_state = model.state_dict()
        for key, ema_value in ema_state.items():
            model_value = model_state[key].detach()
            if ema_value.dtype.is_floating_point:
                ema_value.mul_(self.decay).add_(model_value, alpha=1.0 - self.decay)
            else:
                ema_value.copy_(model_value)


def _is_head_parameter(name: str) -> bool:
    markers = ("head", "classifier", "fc", "last_linear")
    return any(marker in name.lower() for marker in markers)


def set_backbone_trainable(model: nn.Module, trainable: bool) -> None:
    """Freeze or unfreeze non-classifier parameters."""
    for name, param in model.named_parameters():
        if not _is_head_parameter(name):
            param.requires_grad = trainable


def build_optimizer(model: nn.Module, lr: float, weight_decay: float, head_lr_mult: float = 1.0):
    """Build AdamW optimizer with optional higher LR for classifier head."""
    backbone_params = []
    head_params = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if _is_head_parameter(name):
            head_params.append(param)
        else:
            backbone_params.append(param)
    groups = []
    if backbone_params:
        groups.append({"params": backbone_params, "lr": lr, "name": "backbone"})
    if head_params:
        groups.append({"params": head_params, "lr": lr * max(1.0, head_lr_mult), "name": "head"})
    if not groups:
        groups = [{"params": model.parameters(), "lr": lr, "name": "all"}]
    return AdamW(groups, lr=lr, weight_decay=weight_decay)


def build_scheduler(optimizer, epochs: int, min_lr: float, use_warmup: bool = False, warmup_epochs: int = 3):
    """Build cosine scheduler with optional linear warmup."""
    epochs = max(1, int(epochs))
    if not use_warmup or warmup_epochs <= 0 or epochs <= 1:
        return CosineAnnealingLR(optimizer, T_max=epochs, eta_min=min_lr)
    warmup_epochs = min(max(1, int(warmup_epochs)), epochs - 1)
    warmup = LambdaLR(optimizer, lr_lambda=lambda e: float(e + 1) / float(warmup_epochs))
    cosine = CosineAnnealingLR(optimizer, T_max=max(1, epochs - warmup_epochs), eta_min=min_lr)
    return SequentialLR(optimizer, schedulers=[warmup, cosine], milestones=[warmup_epochs])


def compute_class_weights(labels: Iterable[int], num_classes: int, device: torch.device) -> torch.Tensor:
    """Compute inverse-frequency class weights."""
    labels = np.asarray(list(labels), dtype=np.int64)
    counts = np.bincount(labels, minlength=num_classes).astype(np.float32)
    counts[counts == 0] = 1.0
    weights = counts.sum() / (num_classes * counts)
    return torch.tensor(weights, dtype=torch.float32, device=device)


def compute_class_balanced_weights(labels: Iterable[int], num_classes: int, beta: float,
                                   device: torch.device) -> torch.Tensor:
    """Compute effective-number class-balanced weights."""
    labels = np.asarray(list(labels), dtype=np.int64)
    counts = np.bincount(labels, minlength=num_classes).astype(np.float32)
    counts[counts == 0] = 1.0
    effective_num = 1.0 - np.power(beta, counts)
    weights = (1.0 - beta) / np.maximum(effective_num, 1e-8)
    weights = weights / weights.sum() * num_classes
    return torch.tensor(weights, dtype=torch.float32, device=device)


def build_loss_function(num_classes: int, labels: Optional[Iterable[int]], use_class_weight: bool,
                        label_smoothing: float, device: torch.device, loss_type: str = "ce",
                        focal_gamma: float = 2.0, cb_beta: float = 0.9999):
    """Build CE, focal, or class-balanced focal loss."""
    loss_type = str(loss_type or "ce").lower()
    weight = None
    if loss_type == "class_balanced_focal" and labels is not None:
        weight = compute_class_balanced_weights(labels, num_classes, cb_beta, device)
        print(f"Class-balanced focal weights: {weight.detach().cpu().numpy().round(4).tolist()}")
    elif use_class_weight and labels is not None:
        weight = compute_class_weights(labels, num_classes, device)
        print(f"Class weights: {weight.detach().cpu().numpy().round(4).tolist()}")
    if loss_type in {"focal", "class_balanced_focal"}:
        return FocalLoss(gamma=focal_gamma, weight=weight, label_smoothing=label_smoothing)
    return nn.CrossEntropyLoss(weight=weight, label_smoothing=label_smoothing)


def _autocast_context(device: torch.device, use_amp: bool):
    """Return a safe autocast context for different torch versions."""
    enabled = bool(use_amp and device.type == "cuda")
    try:
        return torch.amp.autocast(device_type="cuda", enabled=enabled)
    except Exception:
        return torch.cuda.amp.autocast(enabled=enabled)


def train_one_epoch(model: nn.Module, loader, criterion, optimizer, device: torch.device,
                    scaler, use_amp: bool, epoch: int, grad_accum_steps: int = 1,
                    max_grad_norm: float = 0.0, ema: Optional[ModelEMA] = None) -> Dict[str, float]:
    """Run one training epoch."""
    model.train()
    total_loss = 0.0
    all_true: List[int] = []
    all_pred: List[int] = []
    progress = tqdm(loader, desc=f"Train {epoch}", leave=PROGRESS_LEAVE, dynamic_ncols=True, mininterval=0.5)
    grad_accum_steps = max(1, int(grad_accum_steps))
    optimizer.zero_grad(set_to_none=True)
    for step, (images, labels) in enumerate(progress, start=1):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        with _autocast_context(device, use_amp):
            logits = model(images)
            loss = criterion(logits, labels) / grad_accum_steps
        if scaler is not None and use_amp and device.type == "cuda":
            scaler.scale(loss).backward()
        else:
            loss.backward()
        should_step = step % grad_accum_steps == 0 or step == len(loader)
        if should_step:
            if scaler is not None and use_amp and device.type == "cuda":
                if max_grad_norm and max_grad_norm > 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                scaler.step(optimizer)
                scaler.update()
            else:
                if max_grad_norm and max_grad_norm > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            if ema is not None:
                ema.update(model)
        total_loss += float(loss.item()) * grad_accum_steps * images.size(0)
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
        for images, labels in tqdm(loader, desc="Validate", leave=PROGRESS_LEAVE, dynamic_ncols=True, mininterval=0.5):
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
