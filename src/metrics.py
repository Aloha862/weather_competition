"""Metrics and analysis utilities."""
from __future__ import annotations

import shutil
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score

from src.utils import ensure_dir


def calculate_accuracy(y_true: Sequence[int], y_pred: Sequence[int]) -> float:
    """Calculate accuracy."""
    return float(accuracy_score(y_true, y_pred))


def calculate_f1(y_true: Sequence[int], y_pred: Sequence[int]) -> Tuple[float, float]:
    """Calculate macro F1 and weighted F1."""
    macro_f1 = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    weighted_f1 = float(f1_score(y_true, y_pred, average="weighted", zero_division=0))
    return macro_f1, weighted_f1


def print_classification_report(y_true: Sequence[int], y_pred: Sequence[int], target_names: List[str]) -> str:
    """Print and return sklearn classification report."""
    report = classification_report(y_true, y_pred, target_names=target_names, zero_division=0)
    print(report)
    return report


def plot_confusion_matrix(y_true: Sequence[int], y_pred: Sequence[int], class_names: List[str], save_path: str | Path) -> None:
    """Save confusion matrix figure."""
    save_path = Path(save_path)
    ensure_dir(save_path.parent)
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(class_names))))
    fig, ax = plt.subplots(figsize=(max(6, len(class_names) * 0.8), max(5, len(class_names) * 0.7)))
    im = ax.imshow(cm)
    ax.set_xticks(np.arange(len(class_names)))
    ax.set_yticks(np.arange(len(class_names)))
    ax.set_xticklabels(class_names, rotation=45, ha="right")
    ax.set_yticklabels(class_names)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    for i in range(len(class_names)):
        for j in range(len(class_names)):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center")
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(save_path, dpi=160)
    plt.close(fig)


def plot_training_curves(log_path: str | Path, save_path: str | Path) -> None:
    """Save loss and validation metric curves from the CSV training log."""
    log_path = Path(log_path)
    if not log_path.exists():
        return
    df = pd.read_csv(log_path)
    if df.empty:
        return
    save_path = Path(save_path)
    ensure_dir(save_path.parent)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].plot(df["epoch"], df["train_loss"], marker="o", label="train_loss")
    axes[0].plot(df["epoch"], df["val_loss"], marker="o", label="val_loss")
    axes[0].set_xlabel("epoch")
    axes[0].set_ylabel("loss")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    axes[1].plot(df["epoch"], df["train_macro_f1"], marker="o", label="train_macro_f1")
    axes[1].plot(df["epoch"], df["val_macro_f1"], marker="o", label="val_macro_f1")
    axes[1].plot(df["epoch"], df["val_accuracy"], marker="o", label="val_accuracy")
    axes[1].set_xlabel("epoch")
    axes[1].set_ylabel("score")
    axes[1].set_ylim(0, 1)
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(save_path, dpi=160)
    plt.close(fig)


def export_error_samples(image_paths: Sequence[str], y_true: Sequence[int], y_pred: Sequence[int],
                         idx_to_class: Dict[str, str], output_dir: str | Path, max_per_pair: int = 30) -> None:
    """Copy misclassified validation samples to grouped folders."""
    output_dir = ensure_dir(output_dir)
    counters: Dict[str, int] = {}
    for path, true_idx, pred_idx in zip(image_paths, y_true, y_pred):
        if int(true_idx) == int(pred_idx):
            continue
        true_name = idx_to_class.get(str(int(true_idx)), str(true_idx))
        pred_name = idx_to_class.get(str(int(pred_idx)), str(pred_idx))
        folder_name = f"true_{true_name}__pred_{pred_name}"
        key = folder_name
        counters[key] = counters.get(key, 0) + 1
        if counters[key] > max_per_pair:
            continue
        dst_dir = ensure_dir(output_dir / folder_name)
        src = Path(path)
        if src.exists():
            shutil.copy2(src, dst_dir / src.name)
