"""Metrics and analysis utilities."""
from __future__ import annotations

import shutil
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score, precision_recall_fscore_support

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


def save_per_class_metrics(y_true: Sequence[int], y_pred: Sequence[int], class_names: List[str],
                           save_path: str | Path) -> pd.DataFrame:
    """Save precision/recall/F1/support per class sorted by F1 ascending."""
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=list(range(len(class_names))), zero_division=0
    )
    df = pd.DataFrame({
        "class_name": class_names,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "support": support,
    }).sort_values("f1", ascending=True)
    save_path = Path(save_path)
    ensure_dir(save_path.parent)
    df.to_csv(save_path, index=False)
    return df


def save_validation_predictions(image_paths: Sequence[str], y_true: Sequence[int], y_pred: Sequence[int],
                                idx_to_class: Dict[str, str], save_path: str | Path) -> Path:
    """Save validation predictions for later error analysis."""
    rows = []
    for path, true_idx, pred_idx in zip(image_paths, y_true, y_pred):
        rows.append({
            "path": path,
            "image_id": Path(path).name,
            "true_idx": int(true_idx),
            "pred_idx": int(pred_idx),
            "true_label": idx_to_class.get(str(int(true_idx)), str(true_idx)),
            "pred_label": idx_to_class.get(str(int(pred_idx)), str(pred_idx)),
            "correct": int(int(true_idx) == int(pred_idx)),
        })
    save_path = Path(save_path)
    ensure_dir(save_path.parent)
    pd.DataFrame(rows).to_csv(save_path, index=False)
    return save_path


def top_confusion_pairs(y_true: Sequence[int], y_pred: Sequence[int], class_names: List[str],
                        top_k: int = 10) -> List[Dict[str, object]]:
    """Return most frequent off-diagonal confusion pairs."""
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(class_names))))
    pairs = []
    for i, true_name in enumerate(class_names):
        for j, pred_name in enumerate(class_names):
            if i == j or cm[i, j] == 0:
                continue
            pairs.append({"true_label": true_name, "pred_label": pred_name, "count": int(cm[i, j])})
    return sorted(pairs, key=lambda x: int(x["count"]), reverse=True)[:top_k]


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


def export_focus_error_samples(image_paths: Sequence[str], y_true: Sequence[int], y_pred: Sequence[int],
                               idx_to_class: Dict[str, str], output_dir: str | Path,
                               focus_classes: Sequence[str] = ("rainy", "snowy"),
                               max_per_class: int = 80) -> None:
    """Copy errors involving focus classes for quick rainy/snowy inspection."""
    output_dir = ensure_dir(output_dir)
    counters: Dict[str, int] = {}
    for path, true_idx, pred_idx in zip(image_paths, y_true, y_pred):
        if int(true_idx) == int(pred_idx):
            continue
        true_name = idx_to_class.get(str(int(true_idx)), str(true_idx))
        pred_name = idx_to_class.get(str(int(pred_idx)), str(pred_idx))
        if true_name not in focus_classes and pred_name not in focus_classes:
            continue
        key = f"true_{true_name}__pred_{pred_name}"
        counters[key] = counters.get(key, 0) + 1
        if counters[key] > max_per_class:
            continue
        dst_dir = ensure_dir(output_dir / key)
        src = Path(path)
        if src.exists():
            shutil.copy2(src, dst_dir / src.name)
