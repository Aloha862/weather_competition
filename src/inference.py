"""Inference and CSV generation utilities."""
from __future__ import annotations

from pathlib import Path
import time
from typing import Dict, List, Tuple

import pandas as pd
import torch
from tqdm import tqdm

from src.dataset import create_test_loader
from src.utils import check_submission_format, ensure_dir, save_json


def _model_size_mb(model) -> float:
    return float(sum(p.numel() * p.element_size() for p in model.parameters()) / (1024 ** 2))


def predict_test(model, test_loader, device: torch.device, idx_to_class: Dict[str, str],
                 use_tta: bool = False, tta_mode: str = "none",
                 benchmark_path: str | Path | None = None) -> Tuple[List[str], List[str], List[float]]:
    """Run batch inference on test images."""
    model.eval()
    image_ids: List[str] = []
    labels: List[str] = []
    confs: List[float] = []
    start = time.perf_counter()
    num_images = 0
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    with torch.inference_mode():
        for images, ids in tqdm(test_loader, desc="Inference", leave=False):
            images = images.to(device, non_blocking=True)
            logits = model(images)
            if use_tta and str(tta_mode).lower() == "hflip":
                logits = (logits + model(torch.flip(images, dims=[3]))) / 2.0
            probs = torch.softmax(logits, dim=1)
            scores, preds = probs.max(dim=1)
            num_images += len(ids)
            for img_id, pred_idx, score in zip(ids, preds.cpu().tolist(), scores.cpu().tolist()):
                image_ids.append(str(img_id))
                labels.append(idx_to_class.get(str(int(pred_idx)), str(int(pred_idx))))
                confs.append(float(score))
    elapsed = time.perf_counter() - start
    if benchmark_path is not None:
        peak_mb = 0.0
        if device.type == "cuda":
            peak_mb = float(torch.cuda.max_memory_allocated(device) / (1024 ** 2))
        save_json({
            "num_images": num_images,
            "elapsed_sec": elapsed,
            "time_per_image_ms": (elapsed / max(1, num_images)) * 1000.0,
            "batch_size": getattr(test_loader, "batch_size", None),
            "use_tta": bool(use_tta),
            "tta_mode": tta_mode,
            "model_size_mb": _model_size_mb(model),
            "peak_gpu_memory_mb": peak_mb,
        }, benchmark_path)
    return image_ids, labels, confs


def generate_submission(image_ids: List[str], pred_labels: List[str], sample_submission_path: str | Path,
                        save_path: str | Path) -> Path:
    """Generate a submission file using the sample columns when available."""
    save_path = Path(save_path)
    ensure_dir(save_path.parent)
    sample_path = Path(sample_submission_path)
    if sample_path.exists():
        sample = pd.read_csv(sample_path)
        if len(sample.columns) >= 2:
            id_col, label_col = sample.columns[0], sample.columns[1]
            result = pd.DataFrame({id_col: image_ids, label_col: pred_labels})
            if len(sample) == len(result):
                order_map = {img_id: label for img_id, label in zip(image_ids, pred_labels)}
                result = sample.copy()
                result[label_col] = result[id_col].map(order_map).fillna(result[label_col])
        else:
            result = pd.DataFrame({"image_id": image_ids, "label": pred_labels})
    else:
        result = pd.DataFrame({"image_id": image_ids, "label": pred_labels})
    result.to_csv(save_path, index=False)
    check_submission_format(save_path, sample_path if sample_path.exists() else None)
    return save_path


def build_test_loader_from_config(cfg):
    """Create test dataframe and DataLoader from config module."""
    return create_test_loader(cfg.TEST_DIR, cfg.IMG_SIZE, cfg.BATCH_SIZE, cfg.NUM_WORKERS, cfg.MEAN, cfg.STD)
