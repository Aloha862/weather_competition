"""Utility functions used by training, inference and platform handler."""
import json
import os
import random
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

import numpy as np
import pandas as pd
import torch


def set_seed(seed: int = 42) -> None:
    """Fix random seeds for reproducible experiments."""
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def ensure_dir(path: Path | str) -> Path:
    """Create a directory if it does not exist."""
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def ensure_dirs(paths: Iterable[Path | str]) -> None:
    """Create multiple directories."""
    for path in paths:
        ensure_dir(path)


def save_json(obj: Dict[str, Any], path: Path | str) -> None:
    """Save a Python dict as UTF-8 JSON."""
    path = Path(path)
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def load_json(path: Path | str) -> Dict[str, Any]:
    """Load a UTF-8 JSON file."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"JSON file not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def check_path_exists(path: Path | str, name: str = "path") -> Path:
    """Check that a file or directory exists."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"{name} does not exist: {path}")
    return path


def count_parameters(model: torch.nn.Module) -> int:
    """Count trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def get_device() -> torch.device:
    """Return cuda device when available, otherwise cpu."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def save_checkpoint(model: torch.nn.Module, optimizer: Optional[torch.optim.Optimizer], epoch: int,
                    best_score: float, path: Path | str, extra: Optional[Dict[str, Any]] = None) -> None:
    """Save model checkpoint to results directory."""
    path = Path(path)
    ensure_dir(path.parent)
    payload: Dict[str, Any] = {
        "model_state_dict": model.state_dict(),
        "epoch": epoch,
        "best_score": best_score,
    }
    if optimizer is not None:
        payload["optimizer_state_dict"] = optimizer.state_dict()
    if extra:
        payload.update(extra)
    torch.save(payload, path)


def load_checkpoint(model: torch.nn.Module, path: Path | str, device: torch.device) -> Dict[str, Any]:
    """Load checkpoint into a model and return checkpoint metadata."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    checkpoint = torch.load(path, map_location=device)
    state_dict = checkpoint.get("model_state_dict", checkpoint)
    model.load_state_dict(state_dict)
    return checkpoint


def check_submission_format(submission_path: Path | str, sample_path: Path | str | None = None) -> None:
    """Validate basic submission CSV format before upload."""
    submission_path = Path(submission_path)
    if not submission_path.exists():
        raise FileNotFoundError(f"Submission file not found: {submission_path}")
    df = pd.read_csv(submission_path)
    if df.empty:
        raise ValueError("Submission file is empty.")
    if df.isnull().any().any():
        raise ValueError("Submission file contains empty values.")
    if len(df.columns) < 2:
        raise ValueError("Submission file must contain at least two columns.")
    if df.iloc[:, 0].duplicated().any():
        raise ValueError("Submission file contains duplicated image ids.")
    if sample_path is not None and Path(sample_path).exists():
        sample = pd.read_csv(sample_path)
        if list(df.columns) != list(sample.columns):
            raise ValueError(f"Column mismatch. expected={list(sample.columns)}, got={list(df.columns)}")
        if len(df) != len(sample):
            raise ValueError(f"Row mismatch. expected={len(sample)}, got={len(df)}")
