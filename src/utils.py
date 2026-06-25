"""Utility functions used by training, inference and platform handler."""
import json
import os
import random
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Union

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


def ensure_dir(path: Union[Path, str]) -> Path:
    """Create a directory if it does not exist."""
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def ensure_dirs(paths: Iterable[Union[Path, str]]) -> None:
    """Create multiple directories."""
    for path in paths:
        ensure_dir(path)


def save_json(obj: Dict[str, Any], path: Union[Path, str]) -> None:
    """Save a Python dict as UTF-8 JSON."""
    path = Path(path)
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def load_json(path: Union[Path, str]) -> Dict[str, Any]:
    """Load a UTF-8 JSON file."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"JSON file not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def check_path_exists(path: Union[Path, str], name: str = "path") -> Path:
    """Check that a file or directory exists."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"{name} does not exist: {path}")
    return path


def count_parameters(model: torch.nn.Module) -> int:
    """Count trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def get_device() -> torch.device:
    """Return the requested training device.

    Environment variables:
    - DEVICE=auto|cpu|cuda|cuda:0
    - REQUIRE_CUDA=true fails fast when CUDA is unavailable.
    """
    requested = os.getenv("DEVICE", "auto").strip().lower()
    require_cuda = os.getenv("REQUIRE_CUDA", "").strip().lower() in {"1", "true", "yes", "y", "on"}
    if requested in {"auto", ""}:
        if torch.cuda.is_available():
            return torch.device("cuda")
        if require_cuda:
            raise RuntimeError(
                "REQUIRE_CUDA=true but CUDA is not available. "
                "Use a GPU runtime or install a CUDA-enabled PyTorch build."
            )
        return torch.device("cpu")
    if requested == "cpu":
        if require_cuda:
            raise RuntimeError("REQUIRE_CUDA=true conflicts with DEVICE=cpu.")
        return torch.device("cpu")
    if requested.startswith("cuda"):
        if not torch.cuda.is_available():
            raise RuntimeError(
                f"DEVICE={requested} was requested, but CUDA is not available. "
                "Current PyTorch may be CPU-only, or this machine has no visible NVIDIA GPU."
            )
        return torch.device(requested)
    raise ValueError(f"Unsupported DEVICE value: {requested}. Use auto, cpu, cuda, or cuda:0.")


def save_checkpoint(model: torch.nn.Module, optimizer: Optional[torch.optim.Optimizer], epoch: int,
                    best_score: float, path: Union[Path, str], extra: Optional[Dict[str, Any]] = None) -> None:
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


def load_checkpoint_payload(path: Union[Path, str], device: torch.device) -> Dict[str, Any]:
    """Load a checkpoint file and return its payload."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    return torch.load(path, map_location=device)


def load_checkpoint(model: torch.nn.Module, path: Union[Path, str], device: torch.device) -> Dict[str, Any]:
    """Load checkpoint into a model and return checkpoint metadata."""
    checkpoint = load_checkpoint_payload(path, device)
    state_dict = checkpoint.get("model_state_dict", checkpoint)
    model.load_state_dict(state_dict)
    return checkpoint


def check_submission_format(submission_path: Union[Path, str],
                            sample_path: Optional[Union[Path, str]] = None) -> None:
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
