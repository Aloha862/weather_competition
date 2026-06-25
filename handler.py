"""Mo platform handler for single-image prediction.

The exact input shape may differ by platform test case. This handler accepts a
string path, a dict with key 'image' or 'image_path', or a PIL image object.
"""
from pathlib import Path
from typing import Any, Dict

import torch
from PIL import Image

import config as cfg
from src.dataset import build_transforms
from src.model import build_model
from src.utils import get_device, load_checkpoint, load_checkpoint_payload, load_json

_MODEL = None
_TRANSFORM = None
_IDX_TO_CLASS = None
_DEVICE = None


def _load_runtime():
    """Load model once and reuse it for later calls."""
    global _MODEL, _TRANSFORM, _IDX_TO_CLASS, _DEVICE
    if _MODEL is not None:
        return _MODEL, _TRANSFORM, _IDX_TO_CLASS, _DEVICE
    _DEVICE = get_device()
    _IDX_TO_CLASS = load_json(cfg.IDX_TO_CLASS_PATH)
    num_classes = len(_IDX_TO_CLASS)
    checkpoint = load_checkpoint_payload(cfg.BEST_MODEL_PATH, _DEVICE)
    model_name = checkpoint.get("model_name", cfg.MODEL_NAME)
    _MODEL, _ = build_model(model_name, num_classes, pretrained=False, fallback_model_name=cfg.FALLBACK_MODEL_NAME)
    _MODEL = _MODEL.to(_DEVICE)
    load_checkpoint(_MODEL, cfg.BEST_MODEL_PATH, _DEVICE)
    _MODEL.eval()
    _TRANSFORM = build_transforms(cfg.IMG_SIZE, is_train=False, mean=cfg.MEAN, std=cfg.STD)
    return _MODEL, _TRANSFORM, _IDX_TO_CLASS, _DEVICE


def _read_image(data: Any) -> Image.Image:
    """Read image from several common input formats."""
    if isinstance(data, dict):
        data = data.get("image") or data.get("image_path") or data.get("path")
    if isinstance(data, Image.Image):
        return data.convert("RGB")
    if isinstance(data, (str, Path)):
        return Image.open(data).convert("RGB")
    raise ValueError("Unsupported input. Expected image path, PIL image, or dict with image path.")


def handle(data: Any) -> Dict[str, Any]:
    """Return predicted label and confidence."""
    try:
        model, transform, idx_to_class, device = _load_runtime()
        image = _read_image(data)
        tensor = transform(image).unsqueeze(0).to(device)
        with torch.inference_mode():
            logits = model(tensor)
            probs = torch.softmax(logits, dim=1)
            confidence, pred_idx = probs.max(dim=1)
        label = idx_to_class.get(str(int(pred_idx.item())), str(int(pred_idx.item())))
        return {"label": label, "confidence": float(confidence.item())}
    except Exception as exc:
        return {"error": str(exc)}
