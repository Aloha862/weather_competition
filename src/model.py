"""Model builder with timm first and torchvision fallback."""
from __future__ import annotations

import warnings
from typing import Tuple

import torch
from torch import nn

from src.utils import count_parameters


def _build_torchvision_model(model_name: str, num_classes: int, pretrained: bool) -> nn.Module:
    """Build a torchvision fallback classifier."""
    from torchvision import models

    name = model_name.lower()
    if name in {"resnet50", "resnet"}:
        weights = models.ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
        model = models.resnet50(weights=weights)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        return model
    if name in {"efficientnet_b0", "efficientnet-b0"}:
        weights = models.EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None
        model = models.efficientnet_b0(weights=weights)
        in_features = model.classifier[-1].in_features
        model.classifier[-1] = nn.Linear(in_features, num_classes)
        return model
    raise ValueError(f"Unsupported torchvision fallback model: {model_name}")


def build_model(model_name: str, num_classes: int, pretrained: bool = True,
                fallback_model_name: str = "resnet50") -> Tuple[nn.Module, str]:
    """Build a classifier and return model plus actual model name."""
    if num_classes <= 0:
        raise ValueError("num_classes must be positive.")
    try:
        import timm
        model = timm.create_model(model_name, pretrained=pretrained, num_classes=num_classes)
        actual_name = model_name
    except Exception as exc:
        warnings.warn(f"timm model build failed: {exc}. Falling back to torchvision {fallback_model_name}.")
        model = _build_torchvision_model(fallback_model_name, num_classes, pretrained)
        actual_name = fallback_model_name
    params = count_parameters(model)
    print(f"Model: {actual_name}, trainable parameters: {params:,}")
    return model, actual_name
