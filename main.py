"""Platform entry point for single-image weather prediction.

Do not generate this file from the training notebook. Competition system tests
should call `handle(data)` for inference only.
"""
from typing import Any, Dict

from handler import handle as _handle


def handle(data: Any) -> Dict[str, Any]:
    """Return predicted weather label and confidence for one image."""
    return _handle(data)


def predict(data: Any) -> Dict[str, Any]:
    """Compatibility alias for platforms that call `predict`."""
    return handle(data)

