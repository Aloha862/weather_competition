"""Platform entry point for single-image weather prediction."""
from typing import Any, Dict, Union

from handler import handle as _handle
from handler import predict as _predict


def handle(data: Any) -> Dict[str, Any]:
    """Return predicted weather label and confidence for one image."""
    return _handle(data)


def predict(data: Any) -> Union[str, int]:
    """Return only the class label for scoring scripts."""
    return _predict(data)
