"""Default settings for lightweight JupyterLab smoke tests."""

SMOKE_ENV = {
    "MODEL_NAME": "efficientnet_b0",
    "FALLBACK_MODEL_NAME": "efficientnet_b0",
    "PRETRAINED": "false",
    "IMG_SIZE": "224",
    "BATCH_SIZE": "8",
    "EPOCHS": "1",
    "NUM_WORKERS": "0",
}

