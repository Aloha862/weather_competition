"""Smoke test 推荐配置。

该文件用于说明快速验证项目闭环时应采用的轻量参数。
scripts/smoke_test.py 会临时把这些参数写入 config.py，运行结束后默认恢复原配置。
"""

SMOKE_TEST_OVERRIDES = {
    "MODEL_NAME": "efficientnet_b0",
    "FALLBACK_MODEL_NAME": "efficientnet_b0",
    "PRETRAINED": False,
    "IMG_SIZE": 224,
    "BATCH_SIZE": 8,
    "NUM_WORKERS": 0,
    "EPOCHS": 1,
    "USE_AMP": False,
    "USE_CLASS_WEIGHT": False,
    "EARLY_STOPPING_PATIENCE": 2,
    "TARGET_METRIC": "macro_f1",
}
