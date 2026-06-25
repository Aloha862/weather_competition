import os
from pathlib import Path

def _get_env(name, default, cast=str):
    value = os.getenv(name)
    if value is None or value == "":
        return default
    if cast is bool:
        return value.strip().lower() in {"1", "true", "yes", "y", "on"}
    return cast(value)


PROJECT_ROOT = Path(os.getenv('PROJECT_ROOT', '.'))
DATA_DIR = Path(os.getenv('DATA_DIR', 'data'))
TRAIN_DIR = Path(os.getenv('TRAIN_DIR', 'data/train'))
TEST_DIR = Path(os.getenv('TEST_DIR', 'data/test'))
SAMPLE_SUBMISSION_PATH = Path(os.getenv('SAMPLE_SUBMISSION_PATH', 'data/sample_submission.csv'))
DEVICE = os.getenv('DEVICE', 'auto')
REQUIRE_CUDA = _get_env('REQUIRE_CUDA', False, bool)

RESULTS_DIR = Path(os.getenv('RESULTS_DIR', 'results'))
OUTPUTS_DIR = Path(os.getenv('OUTPUTS_DIR', 'outputs'))
SUBMISSIONS_DIR = Path(os.getenv('SUBMISSIONS_DIR', str(OUTPUTS_DIR / 'submissions')))
ERRORS_DIR = Path(os.getenv('ERRORS_DIR', str(OUTPUTS_DIR / 'errors')))
LOGS_DIR = Path(os.getenv('LOGS_DIR', 'logs'))

BEST_MODEL_PATH = Path(os.getenv('BEST_MODEL_PATH', str(RESULTS_DIR / 'best_model.pth')))
CLASS_TO_IDX_PATH = Path(os.getenv('CLASS_TO_IDX_PATH', str(RESULTS_DIR / 'class_to_idx.json')))
IDX_TO_CLASS_PATH = Path(os.getenv('IDX_TO_CLASS_PATH', str(RESULTS_DIR / 'idx_to_class.json')))
TRAINING_SUMMARY_PATH = Path(os.getenv('TRAINING_SUMMARY_PATH', str(RESULTS_DIR / 'training_summary.json')))
TRAIN_LOG_PATH = Path(os.getenv('TRAIN_LOG_PATH', str(LOGS_DIR / 'train_log.csv')))
SUBMISSION_PATH = Path(os.getenv('SUBMISSION_PATH', str(SUBMISSIONS_DIR / 'submission.csv')))
CONFUSION_MATRIX_PATH = Path(os.getenv('CONFUSION_MATRIX_PATH', str(OUTPUTS_DIR / 'confusion_matrix.png')))
TRAINING_CURVES_PATH = Path(os.getenv('TRAINING_CURVES_PATH', str(OUTPUTS_DIR / 'training_curves.png')))

MODEL_NAME = os.getenv('MODEL_NAME', 'convnext_tiny')
FALLBACK_MODEL_NAME = os.getenv('FALLBACK_MODEL_NAME', 'resnet50')
PRETRAINED = _get_env('PRETRAINED', True, bool)
IMG_SIZE = _get_env('IMG_SIZE', 224, int)

SEED = _get_env('SEED', 42, int)
BATCH_SIZE = _get_env('BATCH_SIZE', 32, int)
NUM_WORKERS = _get_env('NUM_WORKERS', 2, int)
EPOCHS = _get_env('EPOCHS', 30, int)
VAL_RATIO = _get_env('VAL_RATIO', 0.2, float)
LEARNING_RATE = _get_env('LEARNING_RATE', 3e-4, float)
WEIGHT_DECAY = _get_env('WEIGHT_DECAY', 1e-4, float)
MIN_LR = _get_env('MIN_LR', 1e-6, float)
LABEL_SMOOTHING = _get_env('LABEL_SMOOTHING', 0.1, float)
USE_AMP = _get_env('USE_AMP', True, bool)
USE_CLASS_WEIGHT = _get_env('USE_CLASS_WEIGHT', False, bool)
EARLY_STOPPING_PATIENCE = _get_env('EARLY_STOPPING_PATIENCE', 6, int)
TARGET_METRIC = os.getenv('TARGET_METRIC', 'macro_f1')

MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)

RANDAUGMENT_NUM_OPS = 2
RANDAUGMENT_MAGNITUDE = 7
RANDOM_ERASING_P = 0.15

ID_COLUMN_CANDIDATES = ('image_id', 'filename', 'file', 'id', 'name')
LABEL_COLUMN_CANDIDATES = ('label', 'class', 'category', 'weather')
