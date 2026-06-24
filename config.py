from pathlib import Path

PROJECT_ROOT = Path('.')
DATA_DIR = Path('data')
TRAIN_DIR = Path('data/train')
TEST_DIR = Path('data/test')
SAMPLE_SUBMISSION_PATH = Path('data/sample_submission.csv')

RESULTS_DIR = Path('results')
OUTPUTS_DIR = Path('outputs')
SUBMISSIONS_DIR = Path('outputs/submissions')
ERRORS_DIR = Path('outputs/errors')
LOGS_DIR = Path('logs')

BEST_MODEL_PATH = Path('results/best_model.pth')
CLASS_TO_IDX_PATH = Path('results/class_to_idx.json')
IDX_TO_CLASS_PATH = Path('results/idx_to_class.json')
TRAINING_SUMMARY_PATH = Path('results/training_summary.json')
TRAIN_LOG_PATH = Path('logs/train_log.csv')
SUBMISSION_PATH = Path('outputs/submissions/submission.csv')
CONFUSION_MATRIX_PATH = Path('outputs/confusion_matrix.png')

MODEL_NAME = 'convnext_tiny'
FALLBACK_MODEL_NAME = 'resnet50'
PRETRAINED = True
IMG_SIZE = 224

SEED = 42
BATCH_SIZE = 32
NUM_WORKERS = 2
EPOCHS = 30
VAL_RATIO = 0.2
LEARNING_RATE = 3e-4
WEIGHT_DECAY = 1e-4
MIN_LR = 1e-6
LABEL_SMOOTHING = 0.1
USE_AMP = True
USE_CLASS_WEIGHT = False
EARLY_STOPPING_PATIENCE = 6
TARGET_METRIC = 'macro_f1'

MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)

RANDAUGMENT_NUM_OPS = 2
RANDAUGMENT_MAGNITUDE = 7
RANDOM_ERASING_P = 0.15

ID_COLUMN_CANDIDATES = ('image_id', 'filename', 'file', 'id', 'name')
LABEL_COLUMN_CANDIDATES = ('label', 'class', 'category', 'weather')
