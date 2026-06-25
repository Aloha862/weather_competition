# weather_competition

Weather image classification pipeline for the Zhihai / Mo weather recognition competition. The project trains a PyTorch image classifier, saves the best validation checkpoint, and generates a submission CSV for test images.

## Current Status

- Task: 4-class weather image classification: `cloudy`, `rainy`, `snowy`, `sunny`.
- Training data detected locally: 4999 images.
  - `cloudy`: 2184
  - `rainy`: 446
  - `snowy`: 403
  - `sunny`: 1966
- Current local runtime: `.venv` has CPU PyTorch installed; CUDA is not available in this machine.
- Training pipeline smoke-tested end to end with a temporary dataset: train, checkpoint save, inference, and CSV format validation all pass.
- Formal accuracy is not yet certified because full GPU training on the real dataset has not been run in this session. Use the validation `macro_f1` in `results/training_summary.json` as the primary acceptance metric.
- `data/test` currently only contains `.gitkeep`; add the real competition test images before running final inference.

## Project Layout

```text
.
├── config.py                  # paths, model, training hyperparameters
├── train.py                   # training entry
├── infer.py                   # batch inference / submission entry
├── handler.py                 # Mo platform single-image handler
├── src/
│   ├── dataset.py             # data scan, transforms, dataloaders
│   ├── model.py               # timm model builder with torchvision fallback
│   ├── train_utils.py         # optimizer, scheduler, train/valid loops
│   ├── inference.py           # prediction and submission generation
│   ├── metrics.py             # accuracy, F1, confusion matrix
│   └── utils.py               # IO, checkpoints, submission checks
├── data/
│   ├── train/                 # training data
│   ├── test/                  # test images for final submission
│   └── sample_submission.csv  # expected CSV columns
├── results/                   # checkpoints and class mappings
├── outputs/                   # submissions, figures, error samples
└── logs/                      # CSV training logs
```

## Data Format

The preferred structure is:

```text
data/train/
├── cloudy/*.jpg
├── rainy/*.jpg
├── snowy/*.jpg
└── sunny/*.jpg

data/test/*.jpg
```

The loader also supports the currently unpacked nested layout:

```text
data/train/天气识别/train/train/{cloudy,rainy,snowy,sunny}/*.jpg
```

The scanner finds the real class folders recursively when `data/train` does not directly contain class folders.

## Install

Use the existing virtual environment when working locally:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

Or with an activated environment:

```powershell
pip install -r requirements.txt
```

Check the runtime:

```powershell
.\.venv\Scripts\python.exe -c "import torch, timm; print(torch.__version__); print('cuda=', torch.cuda.is_available()); print(timm.__version__)"
```

## Configuration

Defaults live in `config.py`. Most values can be overridden without editing code:

```powershell
$env:MODEL_NAME="convnext_tiny"
$env:EPOCHS="30"
$env:BATCH_SIZE="32"
$env:IMG_SIZE="224"
$env:TRAIN_DIR="data/train"
$env:TEST_DIR="data/test"
```

Useful overrides:

- `MODEL_NAME`: default `convnext_tiny`; good alternatives include `tf_efficientnetv2_s`, `efficientnet_b0`, `resnet50`, `resnet18`.
- `PRETRAINED`: default `true`; set `false` for offline smoke tests.
- `USE_CLASS_WEIGHT`: default `false`; try `true` if minority classes have poor recall.
- `TARGET_METRIC`: default `macro_f1`; keep this for imbalanced classes.
- `RESULTS_DIR`, `OUTPUTS_DIR`, `LOGS_DIR`: redirect generated files for experiments.

## Quick Checks

Check that the real training data is read as 4 classes:

```powershell
.\.venv\Scripts\python.exe -c "from src.dataset import scan_image_folder; df=scan_image_folder('data/train'); print(df['label'].value_counts().sort_index()); print('total=', len(df))"
```

Expected local result:

```text
cloudy    2184
rainy      446
snowy      403
sunny     1966
total=4999
```

Compile-check the code:

```powershell
.\.venv\Scripts\python.exe -m compileall config.py train.py infer.py handler.py src
```

## Train

For full training, use a GPU job when possible:

```powershell
.\.venv\Scripts\python.exe train.py
```

Training writes:

- `results/best_model.pth`
- `results/class_to_idx.json`
- `results/idx_to_class.json`
- `results/training_summary.json`
- `logs/train_log.csv`
- `outputs/confusion_matrix.png`

The best checkpoint is selected by `TARGET_METRIC`, which defaults to `macro_f1`.

For a CPU-only sanity run, lower the workload:

```powershell
$env:MODEL_NAME="resnet18"
$env:PRETRAINED="false"
$env:IMG_SIZE="96"
$env:BATCH_SIZE="8"
$env:EPOCHS="1"
$env:NUM_WORKERS="0"
.\.venv\Scripts\python.exe train.py
```

## Inference

Add official test images under `data/test`, then run:

```powershell
.\.venv\Scripts\python.exe infer.py
```

The generated file is:

```text
outputs/submissions/submission.csv
```

The inference code loads the model architecture recorded inside `results/best_model.pth`, so it remains consistent even if training used a fallback model.

## Accuracy Plan

Use this project as the strong baseline:

1. Start with `convnext_tiny`, `PRETRAINED=true`, `IMG_SIZE=224`, `EPOCHS=30`.
2. Track `val_macro_f1`, `val_weighted_f1`, and `val_accuracy` in `logs/train_log.csv`.
3. Check minority-class recall for `rainy` and `snowy` in the classification report.
4. If minority classes lag, run another experiment with `USE_CLASS_WEIGHT=true`.
5. Try `tf_efficientnetv2_s` if GPU memory allows.
6. Submit only after the validation score is stable across at least one rerun or a stricter split.

A high leaderboard score cannot be guaranteed from code alone. It must be confirmed by real validation results after full training.

## Mo Platform Notes

- Use Notebook for data checks and smoke tests.
- Use GPU Job for full training.
- Keep generated models and summaries under `results/`.
- Keep generated submissions under `outputs/submissions/`.
- Do not hard-code local absolute paths.
- `handler.py` exposes `handle(data)` for single-image prediction and returns:

```json
{"label": "sunny", "confidence": 0.98}
```

## Submission Checklist

Before uploading `outputs/submissions/submission.csv`:

- `data/test` contains the official test images.
- `results/best_model.pth` exists and matches the intended experiment.
- `results/idx_to_class.json` contains the expected weather labels.
- CSV columns match `data/sample_submission.csv`.
- Row count matches the official sample submission.
- No empty labels.
- No duplicated image ids.
- Validation `macro_f1` is high enough for the competition target.
