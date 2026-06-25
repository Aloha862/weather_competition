# weather_competition

Weather image classification pipeline for the Zhihai / Mo weather recognition competition. The project supports training, validation, checkpoint saving, batch inference, submission CSV generation, and `handler.py` single-image prediction.

## Current Status

- Task: 4-class weather image classification: `cloudy`, `rainy`, `snowy`, `sunny`.
- Training data detected locally: 4999 images.
  - `cloudy`: 2184
  - `rainy`: 446
  - `snowy`: 403
  - `sunny`: 1966
- Local runtime used for verification: CPU PyTorch in `.venv`; CUDA is not available on this machine.
- Full pipeline smoke test passed with a temporary synthetic dataset: train, checkpoint save, inference, and submission CSV validation.
- Formal accuracy still needs a full GPU training run on the real dataset. Use validation `macro_f1` in `results/training_summary.json` as the main acceptance metric.
- `data/test` currently only contains `.gitkeep`; add official test images before final inference.

## JupyterLab First

For platform use, open:

```text
coding_here.ipynb
```

The notebook walks through:

1. Runtime and CUDA check.
2. Project and platform path check.
3. Optional `TRAIN_DIR`, `TEST_DIR`, and `SAMPLE_SUBMISSION_PATH` overrides.
4. Training data scan.
5. Lightweight smoke-test settings with `PRETRAINED=false` and `NUM_WORKERS=0`.
6. Isolated synthetic smoke test through `scripts/smoke_test.py`.
7. Full training settings.
8. Training, inference, and submission inspection.

For JupyterLab dependency installation, prefer:

```bash
pip install -r requirements_jupyter.txt
```

This file intentionally does not reinstall `torch` or `torchvision`, because competition platforms often preinstall CUDA-matched PyTorch builds.

## Project Layout

```text
.
|-- coding_here.ipynb
|-- config.py
|-- train.py
|-- infer.py
|-- handler.py
|-- requirements.txt
|-- requirements_jupyter.txt
|-- configs/
|   `-- smoke_test_config.py
|-- scripts/
|   |-- analyze_errors.py
|   |-- benchmark_inference.py
|   |-- check_platform_paths.py
|   |-- download_public_weather_test.py
|   |-- evaluate_labeled_folder.py
|   |-- prepare_public_weather_dataset.py
|   |-- run_kfold.py
|   |-- run_experiments.py
|   `-- smoke_test.py
|-- src/
|   |-- dataset.py
|   |-- model.py
|   |-- train_utils.py
|   |-- inference.py
|   |-- metrics.py
|   `-- utils.py
|-- data/
|   |-- train/
|   |-- test/
|   `-- sample_submission.csv
|-- results/
|-- outputs/
`-- logs/
```

## Data Format

Preferred structure:

```text
data/train/
|-- cloudy/*.jpg
|-- rainy/*.jpg
|-- snowy/*.jpg
`-- sunny/*.jpg

data/test/*.jpg
```

The loader also supports common nested layouts such as:

```text
data/train/<dataset_name>/train/train/{cloudy,rainy,snowy,sunny}/*.jpg
```

When `data/train` does not directly contain class folders, the scanner recursively finds the real class folders.

If an uploaded dataset is nested, prepare a clean project layout with:

```bash
python scripts/prepare_public_weather_dataset.py --source path/to/unpacked_dataset --output data/train
```

Use `--max-per-class 20` for a quick small copy.

## Install

Local full dependencies:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

JupyterLab lightweight dependencies:

```bash
pip install -r requirements_jupyter.txt
```

Check runtime:

```powershell
.\.venv\Scripts\python.exe -c "import torch, timm; print(torch.__version__); print('cuda=', torch.cuda.is_available()); print(timm.__version__)"
```

## Platform Path Check

Run this first after uploading to a platform:

```bash
python scripts/check_platform_paths.py
```

Then set paths in Notebook or shell when needed:

```python
import os
os.environ["TRAIN_DIR"] = "actual_train_dir"
os.environ["TEST_DIR"] = "actual_test_dir"
os.environ["SAMPLE_SUBMISSION_PATH"] = "actual_sample_submission.csv"
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

- `DEVICE`: default `auto`; set `cuda` for GPU training.
- `REQUIRE_CUDA`: default `false`; set `true` to fail fast if the current runtime is not GPU-enabled.
- `MODEL_NAME`: default `convnext_tiny`; good alternatives include `tf_efficientnetv2_s`, `efficientnet_b0`, `resnet50`, `resnet18`.
- `PRETRAINED`: default `true`; set `false` for offline smoke tests.
- `NUM_WORKERS`: use `0` in Notebook; use `2` or `4` in GPU jobs if stable.
- `USE_CLASS_WEIGHT`: default `false`; try `true` if minority classes have poor recall.
- `AUGMENT_PROFILE`: `light`, `weather_safe`, or `strong`; default `weather_safe`.
- `LOSS_TYPE`: `ce`, `focal`, or `class_balanced_focal`.
- `SAMPLER_TYPE`: `none` or `weighted`; do not combine aggressively with class weights unless validated.
- `USE_EMA`: enable exponential moving average validation/checkpoint weights.
- `USE_WARMUP`: enable linear warmup before cosine decay.
- `USE_FREEZE_BACKBONE`: train classifier head first, then unfreeze the backbone.
- `GRAD_ACCUM_STEPS`: accumulate gradients when GPU memory limits batch size.
- `USE_TTA` and `TTA_MODE=hflip`: optional lightweight test-time augmentation.
- `EXPERIMENT_NAME`: used by experiment scripts to separate logs/results.
- `SAVE_TOP_K`: save more than one top checkpoint.
- `TARGET_METRIC`: default `macro_f1`; keep this for imbalanced classes.
- `RESULTS_DIR`, `OUTPUTS_DIR`, `LOGS_DIR`: redirect generated files for experiments.

## F1 Optimization Route

Current EfficientNetV2-S validation is already strong: accuracy around 0.93, weighted F1 around 0.93, and macro F1 around 0.91. The main bottleneck is not the large classes (`cloudy`, `sunny`) but minority classes (`rainy`, `snowy`). Because the training distribution is imbalanced (`cloudy` and `sunny` are much larger), weighted F1 can look healthy while macro F1 is pulled down by small-class precision/recall.

### What Was Added

- Weather-safe augmentation profiles in `src/dataset.py`.
- `WeightedRandomSampler` via `SAMPLER_TYPE=weighted`.
- CE, focal loss, and class-balanced focal loss via `LOSS_TYPE`.
- EMA model weights via `USE_EMA=true`.
- Warmup + cosine scheduler via `USE_WARMUP=true`.
- Optional backbone freeze and higher classifier-head learning rate.
- Gradient accumulation and gradient clipping.
- Per-class metric CSV, validation prediction CSV, confusion pairs, and rainy/snowy focused error export.
- Optional hflip TTA and inference benchmark JSON.
- Batch experiment runner and error-analysis report script.

### Recommended Experiments

Experiment 0, baseline:

```python
os.environ["MODEL_NAME"] = "tf_efficientnetv2_s"
os.environ["IMG_SIZE"] = "224"
os.environ["LOSS_TYPE"] = "ce"
os.environ["AUGMENT_PROFILE"] = "light"
```

Experiment 1, larger input:

```python
os.environ["MODEL_NAME"] = "tf_efficientnetv2_s"
os.environ["IMG_SIZE"] = "300"  # use 320 if memory allows
```

Experiment 2, weather-safe augmentation:

```python
os.environ["AUGMENT_PROFILE"] = "weather_safe"
```

Experiment 3, imbalance handling:

```python
os.environ["LOSS_TYPE"] = "focal"
os.environ["SAMPLER_TYPE"] = "none"
os.environ["USE_CLASS_WEIGHT"] = "false"
```

Alternative:

```python
os.environ["LOSS_TYPE"] = "ce"
os.environ["SAMPLER_TYPE"] = "weighted"
os.environ["USE_CLASS_WEIGHT"] = "false"
```

Experiment 4, EMA + warmup:

```python
os.environ["USE_EMA"] = "true"
os.environ["USE_WARMUP"] = "true"
os.environ["WARMUP_EPOCHS"] = "3"
```

Experiment 5, model comparison:

```python
os.environ["MODEL_NAME"] = "convnext_tiny"       # balanced speed/effect
os.environ["MODEL_NAME"] = "convnext_small"      # potentially higher F1, slower
os.environ["MODEL_NAME"] = "tf_efficientnetv2_s" # stable high-score candidate
```

Experiment 6, speed benchmark:

```bash
python scripts/benchmark_inference.py --batch-size 32
```

Experiment 7, optional hflip TTA:

```bash
python scripts/benchmark_inference.py --tta hflip --batch-size 32
```

Use hflip TTA only if macro F1 improves enough to justify the extra inference cost.

### Suggested Main Configuration

```python
os.environ["DEVICE"] = "cuda"
os.environ["REQUIRE_CUDA"] = "true"
os.environ["MODEL_NAME"] = "tf_efficientnetv2_s"
os.environ["FALLBACK_MODEL_NAME"] = "efficientnet_b0"
os.environ["IMG_SIZE"] = "300"
os.environ["BATCH_SIZE"] = "8"
os.environ["EPOCHS"] = "40"
os.environ["LEARNING_RATE"] = "2e-4"
os.environ["WEIGHT_DECAY"] = "1e-4"
os.environ["LABEL_SMOOTHING"] = "0.05"
os.environ["AUGMENT_PROFILE"] = "weather_safe"
os.environ["LOSS_TYPE"] = "focal"
os.environ["USE_EMA"] = "true"
os.environ["USE_WARMUP"] = "true"
os.environ["USE_CLASS_WEIGHT"] = "false"
os.environ["SAMPLER_TYPE"] = "none"
os.environ["USE_TTA"] = "false"
```

### How To Judge an Experiment

- Primary metric: `val_macro_f1`.
- Secondary metrics: rainy/snowy F1 and recall in `logs/per_class_metrics.csv`.
- Inspect `outputs/errors/focus_rainy_snowy/` after each serious run.
- Run `python scripts/analyze_errors.py` to generate `outputs/error_analysis_report.md`.
- Compare speed with `results/inference_benchmark.json`.

### Why 98 Is Not Guaranteed

Macro-F1 above 0.95 is a reasonable optimization target, but 0.98 depends on label quality, class boundary ambiguity, data leakage risk, and the hidden test distribution. This repo adds the tools to push toward 95+ while keeping inference speed and implementation complexity under control; it does not claim a guaranteed leaderboard score.

### Batch Experiments

Run predefined experiment grids:

```bash
python scripts/run_experiments.py --preset quick
python scripts/run_experiments.py --preset f1
```

Results are appended to:

```text
logs/experiment_record.csv
```

### Optional K-Fold Stability Check

K-fold is for offline stability validation, not for the default fast Notebook path:

```bash
python scripts/run_kfold.py --folds 5 --experiment-name kfold_e2s_weather
```

This writes per-fold outputs under `results/kfold_e2s_weather/`, `logs/kfold_e2s_weather/`, and `outputs/kfold_e2s_weather/`, plus:

```text
results/kfold_e2s_weather/kfold_summary.json
```

## Smoke Test

Run an isolated smoke test that does not require official test images:

```bash
python scripts/smoke_test.py --epochs 1 --batch-size 8 --model-name efficientnet_b0 --fallback-model-name efficientnet_b0
```

The script creates a tiny temporary dataset under `tmp/smoke_test`, trains for one epoch, runs inference, and validates the generated CSV.

To smoke-test the configured real data paths:

```bash
python scripts/smoke_test.py --use-existing-data --epochs 1 --batch-size 8 --skip-infer
```

## External Public Test Set

For a real labeled external test, this project can prepare the public Hugging Face dataset `davidshableski/weatherimages`.

Source notes:

- The repository README for `DavidShableski/weather-image-classification` says the dataset has about 1000 images with train/test split.
- The reported classes are `sunny`, `rainy`, `cloudy`, `snowy`, and `sunrise`.
- This project keeps only the four overlapping labels: `cloudy`, `rainy`, `snowy`, `sunny`.
- The current `Data.zip` file extracts to a `raw/` folder. If no split folder exists, the script prepares a labeled external set from `raw/`.

Download and prepare the external test set:

```bash
python scripts/download_public_weather_test.py --output-dir tmp/public_weather_test
```

For a balanced quick test, cap each class:

```bash
python scripts/download_public_weather_test.py --output-dir tmp/public_weather_test --max-per-class 50
```

Evaluate the trained checkpoint:

```bash
python scripts/evaluate_labeled_folder.py --data-dir tmp/public_weather_test --output-dir outputs/external_test --batch-size 32 --num-workers 0
```

Generated evaluation files:

- `outputs/external_test/external_test_metrics.json`
- `outputs/external_test/external_test_report.txt`
- `outputs/external_test/external_test_predictions.csv`
- `outputs/external_test/external_test_confusion_matrix.png`

This requires `results/best_model.pth` and `results/idx_to_class.json`, so run training first.

## Quick Checks

Check that real training data is read as 4 classes:

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
.\.venv\Scripts\python.exe -m compileall config.py train.py infer.py handler.py src scripts configs
```

## Train

For full training, use a GPU job when possible:

```powershell
.\.venv\Scripts\python.exe train.py
```

GPU full-training environment:

```python
import os

os.environ["DEVICE"] = "cuda"
os.environ["REQUIRE_CUDA"] = "true"
os.environ["MODEL_NAME"] = "tf_efficientnetv2_s"
os.environ["FALLBACK_MODEL_NAME"] = "efficientnet_b0"
os.environ["PRETRAINED"] = "true"
os.environ["IMG_SIZE"] = "224"
os.environ["BATCH_SIZE"] = "8"
os.environ["EPOCHS"] = "30"
os.environ["NUM_WORKERS"] = "2"
os.environ["USE_AMP"] = "true"
os.environ["USE_CLASS_WEIGHT"] = "true"
```

When training starts, the log should show `device=cuda` and the GPU name. If it shows `device=cpu`, you are not training on GPU. If `REQUIRE_CUDA=true` and CUDA is unavailable, training stops immediately with a clear error.

To switch models, only change `MODEL_NAME` and adjust `BATCH_SIZE` if needed:

```python
# EfficientNetV2-S, stronger but heavier
os.environ["MODEL_NAME"] = "tf_efficientnetv2_s"
os.environ["FALLBACK_MODEL_NAME"] = "efficientnet_b0"
os.environ["BATCH_SIZE"] = "8"

# ConvNeXt Tiny, previous baseline
os.environ["MODEL_NAME"] = "convnext_tiny"
os.environ["FALLBACK_MODEL_NAME"] = "resnet50"
os.environ["BATCH_SIZE"] = "16"
```

Training writes:

- `results/best_model.pth`
- `results/class_to_idx.json`
- `results/idx_to_class.json`
- `results/training_summary.json`
- `logs/train_log.csv`
- `outputs/confusion_matrix.png`
- `outputs/errors/` grouped misclassified validation samples

The best checkpoint is selected by `TARGET_METRIC`, which defaults to `macro_f1`.

For a CPU-only sanity run:

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
4. Review `outputs/errors/` to inspect repeated confusion pairs.
5. If minority classes lag, run another experiment with `USE_CLASS_WEIGHT=true`.
6. Try `tf_efficientnetv2_s` if GPU memory allows.
7. Submit only after validation score is stable across at least one rerun or a stricter split.

A high leaderboard score cannot be guaranteed from code alone. It must be confirmed by real validation results after full training.

## Mo Platform Notes

- Use `coding_here.ipynb` for first-run checks and smoke tests.
- Use `requirements_jupyter.txt` before reinstalling full dependencies.
- Use `scripts/check_platform_paths.py` to locate uploaded datasets.
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

- `data/test` contains the official test images or `TEST_DIR` points to them.
- `results/best_model.pth` exists and matches the intended experiment.
- `results/idx_to_class.json` contains the expected weather labels.
- CSV columns match `data/sample_submission.csv`.
- Row count matches the official sample submission.
- No empty labels.
- No duplicated image ids.
- Validation `macro_f1` is high enough for the competition target.
