# Mo / JupyterLab Guide

Use `coding_here.ipynb` first when the platform supports notebooks. This guide is the plain-script fallback.

## 1. Check Runtime

```bash
python -c "import sys, torch; print(sys.version); print(torch.__version__); print(torch.cuda.is_available())"
```

If PyTorch is already installed by the platform, prefer the lightweight dependency file:

```bash
pip install -r requirements_jupyter.txt
```

## 2. Locate Platform Data

```bash
python scripts/check_platform_paths.py
```

If the data is not under `data/train` and `data/test`, set:

```python
import os
os.environ["TRAIN_DIR"] = "actual_train_dir"
os.environ["TEST_DIR"] = "actual_test_dir"
os.environ["SAMPLE_SUBMISSION_PATH"] = "actual_sample_submission.csv"
```

## 3. Check Data Reading

```bash
python -c "from src.dataset import scan_image_folder; df=scan_image_folder('data/train'); print(df['label'].value_counts()); print(len(df))"
```

## 4. Run Smoke Test

This does not require official test images:

```bash
python scripts/smoke_test.py --epochs 1 --batch-size 8 --model-name efficientnet_b0 --fallback-model-name efficientnet_b0
```

For Notebook debugging, use:

```python
import os
os.environ["MODEL_NAME"] = "efficientnet_b0"
os.environ["FALLBACK_MODEL_NAME"] = "efficientnet_b0"
os.environ["PRETRAINED"] = "false"
os.environ["EPOCHS"] = "1"
os.environ["BATCH_SIZE"] = "8"
os.environ["NUM_WORKERS"] = "0"
```

## 5. Run Full Training

Use a GPU job for full training:

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

```bash
python train.py
```

The training log must show `device=cuda` and the GPU name. If CUDA is not visible, `REQUIRE_CUDA=true` stops training instead of silently falling back to CPU.

## 6. Run Inference

Add official test images first, then run:

```bash
python infer.py
```

Generated files:

- `results/best_model.pth`
- `results/training_summary.json`
- `logs/train_log.csv`
- `outputs/confusion_matrix.png`
- `outputs/errors/`
- `outputs/submissions/submission.csv`
