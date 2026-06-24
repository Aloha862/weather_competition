# weather_competition

Weather image classification project for the Zhihai algorithm tuning competition.

## Goal

Build a stable PyTorch training and inference pipeline that can run in Mo platform notebooks and GPU jobs. The project focuses on image classification only.

## Main route

- Framework: PyTorch
- Model library: timm first, torchvision fallback
- Main model: convnext_tiny
- Backup model: tf_efficientnetv2_s
- Fallback model: resnet50 or efficientnet_b0
- Metrics: accuracy, macro_f1, weighted_f1
- Best checkpoint: results/best_model.pth

## Directory

- data/train: folder-style training data, class_name/image files
- data/test: test images
- data/sample_submission.csv: sample CSV columns
- results: model checkpoint and class mappings
- outputs/submissions: generated CSV files
- outputs/errors: copied error samples
- logs: training CSV logs
- src: core modules

## Install

pip install -r requirements.txt

If timm is not available on the platform, the code falls back to torchvision models.

## Train

python train.py

The training script scans data/train, builds class mappings, splits train and validation data, trains the model, logs loss and F1 scores, and saves the best model according to the target F1 metric.

## Inference

python infer.py

The inference script loads results/best_model.pth and generates outputs/submissions/submission.csv.

## Mo platform notes

Use Notebook for short checks only. Use GPU Job for full training. Keep all generated models, mappings and summaries under results. Do not hard-code local absolute paths.

## Submission checks

Before uploading a CSV file, check column names, row count, empty values, duplicated image ids and class mapping consistency.
