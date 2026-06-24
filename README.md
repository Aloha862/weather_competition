# weather_competition

Weather image classification project for an AI competition.

## Goal

Build a PyTorch training and inference pipeline that can run in Mo platform notebooks and GPU jobs.

## Main files

- config.py: project configuration
- train.py: training entry
- infer.py: inference entry
- src/: core modules
- results/: model and mappings
- outputs/: predictions and reports
- logs/: training logs

## Quick start

pip install -r requirements.txt
python train.py
python infer.py
