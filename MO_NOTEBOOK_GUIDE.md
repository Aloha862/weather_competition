# Mo Notebook Guide

Use this guide when the platform notebook file cannot be created automatically.

1. Check environment:

python -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"

2. Check data folders:

python -c "from pathlib import Path; print(Path('data/train').exists()); print(Path('data/test').exists())"

3. Run training:

python train.py

4. Run inference:

python infer.py

Notebook is only recommended for short checks. Use a GPU Job for full training.
