# weather_competition

Weather image classification project for the Zhihai algorithm tuning competition.

## 1. 项目目标

本项目用于“智海算法调优”天气图片分类任务，目标是构建一个稳定、可复现、可在 Mo 平台 Notebook 和 GPU Job 中运行的 PyTorch 训练与推理工程。

项目只做图像分类，不做目标检测、语义分割、图像生成、视频识别或 Web 系统。

## 2. 技术路线

- Framework: PyTorch
- Model library: timm first, torchvision fallback
- Main model: convnext_tiny
- Backup model: tf_efficientnetv2_s
- Fallback model: resnet50 or efficientnet_b0
- Metrics: accuracy, macro_f1, weighted_f1
- Best checkpoint: results/best_model.pth

## 3. 项目目录

```text
weather_competition/
├── data/
│   ├── raw/                  # 手动放置公开数据集 zip，真实数据不提交
│   ├── train/                # 类别文件夹形式训练集，真实数据不提交
│   ├── test/                 # 测试图片，真实数据不提交
│   └── sample_submission.csv
├── results/                  # 模型权重、类别映射、训练摘要，生成物不提交
├── outputs/
│   ├── submissions/          # submission.csv，生成物不提交
│   └── errors/               # 错误样本分析，生成物不提交
├── logs/                     # 训练日志，生成物不提交
├── src/                      # 核心代码模块
├── scripts/                  # 数据准备与 smoke test 脚本
├── configs/                  # 测试配置说明
├── config.py
├── train.py
├── infer.py
├── handler.py
├── app_spec.yml
└── requirements.txt
```

## 4. 安装依赖

```bash
pip install -r requirements.txt
```

如果平台上 `timm` 不可用，代码会尝试回退到 torchvision 模型。

## 5. 数据目录格式

训练集推荐使用类别文件夹形式：

```text
data/train/
├── cloudy/
│   ├── xxx.jpg
│   └── ...
├── rain/
├── shine/
└── sunrise/
```

测试集使用扁平结构：

```text
data/test/
├── image_001.jpg
├── image_002.jpg
└── ...
```

提交样例：

```csv
image_id,label
image_001.jpg,cloudy
image_002.jpg,cloudy
```

## 6. 公开数据集验证测试

### 6.1 测试目的

公开天气数据集只用于验证工程闭环：数据整理、训练、F1 计算、最佳模型保存、推理、submission.csv 生成和 handler.py 单图预测。公开数据集不等于正式比赛数据，最终参赛必须以 Mo 平台正式数据为准。

### 6.2 推荐公开数据集

优先推荐：

1. Kaggle Multi-class Weather Dataset：类别相对简单，适合快速 smoke test。
2. Kaggle Weather Image Recognition：类别更丰富，适合验证模型泛化和混淆矩阵分析。

如果无法使用 Kaggle API，可以在网页手动下载 zip，然后上传到：

```text
data/raw/
```

### 6.3 手动下载后整理数据

将公开数据集 zip 放入：

```text
data/raw/your_weather_dataset.zip
```

运行：

```bash
python scripts/prepare_public_weather_dataset.py --clean
```

脚本会自动：

- 解压 zip；
- 自动识别类别文件夹；
- 过滤非图片和坏图；
- 按类别划分 data/train 与 data/test；
- 生成 data/sample_submission.csv；
- 输出 data/dataset_prepare_report.json。

### 6.4 Kaggle API 下载方式

如本地或 Mo 平台已经配置 Kaggle Token，可参考：

```bash
pip install kaggle
kaggle datasets download -d pratik2901/multiclass-weather-dataset -p data/raw
python scripts/prepare_public_weather_dataset.py --clean
```

若没有 Kaggle Token，请直接手动下载 zip，不要把 `kaggle.json` 提交到 GitHub。

### 6.5 运行 smoke test

```bash
python scripts/smoke_test.py --epochs 1 --batch-size 8 --model-name efficientnet_b0 --fallback-model-name efficientnet_b0
```

smoke test 会临时把 `config.py` 调整为轻量参数，默认 `PRETRAINED=False`，避免因下载预训练权重失败而影响工程验证。运行结束后会自动恢复原始 `config.py`。

输出报告：

```text
outputs/smoke_test_report.json
```

### 6.6 正式训练

公开数据 smoke test 跑通后，可恢复主力方案：

```bash
python train.py
```

训练后应生成：

```text
results/best_model.pth
results/class_to_idx.json
results/idx_to_class.json
results/training_summary.json
logs/train_log.csv
```

### 6.7 推理与提交文件生成

```bash
python infer.py
```

输出：

```text
outputs/submissions/submission.csv
```

提交前必须检查：

- CSV 列名是否与 sample_submission.csv 一致；
- 行数是否与测试集一致；
- 是否存在空值；
- image_id 是否重复；
- 是否多出 Unnamed 索引列；
- 类别映射是否与训练时一致。

## 7. Mo 平台使用建议

- Notebook 只用于短时间调试：依赖检查、路径检查、少量图片读取、小规模训练。
- 长时间训练请使用 GPU Job 运行 `python train.py`。
- 训练结果统一保存到 `results/`。
- 推理输出统一保存到 `outputs/submissions/`。
- 不要写死 Windows 本地绝对路径，例如 `D:/xxx` 或 `C:/xxx`。

## 8. 常见问题

### No module named timm

```bash
pip install timm
```

若平台安装失败，可改用 torchvision fallback 模型。

### CUDA out of memory

降低 `config.py` 中的：

- `BATCH_SIZE`
- `IMG_SIZE`
- 模型大小

### 公开数据 smoke test F1 很低

正常。smoke test 默认 `PRETRAINED=False` 且只训练 1 个 epoch，目标是验证流程，不是追求分数。

### submission.csv 格式错误

运行 `infer.py` 后检查 `outputs/submissions/submission.csv`，确保 `index=False`、列名一致、无空值和重复 ID。

## 9. 后续调参建议

1. 先用 `efficientnet_b0` 或 `resnet50` 验证闭环；
2. 再切换 `convnext_tiny`；
3. 根据类别分布决定是否开启 Class Weight；
4. 根据混淆矩阵分析雨/雪/雾/阴天等易混类别；
5. 最终以 Mo 平台正式数据和平台得分为准。
