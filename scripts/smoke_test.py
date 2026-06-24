"""项目工程闭环 smoke test 脚本。

本脚本用于快速验证 weather_competition 项目是否能完成：
1. 读取 data/train 与 data/test；
2. 运行 1-2 epoch 快速训练；
3. 保存 best_model、类别映射和 train_log；
4. 运行 infer.py 生成 submission.csv；
5. 检查提交文件格式；
6. 可选测试 handler.py 单图预测。

注意：
- smoke test 的目标是验证工程闭环，不追求高 F1。
- 脚本会临时修改 config.py 中的训练参数，运行结束后默认恢复原配置。
- 公开数据和生成结果不要提交到 GitHub。
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "config.py"
REPORT_PATH = ROOT / "outputs" / "smoke_test_report.json"


def ensure_dir(path: Path) -> Path:
    """确保目录存在。"""
    path.mkdir(parents=True, exist_ok=True)
    return path


def run_cmd(cmd: List[str], timeout: int) -> Dict[str, Any]:
    """运行命令并捕获输出。"""
    start = time.time()
    try:
        proc = subprocess.run(
            cmd,
            cwd=ROOT,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=timeout,
        )
        return {
            "cmd": " ".join(cmd),
            "returncode": proc.returncode,
            "seconds": round(time.time() - start, 3),
            "stdout_tail": proc.stdout[-4000:],
            "stderr_tail": proc.stderr[-4000:],
            "success": proc.returncode == 0,
        }
    except subprocess.TimeoutExpired as exc:
        return {
            "cmd": " ".join(cmd),
            "returncode": None,
            "seconds": round(time.time() - start, 3),
            "stdout_tail": (exc.stdout or "")[-4000:] if isinstance(exc.stdout, str) else "",
            "stderr_tail": "timeout",
            "success": False,
        }


def check_import(module_name: str) -> Dict[str, Any]:
    """检查依赖是否可以导入。"""
    spec = importlib.util.find_spec(module_name)
    return {"module": module_name, "available": spec is not None}


def replace_config_value(text: str, name: str, value_repr: str) -> str:
    """替换 config.py 中的简单常量配置。"""
    pattern = re.compile(rf"^{name}\s*=\s*.*$", re.MULTILINE)
    replacement = f"{name} = {value_repr}"
    if pattern.search(text):
        return pattern.sub(replacement, text)
    return text + f"\n{replacement}\n"


def patch_config(args: argparse.Namespace) -> str:
    """备份并临时修改 config.py，以便快速 smoke test。"""
    original = CONFIG_PATH.read_text(encoding="utf-8")
    patched = original
    overrides = {
        "MODEL_NAME": repr(args.model_name),
        "FALLBACK_MODEL_NAME": repr(args.fallback_model_name),
        "PRETRAINED": "False",  # 避免 smoke test 因下载预训练权重而失败
        "IMG_SIZE": str(args.img_size),
        "BATCH_SIZE": str(args.batch_size),
        "NUM_WORKERS": str(args.num_workers),
        "EPOCHS": str(args.epochs),
        "USE_AMP": "False",
        "USE_CLASS_WEIGHT": "False",
        "EARLY_STOPPING_PATIENCE": "2",
        "TARGET_METRIC": repr("macro_f1"),
    }
    for key, value in overrides.items():
        patched = replace_config_value(patched, key, value)
    CONFIG_PATH.write_text(patched, encoding="utf-8")
    return original


def restore_config(original: str) -> None:
    """恢复原始 config.py。"""
    CONFIG_PATH.write_text(original, encoding="utf-8")


def check_required_paths() -> Dict[str, bool]:
    """检查 smoke test 必要路径是否存在。"""
    required = {
        "README.md": ROOT / "README.md",
        "requirements.txt": ROOT / "requirements.txt",
        "config.py": ROOT / "config.py",
        "train.py": ROOT / "train.py",
        "infer.py": ROOT / "infer.py",
        "handler.py": ROOT / "handler.py",
        "app_spec.yml": ROOT / "app_spec.yml",
        "src/dataset.py": ROOT / "src" / "dataset.py",
        "src/model.py": ROOT / "src" / "model.py",
        "src/train_utils.py": ROOT / "src" / "train_utils.py",
        "src/metrics.py": ROOT / "src" / "metrics.py",
        "src/inference.py": ROOT / "src" / "inference.py",
        "src/utils.py": ROOT / "src" / "utils.py",
        "data/train": ROOT / "data" / "train",
        "data/test": ROOT / "data" / "test",
        "data/sample_submission.csv": ROOT / "data" / "sample_submission.csv",
    }
    return {name: path.exists() for name, path in required.items()}


def check_generated_files() -> Dict[str, bool]:
    """检查训练和推理生成物是否存在。"""
    generated = {
        "results/best_model.pth": ROOT / "results" / "best_model.pth",
        "results/class_to_idx.json": ROOT / "results" / "class_to_idx.json",
        "results/idx_to_class.json": ROOT / "results" / "idx_to_class.json",
        "results/training_summary.json": ROOT / "results" / "training_summary.json",
        "logs/train_log.csv": ROOT / "logs" / "train_log.csv",
        "outputs/submissions/submission.csv": ROOT / "outputs" / "submissions" / "submission.csv",
    }
    return {name: path.exists() for name, path in generated.items()}


def validate_submission() -> Dict[str, Any]:
    """检查 submission.csv 是否存在空值、重复 ID、多余索引列和行数错误。"""
    submission_path = ROOT / "outputs" / "submissions" / "submission.csv"
    sample_path = ROOT / "data" / "sample_submission.csv"
    result: Dict[str, Any] = {"path": str(submission_path), "exists": submission_path.exists()}
    if not submission_path.exists():
        return result
    df = pd.read_csv(submission_path)
    result.update({
        "rows": int(len(df)),
        "columns": list(df.columns),
        "has_null": bool(df.isnull().any().any()),
        "has_duplicated_id": bool(df.iloc[:, 0].duplicated().any()) if len(df.columns) else None,
        "has_unnamed_index": any(str(col).startswith("Unnamed") for col in df.columns),
    })
    if sample_path.exists():
        sample = pd.read_csv(sample_path)
        result["sample_rows"] = int(len(sample))
        result["row_match_sample"] = len(df) == len(sample)
        result["columns_match_sample"] = list(df.columns) == list(sample.columns)
    return result


def test_handler() -> Dict[str, Any]:
    """使用 data/test 中第一张图片测试 handler.py。"""
    test_images = sorted((ROOT / "data" / "test").rglob("*"))
    test_images = [p for p in test_images if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}]
    if not test_images:
        return {"success": False, "error": "data/test 中没有图片"}
    code = (
        "from handler import handle\n"
        f"print(handle({{'image': r'{str(test_images[0])}'}}))\n"
    )
    return run_cmd([sys.executable, "-c", code], timeout=120)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="运行 weather_competition 工程闭环 smoke test")
    parser.add_argument("--epochs", type=int, default=1, help="快速训练 epoch 数")
    parser.add_argument("--batch-size", type=int, default=8, help="快速训练 batch size")
    parser.add_argument("--img-size", type=int, default=224, help="输入图片尺寸")
    parser.add_argument("--model-name", default="efficientnet_b0", help="smoke test 模型名，默认轻量模型")
    parser.add_argument("--fallback-model-name", default="efficientnet_b0", help="timm 不可用时的 torchvision fallback")
    parser.add_argument("--num-workers", type=int, default=0, help="DataLoader worker 数，平台调试建议 0")
    parser.add_argument("--train-timeout", type=int, default=1800, help="训练超时时间，秒")
    parser.add_argument("--infer-timeout", type=int, default=600, help="推理超时时间，秒")
    parser.add_argument("--keep-config", action="store_true", help="运行结束后不恢复 config.py，仅调试时使用")
    parser.add_argument("--skip-handler", action="store_true", help="跳过 handler.py 单图测试")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ensure_dir(REPORT_PATH.parent)

    report: Dict[str, Any] = {
        "purpose": "verify_weather_competition_engineering_pipeline",
        "note": "smoke test 只验证工程闭环，不代表正式比赛分数。",
        "config": vars(args),
        "dependency_check": [check_import(m) for m in ["torch", "torchvision", "timm", "sklearn", "pandas", "PIL"]],
        "required_paths": check_required_paths(),
        "commands": [],
    }

    missing = [name for name, ok in report["required_paths"].items() if not ok]
    if missing:
        report["success"] = False
        report["error"] = f"必要文件或数据缺失：{missing}"
        REPORT_PATH.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps(report, ensure_ascii=False, indent=2))
        raise SystemExit(1)

    original_config = patch_config(args)
    try:
        train_result = run_cmd([sys.executable, "train.py"], timeout=args.train_timeout)
        report["commands"].append(train_result)

        infer_result = run_cmd([sys.executable, "infer.py"], timeout=args.infer_timeout)
        report["commands"].append(infer_result)

        report["generated_files"] = check_generated_files()
        report["submission_check"] = validate_submission()
        if not args.skip_handler:
            report["handler_test"] = test_handler()

        report["success"] = bool(train_result["success"] and infer_result["success"])
    finally:
        if not args.keep_config:
            restore_config(original_config)
            report["config_restored"] = True
        else:
            report["config_restored"] = False

    REPORT_PATH.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if not report.get("success"):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
