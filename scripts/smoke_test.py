"""Run a lightweight train/infer smoke test for JupyterLab or platform setup."""
from __future__ import annotations

import argparse
import csv
import os
import shutil
import subprocess
import sys
from pathlib import Path

from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from configs.smoke_test_config import SMOKE_ENV  # noqa: E402


def _write_synthetic_dataset(root: Path) -> tuple[Path, Path, Path]:
    """Create a tiny deterministic dataset that exercises the full pipeline."""
    if root.exists():
        shutil.rmtree(root)
    colors = {
        "cloudy": (120, 120, 130),
        "rainy": (40, 80, 170),
        "snowy": (230, 235, 245),
        "sunny": (245, 190, 40),
    }
    train_dir = root / "train"
    test_dir = root / "test"
    sample_path = root / "sample_submission.csv"
    image_ids: list[str] = []
    for label, color in colors.items():
        class_dir = train_dir / label
        class_dir.mkdir(parents=True, exist_ok=True)
        for i in range(4):
            image = Image.new("RGB", (96, 96), color)
            draw = ImageDraw.Draw(image)
            fill = (255, 255, 255) if sum(color) < 380 else (0, 0, 0)
            draw.text((8, 36), f"{label[:2]}{i}", fill=fill)
            image.save(class_dir / f"{label}_{i}.jpg")
    test_dir.mkdir(parents=True, exist_ok=True)
    for i, (label, color) in enumerate(colors.items()):
        image_id = f"test_{i}.jpg"
        Image.new("RGB", (96, 96), color).save(test_dir / image_id)
        image_ids.append(image_id)
    with sample_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["image_id", "label"])
        for image_id in image_ids:
            writer.writerow([image_id, "unknown"])
    return train_dir, test_dir, sample_path


def _run(cmd: list[str], env: dict[str, str]) -> None:
    print("\n$", " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=ROOT, env=env, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a fast smoke test.")
    parser.add_argument("--epochs", default=SMOKE_ENV["EPOCHS"])
    parser.add_argument("--batch-size", default=SMOKE_ENV["BATCH_SIZE"])
    parser.add_argument("--img-size", default="64")
    parser.add_argument("--model-name", default=SMOKE_ENV["MODEL_NAME"])
    parser.add_argument("--fallback-model-name", default=SMOKE_ENV["FALLBACK_MODEL_NAME"])
    parser.add_argument("--pretrained", default=SMOKE_ENV["PRETRAINED"])
    parser.add_argument("--num-workers", default=SMOKE_ENV["NUM_WORKERS"])
    parser.add_argument("--use-existing-data", action="store_true", help="Use configured data paths instead of synthetic data.")
    parser.add_argument("--skip-infer", action="store_true", help="Only run training.")
    args = parser.parse_args()

    env = os.environ.copy()
    env.update(SMOKE_ENV)
    env.update({
        "MODEL_NAME": str(args.model_name),
        "FALLBACK_MODEL_NAME": str(args.fallback_model_name),
        "PRETRAINED": str(args.pretrained).lower(),
        "IMG_SIZE": str(args.img_size),
        "BATCH_SIZE": str(args.batch_size),
        "EPOCHS": str(args.epochs),
        "NUM_WORKERS": str(args.num_workers),
    })

    smoke_root = ROOT / "tmp" / "smoke_test"
    if not args.use_existing_data:
        train_dir, test_dir, sample_path = _write_synthetic_dataset(smoke_root / "data")
        env.update({
            "TRAIN_DIR": str(train_dir),
            "TEST_DIR": str(test_dir),
            "SAMPLE_SUBMISSION_PATH": str(sample_path),
            "RESULTS_DIR": str(smoke_root / "results"),
            "OUTPUTS_DIR": str(smoke_root / "outputs"),
            "LOGS_DIR": str(smoke_root / "logs"),
        })

    _run([sys.executable, "train.py"], env)
    if not args.skip_infer:
        _run([sys.executable, "infer.py"], env)
    print("\nSmoke test finished.")
    if not args.use_existing_data:
        print(f"Generated files are under: {smoke_root}")


if __name__ == "__main__":
    main()
