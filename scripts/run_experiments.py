"""Run a small grid of F1 optimization experiments.

Examples:
    python scripts/run_experiments.py --preset quick
    python scripts/run_experiments.py --preset f1
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
from pathlib import Path


PRESETS = {
    "quick": [
        {"EXPERIMENT_NAME": "quick_effb0_light", "MODEL_NAME": "efficientnet_b0", "IMG_SIZE": "224", "EPOCHS": "3", "AUGMENT_PROFILE": "light", "LOSS_TYPE": "ce"},
    ],
    "f1": [
        {"EXPERIMENT_NAME": "e2s_224_ce_light", "MODEL_NAME": "tf_efficientnetv2_s", "IMG_SIZE": "224", "AUGMENT_PROFILE": "light", "LOSS_TYPE": "ce"},
        {"EXPERIMENT_NAME": "e2s_300_weather_focal", "MODEL_NAME": "tf_efficientnetv2_s", "IMG_SIZE": "300", "AUGMENT_PROFILE": "weather_safe", "LOSS_TYPE": "focal", "USE_EMA": "true", "USE_WARMUP": "true"},
        {"EXPERIMENT_NAME": "convnext_tiny_weather_focal", "MODEL_NAME": "convnext_tiny", "IMG_SIZE": "224", "AUGMENT_PROFILE": "weather_safe", "LOSS_TYPE": "focal", "USE_EMA": "true"},
        {"EXPERIMENT_NAME": "convnext_small_weather_focal", "MODEL_NAME": "convnext_small", "IMG_SIZE": "300", "AUGMENT_PROFILE": "weather_safe", "LOSS_TYPE": "focal", "USE_EMA": "true"},
    ],
}


FIELDS = [
    "experiment_name", "model_name", "img_size", "augment_profile", "loss_type", "sampler_type",
    "use_ema", "best_accuracy", "best_macro_f1", "best_weighted_f1",
    "inference_time_per_image", "model_size_mb", "notes",
]


def _append_record(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()
    with path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        if not exists:
            writer.writeheader()
        writer.writerow({k: row.get(k, "") for k in FIELDS})


def main() -> None:
    parser = argparse.ArgumentParser(description="Run experiment presets and record metrics.")
    parser.add_argument("--preset", choices=sorted(PRESETS), default="quick")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    record_path = Path("logs/experiment_record.csv")
    for exp in PRESETS[args.preset]:
        env = os.environ.copy()
        env.update(exp)
        env.setdefault("PRETRAINED", "true")
        env.setdefault("BATCH_SIZE", "8")
        env.setdefault("NUM_WORKERS", "2")
        env.setdefault("USE_AMP", "true")
        env.setdefault("SAMPLER_TYPE", "none")
        name = env["EXPERIMENT_NAME"]
        env["RESULTS_DIR"] = f"results/{name}"
        env["OUTPUTS_DIR"] = f"outputs/{name}"
        env["LOGS_DIR"] = f"logs/{name}"
        print(f"\n== {name} ==")
        if args.dry_run:
            print(exp)
            continue
        subprocess.run([sys.executable, "train.py"], env=env, check=True)
        summary_path = Path(env["RESULTS_DIR"]) / "training_summary.json"
        summary = json.loads(summary_path.read_text(encoding="utf-8")) if summary_path.exists() else {}
        bench_path = Path(env["RESULTS_DIR"]) / "inference_benchmark.json"
        bench = json.loads(bench_path.read_text(encoding="utf-8")) if bench_path.exists() else {}
        _append_record(record_path, {
            "experiment_name": name,
            "model_name": exp.get("MODEL_NAME"),
            "img_size": exp.get("IMG_SIZE"),
            "augment_profile": exp.get("AUGMENT_PROFILE"),
            "loss_type": exp.get("LOSS_TYPE"),
            "sampler_type": env.get("SAMPLER_TYPE"),
            "use_ema": env.get("USE_EMA", "false"),
            "best_accuracy": summary.get("best_accuracy"),
            "best_macro_f1": summary.get("best_macro_f1"),
            "best_weighted_f1": summary.get("best_weighted_f1"),
            "inference_time_per_image": bench.get("time_per_image_ms"),
            "model_size_mb": bench.get("model_size_mb"),
            "notes": "",
        })


if __name__ == "__main__":
    main()

