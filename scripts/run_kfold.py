"""Run optional K-fold validation.

Example:
    python scripts/run_kfold.py --folds 5
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.utils import save_json


def main() -> None:
    parser = argparse.ArgumentParser(description="Run K-fold training and summarize metrics.")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--experiment-name", default="kfold")
    args = parser.parse_args()

    rows = []
    for fold in range(args.folds):
        env = os.environ.copy()
        env["USE_KFOLD"] = "true"
        env["NUM_FOLDS"] = str(args.folds)
        env["FOLD_INDEX"] = str(fold)
        env["EXPERIMENT_NAME"] = f"{args.experiment_name}_fold{fold}"
        env["RESULTS_DIR"] = f"results/{args.experiment_name}/fold{fold}"
        env["OUTPUTS_DIR"] = f"outputs/{args.experiment_name}/fold{fold}"
        env["LOGS_DIR"] = f"logs/{args.experiment_name}/fold{fold}"
        print(f"\n== Fold {fold + 1}/{args.folds} ==")
        subprocess.run([sys.executable, "train.py"], env=env, check=True)
        summary_path = Path(env["RESULTS_DIR"]) / "training_summary.json"
        rows.append(json.loads(summary_path.read_text(encoding="utf-8")))

    macro = [float(row["best_macro_f1"]) for row in rows if row.get("best_macro_f1") is not None]
    weighted = [float(row["best_weighted_f1"]) for row in rows if row.get("best_weighted_f1") is not None]
    summary = {
        "num_folds": args.folds,
        "folds": rows,
        "macro_f1_mean": statistics.mean(macro) if macro else None,
        "macro_f1_std": statistics.pstdev(macro) if len(macro) > 1 else 0.0,
        "weighted_f1_mean": statistics.mean(weighted) if weighted else None,
        "weighted_f1_std": statistics.pstdev(weighted) if len(weighted) > 1 else 0.0,
    }
    save_json(summary, Path("results") / args.experiment_name / "kfold_summary.json")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
