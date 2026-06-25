"""Analyze validation errors and generate a Markdown report.

Run after training:
    python scripts/analyze_errors.py
"""
from __future__ import annotations

from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import config as cfg
from src.utils import ensure_dir


def main() -> None:
    pred_path = Path(cfg.VAL_PREDICTIONS_PATH)
    per_class_path = Path(cfg.PER_CLASS_METRICS_PATH)
    output_path = Path("outputs/error_analysis_report.md")
    if not pred_path.exists():
        raise FileNotFoundError(f"Validation predictions not found: {pred_path}. Run train.py first.")
    df = pd.read_csv(pred_path)
    per_class = pd.read_csv(per_class_path) if per_class_path.exists() else pd.DataFrame()
    errors = df[df["correct"] == 0].copy()
    ensure_dir(output_path.parent)

    lines = [
        "# Error Analysis Report",
        "",
        "## Class F1 Ranking",
        "",
    ]
    if not per_class.empty:
        lines.append("```text")
        lines.append(per_class.to_string(index=False))
        lines.append("```")
        worst = per_class.sort_values("f1").iloc[0]
        lines.extend(["", f"Lowest F1 class: `{worst['class_name']}` (f1={worst['f1']:.4f}).", ""])

    lines.extend(["## Top Confusion Pairs", ""])
    if errors.empty:
        lines.append("No validation errors found.")
    else:
        confusion = errors.groupby(["true_label", "pred_label"]).size().reset_index(name="count").sort_values("count", ascending=False)
        lines.append("```text")
        lines.append(confusion.head(20).to_string(index=False))
        lines.append("```")

    lines.extend(["", "## Focus: rainy / snowy", ""])
    for label in ["rainy", "snowy"]:
        subset = errors[(errors["true_label"] == label) | (errors["pred_label"] == label)]
        lines.append(f"### {label}")
        if subset.empty:
            lines.append("No errors involving this class.")
        else:
            table = subset.groupby(["true_label", "pred_label"]).size().reset_index(name="count").sort_values("count", ascending=False)
            lines.append("```text")
            lines.append(table.to_string(index=False))
            lines.append("```")
        lines.append("")

    lines.extend([
        "## Diagnostic Suggestions",
        "",
        "- If rainy precision is low, inspect cloudy/snowy images predicted as rainy; wet ground, dark scenes, and rain streak labels may be noisy.",
        "- If snowy precision is low, inspect bright cloudy/sunny scenes and overexposed snow/sky boundaries.",
        "- If a small class has high recall but low precision, the model may over-predict it; compare focal loss, weighted sampler, and class weights one at a time.",
        "- If large classes dominate weighted F1 while macro F1 lags, optimize rainy/snowy first.",
        "- Check high-confidence errors manually when a `confidence` column is available in prediction CSVs.",
    ])
    output_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"Saved: {output_path}")


if __name__ == "__main__":
    main()
