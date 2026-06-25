"""Evaluate a trained checkpoint on a labeled ImageFolder-style dataset."""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import torch
from sklearn.metrics import accuracy_score, classification_report, f1_score
from torch.utils.data import DataLoader
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import config as cfg  # noqa: E402
from src.dataset import WeatherDataset, build_transforms, scan_image_folder  # noqa: E402
from src.metrics import plot_confusion_matrix  # noqa: E402
from src.model import build_model  # noqa: E402
from src.utils import ensure_dir, get_device, load_checkpoint, load_checkpoint_payload, load_json, save_json  # noqa: E402


def evaluate(folder: Path, output_dir: Path, batch_size: int, num_workers: int) -> dict[str, float]:
    device = get_device()
    idx_to_class = load_json(cfg.IDX_TO_CLASS_PATH)
    known_labels = set(str(v) for v in idx_to_class.values())
    checkpoint = load_checkpoint_payload(cfg.BEST_MODEL_PATH, device)
    model_name = checkpoint.get("model_name", cfg.MODEL_NAME)
    model, _ = build_model(model_name, len(idx_to_class), pretrained=False, fallback_model_name=cfg.FALLBACK_MODEL_NAME)
    model = model.to(device)
    load_checkpoint(model, cfg.BEST_MODEL_PATH, device)
    model.eval()

    df = scan_image_folder(folder)
    unknown = sorted(set(df["label"]) - known_labels)
    if unknown:
        raise ValueError(f"Labels not found in trained class mapping: {unknown}. Known labels: {sorted(known_labels)}")

    dataset = WeatherDataset(
        df["path"].tolist(),
        labels=None,
        image_ids=df["image_id"].tolist(),
        transform=build_transforms(cfg.IMG_SIZE, False, cfg.MEAN, cfg.STD),
    )
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=torch.cuda.is_available())

    y_true: list[str] = []
    y_pred: list[str] = []
    rows: list[dict[str, str | float]] = []
    with torch.inference_mode():
        offset = 0
        for images, image_ids in tqdm(loader, desc="Evaluate", leave=False):
            images = images.to(device, non_blocking=True)
            logits = model(images)
            probs = torch.softmax(logits, dim=1)
            confs, preds = probs.max(dim=1)
            batch_true = df.iloc[offset: offset + len(image_ids)]["label"].astype(str).tolist()
            offset += len(image_ids)
            for image_id, true_label, pred_idx, conf in zip(image_ids, batch_true, preds.cpu().tolist(), confs.cpu().tolist()):
                pred_label = idx_to_class[str(int(pred_idx))]
                y_true.append(true_label)
                y_pred.append(pred_label)
                rows.append({
                    "image_id": str(image_id),
                    "true_label": true_label,
                    "pred_label": pred_label,
                    "confidence": float(conf),
                    "correct": int(true_label == pred_label),
                })

    labels = sorted(known_labels & set(y_true))
    metrics = {
        "num_images": float(len(y_true)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", labels=labels, zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted", labels=labels, zero_division=0)),
    }

    output_dir = ensure_dir(output_dir)
    save_json(metrics, output_dir / "external_test_metrics.json")
    report = classification_report(y_true, y_pred, labels=labels, zero_division=0)
    (output_dir / "external_test_report.txt").write_text(report, encoding="utf-8")
    with (output_dir / "external_test_predictions.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["image_id", "true_label", "pred_label", "confidence", "correct"])
        writer.writeheader()
        writer.writerows(rows)

    label_to_idx = {label: i for i, label in enumerate(labels)}
    plot_confusion_matrix(
        [label_to_idx[x] for x in y_true],
        [label_to_idx.get(x, -1) for x in y_pred],
        labels,
        output_dir / "external_test_confusion_matrix.png",
    )
    print(report)
    print(metrics)
    print(f"Saved evaluation files to: {output_dir.resolve()}")
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate best checkpoint on a labeled folder dataset.")
    parser.add_argument("--data-dir", default="tmp/public_weather_test")
    parser.add_argument("--output-dir", default="outputs/external_test")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=0)
    args = parser.parse_args()
    evaluate(Path(args.data_dir), Path(args.output_dir), args.batch_size, args.num_workers)


if __name__ == "__main__":
    main()

