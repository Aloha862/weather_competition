"""Inference entry. Run: python infer.py"""
import torch

import config as cfg
from src.inference import build_test_loader_from_config, generate_submission, predict_test
from src.model import build_model
from src.utils import ensure_dirs, get_device, load_checkpoint, load_checkpoint_payload, load_json


def main():
    ensure_dirs([cfg.RESULTS_DIR, cfg.OUTPUTS_DIR, cfg.SUBMISSIONS_DIR])
    device = get_device()
    idx_to_class = load_json(cfg.IDX_TO_CLASS_PATH)
    num_classes = len(idx_to_class)
    checkpoint = load_checkpoint_payload(cfg.BEST_MODEL_PATH, device)
    model_name = checkpoint.get("model_name", cfg.MODEL_NAME)
    model, _ = build_model(model_name, num_classes, pretrained=False, fallback_model_name=cfg.FALLBACK_MODEL_NAME)
    model = model.to(device)
    load_checkpoint(model, cfg.BEST_MODEL_PATH, device)

    _, test_loader = build_test_loader_from_config(cfg)
    image_ids, pred_labels, _ = predict_test(model, test_loader, device, idx_to_class)
    save_path = generate_submission(image_ids, pred_labels, cfg.SAMPLE_SUBMISSION_PATH, cfg.SUBMISSION_PATH)
    print(f"Submission saved to: {save_path}")


if __name__ == "__main__":
    main()
