"""Training entry. Run: python train.py"""
from pathlib import Path
import csv
import torch

import config as cfg
from src.dataset import build_class_mapping, create_dataloaders, scan_image_folder, stratified_split
from src.metrics import plot_confusion_matrix, print_classification_report
from src.model import build_model
from src.train_utils import EarlyStopping, build_loss_function, build_optimizer, build_scheduler, train_one_epoch, validate
from src.utils import ensure_dirs, get_device, save_checkpoint, save_json, set_seed


def main():
    set_seed(cfg.SEED)
    ensure_dirs([cfg.RESULTS_DIR, cfg.OUTPUTS_DIR, cfg.SUBMISSIONS_DIR, cfg.ERRORS_DIR, cfg.LOGS_DIR])
    device = get_device()
    print(f"device={device}")

    df = scan_image_folder(cfg.TRAIN_DIR)
    class_to_idx, idx_to_class = build_class_mapping(df["label"].tolist(), cfg.CLASS_TO_IDX_PATH, cfg.IDX_TO_CLASS_PATH)
    df["label_idx"] = df["label"].map(class_to_idx).astype(int)
    num_classes = len(class_to_idx)
    train_df, val_df = stratified_split(df, cfg.VAL_RATIO, cfg.SEED)
    train_loader, val_loader = create_dataloaders(train_df, val_df, cfg.IMG_SIZE, cfg.BATCH_SIZE, cfg.NUM_WORKERS, cfg.MEAN, cfg.STD)

    model, actual_name = build_model(cfg.MODEL_NAME, num_classes, cfg.PRETRAINED, cfg.FALLBACK_MODEL_NAME)
    model = model.to(device)
    criterion = build_loss_function(num_classes, train_df["label_idx"].tolist(), cfg.USE_CLASS_WEIGHT, cfg.LABEL_SMOOTHING, device)
    optimizer = build_optimizer(model, cfg.LEARNING_RATE, cfg.WEIGHT_DECAY)
    scheduler = build_scheduler(optimizer, cfg.EPOCHS, cfg.MIN_LR)
    amp_enabled = bool(cfg.USE_AMP and device.type == "cuda")
    try:
        scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)
    except TypeError:
        scaler = torch.cuda.amp.GradScaler(enabled=amp_enabled)
    stopper = EarlyStopping(patience=cfg.EARLY_STOPPING_PATIENCE)

    fields = ["epoch", "train_loss", "train_accuracy", "train_macro_f1", "train_weighted_f1", "val_loss", "val_accuracy", "val_macro_f1", "val_weighted_f1", "lr"]
    with Path(cfg.TRAIN_LOG_PATH).open("w", newline="", encoding="utf-8") as f:
        csv.DictWriter(f, fieldnames=fields).writeheader()

    best_score = -1.0
    best_epoch = -1
    best_true, best_pred = None, None
    for epoch in range(1, cfg.EPOCHS + 1):
        tm = train_one_epoch(model, train_loader, criterion, optimizer, device, scaler, cfg.USE_AMP, epoch)
        vm, y_true, y_pred = validate(model, val_loader, criterion, device)
        scheduler.step()
        score = vm.get(cfg.TARGET_METRIC, vm["macro_f1"])
        if stopper.step(score):
            best_score = score
            best_epoch = epoch
            best_true, best_pred = y_true, y_pred
            save_checkpoint(model, optimizer, epoch, best_score, cfg.BEST_MODEL_PATH, {"model_name": actual_name, "num_classes": num_classes})
        row = {"epoch": epoch, "train_loss": tm["loss"], "train_accuracy": tm["accuracy"], "train_macro_f1": tm["macro_f1"], "train_weighted_f1": tm["weighted_f1"], "val_loss": vm["loss"], "val_accuracy": vm["accuracy"], "val_macro_f1": vm["macro_f1"], "val_weighted_f1": vm["weighted_f1"], "lr": optimizer.param_groups[0]["lr"]}
        with Path(cfg.TRAIN_LOG_PATH).open("a", newline="", encoding="utf-8") as f:
            csv.DictWriter(f, fieldnames=fields).writerow(row)
        print(row)
        if stopper.should_stop:
            break

    class_names = [idx_to_class[str(i)] for i in range(num_classes)]
    if best_true is not None:
        print_classification_report(best_true, best_pred, class_names)
        plot_confusion_matrix(best_true, best_pred, class_names, cfg.CONFUSION_MATRIX_PATH)
    save_json({"model_name": actual_name, "best_epoch": best_epoch, "best_score": best_score, "num_classes": num_classes}, cfg.TRAINING_SUMMARY_PATH)


if __name__ == "__main__":
    main()
