"""Training entry. Run: python train.py"""
from pathlib import Path
import csv
import torch

import config as cfg
from src.dataset import build_class_mapping, create_dataloaders, kfold_split, scan_image_folder, stratified_split
from src.metrics import (
    export_error_samples,
    export_focus_error_samples,
    plot_confusion_matrix,
    plot_training_curves,
    print_classification_report,
    save_per_class_metrics,
    save_validation_predictions,
    top_confusion_pairs,
)
from src.model import build_model
from src.train_utils import (
    EarlyStopping,
    ModelEMA,
    build_loss_function,
    build_optimizer,
    build_scheduler,
    set_backbone_trainable,
    train_one_epoch,
    validate,
)
from src.utils import ensure_dirs, get_device, save_checkpoint, save_json, set_seed


def _checkpoint_extra(actual_name, num_classes, class_to_idx, idx_to_class):
    return {
        "model_name": actual_name,
        "num_classes": num_classes,
        "class_to_idx": class_to_idx,
        "idx_to_class": idx_to_class,
        "img_size": cfg.IMG_SIZE,
        "mean": cfg.MEAN,
        "std": cfg.STD,
        "augment_profile": cfg.AUGMENT_PROFILE,
        "loss_type": cfg.LOSS_TYPE,
        "target_metric": cfg.TARGET_METRIC,
        "experiment_name": cfg.EXPERIMENT_NAME,
    }


def _topk_path(rank: int) -> Path:
    base = Path(cfg.BEST_MODEL_PATH)
    if rank == 1:
        return base
    return base.with_name(f"{base.stem}_top{rank}{base.suffix}")


def main():
    set_seed(cfg.SEED)
    ensure_dirs([cfg.RESULTS_DIR, cfg.OUTPUTS_DIR, cfg.SUBMISSIONS_DIR, cfg.ERRORS_DIR, cfg.LOGS_DIR])
    device = get_device()
    print(f"device={device}", flush=True)
    if device.type == "cuda":
        index = device.index if device.index is not None else torch.cuda.current_device()
        name = torch.cuda.get_device_name(index)
        props = torch.cuda.get_device_properties(index)
        total_gb = props.total_memory / (1024 ** 3)
        print(f"gpu={name}, total_memory={total_gb:.1f}GB", flush=True)

    df = scan_image_folder(cfg.TRAIN_DIR)
    class_to_idx, idx_to_class = build_class_mapping(df["label"].tolist(), cfg.CLASS_TO_IDX_PATH, cfg.IDX_TO_CLASS_PATH)
    df["label_idx"] = df["label"].map(class_to_idx).astype(int)
    num_classes = len(class_to_idx)
    if cfg.USE_KFOLD:
        train_df, val_df = kfold_split(df, cfg.NUM_FOLDS, cfg.FOLD_INDEX, cfg.SEED)
        print(f"Using K-fold split: fold {cfg.FOLD_INDEX + 1}/{cfg.NUM_FOLDS}", flush=True)
    else:
        train_df, val_df = stratified_split(df, cfg.VAL_RATIO, cfg.SEED)
    train_loader, val_loader = create_dataloaders(
        train_df, val_df, cfg.IMG_SIZE, cfg.BATCH_SIZE, cfg.NUM_WORKERS, cfg.MEAN, cfg.STD,
        cfg.AUGMENT_PROFILE, cfg.SAMPLER_TYPE,
    )

    model, actual_name = build_model(cfg.MODEL_NAME, num_classes, cfg.PRETRAINED, cfg.FALLBACK_MODEL_NAME)
    model = model.to(device)
    if cfg.USE_FREEZE_BACKBONE:
        set_backbone_trainable(model, False)
        print(f"Backbone frozen for first {cfg.FREEZE_EPOCHS} epoch(s).", flush=True)
    criterion = build_loss_function(
        num_classes,
        train_df["label_idx"].tolist(),
        cfg.USE_CLASS_WEIGHT,
        cfg.LABEL_SMOOTHING,
        device,
        cfg.LOSS_TYPE,
        cfg.FOCAL_GAMMA,
        cfg.CB_BETA,
    )
    optimizer = build_optimizer(model, cfg.LEARNING_RATE, cfg.WEIGHT_DECAY, cfg.HEAD_LR_MULT)
    scheduler = build_scheduler(optimizer, cfg.EPOCHS, cfg.MIN_LR, cfg.USE_WARMUP, cfg.WARMUP_EPOCHS)
    amp_enabled = bool(cfg.USE_AMP and device.type == "cuda")
    try:
        scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)
    except TypeError:
        scaler = torch.cuda.amp.GradScaler(enabled=amp_enabled)
    stopper = EarlyStopping(patience=cfg.EARLY_STOPPING_PATIENCE)
    ema = ModelEMA(model, cfg.EMA_DECAY) if cfg.USE_EMA else None

    fields = ["epoch", "train_loss", "train_accuracy", "train_macro_f1", "train_weighted_f1", "val_loss", "val_accuracy", "val_macro_f1", "val_weighted_f1", "lr"]
    with Path(cfg.TRAIN_LOG_PATH).open("w", newline="", encoding="utf-8") as f:
        csv.DictWriter(f, fieldnames=fields).writeheader()

    best_score = -1.0
    best_epoch = -1
    best_metrics = {}
    best_true, best_pred = None, None
    top_records = []
    for epoch in range(1, cfg.EPOCHS + 1):
        if cfg.USE_FREEZE_BACKBONE and epoch == cfg.FREEZE_EPOCHS + 1:
            set_backbone_trainable(model, True)
            optimizer = build_optimizer(model, cfg.LEARNING_RATE, cfg.WEIGHT_DECAY, cfg.HEAD_LR_MULT)
            scheduler = build_scheduler(optimizer, max(1, cfg.EPOCHS - epoch + 1), cfg.MIN_LR, False, 0)
            print("Backbone unfrozen; optimizer rebuilt.", flush=True)
        tm = train_one_epoch(
            model, train_loader, criterion, optimizer, device, scaler, cfg.USE_AMP, epoch,
            cfg.GRAD_ACCUM_STEPS, cfg.MAX_GRAD_NORM, ema,
        )
        eval_model = ema.ema if ema is not None else model
        vm, y_true, y_pred = validate(eval_model, val_loader, criterion, device)
        scheduler.step()
        score = vm.get(cfg.TARGET_METRIC, vm["macro_f1"])
        if stopper.step(score):
            best_score = score
            best_epoch = epoch
            best_metrics = vm
            best_true, best_pred = y_true, y_pred
            save_model = ema.ema if ema is not None else model
            extra = _checkpoint_extra(actual_name, num_classes, class_to_idx, idx_to_class)
            extra.update({"best_metrics": vm, "used_ema_weights": bool(ema is not None)})
            save_checkpoint(save_model, optimizer, epoch, best_score, cfg.BEST_MODEL_PATH, extra)
            if cfg.SAVE_TOP_K > 1:
                epoch_path = Path(cfg.BEST_MODEL_PATH).with_name(f"{Path(cfg.BEST_MODEL_PATH).stem}_epoch{epoch}_score{score:.5f}.pth")
                save_checkpoint(save_model, optimizer, epoch, best_score, epoch_path, extra)
                top_records.append((score, epoch_path))
                top_records = sorted(top_records, key=lambda x: x[0], reverse=True)[:cfg.SAVE_TOP_K]
                for rank, (_, path) in enumerate(top_records, start=1):
                    if path.exists() and path != _topk_path(rank):
                        _topk_path(rank).write_bytes(path.read_bytes())
        row = {"epoch": epoch, "train_loss": tm["loss"], "train_accuracy": tm["accuracy"], "train_macro_f1": tm["macro_f1"], "train_weighted_f1": tm["weighted_f1"], "val_loss": vm["loss"], "val_accuracy": vm["accuracy"], "val_macro_f1": vm["macro_f1"], "val_weighted_f1": vm["weighted_f1"], "lr": optimizer.param_groups[0]["lr"]}
        with Path(cfg.TRAIN_LOG_PATH).open("a", newline="", encoding="utf-8") as f:
            csv.DictWriter(f, fieldnames=fields).writerow(row)
        plot_training_curves(cfg.TRAIN_LOG_PATH, cfg.TRAINING_CURVES_PATH)
        print(row, flush=True)
        if stopper.should_stop:
            break

    class_names = [idx_to_class[str(i)] for i in range(num_classes)]
    if best_true is not None:
        print_classification_report(best_true, best_pred, class_names)
        plot_confusion_matrix(best_true, best_pred, class_names, cfg.CONFUSION_MATRIX_PATH)
        save_per_class_metrics(best_true, best_pred, class_names, cfg.PER_CLASS_METRICS_PATH)
        save_validation_predictions(val_df["path"].tolist(), best_true, best_pred, idx_to_class, cfg.VAL_PREDICTIONS_PATH)
        export_error_samples(val_df["path"].tolist(), best_true, best_pred, idx_to_class, cfg.ERRORS_DIR)
        export_focus_error_samples(val_df["path"].tolist(), best_true, best_pred, idx_to_class, cfg.ERRORS_DIR / "focus_rainy_snowy")
    save_json({
        "model_name": actual_name,
        "best_epoch": best_epoch,
        "best_score": best_score,
        "best_accuracy": best_metrics.get("accuracy"),
        "best_macro_f1": best_metrics.get("macro_f1"),
        "best_weighted_f1": best_metrics.get("weighted_f1"),
        "num_classes": num_classes,
        "augment_profile": cfg.AUGMENT_PROFILE,
        "loss_type": cfg.LOSS_TYPE,
        "sampler_type": cfg.SAMPLER_TYPE,
        "use_ema": cfg.USE_EMA,
        "use_warmup": cfg.USE_WARMUP,
        "confusion_pairs": top_confusion_pairs(best_true, best_pred, class_names) if best_true is not None else [],
    }, cfg.TRAINING_SUMMARY_PATH)


if __name__ == "__main__":
    main()
