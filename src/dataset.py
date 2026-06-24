"""Dataset and data loading utilities for weather image classification."""
from __future__ import annotations

import warnings
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import pandas as pd
from PIL import Image
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from src.utils import save_json

VALID_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


class WeatherDataset(Dataset):
    """PyTorch dataset for train, validation and test images."""

    def __init__(self, image_paths: Sequence[str | Path], labels: Optional[Sequence[int]] = None,
                 image_ids: Optional[Sequence[str]] = None, transform=None):
        self.image_paths = [Path(p) for p in image_paths]
        self.labels = list(labels) if labels is not None else None
        self.image_ids = list(image_ids) if image_ids is not None else [p.name for p in self.image_paths]
        self.transform = transform

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int):
        path = self.image_paths[idx]
        try:
            image = Image.open(path).convert("RGB")
        except Exception as exc:
            warnings.warn(f"Failed to read image {path}: {exc}. A blank image is used instead.")
            image = Image.new("RGB", (224, 224), color=(0, 0, 0))
        if self.transform is not None:
            image = self.transform(image)
        if self.labels is None:
            return image, self.image_ids[idx]
        return image, int(self.labels[idx])


def build_transforms(img_size: int, is_train: bool, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
    """Build transforms. Validation and test transforms are deterministic."""
    if is_train:
        return transforms.Compose([
            transforms.RandomResizedCrop(img_size, scale=(0.75, 1.0)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.10, hue=0.02),
            transforms.RandAugment(num_ops=2, magnitude=7),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
            transforms.RandomErasing(p=0.15, scale=(0.02, 0.12), ratio=(0.3, 3.3)),
        ])
    return transforms.Compose([
        transforms.Resize(int(img_size * 1.14)),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])


def scan_image_folder(train_dir: str | Path) -> pd.DataFrame:
    """Scan data/train/class_name/image files and return a dataframe."""
    train_dir = Path(train_dir)
    if not train_dir.exists():
        raise FileNotFoundError(f"Training directory not found: {train_dir}")
    rows: List[Dict[str, str]] = []
    for class_dir in sorted([p for p in train_dir.iterdir() if p.is_dir()]):
        label = class_dir.name
        for image_path in sorted(class_dir.rglob("*")):
            if image_path.is_file() and image_path.suffix.lower() in VALID_EXTENSIONS:
                rows.append({"path": str(image_path), "label": label, "image_id": image_path.name})
    if not rows:
        raise ValueError(f"No images found in folder-style dataset: {train_dir}")
    return pd.DataFrame(rows)


def scan_test_folder(test_dir: str | Path) -> pd.DataFrame:
    """Scan test image folder."""
    test_dir = Path(test_dir)
    if not test_dir.exists():
        raise FileNotFoundError(f"Test directory not found: {test_dir}")
    rows = []
    for image_path in sorted(test_dir.rglob("*")):
        if image_path.is_file() and image_path.suffix.lower() in VALID_EXTENSIONS:
            rows.append({"path": str(image_path), "image_id": image_path.name})
    if not rows:
        raise ValueError(f"No test images found: {test_dir}")
    return pd.DataFrame(rows)


def load_csv_dataset(csv_path: str | Path, image_root: str | Path,
                     path_col: str = "image", label_col: str = "label") -> pd.DataFrame:
    """Load a CSV label file when the competition data uses table format."""
    csv_path = Path(csv_path)
    image_root = Path(image_root)
    df = pd.read_csv(csv_path)
    if path_col not in df.columns or label_col not in df.columns:
        raise ValueError(f"CSV must contain columns: {path_col}, {label_col}")
    df = df.copy()
    df["path"] = df[path_col].apply(lambda x: str(image_root / str(x)))
    df["label"] = df[label_col].astype(str)
    df["image_id"] = df[path_col].apply(lambda x: Path(str(x)).name)
    return df[["path", "label", "image_id"]]


def build_class_mapping(labels: Sequence[str], class_to_idx_path: str | Path, idx_to_class_path: str | Path) -> Tuple[Dict[str, int], Dict[str, str]]:
    """Build and save stable class mappings."""
    classes = sorted(set(str(x) for x in labels))
    class_to_idx = {cls: idx for idx, cls in enumerate(classes)}
    idx_to_class = {str(idx): cls for cls, idx in class_to_idx.items()}
    save_json(class_to_idx, class_to_idx_path)
    save_json(idx_to_class, idx_to_class_path)
    return class_to_idx, idx_to_class


def stratified_split(df: pd.DataFrame, val_ratio: float, seed: int) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Create a stratified train/validation split."""
    if "label_idx" not in df.columns:
        raise ValueError("DataFrame must contain label_idx before splitting.")
    train_df, val_df = train_test_split(
        df,
        test_size=val_ratio,
        random_state=seed,
        stratify=df["label_idx"],
    )
    return train_df.reset_index(drop=True), val_df.reset_index(drop=True)


def create_dataloaders(train_df: pd.DataFrame, val_df: pd.DataFrame, img_size: int,
                       batch_size: int, num_workers: int, mean, std) -> Tuple[DataLoader, DataLoader]:
    """Create train and validation DataLoaders."""
    train_dataset = WeatherDataset(
        train_df["path"].tolist(), train_df["label_idx"].tolist(), train_df["image_id"].tolist(),
        transform=build_transforms(img_size, True, mean, std),
    )
    val_dataset = WeatherDataset(
        val_df["path"].tolist(), val_df["label_idx"].tolist(), val_df["image_id"].tolist(),
        transform=build_transforms(img_size, False, mean, std),
    )
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=torch.cuda.is_available())
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,
                            num_workers=num_workers, pin_memory=torch.cuda.is_available())
    return train_loader, val_loader


def create_test_loader(test_dir: str | Path, img_size: int, batch_size: int, num_workers: int, mean, std) -> Tuple[pd.DataFrame, DataLoader]:
    """Create test DataLoader for inference."""
    test_df = scan_test_folder(test_dir)
    dataset = WeatherDataset(test_df["path"].tolist(), labels=None, image_ids=test_df["image_id"].tolist(),
                             transform=build_transforms(img_size, False, mean, std))
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False,
                        num_workers=num_workers, pin_memory=torch.cuda.is_available())
    return test_df, loader
