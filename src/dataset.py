"""Dataset and data loading utilities for weather image classification."""
from __future__ import annotations

import warnings
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Union

import pandas as pd
from PIL import Image
from sklearn.model_selection import StratifiedKFold, train_test_split
import torch
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision import transforms

from src.utils import save_json

VALID_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def _has_direct_images(folder: Path) -> bool:
    return any(p.is_file() and p.suffix.lower() in VALID_EXTENSIONS for p in folder.iterdir())


def _discover_class_dirs(train_dir: Path) -> List[Path]:
    """Find class folders, including common nested train/train/class layouts."""
    direct_class_dirs = [
        p for p in sorted(train_dir.iterdir())
        if p.is_dir() and _has_direct_images(p)
    ]
    if direct_class_dirs:
        return direct_class_dirs
    nested_class_dirs = [
        p for p in sorted(train_dir.rglob("*"))
        if p.is_dir() and _has_direct_images(p)
    ]
    return nested_class_dirs


class WeatherDataset(Dataset):
    """PyTorch dataset for train, validation and test images."""

    def __init__(self, image_paths: Sequence[Union[str, Path]], labels: Optional[Sequence[int]] = None,
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


def build_transforms(img_size: int, is_train: bool, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225),
                     augment_profile: str = "weather_safe"):
    """Build transforms. Validation and test transforms are deterministic.

    Weather recognition depends on brightness, sky color, rain/snow texture and
    haze. Training augmentation is therefore intentionally conservative.
    """
    if is_train:
        profile = str(augment_profile or "weather_safe").lower()
        if profile == "light":
            ops = [
                transforms.RandomResizedCrop(img_size, scale=(0.85, 1.0)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ColorJitter(brightness=0.10, contrast=0.10, saturation=0.05, hue=0.0),
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
                transforms.RandomErasing(p=0.08, scale=(0.02, 0.08), ratio=(0.3, 3.3)),
            ]
        elif profile == "strong":
            ops = [
                transforms.RandomResizedCrop(img_size, scale=(0.70, 1.0)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ColorJitter(brightness=0.18, contrast=0.18, saturation=0.12, hue=0.015),
                transforms.RandAugment(num_ops=2, magnitude=8),
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
                transforms.RandomErasing(p=0.20, scale=(0.02, 0.14), ratio=(0.3, 3.3)),
            ]
        else:
            ops = [
                transforms.RandomResizedCrop(img_size, scale=(0.80, 1.0)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ColorJitter(brightness=0.12, contrast=0.12, saturation=0.06, hue=0.0),
                transforms.RandAugment(num_ops=2, magnitude=5),
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
                transforms.RandomErasing(p=0.12, scale=(0.02, 0.10), ratio=(0.3, 3.3)),
            ]
        return transforms.Compose(ops)
    return transforms.Compose([
        transforms.Resize(int(img_size * 1.14)),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])


def scan_image_folder(train_dir: Union[str, Path]) -> pd.DataFrame:
    """Scan data/train/class_name/image files and return a dataframe."""
    train_dir = Path(train_dir)
    if not train_dir.exists():
        raise FileNotFoundError(f"Training directory not found: {train_dir}")
    rows: List[Dict[str, str]] = []
    class_dirs = _discover_class_dirs(train_dir)
    if not class_dirs:
        raise ValueError(f"No class folders with images found under: {train_dir}")
    for class_dir in class_dirs:
        label = class_dir.name
        for image_path in sorted(class_dir.rglob("*")):
            if image_path.is_file() and image_path.suffix.lower() in VALID_EXTENSIONS:
                rows.append({"path": str(image_path), "label": label, "image_id": image_path.name})
    if not rows:
        raise ValueError(f"No images found in folder-style dataset: {train_dir}")
    df = pd.DataFrame(rows)
    if df["label"].nunique() < 2:
        raise ValueError(
            f"Only one class was found under {train_dir}. "
            "Expected at least two class folders such as data/train/sunny/*.jpg."
        )
    return df


def scan_test_folder(test_dir: Union[str, Path]) -> pd.DataFrame:
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


def load_csv_dataset(csv_path: Union[str, Path], image_root: Union[str, Path],
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


def build_class_mapping(labels: Sequence[str], class_to_idx_path: Union[str, Path],
                        idx_to_class_path: Union[str, Path]) -> Tuple[Dict[str, int], Dict[str, str]]:
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
    class_counts = df["label_idx"].value_counts()
    if class_counts.min() < 2:
        raise ValueError("Each class needs at least two images for stratified validation split.")
    train_df, val_df = train_test_split(
        df,
        test_size=val_ratio,
        random_state=seed,
        stratify=df["label_idx"],
    )
    return train_df.reset_index(drop=True), val_df.reset_index(drop=True)


def kfold_split(df: pd.DataFrame, num_folds: int, fold_index: int, seed: int) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Create one stratified K-fold train/validation split."""
    if "label_idx" not in df.columns:
        raise ValueError("DataFrame must contain label_idx before K-fold splitting.")
    if num_folds < 2:
        raise ValueError("num_folds must be at least 2.")
    fold_index = int(fold_index)
    if fold_index < 0 or fold_index >= num_folds:
        raise ValueError(f"fold_index must be in [0, {num_folds - 1}], got {fold_index}")
    splitter = StratifiedKFold(n_splits=num_folds, shuffle=True, random_state=seed)
    splits = list(splitter.split(df, df["label_idx"]))
    train_idx, val_idx = splits[fold_index]
    return df.iloc[train_idx].reset_index(drop=True), df.iloc[val_idx].reset_index(drop=True)


def create_dataloaders(train_df: pd.DataFrame, val_df: pd.DataFrame, img_size: int,
                       batch_size: int, num_workers: int, mean, std, augment_profile: str = "weather_safe",
                       sampler_type: str = "none") -> Tuple[DataLoader, DataLoader]:
    """Create train and validation DataLoaders."""
    train_dataset = WeatherDataset(
        train_df["path"].tolist(), train_df["label_idx"].tolist(), train_df["image_id"].tolist(),
        transform=build_transforms(img_size, True, mean, std, augment_profile),
    )
    val_dataset = WeatherDataset(
        val_df["path"].tolist(), val_df["label_idx"].tolist(), val_df["image_id"].tolist(),
        transform=build_transforms(img_size, False, mean, std),
    )
    sampler = None
    shuffle = True
    if str(sampler_type).lower() == "weighted":
        labels = torch.tensor(train_df["label_idx"].tolist(), dtype=torch.long)
        counts = torch.bincount(labels).float()
        counts[counts == 0] = 1.0
        sample_weights = (1.0 / counts[labels]).double()
        sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)
        shuffle = False
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle, sampler=sampler,
                              num_workers=num_workers, pin_memory=torch.cuda.is_available())
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,
                            num_workers=num_workers, pin_memory=torch.cuda.is_available())
    return train_loader, val_loader


def create_test_loader(test_dir: Union[str, Path], img_size: int, batch_size: int,
                       num_workers: int, mean, std) -> Tuple[pd.DataFrame, DataLoader]:
    """Create test DataLoader for inference."""
    test_df = scan_test_folder(test_dir)
    dataset = WeatherDataset(test_df["path"].tolist(), labels=None, image_ids=test_df["image_id"].tolist(),
                             transform=build_transforms(img_size, False, mean, std))
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False,
                        num_workers=num_workers, pin_memory=torch.cuda.is_available())
    return test_df, loader
