"""Prepare a folder-style weather dataset for this project.

The script copies images from common nested layouts into:

    data/train/<class_name>/*.jpg

It does not download data. Use it after uploading or unpacking public/platform
data into a local directory.
"""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path

VALID_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def _has_direct_images(path: Path) -> bool:
    return any(p.is_file() and p.suffix.lower() in VALID_EXTENSIONS for p in path.iterdir())


def _find_class_dirs(source: Path) -> list[Path]:
    direct = [p for p in sorted(source.iterdir()) if p.is_dir() and _has_direct_images(p)]
    if direct:
        return direct
    return [p for p in sorted(source.rglob("*")) if p.is_dir() and _has_direct_images(p)]


def prepare(source: Path, output: Path, max_per_class: int | None = None, overwrite: bool = False) -> None:
    if not source.exists():
        raise FileNotFoundError(f"Source directory does not exist: {source}")
    class_dirs = _find_class_dirs(source)
    if not class_dirs:
        raise ValueError(f"No class directories with images found under: {source}")
    output.mkdir(parents=True, exist_ok=True)
    for class_dir in class_dirs:
        dst_dir = output / class_dir.name
        dst_dir.mkdir(parents=True, exist_ok=True)
        all_images = [p for p in sorted(class_dir.rglob("*")) if p.is_file() and p.suffix.lower() in VALID_EXTENSIONS]
        images = all_images
        if max_per_class is not None:
            images = images[:max_per_class]
        copied = 0
        for image in images:
            dst = dst_dir / image.name
            if dst.exists() and not overwrite:
                continue
            shutil.copy2(image, dst)
            copied += 1
        print(f"{class_dir.name}: copied {copied}, selected {len(images)}, source images {len(all_images)}")
    print(f"Prepared dataset: {output.resolve()}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Copy nested weather class folders into data/train.")
    parser.add_argument("--source", required=True, help="Unpacked source dataset directory.")
    parser.add_argument("--output", default="data/train", help="Output train directory.")
    parser.add_argument("--max-per-class", type=int, default=None, help="Optional cap for quick tests.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing copied files.")
    args = parser.parse_args()
    prepare(Path(args.source), Path(args.output), args.max_per_class, args.overwrite)


if __name__ == "__main__":
    main()
