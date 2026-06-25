"""Download and prepare a public labeled weather test set.

Default source:
https://huggingface.co/datasets/davidshableski/weatherimages

The source dataset has five classes: sunny, rainy, cloudy, snowy, sunrise.
This project evaluates only the overlapping four classes:
cloudy, rainy, snowy, sunny.
"""
from __future__ import annotations

import argparse
import shutil
import urllib.request
import zipfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple

VALID_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
DEFAULT_URL = "https://huggingface.co/datasets/davidshableski/weatherimages/resolve/main/Data.zip"
CLASS_ALIASES = {
    "cloud": "cloudy",
    "cloudy": "cloudy",
    "rain": "rainy",
    "rainy": "rainy",
    "snow": "snowy",
    "snowy": "snowy",
    "shine": "sunny",
    "sunny": "sunny",
}
TARGET_CLASSES = {"cloudy", "rainy", "snowy", "sunny"}


def _download(url: str, path: Path, force: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and not force:
        print(f"Using existing archive: {path}")
        return
    print(f"Downloading: {url}")
    with urllib.request.urlopen(url) as response, path.open("wb") as f:
        total = int(response.headers.get("Content-Length", "0") or 0)
        done = 0
        while True:
            chunk = response.read(1024 * 1024)
            if not chunk:
                break
            f.write(chunk)
            done += len(chunk)
            if total:
                print(f"\r{done / total:.1%}", end="")
        print()
    print(f"Saved: {path}")


def _extract(archive: Path, extract_dir: Path, force: bool = False) -> None:
    if extract_dir.exists() and force:
        shutil.rmtree(extract_dir)
    if extract_dir.exists() and any(extract_dir.iterdir()):
        print(f"Using existing extracted directory: {extract_dir}")
        return
    extract_dir.mkdir(parents=True, exist_ok=True)
    print(f"Extracting: {archive}")
    with zipfile.ZipFile(archive) as zf:
        zf.extractall(extract_dir)


def _class_name(path: Path) -> Optional[str]:
    return CLASS_ALIASES.get(path.name.strip().lower())


def _find_class_dirs(root: Path, preferred_split: str) -> List[Tuple[Path, str]]:
    candidates: List[Tuple[Path, str, int]] = []
    for folder in root.rglob("*"):
        if not folder.is_dir():
            continue
        label = _class_name(folder)
        if label not in TARGET_CLASSES:
            continue
        image_count = sum(1 for p in folder.rglob("*") if p.is_file() and p.suffix.lower() in VALID_EXTENSIONS)
        if image_count == 0:
            continue
        split_bonus = 0 if preferred_split.lower() in {p.name.lower() for p in folder.parents} else 1
        candidates.append((folder, label, split_bonus))

    # Prefer folders below the requested split, then avoid duplicate labels.
    selected: Dict[str, Path] = {}
    for folder, label, _ in sorted(candidates, key=lambda x: (x[2], len(x[0].parts))):
        selected.setdefault(label, folder)
    return [(folder, label) for label, folder in sorted(selected.items())]


def _copy_dataset(class_dirs: List[Tuple[Path, str]], output_dir: Path,
                  max_per_class: Optional[int], overwrite: bool) -> None:
    if output_dir.exists() and overwrite:
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    for source_dir, label in class_dirs:
        target_dir = output_dir / label
        target_dir.mkdir(parents=True, exist_ok=True)
        images = [p for p in sorted(source_dir.rglob("*")) if p.is_file() and p.suffix.lower() in VALID_EXTENSIONS]
        if max_per_class is not None:
            images = images[:max_per_class]
        copied = 0
        for image in images:
            target = target_dir / image.name
            if target.exists() and not overwrite:
                continue
            shutil.copy2(image, target)
            copied += 1
        print(f"{label}: copied {copied}, selected {len(images)}, source {source_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Download and prepare a public labeled weather test set.")
    parser.add_argument("--url", default=DEFAULT_URL)
    parser.add_argument("--work-dir", default="tmp/public_weather_source")
    parser.add_argument("--output-dir", default="tmp/public_weather_test")
    parser.add_argument("--split", default="test", help="Preferred split folder, usually test or val.")
    parser.add_argument("--max-per-class", type=int, default=None)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    work_dir = Path(args.work_dir)
    archive = work_dir / "Data.zip"
    extract_dir = work_dir / "extracted"
    output_dir = Path(args.output_dir)

    _download(args.url, archive, args.force)
    _extract(archive, extract_dir, args.force)
    class_dirs = _find_class_dirs(extract_dir, args.split)
    labels = {label for _, label in class_dirs}
    missing = TARGET_CLASSES - labels
    if missing:
        raise ValueError(f"Missing target classes after extraction: {sorted(missing)}")
    _copy_dataset(class_dirs, output_dir, args.max_per_class, args.force)
    print(f"Prepared labeled test set: {output_dir.resolve()}")


if __name__ == "__main__":
    main()
