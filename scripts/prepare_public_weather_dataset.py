"""公开天气图片分类数据集整理脚本。

用途：
1. 支持手动下载 Kaggle 等公开天气数据集 zip 后放入 data/raw/。
2. 自动解压、识别类别文件夹、过滤坏图。
3. 自动生成 data/train/类别名/*.jpg、data/test/*.jpg 和 data/sample_submission.csv。
4. 输出 data/dataset_prepare_report.json，便于确认数据准备结果。

注意：
- 本脚本只整理公开数据用于工程闭环 smoke test，不代表正式比赛数据分布。
- 不会强制依赖 Kaggle API；如需 Kaggle API，请先手动下载 zip 或自行下载到 data/raw/。
"""
from __future__ import annotations

import argparse
import json
import random
import shutil
import zipfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import pandas as pd
from PIL import Image

VALID_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def ensure_dir(path: Path) -> Path:
    """确保目录存在。"""
    path.mkdir(parents=True, exist_ok=True)
    return path


def is_image_file(path: Path) -> bool:
    """判断文件是否为常见图片格式。"""
    return path.is_file() and path.suffix.lower() in VALID_EXTENSIONS


def verify_image(path: Path) -> bool:
    """检查图片是否可被 PIL 正常读取，坏图返回 False。"""
    try:
        with Image.open(path) as img:
            img.verify()
        return True
    except Exception:
        return False


def safe_copy(src: Path, dst: Path) -> None:
    """复制文件，若目标重名则自动追加序号，避免覆盖。"""
    ensure_dir(dst.parent)
    final_dst = dst
    index = 1
    while final_dst.exists():
        final_dst = dst.with_name(f"{dst.stem}_{index}{dst.suffix}")
        index += 1
    shutil.copy2(src, final_dst)


def clean_output_dirs(train_dir: Path, test_dir: Path) -> None:
    """清理旧的 train/test 输出目录，避免混入上次整理结果。"""
    for path in [train_dir, test_dir]:
        if path.exists():
            shutil.rmtree(path)
        ensure_dir(path)


def extract_archives(raw_dir: Path, extract_dir: Path, zip_path: Path | None = None) -> List[Path]:
    """解压 zip 文件。若未指定 zip_path，则解压 data/raw/ 下全部 zip。"""
    ensure_dir(raw_dir)
    ensure_dir(extract_dir)
    archives = [zip_path] if zip_path else sorted(raw_dir.glob("*.zip"))
    extracted_roots: List[Path] = []

    if not archives:
        # 允许用户已经手动解压到 data/raw/，此时直接扫描 raw_dir。
        return [raw_dir]

    for archive in archives:
        archive = Path(archive)
        if not archive.exists():
            raise FileNotFoundError(f"找不到压缩包：{archive}")
        target = extract_dir / archive.stem
        ensure_dir(target)
        print(f"[解压] {archive} -> {target}")
        with zipfile.ZipFile(archive, "r") as zf:
            zf.extractall(target)
        extracted_roots.append(target)
    return extracted_roots


def find_candidate_class_dirs(search_roots: Iterable[Path], min_images_per_class: int = 2) -> List[Path]:
    """自动识别类别文件夹。

    规则：扫描所有子目录，只要该目录下直接或递归包含足够数量图片，就视为候选类别目录。
    为避免把父级总目录也误认为类别目录，优先保留“没有更深层候选子目录”的叶子候选目录。
    """
    candidates: List[Tuple[Path, int]] = []
    for root in search_roots:
        root = Path(root)
        if not root.exists():
            continue
        for directory in [root, *[p for p in root.rglob("*") if p.is_dir()]]:
            images = [p for p in directory.rglob("*") if is_image_file(p)]
            if len(images) >= min_images_per_class:
                candidates.append((directory, len(images)))

    candidate_paths = [p for p, _ in candidates]
    leaf_dirs: List[Path] = []
    for path, _count in candidates:
        has_child_candidate = any(other != path and path in other.parents for other in candidate_paths)
        if not has_child_candidate:
            leaf_dirs.append(path)

    # 去重并按路径排序，保证类别映射稳定。
    unique_leaf_dirs = sorted(set(leaf_dirs), key=lambda p: str(p).lower())
    return unique_leaf_dirs


def collect_valid_images(class_dirs: List[Path]) -> Dict[str, List[Path]]:
    """收集每个类别的有效图片，并过滤坏图。"""
    class_to_images: Dict[str, List[Path]] = {}
    bad_images: List[str] = []

    for class_dir in class_dirs:
        label = class_dir.name.strip()
        images = []
        for image_path in sorted(class_dir.rglob("*")):
            if not is_image_file(image_path):
                continue
            if verify_image(image_path):
                images.append(image_path)
            else:
                bad_images.append(str(image_path))
        if images:
            class_to_images[label] = images

    if bad_images:
        print(f"[警告] 检测到 {len(bad_images)} 张坏图，已跳过。")
    return class_to_images


def split_and_copy(class_to_images: Dict[str, List[Path]], train_dir: Path, test_dir: Path,
                   test_ratio: float, seed: int) -> Dict[str, Dict[str, int]]:
    """按类别划分 train/test，并复制到项目标准目录。"""
    random.seed(seed)
    stats: Dict[str, Dict[str, int]] = {}

    for label, images in sorted(class_to_images.items()):
        images = list(images)
        random.shuffle(images)
        if len(images) < 2:
            print(f"[跳过] 类别 {label} 图片少于 2 张，不适合划分。")
            continue
        test_count = max(1, int(round(len(images) * test_ratio)))
        test_count = min(test_count, len(images) - 1)
        test_images = images[:test_count]
        train_images = images[test_count:]

        for src in train_images:
            safe_copy(src, train_dir / label / src.name)
        for src in test_images:
            # 测试集采用扁平结构，同时把真实类别写进文件名，方便人工检查，但不参与推理。
            safe_name = f"{label}__{src.name}"
            safe_copy(src, test_dir / safe_name)

        stats[label] = {"total": len(images), "train": len(train_images), "test": len(test_images)}
    return stats


def build_sample_submission(test_dir: Path, sample_submission_path: Path, default_label: str) -> int:
    """根据 data/test 生成 sample_submission.csv。"""
    image_ids = [p.name for p in sorted(test_dir.rglob("*")) if is_image_file(p)]
    df = pd.DataFrame({"image_id": image_ids, "label": [default_label] * len(image_ids)})
    ensure_dir(sample_submission_path.parent)
    df.to_csv(sample_submission_path, index=False, encoding="utf-8")
    return len(df)


def save_report(report: dict, report_path: Path) -> None:
    """保存数据整理报告。"""
    ensure_dir(report_path.parent)
    with report_path.open("w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="整理公开天气图片分类数据集用于 smoke test")
    parser.add_argument("--raw-dir", default="data/raw", help="原始 zip 或已解压数据目录")
    parser.add_argument("--zip-path", default=None, help="指定单个 zip 文件；不指定则扫描 raw-dir 下全部 zip")
    parser.add_argument("--extract-dir", default="data/raw/extracted", help="zip 解压目录")
    parser.add_argument("--train-dir", default="data/train", help="输出训练集目录")
    parser.add_argument("--test-dir", default="data/test", help="输出测试集目录")
    parser.add_argument("--sample-submission", default="data/sample_submission.csv", help="输出提交样例路径")
    parser.add_argument("--report-path", default="data/dataset_prepare_report.json", help="输出数据整理报告路径")
    parser.add_argument("--test-ratio", type=float, default=0.2, help="每个类别划分到 test 的比例")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    parser.add_argument("--clean", action="store_true", help="整理前清空 data/train 和 data/test")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    raw_dir = Path(args.raw_dir)
    zip_path = Path(args.zip_path) if args.zip_path else None
    extract_dir = Path(args.extract_dir)
    train_dir = Path(args.train_dir)
    test_dir = Path(args.test_dir)
    sample_submission_path = Path(args.sample_submission)
    report_path = Path(args.report_path)

    if args.clean:
        clean_output_dirs(train_dir, test_dir)
    else:
        ensure_dir(train_dir)
        ensure_dir(test_dir)

    roots = extract_archives(raw_dir, extract_dir, zip_path)
    class_dirs = find_candidate_class_dirs(roots)
    if not class_dirs:
        raise RuntimeError("没有识别到类别文件夹。请确认 zip 已下载到 data/raw/，且数据以 类别名/图片 的形式组织。")

    print("[识别到候选类别目录]")
    for path in class_dirs:
        print(f"- {path}")

    class_to_images = collect_valid_images(class_dirs)
    if len(class_to_images) < 2:
        raise RuntimeError("有效类别数少于 2，无法验证分类训练流程。")

    stats = split_and_copy(class_to_images, train_dir, test_dir, args.test_ratio, args.seed)
    if not stats:
        raise RuntimeError("没有成功划分任何类别，请检查每类图片数量。")

    default_label = sorted(stats.keys())[0]
    sample_rows = build_sample_submission(test_dir, sample_submission_path, default_label)

    report = {
        "purpose": "public_weather_dataset_smoke_test_only",
        "raw_dir": str(raw_dir),
        "train_dir": str(train_dir),
        "test_dir": str(test_dir),
        "sample_submission": str(sample_submission_path),
        "test_ratio": args.test_ratio,
        "seed": args.seed,
        "num_classes": len(stats),
        "class_stats": stats,
        "total_train": sum(x["train"] for x in stats.values()),
        "total_test": sum(x["test"] for x in stats.values()),
        "sample_submission_rows": sample_rows,
    }
    save_report(report, report_path)

    print("\n[数据整理完成]")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
