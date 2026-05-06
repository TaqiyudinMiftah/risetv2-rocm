from __future__ import annotations

import random
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path


@dataclass
class ManifestBuildResult:
    rows: list[dict[str, object]]
    diagnostics: dict[str, object]


def _collect_split_rows(
    dataset_root: Path,
    split_name: str,
    image_extensions: tuple[str, ...],
) -> list[dict[str, object]]:
    split_dir = dataset_root / split_name
    if not split_dir.exists():
        return []

    rows: list[dict[str, object]] = []
    for class_dir in sorted(split_dir.iterdir()):
        if not class_dir.is_dir():
            continue
        label = class_dir.name

        for image_path in sorted(class_dir.rglob("*")):
            if not image_path.is_file():
                continue
            if image_path.suffix.lower() not in image_extensions:
                continue

            rel_path = image_path.relative_to(dataset_root).as_posix()
            sample_id = rel_path.replace("/", "__").replace(".", "_")
            rows.append(
                {
                    "sample_id": f"{split_name}__{sample_id}",
                    "image_path": rel_path,
                    "label": label,
                    "split": split_name,
                    "face_bbox": None,
                }
            )
    return rows


def _stratified_holdout(
    rows: list[dict[str, object]],
    holdout_ratio: float,
    seed: int,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    by_label: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        by_label[str(row["label"])].append(row)

    rnd = random.Random(seed)
    keep_rows: list[dict[str, object]] = []
    holdout_rows: list[dict[str, object]] = []

    for label, label_rows in by_label.items():
        pool = list(label_rows)
        rnd.shuffle(pool)
        holdout_n = max(1, int(round(len(pool) * holdout_ratio)))
        holdout_n = min(holdout_n, len(pool) - 1) if len(pool) > 1 else 0
        holdout = pool[:holdout_n]
        keep = pool[holdout_n:]

        for item in holdout:
            copied = dict(item)
            copied["split"] = "val"
            copied["sample_id"] = str(copied["sample_id"]).replace("train__", "val__", 1)
            holdout_rows.append(copied)

        keep_rows.extend(keep)

    return keep_rows, holdout_rows


def _split_counts(rows: list[dict[str, object]]) -> dict[str, int]:
    counter: Counter[str] = Counter()
    for row in rows:
        counter[str(row["split"])] += 1
    return dict(sorted(counter.items()))


def _class_counts(rows: list[dict[str, object]], split_name: str) -> dict[str, int]:
    counter: Counter[str] = Counter()
    for row in rows:
        if str(row["split"]) == split_name:
            counter[str(row["label"])] += 1
    return dict(sorted(counter.items()))


def build_caers_manifest(
    dataset_root: Path,
    image_extensions: tuple[str, ...],
    create_val_split: bool,
    val_ratio: float,
    seed: int,
) -> ManifestBuildResult:
    train_rows = _collect_split_rows(dataset_root, "train", image_extensions)
    test_rows = _collect_split_rows(dataset_root, "test", image_extensions)

    if len(train_rows) == 0 or len(test_rows) == 0:
        raise FileNotFoundError(
            "Missing CAER-S split data. Expected dataset_root/train and dataset_root/test with class folders."
        )

    if create_val_split:
        train_rows, val_rows = _stratified_holdout(train_rows, val_ratio, seed)
    else:
        val_rows = []

    all_rows = train_rows + val_rows + test_rows

    diagnostics = {
        "dataset_root": str(dataset_root),
        "total_samples": len(all_rows),
        "split_counts": _split_counts(all_rows),
        "class_counts": {
            "train": _class_counts(all_rows, "train"),
            "val": _class_counts(all_rows, "val"),
            "test": _class_counts(all_rows, "test"),
        },
        "labels": sorted({str(r["label"]) for r in all_rows}),
    }

    return ManifestBuildResult(rows=all_rows, diagnostics=diagnostics)
