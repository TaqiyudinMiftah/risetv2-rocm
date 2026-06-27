from __future__ import annotations

import random
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path


@dataclass
class ManifestBuildResult:
    rows: list[dict[str, object]]
    diagnostics: dict[str, object]


def _normalize_label(label: str) -> str:
    """Normalize label names to match the repository's 7 emotion classes."""
    label_map = {
        "Anger": "Angry",  # CAER-S official folders sometimes use "Anger".
    }
    return label_map.get(label, label)


def _bbox_from_values(values: list[str]) -> list[int]:
    """Parse bbox coordinates from strings, accepting integer-like floats too."""
    return [int(float(v.strip())) for v in values]


def _is_split_prefixed(rel_path: str) -> bool:
    first_part = rel_path.split("/", 1)[0].lower()
    return first_part in {"train", "val", "valid", "validation", "test"}


def _logical_detection_rel_path(rel_path: str) -> str:
    """Return path without split prefix, e.g. train/Happy/a.png -> Happy/a.png."""
    normalized = rel_path.replace("\\", "/").strip().lstrip("/")
    if _is_split_prefixed(normalized):
        parts = normalized.split("/", 1)
        return parts[1] if len(parts) == 2 else normalized
    return normalized


def _default_image_path_for_detection(split_name: str, rel_path: str) -> str:
    """Map official detector relative paths to dataset-root-relative image paths."""
    normalized = rel_path.replace("\\", "/").strip().lstrip("/")
    if _is_split_prefixed(normalized):
        return normalized

    # Official CAER-S val detections usually point to images physically stored
    # under train/, because CAER-S only ships train/ and test/ folders.
    storage_split = "train" if split_name == "val" else split_name
    return f"{storage_split}/{normalized}"


def _find_detection_image_path(dataset_root: Path, split_name: str, rel_path: str) -> str | None:
    """Find the actual dataset-root-relative image path for an official detection row."""
    normalized = rel_path.replace("\\", "/").strip().lstrip("/")
    logical_rel = _logical_detection_rel_path(normalized)

    candidates: list[Path] = []
    if _is_split_prefixed(normalized):
        candidates.append(dataset_root / normalized)
    else:
        candidates.append(dataset_root / split_name / logical_rel)
        if split_name == "val":
            candidates.append(dataset_root / "train" / logical_rel)

    # Keep the default fallback last so existing configs can still build a
    # manifest even if the image tree is not mounted during dry-run checks.
    for candidate in candidates:
        if candidate.exists():
            return candidate.relative_to(dataset_root).as_posix()
    return None


def _parse_official_detection_file(file_path: Path) -> list[tuple[str, list[int]]]:
    """Parse official CAER-S detector files.

    Expected line format from the CAER PyTorch reproduction repository:
        relative_image_path,label,x1,y1,x2,y2

    The second column can be a class index or label name and is intentionally
    ignored because the class label is derivable from the relative path folder.
    """
    if not file_path.exists():
        raise FileNotFoundError(f"Official face bbox file not found: {file_path}")

    rows: list[tuple[str, list[int]]] = []
    with file_path.open("r", encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue

            parts = [part.strip() for part in line.split(",")]
            if len(parts) != 6:
                raise ValueError(
                    f"Invalid detection row in {file_path} line {line_number}: "
                    "expected 6 comma-separated fields "
                    "(relative_image_path,label,x1,y1,x2,y2)."
                )

            rel_path = parts[0].replace("\\", "/").lstrip("/")
            bbox = _bbox_from_values(parts[2:6])
            rows.append((rel_path, bbox))
    return rows


def _collect_detection_rows(
    dataset_root: Path,
    split_name: str,
    detection_file: Path,
) -> tuple[list[dict[str, object]], dict[str, object]]:
    parsed_rows = _parse_official_detection_file(detection_file)
    rows: list[dict[str, object]] = []
    missing_images: list[str] = []

    for rel_path, bbox in parsed_rows:
        logical_rel = _logical_detection_rel_path(rel_path)
        parts = logical_rel.split("/")
        if len(parts) < 2:
            raise ValueError(
                f"Invalid image path in official detection file {detection_file}: {rel_path}"
            )

        folder_label = parts[0]
        label = _normalize_label(folder_label)
        resolved_image_path = _find_detection_image_path(dataset_root, split_name, rel_path)
        if resolved_image_path is None:
            resolved_image_path = _default_image_path_for_detection(split_name, rel_path)
            missing_images.append(resolved_image_path)

        sample_key = logical_rel.replace("/", "__").replace(".", "_")
        rows.append(
            {
                "sample_id": f"{split_name}__{sample_key}",
                "image_path": resolved_image_path,
                "label": label,
                "split": split_name,
                "face_bbox": bbox,
            }
        )

    stats = {
        "file": str(detection_file),
        "rows": len(rows),
        "missing_images": len(missing_images),
        "missing_image_examples": missing_images[:20],
    }
    return rows, stats


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
        label = _normalize_label(class_dir.name)

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


def _bbox_counts(rows: list[dict[str, object]]) -> dict[str, int]:
    counter: Counter[str] = Counter()
    for row in rows:
        split = str(row["split"])
        if row.get("face_bbox"):
            counter[split] += 1
    return dict(sorted(counter.items()))


def build_caers_manifest(
    dataset_root: Path,
    image_extensions: tuple[str, ...],
    create_val_split: bool,
    val_ratio: float,
    seed: int,
    detection_files: dict[str, Path] | None = None,
) -> ManifestBuildResult:
    detection_files = {k: Path(v) for k, v in (detection_files or {}).items() if v}
    valid_splits = {"train", "val", "test"}
    unknown_splits = sorted(set(detection_files) - valid_splits)
    if unknown_splits:
        raise ValueError(f"Unsupported detection file split(s): {unknown_splits}")

    detection_stats: dict[str, object] = {}

    if detection_files:
        if "train" in detection_files:
            train_rows, detection_stats["train"] = _collect_detection_rows(
                dataset_root, "train", detection_files["train"]
            )
        else:
            train_rows = _collect_split_rows(dataset_root, "train", image_extensions)

        if "test" in detection_files:
            test_rows, detection_stats["test"] = _collect_detection_rows(
                dataset_root, "test", detection_files["test"]
            )
        else:
            test_rows = _collect_split_rows(dataset_root, "test", image_extensions)

        if "val" in detection_files:
            val_rows, detection_stats["val"] = _collect_detection_rows(
                dataset_root, "val", detection_files["val"]
            )
            val_split_source = "official_detection_file"
        elif create_val_split:
            train_rows, val_rows = _stratified_holdout(train_rows, val_ratio, seed)
            val_split_source = "stratified_holdout"
        else:
            val_rows = []
            val_split_source = "none"
    else:
        train_rows = _collect_split_rows(dataset_root, "train", image_extensions)
        test_rows = _collect_split_rows(dataset_root, "test", image_extensions)

        if create_val_split:
            train_rows, val_rows = _stratified_holdout(train_rows, val_ratio, seed)
            val_split_source = "stratified_holdout"
        else:
            val_rows = []
            val_split_source = "none"

    if len(train_rows) == 0 or len(test_rows) == 0:
        raise FileNotFoundError(
            "Missing CAER-S split data. Expected dataset_root/train and dataset_root/test "
            "with class folders, or official detection files for train/test."
        )

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
        "face_bbox_source": "official_detection_files" if detection_files else "none",
        "face_bbox_counts": _bbox_counts(all_rows),
        "val_split_source": val_split_source,
        "detection_files": {k: str(v) for k, v in detection_files.items()},
        "detection_file_stats": detection_stats,
    }

    return ManifestBuildResult(rows=all_rows, diagnostics=diagnostics)
