from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from config.config import load_config
from utils.data_manifest import build_caers_manifest
from utils.io_utils import write_json, write_jsonl
from utils.logger import setup_logger


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build CAER-S manifest and diagnostics")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config")
    parser.add_argument(
        "--train-detections",
        type=str,
        default="",
        help="Official CAER-S train bbox file: relative_image_path,label,x1,y1,x2,y2",
    )
    parser.add_argument(
        "--val-detections",
        type=str,
        default="",
        help="Official CAER-S val bbox file. Overrides random val split when provided.",
    )
    parser.add_argument(
        "--test-detections",
        type=str,
        default="",
        help="Official CAER-S test bbox file: relative_image_path,label,x1,y1,x2,y2",
    )
    return parser.parse_args()


def _resolve_detection_overrides(args: argparse.Namespace) -> dict[str, Path]:
    overrides: dict[str, Path] = {}
    if args.train_detections:
        overrides["train"] = Path(args.train_detections).expanduser()
    if args.val_detections:
        overrides["val"] = Path(args.val_detections).expanduser()
    if args.test_detections:
        overrides["test"] = Path(args.test_detections).expanduser()
    return overrides


def main() -> None:
    args = parse_args()

    logger = setup_logger(name="caers_manifest", log_dir=PROJECT_ROOT / "logs")
    logger.info("Starting manifest build script")
    logger.info("Config path: %s", args.config)

    try:
        cfg = load_config(args.config)
        detection_files = dict(cfg.dataset.detection_files)
        detection_files.update(_resolve_detection_overrides(args))

        result = build_caers_manifest(
            dataset_root=cfg.dataset.dataset_root,
            image_extensions=cfg.dataset.image_extensions,
            create_val_split=cfg.dataset.create_val_split,
            val_ratio=cfg.dataset.val_ratio,
            seed=cfg.seed,
            detection_files=detection_files,
        )

        manifest_path = Path(cfg.outputs.manifest_path)
        diagnostics_path = Path(cfg.outputs.diagnostics_path)

        write_jsonl(manifest_path, result.rows)
        write_json(diagnostics_path, result.diagnostics)

        logger.info("Manifest written: %s", manifest_path)
        logger.info("Diagnostics written: %s", diagnostics_path)
        logger.info("Split counts: %s", result.diagnostics["split_counts"])
        logger.info("Face bbox source: %s", result.diagnostics["face_bbox_source"])
        logger.info("Face bbox counts: %s", result.diagnostics["face_bbox_counts"])
    except Exception:
        logger.exception("Manifest build crashed with an exception")
        raise


if __name__ == "__main__":
    main()
