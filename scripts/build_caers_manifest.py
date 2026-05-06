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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build CAER-S manifest and diagnostics")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)

    result = build_caers_manifest(
        dataset_root=cfg.dataset.dataset_root,
        image_extensions=cfg.dataset.image_extensions,
        create_val_split=cfg.dataset.create_val_split,
        val_ratio=cfg.dataset.val_ratio,
        seed=cfg.seed,
    )

    manifest_path = Path(cfg.outputs.manifest_path)
    diagnostics_path = Path(cfg.outputs.diagnostics_path)

    write_jsonl(manifest_path, result.rows)
    write_json(diagnostics_path, result.diagnostics)

    print(f"Manifest written: {manifest_path}")
    print(f"Diagnostics written: {diagnostics_path}")
    print(f"Split counts: {result.diagnostics['split_counts']}")


if __name__ == "__main__":
    main()
