from __future__ import annotations

import argparse
import sys
from pathlib import Path

from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from config.config import load_config
from datasets.caers_dataset import CAERSTwoStreamDataset
from utils.logger import setup_logger


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Smoke test CAER-S data pipeline")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config")
    parser.add_argument("--batch-size", type=int, default=4, help="Batch size for smoke test")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    logger = setup_logger(name="caers_smoke", log_dir=PROJECT_ROOT / "logs")
    logger.info("Starting smoke test script")
    logger.info("Config path: %s", args.config)

    try:
        cfg = load_config(args.config)

        ds_train = CAERSTwoStreamDataset(
            manifest_path=cfg.outputs.manifest_path,
            dataset_root=cfg.dataset.dataset_root,
            split="train",
            image_size=cfg.dataset.image_size,
        )

        loader = DataLoader(ds_train, batch_size=args.batch_size, shuffle=False, num_workers=0)
        batch = next(iter(loader))

        logger.info("Smoke test success")
        logger.info("Train samples: %d", len(ds_train))
        logger.info("Num classes: %d", len(ds_train.label_to_index))
        logger.info("face_image shape: %s", tuple(batch["face_image"].shape))
        logger.info("context_image shape: %s", tuple(batch["context_image"].shape))
        logger.info("label shape: %s", tuple(batch["label"].shape))
    except Exception:
        logger.exception("Smoke test crashed with an exception")
        raise


if __name__ == "__main__":
    main()
