"""
evaluate.py  –  Evaluate a trained CAER-Net checkpoint.

Usage
-----
    python evaluate.py --checkpoint checkpoints/best.pth --config caernet.yaml
    python evaluate.py --checkpoint checkpoints/best.pth --config caernet.yaml --split test
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path
from typing import Dict, List

import torch
import torch.nn as nn
import yaml
from torch.utils.data import DataLoader

from datasets.caers import (
    EMOTION_CLASSES,
    NUM_CLASSES,
    CAERSDataset,
    FaceDetector,
    discover_samples,
    split_train_val,
)
from models.caernet import CAERNet, SingleStreamNet
from train import build_model, per_class_accuracy

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("evaluate")


# ---------------------------------------------------------------------------

@torch.no_grad()
def run_eval(
    model:    nn.Module,
    loader:   DataLoader,
    device:   torch.device,
) -> Dict:
    model.eval()
    all_preds:  List[int] = []
    all_labels: List[int] = []
    correct = total = 0

    t0 = time.time()
    for batch in loader:
        face    = batch["face"].to(device, non_blocking=True)
        context = batch["context"].to(device, non_blocking=True)
        labels  = batch["label"].to(device, non_blocking=True)

        out    = model(face, context)
        preds  = out["logits"].argmax(1)

        correct += (preds == labels).sum().item()
        total   += labels.size(0)
        all_preds.extend(preds.cpu().tolist())
        all_labels.extend(labels.cpu().tolist())

    acc = correct / total * 100
    elapsed = time.time() - t0

    return {
        "acc":     acc,
        "correct": correct,
        "total":   total,
        "preds":   all_preds,
        "labels":  all_labels,
        "elapsed_s": elapsed,
    }


def confusion_matrix_str(preds: List[int], labels: List[int]) -> str:
    n = NUM_CLASSES
    mat = [[0] * n for _ in range(n)]
    for p, l in zip(preds, labels):
        mat[l][p] += 1

    # Header
    short = [c[:3] for c in EMOTION_CLASSES]
    header = "      " + "  ".join(f"{s:>5}" for s in short)
    lines  = [header, "-" * len(header)]
    for i, row in enumerate(mat):
        cells = "  ".join(f"{v:>5}" for v in row)
        lines.append(f"{EMOTION_CLASSES[i]:>6}  {cells}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--config",     required=True)
    parser.add_argument("--split",      default="test", choices=["train", "val", "test"])
    parser.add_argument("--output",     default=None,
                        help="Save detailed results to this JSON file")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg: Dict = yaml.safe_load(f)

    # Device
    spec   = cfg["train"].get("device", "")
    if spec:
        device = torch.device(spec)
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    logger.info(f"Device: {device}")

    # Dataset
    d    = cfg["dataset"]
    exts = d.get("image_extensions", [".png", ".jpg", ".jpeg", ".bmp", ".webp"])
    sz   = int(d.get("image_size", 224))

    train_s, test_s = discover_samples(d["dataset_root"], exts)

    seed  = int(cfg.get("seed", 42))
    val_s: List = []
    if d.get("create_val_split", True):
        train_s, val_s = split_train_val(train_s, float(d.get("val_ratio", 0.10)), seed)

    split_map = {"train": train_s, "val": val_s or test_s, "test": test_s}
    samples   = split_map[args.split]
    if not samples:
        logger.warning(f"No samples found for split='{args.split}'.")
        return

    detector = FaceDetector()
    dataset  = CAERSDataset(samples, sz, augment=False, face_detector=detector)
    loader   = DataLoader(
        dataset,
        batch_size=int(cfg["train"].get("batch_size", 32)),
        shuffle=False,
        num_workers=int(cfg["train"].get("num_workers", 4)),
        pin_memory=(device.type != "cpu"),
    )
    logger.info(f"Evaluating {len(dataset):,} samples from '{args.split}' split.")

    # Load checkpoint
    ckpt = torch.load(args.checkpoint, map_location=device)
    model = build_model(cfg, NUM_CLASSES).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    logger.info(
        f"Loaded checkpoint from epoch {ckpt.get('epoch', '?')} "
        f"(best val: {ckpt.get('best_val_acc', '?'):.2f}%)"
    )

    results = run_eval(model, loader, device)

    pc_acc = per_class_accuracy(results["preds"], results["labels"])
    cm_str = confusion_matrix_str(results["preds"], results["labels"])

    # Report
    logger.info(f"\n{'='*60}")
    logger.info(f"[{args.split.upper()}] Accuracy: {results['acc']:.2f}%  "
                f"({results['correct']}/{results['total']})  "
                f"[{results['elapsed_s']:.1f}s]")
    logger.info("Per-class accuracy:")
    for cls, acc in pc_acc.items():
        bar = "█" * int(acc / 5)
        logger.info(f"  {cls:>10s}: {acc:6.2f}%  {bar}")
    logger.info("\nConfusion matrix (row=true, col=pred):\n" + cm_str)
    logger.info("=" * 60)

    if args.output:
        out = {
            "checkpoint":    args.checkpoint,
            "split":         args.split,
            "accuracy":      results["acc"],
            "per_class_acc": pc_acc,
        }
        with open(args.output, "w") as f:
            json.dump(out, f, indent=2)
        logger.info(f"Results saved to {args.output}")


if __name__ == "__main__":
    main()
