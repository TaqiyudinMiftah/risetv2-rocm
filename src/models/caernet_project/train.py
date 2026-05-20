"""
train.py  –  CAER-Net training pipeline
Lee et al., ICCV 2019.

Usage
-----
    python train.py --config caernet.yaml
    python train.py --config caernet.yaml --resume checkpoints/last.pth
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import random
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
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

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("train")


# ===========================================================================
# Utilities
# ===========================================================================

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False


def resolve_device(spec: str) -> torch.device:
    if spec:
        return torch.device(spec)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


# ===========================================================================
# Model factory
# ===========================================================================

def build_model(cfg: Dict, num_classes: int) -> nn.Module:
    m   = cfg["model"]
    mode = cfg["train"].get("stream_mode", "multimodal")

    if mode == "multimodal":
        return CAERNet(
            num_classes=num_classes,
            backbone=m.get("backbone", "resnet18"),
            pretrained=m.get("pretrained", True),
            dropout=m.get("dropout", 0.5),
        )
    if mode in ("face", "context"):
        return SingleStreamNet(
            num_classes=num_classes,
            stream=mode,
            backbone=m.get("backbone", "resnet18"),
            pretrained=m.get("pretrained", True),
            dropout=m.get("dropout", 0.5),
        )
    raise ValueError(f"Unknown stream_mode: '{mode}' — use multimodal | face | context")


# ===========================================================================
# Optimizer & scheduler
# ===========================================================================

def build_optimizer(model: nn.Module, cfg: Dict) -> optim.Optimizer:
    t = cfg["train"]
    lr = float(t.get("lr", 1e-4))
    wd = float(t.get("weight_decay", 1e-4))
    # SGD + Nesterov momentum – standard for vision fine-tuning
    return optim.SGD(
        model.parameters(),
        lr=lr,
        momentum=0.9,
        weight_decay=wd,
        nesterov=True,
    )


def build_scheduler(
    optimizer: optim.Optimizer,
    cfg: Dict,
    steps_per_epoch: int,
) -> optim.lr_scheduler._LRScheduler:
    t         = cfg["train"]
    epochs    = int(t.get("num_epochs", 30))
    pretrained = cfg["model"].get("pretrained", True)

    if pretrained:
        # Fine-tuning: warmup (3 epochs) then cosine decay
        warmup_steps = 3 * steps_per_epoch
        total_steps  = epochs * steps_per_epoch

        def lr_lambda(step: int) -> float:
            if step < warmup_steps:
                return (step + 1) / warmup_steps
            progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
            return 0.5 * (1.0 + np.cos(np.pi * progress))

        return optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    else:
        # From scratch (paper): step LR – drop by ×0.1 every 4 epochs
        return optim.lr_scheduler.StepLR(optimizer, step_size=4, gamma=0.1)


# ===========================================================================
# Train / Eval loops
# ===========================================================================

def train_one_epoch(
    model:     nn.Module,
    loader:    DataLoader,
    optimiser: optim.Optimizer,
    scheduler: optim.lr_scheduler._LRScheduler,
    criterion: nn.Module,
    device:    torch.device,
    epoch:     int,
    is_step_scheduler: bool,       # True → step per batch; False → step per epoch
) -> Dict[str, float]:
    model.train()
    total_loss = total_correct = total_n = 0

    for i, batch in enumerate(loader, 1):
        face    = batch["face"].to(device, non_blocking=True)
        context = batch["context"].to(device, non_blocking=True)
        labels  = batch["label"].to(device, non_blocking=True)

        optimiser.zero_grad(set_to_none=True)

        out    = model(face, context)
        logits = out["logits"]
        loss   = criterion(logits, labels)
        loss.backward()

        # Gradient clipping (helps with divergence on smaller batches)
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)

        optimiser.step()
        if is_step_scheduler:
            scheduler.step()

        preds         = logits.argmax(1)
        bs            = labels.size(0)
        total_correct += (preds == labels).sum().item()
        total_loss    += loss.item() * bs
        total_n       += bs

        if i % 100 == 0 or i == len(loader):
            cur_lr = optimiser.param_groups[0]["lr"]
            logger.info(
                f"  [E{epoch:02d} {i:4d}/{len(loader)}]  "
                f"loss={loss.item():.4f}  "
                f"acc={total_correct/total_n*100:.2f}%  "
                f"lr={cur_lr:.2e}"
            )

    return {
        "loss": total_loss / total_n,
        "acc":  total_correct / total_n * 100,
    }


@torch.no_grad()
def evaluate(
    model:     nn.Module,
    loader:    DataLoader,
    criterion: nn.Module,
    device:    torch.device,
) -> Dict[str, object]:
    model.eval()
    total_loss = total_correct = total_n = 0
    all_preds:  List[int] = []
    all_labels: List[int] = []

    for batch in loader:
        face    = batch["face"].to(device, non_blocking=True)
        context = batch["context"].to(device, non_blocking=True)
        labels  = batch["label"].to(device, non_blocking=True)

        out    = model(face, context)
        logits = out["logits"]
        loss   = criterion(logits, labels)
        preds  = logits.argmax(1)

        bs             = labels.size(0)
        total_correct += (preds == labels).sum().item()
        total_loss    += loss.item() * bs
        total_n       += bs
        all_preds.extend(preds.cpu().tolist())
        all_labels.extend(labels.cpu().tolist())

    return {
        "loss":   total_loss / total_n,
        "acc":    total_correct / total_n * 100,
        "preds":  all_preds,
        "labels": all_labels,
    }


def per_class_accuracy(preds: List[int], labels: List[int]) -> Dict[str, float]:
    result: Dict[str, float] = {}
    for idx, cls in enumerate(EMOTION_CLASSES):
        cls_true  = [p == l for p, l in zip(preds, labels) if l == idx]
        result[cls] = (sum(cls_true) / len(cls_true) * 100) if cls_true else 0.0
    return result


# ===========================================================================
# Main
# ===========================================================================

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="caernet.yaml")
    parser.add_argument("--resume", default=None,
                        help="Path to checkpoint (.pth) to resume from")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg: Dict = yaml.safe_load(f)

    logger.info("=" * 60)
    logger.info(f"Config: {args.config}")
    logger.info(json.dumps(cfg, indent=2))
    logger.info("=" * 60)

    # ── Reproducibility ────────────────────────────────────────────────
    seed = int(cfg.get("seed", 42))
    set_seed(seed)

    device = resolve_device(cfg["train"].get("device", ""))
    logger.info(f"Device: {device}")

    # ── Dataset ────────────────────────────────────────────────────────
    d     = cfg["dataset"]
    root  = d["dataset_root"]
    exts  = d.get("image_extensions", [".png", ".jpg", ".jpeg", ".bmp", ".webp"])
    sz    = int(d.get("image_size", 224))

    train_s, test_s = discover_samples(root, exts)
    if not train_s:
        raise RuntimeError(f"No samples found under '{root}'.")

    val_s: List = []
    if d.get("create_val_split", True):
        train_s, val_s = split_train_val(
            train_s,
            val_ratio=float(d.get("val_ratio", 0.10)),
            seed=seed,
        )

    # Shared face detector (avoid re-creating per dataset)
    detector = FaceDetector()
    logger.info("OpenCV Haar-Cascade face detector ready.")

    t   = cfg["train"]
    bsz = int(t.get("batch_size", 32))
    nw  = int(t.get("num_workers", 4))

    train_ds = CAERSDataset(train_s, sz, augment=True,  face_detector=detector)
    val_ds   = CAERSDataset(val_s or test_s, sz, augment=False, face_detector=detector)
    test_ds  = CAERSDataset(test_s,  sz, augment=False, face_detector=detector)

    logger.info(
        f"Splits → train:{len(train_ds):,}  val:{len(val_ds):,}  test:{len(test_ds):,}"
    )

    train_loader = DataLoader(
        train_ds, batch_size=bsz, shuffle=True,
        num_workers=nw, pin_memory=(device.type != "cpu"),
        drop_last=True, persistent_workers=(nw > 0),
    )
    val_loader = DataLoader(
        val_ds, batch_size=bsz, shuffle=False,
        num_workers=nw, pin_memory=(device.type != "cpu"),
        persistent_workers=(nw > 0),
    )
    test_loader = DataLoader(
        test_ds, batch_size=bsz, shuffle=False,
        num_workers=nw, pin_memory=(device.type != "cpu"),
        persistent_workers=(nw > 0),
    )

    # ── Model ─────────────────────────────────────────────────────────
    model = build_model(cfg, NUM_CLASSES).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Model: {model.__class__.__name__}  ({n_params:,} trainable params)")

    # ── Loss ─────────────────────────────────────────────────────────
    # label_smoothing=0.1 acts as regulariser; remove if strict paper replication
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

    # ── Optimiser & Scheduler ─────────────────────────────────────────
    optimiser = build_optimizer(model, cfg)
    pretrained        = cfg["model"].get("pretrained", True)
    is_step_scheduler = pretrained          # step per batch for warmup+cosine
    scheduler = build_scheduler(optimiser, cfg, steps_per_epoch=len(train_loader))

    # ── Resume ────────────────────────────────────────────────────────
    start_epoch   = 1
    best_val_acc  = 0.0
    if args.resume:
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        optimiser.load_state_dict(ckpt["optimizer_state_dict"])
        start_epoch  = ckpt["epoch"] + 1
        best_val_acc = ckpt.get("best_val_acc", 0.0)
        logger.info(f"Resumed from epoch {ckpt['epoch']}  (best val: {best_val_acc:.2f}%)")

    # ── Output paths ──────────────────────────────────────────────────
    save_dir = Path(t.get("save_dir", "checkpoints"))
    save_dir.mkdir(parents=True, exist_ok=True)

    out     = cfg.get("outputs", {})
    manifest_path    = Path(out.get("manifest_path",    "artifacts/caers/manifest_caers.jsonl"))
    diagnostics_path = Path(out.get("diagnostics_path", "artifacts/caers/diagnostics_caers.json"))
    for p in (manifest_path, diagnostics_path):
        p.parent.mkdir(parents=True, exist_ok=True)

    num_epochs = int(t.get("num_epochs", 30))
    history: List[Dict] = []

    # ── Training loop ─────────────────────────────────────────────────
    logger.info(f"Training for {num_epochs} epochs …")
    t0_total = time.time()

    for epoch in range(start_epoch, num_epochs + 1):
        t0 = time.time()

        train_m = train_one_epoch(
            model, train_loader, optimiser, scheduler, criterion,
            device, epoch, is_step_scheduler=is_step_scheduler,
        )
        val_m = evaluate(model, val_loader, criterion, device)

        if not is_step_scheduler:           # epoch-based scheduler (StepLR)
            scheduler.step()

        elapsed = time.time() - t0
        cur_lr  = optimiser.param_groups[0]["lr"]

        logger.info(
            f"Epoch {epoch:02d}/{num_epochs}  "
            f"train_loss={train_m['loss']:.4f}  train_acc={train_m['acc']:.2f}%  "
            f"val_loss={val_m['loss']:.4f}  val_acc={val_m['acc']:.2f}%  "
            f"lr={cur_lr:.2e}  ({elapsed:.1f}s)"
        )

        row = {
            "epoch":      epoch,
            "train_loss": round(train_m["loss"], 5),
            "train_acc":  round(train_m["acc"],  4),
            "val_loss":   round(val_m["loss"],   5),
            "val_acc":    round(val_m["acc"],     4),
            "lr":         cur_lr,
        }
        history.append(row)

        # Write manifest (one JSON line per epoch)
        with open(manifest_path, "a") as f:
            f.write(json.dumps(row) + "\n")

        # Checkpoint
        is_best = val_m["acc"] > best_val_acc
        if is_best:
            best_val_acc = val_m["acc"]

        state = {
            "epoch":               epoch,
            "model_state_dict":    model.state_dict(),
            "optimizer_state_dict":optimiser.state_dict(),
            "val_acc":             val_m["acc"],
            "best_val_acc":        best_val_acc,
            "config":              cfg,
        }
        torch.save(state, save_dir / "last.pth")
        if is_best:
            torch.save(state, save_dir / "best.pth")
            logger.info(f"  ★ New best val acc: {best_val_acc:.2f}%  → best.pth saved")

    total_time = time.time() - t0_total
    logger.info(f"Training done in {total_time/60:.1f} min.")

    # ── Final test evaluation ──────────────────────────────────────────
    logger.info("Loading best.pth for final test evaluation …")
    best_ckpt = torch.load(save_dir / "best.pth", map_location=device)
    model.load_state_dict(best_ckpt["model_state_dict"])

    # Use test set if available, otherwise val set
    final_loader = test_loader if len(test_ds) > 0 else val_loader
    final_split  = "test" if len(test_ds) > 0 else "val"

    test_m = evaluate(model, final_loader, criterion, device)
    pc_acc = per_class_accuracy(test_m["preds"], test_m["labels"])

    logger.info(f"\n{'='*60}")
    logger.info(f"[{final_split.upper()}] Accuracy: {test_m['acc']:.2f}%")
    logger.info("Per-class accuracy:")
    for cls, acc in pc_acc.items():
        logger.info(f"  {cls:>10s}: {acc:.2f}%")
    logger.info(
        f"Face-detector miss rate: {detector.miss_rate*100:.1f}%"
    )
    logger.info("=" * 60)

    # ── Save diagnostics ──────────────────────────────────────────────
    diagnostics = {
        "method":          cfg.get("method", "caernet"),
        "config":          cfg,
        "best_val_acc":    best_val_acc,
        f"{final_split}_acc": test_m["acc"],
        "per_class_acc":   pc_acc,
        "splits": {
            "train": len(train_ds),
            "val":   len(val_ds),
            "test":  len(test_ds),
        },
        "face_detector_miss_rate": detector.miss_rate,
        "total_training_minutes":  total_time / 60,
        "history": history,
    }
    with open(diagnostics_path, "w") as f:
        json.dump(diagnostics, f, indent=2)
    logger.info(f"Diagnostics → {diagnostics_path}")


if __name__ == "__main__":
    main()
