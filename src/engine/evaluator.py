from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from engine.metrics import MetricTracker


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader[Any],
    criterion: nn.Module,
    device: torch.device,
) -> dict[str, float]:
    model.eval()
    tracker = MetricTracker()

    for batch in tqdm(loader, desc="eval", leave=False):
        face = batch["face_image"].to(device)
        context = batch["context_image"].to(device)
        labels = batch["label"].to(device)

        out = model(face, context)
        loss = criterion(out["logits"], labels)
        tracker.update(loss.item(), out["logits"], labels)

    return tracker.summary()


@torch.no_grad()
def evaluate_per_class(
    model: nn.Module,
    loader: DataLoader[Any],
    device: torch.device,
    index_to_label: dict[int, str],
) -> dict[str, Any]:
    model.eval()
    num_classes = len(index_to_label)
    correct_per_class = torch.zeros(num_classes, dtype=torch.long)
    total_per_class = torch.zeros(num_classes, dtype=torch.long)

    for batch in tqdm(loader, desc="eval_per_class", leave=False):
        face = batch["face_image"].to(device)
        context = batch["context_image"].to(device)
        labels = batch["label"].to(device)

        out = model(face, context)
        preds = out["logits"].argmax(dim=1)

        for c in range(num_classes):
            mask = labels == c
            if mask.any():
                total_per_class[c] += mask.sum().item()
                correct_per_class[c] += (preds[mask] == labels[mask]).sum().item()

    per_class_acc: dict[str, float] = {}
    for idx, label_name in index_to_label.items():
        total = int(total_per_class[idx])
        correct = int(correct_per_class[idx])
        per_class_acc[label_name] = (correct / total * 100.0) if total > 0 else 0.0

    overall_acc = float(correct_per_class.sum().item() / max(1, total_per_class.sum().item()) * 100.0)

    return {
        "overall_acc": overall_acc,
        "per_class_acc": per_class_acc,
    }
