from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from engine.metrics import MetricTracker


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader[Any],
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    loss_fn: Any | None = None,
) -> dict[str, float]:
    model.train()
    tracker = MetricTracker()

    pbar = tqdm(loader, desc="train", leave=False)
    for batch in pbar:
        face = batch["face_image"].to(device)
        context = batch["context_image"].to(device)
        labels = batch["label"].to(device)

        optimizer.zero_grad()
        out = model(face, context)
        if loss_fn is not None:
            loss = loss_fn(out, labels)
        else:
            loss = criterion(out["logits"], labels)
        loss.backward()
        optimizer.step()

        tracker.update(loss.item(), out["logits"].detach(), labels)
        pbar.set_postfix({"loss": f"{tracker.avg_loss:.4f}", "acc1": f"{tracker.avg_acc1:.2f}%"})

    return tracker.summary()
