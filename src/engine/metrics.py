from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm


def accuracy_topk(
    logits: torch.Tensor,
    targets: torch.Tensor,
    topk: tuple[int, ...] = (1,),
) -> list[float]:
    """Compute top-k accuracies."""
    maxk = max(topk)
    batch_size = targets.size(0)
    _, pred = logits.topk(maxk, dim=1, largest=True, sorted=True)
    pred = pred.t()
    correct = pred.eq(targets.view(1, -1).expand_as(pred))
    res: list[float] = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
        res.append(float(correct_k.mul_(100.0 / batch_size).item()))
    return res


class MetricTracker:
    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.total_loss = 0.0
        self.total_samples = 0
        self.total_correct_top1 = 0
        self.total_correct_top5 = 0

    def update(self, loss: float, logits: torch.Tensor, labels: torch.Tensor) -> None:
        batch_size = labels.size(0)
        self.total_loss += loss * batch_size
        self.total_samples += batch_size
        acc1, acc5 = accuracy_topk(logits, labels, topk=(1, 5))
        self.total_correct_top1 += acc1 * batch_size / 100.0
        self.total_correct_top5 += acc5 * batch_size / 100.0

    @property
    def avg_loss(self) -> float:
        return self.total_loss / max(1, self.total_samples)

    @property
    def avg_acc1(self) -> float:
        return self.total_correct_top1 / max(1, self.total_samples) * 100.0

    @property
    def avg_acc5(self) -> float:
        return self.total_correct_top5 / max(1, self.total_samples) * 100.0

    def summary(self) -> dict[str, float]:
        return {
            "loss": self.avg_loss,
            "acc1": self.avg_acc1,
            "acc5": self.avg_acc5,
        }
