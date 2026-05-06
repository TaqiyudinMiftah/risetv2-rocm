from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.common import _make_encoder


class AdaptiveFusion(nn.Module):
    """
    Lightweight adaptive fusion module.
    Given face and context features, predicts a soft weighting
    (via softmax) and returns the fused representation.
    """

    def __init__(self, face_dim: int, context_dim: int, hidden_dim: int = 256) -> None:
        super().__init__()
        total_dim = face_dim + context_dim
        self.fc1 = nn.Linear(total_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 2)  # two weights: face, context

    def forward(self, face_feat: torch.Tensor, context_feat: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        combined = torch.cat([face_feat, context_feat], dim=1)
        h = F.relu(self.fc1(combined))
        logits = self.fc2(h)
        weights = F.softmax(logits, dim=1)  # [B, 2]
        w_face = weights[:, 0:1]
        w_context = weights[:, 1:2]
        fused = w_face * face_feat + w_context * context_feat
        return fused, weights


class CAERNet(nn.Module):
    """
    Minimal CAER-Net with two-stream encoders and adaptive fusion.
    """

    def __init__(
        self,
        num_classes: int,
        backbone: str = "resnet18",
        pretrained: bool = False,
        dropout: float = 0.5,
    ) -> None:
        super().__init__()
        self.face_encoder, self.face_dim = _make_encoder(backbone, pretrained=pretrained)
        self.context_encoder, self.context_dim = _make_encoder(backbone, pretrained=pretrained)

        self.fusion = AdaptiveFusion(self.face_dim, self.context_dim)

        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(self.face_dim, num_classes),
        )

    def forward(
        self,
        face_image: torch.Tensor,
        context_image: torch.Tensor,
    ) -> dict[str, Any]:
        face_feat = self.face_encoder(face_image)["feat"]  # [B, C, H, W]
        context_feat = self.context_encoder(context_image)["feat"]  # [B, C, H, W]

        if face_feat.dim() > 2:
            face_feat = F.adaptive_avg_pool2d(face_feat, (1, 1)).flatten(1)
        if context_feat.dim() > 2:
            context_feat = F.adaptive_avg_pool2d(context_feat, (1, 1)).flatten(1)

        fused_feat, fusion_weights = self.fusion(face_feat, context_feat)
        logits = self.classifier(fused_feat)

        return {
            "logits": logits,
            "fusion_weights": fusion_weights,
            "face_feat": face_feat,
            "context_feat": context_feat,
            "fused_feat": fused_feat,
        }


class SingleStreamNet(nn.Module):
    """
    Single-stream baseline (face only or context only) for ablation.
    """

    def __init__(
        self,
        num_classes: int,
        stream: str = "face",
        backbone: str = "resnet18",
        pretrained: bool = False,
        dropout: float = 0.5,
    ) -> None:
        super().__init__()
        if stream not in ("face", "context"):
            raise ValueError("stream must be 'face' or 'context'")
        self.stream = stream
        self.encoder, self.feat_dim = _make_encoder(backbone, pretrained=pretrained)
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(self.feat_dim, num_classes),
        )

    def forward(self, face_image: torch.Tensor, context_image: torch.Tensor) -> dict[str, Any]:
        x = face_image if self.stream == "face" else context_image
        feat = self.encoder(x)["feat"]
        if feat.dim() > 2:
            feat = F.adaptive_avg_pool2d(feat, (1, 1)).flatten(1)
        logits = self.classifier(feat)
        return {
            "logits": logits,
            "feat": feat,
        }
