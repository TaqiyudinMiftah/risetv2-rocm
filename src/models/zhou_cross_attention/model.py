from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.common import _make_encoder

# ---------------------------------------------------------------------------
# Blocks for Zhou et al.
# ---------------------------------------------------------------------------


class CrossAttentionBlock(nn.Module):
    """
    Cross-Attention (CA) block.
    Uses channel statistics from one stream to gate the other stream,
    capturing cross-stream complementarity via cross-channel attention.
    """

    def __init__(self, channels: int, reduction: int = 16) -> None:
        super().__init__()
        self.face_gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(channels, channels // reduction),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels),
            nn.Sigmoid(),
        )
        self.context_gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(channels, channels // reduction),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels),
            nn.Sigmoid(),
        )

    def forward(
        self, face_feat: torch.Tensor, context_feat: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # face_feat gates context, context gates face
        face_att = self.face_gate(face_feat).view(face_feat.size(0), face_feat.size(1), 1, 1)
        context_att = self.context_gate(context_feat).view(context_feat.size(0), context_feat.size(1), 1, 1)
        face_enhanced = face_feat * context_att
        context_enhanced = context_feat * face_att
        return face_enhanced, context_enhanced


class ElementRecalibrationBlock(nn.Module):
    """
    Element Recalibration (ER) block.
    Revises the feature map of each channel by embedding global information
    (SE-like channel attention).
    """

    def __init__(self, channels: int, reduction: int = 16) -> None:
        super().__init__()
        self.fc = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(channels, channels // reduction),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w = self.fc(x).view(x.size(0), x.size(1), 1, 1)
        return x * w


class AdaptiveAttentionBlock(nn.Module):
    """
    Adaptive-Attention (AA) block.
    Infers optimal feature fusion weights via hybrid feature weighting:
    - Global stream weights (face vs context)
    - Channel-wise recalibration weights
    """

    def __init__(self, channels: int, hidden_dim: int = 256) -> None:
        super().__init__()
        self.channels = channels
        self.global_fusion = nn.Sequential(
            nn.Linear(channels * 2, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 2),
            nn.Softmax(dim=1),
        )
        self.channel_weight = nn.Sequential(
            nn.Linear(channels * 2, channels),
            nn.Sigmoid(),
        )

    def forward(
        self, face_feat: torch.Tensor, context_feat: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        face_vec = F.adaptive_avg_pool2d(face_feat, 1).flatten(1)
        context_vec = F.adaptive_avg_pool2d(context_feat, 1).flatten(1)
        combined = torch.cat([face_vec, context_vec], dim=1)

        w_global = self.global_fusion(combined)  # [B, 2]
        w_channel = self.channel_weight(combined).view(-1, self.channels, 1, 1)  # [B, C, 1, 1]

        w_face = w_global[:, 0].view(-1, 1, 1, 1)
        w_context = w_global[:, 1].view(-1, 1, 1, 1)
        fused = w_face * face_feat + w_context * context_feat
        fused = fused * w_channel
        return fused, w_global


class DeepFusionBlock(nn.Module):
    """
    Deep Fusion (DF) block.
    Integrates adaptive emotion features for final prediction.
    """

    def __init__(
        self, channels: int, num_classes: int, dropout: float = 0.5, hidden_dim: int = 512
    ) -> None:
        super().__init__()
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(channels, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(x)


# ---------------------------------------------------------------------------
# Zhou et al. model
# ---------------------------------------------------------------------------


class ZhouCrossAttentionNet(nn.Module):
    """
    Cross-Attention and Hybrid Feature Weighting Network
    (Zhou et al., IJERPH 2023).

    Architecture:
        DBE -> CA -> ER -> AA -> DF
    """

    def __init__(
        self,
        num_classes: int,
        backbone: str = "resnet18",
        pretrained: bool = False,
        dropout: float = 0.5,
        ca_reduction: int = 16,
        er_reduction: int = 16,
        aa_hidden_dim: int = 256,
        df_hidden_dim: int = 512,
    ) -> None:
        super().__init__()
        self.face_encoder, self.face_dim = _make_encoder(backbone, pretrained=pretrained)
        self.context_encoder, self.context_dim = _make_encoder(backbone, pretrained=pretrained)

        if self.face_dim != self.context_dim:
            raise ValueError(
                f"face_dim ({self.face_dim}) must equal context_dim ({self.context_dim}) "
                f"for ZhouCrossAttentionNet"
            )
        channels = self.face_dim

        self.ca = CrossAttentionBlock(channels, reduction=ca_reduction)
        self.er_face = ElementRecalibrationBlock(channels, reduction=er_reduction)
        self.er_context = ElementRecalibrationBlock(channels, reduction=er_reduction)
        self.aa = AdaptiveAttentionBlock(channels, hidden_dim=aa_hidden_dim)
        self.df = DeepFusionBlock(channels, num_classes, dropout=dropout, hidden_dim=df_hidden_dim)

    def forward(
        self,
        face_image: torch.Tensor,
        context_image: torch.Tensor,
    ) -> dict[str, Any]:
        # DBE: dual-branch encoding (spatial feature maps)
        face_feat = self.face_encoder(face_image)["feat"]  # [B, C, H, W]
        context_feat = self.context_encoder(context_image)["feat"]  # [B, C, H, W]

        # CA: cross-attention
        face_ca, context_ca = self.ca(face_feat, context_feat)

        # ER: element recalibration
        face_er = self.er_face(face_ca)
        context_er = self.er_context(context_ca)

        # AA: adaptive attention (hybrid fusion)
        fused, fusion_weights = self.aa(face_er, context_er)

        # DF: deep fusion & classification
        logits = self.df(fused)

        return {
            "logits": logits,
            "fusion_weights": fusion_weights,
            "face_feat": face_feat,
            "context_feat": context_feat,
            "fused_feat": fused,
        }
