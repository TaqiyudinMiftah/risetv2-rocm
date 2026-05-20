from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Paper 5-layer custom CNN (Section 3.2.1)
# ---------------------------------------------------------------------------

class _CAERConvBlock(nn.Module):
    """Conv2D(3x3) + BN + ReLU [+ MaxPool(2x2)]."""

    def __init__(self, in_c: int, out_c: int, pool: bool = False) -> None:
        super().__init__()
        self.conv = nn.Conv2d(in_c, out_c, 3, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(out_c)
        self.pool = nn.MaxPool2d(2, 2) if pool else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.pool(F.relu(self.bn(self.conv(x))))


class _CAERCNN(nn.Module):
    """
    5-layer CNN: 32 -> 64 -> 128 -> 256 -> 256.
    MaxPool on first 4 layers only (no pool on conv5).
    Output: [B, 256, H/16, W/16] for input HxW.
    """

    def __init__(self, in_c: int = 3) -> None:
        super().__init__()
        self.conv1 = _CAERConvBlock(in_c, 32, pool=True)
        self.conv2 = _CAERConvBlock(32, 64, pool=True)
        self.conv3 = _CAERConvBlock(64, 128, pool=True)
        self.conv4 = _CAERConvBlock(128, 256, pool=True)
        self.conv5 = _CAERConvBlock(256, 256, pool=False)  # no pool

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)  # [B,  32, H/2, W/2]
        x = self.conv2(x)  # [B,  64, H/4, W/4]
        x = self.conv3(x)  # [B, 128, H/8, W/8]
        x = self.conv4(x)  # [B, 256, H/16, W/16]
        x = self.conv5(x)  # [B, 256, H/16, W/16]
        return x


class _EncoderWrapper(nn.Module):
    """Wraps encoder to always return a tensor, handling dict outputs from feature extractors."""

    def __init__(self, encoder: nn.Module) -> None:
        super().__init__()
        self.encoder = encoder

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.encoder(x)
        if isinstance(out, dict):
            return out["feat"]
        return out


# ---------------------------------------------------------------------------
# Context Attention Inference Module (Eq. 3-4)
# ---------------------------------------------------------------------------

class ContextAttention(nn.Module):
    """
    Unsupervised spatial attention on the context feature map.

    Forward:
        X_C  -> Conv(C->128, 3x3) + BN + ReLU
              -> Conv(128->1,  3x3)
              -> spatial softmax        (Eq. 3)
              -> element-wise * X_C    (Eq. 4)  -> X̄_C
    """

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(channels, 128, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(128)
        self.conv2 = nn.Conv2d(128, 1, 3, padding=1, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        a = F.relu(self.bn1(self.conv1(x)))  # [B, 128, H, W]
        a = self.conv2(a)                     # [B,   1, H, W]

        # Eq. 3: spatial softmax -> A_i = exp(A_i) / sum_j exp(A_j)
        B, _, H, W = a.shape
        a = F.softmax(a.view(B, -1), dim=1).view(B, 1, H, W)

        # Eq. 4: X̄_C = A * X_C
        return x * a  # [B, C, H, W]


# ---------------------------------------------------------------------------
# Adaptive Fusion Network (Section 3.2.2, Eq. 5-6)
# ---------------------------------------------------------------------------

class AdaptiveFusion(nn.Module):
    """
    Infer per-stream attention weights (lambda_F, lambda_C) via four 1x1 Conv layers,
    then classify the weighted-concatenated feature via two 1x1 Conv layers.

    Total 6 x Conv2d(1x1) as stated in the paper.
    """

    def __init__(self, channels: int, num_classes: int, dropout: float = 0.5) -> None:
        super().__init__()

        # -- lambda_F inference: 2 layers (C->128->1) per stream --
        self.face_w1 = nn.Conv2d(channels, 128, 1, bias=True)
        self.face_w2 = nn.Conv2d(128, 1, 1, bias=True)

        # -- lambda_C inference: 2 layers (C->128->1) per stream --
        self.ctx_w1 = nn.Conv2d(channels, 128, 1, bias=True)
        self.ctx_w2 = nn.Conv2d(128, 1, 1, bias=True)

        # -- Classifier: 2 layers (2C->128->K) --
        self.classifier = nn.Sequential(
            nn.Conv2d(channels * 2, 128, 1, bias=True),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=dropout),
            nn.Conv2d(128, num_classes, 1, bias=True),
        )

    def forward(
        self,
        face_feat: torch.Tensor,  # [B, C, H, W]
        ctx_feat: torch.Tensor,   # [B, C, H, W]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            logits: [B, K]
            fusion_weights: [B, 2] (lambda_F, lambda_C) -- softmax-normalised
        """
        # Compute raw scalar attention for each stream via GAP
        lam_f = self.face_w2(F.relu(self.face_w1(face_feat)))  # [B, 1, H, W]
        lam_c = self.ctx_w2(F.relu(self.ctx_w1(ctx_feat)))     # [B, 1, H, W]

        lam_f = F.adaptive_avg_pool2d(lam_f, 1).flatten(1)     # [B, 1]
        lam_c = F.adaptive_avg_pool2d(lam_c, 1).flatten(1)     # [B, 1]

        # lambda_F + lambda_C = 1 (softmax)
        weights = F.softmax(torch.cat([lam_f, lam_c], dim=1), dim=1)  # [B, 2]
        w_f = weights[:, 0:1, None, None]   # [B, 1, 1, 1]
        w_c = weights[:, 1:2, None, None]

        # Eq. 5: X_A = concat(X_F * lambda_F, X̄_C * lambda_C)
        x_a = torch.cat([face_feat * w_f, ctx_feat * w_c], dim=1)  # [B, 2C, H, W]

        # Eq. 6: y = F(X_A; W_G) via 1x1 conv -> GAP
        logits = self.classifier(x_a)                        # [B, K, H, W]
        logits = F.adaptive_avg_pool2d(logits, 1).flatten(1)  # [B, K]

        return logits, weights


# ---------------------------------------------------------------------------
# CAER-Net / CAER-Net-S
# ---------------------------------------------------------------------------

class CAERNet(nn.Module):
    """
    Full CAER-Net-S (static / 2-D) model.

    Supports both the paper's custom 5-layer CNN and pretrained backbones
    (e.g. ResNet-18) for reproduction fidelity.
    Allows different backbones for face and context branches (e.g. ImageNet
    for face, Places365 for context).
    """

    def __init__(
        self,
        num_classes: int = 7,
        dropout: float = 0.5,
        backbone: str = "custom",
        pretrained: bool = False,
        face_backbone: str | None = None,
        context_backbone: str | None = None,
    ) -> None:
        super().__init__()
        self.backbone = backbone
        # Per-branch overrides take precedence over shared backbone
        face_bb = face_backbone if face_backbone is not None else backbone
        ctx_bb = context_backbone if context_backbone is not None else backbone

        if face_bb == "custom":
            face_enc = _CAERCNN(in_c=3)
            face_channels = 256
        else:
            from models.common import _make_encoder
            face_enc, face_channels = _make_encoder(face_bb, pretrained)

        if ctx_bb == "custom":
            ctx_enc = _CAERCNN(in_c=3)
            ctx_channels = 256
        else:
            from models.common import _make_encoder
            ctx_enc, ctx_channels = _make_encoder(ctx_bb, pretrained)

        assert face_channels == ctx_channels, (
            f"Face encoder channels ({face_channels}) must match "
            f"context encoder channels ({ctx_channels}) for fusion."
        )
        channels = face_channels

        self.face_encoder = _EncoderWrapper(face_enc)
        self.ctx_encoder = _EncoderWrapper(ctx_enc)
        self.ctx_attention = ContextAttention(channels)
        self.fusion = AdaptiveFusion(channels, num_classes, dropout=dropout)

    def forward(
        self,
        face_image: torch.Tensor,    # [B, 3, H, W]  face-cropped
        context_image: torch.Tensor, # [B, 3, H, W]  face-hidden scene
    ) -> dict[str, Any]:

        face_feat = self.face_encoder(face_image)
        ctx_feat = self.ctx_encoder(context_image)

        ctx_feat = self.ctx_attention(ctx_feat)       # apply Eq. 3-4

        logits, fusion_weights = self.fusion(face_feat, ctx_feat)

        return {
            "logits": logits,
            "fusion_weights": fusion_weights,
            "face_feat": F.adaptive_avg_pool2d(face_feat, 1).flatten(1),
            "context_feat": F.adaptive_avg_pool2d(ctx_feat, 1).flatten(1),
        }


class SingleStreamNet(nn.Module):
    """Face-only or context-only baseline for ablation studies."""

    def __init__(
        self,
        num_classes: int = 7,
        stream: str = "face",
        dropout: float = 0.5,
        backbone: str = "custom",
        pretrained: bool = False,
    ) -> None:
        super().__init__()
        if stream not in ("face", "context"):
            raise ValueError("stream must be 'face' or 'context'")
        self.stream = stream

        if backbone == "custom":
            encoder = _CAERCNN(in_c=3)
            channels = 256
        else:
            from models.common import _make_encoder
            encoder, channels = _make_encoder(backbone, pretrained)

        self.encoder = _EncoderWrapper(encoder)
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Dropout(p=dropout),
            nn.Linear(channels, num_classes),
        )

    def forward(self, face_image: torch.Tensor, context_image: torch.Tensor) -> dict[str, Any]:
        x = face_image if self.stream == "face" else context_image
        feat = self.encoder(x)
        logits = self.classifier(feat)
        return {
            "logits": logits,
            "feat": F.adaptive_avg_pool2d(feat, 1).flatten(1),
        }
