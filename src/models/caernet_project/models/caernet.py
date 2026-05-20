"""
models/caernet.py
CAER-Net: Context-Aware Emotion Recognition Networks.
Lee et al., ICCV 2019.

Architecture
------------
Two-stream encoding networks
  ├─ Face stream   : CNN encoder → feature map X_F
  └─ Context stream: CNN encoder → Context Attention → X̄_C
Adaptive Fusion Networks
  └─ Infer λ_F, λ_C → weighted concat → 1×1 Conv classifier
"""

from __future__ import annotations

from typing import Any, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.common import _make_encoder


# ---------------------------------------------------------------------------
# Paper 5-layer custom CNN (Section 3.2.1)
# ---------------------------------------------------------------------------

class _CAERConvBlock(nn.Module):
    """Conv2D(3×3) + BN + ReLU [+ MaxPool(2×2)]."""

    def __init__(self, in_c: int, out_c: int, pool: bool = False) -> None:
        super().__init__()
        self.conv = nn.Conv2d(in_c, out_c, 3, padding=1, bias=False)
        self.bn   = nn.BatchNorm2d(out_c)
        self.pool = nn.MaxPool2d(2, 2) if pool else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.pool(F.relu(self.bn(self.conv(x))))


class _CAERCNN(nn.Module):
    """
    5-layer CNN: 32 → 64 → 128 → 256 → 256.
    MaxPool on first 4 layers only (no pool on conv5).
    Output ∈ R^{256×14×14} for 224×224 input.
    """

    def __init__(self, in_c: int = 3) -> None:
        super().__init__()
        self.conv1 = _CAERConvBlock(in_c,  32,  pool=True)
        self.conv2 = _CAERConvBlock(32,    64,  pool=True)
        self.conv3 = _CAERConvBlock(64,    128, pool=True)
        self.conv4 = _CAERConvBlock(128,   256, pool=True)
        self.conv5 = _CAERConvBlock(256,   256, pool=False)   # no pool

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)   # [B,  32, 112, 112]
        x = self.conv2(x)   # [B,  64,  56,  56]
        x = self.conv3(x)   # [B, 128,  28,  28]
        x = self.conv4(x)   # [B, 256,  14,  14]
        x = self.conv5(x)   # [B, 256,  14,  14]
        return x


# ---------------------------------------------------------------------------
# Context Attention Inference Module (Eq. 3-4)
# ---------------------------------------------------------------------------

class ContextAttention(nn.Module):
    """
    Unsupervised spatial attention on the context feature map.

    Forward:
        X_C  ─► Conv(C→128, 3×3) + BN + ReLU
              ─► Conv(128→1,  3×3)
              ─► spatial softmax        (Eq. 3)
              ─► element-wise ⊙ X_C    (Eq. 4)  → X̄_C
    """

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(channels, 128, 3, padding=1, bias=False)
        self.bn1   = nn.BatchNorm2d(128)
        self.conv2 = nn.Conv2d(128, 1, 3, padding=1, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        a = F.relu(self.bn1(self.conv1(x)))  # [B, 128, H, W]
        a = self.conv2(a)                     # [B,   1, H, W]

        # Eq. 3: spatial softmax → Â_i = exp(A_i) / Σ_j exp(A_j)
        B, _, H, W = a.shape
        a = F.softmax(a.view(B, -1), dim=1).view(B, 1, H, W)

        # Eq. 4: X̄_C = Â ⊙ X_C
        return x * a                          # [B, C, H, W]


# ---------------------------------------------------------------------------
# Adaptive Fusion Network (Section 3.2.2, Eq. 5-6)
# ---------------------------------------------------------------------------

class AdaptiveFusion(nn.Module):
    """
    Infer per-stream attention weights (λ_F, λ_C) via four 1×1 Conv layers,
    then classify the weighted-concatenated feature via two 1×1 Conv layers.

    Total 6 × Conv2d(1×1) as stated in the paper.
    """

    def __init__(self, channels: int, num_classes: int, dropout: float = 0.5) -> None:
        super().__init__()

        # ── λ_F inference: 2 layers (C→128→1) per stream ──
        self.face_w1 = nn.Conv2d(channels, 128, 1, bias=True)
        self.face_w2 = nn.Conv2d(128,        1, 1, bias=True)

        # ── λ_C inference: 2 layers (C→128→1) per stream ──
        self.ctx_w1  = nn.Conv2d(channels, 128, 1, bias=True)
        self.ctx_w2  = nn.Conv2d(128,        1, 1, bias=True)

        # ── Classifier: 2 layers (2C→128→K) ──
        self.classifier = nn.Sequential(
            nn.Conv2d(channels * 2, 128, 1, bias=True),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=dropout),
            nn.Conv2d(128, num_classes, 1, bias=True),
        )

    def forward(
        self,
        face_feat: torch.Tensor,     # [B, C, H, W]
        ctx_feat:  torch.Tensor,     # [B, C, H, W]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Returns
        -------
        logits          : [B, K]
        fusion_weights  : [B, 2]  (λ_F, λ_C) – softmax-normalised
        """
        # Compute raw scalar attention for each stream via GAP
        lam_f = self.face_w2(F.relu(self.face_w1(face_feat)))  # [B, 1, H, W]
        lam_c = self.ctx_w2 (F.relu(self.ctx_w1 (ctx_feat)))   # [B, 1, H, W]

        lam_f = F.adaptive_avg_pool2d(lam_f, 1).flatten(1)     # [B, 1]
        lam_c = F.adaptive_avg_pool2d(lam_c, 1).flatten(1)     # [B, 1]

        # λ_F + λ_C = 1  (softmax)
        weights = F.softmax(torch.cat([lam_f, lam_c], dim=1), dim=1)  # [B, 2]
        w_f = weights[:, 0:1, None, None]   # [B, 1, 1, 1]
        w_c = weights[:, 1:2, None, None]

        # Eq. 5: X_A = Π(X_F ⊙ λ_F,  X̄_C ⊙ λ_C)
        x_a = torch.cat([face_feat * w_f, ctx_feat * w_c], dim=1)  # [B, 2C, H, W]

        # Eq. 6: y = F(X_A; W_G) via 1×1 conv → GAP
        logits = self.classifier(x_a)                           # [B, K, H, W]
        logits = F.adaptive_avg_pool2d(logits, 1).flatten(1)    # [B, K]

        return logits, weights


# ---------------------------------------------------------------------------
# CAER-Net  /  CAER-Net-S
# ---------------------------------------------------------------------------

class CAERNet(nn.Module):
    """
    Full CAER-Net (dynamic) and CAER-Net-S (static / 2-D) model.

    When backbone == 'custom'  → uses the paper's own 5-layer CNN.
    Otherwise                  → uses the specified pretrained backbone.

    Parameters
    ----------
    num_classes : int    – number of emotion categories (7 for CAER dataset)
    backbone    : str    – 'custom' | 'resnet18' | 'resnet50' | 'vgg16' | …
    pretrained  : bool   – load ImageNet weights (ignored for 'custom')
    dropout     : float  – dropout rate in the fusion classifier
    """

    def __init__(
        self,
        num_classes: int,
        backbone:    str   = "resnet18",
        pretrained:  bool  = True,
        dropout:     float = 0.5,
    ) -> None:
        super().__init__()
        self._custom = backbone == "custom"

        if self._custom:
            self.face_encoder    = _CAERCNN(in_c=3)
            self.ctx_encoder_cnn = _CAERCNN(in_c=3)
            channels             = 256
        else:
            self.face_encoder,    self.face_dim    = _make_encoder(backbone, pretrained)
            self.ctx_encoder_cnn, self.context_dim = _make_encoder(backbone, pretrained)
            channels = self.face_dim   # both streams share same backbone ⟹ same dim

        self.ctx_attention = ContextAttention(channels)
        self.fusion        = AdaptiveFusion(channels, num_classes, dropout=dropout)

    def forward(
        self,
        face_image:    torch.Tensor,   # [B, 3, H, W]  face-cropped
        context_image: torch.Tensor,   # [B, 3, H, W]  face-hidden scene
    ) -> Dict[str, Any]:

        if self._custom:
            face_feat = self.face_encoder(face_image)          # [B, 256, 14, 14]
            ctx_feat  = self.ctx_encoder_cnn(context_image)
        else:
            face_feat = self.face_encoder(face_image)["feat"]     # [B, C, H, W]
            ctx_feat  = self.ctx_encoder_cnn(context_image)["feat"]

        ctx_feat = self.ctx_attention(ctx_feat)               # apply Eq. 3-4

        logits, fusion_weights = self.fusion(face_feat, ctx_feat)

        return {
            "logits":          logits,                                          # [B, K]
            "fusion_weights":  fusion_weights,                                  # [B, 2]
            "face_feat":       F.adaptive_avg_pool2d(face_feat, 1).flatten(1), # [B, C]
            "context_feat":    F.adaptive_avg_pool2d(ctx_feat,  1).flatten(1), # [B, C]
        }


# ---------------------------------------------------------------------------
# Single-stream baseline (ablation)
# ---------------------------------------------------------------------------

class SingleStreamNet(nn.Module):
    """Face-only or context-only baseline for ablation studies."""

    def __init__(
        self,
        num_classes: int,
        stream:      str   = "face",          # 'face' | 'context'
        backbone:    str   = "resnet18",
        pretrained:  bool  = True,
        dropout:     float = 0.5,
    ) -> None:
        super().__init__()
        if stream not in ("face", "context"):
            raise ValueError("stream must be 'face' or 'context'")
        self.stream  = stream
        self._custom = backbone == "custom"

        if self._custom:
            self.encoder = _CAERCNN(in_c=3)
            channels     = 256
        else:
            self.encoder, channels = _make_encoder(backbone, pretrained)

        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Dropout(p=dropout),
            nn.Linear(channels, num_classes),
        )

    def forward(
        self,
        face_image:    torch.Tensor,
        context_image: torch.Tensor,
    ) -> Dict[str, Any]:
        x    = face_image if self.stream == "face" else context_image
        feat = self.encoder(x)
        if isinstance(feat, dict):
            feat = feat["feat"]
        logits = self.classifier(feat)
        return {
            "logits": logits,
            "feat":   F.adaptive_avg_pool2d(feat, 1).flatten(1),
        }
