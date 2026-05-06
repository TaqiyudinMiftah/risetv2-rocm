from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.common import _make_encoder


# ---------------------------------------------------------------------------
# GLA (Global-Local Attention) Module
# ---------------------------------------------------------------------------

class GLAModule(nn.Module):
    """
    Global-Local Attention module.

    Given face feature vector and context spatial feature map,
    learns spatial attention over the context conditioned on the face.
    """

    def __init__(self, face_dim: int, context_dim: int, hidden_dim: int = 128) -> None:
        super().__init__()
        self.face_dim = face_dim
        self.context_dim = context_dim

        self.attention_fc1 = nn.Linear(face_dim + context_dim, hidden_dim)
        self.attention_bn = nn.BatchNorm1d(hidden_dim)
        self.attention_fc2 = nn.Linear(hidden_dim, 1)

    def forward(self, face_vector: torch.Tensor, context_feat: torch.Tensor) -> torch.Tensor:
        """
        Args:
            face_vector: [B, face_dim]
            context_feat: [B, context_dim, H, W]
        Returns:
            context_vector: [B, context_dim] — attended context representation
        """
        B, C, H, W = context_feat.shape

        # Flatten context spatial dimensions
        context_flat = context_feat.view(B, C, H * W).permute(0, 2, 1)  # [B, H*W, C]

        # Repeat face vector for each spatial location
        face_repeat = face_vector.unsqueeze(1).expand(-1, H * W, -1)  # [B, H*W, face_dim]

        # Concatenate face and context per location
        concat = torch.cat([face_repeat, context_flat], dim=-1)  # [B, H*W, face_dim + context_dim]

        # Attention weights
        # Reshape for BatchNorm1d: [B, H*W, hidden_dim] -> [B*H*W, hidden_dim]
        attn = self.attention_fc1(concat)  # [B, H*W, hidden_dim]
        attn = attn.view(B * H * W, -1)
        attn = self.attention_bn(attn)
        attn = F.relu(attn)
        attn = attn.view(B, H * W, -1)
        attn = self.attention_fc2(attn).squeeze(-1)  # [B, H*W]
        attn = F.softmax(attn, dim=1)  # [B, H*W]

        # Weighted sum of context features
        attn = attn.unsqueeze(2)  # [B, H*W, 1]
        context_vector = (context_flat * attn).sum(dim=1)  # [B, context_dim]

        return context_vector


# ---------------------------------------------------------------------------
# Fusion Module
# ---------------------------------------------------------------------------

class FusionModule(nn.Module):
    """
    Fusion module that learns adaptive weights for face and context streams.
    """

    def __init__(self, face_dim: int, context_dim: int, hidden_dim: int = 128) -> None:
        super().__init__()
        self.face_weight_fc = nn.Sequential(
            nn.Linear(face_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 1),
        )
        self.context_weight_fc = nn.Sequential(
            nn.Linear(context_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 1),
        )

    def forward(
        self, face_vector: torch.Tensor, context_vector: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        w_f = self.face_weight_fc(face_vector)  # [B, 1]
        w_c = self.context_weight_fc(context_vector)  # [B, 1]
        w_fc = torch.cat([w_f, w_c], dim=1)  # [B, 2]
        w_fc = F.softmax(w_fc, dim=1)

        face_weighted = face_vector * w_fc[:, 0:1]
        context_weighted = context_vector * w_fc[:, 1:2]

        return face_weighted, context_weighted


# ---------------------------------------------------------------------------
# GLAMOR-Net
# ---------------------------------------------------------------------------

class GLAMORNet(nn.Module):
    """
    Global-Local Attention for Emotion Recognition (GLAMOR-Net).
    Le et al., Neural Computing and Applications, 2022.

    Two-stream encoder + Global-Local Attention (GLA) module + Fusion + Classifier.
    """

    def __init__(
        self,
        num_classes: int = 7,
        backbone: str = "resnet18",
        pretrained: bool = False,
        dropout: float = 0.5,
        gla_hidden_dim: int = 128,
        fusion_hidden_dim: int = 128,
        classifier_hidden_dim: int = 128,
    ) -> None:
        super().__init__()
        self.face_encoder, self.face_dim = _make_encoder(backbone, pretrained=pretrained)
        self.context_encoder, self.context_dim = _make_encoder(backbone, pretrained=pretrained)

        # GLA module: attend context conditioned on face
        self.gla = GLAModule(
            face_dim=self.face_dim,
            context_dim=self.context_dim,
            hidden_dim=gla_hidden_dim,
        )

        # Fusion module: adaptive weighting
        self.fusion = FusionModule(
            face_dim=self.face_dim,
            context_dim=self.context_dim,
            hidden_dim=fusion_hidden_dim,
        )

        # Final classifier
        self.classifier = nn.Sequential(
            nn.Linear(self.face_dim + self.context_dim, classifier_hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(classifier_hidden_dim, num_classes),
        )

    def forward(
        self,
        face_image: torch.Tensor,
        context_image: torch.Tensor,
    ) -> dict[str, Any]:
        # Encode face and context (spatial feature maps)
        face_feat = self.face_encoder(face_image)["feat"]  # [B, face_dim, H, W]
        context_feat = self.context_encoder(context_image)["feat"]  # [B, context_dim, H, W]

        # Global average pooling for face
        face_vector = F.adaptive_avg_pool2d(face_feat, 1).flatten(1)  # [B, face_dim]

        # GLA: attend context conditioned on face
        context_vector = self.gla(face_vector, context_feat)  # [B, context_dim]

        # Fusion: adaptive weighting
        face_weighted, context_weighted = self.fusion(face_vector, context_vector)

        # Concatenate and classify
        fused = torch.cat([face_weighted, context_weighted], dim=1)  # [B, face_dim + context_dim]
        logits = self.classifier(fused)

        return {
            "logits": logits,
            "face_vector": face_vector,
            "context_vector": context_vector,
            "fused": fused,
        }
