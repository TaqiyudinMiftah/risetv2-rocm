from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.agcd_net.ag_cim import AGCIM
from models.agcd_net.hybrid_convnext import HybridConvNeXtEncoder


class AGCDNet(nn.Module):
    """
    AGCD-Net: Attention Guided Context Debiasing Network
    Reproduction of Devi et al. (ICIAP 2025).

    Architecture:
        1. Hybrid ConvNeXt encoder (face + context streams)
        2. MHSA on both streams
        3. AG-CIM: face-guided context debiasing
        4. Element-wise addition fusion
        5. FC classifier
    """

    def __init__(
        self,
        num_classes: int = 7,
        convnext_variant: str = "tiny",
        pretrained: bool = True,
        use_stn: bool = True,
        se_reduction: int = 16,
        num_heads: int = 8,
        dropout: float = 0.5,
    ) -> None:
        super().__init__()
        self.num_classes = num_classes

        # Dual Hybrid ConvNeXt encoders
        self.face_encoder = HybridConvNeXtEncoder(
            variant=convnext_variant,
            pretrained=pretrained,
            use_stn=use_stn,
            se_reduction=se_reduction,
        )
        self.context_encoder = HybridConvNeXtEncoder(
            variant=convnext_variant,
            pretrained=pretrained,
            use_stn=use_stn,
            se_reduction=se_reduction,
        )

        self.feature_dim = self.face_encoder.feature_dim

        # Spatial feature pooling and projection for MHSA
        # ConvNeXt output: [B, C, H, W] -> we flatten spatial dims
        # After ConvNeXt stages: H'=7, W'=7 for 224x224 input
        self.spatial_tokens = 7 * 7  # 49 tokens

        # Project to a common dimension for MHSA (paper uses same dim)
        self.face_proj = nn.Linear(self.feature_dim, self.feature_dim)
        self.context_proj = nn.Linear(self.feature_dim, self.feature_dim)

        # AG-CIM module
        self.ag_cim = AGCIM(
            dim=self.feature_dim,
            num_heads=num_heads,
            dropout=dropout,
        )

        # Fusion and classification
        # Paper uses element-wise addition then FC
        self.fusion_norm = nn.LayerNorm(self.feature_dim)
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(self.feature_dim, num_classes),
        )

    def extract_tokens(self, x: torch.Tensor, encoder: nn.Module) -> torch.Tensor:
        """
        Extract spatial tokens from encoder output.
        Args:
            x: [B, 3, H, W] image
        Returns:
            [B, N, C] tokens where N = H'*W'
        """
        feat = encoder(x)["feat"]  # [B, C, H', W']
        B, C, H, W = feat.shape
        tokens = feat.permute(0, 2, 3, 1).reshape(B, H * W, C)  # [B, N, C]
        return tokens

    def forward(
        self,
        face_image: torch.Tensor,
        context_image: torch.Tensor,
    ) -> dict[str, Any]:
        """
        Args:
            face_image: [B, 3, H, W]
            context_image: [B, 3, H, W]
        Returns:
            dict with "logits", "face_att", "context_corr", "h_face", "h_context"
        """
        # Extract features
        face_tokens = self.extract_tokens(face_image, self.face_encoder)    # [B, N, C]
        context_tokens = self.extract_tokens(context_image, self.context_encoder)  # [B, N, C]

        # Project
        face_tokens = self.face_proj(face_tokens)
        context_tokens = self.context_proj(context_tokens)

        # AG-CIM: face-guided context debiasing
        face_att, context_corr, h_face, h_context = self.ag_cim(
            face_tokens, context_tokens
        )

        # Fusion: element-wise addition (per paper Eq. 7)
        fused = face_att + context_corr  # [B, N, C]
        fused = self.fusion_norm(fused)

        # Global average pooling over tokens
        fused_pooled = fused.mean(dim=1)  # [B, C]

        # Classification
        logits = self.classifier(fused_pooled)  # [B, num_classes]

        return {
            "logits": logits,
            "face_att": face_att,
            "context_corr": context_corr,
            "h_face": h_face,
            "h_context": h_context,
        }
