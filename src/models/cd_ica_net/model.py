from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.common import _make_encoder

from .ccim_module import IntegratedCCIM
from .fusion_module import HybridAdaptiveFusion
from .ica_module import ICAModule


# ---------------------------------------------------------------------------
# Context Attention Highlight Module
# ---------------------------------------------------------------------------

class ContextHighlightModule(nn.Module):
    """
    Attention-based highlight module for context branch.
    Generates a spatial attention map to emphasize emotionally relevant regions.
    """

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels // 2, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels // 2)
        self.conv2 = nn.Conv2d(channels // 2, 1, kernel_size=3, padding=1, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, C, H, W]
        Returns:
            highlighted: [B, C, H, W]
        """
        attn = F.relu(self.bn1(self.conv1(x)), inplace=True)
        attn = self.conv2(attn)  # [B, 1, H, W]
        attn = F.softmax(attn.view(x.size(0), 1, -1), dim=-1).view(x.size(0), 1, x.size(2), x.size(3))
        return x * attn


# ---------------------------------------------------------------------------
# CD-ICA-Net Main Model
# ---------------------------------------------------------------------------

class CDICANet(nn.Module):
    """
    Causal Debiasing Iterative Cross-Attention Network (CD-ICA-Net).

    Architecture (5 stages):
        1. Dual-Branch CNN Encoder (face + context with highlight)
        2. Iterative Bidirectional Cross-Attention (ICA, N iterations)
        3. Integrated Causal Debiasing (CCIM, after cross-attention)
        4. Hybrid Adaptive Fusion (AA + DF blocks)
        5. Emotion Classifier

    Returns both conventional logits P(Y|X) and causal logits P(Y|do(X))
    for the causal intervention loss L_ica = KL(P(Y|X) || P(Y|do(X))).
    """

    def __init__(
        self,
        num_classes: int,
        backbone: str = "resnet18",
        pretrained: bool = False,
        dropout: float = 0.5,
        num_iterations: int = 3,
        confounder_dim: int = 512,
        num_confounders: int = 128,
        ccim_strategy: str = "dp_cause",
        aa_hidden_dim: int = 256,
        df_hidden_dim: int = 512,
    ) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.num_iterations = num_iterations
        self.confounder_dim = confounder_dim
        self.num_confounders = num_confounders

        # -------------------------------------------------------------------
        # Stage 1: Dual-Branch CNN Encoder
        # -------------------------------------------------------------------
        self.face_encoder, self.face_dim = _make_encoder(backbone, pretrained=pretrained)
        self.context_encoder, self.context_dim = _make_encoder(backbone, pretrained=pretrained)

        if self.face_dim != self.context_dim:
            raise ValueError(
                f"face_dim ({self.face_dim}) must equal context_dim ({self.context_dim})"
            )
        self.channels = self.face_dim

        # Context highlight module
        self.context_highlight = ContextHighlightModule(self.context_dim)

        # -------------------------------------------------------------------
        # Stage 2: Iterative Bidirectional Cross-Attention
        # -------------------------------------------------------------------
        # Infer spatial size from a dummy forward (assume square feature maps)
        # For ResNet-18 on 224×224 input, spatial size is 7×7
        self.spatial_size = 7  # default for ResNet-18 / typical backbones

        self.ica = ICAModule(
            channels=self.channels,
            spatial_size=self.spatial_size,
            num_iterations=num_iterations,
        )

        # -------------------------------------------------------------------
        # Stage 3: Integrated Causal Debiasing (CCIM)
        # -------------------------------------------------------------------
        self.ccim = IntegratedCCIM(
            feature_dim=self.channels,
            confounder_dim=confounder_dim,
            num_classes=num_classes,
            strategy=ccim_strategy,
            dropout=dropout,
        )

        # -------------------------------------------------------------------
        # Stage 4: Hybrid Adaptive Fusion (AA + DF)
        # -------------------------------------------------------------------
        self.fusion = HybridAdaptiveFusion(
            channels=self.channels,
            num_classes=num_classes,
            dropout=dropout,
            aa_hidden_dim=aa_hidden_dim,
            df_hidden_dim=df_hidden_dim,
        )

        # -------------------------------------------------------------------
        # Confounder dictionary (filled externally before training)
        # -------------------------------------------------------------------
        self.register_buffer("confounder_dict", torch.zeros(num_confounders, confounder_dim))
        self.register_buffer("confounder_prior", torch.ones(num_confounders, 1) / num_confounders)
        self.has_confounder = False

    def set_confounder_dict(
        self, confounder_dict: torch.Tensor, confounder_prior: torch.Tensor
    ) -> None:
        """Set confounder dictionary and prior (built from training data)."""
        if confounder_dict.shape != self.confounder_dict.shape:
            raise ValueError(
                f"Confounder dict shape mismatch: expected {self.confounder_dict.shape}, got {confounder_dict.shape}"
            )
        self.confounder_dict.copy_(confounder_dict)
        self.confounder_prior.copy_(confounder_prior)
        self.has_confounder = True

    def forward(
        self,
        face_image: torch.Tensor,
        context_image: torch.Tensor,
    ) -> dict[str, Any]:
        """
        Args:
            face_image:    [B, 3, H, W]
            context_image: [B, 3, H, W]
        Returns:
            dict with:
                - logits: [B, num_classes]          (P(Y|X) from fusion)
                - causal_logits: [B, num_classes]   (P(Y|do(X)) from CCIM)
                - various intermediate features for debugging/ablation
        """
        # -------------------------------------------------------------------
        # Stage 1: Encode
        # -------------------------------------------------------------------
        face_feat = self.face_encoder(face_image)["feat"]       # [B, C, H, W]
        context_feat = self.context_encoder(context_image)["feat"]  # [B, C, H, W]
        context_feat = self.context_highlight(context_feat)     # [B, C, H, W]

        # Store shallow features for fusion
        shallow_face_vec = F.adaptive_avg_pool2d(face_feat, 1).flatten(1)      # [B, C]
        shallow_ctx_vec = F.adaptive_avg_pool2d(context_feat, 1).flatten(1)    # [B, C]

        # -------------------------------------------------------------------
        # Stage 2: Iterative Cross-Attention
        # -------------------------------------------------------------------
        H_face_star, H_ctx_star, iteration_info = self.ica(face_feat, context_feat)

        # Deep features for fusion
        deep_face_vec = F.adaptive_avg_pool2d(H_face_star, 1).flatten(1)   # [B, C]
        deep_ctx_vec = F.adaptive_avg_pool2d(H_ctx_star, 1).flatten(1)     # [B, C]

        # -------------------------------------------------------------------
        # Stage 3: Causal Debiasing (CCIM)
        # -------------------------------------------------------------------
        if self.training and not self.has_confounder:
            raise RuntimeError(
                "Confounder dictionary not set. Call set_confounder_dict() before training, "
                "or use build_confounder_for_dataset to build it from the dataset."
            )

        ccim_out = self.ccim(
            H_face_star,
            H_ctx_star,
            self.confounder_dict,
            self.confounder_prior,
        )
        causal_logits = ccim_out["causal_logits"]  # [B, num_classes]

        # -------------------------------------------------------------------
        # Stage 4: Hybrid Adaptive Fusion + Stage 5: Classifier
        # -------------------------------------------------------------------
        fusion_out = self.fusion(
            shallow_face_vec,
            shallow_ctx_vec,
            deep_face_vec,
            deep_ctx_vec,
        )
        logits = fusion_out["logits"]  # [B, num_classes]  (P(Y|X))

        return {
            "logits": logits,
            "causal_logits": causal_logits,
            "lambda_shallow": fusion_out["lambda_shallow"],
            "lambda_deep": fusion_out["lambda_deep"],
            "h": ccim_out["h"],
            "E_z": ccim_out["E_z"],
            "shallow_face": shallow_face_vec,
            "shallow_context": shallow_ctx_vec,
            "deep_face": deep_face_vec,
            "deep_context": deep_ctx_vec,
            "iteration_info": iteration_info,
        }
