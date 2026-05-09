from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Adaptive-Attention (AA) Block
# ---------------------------------------------------------------------------

class AdaptiveAttentionBlock(nn.Module):
    """
    Adaptive-Attention block with hybrid feature weighting.
    Computes shallow and deep fusion weights for face and context streams.
    """

    def __init__(self, channels: int, hidden_dim: int = 256) -> None:
        super().__init__()
        self.channels = channels

        # Shallow weights: from shallow features (face & context)
        self.shallow_weight_net = nn.Sequential(
            nn.Linear(channels * 2, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 2),
            nn.Softmax(dim=1),
        )

        # Deep weights: from deep features (face & context)
        self.deep_weight_net = nn.Sequential(
            nn.Linear(channels * 2, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 2),
            nn.Softmax(dim=1),
        )

    def forward(
        self,
        shallow_face: torch.Tensor,
        shallow_context: torch.Tensor,
        deep_face: torch.Tensor,
        deep_context: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            shallow_face:    [B, C] from Stage 1 encoder
            shallow_context: [B, C] from Stage 1 encoder
            deep_face:       [B, C] from Stage 2/3 (ICA enriched)
            deep_context:    [B, C] from Stage 2/3 (ICA enriched)
        Returns:
            f0_shallow, f1_shallow, f0_deep, f1_deep,
            lambda_shallow [B, 2], lambda_deep [B, 2]
        """
        # Compute weights
        shallow_combined = torch.cat([shallow_face, shallow_context], dim=1)  # [B, C*2]
        deep_combined = torch.cat([deep_face, deep_context], dim=1)          # [B, C*2]

        lambda_shallow = self.shallow_weight_net(shallow_combined)  # [B, 2]
        lambda_deep = self.deep_weight_net(deep_combined)           # [B, 2]

        # 4 adaptive features
        # f0_shallow = λ_deep[0] ⊙ shallow_context
        # f1_shallow = λ_deep[1] ⊙ shallow_face
        # f0_deep    = λ_shallow[0] ⊙ deep_context
        # f1_deep    = λ_shallow[1] ⊙ deep_face
        f0_shallow = lambda_deep[:, 0].unsqueeze(1) * shallow_context
        f1_shallow = lambda_deep[:, 1].unsqueeze(1) * shallow_face
        f0_deep = lambda_shallow[:, 0].unsqueeze(1) * deep_context
        f1_deep = lambda_shallow[:, 1].unsqueeze(1) * deep_face

        return f0_shallow, f1_shallow, f0_deep, f1_deep, lambda_shallow, lambda_deep


# ---------------------------------------------------------------------------
# Deep Fusion (DF) Block
# ---------------------------------------------------------------------------

class DeepFusionBlock(nn.Module):
    """
    Deep Fusion block.
    Hierarchically fuses adaptive features and produces classification logits.
    """

    def __init__(
        self,
        channels: int,
        num_classes: int,
        dropout: float = 0.5,
        hidden_dim: int = 512,
    ) -> None:
        super().__init__()
        self.channels = channels

        # f1 = Dropout(ReLU(Conv1D(concat(f0_shallow, f0_deep))))
        self.fusion_branch_1 = nn.Sequential(
            nn.Linear(channels * 2, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
        )

        # f2 = Dropout(ReLU(Conv1D(concat(f1_shallow, f1_deep))))
        self.fusion_branch_2 = nn.Sequential(
            nn.Linear(channels * 2, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
        )

        # X_fusion = concat(f1, f2)
        # x_cls = softmax(Conv1D(Dropout(ReLU(Conv1D(X_fusion)))))
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(
        self,
        f0_shallow: torch.Tensor,
        f1_shallow: torch.Tensor,
        f0_deep: torch.Tensor,
        f1_deep: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            Each input: [B, C]
        Returns:
            logits: [B, num_classes]
        """
        f1 = self.fusion_branch_1(torch.cat([f0_shallow, f0_deep], dim=1))  # [B, hidden_dim]
        f2 = self.fusion_branch_2(torch.cat([f1_shallow, f1_deep], dim=1))  # [B, hidden_dim]

        X_fusion = torch.cat([f1, f2], dim=1)  # [B, hidden_dim*2]
        logits = self.classifier(X_fusion)      # [B, num_classes]

        return logits


# ---------------------------------------------------------------------------
# Hybrid Adaptive Fusion Module (AA + DF)
# ---------------------------------------------------------------------------

class HybridAdaptiveFusion(nn.Module):
    """
    Hybrid Adaptive Fusion combining Adaptive-Attention (AA) and Deep Fusion (DF).
    """

    def __init__(
        self,
        channels: int,
        num_classes: int,
        dropout: float = 0.5,
        aa_hidden_dim: int = 256,
        df_hidden_dim: int = 512,
    ) -> None:
        super().__init__()
        self.aa = AdaptiveAttentionBlock(channels, hidden_dim=aa_hidden_dim)
        self.df = DeepFusionBlock(channels, num_classes, dropout=dropout, hidden_dim=df_hidden_dim)

    def forward(
        self,
        shallow_face: torch.Tensor,
        shallow_context: torch.Tensor,
        deep_face: torch.Tensor,
        deep_context: torch.Tensor,
    ) -> dict[str, Any]:
        """
        Args:
            shallow_face:    [B, C]
            shallow_context: [B, C]
            deep_face:       [B, C]
            deep_context:    [B, C]
        Returns:
            dict with logits, lambda_shallow, lambda_deep
        """
        f0_s, f1_s, f0_d, f1_d, lambda_shallow, lambda_deep = self.aa(
            shallow_face, shallow_context, deep_face, deep_context
        )
        logits = self.df(f0_s, f1_s, f0_d, f1_d)

        return {
            "logits": logits,
            "lambda_shallow": lambda_shallow,
            "lambda_deep": lambda_deep,
        }
