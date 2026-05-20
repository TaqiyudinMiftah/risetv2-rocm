from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Multi-Head Self-Attention (MHSA)
# ---------------------------------------------------------------------------

class MultiHeadSelfAttention(nn.Module):
    """
    Multi-Head Self-Attention module for face and context features.
    Transforms spatial feature maps into query/key/value and applies attention.
    """

    def __init__(self, dim: int, num_heads: int = 8, dropout: float = 0.1) -> None:
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        assert self.head_dim * num_heads == dim, "dim must be divisible by num_heads"

        self.qkv_proj = nn.Linear(dim, dim * 3)
        self.out_proj = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)
        self.scale = self.head_dim ** -0.5

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, N, C] where N = H*W (flattened spatial features)
        Returns:
            Attended features [B, N, C]
        """
        B, N, C = x.shape
        qkv = self.qkv_proj(x).reshape(B, N, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # [3, B, num_heads, N, head_dim]
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)

        out = (attn @ v).transpose(1, 2).reshape(B, N, C)
        out = self.out_proj(out)
        return out


# ---------------------------------------------------------------------------
# Attention Guided - Causal Intervention Module (AG-CIM)
# ---------------------------------------------------------------------------

class AGCIM(nn.Module):
    """
    Attention Guided - Causal Intervention Module.

    Performs instance-level context debiasing guided by face features:
    1. Generate counterfactual context via perturbation
    2. Compute context bias as difference between original and perturbed
    3. Correct context using face-guided attention
    """

    def __init__(self, dim: int, num_heads: int = 8, dropout: float = 0.1) -> None:
        super().__init__()
        self.dim = dim

        # Multi-Head Self-Attention for face and context
        self.face_mhsa = MultiHeadSelfAttention(dim, num_heads=num_heads, dropout=dropout)
        self.context_mhsa = MultiHeadSelfAttention(dim, num_heads=num_heads, dropout=dropout)

        # Layer normalization
        self.face_norm = nn.LayerNorm(dim)
        self.context_norm = nn.LayerNorm(dim)

        # Perturbation weight for counterfactual generation
        self.W_p = nn.Linear(dim, dim)

        # Correction weight
        self.W_c = nn.Linear(dim, dim)

        # Learnable parameter alpha for face influence
        self.alpha = nn.Parameter(torch.ones(1))

    def forward(
        self,
        face_feat: torch.Tensor,
        context_feat: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            face_feat: [B, N, C] face features (flattened spatial)
            context_feat: [B, N, C] context features (flattened spatial)

        Returns:
            face_att: [B, N, C] attended face features
            context_corr: [B, N, C] corrected context features
            h_face: [B, N] attention weights for face (for loss)
            h_context: [B, N] attention weights for context (for loss)
        """
        # Apply MHSA
        face_att = self.face_mhsa(face_feat)
        face_att = self.face_norm(face_att + face_feat)

        context_att = self.context_mhsa(context_feat)
        context_att = self.context_norm(context_att + context_feat)

        # Compute attention weights (mean across heads for regularization loss)
        h_face = face_att.mean(dim=-1)      # [B, N]
        h_context = context_att.mean(dim=-1)  # [B, N]

        # Step 1: Generate counterfactual context via perturbation
        context_pert = self.W_p(context_att)  # [B, N, C]

        # Step 2: Compute context bias
        delta_context = context_att - context_pert  # [B, N, C]

        # Step 3: Bias correction using face attention
        # W_c * delta_context
        correction = self.W_c(delta_context)  # [B, N, C]

        # Face-guided gating: sigmoid(alpha * face_att)
        gate = torch.sigmoid(self.alpha * face_att)  # [B, N, C]

        # Corrected context
        context_corr = context_att - correction * gate  # [B, N, C]

        return face_att, context_corr, h_face, h_context
