from __future__ import annotations

import math
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Causal Intervention implementations (adapted from Yang et al.)
# ---------------------------------------------------------------------------

class DotProductIntervention(nn.Module):
    """Dot-product causal intervention (dp_cause)."""

    def __init__(self, con_size: int, fuse_size: int) -> None:
        super().__init__()
        self.con_size = con_size
        self.fuse_size = fuse_size
        self.query = nn.Linear(fuse_size, 256, bias=False)
        self.key = nn.Linear(con_size, 256, bias=False)

    def forward(
        self, confounder_set: torch.Tensor, fuse_rep: torch.Tensor, probabilities: torch.Tensor
    ) -> torch.Tensor:
        query = self.query(fuse_rep)  # [B, 256]
        key = self.key(confounder_set)  # [K, 256]
        mid = torch.matmul(query, key.t()) / math.sqrt(self.con_size)  # [B, K]
        attention = F.softmax(mid, dim=-1)  # [B, K]
        attention = attention.unsqueeze(2)  # [B, K, 1]
        fin = (attention * confounder_set.unsqueeze(0) * probabilities.unsqueeze(0)).sum(1)  # [B, con_size]
        return fin


class AdditiveIntervention(nn.Module):
    """Additive causal intervention (ad_cause)."""

    def __init__(self, con_size: int, fuse_size: int) -> None:
        super().__init__()
        self.con_size = con_size
        self.fuse_size = fuse_size
        self.query = nn.Linear(fuse_size, 256, bias=False)
        self.key = nn.Linear(con_size, 256, bias=False)
        self.w_t = nn.Linear(256, 1, bias=False)
        self.tanh = nn.Tanh()

    def forward(
        self, confounder_set: torch.Tensor, fuse_rep: torch.Tensor, probabilities: torch.Tensor
    ) -> torch.Tensor:
        query = self.query(fuse_rep)  # [B, 256]
        key = self.key(confounder_set)  # [K, 256]
        query_expand = query.unsqueeze(1)  # [B, 1, 256]
        fuse = query_expand + key.unsqueeze(0)  # [B, K, 256]
        fuse = self.tanh(fuse)
        attention = self.w_t(fuse).squeeze(-1)  # [B, K]
        attention = F.softmax(attention, dim=1)
        fin = (attention.unsqueeze(2) * confounder_set.unsqueeze(0) * probabilities.unsqueeze(0)).sum(1)
        return fin


# ---------------------------------------------------------------------------
# Integrated CCIM Module (placed AFTER iterative cross-attention)
# ---------------------------------------------------------------------------

class IntegratedCCIM(nn.Module):
    """
    Integrated Contextual Causal Intervention Module.

    Unlike Yang et al. (CVPR 2023) where CCIM is a plug-in on raw encoder
    features, this module operates on representations already enriched by
    iterative cross-attention (H_face*, H_ctx*).

    Computes:
        h = φ(concat(GAP(H_face*), GAP(H_ctx*)))
        P(Y|do(X)) = W_h · h + W_g · E_z[g(z)]
    """

    def __init__(
        self,
        feature_dim: int,
        confounder_dim: int,
        num_classes: int,
        strategy: str = "dp_cause",
        dropout: float = 0.5,
    ) -> None:
        super().__init__()
        self.feature_dim = feature_dim
        self.confounder_dim = confounder_dim
        self.num_classes = num_classes

        # Joint representation projection
        # Input: concat(GAP(face), GAP(context)) -> [B, feature_dim * 2]
        self.joint_proj = nn.Sequential(
            nn.Linear(feature_dim * 2, confounder_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
        )

        # Causal intervention
        if strategy == "dp_cause":
            self.causal_intervention = DotProductIntervention(confounder_dim, confounder_dim)
        elif strategy == "ad_cause":
            self.causal_intervention = AdditiveIntervention(confounder_dim, confounder_dim)
        else:
            raise ValueError("strategy must be 'dp_cause' or 'ad_cause'")

        # Causal prediction parameters
        self.w_h = nn.Parameter(torch.Tensor(confounder_dim, num_classes))
        self.w_g = nn.Parameter(torch.Tensor(confounder_dim, num_classes))
        nn.init.xavier_normal_(self.w_h)
        nn.init.xavier_normal_(self.w_g)

        # Optional: classifier head for P(Y|do(X))
        self.causal_classifier = nn.Sequential(
            nn.Linear(confounder_dim, confounder_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(confounder_dim, num_classes),
        )

    def forward(
        self,
        H_face_star: torch.Tensor,
        H_ctx_star: torch.Tensor,
        confounder_dict: torch.Tensor,
        confounder_prior: torch.Tensor,
    ) -> dict[str, Any]:
        """
        Args:
            H_face_star: [B, C, H, W] enriched face features
            H_ctx_star:  [B, C, H, W] enriched context features
            confounder_dict: [K, confounder_dim]
            confounder_prior: [K, 1]
        Returns:
            dict with:
                - causal_logits: [B, num_classes]  (P(Y|do(X)))
                - h: [B, confounder_dim] joint representation
                - E_z: [B, confounder_dim] expected confounder effect
        """
        # Global Average Pooling
        face_vec = F.adaptive_avg_pool2d(H_face_star, 1).flatten(1)  # [B, C]
        ctx_vec = F.adaptive_avg_pool2d(H_ctx_star, 1).flatten(1)   # [B, C]

        # Joint representation
        joint = torch.cat([face_vec, ctx_vec], dim=1)  # [B, C*2]
        h = self.joint_proj(joint)  # [B, confounder_dim]

        # Causal intervention: backdoor adjustment
        E_z = self.causal_intervention(confounder_dict, h, confounder_prior)  # [B, confounder_dim]

        # Causal prediction: P(Y|do(X)) = W_h · h + W_g · E_z[g(z)]
        proj_h = torch.matmul(h, self.w_h)      # [B, num_classes]
        proj_g = torch.matmul(E_z, self.w_g)    # [B, num_classes]
        causal_logits = proj_h + proj_g

        # Also pass through classifier for richer non-linear causal prediction
        causal_logits = causal_logits + self.causal_classifier(h)

        return {
            "causal_logits": causal_logits,
            "h": h,
            "E_z": E_z,
        }
