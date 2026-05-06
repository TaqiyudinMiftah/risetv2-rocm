from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from models.common import _make_encoder


# ---------------------------------------------------------------------------
# Causal Intervention implementations
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
        key = self.key(confounder_set)  # [num_confounders, 256]
        mid = torch.matmul(query, key.t()) / math.sqrt(self.con_size)  # [B, num_confounders]
        attention = F.softmax(mid, dim=-1)  # [B, num_confounders]
        attention = attention.unsqueeze(2)  # [B, num_confounders, 1]
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
        key = self.key(confounder_set)  # [num_confounders, 256]
        query_expand = query.unsqueeze(1)  # [B, 1, 256]
        fuse = query_expand + key.unsqueeze(0)  # [B, num_confounders, 256]
        fuse = self.tanh(fuse)
        attention = self.w_t(fuse).squeeze(-1)  # [B, num_confounders]
        attention = F.softmax(attention, dim=1)
        fin = (attention.unsqueeze(2) * confounder_set.unsqueeze(0) * probabilities.unsqueeze(0)).sum(1)
        return fin


# ---------------------------------------------------------------------------
# Residual classifier block
# ---------------------------------------------------------------------------

class ResidualClassifier(nn.Module):
    def __init__(self, dim: int = 128, dropout: float = 0.5) -> None:
        super().__init__()
        self.fc1 = nn.Linear(dim, dim * 4)
        self.fc2 = nn.Linear(dim * 4, dim)
        self.drop = nn.Dropout(p=dropout)
        self.norm = nn.BatchNorm1d(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = self.norm(x)
        out = gelu(self.fc1(out))
        out = self.drop(out)
        out = self.fc2(out)
        out = self.drop(out)
        out = residual + out * 0.3
        return out


def gelu(x: torch.Tensor) -> torch.Tensor:
    return 0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))


# ---------------------------------------------------------------------------
# CCIM Module
# ---------------------------------------------------------------------------

class CCIM(nn.Module):
    """
    Contextual Causal Intervention Module.
    Plug-in module that takes a joint/fused feature and performs
    backdoor adjustment via confounder dictionary.
    """

    def __init__(
        self,
        num_joint_feature: int,
        num_gz: int,
        strategy: str = "dp_cause",
        dropout: float = 0.5,
    ) -> None:
        super().__init__()
        self.num_joint_feature = num_joint_feature
        self.num_gz = num_gz

        if strategy == "dp_cause":
            self.causal_intervention = DotProductIntervention(num_gz, num_joint_feature)
        elif strategy == "ad_cause":
            self.causal_intervention = AdditiveIntervention(num_gz, num_joint_feature)
        else:
            raise ValueError("strategy must be 'dp_cause' or 'ad_cause'")

        self.w_h = nn.Parameter(torch.Tensor(num_joint_feature, 128))
        self.w_g = nn.Parameter(torch.Tensor(num_gz, 128))
        self.classifier = ResidualClassifier(dim=128, dropout=dropout)
        self.final_fc = nn.Linear(128, 7)

        nn.init.xavier_normal_(self.w_h)
        nn.init.xavier_normal_(self.w_g)

    def forward(
        self,
        joint_feature: torch.Tensor,
        confounder_set: torch.Tensor,
        probabilities: torch.Tensor,
    ) -> torch.Tensor:
        g_z = self.causal_intervention(confounder_set, joint_feature, probabilities)
        proj_h = torch.matmul(joint_feature, self.w_h)  # [B, 128]
        proj_g_z = torch.matmul(g_z, self.w_g)  # [B, 128]
        do_x = proj_h + proj_g_z
        out = self.classifier(do_x)
        logits = self.final_fc(out)
        return logits


# ---------------------------------------------------------------------------
# Yang CCIM Network (full model)
# ---------------------------------------------------------------------------

class YangCCIMNet(nn.Module):
    """
    Full network for Yang et al. CVPR 2023.

    Two-stream encoders -> feature fusion -> CCIM -> logits.
    """

    def __init__(
        self,
        num_classes: int = 7,
        backbone: str = "resnet18",
        pretrained: bool = False,
        dropout: float = 0.5,
        num_confounders: int = 1024,
        confounder_feature_dim: int = 512,
        ccim_strategy: str = "dp_cause",
    ) -> None:
        super().__init__()
        self.face_encoder, self.face_dim = _make_encoder(backbone, pretrained=pretrained)
        self.context_encoder, self.context_dim = _make_encoder(backbone, pretrained=pretrained)

        # Joint feature = concatenation of face + context feature vectors
        self.joint_dim = self.face_dim + self.context_dim

        # Project to fixed dim for CCIM consistency
        self.joint_proj = nn.Linear(self.joint_dim, confounder_feature_dim)

        self.ccim = CCIM(
            num_joint_feature=confounder_feature_dim,
            num_gz=confounder_feature_dim,
            strategy=ccim_strategy,
            dropout=dropout,
        )

        # Register buffers for confounder dictionary (filled externally before training)
        self.register_buffer("confounder_dict", torch.zeros(num_confounders, confounder_feature_dim))
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

    def extract_joint_feature(self, face_image: torch.Tensor, context_image: torch.Tensor) -> torch.Tensor:
        face_feat = self.face_encoder(face_image)["feat"]  # [B, C, H, W]
        context_feat = self.context_encoder(context_image)["feat"]

        # Global average pooling
        face_vec = F.adaptive_avg_pool2d(face_feat, 1).flatten(1)
        context_vec = F.adaptive_avg_pool2d(context_feat, 1).flatten(1)

        joint = torch.cat([face_vec, context_vec], dim=1)  # [B, face_dim + context_dim]
        joint = self.joint_proj(joint)  # [B, confounder_feature_dim]
        return joint

    def forward(
        self,
        face_image: torch.Tensor,
        context_image: torch.Tensor,
    ) -> dict[str, Any]:
        joint_feature = self.extract_joint_feature(face_image, context_image)

        if self.training and not self.has_confounder:
            raise RuntimeError(
                "Confounder dictionary not set. Call set_confounder_dict() before training, "
                "or use ConfounderBuilder to build it from the dataset."
            )

        logits = self.ccim(joint_feature, self.confounder_dict, self.confounder_prior)

        return {
            "logits": logits,
            "joint_feature": joint_feature,
        }
