from __future__ import annotations

import math
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Cross-Attention Block (Channel-wise cross-attention)
# ---------------------------------------------------------------------------

class CrossAttentionBlock(nn.Module):
    """
    Cross-Attention block operating on channel dimension.
    Projects spatial features to Q/K/V, computes C×C attention matrix,
    and transforms the value representation.
    """

    def __init__(self, channels: int, attn_dim: int | None = None) -> None:
        super().__init__()
        self.channels = channels
        self.attn_dim = attn_dim or channels

        # 1×1 conv projections for Q, K, V (keep channel dim, only learn projection)
        self.q_proj = nn.Conv2d(channels, channels, kernel_size=1, bias=False)
        self.k_proj = nn.Conv2d(channels, channels, kernel_size=1, bias=False)
        self.v_proj = nn.Conv2d(channels, channels, kernel_size=1, bias=False)

        # After attention reshape back to spatial
        self.out_conv = nn.Conv2d(channels, channels, kernel_size=1, bias=False)
        self.out_bn = nn.BatchNorm2d(channels)

    def forward(
        self,
        query_feat: torch.Tensor,
        key_feat: torch.Tensor,
        value_feat: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            query_feat: [B, C, H, W]
            key_feat:   [B, C, H, W]
            value_feat: [B, C, H, W]
        Returns:
            out: [B, C, H, W]
        """
        B, C, H, W = query_feat.shape
        D = H * W

        # Project and flatten to [B, C, D]
        Q = self.q_proj(query_feat).view(B, C, D)
        K = self.k_proj(key_feat).view(B, C, D)
        V = self.v_proj(value_feat).view(B, C, D)

        # Scaled dot-product attention on channels: [B, C, C]
        scale = 1.0 / math.sqrt(self.attn_dim)
        attn = torch.bmm(Q, K.transpose(1, 2)) * scale  # [B, C, C]
        attn = F.softmax(attn, dim=-1)

        # Apply attention to values: [B, C, D]
        out = torch.bmm(attn, V)  # [B, C, D]

        # Reshape and project back to spatial feature map
        out = out.view(B, C, H, W)
        out = self.out_conv(out)
        out = self.out_bn(out)
        out = F.relu(out, inplace=True)

        return out


# ---------------------------------------------------------------------------
# Element Recalibration Block (Gram-like global info)
# ---------------------------------------------------------------------------

class ElementRecalibrationBlock(nn.Module):
    """
    Element Recalibration block.
    Embeds global spatial information via a transformation matrix and
    channel-channel self-attention, then recalibrates the feature map.
    """

    def __init__(self, channels: int, spatial_dim: int) -> None:
        super().__init__()
        self.channels = channels
        self.spatial_dim = spatial_dim

        # W^TM: transformation matrix operating on spatial dimension
        # Implemented as a linear layer [D -> D]
        self.spatial_transform = nn.Linear(spatial_dim, spatial_dim, bias=False)

        # Optional: channel projection to reduce compute for gram matrix
        self.channel_proj = nn.Conv2d(channels, channels, kernel_size=1, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, C, H, W]
        Returns:
            R^ER: [B, C, H, W]
        """
        B, C, H, W = x.shape
        D = H * W

        # Flatten spatial: [B, C, D]
        z = x.view(B, C, D)

        # Apply spatial transformation: [B, C, D]
        z_t = self.spatial_transform(z)

        # Compute channel-channel gram-like attention: [B, C, C]
        # G = softmax(Z_flat · W^TM · Z_flat^T)
        gram = torch.bmm(z, z_t.transpose(1, 2))  # [B, C, C]
        gram = F.softmax(gram, dim=-1)

        # Global info extraction: G · Z_flat -> [B, C, D]
        g = torch.bmm(gram, z)

        # Element-wise recalibration
        recalibrated = z * g  # [B, C, D]

        # Reshape back
        out = recalibrated.view(B, C, H, W)

        # Add residual connection for stability
        out = out + x

        return out


# ---------------------------------------------------------------------------
# Iterative Cross-Attention Module (ICA)
# ---------------------------------------------------------------------------

class ICAModule(nn.Module):
    """
    Iterative Bidirectional Cross-Attention Module.
    Performs N rounds of bidirectional cross-attention between face and context,
    each followed by element recalibration.
    """

    def __init__(
        self,
        channels: int,
        spatial_size: int = 7,
        num_iterations: int = 3,
        attn_dim: int | None = None,
    ) -> None:
        super().__init__()
        self.channels = channels
        self.spatial_dim = spatial_size * spatial_size
        self.num_iterations = num_iterations

        # Shared or per-iteration blocks? Use per-iteration for more capacity
        self.ca_f_to_c = nn.ModuleList(
            [CrossAttentionBlock(channels, attn_dim) for _ in range(num_iterations)]
        )
        self.ca_c_to_f = nn.ModuleList(
            [CrossAttentionBlock(channels, attn_dim) for _ in range(num_iterations)]
        )
        self.er_face = nn.ModuleList(
            [ElementRecalibrationBlock(channels, self.spatial_dim) for _ in range(num_iterations)]
        )
        self.er_context = nn.ModuleList(
            [ElementRecalibrationBlock(channels, self.spatial_dim) for _ in range(num_iterations)]
        )

    def forward(
        self,
        face_feat: torch.Tensor,
        context_feat: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, list[dict[str, Any]]]:
        """
        Args:
            face_feat:    [B, C, H, W]
            context_feat: [B, C, H, W]
        Returns:
            H_face_star:  [B, C, H, W]
            H_ctx_star:   [B, C, H, W]
            iteration_info: list of intermediate dicts (for debugging/ablation)
        """
        H_face = face_feat
        H_ctx = context_feat
        iteration_info: list[dict[str, Any]] = []

        for n in range(self.num_iterations):
            # Cross-Attention F → C: face queries context
            Z_face = self.ca_f_to_c[n](H_face, H_ctx, H_face)
            Z_face = self.er_face[n](Z_face)

            # Cross-Attention C → F: context queries updated face
            Z_ctx = self.ca_c_to_f[n](H_ctx, Z_face, H_ctx)
            Z_ctx = self.er_context[n](Z_ctx)

            # Update for next iteration
            H_face = Z_face
            H_ctx = Z_ctx

            iteration_info.append({
                "iteration": n + 1,
                "face_mean": H_face.mean().item(),
                "ctx_mean": H_ctx.mean().item(),
            })

        return H_face, H_ctx, iteration_info
