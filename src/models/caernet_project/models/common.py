"""
models/common.py
Backbone factory for CAER-Net.
Returns an EncoderWrapper whose forward() yields {"feat": Tensor[B,C,H,W]}.
"""

from __future__ import annotations

from typing import Dict, Tuple

import torch
import torch.nn as nn
import torchvision.models as tvm


# ---------------------------------------------------------------------------
# Wrapper so every backbone speaks the same dict interface
# ---------------------------------------------------------------------------

class EncoderWrapper(nn.Module):
    """Wraps a bare CNN trunk to output {'feat': Tensor}."""

    def __init__(self, trunk: nn.Module) -> None:
        super().__init__()
        self.trunk = trunk

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        return {"feat": self.trunk(x)}


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def _make_encoder(
    backbone_name: str,
    pretrained: bool = True,
) -> Tuple[EncoderWrapper, int]:
    """
    Build a spatial-feature encoder (no avgpool / fc).

    Returns
    -------
    encoder : EncoderWrapper  – outputs {"feat": [B, C, H, W]}
    feat_dim : int            – number of feature channels C
                                (7×7 spatial grid for 224×224 input)
    """
    def _weights(cls):
        return cls.DEFAULT if pretrained else None

    if backbone_name == "resnet18":
        base = tvm.resnet18(weights=_weights(tvm.ResNet18_Weights))
        feat_dim = 512
        trunk = nn.Sequential(*list(base.children())[:-2])   # drop avgpool+fc

    elif backbone_name == "resnet34":
        base = tvm.resnet34(weights=_weights(tvm.ResNet34_Weights))
        feat_dim = 512
        trunk = nn.Sequential(*list(base.children())[:-2])

    elif backbone_name == "resnet50":
        base = tvm.resnet50(weights=_weights(tvm.ResNet50_Weights))
        feat_dim = 2048
        trunk = nn.Sequential(*list(base.children())[:-2])

    elif backbone_name == "resnet101":
        base = tvm.resnet101(weights=_weights(tvm.ResNet101_Weights))
        feat_dim = 2048
        trunk = nn.Sequential(*list(base.children())[:-2])

    elif backbone_name == "vgg16":
        base = tvm.vgg16(weights=_weights(tvm.VGG16_Weights))
        feat_dim = 512                  # last conv block → [B,512,7,7]
        trunk = base.features

    elif backbone_name == "vgg19":
        base = tvm.vgg19(weights=_weights(tvm.VGG19_Weights))
        feat_dim = 512
        trunk = base.features

    else:
        raise ValueError(
            f"Unsupported backbone '{backbone_name}'. "
            "Choose: resnet18 | resnet34 | resnet50 | resnet101 | vgg16 | vgg19 | custom"
        )

    return EncoderWrapper(trunk), feat_dim
