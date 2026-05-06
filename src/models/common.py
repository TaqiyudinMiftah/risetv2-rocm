from __future__ import annotations

import torch.nn as nn
from torchvision import models
from torchvision.models.feature_extraction import create_feature_extractor


def _make_encoder(backbone: str = "resnet18", pretrained: bool = False) -> tuple[nn.Module, int]:
    """Create a CNN feature encoder that returns spatial feature maps."""
    if backbone == "resnet18":
        weights = models.ResNet18_Weights.DEFAULT if pretrained else None
        net = models.resnet18(weights=weights)
        encoder = create_feature_extractor(net, return_nodes={"layer4": "feat"})
        dim = 512
        return encoder, dim
    if backbone == "mobilenet_v2":
        weights = models.MobileNet_V2_Weights.DEFAULT if pretrained else None
        net = models.mobilenet_v2(weights=weights)
        encoder = create_feature_extractor(net, return_nodes={"features": "feat"})
        dim = 1280
        return encoder, dim
    raise ValueError(f"Unsupported backbone: {backbone}")
