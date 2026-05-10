from __future__ import annotations

import torch.nn as nn
from torchvision import models
from torchvision.models.feature_extraction import create_feature_extractor


class ShallowCNNEncoder(nn.Module):
    """
    Custom 5-layer CNN encoder used in CAER-Net, GLAMOR-Net, and CAHFW-Net papers.

    Architecture per paper (ICCV 2019 / NCA 2022 / IJERPH 2023):
        Conv1: 32 filters  -> MaxPool
        Conv2: 64 filters  -> MaxPool
        Conv3: 128 filters -> MaxPool
        Conv4: 256 filters -> MaxPool
        Conv5: 256 filters -> (no pool)

    Each conv block: Conv2D(3x3) + BatchNorm + ReLU

    Output: spatial feature maps with 256 channels.
    """

    def __init__(self, channels_list: tuple[int, ...] = (32, 64, 128, 256, 256)) -> None:
        super().__init__()
        self.channels_list = channels_list
        layers: list[nn.Module] = []
        in_ch = 3
        for i, out_ch in enumerate(channels_list):
            layers.append(nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False))
            layers.append(nn.BatchNorm2d(out_ch))
            layers.append(nn.ReLU(inplace=True))
            # MaxPool for first 4 layers, no pool for last layer
            if i < len(channels_list) - 1:
                layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
            in_ch = out_ch
        self.encoder = nn.Sequential(*layers)
        self.out_channels = channels_list[-1]

    def forward(self, x):
        feat = self.encoder(x)
        return {"feat": feat}


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
    if backbone == "resnet50":
        weights = models.ResNet50_Weights.DEFAULT if pretrained else None
        net = models.resnet50(weights=weights)
        encoder = create_feature_extractor(net, return_nodes={"layer4": "feat"})
        dim = 2048
        return encoder, dim
    if backbone == "resnet101":
        weights = models.ResNet101_Weights.DEFAULT if pretrained else None
        net = models.resnet101(weights=weights)
        encoder = create_feature_extractor(net, return_nodes={"layer4": "feat"})
        dim = 2048
        return encoder, dim
    if backbone == "shallow_cnn":
        # Paper-specific custom CNN (no ImageNet pretraining available)
        encoder = ShallowCNNEncoder(channels_list=(32, 64, 128, 256, 256))
        dim = 256
        return encoder, dim
    raise ValueError(f"Unsupported backbone: {backbone}")
