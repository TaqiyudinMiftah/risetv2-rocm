from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Spatial Transformer Network (STN)
# ---------------------------------------------------------------------------

class SpatialTransformer(nn.Module):
    """
    Spatial Transformer Network for learnable spatial transformation.
    Learns affine transformation parameters (scale, rotation, translation)
    to align input images before feature extraction.
    """

    def __init__(self, in_channels: int = 3) -> None:
        super().__init__()
        # Localization network
        self.localization = nn.Sequential(
            nn.Conv2d(in_channels, 8, kernel_size=7, padding=3),
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(True),
            nn.Conv2d(8, 10, kernel_size=5, padding=2),
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(True),
        )
        # Adaptive pooling to ensure fixed-size input to FC
        self.adaptive_pool = nn.AdaptiveAvgPool2d((7, 7))
        # Regressor for the 3x2 affine matrix
        self.fc_loc = nn.Sequential(
            nn.Linear(10 * 7 * 7, 32),
            nn.ReLU(True),
            nn.Linear(32, 6),
        )
        # Initialize to identity transformation
        self.fc_loc[2].weight.data.zero_()
        self.fc_loc[2].bias.data.copy_(torch.tensor([1, 0, 0, 0, 1, 0], dtype=torch.float))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, C, H, W] input image
        Returns:
            Transformed image [B, C, H, W]
        """
        B, C, H, W = x.shape
        xs = self.localization(x)
        xs = self.adaptive_pool(xs)
        xs = xs.view(B, -1)
        theta = self.fc_loc(xs)
        theta = theta.view(B, 2, 3)

        grid = F.affine_grid(theta, x.size(), align_corners=False)
        x_transformed = F.grid_sample(x, grid, align_corners=False)
        return x_transformed


# ---------------------------------------------------------------------------
# Squeeze-and-Excitation (SE) Block
# ---------------------------------------------------------------------------

class SEBlock(nn.Module):
    """
    Squeeze-and-Excitation block for channel-wise feature recalibration.
    """

    def __init__(self, channels: int, reduction: int = 16) -> None:
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, C, H, W] feature maps
        Returns:
            Recalibrated feature maps [B, C, H, W]
        """
        B, C, _, _ = x.size()
        y = self.avg_pool(x).view(B, C)
        y = self.fc(y).view(B, C, 1, 1)
        return x * y.expand_as(x)


# ---------------------------------------------------------------------------
# Hybrid ConvNeXt Encoder
# ---------------------------------------------------------------------------

class HybridConvNeXtEncoder(nn.Module):
    """
    Hybrid ConvNeXt encoder with STN and SE blocks.
    Based on AGCD-Net paper (ICIAP 2025).

    Architecture:
        1. STN (optional) for spatial alignment
        2. ConvNeXt backbone for feature extraction
        3. SE blocks after feature extraction for channel recalibration
    """

    def __init__(
        self,
        variant: str = "tiny",
        pretrained: bool = True,
        use_stn: bool = True,
        se_reduction: int = 16,
    ) -> None:
        super().__init__()
        self.use_stn = use_stn
        self.variant = variant

        # Spatial Transformer Network
        if use_stn:
            self.stn = SpatialTransformer(in_channels=3)

        # ConvNeXt backbone
        from torchvision import models
        if variant == "tiny":
            weights = models.ConvNeXt_Tiny_Weights.DEFAULT if pretrained else None
            backbone = models.convnext_tiny(weights=weights)
            self.feature_dim = 768
        elif variant == "small":
            weights = models.ConvNeXt_Small_Weights.DEFAULT if pretrained else None
            backbone = models.convnext_small(weights=weights)
            self.feature_dim = 768
        elif variant == "base":
            weights = models.ConvNeXt_Base_Weights.DEFAULT if pretrained else None
            backbone = models.convnext_base(weights=weights)
            self.feature_dim = 1024
        else:
            raise ValueError(f"Unsupported ConvNeXt variant: {variant}")

        # Remove classifier head, keep only features
        self.backbone = backbone.features

        # SE block after the last stage
        self.se = SEBlock(self.feature_dim, reduction=se_reduction)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        """
        Args:
            x: [B, 3, H, W] input image
        Returns:
            dict with "feat" key containing [B, C, H', W'] feature maps
        """
        if self.use_stn:
            x = self.stn(x)

        feat = self.backbone(x)  # [B, C, H', W']
        feat = self.se(feat)     # Channel recalibration

        return {"feat": feat}
