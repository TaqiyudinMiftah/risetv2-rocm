"""
Official CAER-Net-S reproduction matching https://github.com/ndkhanh360/CAER.git

Key differences from our ResNet variant:
- Custom 5-layer CNN encoder (NOT ResNet backbone)
- Face: 96x96, Context: 112x112 (random crop from 128x171)
- Attention inference module on context features
- Fusion: GAP -> BN -> learned scalar weights (FC) -> softmax -> weighted concat -> FC
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class Encoder(nn.Module):
    """Custom CNN encoder from official repo."""
    def __init__(self, num_kernels, kernel_size=3, bn=True, max_pool=True, maxpool_kernel_size=2):
        super().__init__()
        padding = (kernel_size - 1) // 2
        n = len(num_kernels) - 1
        self.convs = nn.ModuleList([
            nn.Conv2d(num_kernels[i], num_kernels[i+1], kernel_size, padding=padding)
            for i in range(n)
        ])
        self.bn = nn.ModuleList([
            nn.BatchNorm2d(num_kernels[i+1]) for i in range(n)
        ]) if bn else None
        self.max_pool = nn.MaxPool2d(maxpool_kernel_size) if max_pool else None

    def forward(self, x):
        n = len(self.convs)
        for i in range(n):
            x = self.convs[i](x)
            if self.bn is not None:
                x = self.bn[i](x)
            x = F.relu(x)
            if self.max_pool is not None and i < n - 1:
                x = self.max_pool(x)
        return x


class TwoStreamNetwork(nn.Module):
    """Face + Context encoding with attention on context."""
    def __init__(self):
        super().__init__()
        num_kernels = [3, 32, 64, 128, 256, 256]
        self.face_encoding_module = Encoder(num_kernels)
        self.context_encoding_module = Encoder(num_kernels)
        self.attention_inference_module = Encoder([256, 128, 1], max_pool=False)

    def forward(self, face, context):
        face = self.face_encoding_module(face)
        context = self.context_encoding_module(context)
        attention = self.attention_inference_module(context)
        N, C, H, W = attention.shape
        attention = F.softmax(attention.view(N, -1), dim=-1).view(N, C, H, W)
        context = context * attention
        return face, context


class FusionNetwork(nn.Module):
    """Fusion with learned scalar weights."""
    def __init__(self, num_class=7, dropout=0.5):
        super().__init__()
        self.face_bn = nn.BatchNorm1d(256)
        self.context_bn = nn.BatchNorm1d(256)

        self.face_1 = nn.Linear(256, 128)
        self.face_2 = nn.Linear(128, 1)

        self.context_1 = nn.Linear(256, 128)
        self.context_2 = nn.Linear(128, 1)

        self.fc1 = nn.Linear(512, 128)
        self.fc2 = nn.Linear(128, num_class)

        self.dropout = nn.Dropout(dropout)

    def forward(self, face, context):
        face = F.avg_pool2d(face, face.shape[2]).view(face.shape[0], -1)
        context = F.avg_pool2d(context, context.shape[2]).view(context.shape[0], -1)

        face, context = self.face_bn(face), self.context_bn(context)

        lambda_f = F.relu(self.face_1(face))
        lambda_c = F.relu(self.context_1(context))

        lambda_f = self.face_2(lambda_f)
        lambda_c = self.context_2(lambda_c)

        weights = torch.cat([lambda_f, lambda_c], dim=-1)
        weights = F.softmax(weights, dim=-1)
        face = face * weights[:, 0].unsqueeze(dim=-1)
        context = context * weights[:, 1].unsqueeze(dim=-1)

        features = torch.cat([face, context], dim=-1)
        features = F.relu(self.fc1(features))
        features = self.dropout(features)

        return self.fc2(features)


class CAERNetOfficial(nn.Module):
    """Official CAER-Net-S."""
    def __init__(self, num_classes=7, dropout=0.5):
        super().__init__()
        self.two_stream_net = TwoStreamNetwork()
        self.fusion_net = FusionNetwork(num_class=num_classes, dropout=dropout)

    def forward(self, face_image, context_image):
        face, context = self.two_stream_net(face_image, context_image)
        logits = self.fusion_net(face, context)
        return {"logits": logits}
