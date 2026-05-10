"""Utility to load/download Places365 pretrained models for confounder extraction.

Paper reference (Yang et al. CVPR 2023):
"We use ResNet-152 pretrained on Places365 to extract context features
for building the confounder dictionary."

Weights source:
- CSAILVision/places365: http://places2.csail.mit.edu/models_places365/
- Fallback: torchvision ResNet-152 ImageNet pretrained
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from torchvision import models
from torchvision.models.feature_extraction import create_feature_extractor


def _download_places365_resnet152(cache_dir: Path | None = None) -> Path | None:
    """Download Places365 ResNet-152 weights if not cached."""
    if cache_dir is None:
        cache_dir = Path.home() / ".cache" / "cd_ica_net" / "places365"
    cache_dir.mkdir(parents=True, exist_ok=True)

    weight_path = cache_dir / "resnet152_places365.pth"
    if weight_path.exists():
        return weight_path

    # Try downloading from CSAILVision (multiple possible URLs)
    urls = [
        "https://github.com/CSAILVision/places365/raw/master/models/resnet152_places365.pth.tar",
        "http://places2.csail.mit.edu/models_places365/resnet152_places365.pth.tar",
    ]
    for url in urls:
        try:
            import urllib.request

            print(f"Downloading Places365 ResNet-152 weights from {url}...")
            print("This may take a few minutes (~230MB)...")
            tar_path = cache_dir / "resnet152_places365.pth.tar"
            urllib.request.urlretrieve(url, str(tar_path))

            # Extract .pth from .tar
            import tarfile

            with tarfile.open(str(tar_path), "r") as tar:
                tar.extractall(path=str(cache_dir))
            tar_path.unlink()  # remove tar after extraction

            if weight_path.exists():
                print(f"Places365 weights saved to {weight_path}")
                return weight_path
        except Exception as e:
            print(f"Warning: Failed to download from {url}: {e}")
            continue

    print("\n" + "=" * 70)
    print("Could not auto-download Places365 weights.")
    print("Please download manually and place at:")
    print(f"  {weight_path}")
    print("\nDownload links:")
    print("  https://github.com/CSAILVision/places365/tree/master/models")
    print("  https://drive.google.com/drive/folders/1H6EvH9oMt_5GY-FdW8Er3dF7_Y8n3fU1")
    print("=" * 70 + "\n")
    return None


def _load_places365_state_dict(weight_path: Path) -> dict[str, Any]:
    """Load Places365 state dict and remap keys for torchvision ResNet-152."""
    checkpoint = torch.load(str(weight_path), map_location="cpu", weights_only=False)

    # Places365 checkpoint may have different key formats
    if "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    elif "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    else:
        state_dict = checkpoint

    # Remap keys if needed
    remapped: dict[str, Any] = {}
    for k, v in state_dict.items():
        # Remove 'module.' prefix if present
        if k.startswith("module."):
            k = k[7:]
        # Handle various key naming conventions
        if k.startswith("features."):
            k = k.replace("features.", "")
        if k.startswith("classifier."):
            k = k.replace("classifier.", "fc.")
        remapped[k] = v

    return remapped


def make_places365_encoder(
    device: torch.device,
    cache_dir: Path | None = None,
) -> tuple[nn.Module, int]:
    """
    Create a ResNet-152 feature encoder for confounder extraction.

    Priority:
    1. Load Places365 pretrained weights (if available/downloaded)
    2. Fallback to ImageNet pretrained ResNet-152
    3. Fallback to uninitialized ResNet-152
    """
    feature_dim = 2048

    # Try loading Places365 weights
    places365_path = None
    if cache_dir is not None:
        places365_path = cache_dir / "resnet152_places365.pth"
    else:
        default_cache = Path.home() / ".cache" / "cd_ica_net" / "places365"
        places365_path = default_cache / "resnet152_places365.pth"

    if places365_path and places365_path.exists():
        try:
            print(f"Loading Places365 ResNet-152 weights from {places365_path}...")
            net = models.resnet152(weights=None)
            state_dict = _load_places365_state_dict(places365_path)

            # Load weights (skip 'fc' layer since we only use features)
            model_dict = net.state_dict()
            pretrained_dict = {k: v for k, v in state_dict.items() if k in model_dict and v.shape == model_dict[k].shape}
            model_dict.update(pretrained_dict)
            net.load_state_dict(model_dict)

            encoder = create_feature_extractor(net, return_nodes={"layer4": "feat"})
            print(f"  Loaded {len(pretrained_dict)}/{len(model_dict)} layers from Places365 weights.")
            return encoder.to(device), feature_dim
        except Exception as e:
            print(f"Warning: Failed to load Places365 weights: {e}")

    # Try downloading Places365
    downloaded_path = _download_places365_resnet152(cache_dir)
    if downloaded_path and downloaded_path.exists():
        return make_places365_encoder(device, cache_dir)

    # Fallback to ImageNet pretrained
    try:
        print("Loading ResNet-152 (ImageNet pretrained) for confounder extraction...")
        weights = models.ResNet152_Weights.DEFAULT
        net = models.resnet152(weights=weights)
        encoder = create_feature_extractor(net, return_nodes={"layer4": "feat"})
        print("  NOTE: Using ImageNet weights. For exact paper reproduction, download Places365 weights:")
        print("    wget http://places2.csail.mit.edu/models_places365/resnet152_places365.pth.tar")
        return encoder.to(device), feature_dim
    except Exception as e:
        print(f"Warning: Could not load pretrained ResNet-152: {e}")

    # Final fallback: uninitialized
    net = models.resnet152(weights=None)
    encoder = create_feature_extractor(net, return_nodes={"layer4": "feat"})
    return encoder.to(device), feature_dim
