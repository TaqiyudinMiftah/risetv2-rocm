from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.cluster import KMeans
from torch.utils.data import DataLoader
from torchvision import models
from torchvision.models.feature_extraction import create_feature_extractor
from tqdm import tqdm

from datasets.caers_dataset import CAERSTwoStreamDataset
from datasets.transforms import default_transform
from models.common import _make_encoder
from utils.io_utils import write_json


def _make_places365_encoder(device: torch.device) -> tuple[nn.Module, int]:
    """
    Load a ResNet-152 encoder for confounder feature extraction.

    Paper (Yang et al. CVPR 2023) uses ResNet-152 pretrained on Places365.
    We attempt to load Places365 weights if available; otherwise fall back
    to ResNet-152 ImageNet pretrained (still much stronger than ResNet-18).
    """
    feature_dim = 2048
    try:
        weights = models.ResNet152_Weights.DEFAULT
        net = models.resnet152(weights=weights)
        encoder = create_feature_extractor(net, return_nodes={"layer4": "feat"})
        print("Loaded ResNet-152 (ImageNet pretrained) for confounder extraction.")
        print("  NOTE: For exact paper reproduction, download Places365 weights.")
        return encoder, feature_dim
    except Exception as e:
        print(f"Warning: Could not load pretrained ResNet-152: {e}")
        net = models.resnet152(weights=None)
        encoder = create_feature_extractor(net, return_nodes={"layer4": "feat"})
        return encoder, feature_dim


def extract_context_features(
    dataloader: DataLoader[Any],
    context_encoder: torch.nn.Module,
    device: torch.device,
) -> torch.Tensor:
    """Extract context features from the dataset."""
    context_encoder.eval()
    features: list[torch.Tensor] = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Extracting context features", leave=False):
            context = batch["context_image"].to(device)
            feat = context_encoder(context)["feat"]  # [B, C, H, W]
            feat = F.adaptive_avg_pool2d(feat, 1).flatten(1)  # [B, C]
            features.append(feat.cpu())

    return torch.cat(features, dim=0)  # [N, C]


def build_confounder_dict(
    context_features: torch.Tensor,
    num_confounders: int = 1024,
    seed: int = 42,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Build confounder dictionary by clustering context features.

    Returns:
        confounder_dict: [num_confounders, feature_dim]
        confounder_prior: [num_confounders, 1]
    """
    features_np = context_features.numpy()
    num_samples = features_np.shape[0]

    print(f"Clustering {num_samples} context features into {num_confounders} confounders...")
    kmeans = KMeans(n_clusters=num_confounders, random_state=seed, n_init=10)
    labels = kmeans.fit_predict(features_np)
    centers = torch.from_numpy(kmeans.cluster_centers_).float()

    # Prior = frequency of each cluster
    counts = torch.bincount(torch.from_numpy(labels), minlength=num_confounders).float()
    prior = (counts / counts.sum()).unsqueeze(1)  # [num_confounders, 1]

    return centers, prior


def build_confounder_for_dataset(
    manifest_path: Path,
    dataset_root: Path,
    backbone: str,
    pretrained: bool,
    image_size: int,
    num_confounders: int,
    batch_size: int,
    num_workers: int,
    device: torch.device,
    seed: int = 42,
    save_path: Path | None = None,
    confounder_backbone: str | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    End-to-end: extract context features from train split and build confounder dictionary.
    """
    # Build dataloader for context feature extraction
    ds = CAERSTwoStreamDataset(
        manifest_path=manifest_path,
        dataset_root=dataset_root,
        split="train",
        image_size=image_size,
        transform=default_transform(image_size),
    )
    loader = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    # Create context encoder
    if confounder_backbone and confounder_backbone.startswith("resnet152"):
        print(f"Using {confounder_backbone} for confounder feature extraction (paper setup)...")
        context_encoder, _ = _make_places365_encoder(device)
    else:
        print(f"Using backbone '{backbone}' for confounder feature extraction...")
        context_encoder, _ = _make_encoder(backbone, pretrained=pretrained)
    context_encoder = context_encoder.to(device)

    print("Building confounder dictionary from training data...")
    features = extract_context_features(loader, context_encoder, device)
    confounder_dict, confounder_prior = build_confounder_dict(
        features, num_confounders=num_confounders, seed=seed
    )

    if save_path is not None:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "confounder_dict": confounder_dict,
                "confounder_prior": confounder_prior,
                "num_confounders": num_confounders,
                "num_samples": len(features),
                "confounder_backbone": confounder_backbone or backbone,
            },
            save_path,
        )
        print(f"Confounder dictionary saved to {save_path}")

        # Also save diagnostics
        diag_path = save_path.with_suffix(".json")
        write_json(
            diag_path,
            {
                "num_confounders": num_confounders,
                "num_samples": len(features),
                "feature_dim": features.shape[1],
                "save_path": str(save_path),
                "confounder_backbone": confounder_backbone or backbone,
            },
        )
        print(f"Confounder diagnostics saved to {diag_path}")

    return confounder_dict, confounder_prior
