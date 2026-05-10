from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


# ---------------------------------------------------------------------------
# Shared configs
# ---------------------------------------------------------------------------


@dataclass
class DatasetConfig:
    dataset_root: Path
    image_extensions: tuple[str, ...]
    create_val_split: bool
    val_ratio: float
    image_size: int


@dataclass
class OutputConfig:
    manifest_path: Path
    diagnostics_path: Path


@dataclass
class TrainConfig:
    batch_size: int = 32
    num_epochs: int = 30
    lr: float = 1e-3
    weight_decay: float = 1e-4
    num_workers: int = 4
    device: str = ""  # auto-detect (cuda / mps / cpu)
    seed: int = 42
    save_dir: Path = field(default_factory=lambda: Path("checkpoints"))
    stream_mode: str = "multimodal"  # for backward compat with CAER-Net


# ---------------------------------------------------------------------------
# Method-specific configs (extensible)
# ---------------------------------------------------------------------------


@dataclass
class CAERNetConfig:
    backbone: str = "resnet18"
    pretrained: bool = True
    dropout: float = 0.5
    face_size: int = 96  # face crop size per paper


@dataclass
class ZhouCrossAttentionConfig:
    backbone: str = "resnet18"
    pretrained: bool = True
    dropout: float = 0.5
    ca_reduction: int = 16          # Cross-Attention channel reduction
    er_reduction: int = 16          # Element Recalibration reduction
    aa_hidden_dim: int = 256        # Adaptive-Attention hidden dim
    df_hidden_dim: int = 512        # Deep Fusion hidden dim


@dataclass
class YangCCIMConfig:
    backbone: str = "resnet18"
    pretrained: bool = True
    dropout: float = 0.5
    num_confounders: int = 1024
    confounder_feature_dim: int = 512
    ccim_strategy: str = "dp_cause"  # "dp_cause" or "ad_cause"


@dataclass
class GLAMORNetConfig:
    backbone: str = "resnet18"
    pretrained: bool = True
    dropout: float = 0.5
    face_size: int = 96              # face crop size per paper
    gla_hidden_dim: int = 128        # GLA attention hidden dim
    fusion_hidden_dim: int = 128     # Fusion module hidden dim
    classifier_hidden_dim: int = 128 # Final classifier hidden dim


@dataclass
class CDICANetConfig:
    backbone: str = "resnet18"
    pretrained: bool = True
    dropout: float = 0.5
    num_iterations: int = 3          # N: iterative CA rounds
    confounder_dim: int = 512        # CCIM feature / confounder dimension
    num_confounders: int = 128       # K: confounder dictionary size
    ccim_strategy: str = "dp_cause"  # "dp_cause" or "ad_cause"
    aa_hidden_dim: int = 256         # Adaptive-Attention hidden dim
    df_hidden_dim: int = 512         # Deep Fusion hidden dim
    alpha_ica: float = 0.5           # Weight for L_ica
    beta_reg: float = 0.1            # Weight for L_reg
    flood_level: float = 0.05        # Flooding level for L_ce


# ---------------------------------------------------------------------------
# App config
# ---------------------------------------------------------------------------


@dataclass
class AppConfig:
    method: str
    seed: int
    dataset: DatasetConfig
    outputs: OutputConfig
    train: TrainConfig
    model: Any = None  # method-specific model config


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _as_tuple(value: Any) -> tuple[str, ...]:
    if not isinstance(value, list):
        raise ValueError("image_extensions must be a list of strings")
    ext = tuple(str(x).lower() for x in value)
    if len(ext) == 0:
        raise ValueError("image_extensions cannot be empty")
    return ext


def _build_dataset_cfg(raw: dict[str, Any]) -> DatasetConfig:
    return DatasetConfig(
        dataset_root=Path(str(raw.get("dataset_root", ""))).expanduser(),
        image_extensions=_as_tuple(raw.get("image_extensions", [])),
        create_val_split=bool(raw.get("create_val_split", True)),
        val_ratio=float(raw.get("val_ratio", 0.1)),
        image_size=int(raw.get("image_size", 224)),
    )


def _build_output_cfg(raw: dict[str, Any]) -> OutputConfig:
    return OutputConfig(
        manifest_path=Path(str(raw.get("manifest_path", "artifacts/manifest.jsonl"))),
        diagnostics_path=Path(str(raw.get("diagnostics_path", "artifacts/diagnostics.json"))),
    )


def _build_train_cfg(raw: dict[str, Any], seed: int) -> TrainConfig:
    return TrainConfig(
        batch_size=int(raw.get("batch_size", 32)),
        num_epochs=int(raw.get("num_epochs", 30)),
        lr=float(raw.get("lr", 1e-3)),
        weight_decay=float(raw.get("weight_decay", 1e-4)),
        num_workers=int(raw.get("num_workers", 4)),
        device=str(raw.get("device", "")),
        seed=seed,
        save_dir=Path(str(raw.get("save_dir", "checkpoints"))),
        stream_mode=str(raw.get("stream_mode", "multimodal")),
    )


# ---------------------------------------------------------------------------
# Loader
# ---------------------------------------------------------------------------


def load_config(config_path: str | Path) -> AppConfig:
    cfg_path = Path(config_path)
    if not cfg_path.exists():
        raise FileNotFoundError(f"Config not found: {cfg_path}")

    with cfg_path.open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)

    if not isinstance(raw, dict):
        raise ValueError("Config root must be a mapping")

    method = str(raw.get("method", "caernet")).lower()
    seed = int(raw.get("seed", 42))

    dataset_raw = raw.get("dataset", {})
    outputs_raw = raw.get("outputs", {})
    train_raw = raw.get("train", {})
    model_raw = raw.get("model", {})

    if not isinstance(dataset_raw, dict) or not isinstance(outputs_raw, dict):
        raise ValueError("dataset and outputs config sections must be mappings")
    if not isinstance(train_raw, dict):
        raise ValueError("train config section must be a mapping")

    dataset_cfg = _build_dataset_cfg(dataset_raw)
    output_cfg = _build_output_cfg(outputs_raw)
    train_cfg = _build_train_cfg(train_raw, seed)

    if not 0.0 < dataset_cfg.val_ratio < 0.5:
        raise ValueError("val_ratio must be in (0.0, 0.5)")

    # Method-specific model config
    model_cfg: Any = None
    if method == "caernet":
        model_cfg = CAERNetConfig(
            backbone=str(model_raw.get("backbone", "resnet18")),
            pretrained=bool(model_raw.get("pretrained", True)),
            dropout=float(model_raw.get("dropout", 0.5)),
            face_size=int(model_raw.get("face_size", 96)),
        )
    elif method == "zhou_cross_attention":
        model_cfg = ZhouCrossAttentionConfig(
            backbone=str(model_raw.get("backbone", "resnet18")),
            pretrained=bool(model_raw.get("pretrained", True)),
            dropout=float(model_raw.get("dropout", 0.5)),
            ca_reduction=int(model_raw.get("ca_reduction", 16)),
            er_reduction=int(model_raw.get("er_reduction", 16)),
            aa_hidden_dim=int(model_raw.get("aa_hidden_dim", 256)),
            df_hidden_dim=int(model_raw.get("df_hidden_dim", 512)),
        )
    elif method == "yang_ccim":
        model_cfg = YangCCIMConfig(
            backbone=str(model_raw.get("backbone", "resnet18")),
            pretrained=bool(model_raw.get("pretrained", True)),
            dropout=float(model_raw.get("dropout", 0.5)),
            num_confounders=int(model_raw.get("num_confounders", 1024)),
            confounder_feature_dim=int(model_raw.get("confounder_feature_dim", 512)),
            ccim_strategy=str(model_raw.get("ccim_strategy", "dp_cause")),
        )
    elif method == "glamor_net":
        model_cfg = GLAMORNetConfig(
            backbone=str(model_raw.get("backbone", "resnet18")),
            pretrained=bool(model_raw.get("pretrained", True)),
            dropout=float(model_raw.get("dropout", 0.5)),
            face_size=int(model_raw.get("face_size", 96)),
            gla_hidden_dim=int(model_raw.get("gla_hidden_dim", 128)),
            fusion_hidden_dim=int(model_raw.get("fusion_hidden_dim", 128)),
            classifier_hidden_dim=int(model_raw.get("classifier_hidden_dim", 128)),
        )
    elif method == "cd_ica_net":
        model_cfg = CDICANetConfig(
            backbone=str(model_raw.get("backbone", "resnet18")),
            pretrained=bool(model_raw.get("pretrained", True)),
            dropout=float(model_raw.get("dropout", 0.5)),
            num_iterations=int(model_raw.get("num_iterations", 3)),
            confounder_dim=int(model_raw.get("confounder_dim", 512)),
            num_confounders=int(model_raw.get("num_confounders", 128)),
            ccim_strategy=str(model_raw.get("ccim_strategy", "dp_cause")),
            aa_hidden_dim=int(model_raw.get("aa_hidden_dim", 256)),
            df_hidden_dim=int(model_raw.get("df_hidden_dim", 512)),
            alpha_ica=float(model_raw.get("alpha_ica", 0.5)),
            beta_reg=float(model_raw.get("beta_reg", 0.1)),
            flood_level=float(model_raw.get("flood_level", 0.05)),
        )
    else:
        raise ValueError(f"Unsupported method: {method}")

    # Save dir sub-folder per method
    train_cfg.save_dir = train_cfg.save_dir / method

    return AppConfig(
        method=method,
        seed=seed,
        dataset=dataset_cfg,
        outputs=output_cfg,
        train=train_cfg,
        model=model_cfg,
    )
