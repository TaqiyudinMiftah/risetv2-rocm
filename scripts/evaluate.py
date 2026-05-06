from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import torch
import torch.nn as nn
import wandb
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from config.config import load_config
from datasets.caers_dataset import CAERSTwoStreamDataset
from datasets.transforms import default_transform
from engine.evaluator import evaluate, evaluate_per_class
from utils.device_utils import get_device
from utils.io_utils import write_json


def build_model(num_classes: int, cfg: object) -> nn.Module:
    method = cfg.method
    model_cfg = cfg.model

    if method == "caernet":
        from models.caernet.model import CAERNet, SingleStreamNet

        if cfg.train.stream_mode == "multimodal":
            return CAERNet(
                num_classes=num_classes,
                backbone=model_cfg.backbone,
                pretrained=model_cfg.pretrained,
                dropout=model_cfg.dropout,
            )
        return SingleStreamNet(
            num_classes=num_classes,
            stream=cfg.train.stream_mode,
            backbone=model_cfg.backbone,
            pretrained=model_cfg.pretrained,
            dropout=model_cfg.dropout,
        )

    if method == "zhou_cross_attention":
        from models.zhou_cross_attention.model import ZhouCrossAttentionNet

        return ZhouCrossAttentionNet(
            num_classes=num_classes,
            backbone=model_cfg.backbone,
            pretrained=model_cfg.pretrained,
            dropout=model_cfg.dropout,
            ca_reduction=model_cfg.ca_reduction,
            er_reduction=model_cfg.er_reduction,
            aa_hidden_dim=model_cfg.aa_hidden_dim,
            df_hidden_dim=model_cfg.df_hidden_dim,
        )

    if method == "yang_ccim":
        from models.yang_ccim.model import YangCCIMNet

        return YangCCIMNet(
            num_classes=num_classes,
            backbone=model_cfg.backbone,
            pretrained=model_cfg.pretrained,
            dropout=model_cfg.dropout,
            num_confounders=model_cfg.num_confounders,
            confounder_feature_dim=model_cfg.confounder_feature_dim,
            ccim_strategy=model_cfg.ccim_strategy,
        )

    if method == "glamor_net":
        from models.glamor_net.model import GLAMORNet

        return GLAMORNet(
            num_classes=num_classes,
            backbone=model_cfg.backbone,
            pretrained=model_cfg.pretrained,
            dropout=model_cfg.dropout,
            gla_hidden_dim=model_cfg.gla_hidden_dim,
            fusion_hidden_dim=model_cfg.fusion_hidden_dim,
            classifier_hidden_dim=model_cfg.classifier_hidden_dim,
        )

    raise ValueError(f"Unsupported method: {method}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate emotion recognition model")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--split", type=str, default="test", help="Split to evaluate: test or val")
    parser.add_argument(
        "--wandb-api-key",
        type=str,
        default=os.environ.get("WANDB_API_KEY", ""),
        help="W&B API key (falls back to WANDB_API_KEY env var)",
    )
    parser.add_argument("--wandb-project", type=str, default="caers-emotion-recognition", help="W&B project name")
    parser.add_argument("--wandb-entity", type=str, default="", help="W&B entity/team name")
    parser.add_argument("--wandb-run-name", type=str, default="", help="W&B run name (auto-generated if empty)")
    parser.add_argument("--wandb-offline", action="store_true", help="Run W&B in offline mode")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)

    device = get_device(cfg.train.device)

    # Login to W&B if API key provided
    if args.wandb_api_key and not args.wandb_offline:
        wandb.login(key=args.wandb_api_key)

    # Initialize W&B
    wandb_mode = "offline" if args.wandb_offline else "online"
    run_name = args.wandb_run_name or f"eval-{cfg.method}-{cfg.model.backbone}"
    run = wandb.init(
        project=args.wandb_project,
        entity=args.wandb_entity or None,
        name=run_name,
        mode=wandb_mode,
        job_type="evaluation",
        config={
            "method": cfg.method,
            "eval_split": args.split,
            "checkpoint": args.checkpoint,
            "backbone": cfg.model.backbone,
            "pretrained": cfg.model.pretrained,
            "dropout": cfg.model.dropout,
            "batch_size": cfg.train.batch_size,
            "stream_mode": cfg.train.stream_mode,
            "image_size": cfg.dataset.image_size,
        },
    )

    ds_test = CAERSTwoStreamDataset(
        manifest_path=cfg.outputs.manifest_path,
        dataset_root=cfg.dataset.dataset_root,
        split=args.split,
        image_size=cfg.dataset.image_size,
        transform=default_transform(cfg.dataset.image_size),
    )
    loader_test = DataLoader(
        ds_test,
        batch_size=cfg.train.batch_size,
        shuffle=False,
        num_workers=cfg.train.num_workers,
        pin_memory=True,
    )

    num_classes = len(ds_test.label_to_index)
    model = build_model(num_classes, cfg).to(device)

    # For Yang CCIM: load confounder dictionary
    if cfg.method == "yang_ccim":
        confounder_path = cfg.train.save_dir / "confounder_dict.pt"
        if confounder_path.exists():
            print(f"Loading confounder dictionary from {confounder_path}")
            ckpt_conf = torch.load(confounder_path, map_location="cpu")
            model.set_confounder_dict(ckpt_conf["confounder_dict"], ckpt_conf["confounder_prior"])
        else:
            print("WARNING: Confounder dictionary not found. Building from scratch...")
            from models.yang_ccim.confounder_builder import build_confounder_for_dataset

            conf_dict, conf_prior = build_confounder_for_dataset(
                manifest_path=cfg.outputs.manifest_path,
                dataset_root=cfg.dataset.dataset_root,
                backbone=cfg.model.backbone,
                pretrained=cfg.model.pretrained,
                image_size=cfg.dataset.image_size,
                num_confounders=cfg.model.num_confounders,
                batch_size=cfg.train.batch_size,
                num_workers=cfg.train.num_workers,
                device=device,
                seed=cfg.seed,
            )
            model.set_confounder_dict(conf_dict, conf_prior)

    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    label_to_index = checkpoint.get("label_to_index", ds_test.label_to_index)
    index_to_label = {int(k): v for k, v in checkpoint.get("index_to_label", ds_test.index_to_label).items()}

    criterion = nn.CrossEntropyLoss()
    metrics = evaluate(model, loader_test, criterion, device)
    per_class = evaluate_per_class(model, loader_test, device, index_to_label)

    print(f"Results on '{args.split}' ({cfg.method}):")
    print(f"  Loss: {metrics['loss']:.4f}")
    print(f"  Top-1 Accuracy: {metrics['acc1']:.2f}%")
    print(f"  Top-5 Accuracy: {metrics['acc5']:.2f}%")
    print("  Per-class Accuracy:")
    for label_name, acc in per_class["per_class_acc"].items():
        print(f"    {label_name}: {acc:.2f}%")

    # Log metrics to W&B
    wandb.log({
        f"{args.split}/loss": metrics["loss"],
        f"{args.split}/acc1": metrics["acc1"],
        f"{args.split}/acc5": metrics["acc5"],
        f"{args.split}/overall_acc": per_class["overall_acc"],
    })

    # Log per-class accuracy as a table
    class_data = []
    for label_name, acc in per_class["per_class_acc"].items():
        class_data.append([label_name, acc])

    class_table = wandb.Table(columns=["class", "accuracy"], data=class_data)
    wandb.log({f"{args.split}/per_class_accuracy": class_table})

    # Log per-class accuracy as bar chart
    wandb.log({
        f"{args.split}/per_class_acc_bar": wandb.plot.bar(
            class_table, "class", "accuracy", title=f"Per-Class Accuracy ({args.split})"
        )
    })

    # Log summary stats
    wandb.run.summary[f"{args.split}_loss"] = metrics["loss"]
    wandb.run.summary[f"{args.split}_acc1"] = metrics["acc1"]
    wandb.run.summary[f"{args.split}_acc5"] = metrics["acc5"]
    wandb.run.summary[f"{args.split}_overall_acc"] = per_class["overall_acc"]

    out = {
        "method": cfg.method,
        "split": args.split,
        "metrics": metrics,
        "per_class_acc": per_class["per_class_acc"],
        "overall_acc": per_class["overall_acc"],
    }
    out_path = cfg.train.save_dir / f"eval_{args.split}.json"
    write_json(out_path, out)

    # Log evaluation JSON as artifact
    artifact = wandb.Artifact(
        name=f"{cfg.method}-eval-{wandb.run.id}",
        type="evaluation",
        metadata={
            "method": cfg.method,
            "split": args.split,
            "acc1": metrics["acc1"],
            "overall_acc": per_class["overall_acc"],
        },
    )
    artifact.add_file(str(out_path))
    wandb.log_artifact(artifact)

    print(f"Evaluation saved to {out_path}")

    wandb.finish()


if __name__ == "__main__":
    main()
