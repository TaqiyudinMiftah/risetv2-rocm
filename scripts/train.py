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
from datasets.transforms import default_transform, augmented_transform, caer_net_transforms
from engine.evaluator import evaluate, evaluate_per_class
from engine.trainer import train_one_epoch
from utils.device_utils import get_device, set_seed_all
from utils.io_utils import ensure_parent_dir, write_json
from utils.logger import setup_logger


def set_seed(seed: int) -> None:
    set_seed_all(seed)


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

    if method == "cd_ica_net":
        from models.cd_ica_net.model import CDICANet

        return CDICANet(
            num_classes=num_classes,
            backbone=model_cfg.backbone,
            pretrained=model_cfg.pretrained,
            dropout=model_cfg.dropout,
            num_iterations=model_cfg.num_iterations,
            confounder_dim=model_cfg.confounder_dim,
            num_confounders=model_cfg.num_confounders,
            ccim_strategy=model_cfg.ccim_strategy,
            aa_hidden_dim=model_cfg.aa_hidden_dim,
            df_hidden_dim=model_cfg.df_hidden_dim,
        )

    raise ValueError(f"Unsupported method: {method}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train emotion recognition model")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config")
    parser.add_argument("--resume", type=str, default="", help="Path to checkpoint to resume")
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
    parser.add_argument("--augment", action="store_true", help="Use data augmentation for training")
    parser.add_argument("--eval-after-train", action="store_true", help="Evaluate on test split after training and log to the same W&B run")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    logger = setup_logger(name="caers_train", log_dir=PROJECT_ROOT / "logs")
    logger.info("Starting training script")
    logger.info("Config path: %s", args.config)
    if args.resume:
        logger.info("Resume checkpoint: %s", args.resume)

    try:
        cfg = load_config(args.config)
        set_seed(cfg.seed)

        device = get_device(cfg.train.device)
        logger.info("Using device: %s", device)

        # Login to W&B if API key provided
        if args.wandb_api_key and not args.wandb_offline:
            wandb.login(key=args.wandb_api_key)

        # Initialize W&B
        wandb_mode = "offline" if args.wandb_offline else "online"
        run_name = args.wandb_run_name or f"{cfg.method}-{cfg.model.backbone}"
        run = wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity or None,
            name=run_name,
            mode=wandb_mode,
            config={
                "method": cfg.method,
                "seed": cfg.seed,
                "backbone": cfg.model.backbone,
                "pretrained": cfg.model.pretrained,
                "dropout": cfg.model.dropout,
                "batch_size": cfg.train.batch_size,
                "num_epochs": cfg.train.num_epochs,
                "lr": cfg.train.lr,
                "weight_decay": cfg.train.weight_decay,
                "optimizer": cfg.train.optimizer,
                "momentum": cfg.train.momentum,
                "nesterov": cfg.train.nesterov,
                "scheduler": cfg.train.scheduler,
                "grad_clip_norm": cfg.train.grad_clip_norm,
                "stream_mode": cfg.train.stream_mode,
                "image_size": cfg.dataset.image_size,
                "val_ratio": cfg.dataset.val_ratio,
                **({
                    "face_size": cfg.model.face_size,
                } if cfg.method in ("caernet", "glamor_net") else {}),
                **({
                    "num_iterations": cfg.model.num_iterations,
                    "confounder_dim": cfg.model.confounder_dim,
                    "num_confounders": cfg.model.num_confounders,
                    "ccim_strategy": cfg.model.ccim_strategy,
                    "alpha_ica": cfg.model.alpha_ica,
                    "beta_reg": cfg.model.beta_reg,
                    "flood_level": cfg.model.flood_level,
                } if cfg.method == "cd_ica_net" else {}),
            },
        )

        # Per-method transform setup
        # CAER-Net and GLAMOR-Net use paper-specific face crop (96x96) + context (224x224)
        if cfg.method in ("caernet", "glamor_net"):
            face_size = getattr(cfg.model, "face_size", 96)
            train_face_t, train_ctx_t = caer_net_transforms(
                image_size=cfg.dataset.image_size, face_size=face_size, augment=args.augment
            )
            val_face_t, val_ctx_t = caer_net_transforms(
                image_size=cfg.dataset.image_size, face_size=face_size, augment=False
            )
            ds_train = CAERSTwoStreamDataset(
                manifest_path=cfg.outputs.manifest_path,
                dataset_root=cfg.dataset.dataset_root,
                split="train",
                image_size=cfg.dataset.image_size,
                face_size=face_size,
                face_transform=train_face_t,
                context_transform=train_ctx_t,
            )
            ds_val = CAERSTwoStreamDataset(
                manifest_path=cfg.outputs.manifest_path,
                dataset_root=cfg.dataset.dataset_root,
                split="val",
                image_size=cfg.dataset.image_size,
                face_size=face_size,
                face_transform=val_face_t,
                context_transform=val_ctx_t,
            )
        else:
            train_transform = augmented_transform(cfg.dataset.image_size) if args.augment else default_transform(cfg.dataset.image_size)
            val_transform = default_transform(cfg.dataset.image_size)
            ds_train = CAERSTwoStreamDataset(
                manifest_path=cfg.outputs.manifest_path,
                dataset_root=cfg.dataset.dataset_root,
                split="train",
                image_size=cfg.dataset.image_size,
                transform=train_transform,
            )
            ds_val = CAERSTwoStreamDataset(
                manifest_path=cfg.outputs.manifest_path,
                dataset_root=cfg.dataset.dataset_root,
                split="val",
                image_size=cfg.dataset.image_size,
                transform=val_transform,
            )

        loader_train = DataLoader(
            ds_train,
            batch_size=cfg.train.batch_size,
            shuffle=True,
            num_workers=cfg.train.num_workers,
            pin_memory=True,
        )
        loader_val = DataLoader(
            ds_val,
            batch_size=cfg.train.batch_size,
            shuffle=False,
            num_workers=cfg.train.num_workers,
            pin_memory=True,
        )

        num_classes = len(ds_train.label_to_index)
        model = build_model(num_classes, cfg).to(device)
        logger.info("Model built: method=%s backbone=%s classes=%d", cfg.method, cfg.model.backbone, num_classes)

        # For Yang CCIM: build confounder dictionary if not already set
        if cfg.method == "yang_ccim":
            confounder_path = cfg.train.save_dir / "confounder_dict.pt"
            if confounder_path.exists() and not args.resume:
                logger.info("Loading existing confounder dictionary from %s", confounder_path)
                ckpt = torch.load(confounder_path, map_location="cpu")
                model.set_confounder_dict(ckpt["confounder_dict"], ckpt["confounder_prior"])
            else:
                logger.info("Building confounder dictionary from training data...")
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
                    save_path=confounder_path,
                )
                model.set_confounder_dict(conf_dict, conf_prior)
                logger.info("Confounder dictionary built and saved to %s", confounder_path)

        # For CD-ICA-Net: build confounder dictionary if not already set
        if cfg.method == "cd_ica_net":
            confounder_path = cfg.train.save_dir / "confounder_dict.pt"
            if confounder_path.exists() and not args.resume:
                logger.info("Loading existing confounder dictionary from %s", confounder_path)
                ckpt = torch.load(confounder_path, map_location="cpu")
                model.set_confounder_dict(ckpt["confounder_dict"], ckpt["confounder_prior"])
            else:
                logger.info("Building confounder dictionary from training data...")
                from models.cd_ica_net.confounder_builder import build_confounder_for_dataset

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
                    save_path=confounder_path,
                )
                model.set_confounder_dict(conf_dict, conf_prior)
                logger.info("Confounder dictionary built and saved to %s", confounder_path)

        criterion = nn.CrossEntropyLoss()

        # Optimizer selection
        if cfg.train.optimizer.lower() == "sgd":
            optimizer = torch.optim.SGD(
                model.parameters(),
                lr=cfg.train.lr,
                momentum=cfg.train.momentum,
                nesterov=cfg.train.nesterov,
                weight_decay=cfg.train.weight_decay,
            )
            logger.info("Optimizer: SGD(lr=%.4f, momentum=%.2f, nesterov=%s, wd=%.5f)", cfg.train.lr, cfg.train.momentum, cfg.train.nesterov, cfg.train.weight_decay)
        else:
            optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.train.lr, weight_decay=cfg.train.weight_decay)
            logger.info("Optimizer: AdamW(lr=%.4f, wd=%.5f)", cfg.train.lr, cfg.train.weight_decay)

        # LR scheduler
        scheduler = None
        if cfg.train.scheduler.lower() == "cosine":
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=cfg.train.num_epochs, eta_min=cfg.train.eta_min
            )
            logger.info("Scheduler: CosineAnnealingLR(T_max=%d, eta_min=%.2e)", cfg.train.num_epochs, cfg.train.eta_min)

        # Custom loss function for CD-ICA-Net
        loss_fn = None
        if cfg.method == "cd_ica_net":
            import torch.nn.functional as F

            alpha_ica = cfg.model.alpha_ica
            beta_reg = cfg.model.beta_reg
            flood_level = cfg.model.flood_level

            def _cd_ica_loss_fn(out: dict[str, Any], labels: torch.Tensor) -> torch.Tensor:
                ce = F.cross_entropy(out["logits"], labels)
                # Flooding (Ishida et al. 2020)
                ce = (ce - flood_level).abs() + flood_level

                # Causal intervention loss: KL(P(Y|X) || P(Y|do(X)))
                loss_ica = torch.tensor(0.0, device=ce.device)
                if "causal_logits" in out:
                    p_standard = F.log_softmax(out["logits"], dim=1)
                    p_causal = F.softmax(out["causal_logits"], dim=1)
                    loss_ica = F.kl_div(p_standard, p_causal, reduction="batchmean")

                # Regularization on causal projection weights
                l_reg = torch.tensor(0.0, device=ce.device)
                if hasattr(model, "ccim"):
                    l_reg = model.ccim.w_h.pow(2).sum() + model.ccim.w_g.pow(2).sum()

                return ce + alpha_ica * loss_ica + beta_reg * l_reg

            loss_fn = _cd_ica_loss_fn
            logger.info(
                "CD-ICA-Net custom loss | alpha_ica=%.3f beta_reg=%.3f flood=%.3f",
                alpha_ica, beta_reg, flood_level,
            )

        # Log model architecture
        wandb.watch(model, log="all", log_freq=100)

        start_epoch = 0
        best_val_acc = 0.0
        history: list[dict[str, object]] = []

        if args.resume:
            checkpoint = torch.load(args.resume, map_location=device)
            model.load_state_dict(checkpoint["model_state_dict"])
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            start_epoch = checkpoint.get("epoch", 0) + 1
            best_val_acc = checkpoint.get("best_val_acc", 0.0)
            logger.info("Resumed from epoch %d, best_val_acc=%.2f%%", start_epoch, best_val_acc)
            wandb.run.summary["resumed_from_epoch"] = start_epoch
            wandb.run.summary["resumed_best_val_acc"] = best_val_acc

        ensure_parent_dir(cfg.train.save_dir / "placeholder")

        for epoch in range(start_epoch, cfg.train.num_epochs):
            logger.info("Epoch %d/%d started", epoch + 1, cfg.train.num_epochs)
            train_metrics = train_one_epoch(
                model, loader_train, optimizer, criterion, device,
                loss_fn=loss_fn, grad_clip_norm=cfg.train.grad_clip_norm,
            )
            val_metrics = evaluate(model, loader_val, criterion, device)

            logger.info(
                "Epoch %d/%d finished | train loss=%.4f acc1=%.2f%% | val loss=%.4f acc1=%.2f%%",
                epoch + 1,
                cfg.train.num_epochs,
                train_metrics["loss"],
                train_metrics["acc1"],
                val_metrics["loss"],
                val_metrics["acc1"],
            )

            # Step scheduler
            current_lr = optimizer.param_groups[0]["lr"]
            if scheduler is not None:
                scheduler.step()

            # Log metrics to W&B
            log_dict = {
                "epoch": epoch + 1,
                "train/loss": train_metrics["loss"],
                "train/acc1": train_metrics["acc1"],
                "train/acc5": train_metrics["acc5"],
                "val/loss": val_metrics["loss"],
                "val/acc1": val_metrics["acc1"],
                "val/acc5": val_metrics["acc5"],
                "train/lr": current_lr,
            }
            wandb.log(log_dict)

            history.append({
                "epoch": epoch + 1,
                "train": train_metrics,
                "val": val_metrics,
            })

            if val_metrics["acc1"] > best_val_acc:
                best_val_acc = val_metrics["acc1"]
                ckpt_path = cfg.train.save_dir / "best_model.pt"
                torch.save({
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "best_val_acc": best_val_acc,
                    "label_to_index": ds_train.label_to_index,
                    "index_to_label": ds_train.index_to_label,
                }, ckpt_path)
                logger.info("Saved best model -> %s (val_acc1=%.2f%%)", ckpt_path, best_val_acc)

                # Log best model as W&B artifact
                artifact = wandb.Artifact(
                    name=f"{cfg.method}-model-{wandb.run.id}",
                    type="model",
                    metadata={
                        "epoch": epoch + 1,
                        "val_acc1": best_val_acc,
                        "method": cfg.method,
                    },
                )
                artifact.add_file(str(ckpt_path))
                wandb.log_artifact(artifact, aliases=["best"])

        history_path = cfg.train.save_dir / "history.json"
        write_json(history_path, {"history": history, "best_val_acc": best_val_acc})
        logger.info("Training complete. History saved to %s", history_path)

        # Log final history and summary
        wandb.run.summary["best_val_acc"] = best_val_acc
        wandb.run.summary["total_epochs"] = cfg.train.num_epochs
        wandb.save(str(history_path))

        # -------------------------------------------------------------------
        # Evaluate on test set (same W&B run)
        # -------------------------------------------------------------------
        if args.eval_after_train:
            logger.info("Evaluating on test split (same W&B run)...")
            from engine.evaluator import evaluate_per_class

            if cfg.method in ("caernet", "glamor_net"):
                ds_test = CAERSTwoStreamDataset(
                    manifest_path=cfg.outputs.manifest_path,
                    dataset_root=cfg.dataset.dataset_root,
                    split="test",
                    image_size=cfg.dataset.image_size,
                    face_size=face_size,
                    face_transform=val_face_t,
                    context_transform=val_ctx_t,
                )
            else:
                ds_test = CAERSTwoStreamDataset(
                    manifest_path=cfg.outputs.manifest_path,
                    dataset_root=cfg.dataset.dataset_root,
                    split="test",
                    image_size=cfg.dataset.image_size,
                    transform=val_transform,
                )
            loader_test = DataLoader(
                ds_test,
                batch_size=cfg.train.batch_size,
                shuffle=False,
                num_workers=cfg.train.num_workers,
                pin_memory=True,
            )

            test_metrics = evaluate(model, loader_test, criterion, device)
            test_per_class = evaluate_per_class(model, loader_test, device, ds_test.index_to_label)

            logger.info("Test results | loss=%.4f acc1=%.2f%% acc5=%.2f%%", test_metrics["loss"], test_metrics["acc1"], test_metrics["acc5"])
            for label_name, acc in test_per_class["per_class_acc"].items():
                logger.info("  %s: %.2f%%", label_name, acc)

            # Log test metrics to the SAME W&B run
            wandb.log({
                "test/loss": test_metrics["loss"],
                "test/acc1": test_metrics["acc1"],
                "test/acc5": test_metrics["acc5"],
                "test/overall_acc": test_per_class["overall_acc"],
            })

            # Per-class accuracy table
            class_data = [[label_name, acc] for label_name, acc in test_per_class["per_class_acc"].items()]
            class_table = wandb.Table(columns=["class", "accuracy"], data=class_data)
            wandb.log({"test/per_class_accuracy": class_table})

            # Update run summary
            wandb.run.summary["test_acc1"] = test_metrics["acc1"]
            wandb.run.summary["test_acc5"] = test_metrics["acc5"]
            wandb.run.summary["test_overall_acc"] = test_per_class["overall_acc"]

            # Save test eval JSON
            out = {
                "method": cfg.method,
                "split": "test",
                "metrics": test_metrics,
                "per_class_acc": test_per_class["per_class_acc"],
                "overall_acc": test_per_class["overall_acc"],
            }
            test_out_path = cfg.train.save_dir / "eval_test.json"
            write_json(test_out_path, out)
            wandb.save(str(test_out_path))
            logger.info("Test evaluation saved to %s", test_out_path)

        wandb.finish()
    except Exception:
        logger.exception("Training crashed with an exception")
        raise


if __name__ == "__main__":
    main()
