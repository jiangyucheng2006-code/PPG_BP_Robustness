"""Train and evaluate PPG BP robustness experiments."""

from __future__ import annotations

import argparse
import json
import os
import random
from pathlib import Path

import numpy as np
import torch
import yaml
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from ppg_bp.data import PulseDBMemmapDataset, build_augmenter
from ppg_bp.models import build_model
from ppg_bp.training.losses import build_regression_loss
from ppg_bp.training.metrics import regression_metrics


def load_config(path: Path) -> dict:
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    config["data"]["root"] = os.path.expandvars(config["data"]["root"])
    if "$" in config["data"]["root"]:
        raise EnvironmentError("Set PPG_BP_DATA_ROOT before running training")
    return config


def choose_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(requested)


def bp_output(model: nn.Module, inputs: torch.Tensor) -> torch.Tensor:
    outputs = model(inputs)
    if isinstance(outputs, dict):
        return outputs["bp"]
    return outputs


@torch.no_grad()
def collect_predictions(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    target_mean: torch.Tensor,
    target_std: torch.Tensor,
    *,
    augmenter=None,
    transform: str | None = None,
    severity: float | None = None,
    seed: int = 2026,
) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    predictions: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    for batch_index, (signals, labels) in enumerate(loader):
        signals = signals.to(device, non_blocking=True)
        if augmenter is not None and transform is not None:
            signals, _, _ = augmenter(
                signals,
                forced_transform=transform,
                forced_severity=severity,
                seed=seed + batch_index,
            )
        normalized = bp_output(model, signals)
        predictions.append(
            (normalized * target_std + target_mean).cpu().numpy()
        )
        targets.append(labels.numpy())
    return np.concatenate(predictions), np.concatenate(targets)


@torch.no_grad()
def evaluate_pair(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    target_mean: torch.Tensor,
    target_std: torch.Tensor,
    augmenter,
    *,
    transform: str,
    severity: float,
    seed: int = 2026,
    clean_predictions: np.ndarray | None = None,
    targets: np.ndarray | None = None,
) -> tuple[dict[str, float | bool | str], float]:
    if clean_predictions is None or targets is None:
        clean_predictions, targets = collect_predictions(
            model,
            loader,
            device,
            target_mean,
            target_std,
        )
    corrupted_predictions, corrupted_targets = collect_predictions(
        model,
        loader,
        device,
        target_mean,
        target_std,
        augmenter=augmenter,
        transform=transform,
        severity=severity,
        seed=seed,
    )
    if not np.array_equal(targets, corrupted_targets):
        raise RuntimeError("Clean and corrupted evaluation targets are misaligned")
    shift = float(np.abs(corrupted_predictions - clean_predictions).mean())
    return regression_metrics(corrupted_predictions, targets), shift


def load_pretrained_backbone(
    model: nn.Module,
    checkpoint_path: str | None,
    device: torch.device,
) -> None:
    if not checkpoint_path:
        return
    expanded = os.path.expandvars(checkpoint_path)
    if "$" in expanded:
        raise EnvironmentError(f"Unresolved checkpoint path: {checkpoint_path}")
    checkpoint = torch.load(
        Path(expanded),
        map_location=device,
        weights_only=False,
    )
    backbone = getattr(model, "backbone", None)
    if backbone is None:
        raise TypeError("Configured model does not expose a backbone")
    state = checkpoint.get("backbone", checkpoint.get("model", checkpoint))
    missing, unexpected = backbone.load_state_dict(state, strict=False)
    print(
        json.dumps(
            {
                "pretrained_checkpoint": expanded,
                "missing_backbone_keys": missing,
                "unexpected_backbone_keys": unexpected,
            }
        )
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    config = load_config(args.config)
    if args.seed is not None:
        config["seed"] = args.seed

    seed = int(config["seed"])
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    device = choose_device(config["training"]["device"])

    root = Path(config["data"]["root"])
    datasets = {
        name: PulseDBMemmapDataset(
            root,
            name,
            config["data"]["normalization"],
            config["data"].get("label_filter"),
            split_filename=str(config["data"].get("split_filename", "split.npy")),
            input_representation=str(
                config["data"].get("input_representation", "ppg")
            ),
            derivative_normalization=str(
                config["data"].get(
                    "derivative_normalization",
                    "per_segment_zscore",
                )
            ),
        )
        for name in ("train", "val", "test")
    }
    batch_size = int(config["training"]["batch_size"])
    workers = int(config["training"]["num_workers"])
    evaluation_workers = int(config["training"].get("evaluation_num_workers", 0))
    loaders = {
        name: DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=name == "train",
            drop_last=name == "train",
            num_workers=workers if name == "train" else evaluation_workers,
            pin_memory=device.type == "cuda",
            persistent_workers=name == "train" and workers > 0,
        )
        for name, dataset in datasets.items()
    }

    train_targets = np.asarray(
        datasets["train"].labels[datasets["train"].indices],
        dtype=np.float32,
    )
    train_target_mean = torch.from_numpy(train_targets.mean(axis=0)).to(device)
    train_target_std = torch.from_numpy(
        train_targets.std(axis=0).clip(min=1.0)
    ).to(device)
    if bool(config["training"].get("target_standardization", True)):
        target_mean = train_target_mean
        target_std = train_target_std
    else:
        target_mean = torch.zeros(2, dtype=torch.float32, device=device)
        target_std = torch.ones(2, dtype=torch.float32, device=device)

    robustness = config["robustness"]
    mode = str(robustness["mode"])
    augmenter = build_augmenter(
        robustness["augmentation"],
        sampling_rate=float(config["data"]["sampling_rate"]),
    )
    evaluation_augmenter = build_augmenter(
        robustness.get(
            "evaluation_augmentation",
            robustness["augmentation"],
        ),
        sampling_rate=float(config["data"]["sampling_rate"]),
    )
    model = build_model(config["model"]).to(device)
    if getattr(model, "artifact_classes", 0) not in (0, augmenter.class_count):
        raise ValueError(
            "artifact_classes must be zero or match augmentation classes"
        )
    load_pretrained_backbone(
        model,
        config["training"].get("pretrained_checkpoint"),
        device,
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config["training"]["learning_rate"]),
        weight_decay=float(config["training"]["weight_decay"]),
    )
    criterion, loss_name, loss_metadata = build_regression_loss(
        config["training"],
        train_targets,
    )
    criterion = criterion.to(device)
    scheduler_config = config["training"].get("scheduler", {"name": "constant"})
    scheduler_name = str(scheduler_config.get("name", "constant")).lower()
    if scheduler_name == "constant":
        scheduler = None
    elif scheduler_name == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=int(config["training"]["epochs"]),
            eta_min=float(scheduler_config.get("min_learning_rate", 1e-5)),
        )
    else:
        raise ValueError(f"Unsupported scheduler: {scheduler_name}")

    use_amp = bool(config["training"]["amp"]) and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    accumulation_steps = int(
        config["training"].get("gradient_accumulation_steps", 1)
    )
    early_stopping = config["training"].get("early_stopping", {})
    patience = int(early_stopping.get("patience", 0))
    min_delta = float(early_stopping.get("min_delta", 0.0))
    clean_selection_weight = float(
        robustness.get("clean_selection_weight", 0.5)
    )
    robust_selection_weight = 1.0 - clean_selection_weight
    validation_transform = str(
        robustness.get("validation_transform", "motion_artifact")
    )
    validation_severity = float(
        robustness.get("validation_severity", 0.7)
    )
    augmented_supervision_weight = float(
        robustness.get("augmented_supervision_weight", 0.5)
    )
    consistency_weight = float(robustness.get("consistency_weight", 0.1))
    artifact_type_weight = float(
        robustness.get("artifact_type_weight", 0.1)
    )
    artifact_severity_weight = float(
        robustness.get("artifact_severity_weight", 0.05)
    )

    args.output.mkdir(parents=True, exist_ok=True)
    best_score = float("inf")
    epochs_without_improvement = 0
    history: list[dict[str, object]] = []
    start_epoch = 1
    last_checkpoint = args.output / "last.pt"
    if args.resume and last_checkpoint.exists():
        checkpoint = torch.load(
            last_checkpoint,
            map_location=device,
            weights_only=False,
        )
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        scaler.load_state_dict(checkpoint["scaler"])
        if scheduler is not None and checkpoint.get("scheduler") is not None:
            scheduler.load_state_dict(checkpoint["scheduler"])
        best_score = float(checkpoint["best_score"])
        epochs_without_improvement = int(
            checkpoint.get("epochs_without_improvement", 0)
        )
        history = checkpoint["history"]
        start_epoch = int(checkpoint["epoch"]) + 1
        print(f"Resuming from epoch {start_epoch}")

    for epoch in range(start_epoch, int(config["training"]["epochs"]) + 1):
        model.train()
        running = {
            "total": 0.0,
            "clean_bp": 0.0,
            "augmented_bp": 0.0,
            "consistency": 0.0,
            "artifact_type": 0.0,
            "artifact_severity": 0.0,
        }
        optimizer.zero_grad(set_to_none=True)
        train_loader = loaders["train"]
        for step, (signals, labels) in enumerate(
            tqdm(train_loader, desc=f"Epoch {epoch}", mininterval=5),
            start=1,
        ):
            signals = signals.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            augmented, artifact_labels, artifact_severity = augmenter(signals)
            with torch.autocast(device_type=device.type, enabled=use_amp):
                normalized_labels = (labels - target_mean) / target_std
                losses: dict[str, torch.Tensor] = {}
                if mode == "supervised_augmentation":
                    augmented_outputs = model(augmented)
                    losses["augmented_bp"] = criterion(
                        augmented_outputs["bp"],
                        normalized_labels,
                        labels,
                    )
                    total_loss = losses["augmented_bp"]
                elif mode in {"consistency", "artifact_consistency"}:
                    clean_outputs = model(signals)
                    augmented_outputs = model(augmented)
                    losses["clean_bp"] = criterion(
                        clean_outputs["bp"],
                        normalized_labels,
                        labels,
                    )
                    losses["augmented_bp"] = criterion(
                        augmented_outputs["bp"],
                        normalized_labels,
                        labels,
                    )
                    normalized_shift = (
                        augmented_outputs["bp"]
                        - clean_outputs["bp"].detach()
                    ) / train_target_std
                    losses["consistency"] = F.smooth_l1_loss(
                        normalized_shift,
                        torch.zeros_like(normalized_shift),
                        beta=0.5,
                    )
                    total_loss = (
                        losses["clean_bp"]
                        + augmented_supervision_weight
                        * losses["augmented_bp"]
                    ) / (1.0 + augmented_supervision_weight)
                    total_loss = (
                        total_loss
                        + consistency_weight * losses["consistency"]
                    )
                    if mode == "artifact_consistency":
                        if "artifact_type" not in augmented_outputs:
                            raise TypeError(
                                "artifact_consistency requires artifact heads"
                            )
                        losses["artifact_type"] = F.cross_entropy(
                            augmented_outputs["artifact_type"],
                            artifact_labels,
                        )
                        losses["artifact_severity"] = F.mse_loss(
                            augmented_outputs["artifact_severity"],
                            artifact_severity,
                        )
                        total_loss = (
                            total_loss
                            + artifact_type_weight
                            * losses["artifact_type"]
                            + artifact_severity_weight
                            * losses["artifact_severity"]
                        )
                else:
                    raise ValueError(f"Unsupported robustness mode: {mode}")
                scaled_loss = total_loss / accumulation_steps

            scaler.scale(scaled_loss).backward()
            if step % accumulation_steps == 0 or step == len(train_loader):
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
            batch_samples = len(signals)
            running["total"] += float(total_loss.detach()) * batch_samples
            for name, loss in losses.items():
                running[name] += float(loss.detach()) * batch_samples

        clean_predictions, validation_targets = collect_predictions(
            model,
            loaders["val"],
            device,
            target_mean,
            target_std,
        )
        clean_validation = regression_metrics(
            clean_predictions,
            validation_targets,
        )
        robust_validation, validation_shift = evaluate_pair(
            model,
            loaders["val"],
            device,
            target_mean,
            target_std,
            augmenter,
            transform=validation_transform,
            severity=validation_severity,
            clean_predictions=clean_predictions,
            targets=validation_targets,
        )
        selection_score = (
            clean_selection_weight * float(clean_validation["mean_mae"])
            + robust_selection_weight * float(robust_validation["mean_mae"])
        )
        row: dict[str, object] = {
            "epoch": epoch,
            "train_total_loss": running["total"] / len(datasets["train"]),
            **{
                f"train_{name}_loss": value / len(datasets["train"])
                for name, value in running.items()
                if name != "total"
            },
            "learning_rate": float(optimizer.param_groups[0]["lr"]),
            "mean_mae": clean_validation["mean_mae"],
            "clean_sbp_mae": clean_validation["sbp_mae"],
            "clean_dbp_mae": clean_validation["dbp_mae"],
            "robust_mean_mae": robust_validation["mean_mae"],
            "robust_sbp_mae": robust_validation["sbp_mae"],
            "robust_dbp_mae": robust_validation["dbp_mae"],
            "prediction_shift_mae": validation_shift,
            "selection_score": selection_score,
        }
        history.append(row)
        print(json.dumps(row))
        improved = selection_score < best_score - min_delta
        if improved:
            best_score = selection_score
            epochs_without_improvement = 0
            torch.save(
                {
                    "model": model.state_dict(),
                    "config": config,
                    "epoch": epoch,
                    "target_mean": target_mean.cpu(),
                    "target_std": target_std.cpu(),
                },
                args.output / "best.pt",
            )
        else:
            epochs_without_improvement += 1
        if scheduler is not None:
            scheduler.step()
        torch.save(
            {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scaler": scaler.state_dict(),
                "scheduler": scheduler.state_dict() if scheduler is not None else None,
                "config": config,
                "epoch": epoch,
                "best_score": best_score,
                "epochs_without_improvement": epochs_without_improvement,
                "history": history,
                "target_mean": target_mean.cpu(),
                "target_std": target_std.cpu(),
            },
            last_checkpoint,
        )
        (args.output / "history.json").write_text(
            json.dumps(history, indent=2),
            encoding="utf-8",
        )
        if patience > 0 and epochs_without_improvement >= patience:
            print(
                f"Early stopping at epoch {epoch}; "
                f"selection score did not improve for {patience} epochs"
            )
            break

    checkpoint = torch.load(
        args.output / "best.pt",
        map_location=device,
        weights_only=False,
    )
    model.load_state_dict(checkpoint["model"])
    clean_predictions, test_targets = collect_predictions(
        model,
        loaders["test"],
        device,
        target_mean,
        target_std,
    )
    clean_test = regression_metrics(clean_predictions, test_targets)
    suite_config = robustness.get(
        "evaluation_suite",
        {
            "transforms": [
                "motion_artifact",
                "contact_compression",
                "baseline_drift",
                "gaussian_noise",
                "clipping",
                "sensor_gain_offset",
            ],
            "severities": [0.35, 0.7, 1.0],
        },
    )
    robustness_test: dict[str, object] = {}
    for transform in suite_config["transforms"]:
        robustness_test[transform] = {}
        for severity in suite_config["severities"]:
            condition_metrics, prediction_shift = evaluate_pair(
                model,
                loaders["test"],
                device,
                target_mean,
                target_std,
                evaluation_augmenter,
                transform=str(transform),
                severity=float(severity),
                clean_predictions=clean_predictions,
                targets=test_targets,
            )
            robustness_test[transform][str(severity)] = {
                "metrics": condition_metrics,
                "prediction_shift_mae": prediction_shift,
            }

    result = {
        "experiment": config.get("experiment", {}),
        "device": str(device),
        "mode": mode,
        "best_epoch": int(checkpoint["epoch"]),
        "completed_epochs": len(history),
        "stopped_early": len(history) < int(config["training"]["epochs"]),
        "loss": loss_name,
        "loss_metadata": loss_metadata,
        "clean_test": clean_test,
        "robustness_test": robustness_test,
        "history": history,
    }
    (args.output / "metrics.json").write_text(
        json.dumps(result, indent=2),
        encoding="utf-8",
    )
    print(json.dumps({"clean_test": clean_test}, indent=2))


if __name__ == "__main__":
    main()
