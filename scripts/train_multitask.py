"""Train PPG-only blood pressure models with physiological auxiliary tasks."""

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
from torch.utils.data import DataLoader
from tqdm import tqdm

from ppg_bp.data.pulsedb import PulseDBAuxiliaryDataset
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


def age_group_targets(
    ages: torch.Tensor,
    boundaries: torch.Tensor,
) -> torch.Tensor:
    return torch.bucketize(ages, boundaries)


def bp_class_targets(labels: torch.Tensor) -> torch.Tensor:
    """Low, normal/intermediate, or high BP pattern."""

    low = (labels[:, 0] < 90.0) | (labels[:, 1] < 60.0)
    high = (labels[:, 0] >= 140.0) | (labels[:, 1] >= 90.0)
    classes = torch.ones(len(labels), dtype=torch.long, device=labels.device)
    classes[low] = 0
    classes[high] = 2
    return classes


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    target_mean: torch.Tensor,
    target_std: torch.Tensor,
    tasks: tuple[str, ...],
    heart_rate_mean: torch.Tensor,
    heart_rate_std: torch.Tensor,
    age_boundaries: torch.Tensor,
) -> tuple[dict[str, float | bool | str], dict[str, float]]:
    model.eval()
    predictions: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    heart_rate_absolute_error = 0.0
    heart_rate_count = 0
    age_correct = 0
    age_count = 0
    bp_class_correct = 0
    bp_class_count = 0

    for signals, labels, auxiliary in loader:
        labels_device = labels.to(device, non_blocking=True)
        outputs = model(signals.to(device, non_blocking=True))
        normalized = outputs["bp"]
        predictions.append((normalized * target_std + target_mean).cpu().numpy())
        targets.append(labels.numpy())

        if "heart_rate" in tasks:
            valid = auxiliary["heart_rate_valid"].to(device)
            if bool(valid.any()):
                raw_prediction = (
                    outputs["heart_rate"].flatten() * heart_rate_std
                    + heart_rate_mean
                )
                raw_target = auxiliary["heart_rate"].to(device)
                heart_rate_absolute_error += float(
                    torch.abs(raw_prediction[valid] - raw_target[valid]).sum()
                )
                heart_rate_count += int(valid.sum())
        if "age_group" in tasks:
            ages = auxiliary["age"].to(device)
            valid = torch.isfinite(ages)
            expected = age_group_targets(ages[valid], age_boundaries)
            predicted = outputs["age_group"][valid].argmax(dim=1)
            age_correct += int((predicted == expected).sum())
            age_count += int(valid.sum())
        if "bp_class" in tasks:
            expected = bp_class_targets(labels_device)
            predicted = outputs["bp_class"].argmax(dim=1)
            bp_class_correct += int((predicted == expected).sum())
            bp_class_count += len(labels)

    auxiliary_metrics: dict[str, float] = {}
    if heart_rate_count:
        auxiliary_metrics["heart_rate_mae_bpm"] = (
            heart_rate_absolute_error / heart_rate_count
        )
    if age_count:
        auxiliary_metrics["age_group_accuracy"] = age_correct / age_count
    if bp_class_count:
        auxiliary_metrics["bp_class_accuracy"] = bp_class_correct / bp_class_count
    return (
        regression_metrics(np.concatenate(predictions), np.concatenate(targets)),
        auxiliary_metrics,
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
    tasks = tuple(config["model"].get("tasks", ()))
    datasets = {
        name: PulseDBAuxiliaryDataset(
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
            auxiliary_tasks=tasks,
        )
        for name in ("train", "val", "test")
    }
    if any(len(dataset) == 0 for dataset in datasets.values()):
        raise ValueError("Every subject-wise split must contain at least one sample")

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

    train_indices = datasets["train"].indices
    train_targets = np.asarray(
        datasets["train"].labels[train_indices],
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

    heart_rate_mean = torch.tensor(0.0, device=device)
    heart_rate_std = torch.tensor(1.0, device=device)
    if "heart_rate" in tasks:
        assert datasets["train"].heart_rate is not None
        assert datasets["train"].heart_rate_valid is not None
        heart_rate_values = np.asarray(
            datasets["train"].heart_rate[train_indices],
            dtype=np.float32,
        )
        valid = np.asarray(
            datasets["train"].heart_rate_valid[train_indices],
            dtype=bool,
        )
        heart_rate_mean = torch.tensor(
            float(heart_rate_values[valid].mean()),
            device=device,
        )
        heart_rate_std = torch.tensor(
            max(float(heart_rate_values[valid].std()), 1.0),
            device=device,
        )

    age_boundaries = torch.tensor(
        config["training"].get("age_group_boundaries", [40.0, 60.0, 80.0]),
        dtype=torch.float32,
        device=device,
    )
    model = build_model(config["model"]).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config["training"]["learning_rate"]),
        weight_decay=float(config["training"]["weight_decay"]),
    )
    bp_criterion, loss_name, loss_metadata = build_regression_loss(
        config["training"],
        train_targets,
    )
    bp_criterion = bp_criterion.to(device)
    auxiliary_weights = {
        str(key): float(value)
        for key, value in config["training"]
        .get("auxiliary_loss_weights", {})
        .items()
    }
    for task in tasks:
        if task not in auxiliary_weights:
            raise KeyError(f"Missing auxiliary loss weight for {task}")

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
    primary_metric = str(config["evaluation"].get("primary_metric", "mean_mae"))
    early_stopping = config["training"].get("early_stopping", {})
    patience = int(early_stopping.get("patience", 0))
    min_delta = float(early_stopping.get("min_delta", 0.0))

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
        running_total_loss = 0.0
        running_bp_loss = 0.0
        running_auxiliary = {task: 0.0 for task in tasks}
        optimizer.zero_grad(set_to_none=True)
        train_loader = loaders["train"]

        for step, (signals, labels, auxiliary) in enumerate(
            tqdm(train_loader, desc=f"Epoch {epoch}", mininterval=5),
            start=1,
        ):
            signals = signals.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            with torch.autocast(device_type=device.type, enabled=use_amp):
                outputs = model(signals)
                normalized_labels = (labels - target_mean) / target_std
                bp_loss = bp_criterion(
                    outputs["bp"],
                    normalized_labels,
                    labels,
                )
                total_loss = bp_loss
                task_losses: dict[str, torch.Tensor] = {}
                if "heart_rate" in tasks:
                    valid = auxiliary["heart_rate_valid"].to(device)
                    if bool(valid.any()):
                        raw_hr = auxiliary["heart_rate"].to(device)
                        normalized_hr = (
                            raw_hr[valid] - heart_rate_mean
                        ) / heart_rate_std
                        task_losses["heart_rate"] = nn.functional.mse_loss(
                            outputs["heart_rate"].flatten()[valid],
                            normalized_hr,
                        )
                if "age_group" in tasks:
                    ages = auxiliary["age"].to(device)
                    valid = torch.isfinite(ages)
                    task_losses["age_group"] = nn.functional.cross_entropy(
                        outputs["age_group"][valid],
                        age_group_targets(ages[valid], age_boundaries),
                    )
                if "bp_class" in tasks:
                    task_losses["bp_class"] = nn.functional.cross_entropy(
                        outputs["bp_class"],
                        bp_class_targets(labels),
                    )
                for task, task_loss in task_losses.items():
                    total_loss = (
                        total_loss + auxiliary_weights[task] * task_loss
                    )
                scaled_loss = total_loss / accumulation_steps

            scaler.scale(scaled_loss).backward()
            if step % accumulation_steps == 0 or step == len(train_loader):
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
            batch_samples = len(signals)
            running_total_loss += float(total_loss.detach()) * batch_samples
            running_bp_loss += float(bp_loss.detach()) * batch_samples
            for task, task_loss in task_losses.items():
                running_auxiliary[task] += (
                    float(task_loss.detach()) * batch_samples
                )

        validation, auxiliary_validation = evaluate(
            model,
            loaders["val"],
            device,
            target_mean,
            target_std,
            tasks,
            heart_rate_mean,
            heart_rate_std,
            age_boundaries,
        )
        row: dict[str, object] = {
            "epoch": epoch,
            "train_total_loss": running_total_loss / len(datasets["train"]),
            "train_bp_loss": running_bp_loss / len(datasets["train"]),
            "learning_rate": float(optimizer.param_groups[0]["lr"]),
            **{
                f"train_{task}_loss": value / len(datasets["train"])
                for task, value in running_auxiliary.items()
            },
            **validation,
            **{f"val_{key}": value for key, value in auxiliary_validation.items()},
        }
        history.append(row)
        print(json.dumps(row))
        if primary_metric not in validation:
            raise KeyError(f"Unknown primary metric: {primary_metric}")
        improved = float(validation[primary_metric]) < best_score - min_delta
        if improved:
            best_score = float(validation[primary_metric])
            epochs_without_improvement = 0
            torch.save(
                {
                    "model": model.state_dict(),
                    "config": config,
                    "epoch": epoch,
                    "target_mean": target_mean.cpu(),
                    "target_std": target_std.cpu(),
                    "heart_rate_mean": heart_rate_mean.cpu(),
                    "heart_rate_std": heart_rate_std.cpu(),
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
                "heart_rate_mean": heart_rate_mean.cpu(),
                "heart_rate_std": heart_rate_std.cpu(),
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
                f"{primary_metric} did not improve for {patience} epochs"
            )
            break

    checkpoint = torch.load(
        args.output / "best.pt",
        map_location=device,
        weights_only=False,
    )
    model.load_state_dict(checkpoint["model"])
    test_metrics, auxiliary_test = evaluate(
        model,
        loaders["test"],
        device,
        target_mean,
        target_std,
        tasks,
        heart_rate_mean,
        heart_rate_std,
        age_boundaries,
    )
    result = {
        "experiment": config.get("experiment", {}),
        "device": str(device),
        "tasks": list(tasks),
        "best_epoch": int(checkpoint["epoch"]),
        "completed_epochs": len(history),
        "stopped_early": len(history) < int(config["training"]["epochs"]),
        "loss": loss_name,
        "loss_metadata": loss_metadata,
        "auxiliary_loss_weights": auxiliary_weights,
        "train_target_mean": train_target_mean.cpu().tolist(),
        "train_target_std": train_target_std.cpu().tolist(),
        "heart_rate_mean": float(heart_rate_mean),
        "heart_rate_std": float(heart_rate_std),
        "test": test_metrics,
        "auxiliary_test": auxiliary_test,
        "history": history,
    }
    (args.output / "metrics.json").write_text(
        json.dumps(result, indent=2),
        encoding="utf-8",
    )
    print(json.dumps({"test": test_metrics, "auxiliary_test": auxiliary_test}, indent=2))


if __name__ == "__main__":
    main()
