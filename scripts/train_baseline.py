"""Train the PPG-only XResNet baseline on prepared PulseDB arrays."""

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

from ppg_bp.data.pulsedb import PulseDBMemmapDataset
from ppg_bp.models import (
    qumphy_multiscale_xresnet1d50,
    qumphy_xresnet1d50,
    qumphy_xresnet1d101,
    xresnet1d50,
    xresnet1d101,
)
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


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    target_mean: torch.Tensor,
    target_std: torch.Tensor,
) -> dict[str, float]:
    model.eval()
    predictions: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    for signals, labels in loader:
        normalized = model(signals.to(device))
        predictions.append((normalized * target_std + target_mean).cpu().numpy())
        targets.append(labels.numpy())
    return regression_metrics(np.concatenate(predictions), np.concatenate(targets))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("outputs/baseline"))
    parser.add_argument("--resume", action="store_true", help="Resume from output/last.pt when available")
    args = parser.parse_args()
    config = load_config(args.config)

    seed = int(config["seed"])
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    device = choose_device(config["training"]["device"])
    root = Path(config["data"]["root"])
    normalization = config["data"]["normalization"]
    label_filter = config["data"].get("label_filter")
    split_filename = str(config["data"].get("split_filename", "split.npy"))
    datasets = {
        name: PulseDBMemmapDataset(
            root,
            name,
            normalization,
            label_filter,
            split_filename=split_filename,
        )
        for name in ("train", "val", "test")
    }
    if any(len(dataset) == 0 for dataset in datasets.values()):
        raise ValueError("Every subject-wise split must contain at least one sample")

    batch_size = int(config["training"]["batch_size"])
    workers = int(config["training"]["num_workers"])
    evaluation_workers = int(config["training"].get("evaluation_num_workers", 0))
    loaders = {}
    for name, dataset in datasets.items():
        loader_workers = workers if name == "train" else evaluation_workers
        loaders[name] = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=name == "train",
            drop_last=name == "train",
            num_workers=loader_workers,
            pin_memory=device.type == "cuda",
            persistent_workers=name == "train" and loader_workers > 0,
        )
    train_targets = np.asarray(datasets["train"].labels[datasets["train"].indices], dtype=np.float32)
    train_target_mean = torch.from_numpy(train_targets.mean(axis=0)).to(device)
    train_target_std = torch.from_numpy(train_targets.std(axis=0).clip(min=1.0)).to(device)
    standardize_targets = bool(config["training"].get("target_standardization", True))
    if standardize_targets:
        target_mean = train_target_mean
        target_std = train_target_std
    else:
        target_mean = torch.zeros(2, dtype=torch.float32, device=device)
        target_std = torch.ones(2, dtype=torch.float32, device=device)

    depth = int(config["model"]["depth"])
    model_name = str(config["model"]["name"])
    factories = {
        "xresnet1d": {50: xresnet1d50, 101: xresnet1d101},
        "qumphy_xresnet1d": {50: qumphy_xresnet1d50, 101: qumphy_xresnet1d101},
        "qumphy_multiscale_xresnet1d": {50: qumphy_multiscale_xresnet1d50},
    }
    if model_name not in factories:
        raise ValueError(f"Unsupported model name: {model_name}")
    factory = factories[model_name].get(depth)
    if factory is None:
        raise ValueError("Supported XResNet depths are 50 and 101")
    model = factory(
        input_channels=int(config["model"].get("input_channels", 1)),
        outputs=int(config["model"].get("outputs", 2)),
        dropout=float(config["model"].get("dropout", 0.5 if model_name == "qumphy_xresnet1d" else 0.2)),
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config["training"]["learning_rate"]),
        weight_decay=float(config["training"]["weight_decay"]),
    )
    loss_config = config["training"].get("loss", {"name": "mse"})
    if isinstance(loss_config, str):
        loss_config = {"name": loss_config}
    loss_name = str(loss_config.get("name", "mse")).lower()
    if loss_name == "mse":
        criterion: nn.Module = nn.MSELoss()
    elif loss_name == "huber":
        criterion = nn.HuberLoss(delta=float(loss_config.get("delta", 5.0)))
    else:
        raise ValueError(f"Unsupported loss: {loss_name}")

    scheduler_config = config["training"].get("scheduler", {"name": "constant"})
    if isinstance(scheduler_config, str):
        scheduler_config = {"name": scheduler_config}
    scheduler_name = str(scheduler_config.get("name", "constant")).lower()
    if scheduler_name == "constant":
        scheduler: torch.optim.lr_scheduler.LRScheduler | None = None
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
    accumulation_steps = int(config["training"].get("gradient_accumulation_steps", 1))
    if accumulation_steps < 1:
        raise ValueError("gradient_accumulation_steps must be at least 1")
    primary_metric = str(config["evaluation"].get("primary_metric", "mean_mae"))
    early_stopping = config["training"].get("early_stopping", {})
    patience = int(early_stopping.get("patience", 0))
    min_delta = float(early_stopping.get("min_delta", 0.0))

    args.output.mkdir(parents=True, exist_ok=True)
    best_score = float("inf")
    epochs_without_improvement = 0
    history: list[dict[str, float]] = []
    start_epoch = 1
    last_checkpoint = args.output / "last.pt"
    if args.resume and last_checkpoint.exists():
        checkpoint = torch.load(last_checkpoint, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        scaler.load_state_dict(checkpoint["scaler"])
        if scheduler is not None and checkpoint.get("scheduler") is not None:
            scheduler.load_state_dict(checkpoint["scheduler"])
        best_score = float(checkpoint["best_score"])
        epochs_without_improvement = int(checkpoint.get("epochs_without_improvement", 0))
        history = checkpoint["history"]
        start_epoch = int(checkpoint["epoch"]) + 1
        print(f"Resuming from epoch {start_epoch}")

    for epoch in range(start_epoch, int(config["training"]["epochs"]) + 1):
        model.train()
        running_loss = 0.0
        optimizer.zero_grad(set_to_none=True)
        train_loader = loaders["train"]
        for step, (signals, labels) in enumerate(
            tqdm(train_loader, desc=f"Epoch {epoch}", mininterval=5), start=1
        ):
            signals = signals.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            with torch.autocast(device_type=device.type, enabled=use_amp):
                predictions = model(signals)
                normalized_labels = (labels - target_mean) / target_std
                loss = criterion(predictions, normalized_labels)
                scaled_loss = loss / accumulation_steps
            scaler.scale(scaled_loss).backward()
            if step % accumulation_steps == 0 or step == len(train_loader):
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
            running_loss += float(loss.detach()) * len(signals)

        validation = evaluate(model, loaders["val"], device, target_mean, target_std)
        row = {
            "epoch": epoch,
            "train_loss": running_loss / len(datasets["train"]),
            "loss": loss_name,
            "learning_rate": float(optimizer.param_groups[0]["lr"]),
            **validation,
        }
        if loss_name == "mse":
            row["train_mse"] = row["train_loss"]
        history.append(row)
        print(json.dumps(row))
        if primary_metric not in validation:
            raise KeyError(f"Unknown primary metric: {primary_metric}")
        improved = validation[primary_metric] < best_score - min_delta
        if improved:
            best_score = validation[primary_metric]
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
        (args.output / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
        if patience > 0 and epochs_without_improvement >= patience:
            print(
                f"Early stopping at epoch {epoch}; "
                f"{primary_metric} did not improve by {min_delta} for {patience} epochs"
            )
            break

    checkpoint = torch.load(args.output / "best.pt", map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model"])
    test_metrics = evaluate(model, loaders["test"], device, target_mean, target_std)
    test_targets = np.asarray(datasets["test"].labels[datasets["test"].indices], dtype=np.float32)
    mean_predictions = np.broadcast_to(train_target_mean.cpu().numpy(), test_targets.shape)
    mean_baseline = regression_metrics(mean_predictions, test_targets)
    result = {
        "device": str(device),
        "best_epoch": int(checkpoint["epoch"]),
        "completed_epochs": len(history),
        "stopped_early": len(history) < int(config["training"]["epochs"]),
        "loss": loss_name,
        "scheduler": scheduler_name,
        "target_standardization": standardize_targets,
        "train_target_mean": train_target_mean.cpu().tolist(),
        "train_target_std": train_target_std.cpu().tolist(),
        "mean_predictor_test": mean_baseline,
        "test": test_metrics,
        "history": history,
    }
    (args.output / "metrics.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result["test"], indent=2))


if __name__ == "__main__":
    main()
