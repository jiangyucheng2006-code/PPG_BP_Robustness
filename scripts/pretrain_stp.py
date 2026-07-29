"""STP-inspired transformation-recognition pretraining for PPG encoders."""

from __future__ import annotations

import argparse
import json
import os
import random
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.nn import functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from ppg_bp.data import PulseDBMemmapDataset, build_augmenter
from ppg_bp.models import build_model


def load_config(path: Path) -> dict:
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    config["data"]["root"] = os.path.expandvars(config["data"]["root"])
    if "$" in config["data"]["root"]:
        raise EnvironmentError("Set PPG_BP_DATA_ROOT before pretraining")
    return config


@torch.no_grad()
def evaluate(model, loader, augmenter, device) -> tuple[float, float]:
    model.eval()
    correct = 0
    count = 0
    severity_error = 0.0
    for signals, _ in loader:
        signals = signals.to(device, non_blocking=True)
        augmented, labels, severity = augmenter(signals)
        outputs = model(augmented)
        correct += int((outputs["artifact_type"].argmax(dim=1) == labels).sum())
        severity_error += float(
            torch.abs(outputs["artifact_severity"] - severity).sum()
        )
        count += len(signals)
    return correct / count, severity_error / count


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    config = load_config(args.config)

    seed = int(config["seed"])
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    device = torch.device(
        "cuda"
        if config["training"]["device"] == "auto" and torch.cuda.is_available()
        else config["training"]["device"]
        if config["training"]["device"] != "auto"
        else "cpu"
    )

    root = Path(config["data"]["root"])
    datasets = {
        name: PulseDBMemmapDataset(
            root,
            name,
            config["data"]["normalization"],
            split_filename=str(config["data"].get("split_filename", "split.npy")),
            input_representation=str(config["data"]["input_representation"]),
            derivative_normalization=str(
                config["data"]["derivative_normalization"]
            ),
        )
        for name in ("train", "val")
    }
    workers = int(config["training"]["num_workers"])
    loaders = {
        name: DataLoader(
            dataset,
            batch_size=int(config["training"]["batch_size"]),
            shuffle=name == "train",
            drop_last=name == "train",
            num_workers=workers if name == "train" else 0,
            pin_memory=device.type == "cuda",
            persistent_workers=name == "train" and workers > 0,
        )
        for name, dataset in datasets.items()
    }
    augmenter = build_augmenter(
        config["pretraining"]["augmentation"],
        sampling_rate=float(config["data"]["sampling_rate"]),
    )
    model = build_model(config["model"]).to(device)
    if getattr(model, "artifact_classes", 0) != augmenter.class_count:
        raise ValueError("artifact_classes must match pretraining transforms")
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config["training"]["learning_rate"]),
        weight_decay=float(config["training"]["weight_decay"]),
    )
    use_amp = bool(config["training"]["amp"]) and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    severity_weight = float(
        config["pretraining"].get("severity_loss_weight", 0.1)
    )

    args.output.mkdir(parents=True, exist_ok=True)
    last_checkpoint = args.output / "last.pt"
    start_epoch = 1
    best_accuracy = 0.0
    history: list[dict[str, float]] = []
    if args.resume and last_checkpoint.exists():
        checkpoint = torch.load(
            last_checkpoint,
            map_location=device,
            weights_only=False,
        )
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        scaler.load_state_dict(checkpoint["scaler"])
        start_epoch = int(checkpoint["epoch"]) + 1
        best_accuracy = float(checkpoint["best_accuracy"])
        history = checkpoint["history"]

    for epoch in range(start_epoch, int(config["training"]["epochs"]) + 1):
        model.train()
        total_loss = 0.0
        total_correct = 0
        total_samples = 0
        for signals, _ in tqdm(
            loaders["train"],
            desc=f"Pretrain epoch {epoch}",
            mininterval=5,
        ):
            signals = signals.to(device, non_blocking=True)
            augmented, labels, severity = augmenter(signals)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, enabled=use_amp):
                outputs = model(augmented)
                classification_loss = F.cross_entropy(
                    outputs["artifact_type"],
                    labels,
                )
                severity_loss = F.mse_loss(
                    outputs["artifact_severity"],
                    severity,
                )
                loss = classification_loss + severity_weight * severity_loss
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            total_loss += float(loss.detach()) * len(signals)
            total_correct += int(
                (outputs["artifact_type"].argmax(dim=1) == labels).sum()
            )
            total_samples += len(signals)

        val_accuracy, val_severity_mae = evaluate(
            model,
            loaders["val"],
            augmenter,
            device,
        )
        row = {
            "epoch": epoch,
            "train_loss": total_loss / total_samples,
            "train_accuracy": total_correct / total_samples,
            "val_accuracy": val_accuracy,
            "val_severity_mae": val_severity_mae,
        }
        history.append(row)
        print(json.dumps(row))
        if val_accuracy > best_accuracy:
            best_accuracy = val_accuracy
            torch.save(
                {
                    "backbone": model.backbone.state_dict(),
                    "model": model.state_dict(),
                    "config": config,
                    "epoch": epoch,
                    "val_accuracy": val_accuracy,
                },
                args.output / "best.pt",
            )
        torch.save(
            {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scaler": scaler.state_dict(),
                "config": config,
                "epoch": epoch,
                "best_accuracy": best_accuracy,
                "history": history,
            },
            last_checkpoint,
        )
        (args.output / "history.json").write_text(
            json.dumps(history, indent=2),
            encoding="utf-8",
        )

    summary = {
        "best_validation_accuracy": best_accuracy,
        "completed_epochs": len(history),
        "history": history,
    }
    (args.output / "metrics.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
