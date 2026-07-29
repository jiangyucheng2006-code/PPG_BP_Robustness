"""Train one stage of the three-stage STP reproduction."""

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

from ppg_bp.data import STPWindowDataset, build_stp_transform_bank
from ppg_bp.models import build_model, transfer_encoder
from ppg_bp.training.metrics import regression_metrics


def expand_environment(value):
    if isinstance(value, dict):
        return {key: expand_environment(item) for key, item in value.items()}
    if isinstance(value, list):
        return [expand_environment(item) for item in value]
    if isinstance(value, str):
        expanded = os.path.expandvars(value)
        if "$" in expanded:
            raise EnvironmentError(f"Unresolved environment variable in {value}")
        return expanded
    return value


def load_config(path: Path) -> dict:
    return expand_environment(yaml.safe_load(path.read_text(encoding="utf-8")))


def choose_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(requested)


def build_loaders(config: dict, stage: str, device: torch.device):
    root = Path(config["data"]["root"])
    sources = tuple(config["data"].get("sources", ()))
    maximum_samples = config["data"].get("maximum_samples")
    datasets = {
        split: STPWindowDataset(
            root,
            split,
            stage,
            sources=sources,
            maximum_samples=maximum_samples,
        )
        for split in ("train", "val", "test")
    }
    workers = int(config["training"].get("num_workers", 0))
    loaders = {
        split: DataLoader(
            dataset,
            batch_size=int(config["training"]["batch_size"]),
            shuffle=split == "train",
            drop_last=split == "train",
            num_workers=workers if split == "train" else 0,
            persistent_workers=split == "train" and workers > 0,
            pin_memory=device.type == "cuda",
        )
        for split, dataset in datasets.items()
    }
    return datasets, loaders


@torch.no_grad()
def evaluate_pretrain(model, loader, transforms, device) -> dict[str, float]:
    model.eval()
    squared_error = 0.0
    absolute_error = 0.0
    count = 0
    for clean in loader:
        clean = clean.to(device, non_blocking=True)
        transformed, target, _ = transforms(clean)
        reconstruction = model(transformed)
        squared_error += float(F.mse_loss(reconstruction, target, reduction="sum"))
        absolute_error += float(F.l1_loss(reconstruction, target, reduction="sum"))
        count += target.numel()
    return {
        "reconstruction_mse": squared_error / count,
        "reconstruction_mae": absolute_error / count,
    }


@torch.no_grad()
def evaluate_pattern(model, loader, device) -> dict[str, object]:
    model.eval()
    correct = 0
    count = 0
    total_loss = 0.0
    confusion = torch.zeros(3, 3, dtype=torch.long)
    for signals, labels in loader:
        signals = signals.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        logits = model(signals)
        total_loss += float(F.cross_entropy(logits, labels, reduction="sum"))
        predictions = logits.argmax(dim=1)
        correct += int((predictions == labels).sum())
        count += len(labels)
        indices = labels.cpu() * 3 + predictions.cpu()
        confusion += torch.bincount(indices, minlength=9).reshape(3, 3)
    recalls = confusion.diag() / confusion.sum(dim=1).clamp_min(1)
    return {
        "pattern_loss": total_loss / count,
        "pattern_accuracy": correct / count,
        "pattern_macro_recall": float(recalls.mean()),
        "confusion_matrix": confusion.tolist(),
    }


@torch.no_grad()
def evaluate_bp(
    model,
    loader,
    device,
    target_mean: torch.Tensor,
    target_std: torch.Tensor,
) -> dict[str, object]:
    model.eval()
    predictions: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    for signals, labels in loader:
        normalized = model(signals.to(device, non_blocking=True))
        predictions.append(
            (normalized * target_std + target_mean).cpu().numpy()
        )
        targets.append(labels.numpy())
    return regression_metrics(
        np.concatenate(predictions),
        np.concatenate(targets),
    )


def validation_score(stage: str, metrics: dict[str, object]) -> float:
    if stage == "pretrain":
        return float(metrics["reconstruction_mse"])
    if stage == "pattern":
        return -float(metrics["pattern_macro_recall"])
    return float(metrics["mean_mae"])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    config = load_config(args.config)
    stage = str(config["model"]["stage"]).lower()

    seed = int(config.get("seed", 42))
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    device = choose_device(str(config["training"].get("device", "auto")))
    datasets, loaders = build_loaders(config, stage, device)
    model = build_model(config["model"]).to(device)

    source_checkpoint = config["training"].get("source_checkpoint")
    if stage != "pretrain":
        if not source_checkpoint:
            raise ValueError(f"{stage} stage requires source_checkpoint")
        checkpoint_path = Path(source_checkpoint)
        if not checkpoint_path.exists():
            raise FileNotFoundError(checkpoint_path)
        source = torch.load(
            checkpoint_path,
            map_location=device,
            weights_only=False,
        )
        transfer_encoder(model, source)
        print(
            f"Transferred encoder from {checkpoint_path} "
            f"(stage={source.get('stage', 'unknown')})"
        )

    transforms = None
    if stage == "pretrain":
        transforms = build_stp_transform_bank(
            config["pretraining"]["transforms"],
            sampling_rate=float(config["data"].get("sampling_rate", 125)),
        )

    target_mean = torch.zeros(2, device=device)
    target_std = torch.ones(2, device=device)
    if stage == "bp":
        train_labels = np.concatenate(
            [
                np.asarray(
                    np.load(
                        datasets["train"].root / entry["labels"],
                        mmap_mode="r",
                    ),
                    dtype=np.float32,
                )
                for entry in datasets["train"].subjects
            ]
        )
        target_mean = torch.from_numpy(train_labels.mean(axis=0)).to(device)
        target_std = torch.from_numpy(
            train_labels.std(axis=0).clip(min=1.0)
        ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config["training"]["learning_rate"]),
        weight_decay=float(config["training"].get("weight_decay", 0)),
    )
    epochs = int(config["training"]["epochs"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=epochs,
        eta_min=float(config["training"].get("minimum_learning_rate", 1e-6)),
    )
    use_amp = bool(config["training"].get("amp", True)) and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    patience = int(config["training"].get("early_stopping_patience", 10))
    args.output.mkdir(parents=True, exist_ok=True)
    best_score = float("inf")
    epochs_without_improvement = 0
    start_epoch = 1
    history: list[dict[str, object]] = []
    last_path = args.output / "last.pt"
    if args.resume and last_path.exists():
        checkpoint = torch.load(last_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        scaler.load_state_dict(checkpoint["scaler"])
        best_score = float(checkpoint["best_score"])
        epochs_without_improvement = int(checkpoint["epochs_without_improvement"])
        start_epoch = int(checkpoint["epoch"]) + 1
        history = checkpoint["history"]
        print(f"Resuming from epoch {start_epoch}")

    for epoch in range(start_epoch, epochs + 1):
        model.train()
        running_loss = 0.0
        samples = 0
        for batch in tqdm(
            loaders["train"],
            desc=f"STP {stage} epoch {epoch}",
            mininterval=2,
        ):
            optimizer.zero_grad(set_to_none=True)
            if stage == "pretrain":
                clean = batch.to(device, non_blocking=True)
                transformed, target, _ = transforms(clean)
                with torch.autocast(device_type=device.type, enabled=use_amp):
                    prediction = model(transformed)
                    loss = F.mse_loss(prediction, target)
                batch_samples = len(clean)
            elif stage == "pattern":
                signals, labels = batch
                signals = signals.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)
                with torch.autocast(device_type=device.type, enabled=use_amp):
                    prediction = model(signals)
                    # The paper calls this the pattern adversarial loss.  The
                    # accessible figure confirms the three-way discriminator;
                    # its unpublished exact equation remains configurable.
                    loss = F.cross_entropy(
                        prediction,
                        labels,
                        label_smoothing=float(
                            config["training"].get("label_smoothing", 0)
                        ),
                    )
                batch_samples = len(signals)
            else:
                signals, labels = batch
                signals = signals.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)
                normalized_labels = (labels - target_mean) / target_std
                with torch.autocast(device_type=device.type, enabled=use_amp):
                    prediction = model(signals)
                    loss = F.mse_loss(prediction, normalized_labels)
                batch_samples = len(signals)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(
                model.parameters(),
                float(config["training"].get("gradient_clip_norm", 1.0)),
            )
            scaler.step(optimizer)
            scaler.update()
            running_loss += float(loss.detach()) * batch_samples
            samples += batch_samples

        if stage == "pretrain":
            validation = evaluate_pretrain(
                model,
                loaders["val"],
                transforms,
                device,
            )
        elif stage == "pattern":
            validation = evaluate_pattern(model, loaders["val"], device)
        else:
            validation = evaluate_bp(
                model,
                loaders["val"],
                device,
                target_mean,
                target_std,
            )
        row = {
            "epoch": epoch,
            "train_loss": running_loss / samples,
            "learning_rate": optimizer.param_groups[0]["lr"],
            **validation,
        }
        history.append(row)
        print(json.dumps(row))
        score = validation_score(stage, validation)
        if score < best_score:
            best_score = score
            epochs_without_improvement = 0
            torch.save(
                {
                    "stage": stage,
                    "epoch": epoch,
                    "model": model.state_dict(),
                    "encoder": model.encoder.state_dict(),
                    "config": config,
                    "target_mean": target_mean.cpu(),
                    "target_std": target_std.cpu(),
                    "validation": validation,
                },
                args.output / "best.pt",
            )
        else:
            epochs_without_improvement += 1
        scheduler.step()
        torch.save(
            {
                "stage": stage,
                "epoch": epoch,
                "model": model.state_dict(),
                "encoder": model.encoder.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "scaler": scaler.state_dict(),
                "config": config,
                "target_mean": target_mean.cpu(),
                "target_std": target_std.cpu(),
                "best_score": best_score,
                "epochs_without_improvement": epochs_without_improvement,
                "history": history,
            },
            last_path,
        )
        (args.output / "history.json").write_text(
            json.dumps(history, indent=2),
            encoding="utf-8",
        )
        if patience > 0 and epochs_without_improvement >= patience:
            print(f"Early stopping at epoch {epoch}")
            break

    best = torch.load(
        args.output / "best.pt",
        map_location=device,
        weights_only=False,
    )
    model.load_state_dict(best["model"])
    if stage == "pretrain":
        test = evaluate_pretrain(model, loaders["test"], transforms, device)
    elif stage == "pattern":
        test = evaluate_pattern(model, loaders["test"], device)
    else:
        test = evaluate_bp(model, loaders["test"], device, target_mean, target_std)
    metrics = {
        "stage": stage,
        "device": str(device),
        "best_epoch": int(best["epoch"]),
        "completed_epochs": len(history),
        "subjects": {
            split: len(dataset.subjects) for split, dataset in datasets.items()
        },
        "samples": {split: len(dataset) for split, dataset in datasets.items()},
        "test": test,
        "history": history,
    }
    (args.output / "metrics.json").write_text(
        json.dumps(metrics, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(test, indent=2))


if __name__ == "__main__":
    main()
