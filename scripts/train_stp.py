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
from torch.utils.data import DataLoader, WeightedRandomSampler
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
    roles = tuple(config["data"].get("roles", ()))
    maximum_samples = config["data"].get("maximum_samples")
    datasets = {
        split: STPWindowDataset(
            root,
            split,
            stage,
            sources=sources,
            roles=roles,
            maximum_samples=maximum_samples,
        )
        for split in ("train", "val", "test")
    }
    workers = int(config["training"].get("num_workers", 0))
    loaders = {}
    for split, dataset in datasets.items():
        sampler = None
        if (
            split == "train"
            and bool(config["training"].get("subject_balanced_sampling", False))
        ):
            subject_indices = np.fromiter(
                (subject_index for subject_index, _ in dataset.index),
                dtype=np.int64,
                count=len(dataset.index),
            )
            counts = np.bincount(
                subject_indices,
                minlength=len(dataset.subjects),
            ).clip(min=1)
            sample_weights = 1.0 / counts[subject_indices]
            generator = torch.Generator()
            generator.manual_seed(int(config.get("seed", 42)))
            sampler = WeightedRandomSampler(
                torch.from_numpy(sample_weights).double(),
                num_samples=len(dataset),
                replacement=True,
                generator=generator,
            )
        loaders[split] = DataLoader(
            dataset,
            batch_size=int(config["training"]["batch_size"]),
            shuffle=split == "train" and sampler is None,
            sampler=sampler,
            drop_last=split == "train",
            num_workers=workers if split == "train" else 0,
            persistent_workers=split == "train" and workers > 0,
            pin_memory=device.type == "cuda",
        )
    return datasets, loaders


def pattern_loss_weights(
    dataset: STPWindowDataset,
    mode: str,
    device: torch.device,
) -> torch.Tensor | None:
    mode = str(mode).lower()
    if mode in {"", "none"}:
        return None
    counts = np.bincount(
        dataset.target_values("patterns").astype(np.int64),
        minlength=3,
    ).clip(min=1)
    inverse = counts.sum() / (len(counts) * counts)
    if mode == "sqrt_inverse":
        inverse = np.sqrt(inverse)
    elif mode != "balanced":
        raise ValueError(
            "class_weighting must be none, balanced, or sqrt_inverse"
        )
    inverse = inverse / inverse.mean()
    return torch.as_tensor(inverse, dtype=torch.float32, device=device)


def regression_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    *,
    mode: str,
    huber_delta: float,
) -> torch.Tensor:
    mode = str(mode).lower()
    if mode == "mse":
        return F.mse_loss(prediction, target)
    if mode in {"huber", "smooth_l1"}:
        return F.huber_loss(prediction, target, delta=huber_delta)
    if mode in {"mae", "l1"}:
        return F.l1_loss(prediction, target)
    raise ValueError(f"Unsupported BP regression loss: {mode}")


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
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()
    config = load_config(args.config)
    stage = str(config["model"]["stage"]).lower()

    seed = int(args.seed if args.seed is not None else config.get("seed", 42))
    config["seed"] = seed
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
            if not bool(config["training"].get("allow_random_init", False)):
                raise ValueError(f"{stage} stage requires source_checkpoint")
            print("Using a randomly initialized encoder (explicit ablation).")
        else:
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
        train_labels = np.asarray(
            datasets["train"].target_values("labels"),
            dtype=np.float32,
        )
        if bool(config["training"].get("normalize_targets", True)):
            target_mean = torch.from_numpy(train_labels.mean(axis=0)).to(device)
            target_std = torch.from_numpy(
                train_labels.std(axis=0).clip(min=1.0)
            ).to(device)
    pulse_pressure_std = torch.tensor(1.0, device=device)
    if stage == "bp":
        pulse_pressure_std = torch.tensor(
            max(float(np.std(train_labels[:, 0] - train_labels[:, 1])), 1.0),
            device=device,
        )

    learning_rate = float(config["training"]["learning_rate"])
    encoder_lr_scale = float(
        config["training"].get("encoder_learning_rate_scale", 1.0)
    )
    if stage != "pretrain" and encoder_lr_scale != 1.0:
        encoder_parameters = list(model.encoder.parameters())
        encoder_ids = {id(parameter) for parameter in encoder_parameters}
        head_parameters = [
            parameter
            for parameter in model.parameters()
            if id(parameter) not in encoder_ids
        ]
        parameter_groups = [
            {
                "params": encoder_parameters,
                "lr": learning_rate * encoder_lr_scale,
            },
            {"params": head_parameters, "lr": learning_rate},
        ]
    else:
        parameter_groups = model.parameters()
    optimizer_name = str(config["training"].get("optimizer", "adamw")).lower()
    betas = tuple(float(value) for value in config["training"].get("betas", (0.9, 0.999)))
    optimizer_class = {
        "adam": torch.optim.Adam,
        "adamw": torch.optim.AdamW,
    }.get(optimizer_name)
    if optimizer_class is None:
        raise ValueError("optimizer must be 'adam' or 'adamw'")
    optimizer = optimizer_class(
        parameter_groups,
        lr=learning_rate,
        betas=betas,
        weight_decay=float(config["training"].get("weight_decay", 0)),
    )
    epochs = int(config["training"]["epochs"])
    scheduler_name = str(config["training"].get("scheduler", "cosine")).lower()
    if scheduler_name == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=epochs,
            eta_min=float(config["training"].get("minimum_learning_rate", 1e-6)),
        )
    elif scheduler_name in {"none", "constant"}:
        scheduler = None
    else:
        raise ValueError("scheduler must be 'cosine' or 'none'")
    use_amp = bool(config["training"].get("amp", True)) and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    patience = int(config["training"].get("early_stopping_patience", 10))
    minimum_improvement = float(
        config["training"].get("minimum_improvement", 0.0)
    )
    freeze_encoder_epochs = int(
        config["training"].get("freeze_encoder_epochs", 0)
    )
    class_weights = (
        pattern_loss_weights(
            datasets["train"],
            str(config["training"].get("class_weighting", "none")),
            device,
        )
        if stage == "pattern"
        else None
    )
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
        if scheduler is not None and checkpoint.get("scheduler") is not None:
            scheduler.load_state_dict(checkpoint["scheduler"])
        scaler.load_state_dict(checkpoint["scaler"])
        best_score = float(checkpoint["best_score"])
        epochs_without_improvement = int(checkpoint["epochs_without_improvement"])
        start_epoch = int(checkpoint["epoch"]) + 1
        history = checkpoint["history"]
        print(f"Resuming from epoch {start_epoch}")

    for epoch in range(start_epoch, epochs + 1):
        if stage != "pretrain":
            encoder_trainable = epoch > freeze_encoder_epochs
            for parameter in model.encoder.parameters():
                parameter.requires_grad_(encoder_trainable)
        model.train()
        if stage != "pretrain" and not encoder_trainable:
            model.encoder.eval()
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
                        weight=class_weights,
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
                    loss = regression_loss(
                        prediction,
                        normalized_labels,
                        mode=str(config["training"].get("loss", "mse")),
                        huber_delta=float(
                            config["training"].get("huber_delta", 1.0)
                        ),
                    )
                    pulse_pressure_weight = float(
                        config["training"].get(
                            "pulse_pressure_loss_weight",
                            0.0,
                        )
                    )
                    if pulse_pressure_weight > 0:
                        physical_prediction = (
                            prediction * target_std + target_mean
                        )
                        predicted_pulse_pressure = (
                            physical_prediction[:, 0]
                            - physical_prediction[:, 1]
                        )
                        target_pulse_pressure = labels[:, 0] - labels[:, 1]
                        loss = loss + pulse_pressure_weight * F.huber_loss(
                            predicted_pulse_pressure / pulse_pressure_std,
                            target_pulse_pressure / pulse_pressure_std,
                            delta=float(
                                config["training"].get("huber_delta", 1.0)
                            ),
                        )
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
        if score < best_score - minimum_improvement:
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
        if scheduler is not None:
            scheduler.step()
        torch.save(
            {
                "stage": stage,
                "epoch": epoch,
                "model": model.state_dict(),
                "encoder": model.encoder.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict() if scheduler is not None else None,
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
