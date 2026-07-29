"""Evaluate an existing BP checkpoint under a fixed corruption suite."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import torch
import yaml
from torch.utils.data import DataLoader

from ppg_bp.data import PulseDBMemmapDataset, build_augmenter
from ppg_bp.models import build_model
from ppg_bp.training.metrics import regression_metrics
from train_robustness import collect_predictions, evaluate_pair


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--suite-config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    suite_config = yaml.safe_load(
        args.suite_config.read_text(encoding="utf-8")
    )
    root_text = os.path.expandvars(suite_config["data"]["root"])
    if "$" in root_text:
        raise EnvironmentError("Set PPG_BP_DATA_ROOT before evaluation")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(
        args.checkpoint,
        map_location=device,
        weights_only=False,
    )
    model = build_model(checkpoint["config"]["model"]).to(device)
    model.load_state_dict(checkpoint["model"])
    target_mean = checkpoint["target_mean"].to(device)
    target_std = checkpoint["target_std"].to(device)

    data = suite_config["data"]
    dataset = PulseDBMemmapDataset(
        Path(root_text),
        "test",
        data["normalization"],
        data.get("label_filter"),
        split_filename=str(data.get("split_filename", "split.npy")),
        input_representation=str(data["input_representation"]),
        derivative_normalization=str(data["derivative_normalization"]),
    )
    loader = DataLoader(
        dataset,
        batch_size=int(suite_config["training"]["batch_size"]),
        shuffle=False,
        num_workers=0,
        pin_memory=device.type == "cuda",
    )
    robustness = suite_config["robustness"]
    augmenter = build_augmenter(
        robustness.get(
            "evaluation_augmentation",
            robustness["augmentation"],
        ),
        sampling_rate=float(data["sampling_rate"]),
    )
    clean_predictions, targets = collect_predictions(
        model,
        loader,
        device,
        target_mean,
        target_std,
    )
    clean_metrics = regression_metrics(clean_predictions, targets)
    conditions: dict[str, object] = {}
    evaluation_suite = robustness["evaluation_suite"]
    for transform in evaluation_suite["transforms"]:
        conditions[transform] = {}
        for severity in evaluation_suite["severities"]:
            metrics, shift = evaluate_pair(
                model,
                loader,
                device,
                target_mean,
                target_std,
                augmenter,
                transform=str(transform),
                severity=float(severity),
                clean_predictions=clean_predictions,
                targets=targets,
            )
            conditions[transform][str(severity)] = {
                "metrics": metrics,
                "prediction_shift_mae": shift,
            }

    args.output.mkdir(parents=True, exist_ok=True)
    result = {
        "checkpoint": str(args.checkpoint),
        "clean_test": clean_metrics,
        "robustness_test": conditions,
    }
    (args.output / "metrics.json").write_text(
        json.dumps(result, indent=2),
        encoding="utf-8",
    )
    print(json.dumps({"clean_test": clean_metrics}, indent=2))


if __name__ == "__main__":
    main()
