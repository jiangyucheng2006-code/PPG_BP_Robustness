"""Evaluate a trained checkpoint on an official PulseDB subset MAT file."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import h5py
import numpy as np
import torch

from ppg_bp.data.pulsedb import ppg_representation
from ppg_bp.models import build_model
from ppg_bp.training.metrics import regression_metrics


def decode_matlab_text(handle: h5py.File, reference: h5py.Reference) -> str:
    values = np.asarray(handle[reference][()]).reshape(-1)
    return "".join(chr(int(value)) for value in values if int(value) != 0)


def load_subset(path: Path) -> tuple[np.ndarray, np.ndarray, list[str]]:
    with h5py.File(path, "r") as handle:
        group = handle["Subset"]
        signals = np.asarray(group["Signals"][:, 1, :], dtype=np.float32).T
        labels = np.column_stack(
            (
                np.asarray(group["SBP"][0], dtype=np.float32),
                np.asarray(group["DBP"][0], dtype=np.float32),
            )
        )
        subjects = [
            decode_matlab_text(handle, reference)
            for reference in group["Subject"][0]
        ]
    return signals, labels, subjects


def prepare_batch(
    signals: np.ndarray,
    *,
    normalization: str,
    input_representation: str,
    derivative_normalization: str,
) -> np.ndarray:
    prepared = []
    for signal in signals:
        if normalization == "per_segment_zscore":
            signal = (signal - signal.mean()) / max(float(signal.std()), 1e-6)
        prepared.append(
            ppg_representation(
                signal,
                input_representation,
                derivative_normalization=derivative_normalization,
            )
        )
    return np.stack(prepared)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--subset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    device = torch.device(
        "cuda" if args.device == "auto" and torch.cuda.is_available()
        else "cpu" if args.device == "auto"
        else args.device
    )
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    config = checkpoint["config"]
    model = build_model(config["model"]).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()

    signals, labels, subjects = load_subset(args.subset)
    data_config = config["data"]
    input_representation = str(data_config.get("input_representation", "ppg"))
    normalization = str(data_config.get("normalization", "none"))
    derivative_normalization = str(
        data_config.get("derivative_normalization", "per_segment_zscore")
    )
    target_mean = checkpoint["target_mean"].to(device)
    target_std = checkpoint["target_std"].to(device)

    predictions = []
    with torch.no_grad():
        for start in range(0, len(signals), args.batch_size):
            batch = prepare_batch(
                signals[start : start + args.batch_size],
                normalization=normalization,
                input_representation=input_representation,
                derivative_normalization=derivative_normalization,
            )
            normalized = model(torch.from_numpy(batch).to(device))
            predictions.append(
                (normalized * target_std + target_mean).cpu().numpy()
            )
    metrics = regression_metrics(np.concatenate(predictions), labels)
    subject_counts = Counter(subjects)
    distribution_checks = {
        "sbp_le_100_pct": float((labels[:, 0] <= 100.0).mean() * 100.0),
        "sbp_ge_160_pct": float((labels[:, 0] >= 160.0).mean() * 100.0),
        "sbp_ge_140_pct": float((labels[:, 0] >= 140.0).mean() * 100.0),
        "dbp_le_60_pct": float((labels[:, 1] <= 60.0).mean() * 100.0),
        "dbp_ge_100_pct": float((labels[:, 1] >= 100.0).mean() * 100.0),
        "dbp_ge_85_pct": float((labels[:, 1] >= 85.0).mean() * 100.0),
    }
    distribution_eligible = (
        distribution_checks["sbp_le_100_pct"] >= 5.0
        and distribution_checks["sbp_ge_160_pct"] >= 5.0
        and distribution_checks["sbp_ge_140_pct"] >= 20.0
        and distribution_checks["dbp_le_60_pct"] >= 5.0
        and distribution_checks["dbp_ge_100_pct"] >= 5.0
        and distribution_checks["dbp_ge_85_pct"] >= 20.0
    )
    result = {
        "checkpoint": str(args.checkpoint),
        "subset": str(args.subset),
        "input_representation": input_representation,
        "n_samples": len(signals),
        "n_subjects": len(subject_counts),
        "min_measurements_per_subject": min(subject_counts.values()),
        "aami_distribution_checks": distribution_checks,
        "aami_sample_protocol_eligible": (
            len(subject_counts) >= 85
            and min(subject_counts.values()) >= 3
            and distribution_eligible
        ),
        "metrics": metrics,
        "note": (
            "AAMI numerical pass reports only |mean error| <= 5 mmHg and "
            "error SD <= 8 mmHg. It is not a clinical device certification."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
