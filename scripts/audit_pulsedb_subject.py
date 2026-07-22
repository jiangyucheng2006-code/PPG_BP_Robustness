"""Inspect one real PulseDB subject file and save a waveform preview."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from ppg_bp.data.pulsedb import read_pulsedb_subject


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    subject = read_pulsedb_subject(args.input)
    summary = {
        "subject_id": subject.subject_id,
        "segments": int(len(subject.ppg)),
        "samples_per_segment": int(subject.ppg.shape[1]),
        "sbp_min": float(subject.labels[:, 0].min()),
        "sbp_mean": float(subject.labels[:, 0].mean()),
        "sbp_max": float(subject.labels[:, 0].max()),
        "dbp_min": float(subject.labels[:, 1].min()),
        "dbp_mean": float(subject.labels[:, 1].mean()),
        "dbp_max": float(subject.labels[:, 1].max()),
        "mean_ppg_abp_correlation": float(np.nanmean(subject.correlation)),
    }
    args.output.mkdir(parents=True, exist_ok=True)
    with (args.output / "summary.json").open("w", encoding="utf-8") as stream:
        json.dump(summary, stream, indent=2)

    seconds = np.arange(subject.ppg.shape[1]) / 125.0
    plt.figure(figsize=(10, 3.5))
    plt.plot(seconds, subject.ppg[0], linewidth=1.2)
    plt.xlabel("Time (s)")
    plt.ylabel("Normalized PPG")
    plt.title(f"PulseDB {subject.subject_id}: SBP {subject.labels[0,0]:.1f}, DBP {subject.labels[0,1]:.1f} mmHg")
    plt.tight_layout()
    plt.savefig(args.output / "first_segment.png", dpi=160)
    plt.close()
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

