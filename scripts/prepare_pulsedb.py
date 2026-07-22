"""Convert official PulseDB v2 subject files into memory-mapped arrays."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
from numpy.lib.format import open_memmap
from tqdm import tqdm

from ppg_bp.data.pulsedb import read_pulsedb_subject


def deterministic_split(subject_id: str) -> int:
    """Temporary subject-wise 80/10/10 split for engineering smoke tests.

    The publication experiments must replace this with PulseDB's official
    calibration-free subject lists.
    """

    bucket = int(hashlib.sha1(subject_id.encode("utf-8")).hexdigest()[:8], 16) % 100
    return 0 if bucket < 80 else 1 if bucket < 90 else 2


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True, help="Folder containing PulseDB subject .mat files")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--ppg-field", default="PPG_F", choices=("PPG_F", "PPG_Record_F", "PPG_Record"))
    parser.add_argument("--max-subjects", type=int)
    parser.add_argument("--max-segments-per-subject", type=int)
    args = parser.parse_args()

    paths = sorted(args.input.rglob("*.mat"))
    if args.max_subjects is not None:
        paths = paths[: args.max_subjects]
    if not paths:
        raise FileNotFoundError(f"No .mat files found below {args.input}")

    counts: list[int] = []
    for path in tqdm(paths, desc="Counting subject segments"):
        import h5py

        with h5py.File(path, "r") as handle:
            dataset = handle["Subj_Wins"][args.ppg_field]
            if h5py.check_dtype(ref=dataset.dtype) is not None:
                count = int(dataset.size)
            elif dataset.ndim == 2 and 1250 in dataset.shape:
                count = int(dataset.shape[1] if dataset.shape[0] == 1250 else dataset.shape[0])
            else:
                count = 1
        if args.max_segments_per_subject is not None:
            count = min(count, args.max_segments_per_subject)
        counts.append(count)

    capacity = sum(counts)
    args.output.mkdir(parents=True, exist_ok=True)
    ppg = open_memmap(args.output / "ppg.npy", mode="w+", dtype=np.float32, shape=(capacity, 1250))
    labels = open_memmap(args.output / "labels.npy", mode="w+", dtype=np.float32, shape=(capacity, 2))
    split = open_memmap(args.output / "split.npy", mode="w+", dtype=np.uint8, shape=(capacity,))

    cursor = 0
    manifest_rows: list[dict[str, object]] = []
    for path in tqdm(paths, desc="Converting PulseDB subjects"):
        try:
            subject = read_pulsedb_subject(
                path,
                ppg_field=args.ppg_field,
                include_only=True,
                max_segments=args.max_segments_per_subject,
            )
        except (KeyError, ValueError) as error:
            print(f"Skipping {path.name}: {error}")
            continue
        keep = len(subject.ppg)
        if args.max_segments_per_subject is not None:
            keep = min(keep, args.max_segments_per_subject)
        subject_ppg = subject.ppg[:keep]
        subject_labels = subject.labels[:keep]
        plausible = (
            np.isfinite(subject_ppg).all(axis=1)
            & np.isfinite(subject_labels).all(axis=1)
            & (subject_labels[:, 0] >= 60)
            & (subject_labels[:, 0] <= 250)
            & (subject_labels[:, 1] >= 30)
            & (subject_labels[:, 1] <= 150)
            & (subject_labels[:, 0] > subject_labels[:, 1])
        )
        subject_ppg = subject_ppg[plausible]
        subject_labels = subject_labels[plausible]
        keep = len(subject_ppg)
        if keep == 0:
            continue

        stop = cursor + keep
        ppg[cursor:stop] = subject_ppg
        labels[cursor:stop] = subject_labels
        split[cursor:stop] = deterministic_split(subject.subject_id)
        manifest_rows.append(
            {
                "subject_id": subject.subject_id,
                "start": cursor,
                "stop": stop,
                "source_file": str(path),
                "split": ("train", "val", "test")[int(split[cursor])],
            }
        )
        cursor = stop

    ppg.flush()
    labels.flush()
    split.flush()
    metadata = {
        "dataset": "PulseDB v2.0",
        "n_samples": cursor,
        "capacity": capacity,
        "n_subjects": len(manifest_rows),
        "window_samples": 1250,
        "sampling_rate_hz": 125,
        "ppg_field": args.ppg_field,
        "label_order": ["SBP", "DBP"],
        "split_strategy": "deterministic_subject_hash_80_10_10_ENGINEERING_ONLY",
    }
    with (args.output / "dataset_meta.json").open("w", encoding="utf-8") as stream:
        json.dump(metadata, stream, indent=2)
    with (args.output / "subjects.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=("subject_id", "start", "stop", "source_file", "split"))
        writer.writeheader()
        writer.writerows(manifest_rows)
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
