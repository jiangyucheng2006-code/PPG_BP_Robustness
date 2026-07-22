"""Convert official PulseDB subset MAT files to model-ready PPG arrays.

The official ``Signals`` field is ordered as ECG, PPG, and ABP. This converter
keeps only channel index 1 (PPG) and the paired SBP/DBP labels. The official
calibration-free test subset remains untouched; validation subjects are held
out deterministically from the official training subset.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import h5py
import numpy as np
from numpy.lib.format import open_memmap
from tqdm import tqdm


def decode_matlab_text(handle: h5py.File, reference: h5py.Reference) -> str:
    values = np.asarray(handle[reference][()]).reshape(-1)
    return "".join(chr(int(value)) for value in values if int(value) != 0)


def is_validation_subject(subject_id: str, fraction: float) -> bool:
    bucket = int(hashlib.sha1(subject_id.encode("utf-8")).hexdigest()[:8], 16) / 0xFFFFFFFF
    return bucket < fraction


def validate_subset(group: h5py.Group) -> int:
    required = {"Signals", "SBP", "DBP", "Subject"}
    missing = required.difference(group.keys())
    if missing:
        raise KeyError(f"Missing PulseDB subset fields: {sorted(missing)}")
    signals = group["Signals"]
    if signals.ndim != 3 or signals.shape[0] != 1250 or signals.shape[1] != 3:
        raise ValueError(f"Unexpected Signals shape: {signals.shape}; expected [1250, 3, N]")
    return int(signals.shape[2])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", type=Path, required=True, help="Official VitalDB_Train_Subset.mat")
    parser.add_argument("--test", type=Path, required=True, help="Official CalFree or AAMI test subset")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--validation-fraction", type=float, default=0.1)
    parser.add_argument("--chunk-size", type=int, default=2048)
    parser.add_argument("--max-train-samples", type=int, help="Engineering check only")
    parser.add_argument("--max-test-samples", type=int, help="Engineering check only")
    args = parser.parse_args()
    if not 0 < args.validation_fraction < 0.5:
        raise ValueError("validation-fraction must be between 0 and 0.5")

    with h5py.File(args.train, "r") as train_handle, h5py.File(args.test, "r") as test_handle:
        train_group = train_handle["Subset"]
        test_group = test_handle["Subset"]
        train_count = validate_subset(train_group)
        test_count = validate_subset(test_group)
        if args.max_train_samples is not None:
            train_count = min(train_count, args.max_train_samples)
        if args.max_test_samples is not None:
            test_count = min(test_count, args.max_test_samples)
        total = train_count + test_count

        args.output.mkdir(parents=True, exist_ok=True)
        ppg = open_memmap(args.output / "ppg.npy", mode="w+", dtype=np.float32, shape=(total, 1250))
        labels = open_memmap(args.output / "labels.npy", mode="w+", dtype=np.float32, shape=(total, 2))
        split = open_memmap(args.output / "split.npy", mode="w+", dtype=np.uint8, shape=(total,))

        unique_train_subjects: set[str] = set()
        unique_val_subjects: set[str] = set()
        subject_refs = train_group["Subject"]
        for index in tqdm(range(train_count), desc="Assigning train/validation subjects"):
            subject_id = decode_matlab_text(train_handle, subject_refs[0, index])
            if is_validation_subject(subject_id, args.validation_fraction):
                split[index] = 1
                unique_val_subjects.add(subject_id)
            else:
                split[index] = 0
                unique_train_subjects.add(subject_id)

        sources = ((train_group, 0, train_count, "train"), (test_group, train_count, test_count, "test"))
        for group, offset, count, name in sources:
            signals = group["Signals"]
            sbp = group["SBP"]
            dbp = group["DBP"]
            for start in tqdm(range(0, count, args.chunk_size), desc=f"Converting {name} PPG"):
                stop = min(start + args.chunk_size, count)
                destination = slice(offset + start, offset + stop)
                ppg[destination] = np.asarray(signals[:, 1, start:stop], dtype=np.float32).T
                labels[destination, 0] = np.asarray(sbp[0, start:stop], dtype=np.float32)
                labels[destination, 1] = np.asarray(dbp[0, start:stop], dtype=np.float32)
            if name == "test":
                split[offset : offset + count] = 2

    ppg.flush()
    labels.flush()
    split.flush()
    metadata = {
        "dataset": "PulseDB v2.0 VitalDB supplementary subsets",
        "n_samples": total,
        "n_train_source_samples": train_count,
        "n_test_samples": test_count,
        "n_training_subjects": len(unique_train_subjects),
        "n_validation_subjects": len(unique_val_subjects),
        "window_samples": 1250,
        "sampling_rate_hz": 125,
        "signal_source": "Signals channel 1 (PPG_F)",
        "label_order": ["SBP", "DBP"],
        "split_strategy": "official test subset; deterministic subject-wise validation holdout from official train",
        "train_file": str(args.train),
        "test_file": str(args.test),
    }
    (args.output / "dataset_meta.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()

