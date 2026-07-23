"""Create the QUMPHY calibration-free VitalDB subject split.

The upstream protocol randomly holds out 144 subjects from the official
training subset, uses 48 as a calibration set, and uses the remaining 96 for
validation. Calibration subjects are excluded from calibration-free training.
The upstream script does not publish a random seed, so this implementation
records an explicit seed for reproducibility.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import numpy as np
from tqdm import tqdm


def decode_matlab_text(handle: h5py.File, reference: h5py.Reference) -> str:
    values = np.asarray(handle[reference][()]).reshape(-1)
    return "".join(chr(int(value)) for value in values if int(value) != 0)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--test", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    with h5py.File(args.train, "r") as train_handle, h5py.File(args.test, "r") as test_handle:
        train_subject_refs = train_handle["Subset"]["Subject"]
        train_count = int(train_subject_refs.size)
        test_count = int(test_handle["Subset"]["Subject"].size)

        subject_ids: list[str] = []
        subject_to_code: dict[str, int] = {}
        segment_subject_codes = np.empty(train_count, dtype=np.int32)
        for index in tqdm(range(train_count), desc="Reading training subjects"):
            subject_id = decode_matlab_text(train_handle, train_subject_refs[0, index])
            if subject_id not in subject_to_code:
                subject_to_code[subject_id] = len(subject_ids)
                subject_ids.append(subject_id)
            segment_subject_codes[index] = subject_to_code[subject_id]

    rng = np.random.RandomState(args.seed)
    held_out = rng.permutation(len(subject_ids))[:144]
    calibration_subjects = set(int(value) for value in held_out[:48])
    validation_subjects = set(int(value) for value in held_out[48:])

    split = np.full(train_count + test_count, 3, dtype=np.uint8)
    train_mask = np.array(
        [
            code not in calibration_subjects and code not in validation_subjects
            for code in segment_subject_codes
        ],
        dtype=bool,
    )
    val_mask = np.array([code in validation_subjects for code in segment_subject_codes], dtype=bool)
    split[:train_count][train_mask] = 0
    split[:train_count][val_mask] = 1
    split[train_count:] = 2

    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.save(args.output, split)
    audit = {
        "protocol": "QUMPHY PulseDB calibration-free VitalDB",
        "seed": args.seed,
        "upstream_seed_available": False,
        "codes": {"train": 0, "validation": 1, "test": 2, "calibration_excluded": 3},
        "subjects": {
            "total_official_train": len(subject_ids),
            "train": len(subject_ids) - 144,
            "validation": 96,
            "calibration_excluded": 48,
        },
        "windows": {
            "train": int(np.sum(split == 0)),
            "validation": int(np.sum(split == 1)),
            "test": int(np.sum(split == 2)),
            "calibration_excluded": int(np.sum(split == 3)),
        },
    }
    args.output.with_suffix(".json").write_text(json.dumps(audit, indent=2), encoding="utf-8")
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
