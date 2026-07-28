"""Create compact training-only physiological labels for PulseDB.

Age is copied from the official subset metadata. Heart rate is estimated from
the synchronized ECG, with PPG periodicity used only as a quality check. The
resulting arrays align exactly with the already prepared PPG arrays.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import numpy as np
from numpy.lib.format import open_memmap
from tqdm import tqdm


def periodic_rate(
    signals: np.ndarray,
    *,
    sampling_rate: float,
    minimum_bpm: float,
    maximum_bpm: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Estimate dominant periodic rate using FFT autocorrelation."""

    centered = signals.astype(np.float32, copy=False)
    centered = centered - centered.mean(axis=1, keepdims=True)
    nfft = 1 << int(np.ceil(np.log2(centered.shape[1])))
    spectrum = np.fft.rfft(centered, n=nfft, axis=1)
    autocorrelation = np.fft.irfft(
        spectrum * np.conjugate(spectrum),
        n=nfft,
        axis=1,
    )[:, : centered.shape[1]]
    minimum_lag = int(np.ceil(sampling_rate * 60.0 / maximum_bpm))
    maximum_lag = int(np.floor(sampling_rate * 60.0 / minimum_bpm))
    candidates = autocorrelation[:, minimum_lag : maximum_lag + 1]
    local_index = candidates.argmax(axis=1)
    lag = local_index + minimum_lag
    peak = candidates[np.arange(len(candidates)), local_index]
    quality = peak / np.maximum(autocorrelation[:, 0], 1e-6)
    return (sampling_rate * 60.0 / lag).astype(np.float32), quality.astype(np.float32)


def subset_count(group: h5py.Group) -> int:
    required = {"Signals", "Age"}
    missing = required.difference(group.keys())
    if missing:
        raise KeyError(f"Missing PulseDB fields: {sorted(missing)}")
    return int(group["Signals"].shape[2])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--chunk-size", type=int, default=2048)
    parser.add_argument("--minimum-bpm", type=float, default=40.0)
    parser.add_argument("--maximum-bpm", type=float, default=180.0)
    parser.add_argument("--maximum-ecg-ppg-difference", type=float, default=8.0)
    parser.add_argument("--minimum-autocorrelation-quality", type=float, default=0.1)
    args = parser.parse_args()

    metadata_path = args.dataset_root / "dataset_meta.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    sources = (
        (Path(metadata["train_file"]), int(metadata["n_train_source_samples"])),
        (Path(metadata["test_file"]), int(metadata["n_test_samples"])),
    )
    expected_total = int(metadata["n_samples"])
    if sum(count for _, count in sources) != expected_total:
        raise ValueError("Prepared dataset counts do not match source subset counts")

    temporary_age = args.dataset_root / "age.building.npy"
    temporary_hr = args.dataset_root / "heart_rate.building.npy"
    temporary_valid = args.dataset_root / "heart_rate_valid.building.npy"
    age = open_memmap(temporary_age, mode="w+", dtype=np.float32, shape=(expected_total,))
    heart_rate = open_memmap(
        temporary_hr,
        mode="w+",
        dtype=np.float32,
        shape=(expected_total,),
    )
    heart_rate_valid = open_memmap(
        temporary_valid,
        mode="w+",
        dtype=np.uint8,
        shape=(expected_total,),
    )

    offset = 0
    for path, expected_count in sources:
        with h5py.File(path, "r") as handle:
            group = handle["Subset"]
            count = subset_count(group)
            if count != expected_count:
                raise ValueError(f"Unexpected sample count in {path}: {count}")
            for start in tqdm(
                range(0, count, args.chunk_size),
                desc=f"Auxiliary labels: {path.stem}",
            ):
                stop = min(start + args.chunk_size, count)
                destination = slice(offset + start, offset + stop)
                age[destination] = np.asarray(
                    group["Age"][0, start:stop],
                    dtype=np.float32,
                )
                ecg = np.asarray(
                    group["Signals"][:, 0, start:stop],
                    dtype=np.float32,
                ).T
                ppg = np.asarray(
                    group["Signals"][:, 1, start:stop],
                    dtype=np.float32,
                ).T
                ecg_rate, ecg_quality = periodic_rate(
                    ecg,
                    sampling_rate=float(metadata["sampling_rate_hz"]),
                    minimum_bpm=args.minimum_bpm,
                    maximum_bpm=args.maximum_bpm,
                )
                ppg_rate, ppg_quality = periodic_rate(
                    ppg,
                    sampling_rate=float(metadata["sampling_rate_hz"]),
                    minimum_bpm=args.minimum_bpm,
                    maximum_bpm=args.maximum_bpm,
                )
                valid = (
                    (np.abs(ecg_rate - ppg_rate) <= args.maximum_ecg_ppg_difference)
                    & (ecg_quality >= args.minimum_autocorrelation_quality)
                    & (ppg_quality >= args.minimum_autocorrelation_quality)
                    & np.isfinite(ecg_rate)
                )
                heart_rate[destination] = ecg_rate
                heart_rate_valid[destination] = valid.astype(np.uint8)
        offset += expected_count

    age.flush()
    heart_rate.flush()
    heart_rate_valid.flush()
    del age, heart_rate, heart_rate_valid
    temporary_age.replace(args.dataset_root / "age.npy")
    temporary_hr.replace(args.dataset_root / "heart_rate.npy")
    temporary_valid.replace(args.dataset_root / "heart_rate_valid.npy")

    ages = np.load(args.dataset_root / "age.npy", mmap_mode="r")
    rates = np.load(args.dataset_root / "heart_rate.npy", mmap_mode="r")
    valid = np.load(args.dataset_root / "heart_rate_valid.npy", mmap_mode="r").astype(bool)
    metadata["auxiliary_labels"] = {
        "age": "official PulseDB Age metadata",
        "heart_rate": "ECG autocorrelation, checked against synchronized PPG periodicity",
        "heart_rate_valid_fraction": float(valid.mean()),
        "heart_rate_bpm_valid_percentiles": [
            float(value) for value in np.percentile(rates[valid], [0, 1, 50, 99, 100])
        ],
        "age_years_percentiles": [
            float(value) for value in np.percentile(ages[np.isfinite(ages)], [0, 1, 50, 99, 100])
        ],
    }
    temporary_meta = metadata_path.with_suffix(".building.json")
    temporary_meta.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    temporary_meta.replace(metadata_path)
    print(json.dumps(metadata["auxiliary_labels"], indent=2))


if __name__ == "__main__":
    main()
