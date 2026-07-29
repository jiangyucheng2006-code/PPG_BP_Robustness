"""Cycle-window extraction and manifest-backed datasets for STP."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Literal

import numpy as np
import torch
from scipy.signal import butter, find_peaks, resample, sosfiltfilt
from torch.utils.data import Dataset


STP_SPLIT_TO_CODE = {"train": 0, "val": 1, "test": 2}


def normalize_ppg(signal: np.ndarray) -> np.ndarray:
    """Min-max normalization used by the paper's transformation figures."""

    signal = np.asarray(signal, dtype=np.float32)
    minimum = float(np.nanmin(signal))
    maximum = float(np.nanmax(signal))
    scale = max(maximum - minimum, 1e-6)
    return ((signal - minimum) / scale).astype(np.float32)


def filter_ppg(
    signal: np.ndarray,
    sampling_rate: float,
    *,
    low_hz: float = 0.5,
    high_hz: float = 8.0,
    order: int = 4,
) -> np.ndarray:
    """Remove baseline and high-frequency noise before cycle extraction."""

    signal = np.asarray(signal, dtype=np.float64)
    finite = np.isfinite(signal)
    if not finite.all():
        fill = float(np.nanmedian(signal[finite])) if finite.any() else 0.0
        signal = np.where(finite, signal, fill)
    median = float(np.median(signal))
    mad = max(float(np.median(np.abs(signal - median))), 1e-6)
    signal = np.clip(signal, median - 8 * mad, median + 8 * mad)
    nyquist = sampling_rate / 2
    high_hz = min(high_hz, nyquist * 0.95)
    sos = butter(
        order,
        (low_hz / nyquist, high_hz / nyquist),
        btype="bandpass",
        output="sos",
    )
    return sosfiltfilt(sos, signal).astype(np.float32)


def cycle_windows(
    signal: np.ndarray,
    sampling_rate: float,
    *,
    cycles: int = 5,
    stride_cycles: int = 2,
    output_samples: int = 512,
) -> tuple[np.ndarray, np.ndarray]:
    """Extract five-cycle windows with a two-cycle stride.

    Returns the resampled windows and their original ``[start, stop)`` sample
    locations.  The cycle/stride values are directly confirmed by Fig. 1 of
    the paper.
    """

    filtered = filter_ppg(signal, sampling_rate)
    prominence = max(float(np.ptp(filtered)) * 0.08, 1e-6)
    peaks, _ = find_peaks(
        filtered,
        distance=max(1, int(0.35 * sampling_rate)),
        prominence=prominence,
    )
    windows: list[np.ndarray] = []
    bounds: list[tuple[int, int]] = []
    for peak_index in range(0, max(0, len(peaks) - cycles), stride_cycles):
        start = int(peaks[peak_index])
        stop = int(peaks[peak_index + cycles])
        if stop - start < max(8, int(1.5 * sampling_rate)):
            continue
        segment = normalize_ppg(filtered[start:stop])
        windows.append(resample(segment, output_samples).astype(np.float32))
        bounds.append((start, stop))
    if not windows:
        return (
            np.empty((0, output_samples), dtype=np.float32),
            np.empty((0, 2), dtype=np.int64),
        )
    return np.stack(windows), np.asarray(bounds, dtype=np.int64)


def bp_pattern_labels(
    labels: np.ndarray,
    *,
    hypotensive_sbp: float = 90.0,
    hypotensive_dbp: float = 60.0,
    hypertensive_sbp: float = 140.0,
    hypertensive_dbp: float = 90.0,
) -> np.ndarray:
    """Map SBP/DBP to hypo-, normo-, and hypertensive classes."""

    labels = np.asarray(labels)
    patterns = np.ones(len(labels), dtype=np.int64)
    patterns[
        (labels[:, 0] < hypotensive_sbp)
        | (labels[:, 1] < hypotensive_dbp)
    ] = 0
    patterns[
        (labels[:, 0] >= hypertensive_sbp)
        | (labels[:, 1] >= hypertensive_dbp)
    ] = 2
    return patterns


class STPWindowDataset(Dataset):
    """Read public STP windows from a subject-wise manifest."""

    def __init__(
        self,
        root: str | Path,
        split: Literal["train", "val", "test"],
        stage: Literal["pretrain", "pattern", "bp"],
        *,
        sources: tuple[str, ...] = (),
        maximum_samples: int | None = None,
    ) -> None:
        if split not in STP_SPLIT_TO_CODE:
            raise ValueError(f"Unknown split: {split}")
        if stage not in {"pretrain", "pattern", "bp"}:
            raise ValueError(f"Unknown STP stage: {stage}")
        self.root = Path(root)
        self.stage = stage
        self.manifest = json.loads(
            (self.root / "manifest.json").read_text(encoding="utf-8")
        )
        requested_sources = {source.lower() for source in sources}
        self.subjects: list[dict] = []
        self.index: list[tuple[int, int]] = []
        for entry in self.manifest["subjects"]:
            if entry["split"] != split:
                continue
            if requested_sources and entry["source"].lower() not in requested_sources:
                continue
            if stage in {"pattern", "bp"} and not entry.get("labels"):
                continue
            subject_index = len(self.subjects)
            self.subjects.append(entry)
            for row in range(int(entry["windows"])):
                self.index.append((subject_index, row))
                if maximum_samples is not None and len(self.index) >= maximum_samples:
                    break
            if maximum_samples is not None and len(self.index) >= maximum_samples:
                break
        if not self.index:
            raise ValueError(
                f"No {stage} samples found for split={split!r}, sources={sources!r}"
            )
        self._signals: dict[int, np.ndarray] = {}
        self._labels: dict[int, np.ndarray] = {}
        self._patterns: dict[int, np.ndarray] = {}

    def __len__(self) -> int:
        return len(self.index)

    def _array(self, subject_index: int, key: str) -> np.ndarray:
        caches = {
            "signals": self._signals,
            "labels": self._labels,
            "patterns": self._patterns,
        }
        cache = caches[key]
        if subject_index not in cache:
            relative = self.subjects[subject_index].get(key)
            if not relative:
                raise KeyError(f"Subject does not contain {key}")
            cache[subject_index] = np.load(
                self.root / relative,
                mmap_mode="r",
            )
        return cache[subject_index]

    def __getitem__(self, item: int):
        subject_index, row = self.index[item]
        signal = np.asarray(
            self._array(subject_index, "signals")[row],
            dtype=np.float32,
        ).copy()
        signal_tensor = torch.from_numpy(signal[None])
        if self.stage == "pretrain":
            return signal_tensor
        if self.stage == "pattern":
            pattern = int(self._array(subject_index, "patterns")[row])
            return signal_tensor, torch.tensor(pattern, dtype=torch.long)
        label = np.asarray(
            self._array(subject_index, "labels")[row],
            dtype=np.float32,
        ).copy()
        return signal_tensor, torch.from_numpy(label)
