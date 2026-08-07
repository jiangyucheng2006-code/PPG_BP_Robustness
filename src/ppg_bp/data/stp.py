"""Cycle-window extraction and manifest-backed datasets for STP."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Literal

import numpy as np
import pywt
import torch
from scipy.signal import (
    butter,
    filtfilt,
    find_peaks,
    firwin,
    resample,
    sosfiltfilt,
)
from torch.utils.data import Dataset


STP_SPLIT_TO_CODE = {"train": 0, "val": 1, "test": 2}


def repair_nonfinite(signal: np.ndarray) -> np.ndarray:
    """Interpolate isolated missing samples without hiding an empty record."""

    signal = np.asarray(signal, dtype=np.float64).reshape(-1)
    finite = np.isfinite(signal)
    if not finite.any():
        raise ValueError("Signal contains no finite samples")
    if not finite.all():
        indices = np.arange(len(signal))
        signal = np.interp(indices, indices[finite], signal[finite])
    return signal


def resample_to_125hz(
    signal: np.ndarray,
    sampling_rate: float,
    *,
    target_rate: float = 125.0,
) -> np.ndarray:
    """Resample a complete record to the 125 Hz rate used by STP."""

    signal = repair_nonfinite(signal)
    if np.isclose(sampling_rate, target_rate):
        return signal.astype(np.float32)
    samples = max(1, int(round(len(signal) * target_rate / sampling_rate)))
    return resample(signal, samples).astype(np.float32)


def wavelet_filter_ppg(
    signal: np.ndarray,
    *,
    wavelet: str = "db8",
    level: int = 9,
) -> np.ndarray:
    """Apply the nine-level db8 denoising procedure disclosed for STP.

    ``pywt.wavedec`` orders its coefficients as ``cA9, cD9, ..., cD1``.
    The paper/patent says to zero the first approximation coefficient and
    detail coefficient groups 7--9.  In this ordering those detail groups
    are ``cD3``, ``cD2`` and ``cD1``, i.e. the high-frequency bands.
    """

    signal = repair_nonfinite(signal)
    maximum_level = pywt.dwt_max_level(len(signal), pywt.Wavelet(wavelet).dec_len)
    if maximum_level < level:
        raise ValueError(
            f"A {level}-level {wavelet} decomposition needs a longer signal "
            f"(maximum level is {maximum_level})"
        )
    coefficients = pywt.wavedec(signal, wavelet, level=level, mode="symmetric")
    coefficients[0] = np.zeros_like(coefficients[0])
    for coefficient_index in (7, 8, 9):
        coefficients[coefficient_index] = np.zeros_like(
            coefficients[coefficient_index]
        )
    reconstructed = pywt.waverec(coefficients, wavelet, mode="symmetric")
    return reconstructed[: len(signal)].astype(np.float32)


def filter_abp_fir(
    signal: np.ndarray,
    sampling_rate: float = 125.0,
    *,
    low_hz: float = 0.5,
    high_hz: float = 35.0,
    taps: int = 401,
) -> np.ndarray:
    """Apply the disclosed 0.5--35 Hz FIR filter to invasive BP."""

    signal = repair_nonfinite(signal)
    high_hz = min(high_hz, sampling_rate * 0.49)
    coefficients = firwin(
        taps,
        [low_hz, high_hz],
        pass_zero=False,
        fs=sampling_rate,
    )
    return filtfilt(coefficients, [1.0], signal).astype(np.float32)


def is_flatline(
    signal: np.ndarray,
    *,
    tolerance: float = 1e-7,
    maximum_flat_fraction: float = 0.95,
) -> bool:
    """Detect the horizontal missing-signal segments described by STP."""

    signal = np.asarray(signal)
    if len(signal) < 2 or not np.isfinite(signal).all():
        return True
    return bool(np.mean(np.abs(np.diff(signal)) <= tolerance) >= maximum_flat_fraction)


def template_quality_mask(
    windows: np.ndarray,
    *,
    standard_deviations: float = 3.0,
) -> tuple[np.ndarray, dict[str, float]]:
    """Remove waveform outliers using an average template and 3-sigma rules."""

    windows = np.asarray(windows, dtype=np.float64)
    if windows.ndim != 2:
        raise ValueError("Expected [windows, samples]")
    if len(windows) < 3:
        return np.ones(len(windows), dtype=bool), {
            "difference_upper": float("inf"),
            "correlation_lower": -1.0,
        }
    template = windows.mean(axis=0)
    differences = np.mean(np.abs(windows - template[None]), axis=1)
    centered_windows = windows - windows.mean(axis=1, keepdims=True)
    centered_template = template - template.mean()
    denominator = (
        np.linalg.norm(centered_windows, axis=1)
        * max(float(np.linalg.norm(centered_template)), 1e-12)
    )
    correlations = (centered_windows @ centered_template) / np.maximum(
        denominator,
        1e-12,
    )
    difference_upper = float(
        differences.mean() + standard_deviations * differences.std()
    )
    correlation_lower = float(
        correlations.mean() - standard_deviations * correlations.std()
    )
    valid = (
        np.isfinite(differences)
        & np.isfinite(correlations)
        & (differences <= difference_upper)
        & (correlations >= correlation_lower)
    )
    return valid, {
        "difference_upper": difference_upper,
        "correlation_lower": correlation_lower,
        "difference_mean": float(differences.mean()),
        "correlation_mean": float(correlations.mean()),
    }


def normalize_ppg(signal: np.ndarray) -> np.ndarray:
    """Min-max normalization used by the paper's transformation figures."""

    signal = np.asarray(signal, dtype=np.float32)
    minimum = float(np.nanmin(signal))
    maximum = float(np.nanmax(signal))
    scale = max(maximum - minimum, 1e-6)
    return ((signal - minimum) / scale).astype(np.float32)


def fixed_length_windows(
    signal: np.ndarray,
    *,
    window_samples: int = 512,
    stride_samples: int | None = None,
    normalize_windows: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """Cut the self-supervised PPG source into fixed-length windows."""

    signal = repair_nonfinite(signal).astype(np.float32)
    if window_samples <= 0:
        raise ValueError("window_samples must be positive")
    stride = window_samples if stride_samples is None else int(stride_samples)
    if stride <= 0:
        raise ValueError("stride_samples must be positive")
    windows: list[np.ndarray] = []
    bounds: list[tuple[int, int]] = []
    for start in range(0, max(0, len(signal) - window_samples + 1), stride):
        stop = start + window_samples
        segment = signal[start:stop]
        if normalize_windows:
            segment = normalize_ppg(segment)
        windows.append(np.asarray(segment, dtype=np.float32))
        bounds.append((start, stop))
    if not windows:
        return (
            np.empty((0, window_samples), dtype=np.float32),
            np.empty((0, 2), dtype=np.int64),
        )
    return np.stack(windows), np.asarray(bounds, dtype=np.int64)


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
    overlap_cycles: int = 2,
    output_samples: int = 512,
    filter_method: str = "butter",
    normalize_windows: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """Extract five-cycle windows with a two-cycle overlap.

    Returns the resampled windows and their original ``[start, stop)`` sample
    locations.  The cycle/stride values are directly confirmed by Fig. 1 of
    the paper.
    """

    if cycles <= 0 or not 0 <= overlap_cycles < cycles:
        raise ValueError("overlap_cycles must satisfy 0 <= overlap < cycles")
    if filter_method == "butter":
        filtered = filter_ppg(signal, sampling_rate)
    elif filter_method == "stp_db8":
        filtered = wavelet_filter_ppg(signal)
    elif filter_method == "none":
        filtered = repair_nonfinite(signal).astype(np.float32)
    else:
        raise ValueError(f"Unknown PPG filter method: {filter_method}")
    prominence = max(float(np.ptp(filtered)) * 0.08, 1e-6)
    peaks, _ = find_peaks(
        filtered,
        distance=max(1, int(0.35 * sampling_rate)),
        prominence=prominence,
    )
    windows: list[np.ndarray] = []
    bounds: list[tuple[int, int]] = []
    step_cycles = cycles - overlap_cycles
    for peak_index in range(0, max(0, len(peaks) - cycles), step_cycles):
        start = int(peaks[peak_index])
        stop = int(peaks[peak_index + cycles])
        if stop - start < max(8, int(1.5 * sampling_rate)):
            continue
        segment = resample(filtered[start:stop], output_samples).astype(np.float32)
        if normalize_windows:
            segment = normalize_ppg(segment)
        windows.append(segment)
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
    hypertensive_sbp: float = 130.0,
    hypertensive_dbp: float = 80.0,
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
        roles: tuple[str, ...] = (),
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
        requested_roles = {role.lower() for role in roles}
        self.subjects: list[dict] = []
        self.index: list[tuple[int, int]] = []
        for entry in self.manifest["subjects"]:
            if entry["split"] != split:
                continue
            if requested_sources and entry["source"].lower() not in requested_sources:
                continue
            role = str(entry.get("role", "")).lower()
            if requested_roles and role not in requested_roles:
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

    def target_values(self, key: Literal["labels", "patterns"]) -> np.ndarray:
        """Return targets for exactly the samples selected by this dataset."""

        rows_by_subject: dict[int, list[int]] = {}
        for subject_index, row in self.index:
            rows_by_subject.setdefault(subject_index, []).append(row)
        values = [
            np.asarray(self._array(subject_index, key))[rows]
            for subject_index, rows in rows_by_subject.items()
        ]
        return np.concatenate(values)

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
