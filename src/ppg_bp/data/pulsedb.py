"""PulseDB readers and a memory-mapped training dataset.

PulseDB v2 subject files are MATLAB v7.3/HDF5 files. Each field in
``Subj_Wins`` contains object references, one reference per 10-second segment.
The preparation script converts those references to contiguous NumPy arrays so
that training does not repeatedly parse MATLAB files.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset


SPLIT_TO_CODE = {"train": 0, "val": 1, "test": 2}


@dataclass(frozen=True)
class SubjectSegments:
    subject_id: str
    ppg: np.ndarray
    labels: np.ndarray
    include: np.ndarray
    correlation: np.ndarray


def _is_reference_dataset(dataset: h5py.Dataset) -> bool:
    return h5py.check_dtype(ref=dataset.dtype) is not None


def _segment_count(dataset: h5py.Dataset) -> int:
    if _is_reference_dataset(dataset):
        return int(dataset.size)
    shape = dataset.shape
    if dataset.ndim == 2 and 1250 in shape:
        return int(shape[1] if shape[0] == 1250 else shape[0])
    return 1


def _read_reference(
    handle: h5py.File,
    reference_dataset: h5py.Dataset,
    index: int,
    count: int,
) -> np.ndarray:
    if _is_reference_dataset(reference_dataset):
        reference = np.asarray(reference_dataset[()]).reshape(-1)[index]
        return np.asarray(handle[reference][()])

    values = np.asarray(reference_dataset[()])
    if count == 1:
        return values
    if values.ndim >= 2 and values.shape[0] == count:
        return np.asarray(values[index])
    if values.ndim >= 2 and values.shape[1] == count:
        return np.asarray(values[:, index])
    return np.asarray(values).reshape(count, -1)[index]


def _decode_matlab_text(values: np.ndarray) -> str:
    flat = np.asarray(values).reshape(-1)
    if np.issubdtype(flat.dtype, np.integer):
        return "".join(chr(int(value)) for value in flat if int(value) != 0)
    return str(flat[0])


def read_pulsedb_subject(
    path: str | Path,
    *,
    ppg_field: str = "PPG_F",
    include_only: bool = True,
    max_segments: int | None = None,
) -> SubjectSegments:
    """Read one official PulseDB v2 subject file.

    Parameters
    ----------
    path:
        Path to one ``pXXXXXX.mat`` or VitalDB subject file.
    ppg_field:
        ``PPG_F`` reproduces the normalized, filtered signal used by the
        original PulseDB subsets. ``PPG_Record_F`` retains absolute amplitude.
    include_only:
        Keep only segments marked by PulseDB's ``IncludeFlag``.
    """

    path = Path(path)
    with h5py.File(path, "r") as handle:
        group = handle["Subj_Wins"]
        if ppg_field not in group:
            raise KeyError(f"{ppg_field!r} is not present in {path.name}")

        count = _segment_count(group[ppg_field])
        ppg: list[np.ndarray] = []
        labels: list[tuple[float, float]] = []
        include: list[bool] = []
        correlation: list[float] = []
        subject_id = path.stem

        for index in range(count):
            flag = bool(_read_reference(handle, group["IncludeFlag"], index, count).reshape(-1)[0])
            if include_only and not flag:
                continue

            signal = _read_reference(handle, group[ppg_field], index, count).astype(np.float32).reshape(-1)
            sbp = float(_read_reference(handle, group["SegSBP"], index, count).reshape(-1)[0])
            dbp = float(_read_reference(handle, group["SegDBP"], index, count).reshape(-1)[0])
            corr = float(_read_reference(handle, group["PPG_ABP_Corr"], index, count).reshape(-1)[0])

            if index == 0 and "SubjectID" in group:
                subject_id = _decode_matlab_text(_read_reference(handle, group["SubjectID"], index, count))

            ppg.append(signal)
            labels.append((sbp, dbp))
            include.append(flag)
            correlation.append(corr)
            if max_segments is not None and len(ppg) >= max_segments:
                break

    if not ppg:
        raise ValueError(f"No usable PulseDB segments found in {path}")

    return SubjectSegments(
        subject_id=subject_id,
        ppg=np.stack(ppg),
        labels=np.asarray(labels, dtype=np.float32),
        include=np.asarray(include, dtype=bool),
        correlation=np.asarray(correlation, dtype=np.float32),
    )


class PulseDBMemmapDataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    """Model-ready PulseDB arrays produced by ``prepare_pulsedb.py``."""

    def __init__(self, root: str | Path, split: str, normalization: str = "per_segment_zscore") -> None:
        if split not in SPLIT_TO_CODE:
            raise ValueError(f"Unknown split {split!r}; expected train, val, or test")
        if normalization not in {"none", "per_segment_zscore"}:
            raise ValueError(f"Unsupported normalization: {normalization}")

        self.root = Path(root)
        with (self.root / "dataset_meta.json").open("r", encoding="utf-8") as stream:
            self.meta: dict[str, Any] = json.load(stream)

        count = int(self.meta["n_samples"])
        self.ppg = np.load(self.root / "ppg.npy", mmap_mode="r")
        self.labels = np.load(self.root / "labels.npy", mmap_mode="r")
        split_codes = np.load(self.root / "split.npy", mmap_mode="r")
        self.indices = np.flatnonzero(split_codes[:count] == SPLIT_TO_CODE[split])
        self.normalization = normalization

        if self.ppg.shape[1] != int(self.meta["window_samples"]):
            raise ValueError("PPG array shape does not match dataset_meta.json")

    def __len__(self) -> int:
        return int(self.indices.size)

    def __getitem__(self, item: int) -> tuple[torch.Tensor, torch.Tensor]:
        index = int(self.indices[item])
        signal = np.asarray(self.ppg[index], dtype=np.float32).copy()
        target = np.asarray(self.labels[index], dtype=np.float32).copy()
        if self.normalization == "per_segment_zscore":
            signal = (signal - signal.mean()) / max(float(signal.std()), 1e-6)
        return torch.from_numpy(signal[None, :]), torch.from_numpy(target)
