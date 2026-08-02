"""Prepare the public-data portion of the STP experiment.

The original study combines WESAD and PPG-DaLiA as unpaired PPG sources with
paired PPG/ABP records.  This script prepares those sources using the
paper-confirmed five-cycle window and two-cycle stride.  The private Mindray
cohort is intentionally not required.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
import pickle
import re
import time
import zipfile
from pathlib import Path

import numpy as np
import requests
import wfdb
from scipy.signal import correlate, correlation_lags, find_peaks
from tqdm import tqdm

from ppg_bp.data.stp import (
    bp_pattern_labels,
    cycle_windows,
    filter_abp_fir,
    is_flatline,
    resample_to_125hz,
    template_quality_mask,
    wavelet_filter_ppg,
)


WESAD_URL = "https://uni-siegen.sciebo.de/s/HGdUkoNlW1Ub0Gx/download"
PPG_DALIA_URL = "https://archive.ics.uci.edu/static/public/495/ppg+dalia.zip"
MIMIC_PAIRED_INDEX_URL = (
    "https://raw.githubusercontent.com/v3551G/BP-prediction-survey/"
    "main/MIMIC-III_ppg_dataset_records.txt"
)


def progress_download(
    url: str,
    destination: Path,
    *,
    retries: int = 12,
) -> None:
    """Download a large public file with retry and HTTP range resumption."""

    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        return
    temporary = destination.with_suffix(destination.suffix + ".part")
    for attempt in range(1, retries + 1):
        offset = temporary.stat().st_size if temporary.exists() else 0
        headers = {"Range": f"bytes={offset}-"} if offset else {}
        try:
            with requests.get(
                url,
                headers=headers,
                stream=True,
                timeout=(30, 90),
            ) as response:
                response.raise_for_status()
                resumed = offset > 0 and response.status_code == 206
                if offset and not resumed:
                    offset = 0
                content_range = response.headers.get("Content-Range", "")
                total = None
                if "/" in content_range:
                    value = content_range.rsplit("/", 1)[-1]
                    if value.isdigit():
                        total = int(value)
                if total is None:
                    remaining = response.headers.get("Content-Length")
                    if remaining and remaining.isdigit():
                        total = offset + int(remaining)
                mode = "ab" if resumed else "wb"
                with temporary.open(mode) as stream, tqdm(
                    total=total,
                    initial=offset,
                    unit="B",
                    unit_scale=True,
                    unit_divisor=1024,
                    desc=f"Downloading {destination.name}",
                ) as progress:
                    for chunk in response.iter_content(chunk_size=1024 * 1024):
                        if not chunk:
                            continue
                        stream.write(chunk)
                        progress.update(len(chunk))
            if total is not None and temporary.stat().st_size < total:
                raise IOError(
                    f"incomplete file: {temporary.stat().st_size}/{total} bytes"
                )
            temporary.replace(destination)
            return
        except (OSError, requests.RequestException) as error:
            if attempt >= retries:
                raise
            delay = min(60, 5 * attempt)
            print(
                f"Download interrupted ({error}); retaining "
                f"{temporary.stat().st_size / 2**30:.2f} GB and retrying "
                f"in {delay} s ({attempt}/{retries})."
            )
            time.sleep(delay)


def extract_archive(archive: Path, destination: Path) -> None:
    marker = destination / ".complete"
    nested_archives = list(destination.glob("*.zip")) if destination.exists() else []
    if marker.exists() and not nested_archives:
        return
    destination.mkdir(parents=True, exist_ok=True)
    if not marker.exists():
        with zipfile.ZipFile(archive) as stream:
            stream.extractall(destination)
    # UCI's PPG-DaLiA download is a ZIP containing another data.zip. Extract
    # such top-level archives as well so the subject pickle files are visible.
    for nested_archive in destination.glob("*.zip"):
        nested_marker = destination / f".{nested_archive.stem}.complete"
        if nested_marker.exists():
            continue
        with zipfile.ZipFile(nested_archive) as stream:
            stream.extractall(destination)
        nested_marker.touch()
    marker.touch()


def split_for_subject(subject_id: str, seed: int) -> str:
    digest = hashlib.sha256(f"{seed}:{subject_id}".encode()).digest()
    value = int.from_bytes(digest[:4], "big") % 100
    if value < 70:
        return "train"
    if value < 85:
        return "val"
    return "test"


def balanced_subject_splits(
    subject_ids: list[str],
    seed: int,
) -> dict[str, str]:
    """Create a deterministic 70/15/15 subject-wise split."""

    ordered = sorted(set(subject_ids))
    random = np.random.default_rng(seed)
    random.shuffle(ordered)
    train_count = int(round(0.70 * len(ordered)))
    validation_count = int(round(0.15 * len(ordered)))
    mapping: dict[str, str] = {}
    for index, subject_id in enumerate(ordered):
        if index < train_count:
            split = "train"
        elif index < train_count + validation_count:
            split = "val"
        else:
            split = "test"
        mapping[subject_id] = split
    return mapping


def read_bvp_pickle(path: Path) -> np.ndarray:
    with path.open("rb") as stream:
        data = pickle.load(stream, encoding="latin1")
    try:
        bvp = data["signal"]["wrist"]["BVP"]
    except KeyError as error:
        raise KeyError(f"Could not find signal/wrist/BVP in {path}") from error
    return np.asarray(bvp, dtype=np.float32).reshape(-1)


def save_subject(
    output: Path,
    *,
    subject_id: str,
    source: str,
    split: str,
    role: str,
    signals: np.ndarray,
    labels: np.ndarray | None = None,
    patterns: np.ndarray | None = None,
    metadata: dict | None = None,
) -> dict:
    safe_id = re.sub(r"[^A-Za-z0-9_.-]+", "_", subject_id)
    relative_directory = Path("subjects") / source.lower()
    directory = output / relative_directory
    directory.mkdir(parents=True, exist_ok=True)
    signal_relative = relative_directory / f"{safe_id}_signals.npy"
    np.save(output / signal_relative, signals.astype(np.float32, copy=False))
    entry = {
        "subject_id": subject_id,
        "source": source,
        "split": split,
        "role": role,
        "windows": int(len(signals)),
        "signals": signal_relative.as_posix(),
        "labels": None,
        "patterns": None,
        "metadata": metadata or {},
    }
    if labels is not None:
        label_relative = relative_directory / f"{safe_id}_labels.npy"
        np.save(output / label_relative, labels.astype(np.float32, copy=False))
        entry["labels"] = label_relative.as_posix()
    if patterns is not None:
        pattern_relative = relative_directory / f"{safe_id}_patterns.npy"
        np.save(output / pattern_relative, patterns.astype(np.int64, copy=False))
        entry["patterns"] = pattern_relative.as_posix()
    return entry


def prepare_unpaired_source(
    root: Path,
    output: Path,
    *,
    source: str,
    seed: int,
    output_samples: int,
    maximum_windows_per_subject: int | None,
) -> list[dict]:
    subject_files = sorted(root.rglob("S*.pkl"))
    if not subject_files:
        raise FileNotFoundError(f"No subject pickle files found below {root}")
    valid_files = [
        path for path in subject_files if re.fullmatch(r"S(\d+)", path.stem)
    ]
    subject_ids = [f"{source}_{path.stem}" for path in valid_files]
    split_map = balanced_subject_splits(subject_ids, seed)
    entries: list[dict] = []
    for path, subject_id in tqdm(
        zip(valid_files, subject_ids),
        total=len(valid_files),
        desc=f"Preparing {source}",
    ):
        match = re.fullmatch(r"S(\d+)", path.stem)
        assert match is not None
        resampled = resample_to_125hz(read_bvp_pickle(path), 64.0)
        windows, _ = cycle_windows(
            resampled,
            125.0,
            cycles=5,
            stride_cycles=2,
            output_samples=output_samples,
            filter_method="stp_db8",
        )
        if maximum_windows_per_subject is not None:
            windows = windows[:maximum_windows_per_subject]
        if not len(windows):
            continue
        entries.append(
            save_subject(
                output,
                subject_id=subject_id,
                source=source,
                split=split_map[subject_id],
                role="pretrain_unpaired",
                signals=windows,
                metadata={
                    "input_sampling_rate": 64.0,
                    "processed_sampling_rate": 125.0,
                    "ppg_filter": "db8 level 9; cA9 and cD3-cD1 zeroed",
                    "paired_bp": False,
                    "source_file": str(path),
                },
            )
        )
    return entries


def channel_index(names: list[str], candidates: tuple[str, ...]) -> int | None:
    upper = [name.upper() for name in names]
    for candidate in candidates:
        if candidate in upper:
            return upper.index(candidate)
    for index, name in enumerate(upper):
        if any(candidate in name for candidate in candidates):
            return index
    return None


def estimate_phase_lag(
    ppg: np.ndarray,
    abp: np.ndarray,
    sampling_rate: float,
    maximum_delay_seconds: float = 0.5,
) -> int:
    limit = min(len(ppg), int(60 * sampling_rate))
    first = ppg[:limit]
    second = abp[:limit]
    first = (first - np.mean(first)) / max(float(np.std(first)), 1e-6)
    second = (second - np.mean(second)) / max(float(np.std(second)), 1e-6)
    values = correlate(first, second, mode="full", method="fft")
    lags = correlation_lags(len(first), len(second), mode="full")
    permitted = np.abs(lags) <= int(maximum_delay_seconds * sampling_rate)
    return int(lags[permitted][np.argmax(values[permitted])])


def align_paired_signals(
    ppg: np.ndarray,
    abp: np.ndarray,
    sampling_rate: float,
) -> tuple[np.ndarray, np.ndarray, int]:
    """Align paired signals by maximum cross-correlation without wraparound."""

    lag = estimate_phase_lag(ppg, abp, sampling_rate)
    if lag > 0:
        return ppg[lag:], abp[:-lag], lag
    if lag < 0:
        return ppg[:lag], abp[-lag:], lag
    return ppg, abp, lag


def apply_phase_lag(
    ppg: np.ndarray,
    paired_signal: np.ndarray,
    lag: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Apply a previously estimated lag to any paired reference signal."""

    if lag > 0:
        return ppg[lag:], paired_signal[:-lag]
    if lag < 0:
        return ppg[:lag], paired_signal[-lag:]
    return ppg, paired_signal


def bp_from_abp_window(
    abp: np.ndarray,
    sampling_rate: float,
) -> tuple[float, float]:
    distance = max(1, int(0.35 * sampling_rate))
    systolic_indices, _ = find_peaks(abp, distance=distance)
    diastolic_indices, _ = find_peaks(-abp, distance=distance)
    if len(systolic_indices) >= 3 and len(diastolic_indices) >= 3:
        sbp = float(np.median(abp[systolic_indices]))
        dbp = float(np.median(abp[diastolic_indices]))
    else:
        sbp = float(np.percentile(abp, 95))
        dbp = float(np.percentile(abp, 5))
    return sbp, dbp


def subject_from_record(record: str) -> str:
    matches = re.findall(r"p\d{6}", record.lower())
    return matches[-1] if matches else Path(record).name


def public_mimic_records(
    pn_dir: str,
    maximum_subjects: int,
    workers: int,
    *,
    require_abp: bool = True,
    excluded_subjects: set[str] | None = None,
) -> list[tuple[str, str]]:
    """Find one usable waveform record per public MIMIC subject."""

    base_url = f"https://physionet.org/files/{pn_dir.rstrip('/')}/"
    response = requests.get(base_url + "RECORDS-waveforms", timeout=60)
    response.raise_for_status()
    records_by_subject: dict[str, list[str]] = {}
    for value in response.text.splitlines():
        value = value.strip()
        if not value:
            continue
        subject = subject_from_record(value)
        records_by_subject.setdefault(subject, []).append(value)
    excluded_subjects = excluded_subjects or set()

    def fetch_header(relative: str) -> list[str] | None:
        url = base_url + relative.strip("/") + ".hea"
        for attempt in range(2):
            try:
                value = requests.get(url, timeout=(5, 15))
                value.raise_for_status()
                return [
                    line.strip()
                    for line in value.text.splitlines()
                    if line.strip() and not line.lstrip().startswith("#")
                ]
            except requests.RequestException:
                if attempt:
                    return None
        return None

    def inspect_subject(item: tuple[str, list[str]]) -> tuple[str, str] | None:
        subject, master_records = item
        if subject in excluded_subjects:
            return None
        for master_value in master_records:
            folder, master_record = master_value.rsplit("/", 1)
            subject_pn_dir = f"{pn_dir.rstrip('/')}/{folder}"
            master_lines = fetch_header(master_value)
            if not master_lines:
                continue
            first_fields = master_lines[0].split()
            try:
                sampling_rate = float(first_fields[2].split("/", 1)[0])
            except (IndexError, ValueError):
                sampling_rate = 125.0
            segments: list[tuple[str, int]] = []
            for line in master_lines[1:]:
                fields = line.split()
                if len(fields) < 2:
                    continue
                try:
                    segments.append((fields[0], int(fields[1])))
                except ValueError:
                    continue
            segment_names = [name for name, _ in segments]
            layout_names: list[str] = []
            for layout in segment_names:
                if not layout.endswith("_layout"):
                    continue
                layout_lines = fetch_header(f"{folder}/{layout}")
                if not layout_lines:
                    continue
                layout_names.extend(
                    line.split()[-1] for line in layout_lines[1:] if line.split()
                )
            layout_has_ppg = (
                channel_index(layout_names, ("PLETH", "PPG")) is not None
            )
            layout_has_abp = (
                channel_index(layout_names, ("ABP", "ART", "ART1")) is not None
            )
            if not layout_has_ppg:
                continue
            if require_abp != layout_has_abp:
                continue
            minimum_samples = int(10 * 60 * sampling_rate)
            candidates = sorted(
                (
                    (candidate, length)
                    for candidate, length in segments
                    if candidate != "~" and not candidate.endswith("_layout")
                ),
                key=lambda item: item[1],
                reverse=True,
            )
            for candidate, length in candidates:
                if candidate == "~" or candidate.endswith("_layout"):
                    continue
                if length < minimum_samples:
                    continue
                candidate_lines = fetch_header(f"{folder}/{candidate}")
                if not candidate_lines:
                    continue
                names = [
                    line.split()[-1]
                    for line in candidate_lines[1:]
                    if line.split()
                ]
                has_ppg = channel_index(names, ("PLETH", "PPG")) is not None
                has_abp = channel_index(names, ("ABP", "ART", "ART1")) is not None
                if (
                    has_ppg
                    and (has_abp if require_abp else not has_abp)
                ):
                    return candidate, subject_pn_dir
        return None

    selected: list[tuple[str, str]] = []
    subject_items = sorted(records_by_subject.items())
    batch_size = max(workers * 4, 64)
    progress = tqdm(
        total=len(subject_items),
        desc="Indexing MIMIC-III",
    )
    for start in range(0, len(subject_items), batch_size):
        batch = subject_items[start : start + batch_size]
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = [
                executor.submit(inspect_subject, directory)
                for directory in batch
            ]
            for future in as_completed(futures):
                result = future.result()
                progress.update(1)
                if result is not None:
                    selected.append(result)
        if len(selected) >= maximum_subjects:
            break
    progress.close()
    selected.sort(key=lambda pair: pair[1])
    selected = selected[:maximum_subjects]
    return selected


def recover_prepared_entries(
    output: Path,
    *,
    source: str,
    role: str,
    seed: int,
    maximum_subjects: int | None = None,
) -> list[dict]:
    """Recover completed arrays after an interrupted preparation run."""

    manifest_path = output / "manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        matching = [
            entry
            for entry in manifest.get("subjects", ())
            if entry.get("source") == source and entry.get("role") == role
        ]
        matching.sort(key=lambda entry: str(entry.get("subject_id", "")))
        if maximum_subjects is not None:
            matching = matching[:maximum_subjects]
        complete = [
            entry
            for entry in matching
            if entry.get("signals")
            and (output / entry["signals"]).exists()
            and (
                not entry.get("labels")
                or (output / entry["labels"]).exists()
            )
            and (
                not entry.get("patterns")
                or (output / entry["patterns"]).exists()
            )
        ]
        if len(complete) == len(matching) and complete:
            return complete

    directory = output / "subjects" / source.lower()
    signal_files = sorted(directory.glob("*_signals.npy"))
    if maximum_subjects is not None:
        signal_files = signal_files[:maximum_subjects]
    subject_ids = [
        path.name.removesuffix("_signals.npy") for path in signal_files
    ]
    split_map = balanced_subject_splits(subject_ids, seed)
    entries: list[dict] = []
    for signal_path, subject_id in zip(signal_files, subject_ids):
        relative_directory = Path("subjects") / source.lower()
        label_path = directory / f"{subject_id}_labels.npy"
        pattern_path = directory / f"{subject_id}_patterns.npy"
        signals = np.load(signal_path, mmap_mode="r")
        entries.append(
            {
                "subject_id": subject_id,
                "source": source,
                "split": split_map[subject_id],
                "role": role,
                "windows": int(len(signals)),
                "signals": (relative_directory / signal_path.name).as_posix(),
                "labels": (
                    (relative_directory / label_path.name).as_posix()
                    if label_path.exists()
                    else None
                ),
                "patterns": (
                    (relative_directory / pattern_path.name).as_posix()
                    if pattern_path.exists()
                    else None
                ),
                "metadata": {
                    "recovered_after_interrupted_run": True,
                    "processing_protocol": "STP disclosed public preprocessing",
                },
            }
        )
    return entries


def local_mimic_records(
    root: Path,
    maximum_subjects: int,
) -> list[tuple[str, None]]:
    selected: list[tuple[str, None]] = []
    subjects: set[str] = set()
    for header in sorted(root.rglob("*.hea")):
        record = str(header.with_suffix(""))
        subject = subject_from_record(record)
        if subject in subjects:
            continue
        subjects.add(subject)
        selected.append((record, None))
        if len(subjects) >= maximum_subjects:
            break
    return selected


def indexed_mimic_records(
    path: Path,
    *,
    pn_dir: str,
    maximum_subjects: int,
    seed: int,
) -> list[tuple[str, str]]:
    """Choose one paired waveform segment per MIMIC record group."""

    groups: dict[str, tuple[str, str]] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        value = line.strip()
        if not value or value.startswith("#") or "/" not in value:
            continue
        intermediate, segment = value.split("/", 1)
        if "_" not in segment:
            continue
        record_group = segment.split("_", 1)[0]
        groups.setdefault(
            record_group,
            (
                segment,
                f"{pn_dir.rstrip('/')}/{intermediate}/{record_group}",
            ),
        )
    values = list(groups.values())
    random = np.random.default_rng(seed)
    random.shuffle(values)
    return values[:maximum_subjects]


def prepare_mimic(
    output: Path,
    *,
    records: list[tuple[str, str | None]],
    seed: int,
    output_samples: int,
    minutes_per_record: float,
    maximum_windows_per_subject: int | None,
    maximum_subjects: int | None = None,
) -> list[dict]:
    entries: list[dict] = []
    subject_ids = [
        f"MIMICIII_{Path(record).name.split('_', 1)[0]}"
        for record, _ in records
    ]
    split_map = balanced_subject_splits(subject_ids, seed)
    for record, record_pn_dir in tqdm(records, desc="Preparing MIMIC-III"):
        try:
            header = wfdb.rdheader(record, pn_dir=record_pn_dir)
            names = list(header.sig_name)
            ppg_index = channel_index(names, ("PLETH", "PPG"))
            abp_index = channel_index(names, ("ABP", "ART", "ART1"))
            if ppg_index is None or abp_index is None:
                continue
            sampling_rate = float(header.fs)
            stop = min(
                int(header.sig_len),
                int(minutes_per_record * 60 * sampling_rate),
            )
            data = wfdb.rdrecord(
                record,
                sampfrom=0,
                sampto=stop,
                channels=[ppg_index, abp_index],
                pn_dir=record_pn_dir,
            ).p_signal
        except Exception as error:
            print(f"\nSkipping {record}: {error}")
            continue
        if data is None:
            continue
        finite = np.isfinite(data).all(axis=1)
        if finite.sum() < 10 * sampling_rate:
            continue
        ppg = np.interp(
            np.arange(len(data)),
            np.flatnonzero(finite),
            data[finite, 0],
        ).astype(np.float32)
        abp = np.interp(
            np.arange(len(data)),
            np.flatnonzero(finite),
            data[finite, 1],
        ).astype(np.float32)
        ppg = wavelet_filter_ppg(resample_to_125hz(ppg, sampling_rate))
        abp_reference = resample_to_125hz(abp, sampling_rate)
        abp_filtered = filter_abp_fir(abp_reference, 125.0)
        sampling_rate = 125.0
        lag = estimate_phase_lag(
            ppg,
            abp_filtered,
            sampling_rate,
        )
        ppg, aligned_abp = apply_phase_lag(ppg, abp_reference, lag)
        windows, bounds = cycle_windows(
            ppg,
            sampling_rate,
            cycles=5,
            stride_cycles=2,
            output_samples=output_samples,
            filter_method="none",
        )
        labels = np.asarray(
            [
                bp_from_abp_window(aligned_abp[start:stop], sampling_rate)
                for start, stop in bounds
            ],
            dtype=np.float32,
        )
        not_flat = np.asarray(
            [not is_flatline(ppg[start:stop]) for start, stop in bounds],
            dtype=bool,
        )
        valid = (
            np.isfinite(labels).all(axis=1)
            & (labels[:, 1] >= 25)
            & (labels[:, 0] > labels[:, 1])
            & not_flat
        )
        windows = windows[valid]
        labels = labels[valid]
        template_valid, template_statistics = template_quality_mask(windows)
        windows = windows[template_valid]
        labels = labels[template_valid]
        if maximum_windows_per_subject is not None:
            windows = windows[:maximum_windows_per_subject]
            labels = labels[:maximum_windows_per_subject]
        if not len(windows):
            continue
        mimic_group = Path(record).name.split("_", 1)[0]
        subject_id = f"MIMICIII_{mimic_group}"
        entries.append(
            save_subject(
                output,
                subject_id=subject_id,
                source="MIMICIII",
                split=split_map[subject_id],
                role="supervised_paired",
                signals=windows,
                labels=labels,
                patterns=bp_pattern_labels(labels),
                metadata={
                    "input_sampling_rate": float(header.fs),
                    "processed_sampling_rate": 125.0,
                    "paired_bp": True,
                    "record": record,
                    "pn_dir": record_pn_dir,
                    "phase_lag_samples": lag,
                    "maximum_alignment_delay_ms": 500,
                    "ppg_filter": "db8 level 9; cA9 and cD3-cD1 zeroed",
                    "abp_filter": "FIR 0.5-35 Hz for phase alignment; calibrated ABP retained for labels",
                    "quality_filter": {
                        "flatline_first_difference": True,
                        "minimum_dbp_mmHg": 25,
                        "template_rule": "mean difference/correlation 3 sigma",
                        **template_statistics,
                        "candidate_windows": int(len(valid)),
                        "retained_windows": int(len(windows)),
                    },
                },
            )
        )
        if maximum_subjects is not None and len(entries) >= maximum_subjects:
            break
    retained_split_map = balanced_subject_splits(
        [entry["subject_id"] for entry in entries],
        seed,
    )
    for entry in entries:
        entry["split"] = retained_split_map[entry["subject_id"]]
    return entries


def prepare_mimic_unpaired(
    output: Path,
    *,
    records: list[tuple[str, str | None]],
    seed: int,
    output_samples: int,
    minutes_per_record: float,
    maximum_windows_per_subject: int | None,
    maximum_subjects: int | None = None,
) -> list[dict]:
    """Prepare MIMIC PPG records whose invasive BP channel is missing."""

    subject_ids = [
        f"MIMICIII_UNPAIRED_{subject_from_record(record)}"
        for record, _ in records
    ]
    split_map = balanced_subject_splits(subject_ids, seed)
    entries: list[dict] = []
    for (record, record_pn_dir), subject_id in tqdm(
        zip(records, subject_ids),
        total=len(records),
        desc="Preparing unpaired MIMIC-III",
    ):
        try:
            header = wfdb.rdheader(record, pn_dir=record_pn_dir)
            names = list(header.sig_name)
            ppg_index = channel_index(names, ("PLETH", "PPG"))
            if ppg_index is None:
                continue
            sampling_rate = float(header.fs)
            stop = min(
                int(header.sig_len),
                int(minutes_per_record * 60 * sampling_rate),
            )
            record_data = wfdb.rdrecord(
                record,
                sampfrom=0,
                sampto=stop,
                channels=[ppg_index],
                pn_dir=record_pn_dir,
            ).p_signal
        except Exception as error:
            print(f"\nSkipping unpaired {record}: {error}")
            continue
        if record_data is None:
            continue
        try:
            ppg = wavelet_filter_ppg(
                resample_to_125hz(record_data[:, 0], sampling_rate)
            )
        except ValueError:
            continue
        windows, _ = cycle_windows(
            ppg,
            125.0,
            cycles=5,
            stride_cycles=2,
            output_samples=output_samples,
            filter_method="none",
        )
        if len(windows):
            template_valid, template_statistics = template_quality_mask(windows)
            windows = windows[template_valid]
        else:
            template_statistics = {}
        if maximum_windows_per_subject is not None:
            windows = windows[:maximum_windows_per_subject]
        if not len(windows):
            continue
        entries.append(
            save_subject(
                output,
                subject_id=subject_id,
                source="MIMICIII",
                split=split_map[subject_id],
                role="pretrain_unpaired",
                signals=windows,
                metadata={
                    "input_sampling_rate": sampling_rate,
                    "processed_sampling_rate": 125.0,
                    "paired_bp": False,
                    "record": record,
                    "pn_dir": record_pn_dir,
                    "ppg_filter": "db8 level 9; cA9 and cD3-cD1 zeroed",
                    "quality_filter": {
                        "template_rule": "mean difference/correlation 3 sigma",
                        **template_statistics,
                    },
                },
            )
        )
        if maximum_subjects is not None and len(entries) >= maximum_subjects:
            break
    retained_split_map = balanced_subject_splits(
        [entry["subject_id"] for entry in entries],
        seed,
    )
    for entry in entries:
        entry["split"] = retained_split_map[entry["subject_id"]]
    return entries


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--raw-root", type=Path, required=True)
    parser.add_argument("--download-unpaired", action="store_true")
    parser.add_argument(
        "--reuse-prepared",
        action="store_true",
        help="Reuse completed subject arrays after an interrupted run",
    )
    parser.add_argument("--skip-wesad", action="store_true")
    parser.add_argument("--skip-ppg-dalia", action="store_true")
    parser.add_argument("--skip-mimic", action="store_true")
    parser.add_argument("--mimic-root", type=Path)
    parser.add_argument(
        "--mimic-pn-dir",
        default="mimic3wdb/1.0",
        help="PhysioNet directory used with the paired record index",
    )
    parser.add_argument("--mimic-record-list", type=Path)
    parser.add_argument("--mimic-record-index", type=Path)
    parser.add_argument(
        "--discover-mimic-records",
        action="store_true",
        help="Slowly inspect the matched database instead of using the public index",
    )
    parser.add_argument(
        "--mimic-discovery-pn-dir",
        default="mimic3wdb-matched/1.0",
    )
    parser.add_argument(
        "--mimic-max-subjects",
        type=int,
        default=200,
        help="Number of paired MIMIC subjects (the paper uses 200)",
    )
    parser.add_argument(
        "--discover-mimic-unpaired",
        action="store_true",
        help="Also discover MIMIC records with PPG but no invasive BP",
    )
    parser.add_argument(
        "--mimic-unpaired-subjects",
        type=int,
        default=300,
        help="Number of unpaired MIMIC subjects used by STP pretraining",
    )
    parser.add_argument(
        "--mimic-unpaired-pn-dir",
        default="mimic3wdb-matched/1.0",
    )
    parser.add_argument("--mimic-index-workers", type=int, default=16)
    parser.add_argument("--mimic-minutes-per-record", type=float, default=30)
    parser.add_argument("--window-samples", type=int, default=512)
    parser.add_argument("--maximum-windows-per-subject", type=int)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--strict-stp-counts",
        action="store_true",
        help="Fail unless the public 300+15+15 unpaired and 200 paired roles exist",
    )
    args = parser.parse_args()

    raw_root = args.raw_root.resolve()
    output = args.output.resolve()
    raw_root.mkdir(parents=True, exist_ok=True)
    output.mkdir(parents=True, exist_ok=True)
    entries: list[dict] = []

    if not args.skip_wesad:
        archive = raw_root / "WESAD.zip"
        extracted = raw_root / "WESAD"
        if args.download_unpaired:
            progress_download(WESAD_URL, archive)
            extract_archive(archive, extracted)
        recovered = (
            recover_prepared_entries(
                output,
                source="WESAD",
                role="pretrain_unpaired",
                seed=args.seed,
                maximum_subjects=15,
            )
            if args.reuse_prepared
            else []
        )
        entries.extend(
            recovered
            if len(recovered) == 15
            else prepare_unpaired_source(
                extracted,
                output,
                source="WESAD",
                seed=args.seed,
                output_samples=args.window_samples,
                maximum_windows_per_subject=args.maximum_windows_per_subject,
            )
        )

    if not args.skip_ppg_dalia:
        archive = raw_root / "PPG-DaLiA.zip"
        extracted = raw_root / "PPG-DaLiA"
        if args.download_unpaired:
            progress_download(PPG_DALIA_URL, archive)
            extract_archive(archive, extracted)
        recovered = (
            recover_prepared_entries(
                output,
                source="PPGDaLiA",
                role="pretrain_unpaired",
                seed=args.seed,
                maximum_subjects=15,
            )
            if args.reuse_prepared
            else []
        )
        entries.extend(
            recovered
            if len(recovered) == 15
            else prepare_unpaired_source(
                extracted,
                output,
                source="PPGDaLiA",
                seed=args.seed,
                output_samples=args.window_samples,
                maximum_windows_per_subject=args.maximum_windows_per_subject,
            )
        )

    if not args.skip_mimic:
        recovered_paired = (
            recover_prepared_entries(
                output,
                source="MIMICIII",
                role="supervised_paired",
                seed=args.seed,
                maximum_subjects=args.mimic_max_subjects,
            )
            if args.reuse_prepared
            else []
        )
        recovered_paired = [
            entry for entry in recovered_paired if entry["labels"]
        ]
        candidate_subjects = max(
            args.mimic_max_subjects,
            2 * args.mimic_max_subjects,
        )
        if args.mimic_record_list:
            records = [
                (
                    line.split("|", 1)[-1].strip(),
                    line.split("|", 1)[0].strip() if "|" in line else None,
                )
                for line in args.mimic_record_list.read_text().splitlines()
                if line.strip() and not line.lstrip().startswith("#")
            ]
        elif args.mimic_root:
            records = local_mimic_records(
                args.mimic_root,
                candidate_subjects,
            )
        elif args.discover_mimic_records:
            records = public_mimic_records(
                args.mimic_discovery_pn_dir,
                candidate_subjects,
                args.mimic_index_workers,
            )
        else:
            index_path = (
                args.mimic_record_index
                if args.mimic_record_index
                else raw_root / "MIMIC-III_ppg_dataset_records.txt"
            )
            if not index_path.exists():
                progress_download(MIMIC_PAIRED_INDEX_URL, index_path)
            records = indexed_mimic_records(
                index_path,
                pn_dir=args.mimic_pn_dir,
                maximum_subjects=candidate_subjects,
                seed=args.seed,
            )
        paired_entries = (
            recovered_paired
            if len(recovered_paired) == args.mimic_max_subjects
            else prepare_mimic(
                output,
                records=records,
                seed=args.seed,
                output_samples=args.window_samples,
                minutes_per_record=args.mimic_minutes_per_record,
                maximum_windows_per_subject=args.maximum_windows_per_subject,
                maximum_subjects=args.mimic_max_subjects,
            )
        )
        entries.extend(paired_entries)
        if args.discover_mimic_unpaired:
            paired_subjects = {
                subject_from_record(record) for record, _ in records
            }
            unpaired_records = public_mimic_records(
                args.mimic_unpaired_pn_dir,
                2 * args.mimic_unpaired_subjects,
                args.mimic_index_workers,
                require_abp=False,
                excluded_subjects=paired_subjects,
            )
            entries.extend(
                prepare_mimic_unpaired(
                    output,
                    records=unpaired_records,
                    seed=args.seed,
                    output_samples=args.window_samples,
                    minutes_per_record=args.mimic_minutes_per_record,
                    maximum_windows_per_subject=args.maximum_windows_per_subject,
                    maximum_subjects=args.mimic_unpaired_subjects,
                )
            )

    manifest_path = output / "manifest.json"
    if manifest_path.exists():
        existing = json.loads(manifest_path.read_text(encoding="utf-8"))
        merged = {
            (entry["source"], entry["subject_id"]): entry
            for entry in existing.get("subjects", ())
        }
        merged.update(
            {
                (entry["source"], entry["subject_id"]): entry
                for entry in entries
            }
        )
        entries = sorted(
            merged.values(),
            key=lambda entry: (entry["source"], entry["subject_id"]),
        )
    manifest = {
        "format_version": 2,
        "method": "STP public-data faithful reproduction",
        "target_sampling_rate": 125,
        "ppg_filter": "db8 level 9; cA9 and cD3-cD1 zeroed",
        "abp_filter": "FIR 0.5-35 Hz",
        "quality_screening": "flatline, DBP >=25, average-template 3 sigma",
        "window_cycles": 5,
        "stride_cycles": 2,
        "window_samples": args.window_samples,
        "split_seed": args.seed,
        "split_protocol": "subject-wise 70/15/15",
        "subjects": entries,
    }
    manifest_path.write_text(
        json.dumps(manifest, indent=2),
        encoding="utf-8",
    )
    counts: dict[str, dict[str, int]] = {}
    for entry in entries:
        source = entry["source"]
        counts.setdefault(source, {"subjects": 0, "windows": 0})
        counts[source]["subjects"] += 1
        counts[source]["windows"] += entry["windows"]
    print(json.dumps(counts, indent=2))
    if args.strict_stp_counts:
        role_source_counts: dict[tuple[str, str], int] = {}
        for entry in entries:
            key = (entry["source"], entry.get("role", ""))
            role_source_counts[key] = role_source_counts.get(key, 0) + 1
        expected = {
            ("MIMICIII", "pretrain_unpaired"): 300,
            ("WESAD", "pretrain_unpaired"): 15,
            ("PPGDaLiA", "pretrain_unpaired"): 15,
            ("MIMICIII", "supervised_paired"): 200,
        }
        mismatches = {
            f"{source}/{role}": {
                "expected": expected_count,
                "actual": role_source_counts.get((source, role), 0),
            }
            for (source, role), expected_count in expected.items()
            if role_source_counts.get((source, role), 0) != expected_count
        }
        if mismatches:
            raise RuntimeError(
                "STP public cohort is incomplete: "
                + json.dumps(mismatches, sort_keys=True)
            )


if __name__ == "__main__":
    main()
