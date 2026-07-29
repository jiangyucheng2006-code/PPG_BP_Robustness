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
import shutil
import urllib.request
import zipfile
from pathlib import Path

import numpy as np
import wfdb
from scipy.signal import correlate, correlation_lags, find_peaks
from tqdm import tqdm

from ppg_bp.data.stp import bp_pattern_labels, cycle_windows


WESAD_URL = "https://uni-siegen.sciebo.de/s/HGdUkoNlW1Ub0Gx/download"
PPG_DALIA_URL = "https://archive.ics.uci.edu/static/public/495/ppg+dalia.zip"
MIMIC_PAIRED_INDEX_URL = (
    "https://raw.githubusercontent.com/v3551G/BP-prediction-survey/"
    "main/MIMIC-III_ppg_dataset_records.txt"
)


def progress_download(url: str, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        return
    temporary = destination.with_suffix(destination.suffix + ".part")

    def report(blocks: int, block_size: int, total: int) -> None:
        downloaded = blocks * block_size
        percent = 100 * downloaded / total if total > 0 else 0
        print(
            f"\rDownloading {destination.name}: "
            f"{downloaded / 2**30:.2f} / {total / 2**30:.2f} GB "
            f"({percent:.1f}%)",
            end="",
        )

    urllib.request.urlretrieve(url, temporary, reporthook=report)
    print()
    temporary.replace(destination)


def extract_archive(archive: Path, destination: Path) -> None:
    marker = destination / ".complete"
    if marker.exists():
        return
    destination.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive) as stream:
        stream.extractall(destination)
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
        windows, _ = cycle_windows(
            read_bvp_pickle(path),
            64.0,
            cycles=5,
            stride_cycles=2,
            output_samples=output_samples,
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
                signals=windows,
                metadata={
                    "input_sampling_rate": 64.0,
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
) -> int:
    limit = min(len(ppg), int(60 * sampling_rate))
    first = ppg[:limit]
    second = abp[:limit]
    first = (first - np.mean(first)) / max(float(np.std(first)), 1e-6)
    second = (second - np.mean(second)) / max(float(np.std(second)), 1e-6)
    values = correlate(first, second, mode="full", method="fft")
    lags = correlation_lags(len(first), len(second), mode="full")
    permitted = np.abs(lags) <= int(2 * sampling_rate)
    return int(lags[permitted][np.argmax(values[permitted])])


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
) -> list[tuple[str, str]]:
    """Find one paired PLETH/ABP waveform record per public subject."""

    database = pn_dir.split("/")[0]
    subject_directories = wfdb.get_record_list(database)
    def inspect_subject(directory: str) -> tuple[str, str] | None:
        subject_pn_dir = f"{pn_dir.rstrip('/')}/{directory.strip('/')}"
        try:
            candidates = wfdb.get_record_list(subject_pn_dir)
        except Exception:
            return None
        for candidate in candidates:
            if candidate.endswith("n"):
                continue
            try:
                header = wfdb.rdheader(candidate, pn_dir=subject_pn_dir)
            except Exception:
                continue
            if not header.sig_name:
                continue
            names = list(header.sig_name)
            if (
                channel_index(names, ("PLETH", "PPG")) is not None
                and channel_index(names, ("ABP", "ART", "ART1")) is not None
                and int(header.sig_len) >= int(10 * 60 * float(header.fs))
            ):
                return candidate, subject_pn_dir
        return None

    selected: list[tuple[str, str]] = []
    batch_size = max(workers * 4, 64)
    progress = tqdm(
        total=len(subject_directories),
        desc="Indexing MIMIC-III",
    )
    for start in range(0, len(subject_directories), batch_size):
        batch = subject_directories[start : start + batch_size]
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
        lag = estimate_phase_lag(ppg, abp, sampling_rate)
        aligned_abp = np.roll(abp, lag)
        windows, bounds = cycle_windows(
            ppg,
            sampling_rate,
            cycles=5,
            stride_cycles=2,
            output_samples=output_samples,
        )
        labels = np.asarray(
            [
                bp_from_abp_window(aligned_abp[start:stop], sampling_rate)
                for start, stop in bounds
            ],
            dtype=np.float32,
        )
        valid = (
            (labels[:, 0] >= 60)
            & (labels[:, 0] <= 240)
            & (labels[:, 1] >= 30)
            & (labels[:, 1] <= 160)
            & (labels[:, 0] > labels[:, 1])
        )
        windows = windows[valid]
        labels = labels[valid]
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
                signals=windows,
                labels=labels,
                patterns=bp_pattern_labels(labels),
                metadata={
                    "input_sampling_rate": sampling_rate,
                    "paired_bp": True,
                    "record": record,
                    "pn_dir": record_pn_dir,
                    "phase_lag_samples": lag,
                },
            )
        )
    return entries


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--raw-root", type=Path, required=True)
    parser.add_argument("--download-unpaired", action="store_true")
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
    parser.add_argument("--mimic-max-subjects", type=int, default=500)
    parser.add_argument("--mimic-index-workers", type=int, default=16)
    parser.add_argument("--mimic-minutes-per-record", type=float, default=30)
    parser.add_argument("--window-samples", type=int, default=512)
    parser.add_argument("--maximum-windows-per-subject", type=int)
    parser.add_argument("--seed", type=int, default=42)
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
        entries.extend(
            prepare_unpaired_source(
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
        entries.extend(
            prepare_unpaired_source(
                extracted,
                output,
                source="PPGDaLiA",
                seed=args.seed,
                output_samples=args.window_samples,
                maximum_windows_per_subject=args.maximum_windows_per_subject,
            )
        )

    if not args.skip_mimic:
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
                args.mimic_max_subjects,
            )
        elif args.discover_mimic_records:
            records = public_mimic_records(
                args.mimic_discovery_pn_dir,
                args.mimic_max_subjects,
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
                maximum_subjects=args.mimic_max_subjects,
                seed=args.seed,
            )
        entries.extend(
            prepare_mimic(
                output,
                records=records,
                seed=args.seed,
                output_samples=args.window_samples,
                minutes_per_record=args.mimic_minutes_per_record,
                maximum_windows_per_subject=args.maximum_windows_per_subject,
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
        "format_version": 1,
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


if __name__ == "__main__":
    main()
