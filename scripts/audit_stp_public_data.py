"""Audit the public STP cohort before a long three-stage run."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from ppg_bp.data.stp import bp_pattern_labels


EXPECTED_ROLES = {
    ("MIMICIII", "pretrain_unpaired"): 300,
    ("WESAD", "pretrain_unpaired"): 15,
    ("PPGDaLiA", "pretrain_unpaired"): 15,
    ("MIMICIII", "supervised_paired"): 200,
}

EXPECTED_ROLE_SPLITS = {
    ("MIMICIII", "pretrain_unpaired", "train"): 210,
    ("MIMICIII", "pretrain_unpaired", "val"): 45,
    ("MIMICIII", "pretrain_unpaired", "test"): 45,
    ("WESAD", "pretrain_unpaired", "train"): 10,
    ("WESAD", "pretrain_unpaired", "val"): 2,
    ("WESAD", "pretrain_unpaired", "test"): 3,
    ("PPGDaLiA", "pretrain_unpaired", "train"): 10,
    ("PPGDaLiA", "pretrain_unpaired", "val"): 2,
    ("PPGDaLiA", "pretrain_unpaired", "test"): 3,
    ("MIMICIII", "supervised_paired", "train"): 140,
    ("MIMICIII", "supervised_paired", "val"): 30,
    ("MIMICIII", "supervised_paired", "test"): 30,
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--strict", action="store_true")
    args = parser.parse_args()

    root = args.root.resolve()
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    subjects = manifest.get("subjects", [])
    role_counts = Counter(
        (entry.get("source"), entry.get("role")) for entry in subjects
    )
    split_counts: dict[str, dict[str, int]] = defaultdict(
        lambda: {"subjects": 0, "windows": 0}
    )
    missing_files: list[str] = []
    duplicate_keys: list[str] = []
    protocol_issues: list[str] = []
    array_issues: list[str] = []
    seen: set[tuple[str, str]] = set()
    paired_mimic_subjects: set[str] = set()
    unpaired_mimic_subjects: set[str] = set()
    canonical_splits: dict[str, set[str]] = defaultdict(set)
    role_split_subjects: Counter[tuple[str, str, str]] = Counter()
    expected_window_samples = int(manifest.get("window_samples", 0))

    for entry in subjects:
        key = (str(entry.get("source")), str(entry.get("subject_id")))
        if key in seen:
            duplicate_keys.append("/".join(key))
        seen.add(key)
        split = str(entry.get("split"))
        role = str(entry.get("role"))
        source = str(entry.get("source"))
        if split not in {"train", "val", "test"}:
            protocol_issues.append(f"{key[0]}/{key[1]}: invalid split={split!r}")
        split_counts[split]["subjects"] += 1
        split_counts[split]["windows"] += int(entry.get("windows", 0))
        role_split_subjects[(source, role, split)] += 1
        for field in ("signals", "labels", "patterns"):
            relative = entry.get(field)
            if relative and not (root / relative).exists():
                missing_files.append(str(relative))
        metadata = entry.get("metadata", {})
        expected_segmentation = (
            "fixed_length"
            if entry.get("role") == "pretrain_unpaired"
            else "five_cycles_two_cycle_overlap"
        )
        if metadata.get("segmentation") != expected_segmentation:
            protocol_issues.append(
                f"{key[0]}/{key[1]}: segmentation="
                f"{metadata.get('segmentation')!r}, expected={expected_segmentation!r}"
            )
        if int(metadata.get("processing_protocol_version", 0)) < 4:
            protocol_issues.append(
                f"{key[0]}/{key[1]}: processing protocol is older than v4"
            )
        if entry.get("source") == "MIMICIII":
            canonical = str(metadata.get("canonical_subject_id", ""))
            if not canonical:
                protocol_issues.append(
                    f"{key[0]}/{key[1]}: canonical MIMIC subject ID is missing"
                )
            elif entry.get("role") == "supervised_paired":
                paired_mimic_subjects.add(canonical)
            elif entry.get("role") == "pretrain_unpaired":
                unpaired_mimic_subjects.add(canonical)
            if canonical:
                canonical_splits[f"MIMICIII/{canonical}"].add(split)
        else:
            subject_key = f"{source}/{entry.get('subject_id')}"
            canonical_splits[subject_key].add(split)

        windows = int(entry.get("windows", 0))
        signal_path = root / str(entry.get("signals", ""))
        if windows <= 0:
            array_issues.append(f"{key[0]}/{key[1]}: no retained windows")
        if signal_path.exists():
            try:
                signals = np.load(signal_path, mmap_mode="r")
                if (
                    signals.ndim != 2
                    or signals.shape[0] != windows
                    or signals.shape[1] != expected_window_samples
                ):
                    array_issues.append(
                        f"{key[0]}/{key[1]}: signals shape={signals.shape}, "
                        f"expected=({windows}, {expected_window_samples})"
                    )
                elif not np.isfinite(signals).all():
                    array_issues.append(f"{key[0]}/{key[1]}: non-finite signals")
                elif float(signals.min()) < -1e-5 or float(signals.max()) > 1.00001:
                    array_issues.append(
                        f"{key[0]}/{key[1]}: signals outside [0, 1]"
                    )
            except (OSError, ValueError) as error:
                array_issues.append(
                    f"{key[0]}/{key[1]}: cannot read signals ({error})"
                )

        paired = role == "supervised_paired"
        if paired and (not entry.get("labels") or not entry.get("patterns")):
            array_issues.append(
                f"{key[0]}/{key[1]}: paired entry lacks labels or patterns"
            )
        if not paired and (entry.get("labels") or entry.get("patterns")):
            array_issues.append(
                f"{key[0]}/{key[1]}: unpaired entry unexpectedly has targets"
            )
        label_path = root / str(entry.get("labels", ""))
        pattern_path = root / str(entry.get("patterns", ""))
        if paired and label_path.exists() and pattern_path.exists():
            try:
                labels = np.asarray(np.load(label_path, mmap_mode="r"))
                patterns = np.asarray(np.load(pattern_path, mmap_mode="r"))
                if labels.shape != (windows, 2):
                    array_issues.append(
                        f"{key[0]}/{key[1]}: labels shape={labels.shape}"
                    )
                elif (
                    not np.isfinite(labels).all()
                    or bool(np.any(labels[:, 1] < 25))
                    or bool(np.any(labels[:, 0] <= labels[:, 1]))
                ):
                    array_issues.append(
                        f"{key[0]}/{key[1]}: invalid BP targets"
                    )
                if patterns.shape != (windows,):
                    array_issues.append(
                        f"{key[0]}/{key[1]}: patterns shape={patterns.shape}"
                    )
                elif labels.shape == (windows, 2) and not np.array_equal(
                    patterns,
                    bp_pattern_labels(labels),
                ):
                    array_issues.append(
                        f"{key[0]}/{key[1]}: pattern labels disagree with BP"
                    )
            except (OSError, ValueError) as error:
                array_issues.append(
                    f"{key[0]}/{key[1]}: cannot read targets ({error})"
                )

    paired_test = [
        entry
        for entry in subjects
        if entry.get("role") == "supervised_paired"
        and entry.get("split") == "test"
    ]
    if paired_test:
        test_labels = np.concatenate(
            [np.load(root / entry["labels"], mmap_mode="r") for entry in paired_test]
        )
        sbp = test_labels[:, 0]
        dbp = test_labels[:, 1]
        distribution = {
            "measurements": int(len(test_labels)),
            "subjects": len(paired_test),
            "sbp_above_160_percent": float(100 * np.mean(sbp > 160)),
            "sbp_below_100_percent": float(100 * np.mean(sbp < 100)),
            "dbp_above_90_percent": float(100 * np.mean(dbp > 90)),
            "dbp_below_60_percent": float(100 * np.mean(dbp < 60)),
        }
    else:
        distribution = {
            "measurements": 0,
            "subjects": 0,
            "sbp_above_160_percent": 0.0,
            "sbp_below_100_percent": 0.0,
            "dbp_above_90_percent": 0.0,
            "dbp_below_60_percent": 0.0,
        }
    distribution["aami_distribution_pass"] = bool(
        distribution["measurements"] >= 150
        and distribution["subjects"] >= 85
        and distribution["sbp_above_160_percent"] >= 10
        and distribution["sbp_below_100_percent"] >= 10
        and distribution["dbp_above_90_percent"] >= 10
        and distribution["dbp_below_60_percent"] >= 10
    )

    mismatches = {
        f"{source}/{role}": {
            "expected": expected,
            "actual": role_counts.get((source, role), 0),
        }
        for (source, role), expected in EXPECTED_ROLES.items()
        if role_counts.get((source, role), 0) != expected
    }
    split_mismatches = {
        f"{source}/{role}/{split}": {
            "expected": expected,
            "actual": role_split_subjects.get((source, role, split), 0),
        }
        for (source, role, split), expected in EXPECTED_ROLE_SPLITS.items()
        if role_split_subjects.get((source, role, split), 0) != expected
    }
    cross_role_subject_overlap = sorted(
        paired_mimic_subjects.intersection(unpaired_mimic_subjects)
    )
    cross_split_subject_overlap = sorted(
        subject
        for subject, splits in canonical_splits.items()
        if len(splits) > 1
    )
    report = {
        "root": str(root),
        "public_subjects": len(subjects),
        "expected_public_subjects": 530,
        "role_counts": {
            f"{source}/{role}": count
            for (source, role), count in sorted(role_counts.items())
        },
        "split_counts": dict(sorted(split_counts.items())),
        "missing_files": missing_files,
        "duplicate_subject_keys": duplicate_keys,
        "protocol_issues": protocol_issues,
        "array_issues": array_issues,
        "mimic_cross_role_subject_overlap": cross_role_subject_overlap,
        "cross_split_subject_overlap": cross_split_subject_overlap,
        "role_split_subject_counts": {
            f"{source}/{role}/{split}": count
            for (source, role, split), count in sorted(role_split_subjects.items())
        },
        "paired_test_distribution": distribution,
        "private_683_subject_cohort_available": False,
        "strict_public_cohort_pass": bool(
            len(subjects) == 530
            and not mismatches
            and not missing_files
            and not duplicate_keys
            and not protocol_issues
            and not array_issues
            and not cross_role_subject_overlap
            and not cross_split_subject_overlap
            and not split_mismatches
            and int(manifest.get("format_version", 0)) >= 4
            and manifest.get("pretrain_segmentation") == "fixed_length"
            and manifest.get("paired_window_cycles") == 5
            and manifest.get("paired_overlap_cycles") == 2
        ),
        "role_mismatches": mismatches,
        "role_split_mismatches": split_mismatches,
    }
    rendered = json.dumps(report, indent=2)
    print(rendered)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    if args.strict and not report["strict_public_cohort_pass"]:
        raise SystemExit("Public STP cohort audit failed")


if __name__ == "__main__":
    main()
