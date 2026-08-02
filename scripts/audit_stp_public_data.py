"""Audit the public STP cohort before a long three-stage run."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np


EXPECTED_ROLES = {
    ("MIMICIII", "pretrain_unpaired"): 300,
    ("WESAD", "pretrain_unpaired"): 15,
    ("PPGDaLiA", "pretrain_unpaired"): 15,
    ("MIMICIII", "supervised_paired"): 200,
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
    seen: set[tuple[str, str]] = set()

    for entry in subjects:
        key = (str(entry.get("source")), str(entry.get("subject_id")))
        if key in seen:
            duplicate_keys.append("/".join(key))
        seen.add(key)
        split = str(entry.get("split"))
        split_counts[split]["subjects"] += 1
        split_counts[split]["windows"] += int(entry.get("windows", 0))
        for field in ("signals", "labels", "patterns"):
            relative = entry.get(field)
            if relative and not (root / relative).exists():
                missing_files.append(str(relative))

    paired_test = [
        entry
        for entry in subjects
        if entry.get("role") == "supervised_paired"
        and entry.get("split") == "test"
    ]
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
        "paired_test_distribution": distribution,
        "private_683_subject_cohort_available": False,
        "strict_public_cohort_pass": bool(
            len(subjects) == 530
            and not mismatches
            and not missing_files
            and not duplicate_keys
        ),
        "role_mismatches": mismatches,
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
