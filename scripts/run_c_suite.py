"""Run the full C-series robustness suite sequentially."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--status", type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    commands = [
        (
            "C0 robustness evaluation",
            [
                "scripts/evaluate_robustness.py",
                "--checkpoint",
                "outputs/pulsedb_m0_bp_only/best.pt",
                "--suite-config",
                "configs/pulsedb_c2_full_artifacts.yaml",
                "--output",
                "outputs/pulsedb_c0_baseline_robustness",
            ],
            Path("outputs/pulsedb_c0_baseline_robustness"),
        ),
        *[
            (
                name,
                [
                    "scripts/train_robustness.py",
                    "--config",
                    config,
                    "--output",
                    output,
                    "--resume",
                ],
                Path(output),
            )
            for name, config, output in (
                (
                    "C1 weak artifacts",
                    "configs/pulsedb_c1_weak_artifacts.yaml",
                    "outputs/pulsedb_c1_weak_artifacts",
                ),
                (
                    "C2 full artifacts",
                    "configs/pulsedb_c2_full_artifacts.yaml",
                    "outputs/pulsedb_c2_full_artifacts",
                ),
                (
                    "C3 contact and motion",
                    "configs/pulsedb_c3_contact_motion.yaml",
                    "outputs/pulsedb_c3_contact_motion",
                ),
                (
                    "C4 consistency",
                    "configs/pulsedb_c4_consistency.yaml",
                    "outputs/pulsedb_c4_consistency",
                ),
                (
                    "C5 artifact aware",
                    "configs/pulsedb_c5_artifact_aware.yaml",
                    "outputs/pulsedb_c5_artifact_aware",
                ),
            )
        ],
        (
            "C6 STP pretraining",
            [
                "scripts/pretrain_stp.py",
                "--config",
                "configs/pulsedb_c6_stp_pretrain.yaml",
                "--output",
                "outputs/pulsedb_c6_stp_pretrain",
                "--resume",
            ],
            Path("outputs/pulsedb_c6_stp_pretrain"),
        ),
        (
            "C6 STP fine-tuning",
            [
                "scripts/train_robustness.py",
                "--config",
                "configs/pulsedb_c6_stp_finetune.yaml",
                "--output",
                "outputs/pulsedb_c6_stp_finetune",
                "--resume",
            ],
            Path("outputs/pulsedb_c6_stp_finetune"),
        ),
    ]

    args.status.parent.mkdir(parents=True, exist_ok=True)
    status: dict[str, object] = {"state": "running", "runs": []}
    for name, arguments, output in commands:
        output.mkdir(parents=True, exist_ok=True)
        run = {
            "name": name,
            "output": str(output),
            "state": "running",
            "started_at": datetime.now().isoformat(timespec="seconds"),
        }
        runs = status["runs"]
        assert isinstance(runs, list)
        runs.append(run)
        args.status.write_text(json.dumps(status, indent=2), encoding="utf-8")
        with (output / "train_stdout.log").open("a", encoding="utf-8") as stdout, (
            output / "train_stderr.log"
        ).open("a", encoding="utf-8") as stderr:
            result = subprocess.run(
                [sys.executable, "-u", *arguments],
                cwd=root,
                stdout=stdout,
                stderr=stderr,
                check=False,
            )
        run["finished_at"] = datetime.now().isoformat(timespec="seconds")
        run["return_code"] = result.returncode
        run["state"] = "completed" if result.returncode == 0 else "failed"
        args.status.write_text(json.dumps(status, indent=2), encoding="utf-8")
        if result.returncode != 0:
            status["state"] = "failed"
            args.status.write_text(
                json.dumps(status, indent=2),
                encoding="utf-8",
            )
            raise SystemExit(result.returncode)
    status["state"] = "completed"
    args.status.write_text(json.dumps(status, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
