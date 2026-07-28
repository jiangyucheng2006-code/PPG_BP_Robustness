"""Run training experiments sequentially while keeping separate logs."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--run",
        action="append",
        nargs=2,
        metavar=("CONFIG", "OUTPUT"),
    )
    parser.add_argument(
        "--run-seed",
        action="append",
        nargs=3,
        metavar=("CONFIG", "OUTPUT", "SEED"),
        help="Run one config with an explicit random seed override",
    )
    parser.add_argument("--status", type=Path, required=True)
    parser.add_argument(
        "--trainer",
        type=Path,
        default=Path("scripts/train_baseline.py"),
        help="Training entry point, relative to the repository root",
    )
    args = parser.parse_args()
    scheduled_runs: list[tuple[str, str, int | None]] = [
        (config, output, None) for config, output in (args.run or [])
    ]
    scheduled_runs.extend(
        (config, output, int(seed))
        for config, output, seed in (args.run_seed or [])
    )
    if not scheduled_runs:
        parser.error("at least one --run or --run-seed is required")

    root = Path(__file__).resolve().parents[1]
    args.status.parent.mkdir(parents=True, exist_ok=True)
    queue_status: dict[str, object] = {"state": "running", "runs": []}

    for config_text, output_text, seed in scheduled_runs:
        config = Path(config_text)
        output = Path(output_text)
        output.mkdir(parents=True, exist_ok=True)
        run_status = {
            "config": str(config),
            "output": str(output),
            "seed": seed,
            "state": "running",
            "started_at": datetime.now().isoformat(timespec="seconds"),
        }
        runs = queue_status["runs"]
        assert isinstance(runs, list)
        runs.append(run_status)
        args.status.write_text(json.dumps(queue_status, indent=2), encoding="utf-8")

        with (output / "train_stdout.log").open("a", encoding="utf-8") as stdout, (
            output / "train_stderr.log"
        ).open("a", encoding="utf-8") as stderr:
            command = [
                sys.executable,
                "-u",
                str(args.trainer),
                "--config",
                str(config),
                "--output",
                str(output),
                "--resume",
            ]
            if seed is not None:
                command.extend(["--seed", str(seed)])
            result = subprocess.run(
                command,
                cwd=root,
                stdout=stdout,
                stderr=stderr,
                check=False,
            )

        run_status["finished_at"] = datetime.now().isoformat(timespec="seconds")
        run_status["return_code"] = result.returncode
        run_status["state"] = "completed" if result.returncode == 0 else "failed"
        args.status.write_text(json.dumps(queue_status, indent=2), encoding="utf-8")
        if result.returncode != 0:
            queue_status["state"] = "failed"
            args.status.write_text(json.dumps(queue_status, indent=2), encoding="utf-8")
            raise SystemExit(result.returncode)

    queue_status["state"] = "completed"
    args.status.write_text(json.dumps(queue_status, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
