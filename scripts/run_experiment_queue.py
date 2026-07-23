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
        required=True,
    )
    parser.add_argument("--status", type=Path, required=True)
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    args.status.parent.mkdir(parents=True, exist_ok=True)
    queue_status: dict[str, object] = {"state": "running", "runs": []}

    for config_text, output_text in args.run:
        config = Path(config_text)
        output = Path(output_text)
        output.mkdir(parents=True, exist_ok=True)
        run_status = {
            "config": str(config),
            "output": str(output),
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
            result = subprocess.run(
                [
                    sys.executable,
                    "-u",
                    "scripts/train_baseline.py",
                    "--config",
                    str(config),
                    "--output",
                    str(output),
                    "--resume",
                ],
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
