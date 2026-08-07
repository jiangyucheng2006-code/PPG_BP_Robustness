"""Run the repeated STP downstream optimization matrix sequentially."""

from __future__ import annotations

import argparse
import atexit
import copy
import ctypes
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "stp_optimization_suite.yaml"
DEFAULT_OUTPUT = PROJECT_ROOT / "outputs" / "stp_optimization_suite"
S1_CHECKPOINT = PROJECT_ROOT / "outputs" / "stp_s1_public_pretrain" / "best.pt"


def prevent_idle_sleep() -> None:
    """Keep Windows awake while the queue process is alive."""

    if os.name != "nt":
        return
    execution_state_continuous = 0x80000000
    execution_state_system_required = 0x00000001
    kernel = ctypes.windll.kernel32
    kernel.SetThreadExecutionState(
        execution_state_continuous | execution_state_system_required
    )
    atexit.register(
        kernel.SetThreadExecutionState,
        execution_state_continuous,
    )


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def merge(target: dict, update: dict) -> dict:
    result = copy.deepcopy(target)
    for key, value in update.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = merge(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2), encoding="utf-8")
    temporary.replace(path)


def concise_metrics(stage: str, metrics: dict) -> dict:
    test = metrics["test"]
    result = {
        "best_epoch": metrics["best_epoch"],
        "completed_epochs": metrics["completed_epochs"],
    }
    if stage == "S2":
        result.update(
            pattern_accuracy=test["pattern_accuracy"],
            pattern_macro_recall=test["pattern_macro_recall"],
        )
    else:
        result.update(
            sbp_mae=test["sbp_mae"],
            dbp_mae=test["dbp_mae"],
            mean_mae=test["mean_mae"],
            sbp_aami_numerical_pass=test["sbp_aami_numerical_pass"],
            dbp_aami_numerical_pass=test["dbp_aami_numerical_pass"],
        )
    return result


def update_summary(output_root: Path, status: dict) -> None:
    bp_records = [
        record
        for record in status["records"]
        if record["stage"] == "S3" and record["state"] == "complete"
    ]
    best_by_path = {}
    for path in ("direct_s1", "via_s2"):
        candidates = [
            record for record in bp_records if record["transfer_path"] == path
        ]
        if candidates:
            best_by_path[path] = min(
                candidates,
                key=lambda record: record["metrics"]["mean_mae"],
            )
    summary = {
        "updated_at": utc_now(),
        "state": status["state"],
        "completed_stages": status["completed_stages"],
        "planned_stages": status["planned_stages"],
        "best_by_transfer_path": best_by_path,
        "records": status["records"],
    }
    write_json(output_root / "summary.json", summary)


def run_stage(
    *,
    stage: str,
    seed: int,
    variant: dict,
    transfer_path: str,
    base_config: dict,
    common_training: dict,
    source_checkpoint: Path,
    output: Path,
    status_path: Path,
    status: dict,
    smoke: bool,
) -> dict:
    config = copy.deepcopy(base_config)
    config["seed"] = seed
    config["experiment"]["id"] = (
        f"{variant['id']}_{transfer_path}_seed{seed}_{stage}"
    )
    config["experiment"]["description"] = variant["description"]
    config["model"] = merge(config["model"], variant.get("model", {}))
    config["training"] = merge(config["training"], common_training)
    config["training"] = merge(
        config["training"],
        variant.get("training", {}),
    )
    config["training"]["source_checkpoint"] = str(source_checkpoint)
    if smoke:
        config["data"]["maximum_samples"] = 512
        config["training"].update(
            epochs=1,
            batch_size=64,
            early_stopping_patience=0,
            num_workers=0,
        )

    output.mkdir(parents=True, exist_ok=True)
    resolved_config = output / "resolved_config.yaml"
    resolved_config.write_text(
        yaml.safe_dump(config, sort_keys=False),
        encoding="utf-8",
    )
    metrics_path = output / "metrics.json"
    record = {
        "stage": stage,
        "seed": seed,
        "variant": variant["id"],
        "description": variant["description"],
        "transfer_path": transfer_path,
        "output": str(output),
    }
    if metrics_path.exists():
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        return {
            **record,
            "state": "complete",
            "metrics": concise_metrics(stage, metrics),
            "resumed_existing_result": True,
        }

    status["current"] = {
        **record,
        "started_at": utc_now(),
    }
    status["updated_at"] = utc_now()
    write_json(status_path, status)

    environment = os.environ.copy()
    environment["PPG_BP_DATA_ROOT"] = str(
        Path(r"D:\Datasets\PPG_BP_Robustness")
    )
    environment["PYTHONUNBUFFERED"] = "1"
    started = time.monotonic()
    with (output / "train.out.log").open("a", encoding="utf-8") as stdout, (
        output / "train.err.log"
    ).open("a", encoding="utf-8") as stderr:
        completed = subprocess.run(
            [
                sys.executable,
                "-u",
                "scripts/train_stp.py",
                "--config",
                str(resolved_config),
                "--output",
                str(output),
                "--resume",
            ],
            cwd=PROJECT_ROOT,
            env=environment,
            stdout=stdout,
            stderr=stderr,
            check=False,
        )
    if completed.returncode != 0 or not metrics_path.exists():
        return {
            **record,
            "state": "failed",
            "exit_code": completed.returncode,
            "elapsed_minutes": (time.monotonic() - started) / 60,
        }
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    return {
        **record,
        "state": "complete",
        "elapsed_minutes": (time.monotonic() - started) / 60,
        "metrics": concise_metrics(stage, metrics),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--status", type=Path)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    prevent_idle_sleep()

    suite = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    output_root = args.output.resolve()
    status_path = (
        args.status.resolve()
        if args.status
        else output_root / "status.json"
    )
    if not S1_CHECKPOINT.exists():
        raise FileNotFoundError(S1_CHECKPOINT)

    seeds = suite["seeds"][:1] if args.smoke else suite["seeds"]
    variants = suite["variants"]
    transfer_paths = suite["transfer_paths"]
    stages_per_variant = sum(
        1 if path == "direct_s1" else 2 for path in transfer_paths
    )
    planned_stages = len(seeds) * len(variants) * stages_per_variant
    status = {
        "state": "running",
        "started_at": utc_now(),
        "updated_at": utc_now(),
        "estimated_runtime_hours": (
            "<1 smoke test"
            if args.smoke
            else suite["experiment"]["estimated_runtime_hours"]
        ),
        "planned_stages": planned_stages,
        "completed_stages": 0,
        "current": None,
        "records": [],
    }
    if status_path.exists():
        previous = json.loads(status_path.read_text(encoding="utf-8"))
        status["started_at"] = previous.get("started_at", status["started_at"])
        status["records"] = previous.get("records", [])
    known_records = {
        (
            record["stage"],
            record["seed"],
            record["variant"],
            record["transfer_path"],
        ): record
        for record in status["records"]
    }
    status["completed_stages"] = sum(
        record["state"] == "complete" for record in known_records.values()
    )
    write_json(status_path, status)

    base_s2 = yaml.safe_load(
        (PROJECT_ROOT / "configs" / "stp_s2_public_pattern.yaml").read_text(
            encoding="utf-8"
        )
    )
    base_s3 = yaml.safe_load(
        (PROJECT_ROOT / "configs" / "stp_s3_public_bp.yaml").read_text(
            encoding="utf-8"
        )
    )
    for seed in seeds:
        for variant in variants:
            variant_root = output_root / f"seed_{seed}" / variant["id"]
            for transfer_path in transfer_paths:
                source_checkpoint = S1_CHECKPOINT
                if transfer_path == "via_s2":
                    s2_output = variant_root / transfer_path / "S2"
                    s2_record = run_stage(
                        stage="S2",
                        seed=seed,
                        variant=variant,
                        transfer_path=transfer_path,
                        base_config=base_s2,
                        common_training=suite["common_training"],
                        source_checkpoint=S1_CHECKPOINT,
                        output=s2_output,
                        status_path=status_path,
                        status=status,
                        smoke=args.smoke,
                    )
                    key = ("S2", seed, variant["id"], transfer_path)
                    known_records[key] = s2_record
                    status["records"] = list(known_records.values())
                    status["completed_stages"] = sum(
                        record["state"] == "complete"
                        for record in known_records.values()
                    )
                    status["updated_at"] = utc_now()
                    write_json(status_path, status)
                    update_summary(output_root, status)
                    if s2_record["state"] != "complete":
                        continue
                    source_checkpoint = s2_output / "best.pt"

                s3_output = variant_root / transfer_path / "S3"
                s3_record = run_stage(
                    stage="S3",
                    seed=seed,
                    variant=variant,
                    transfer_path=transfer_path,
                    base_config=base_s3,
                    common_training=suite["common_training"],
                    source_checkpoint=source_checkpoint,
                    output=s3_output,
                    status_path=status_path,
                    status=status,
                    smoke=args.smoke,
                )
                key = ("S3", seed, variant["id"], transfer_path)
                known_records[key] = s3_record
                status["records"] = list(known_records.values())
                status["completed_stages"] = sum(
                    record["state"] == "complete"
                    for record in known_records.values()
                )
                status["updated_at"] = utc_now()
                write_json(status_path, status)
                update_summary(output_root, status)

    status["state"] = (
        "complete"
        if all(record["state"] == "complete" for record in known_records.values())
        and len(known_records) == planned_stages
        else "complete_with_failures"
    )
    status["current"] = None
    status["updated_at"] = utc_now()
    status["records"] = list(known_records.values())
    write_json(status_path, status)
    update_summary(output_root, status)


if __name__ == "__main__":
    main()
