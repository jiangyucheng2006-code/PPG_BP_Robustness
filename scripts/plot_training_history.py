from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot training and validation curves from history.json."
    )
    parser.add_argument("history", type=Path, help="Path to history.json")
    parser.add_argument("output", type=Path, help="Output PNG path")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    history = json.loads(args.history.read_text(encoding="utf-8"))
    if isinstance(history, dict):
        history = history["history"]

    epochs = [row["epoch"] for row in history]
    train_mse = [row["train_mse"] for row in history]
    sbp_mae = [row["sbp_mae"] for row in history]
    dbp_mae = [row["dbp_mae"] for row in history]
    mean_mae = [row["mean_mae"] for row in history]
    best_index = min(range(len(history)), key=lambda index: mean_mae[index])
    best_epoch = epochs[best_index]

    plt.style.use("seaborn-v0_8-whitegrid")
    figure, axes = plt.subplots(1, 2, figsize=(12, 4.5), dpi=160)

    axes[0].plot(epochs, train_mse, color="#2458A6", linewidth=2.2)
    axes[0].axvline(best_epoch, color="#D95F02", linestyle="--", linewidth=1.5)
    axes[0].set(
        title="Training loss",
        xlabel="Epoch",
        ylabel="Standardized MSE",
        yscale="log",
    )

    axes[1].plot(epochs, sbp_mae, label="SBP", color="#2458A6", linewidth=1.8)
    axes[1].plot(epochs, dbp_mae, label="DBP", color="#2A9D8F", linewidth=1.8)
    axes[1].plot(
        epochs,
        mean_mae,
        label="Mean",
        color="#222222",
        linewidth=2.2,
    )
    axes[1].scatter(
        [best_epoch],
        [mean_mae[best_index]],
        color="#D95F02",
        zorder=5,
        label=f"Best epoch ({best_epoch})",
    )
    axes[1].axvline(best_epoch, color="#D95F02", linestyle="--", linewidth=1.5)
    axes[1].set(
        title="Validation error",
        xlabel="Epoch",
        ylabel="MAE (mmHg)",
    )
    axes[1].legend(frameon=True, ncol=2)

    figure.suptitle("XResNet-101 baseline training", fontsize=14, fontweight="bold")
    figure.tight_layout()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output, bbox_inches="tight")
    plt.close(figure)


if __name__ == "__main__":
    main()
