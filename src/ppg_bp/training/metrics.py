from __future__ import annotations

import numpy as np


def _bhs_grade(within_5: float, within_10: float, within_15: float) -> str:
    for grade, thresholds in (
        ("A", (60.0, 85.0, 95.0)),
        ("B", (50.0, 75.0, 90.0)),
        ("C", (40.0, 65.0, 85.0)),
    ):
        if all(value >= threshold for value, threshold in zip(
            (within_5, within_10, within_15),
            thresholds,
            strict=True,
        )):
            return grade
    return "D"


def regression_metrics(
    predictions: np.ndarray,
    targets: np.ndarray,
) -> dict[str, float | bool | str]:
    predictions = np.asarray(predictions, dtype=np.float64)
    targets = np.asarray(targets, dtype=np.float64)
    if predictions.shape != targets.shape or predictions.ndim != 2 or predictions.shape[1] != 2:
        raise ValueError("predictions and targets must both have shape [N, 2]")

    error = predictions - targets
    absolute_error = np.abs(error)
    squared_error = error**2
    names = ("sbp", "dbp")
    result: dict[str, float] = {}
    for index, name in enumerate(names):
        result[f"{name}_mae"] = float(absolute_error[:, index].mean())
        result[f"{name}_rmse"] = float(np.sqrt(squared_error[:, index].mean()))
        result[f"{name}_bias"] = float(error[:, index].mean())
        result[f"{name}_error_std"] = float(error[:, index].std(ddof=1)) if len(error) > 1 else 0.0
        within_5 = float((absolute_error[:, index] <= 5.0).mean() * 100.0)
        within_10 = float((absolute_error[:, index] <= 10.0).mean() * 100.0)
        within_15 = float((absolute_error[:, index] <= 15.0).mean() * 100.0)
        result[f"{name}_within_5_pct"] = within_5
        result[f"{name}_within_10_pct"] = within_10
        result[f"{name}_within_15_pct"] = within_15
        result[f"{name}_bhs_grade"] = _bhs_grade(within_5, within_10, within_15)
        result[f"{name}_aami_numerical_pass"] = bool(
            abs(result[f"{name}_bias"]) <= 5.0
            and result[f"{name}_error_std"] <= 8.0
        )
    result["mean_mae"] = 0.5 * (result["sbp_mae"] + result["dbp_mae"])
    return result
