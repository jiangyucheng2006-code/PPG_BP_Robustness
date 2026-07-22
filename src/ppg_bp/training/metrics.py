from __future__ import annotations

import numpy as np


def regression_metrics(predictions: np.ndarray, targets: np.ndarray) -> dict[str, float]:
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
    result["mean_mae"] = 0.5 * (result["sbp_mae"] + result["dbp_mae"])
    return result

