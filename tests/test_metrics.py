import numpy as np

from ppg_bp.training.metrics import regression_metrics


def test_regression_metrics() -> None:
    targets = np.array([[120.0, 80.0], [130.0, 70.0]])
    predictions = np.array([[122.0, 79.0], [126.0, 73.0]])
    metrics = regression_metrics(predictions, targets)
    assert metrics["sbp_mae"] == 3.0
    assert metrics["dbp_mae"] == 2.0
    assert metrics["mean_mae"] == 2.5
    assert metrics["sbp_aami_numerical_pass"] is True
    assert metrics["dbp_aami_numerical_pass"] is True
    assert metrics["sbp_within_5_pct"] == 100.0
    assert metrics["dbp_bhs_grade"] == "A"
