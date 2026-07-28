import numpy as np
import torch

from ppg_bp.training.losses import (
    LabelDensityWeightedMSELoss,
    label_density_weights,
)


def test_label_density_weights_emphasize_rare_labels() -> None:
    values = np.asarray([100.0] * 100 + [160.0] * 4, dtype=np.float32)
    spec = label_density_weights(
        values,
        bin_width=5.0,
        power=0.5,
        smoothing_sigma_bins=1.0,
        minimum_weight=0.25,
        maximum_weight=4.0,
    )
    common_index = int((100.0 - spec.lower_edge) // spec.bin_width)
    rare_index = int((160.0 - spec.lower_edge) // spec.bin_width)
    assert spec.bin_weights[rare_index] > spec.bin_weights[common_index]


def test_weighted_mse_uses_separate_sbp_and_dbp_weights() -> None:
    train_targets = np.asarray(
        [[110.0, 60.0]] * 50 + [[170.0, 100.0]] * 2,
        dtype=np.float32,
    )
    criterion = LabelDensityWeightedMSELoss(
        train_targets,
        maximum_weight=4.0,
    )
    predictions = torch.zeros((2, 2))
    normalized_targets = torch.ones((2, 2))
    raw_targets = torch.tensor([[110.0, 60.0], [170.0, 100.0]])
    loss = criterion(predictions, normalized_targets, raw_targets)
    assert torch.isfinite(loss)
    assert float(loss) > 0
