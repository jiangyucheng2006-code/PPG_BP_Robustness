"""Regression losses used by the PPG blood-pressure experiments."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


class RegressionLoss(nn.Module):
    """Common loss interface that keeps raw BP labels available for weighting."""

    def forward(
        self,
        predictions: torch.Tensor,
        normalized_targets: torch.Tensor,
        raw_targets: torch.Tensor,
    ) -> torch.Tensor:
        raise NotImplementedError


class MeanSquaredRegressionLoss(RegressionLoss):
    def forward(
        self,
        predictions: torch.Tensor,
        normalized_targets: torch.Tensor,
        raw_targets: torch.Tensor,
    ) -> torch.Tensor:
        del raw_targets
        return F.mse_loss(predictions, normalized_targets)


class HuberRegressionLoss(RegressionLoss):
    def __init__(self, delta: float = 5.0) -> None:
        super().__init__()
        self.delta = float(delta)

    def forward(
        self,
        predictions: torch.Tensor,
        normalized_targets: torch.Tensor,
        raw_targets: torch.Tensor,
    ) -> torch.Tensor:
        del raw_targets
        return F.huber_loss(predictions, normalized_targets, delta=self.delta)


@dataclass(frozen=True)
class DensityWeightSpec:
    lower_edge: float
    bin_width: float
    bin_weights: np.ndarray
    sample_weight_quantiles: tuple[float, ...]


def label_density_weights(
    values: np.ndarray,
    *,
    bin_width: float,
    power: float,
    smoothing_sigma_bins: float,
    minimum_weight: float,
    maximum_weight: float,
) -> DensityWeightSpec:
    """Create inverse-density weights from training labels only.

    Gaussian smoothing avoids unstable weights for isolated or empty histogram
    bins. Weights are normalized to a training-sample mean of one, then clipped
    to keep rare labels from destabilizing optimization.
    """

    values = np.asarray(values, dtype=np.float64).reshape(-1)
    if values.size == 0:
        raise ValueError("At least one training label is required")
    if bin_width <= 0:
        raise ValueError("bin_width must be positive")
    if not 0 <= power <= 1:
        raise ValueError("power must be between 0 and 1")
    if smoothing_sigma_bins < 0:
        raise ValueError("smoothing_sigma_bins cannot be negative")
    if minimum_weight <= 0 or maximum_weight < minimum_weight:
        raise ValueError("Invalid weight clipping range")

    lower_edge = float(np.floor(values.min() / bin_width) * bin_width)
    upper_edge = float(np.ceil(values.max() / bin_width) * bin_width)
    bin_count = max(1, int(round((upper_edge - lower_edge) / bin_width)) + 1)
    indices = np.floor((values - lower_edge) / bin_width).astype(np.int64)
    indices = np.clip(indices, 0, bin_count - 1)
    density = np.bincount(indices, minlength=bin_count).astype(np.float64)

    if smoothing_sigma_bins > 0:
        radius = max(1, int(np.ceil(3 * smoothing_sigma_bins)))
        offsets = np.arange(-radius, radius + 1, dtype=np.float64)
        kernel = np.exp(-0.5 * np.square(offsets / smoothing_sigma_bins))
        kernel /= kernel.sum()
        density = np.convolve(density, kernel, mode="same")

    positive = density[density > 0]
    reference_density = float(np.median(positive))
    stable_density = np.maximum(density, max(float(positive.min()), 1e-12))
    weights = np.power(reference_density / stable_density, power)
    weights /= float(weights[indices].mean())
    weights = np.clip(weights, minimum_weight, maximum_weight)

    sample_weights = weights[indices]
    quantiles = tuple(
        float(value)
        for value in np.quantile(sample_weights, [0.0, 0.5, 0.9, 0.99, 1.0])
    )
    return DensityWeightSpec(
        lower_edge=lower_edge,
        bin_width=float(bin_width),
        bin_weights=weights.astype(np.float32),
        sample_weight_quantiles=quantiles,
    )


class LabelDensityWeightedMSELoss(RegressionLoss):
    """MSE with separate inverse-density weights for SBP and DBP labels."""

    def __init__(
        self,
        train_targets: np.ndarray,
        *,
        bin_width: float = 5.0,
        power: float = 0.5,
        smoothing_sigma_bins: float = 1.0,
        minimum_weight: float = 0.25,
        maximum_weight: float = 4.0,
    ) -> None:
        super().__init__()
        train_targets = np.asarray(train_targets, dtype=np.float32)
        if train_targets.ndim != 2 or train_targets.shape[1] != 2:
            raise ValueError("train_targets must have shape [samples, 2]")

        self.sbp_spec = label_density_weights(
            train_targets[:, 0],
            bin_width=bin_width,
            power=power,
            smoothing_sigma_bins=smoothing_sigma_bins,
            minimum_weight=minimum_weight,
            maximum_weight=maximum_weight,
        )
        self.dbp_spec = label_density_weights(
            train_targets[:, 1],
            bin_width=bin_width,
            power=power,
            smoothing_sigma_bins=smoothing_sigma_bins,
            minimum_weight=minimum_weight,
            maximum_weight=maximum_weight,
        )
        self.register_buffer(
            "sbp_bin_weights", torch.from_numpy(self.sbp_spec.bin_weights)
        )
        self.register_buffer(
            "dbp_bin_weights", torch.from_numpy(self.dbp_spec.bin_weights)
        )

    @staticmethod
    def _lookup(
        values: torch.Tensor,
        weights: torch.Tensor,
        lower_edge: float,
        bin_width: float,
    ) -> torch.Tensor:
        indices = torch.floor((values - lower_edge) / bin_width).to(torch.long)
        return weights[indices.clamp(0, weights.numel() - 1)]

    def forward(
        self,
        predictions: torch.Tensor,
        normalized_targets: torch.Tensor,
        raw_targets: torch.Tensor,
    ) -> torch.Tensor:
        sbp_weights = self._lookup(
            raw_targets[:, 0],
            self.sbp_bin_weights,
            self.sbp_spec.lower_edge,
            self.sbp_spec.bin_width,
        )
        dbp_weights = self._lookup(
            raw_targets[:, 1],
            self.dbp_bin_weights,
            self.dbp_spec.lower_edge,
            self.dbp_spec.bin_width,
        )
        output_weights = torch.stack((sbp_weights, dbp_weights), dim=1)
        return (torch.square(predictions - normalized_targets) * output_weights).mean()

    def summary(self) -> dict[str, object]:
        return {
            "sbp_sample_weight_quantiles": list(
                self.sbp_spec.sample_weight_quantiles
            ),
            "dbp_sample_weight_quantiles": list(
                self.dbp_spec.sample_weight_quantiles
            ),
        }


def build_regression_loss(
    config: dict,
    train_targets: np.ndarray,
) -> tuple[RegressionLoss, str, dict[str, object]]:
    """Build a configured loss and return compact reproducibility metadata."""

    loss_config = config.get("loss", {"name": "mse"})
    if isinstance(loss_config, str):
        loss_config = {"name": loss_config}
    loss_name = str(loss_config.get("name", "mse")).lower()

    if loss_name == "mse":
        return MeanSquaredRegressionLoss(), loss_name, {}
    if loss_name == "huber":
        delta = float(loss_config.get("delta", 5.0))
        return HuberRegressionLoss(delta), loss_name, {"delta": delta}
    if loss_name == "label_density_mse":
        criterion = LabelDensityWeightedMSELoss(
            train_targets,
            bin_width=float(loss_config.get("bin_width", 5.0)),
            power=float(loss_config.get("power", 0.5)),
            smoothing_sigma_bins=float(
                loss_config.get("smoothing_sigma_bins", 1.0)
            ),
            minimum_weight=float(loss_config.get("minimum_weight", 0.25)),
            maximum_weight=float(loss_config.get("maximum_weight", 4.0)),
        )
        return criterion, loss_name, criterion.summary()
    raise ValueError(f"Unsupported loss: {loss_name}")
