"""XResNet1D architecture used by the QUMPHY PulseDB benchmark.

This implementation follows the published upstream topology while keeping the
project independent of the upstream training package. In particular, all four
residual stages use 64 bottleneck channels instead of the conventional
64/128/256/512 progression.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import nn


def _benchmark_batch_norm(channels: int, *, zero: bool = False) -> nn.BatchNorm1d:
    layer = nn.BatchNorm1d(channels)
    with torch.no_grad():
        layer.weight.fill_(0.0 if zero else 1.0)
        layer.bias.fill_(1e-3)
    return layer


class BenchmarkConvLayer(nn.Sequential):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        *,
        stride: int = 1,
        activation: bool = True,
        zero_norm: bool = False,
    ) -> None:
        padding = (kernel_size - 1) // 2
        layers: list[nn.Module] = [
            nn.Conv1d(
                in_channels,
                out_channels,
                kernel_size,
                stride=stride,
                padding=padding,
                bias=False,
            ),
            _benchmark_batch_norm(out_channels, zero=zero_norm),
        ]
        if activation:
            layers.append(nn.ReLU())
        super().__init__(*layers)


class BenchmarkBottleneck1D(nn.Module):
    expansion = 4

    def __init__(self, in_channels: int, channels: int, *, stride: int = 1) -> None:
        super().__init__()
        out_channels = channels * self.expansion
        self.main = nn.Sequential(
            BenchmarkConvLayer(in_channels, channels, 1),
            BenchmarkConvLayer(channels, channels, 5, stride=stride),
            BenchmarkConvLayer(channels, out_channels, 1, activation=False, zero_norm=True),
        )

        shortcut: list[nn.Module] = []
        if stride != 1:
            shortcut.append(nn.AvgPool1d(2, ceil_mode=True))
        if in_channels != out_channels:
            shortcut.append(BenchmarkConvLayer(in_channels, out_channels, 1, activation=False))
        self.shortcut = nn.Sequential(*shortcut)
        self.activation = nn.ReLU(inplace=True)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.activation(self.main(inputs) + self.shortcut(inputs))


class QumphyXResNet1D(nn.Module):
    """Benchmark-compatible 1-D XResNet for two-output BP regression."""

    def __init__(
        self,
        layers: Sequence[int],
        *,
        input_channels: int = 1,
        outputs: int = 2,
        dropout: float = 0.5,
    ) -> None:
        super().__init__()
        self.stem = nn.Sequential(
            BenchmarkConvLayer(input_channels, 32, 5, stride=2),
            BenchmarkConvLayer(32, 32, 5),
            BenchmarkConvLayer(32, 64, 5),
            nn.MaxPool1d(kernel_size=3, stride=2, padding=1),
        )

        in_channels = 64
        stages: list[nn.Module] = []
        for stage_index, block_count in enumerate(layers):
            blocks: list[nn.Module] = []
            for block_index in range(block_count):
                stride = 2 if stage_index > 0 and block_index == 0 else 1
                blocks.append(BenchmarkBottleneck1D(in_channels, 64, stride=stride))
                in_channels = 64 * BenchmarkBottleneck1D.expansion
            stages.append(nn.Sequential(*blocks))
        self.stages = nn.Sequential(*stages)

        self.max_pool = nn.AdaptiveMaxPool1d(1)
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.head = nn.Sequential(
            nn.BatchNorm1d(in_channels * 2),
            nn.Dropout(dropout),
            nn.Linear(in_channels * 2, outputs),
        )
        self._initialize_weights()

    def _initialize_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, (nn.Conv1d, nn.Linear)):
                nn.init.kaiming_normal_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward_features(self, inputs: torch.Tensor) -> torch.Tensor:
        features = self.stages(self.stem(inputs))
        pooled = torch.cat((self.max_pool(features), self.avg_pool(features)), dim=1)
        return pooled.flatten(1)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.head(self.forward_features(inputs))


def qumphy_xresnet1d50(
    *,
    input_channels: int = 1,
    outputs: int = 2,
    dropout: float = 0.5,
) -> QumphyXResNet1D:
    return QumphyXResNet1D(
        (3, 4, 6, 3),
        input_channels=input_channels,
        outputs=outputs,
        dropout=dropout,
    )


def qumphy_xresnet1d101(
    *,
    input_channels: int = 1,
    outputs: int = 2,
    dropout: float = 0.5,
) -> QumphyXResNet1D:
    return QumphyXResNet1D(
        (3, 4, 23, 3),
        input_channels=input_channels,
        outputs=outputs,
        dropout=dropout,
    )
