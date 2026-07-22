"""A clean 1-D XResNet-style regression backbone.

This is an independent PyTorch implementation of the deep-stem residual design
used as a strong PPG baseline in the 2025 OOD-generalization benchmark. It
accepts one PPG channel and jointly predicts SBP and DBP.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import nn


class ConvNormAct(nn.Sequential):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int = 1) -> None:
        padding = kernel_size // 2
        super().__init__(
            nn.Conv1d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=False),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True),
        )


class Bottleneck1D(nn.Module):
    expansion = 4

    def __init__(self, in_channels: int, channels: int, stride: int = 1, kernel_size: int = 5) -> None:
        super().__init__()
        out_channels = channels * self.expansion
        self.main = nn.Sequential(
            ConvNormAct(in_channels, channels, kernel_size=1),
            ConvNormAct(channels, channels, kernel_size=kernel_size, stride=stride),
            nn.Conv1d(channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm1d(out_channels),
        )
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.AvgPool1d(kernel_size=stride, stride=stride, ceil_mode=True) if stride > 1 else nn.Identity(),
                nn.Conv1d(in_channels, out_channels, kernel_size=1, bias=False),
                nn.BatchNorm1d(out_channels),
            )
        else:
            self.shortcut = nn.Identity()
        self.activation = nn.ReLU(inplace=True)

        nn.init.zeros_(self.main[-1].weight)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.activation(self.main(inputs) + self.shortcut(inputs))


class XResNet1D(nn.Module):
    def __init__(
        self,
        layers: Sequence[int],
        *,
        input_channels: int = 1,
        outputs: int = 2,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.stem = nn.Sequential(
            ConvNormAct(input_channels, 32, kernel_size=5, stride=2),
            ConvNormAct(32, 32, kernel_size=5),
            ConvNormAct(32, 64, kernel_size=5),
            nn.MaxPool1d(kernel_size=3, stride=2, padding=1),
        )

        stage_channels = (64, 128, 256, 512)
        in_channels = 64
        stages: list[nn.Module] = []
        for stage_index, (channels, block_count) in enumerate(zip(stage_channels, layers, strict=True)):
            blocks: list[nn.Module] = []
            for block_index in range(block_count):
                stride = 2 if stage_index > 0 and block_index == 0 else 1
                blocks.append(Bottleneck1D(in_channels, channels, stride=stride))
                in_channels = channels * Bottleneck1D.expansion
            stages.append(nn.Sequential(*blocks))
        self.stages = nn.Sequential(*stages)

        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.max_pool = nn.AdaptiveMaxPool1d(1)
        self.head = nn.Sequential(
            nn.BatchNorm1d(in_channels * 2),
            nn.Dropout(dropout),
            nn.Linear(in_channels * 2, outputs),
        )
        self._initialize_weights()

    def _initialize_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Conv1d):
                nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(module, nn.BatchNorm1d):
                if module.weight is not None and not torch.all(module.weight == 0):
                    nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, std=0.01)
                nn.init.zeros_(module.bias)

    def forward_features(self, inputs: torch.Tensor) -> torch.Tensor:
        features = self.stages(self.stem(inputs))
        return torch.cat((self.avg_pool(features), self.max_pool(features)), dim=1).flatten(1)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.head(self.forward_features(inputs))


def xresnet1d50(*, input_channels: int = 1, outputs: int = 2, dropout: float = 0.2) -> XResNet1D:
    return XResNet1D((3, 4, 6, 3), input_channels=input_channels, outputs=outputs, dropout=dropout)


def xresnet1d101(*, input_channels: int = 1, outputs: int = 2, dropout: float = 0.2) -> XResNet1D:
    return XResNet1D((3, 4, 23, 3), input_channels=input_channels, outputs=outputs, dropout=dropout)

