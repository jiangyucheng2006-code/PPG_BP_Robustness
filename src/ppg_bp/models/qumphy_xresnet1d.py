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


class MultiScaleInputStem(nn.Module):
    """Fuse short-, medium-, and long-range PPG patterns before XResNet."""

    def __init__(
        self,
        input_channels: int,
        *,
        branch_channels: int = 16,
        output_channels: int = 32,
        kernel_sizes: Sequence[int] = (3, 7, 15),
    ) -> None:
        super().__init__()
        self.branches = nn.ModuleList(
            [
                BenchmarkConvLayer(
                    input_channels,
                    branch_channels,
                    kernel_size,
                    stride=2,
                )
                for kernel_size in kernel_sizes
            ]
        )
        self.fusion = BenchmarkConvLayer(
            branch_channels * len(kernel_sizes),
            output_channels,
            1,
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        features = torch.cat([branch(inputs) for branch in self.branches], dim=1)
        return self.fusion(features)


class AdaptiveScaleAttentionStem(MultiScaleInputStem):
    """Weight PPG receptive-field scales independently for every segment."""

    def __init__(
        self,
        input_channels: int,
        *,
        branch_channels: int = 16,
        output_channels: int = 32,
        kernel_sizes: Sequence[int] = (3, 7, 15),
        attention_hidden: int = 24,
    ) -> None:
        super().__init__(
            input_channels,
            branch_channels=branch_channels,
            output_channels=output_channels,
            kernel_sizes=kernel_sizes,
        )
        descriptor_channels = branch_channels * len(kernel_sizes) * 2
        self.scale_attention = nn.Sequential(
            nn.Linear(descriptor_channels, attention_hidden),
            nn.ReLU(),
            nn.Linear(attention_hidden, len(kernel_sizes)),
        )

    def extract_scale_features(
        self,
        inputs: torch.Tensor,
    ) -> tuple[list[torch.Tensor], torch.Tensor]:
        features = [branch(inputs) for branch in self.branches]
        descriptor = torch.cat(
            [
                statistic
                for feature in features
                for statistic in (feature.mean(dim=-1), feature.amax(dim=-1))
            ],
            dim=1,
        )
        weights = torch.softmax(self.scale_attention(descriptor), dim=1)
        return features, weights

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        features, weights = self.extract_scale_features(inputs)
        scale_count = len(features)
        weighted = [
            feature * weights[:, index, None, None] * scale_count
            for index, feature in enumerate(features)
        ]
        return self.fusion(torch.cat(weighted, dim=1))


class QumphyXResNet1D(nn.Module):
    """Benchmark-compatible 1-D XResNet for two-output BP regression."""

    def __init__(
        self,
        layers: Sequence[int],
        *,
        input_channels: int = 1,
        outputs: int = 2,
        dropout: float = 0.5,
        multiscale_stem: bool = False,
        adaptive_scale_attention: bool = False,
    ) -> None:
        super().__init__()
        first_stem: nn.Module
        if adaptive_scale_attention:
            first_stem = AdaptiveScaleAttentionStem(input_channels)
        elif multiscale_stem:
            first_stem = MultiScaleInputStem(input_channels)
        else:
            first_stem = BenchmarkConvLayer(input_channels, 32, 5, stride=2)
        self.stem = nn.Sequential(
            first_stem,
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
        if isinstance(first_stem, AdaptiveScaleAttentionStem):
            output_layer = first_stem.scale_attention[-1]
            assert isinstance(output_layer, nn.Linear)
            nn.init.zeros_(output_layer.weight)
            nn.init.zeros_(output_layer.bias)

    def _initialize_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, (nn.Conv1d, nn.Linear)):
                nn.init.kaiming_normal_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward_feature_map(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.stages(self.stem(inputs))

    def forward_features(self, inputs: torch.Tensor) -> torch.Tensor:
        features = self.forward_feature_map(inputs)
        pooled = torch.cat((self.max_pool(features), self.avg_pool(features)), dim=1)
        return pooled.flatten(1)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.head(self.forward_features(inputs))


class TaskSpecificAttentionHead1D(nn.Module):
    """Channel-temporal attention and regression head for one BP target."""

    def __init__(
        self,
        channels: int,
        *,
        reduction: int = 8,
        temporal_kernel_size: int = 7,
        dropout: float = 0.5,
    ) -> None:
        super().__init__()
        hidden_channels = max(channels // reduction, 16)
        self.channel_attention = nn.Sequential(
            nn.Conv1d(channels, hidden_channels, 1, bias=False),
            nn.ReLU(),
            nn.Conv1d(hidden_channels, channels, 1),
        )
        self.temporal_attention = nn.Conv1d(
            2,
            1,
            temporal_kernel_size,
            padding=(temporal_kernel_size - 1) // 2,
        )
        self.max_pool = nn.AdaptiveMaxPool1d(1)
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.regressor = nn.Sequential(
            nn.BatchNorm1d(channels * 2),
            nn.Dropout(dropout),
            nn.Linear(channels * 2, 1),
        )

    def reset_attention_to_identity(self) -> None:
        channel_output = self.channel_attention[-1]
        assert isinstance(channel_output, nn.Conv1d)
        nn.init.zeros_(channel_output.weight)
        nn.init.zeros_(channel_output.bias)
        nn.init.zeros_(self.temporal_attention.weight)
        nn.init.zeros_(self.temporal_attention.bias)

    def attention_maps(
        self,
        features: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        average_descriptor = self.avg_pool(features)
        maximum_descriptor = self.max_pool(features)
        channel_logits = self.channel_attention(average_descriptor)
        channel_logits = channel_logits + self.channel_attention(maximum_descriptor)
        channel_scale = 0.5 + torch.sigmoid(channel_logits)

        channel_refined = features * channel_scale
        temporal_descriptor = torch.cat(
            (
                channel_refined.mean(dim=1, keepdim=True),
                channel_refined.amax(dim=1, keepdim=True),
            ),
            dim=1,
        )
        temporal_scale = 0.5 + torch.sigmoid(
            self.temporal_attention(temporal_descriptor)
        )
        return channel_scale, temporal_scale

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        channel_scale, temporal_scale = self.attention_maps(features)
        attended = features * channel_scale * temporal_scale
        pooled = torch.cat((self.max_pool(attended), self.avg_pool(attended)), dim=1)
        return self.regressor(pooled.flatten(1))


class QumphyTaskAttentionXResNet1D(QumphyXResNet1D):
    """Multiscale XResNet with separate SBP and DBP attention heads."""

    def __init__(
        self,
        layers: Sequence[int],
        *,
        input_channels: int = 1,
        outputs: int = 2,
        dropout: float = 0.5,
    ) -> None:
        if outputs != 2:
            raise ValueError("Task-specific BP attention requires exactly two outputs")
        super().__init__(
            layers,
            input_channels=input_channels,
            outputs=outputs,
            dropout=dropout,
            multiscale_stem=True,
        )
        self.head = nn.Identity()
        self.task_heads = nn.ModuleList(
            [
                TaskSpecificAttentionHead1D(256, dropout=dropout),
                TaskSpecificAttentionHead1D(256, dropout=dropout),
            ]
        )
        for module in self.task_heads.modules():
            if isinstance(module, (nn.Conv1d, nn.Linear)):
                nn.init.kaiming_normal_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
        for task_head in self.task_heads:
            task_head.reset_attention_to_identity()

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        features = self.forward_feature_map(inputs)
        return torch.cat([head(features) for head in self.task_heads], dim=1)


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


def qumphy_multiscale_xresnet1d50(
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
        multiscale_stem=True,
    )


def qumphy_attention_multiscale_xresnet1d50(
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
        adaptive_scale_attention=True,
    )


def qumphy_task_attention_multiscale_xresnet1d50(
    *,
    input_channels: int = 1,
    outputs: int = 2,
    dropout: float = 0.5,
) -> QumphyTaskAttentionXResNet1D:
    return QumphyTaskAttentionXResNet1D(
        (3, 4, 6, 3),
        input_channels=input_channels,
        outputs=outputs,
        dropout=dropout,
    )
