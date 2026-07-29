"""Batch PPG corruptions for robustness training and evaluation."""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import nn
from torch.nn import functional as F


LABEL_PRESERVING_TRANSFORMS = (
    "gaussian_noise",
    "powerline_noise",
    "baseline_drift",
    "motion_artifact",
    "spike_artifact",
    "random_mask",
    "clipping",
    "contact_compression",
    "sensor_gain_offset",
    "quantization",
    "lowpass_response",
    "respiratory_modulation",
)

STRUCTURAL_TRANSFORMS = (
    "amplitude_inversion",
    "time_reversal",
    "segment_shuffle",
    "time_scaling",
)

ALL_STP_TRANSFORMS = LABEL_PRESERVING_TRANSFORMS + STRUCTURAL_TRANSFORMS


def rebuild_derivative_channels(ppg: torch.Tensor, channels: int) -> torch.Tensor:
    """Recalculate VPG/APG after perturbing the raw PPG channel."""

    outputs = [ppg]
    if channels >= 2:
        vpg = torch.empty_like(ppg)
        vpg[:, 1:-1] = 0.5 * (ppg[:, 2:] - ppg[:, :-2])
        vpg[:, 0] = ppg[:, 1] - ppg[:, 0]
        vpg[:, -1] = ppg[:, -1] - ppg[:, -2]
        vpg = (vpg - vpg.mean(dim=1, keepdim=True)) / vpg.std(
            dim=1,
            keepdim=True,
        ).clamp_min(1e-6)
        outputs.append(vpg)
    if channels >= 3:
        apg = torch.empty_like(ppg)
        apg[:, 1:-1] = 0.5 * (vpg[:, 2:] - vpg[:, :-2])
        apg[:, 0] = vpg[:, 1] - vpg[:, 0]
        apg[:, -1] = vpg[:, -1] - vpg[:, -2]
        apg = (apg - apg.mean(dim=1, keepdim=True)) / apg.std(
            dim=1,
            keepdim=True,
        ).clamp_min(1e-6)
        outputs.append(apg)
    return torch.stack(outputs, dim=1)


class PPGBatchAugmenter:
    """Apply one or more controlled perturbations to PPG batches."""

    def __init__(
        self,
        transforms: Sequence[str],
        *,
        sampling_rate: float = 125.0,
        probability: float = 0.8,
        maximum_transforms: int = 1,
        minimum_severity: float = 0.25,
        maximum_severity: float = 1.0,
    ) -> None:
        transforms = tuple(transforms)
        unknown = set(transforms).difference(ALL_STP_TRANSFORMS)
        if unknown:
            raise ValueError(f"Unsupported PPG transforms: {sorted(unknown)}")
        if not transforms:
            raise ValueError("At least one transform is required")
        if not 0.0 <= probability <= 1.0:
            raise ValueError("probability must be between zero and one")
        if not 1 <= maximum_transforms <= len(transforms):
            raise ValueError("maximum_transforms must fit the transform list")
        if not 0.0 <= minimum_severity <= maximum_severity <= 1.0:
            raise ValueError("severity range must be within [0, 1]")
        self.transforms = transforms
        self.sampling_rate = float(sampling_rate)
        self.probability = float(probability)
        self.maximum_transforms = int(maximum_transforms)
        self.minimum_severity = float(minimum_severity)
        self.maximum_severity = float(maximum_severity)

    @property
    def class_count(self) -> int:
        return len(self.transforms) + 1

    def transform_name(self, class_index: int) -> str:
        if class_index == 0:
            return "clean"
        return self.transforms[class_index - 1]

    @staticmethod
    def _generator(device: torch.device, seed: int | None) -> torch.Generator | None:
        if seed is None:
            return None
        generator = torch.Generator(device=device)
        generator.manual_seed(seed)
        return generator

    @staticmethod
    def _range(ppg: torch.Tensor) -> torch.Tensor:
        return (ppg.amax(dim=1) - ppg.amin(dim=1)).clamp_min(1e-3)

    def __call__(
        self,
        inputs: torch.Tensor,
        *,
        forced_transform: str | None = None,
        forced_severity: float | None = None,
        seed: int | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if inputs.ndim != 3:
            raise ValueError("Expected inputs with shape [batch, channels, samples]")
        device = inputs.device
        generator = self._generator(device, seed)
        ppg = inputs[:, 0, :].clone()
        batch = len(ppg)
        labels = torch.zeros(batch, dtype=torch.long, device=device)
        severities = torch.zeros(batch, dtype=ppg.dtype, device=device)

        if forced_transform is not None:
            if forced_transform not in self.transforms:
                raise ValueError(f"Transform not configured: {forced_transform}")
            indices = torch.full(
                (batch,),
                self.transforms.index(forced_transform),
                dtype=torch.long,
                device=device,
            )
            apply = torch.ones(batch, dtype=torch.bool, device=device)
            severity = torch.full(
                (batch,),
                float(forced_severity if forced_severity is not None else 1.0),
                dtype=ppg.dtype,
                device=device,
            )
            ppg = self._apply_selected(ppg, indices, severity, apply, generator)
            labels[:] = indices + 1
            severities[:] = severity
            return rebuild_derivative_channels(ppg, inputs.shape[1]), labels, severities

        apply = torch.rand(batch, device=device, generator=generator) < self.probability
        transform_count = torch.randint(
            1,
            self.maximum_transforms + 1,
            (batch,),
            device=device,
            generator=generator,
        )
        severity = self.minimum_severity + (
            self.maximum_severity - self.minimum_severity
        ) * torch.rand(batch, device=device, generator=generator)
        for pass_index in range(self.maximum_transforms):
            pass_apply = apply & (transform_count > pass_index)
            indices = torch.randint(
                0,
                len(self.transforms),
                (batch,),
                device=device,
                generator=generator,
            )
            ppg = self._apply_selected(
                ppg,
                indices,
                severity,
                pass_apply,
                generator,
            )
            if pass_index == 0:
                labels[pass_apply] = indices[pass_apply] + 1
                severities[pass_apply] = severity[pass_apply]
        return rebuild_derivative_channels(ppg, inputs.shape[1]), labels, severities

    def _apply_selected(
        self,
        ppg: torch.Tensor,
        indices: torch.Tensor,
        severity: torch.Tensor,
        apply: torch.Tensor,
        generator: torch.Generator | None,
    ) -> torch.Tensor:
        output = ppg
        for transform_index, name in enumerate(self.transforms):
            selected = apply & (indices == transform_index)
            if not bool(selected.any()):
                continue
            transformed = getattr(self, f"_{name}")(
                output[selected],
                severity[selected],
                generator,
            )
            output = output.clone()
            output[selected] = transformed
        return output

    def _gaussian_noise(
        self,
        ppg: torch.Tensor,
        severity: torch.Tensor,
        generator: torch.Generator | None,
    ) -> torch.Tensor:
        noise = torch.randn(
            ppg.shape,
            dtype=ppg.dtype,
            device=ppg.device,
            generator=generator,
        )
        return ppg + noise * (0.005 + 0.045 * severity)[:, None] * self._range(ppg)[:, None]

    def _powerline_noise(
        self,
        ppg: torch.Tensor,
        severity: torch.Tensor,
        generator: torch.Generator | None,
    ) -> torch.Tensor:
        batch, samples = ppg.shape
        frequency = torch.where(
            torch.rand(batch, device=ppg.device, generator=generator) < 0.5,
            50.0,
            60.0,
        )
        phase = 2.0 * torch.pi * torch.rand(
            batch,
            device=ppg.device,
            generator=generator,
        )
        time = torch.arange(samples, device=ppg.device) / self.sampling_rate
        wave = torch.sin(
            2.0 * torch.pi * frequency[:, None] * time[None, :]
            + phase[:, None]
        )
        return ppg + wave * (0.003 + 0.027 * severity)[:, None] * self._range(ppg)[:, None]

    def _baseline_drift(
        self,
        ppg: torch.Tensor,
        severity: torch.Tensor,
        generator: torch.Generator | None,
    ) -> torch.Tensor:
        batch, samples = ppg.shape
        frequency = 0.05 + 0.45 * torch.rand(
            batch,
            device=ppg.device,
            generator=generator,
        )
        phase = 2.0 * torch.pi * torch.rand(
            batch,
            device=ppg.device,
            generator=generator,
        )
        time = torch.arange(samples, device=ppg.device) / self.sampling_rate
        drift = torch.sin(
            2.0 * torch.pi * frequency[:, None] * time[None, :]
            + phase[:, None]
        )
        return ppg + drift * (0.02 + 0.18 * severity)[:, None] * self._range(ppg)[:, None]

    def _motion_artifact(
        self,
        ppg: torch.Tensor,
        severity: torch.Tensor,
        generator: torch.Generator | None,
    ) -> torch.Tensor:
        noise = torch.randn(
            ppg.shape,
            dtype=ppg.dtype,
            device=ppg.device,
            generator=generator,
        )
        colored = F.avg_pool1d(
            noise[:, None, :],
            kernel_size=25,
            stride=1,
            padding=12,
        )[:, 0, :]
        colored = colored / colored.std(dim=1, keepdim=True).clamp_min(1e-6)
        return ppg + colored * (0.03 + 0.22 * severity)[:, None] * self._range(ppg)[:, None]

    def _spike_artifact(
        self,
        ppg: torch.Tensor,
        severity: torch.Tensor,
        generator: torch.Generator | None,
    ) -> torch.Tensor:
        output = ppg.clone()
        samples = ppg.shape[1]
        radius = 8
        offsets = torch.arange(-radius, radius + 1, device=ppg.device)
        kernel = torch.exp(-0.5 * (offsets / 2.5).square())
        for row in range(len(ppg)):
            count = 1 + int(float(severity[row]) * 3)
            centers = torch.randint(
                radius,
                samples - radius,
                (count,),
                device=ppg.device,
                generator=generator,
            )
            for center in centers:
                sign = -1.0 if bool(
                    torch.rand((), device=ppg.device, generator=generator) < 0.5
                ) else 1.0
                output[row, center - radius : center + radius + 1] += (
                    sign
                    * kernel
                    * (0.1 + 0.4 * severity[row])
                    * self._range(ppg[row : row + 1])[0]
                )
        return output

    def _random_mask(
        self,
        ppg: torch.Tensor,
        severity: torch.Tensor,
        generator: torch.Generator | None,
    ) -> torch.Tensor:
        output = ppg.clone()
        samples = ppg.shape[1]
        for row in range(len(ppg)):
            length = max(2, int((0.01 + 0.12 * float(severity[row])) * samples))
            start = int(
                torch.randint(
                    0,
                    max(1, samples - length),
                    (),
                    device=ppg.device,
                    generator=generator,
                )
            )
            left = output[row, max(start - 1, 0)]
            right = output[row, min(start + length, samples - 1)]
            output[row, start : start + length] = torch.linspace(
                float(left),
                float(right),
                length,
                device=ppg.device,
            )
        return output

    def _clipping(
        self,
        ppg: torch.Tensor,
        severity: torch.Tensor,
        generator: torch.Generator | None,
    ) -> torch.Tensor:
        del generator
        fraction = 0.005 + 0.095 * severity
        output = ppg.clone()
        for row in range(len(ppg)):
            lower = torch.quantile(ppg[row], fraction[row])
            upper = torch.quantile(ppg[row], 1.0 - fraction[row])
            output[row] = ppg[row].clamp(lower, upper)
        return output

    def _contact_compression(
        self,
        ppg: torch.Tensor,
        severity: torch.Tensor,
        generator: torch.Generator | None,
    ) -> torch.Tensor:
        minimum = ppg.amin(dim=1, keepdim=True)
        scale = self._range(ppg)[:, None]
        normalized = ((ppg - minimum) / scale).clamp(0.0, 1.0)
        direction = torch.where(
            torch.rand(len(ppg), device=ppg.device, generator=generator) < 0.5,
            -1.0,
            1.0,
        )
        gamma = (1.0 + direction * 0.55 * severity).clamp(0.45, 1.55)
        compressed = normalized.pow(gamma[:, None])
        smoothed = F.avg_pool1d(
            compressed[:, None, :],
            kernel_size=5,
            stride=1,
            padding=2,
        )[:, 0, :]
        mixed = compressed * (1.0 - 0.45 * severity[:, None]) + smoothed * (
            0.45 * severity[:, None]
        )
        gain = 1.0 - 0.4 * severity
        return minimum + scale * gain[:, None] * mixed

    def _sensor_gain_offset(
        self,
        ppg: torch.Tensor,
        severity: torch.Tensor,
        generator: torch.Generator | None,
    ) -> torch.Tensor:
        random = 2.0 * torch.rand(
            (len(ppg), 2),
            device=ppg.device,
            generator=generator,
        ) - 1.0
        gain = 1.0 + random[:, 0] * 0.35 * severity
        offset = random[:, 1] * 0.12 * severity * self._range(ppg)
        return ppg * gain[:, None] + offset[:, None]

    def _quantization(
        self,
        ppg: torch.Tensor,
        severity: torch.Tensor,
        generator: torch.Generator | None,
    ) -> torch.Tensor:
        del generator
        minimum = ppg.amin(dim=1, keepdim=True)
        scale = self._range(ppg)[:, None]
        levels = (256.0 - 240.0 * severity).round().clamp_min(16.0)
        normalized = (ppg - minimum) / scale
        return minimum + scale * (
            torch.round(normalized * (levels[:, None] - 1.0))
            / (levels[:, None] - 1.0)
        )

    def _lowpass_response(
        self,
        ppg: torch.Tensor,
        severity: torch.Tensor,
        generator: torch.Generator | None,
    ) -> torch.Tensor:
        del generator
        smoothed = F.avg_pool1d(
            ppg[:, None, :],
            kernel_size=7,
            stride=1,
            padding=3,
        )[:, 0, :]
        return ppg * (1.0 - 0.8 * severity[:, None]) + smoothed * (
            0.8 * severity[:, None]
        )

    def _respiratory_modulation(
        self,
        ppg: torch.Tensor,
        severity: torch.Tensor,
        generator: torch.Generator | None,
    ) -> torch.Tensor:
        batch, samples = ppg.shape
        frequency = 0.1 + 0.3 * torch.rand(
            batch,
            device=ppg.device,
            generator=generator,
        )
        phase = 2.0 * torch.pi * torch.rand(
            batch,
            device=ppg.device,
            generator=generator,
        )
        time = torch.arange(samples, device=ppg.device) / self.sampling_rate
        modulation = 1.0 + (0.02 + 0.15 * severity)[:, None] * torch.sin(
            2.0 * torch.pi * frequency[:, None] * time[None, :]
            + phase[:, None]
        )
        return ppg * modulation

    def _amplitude_inversion(
        self,
        ppg: torch.Tensor,
        severity: torch.Tensor,
        generator: torch.Generator | None,
    ) -> torch.Tensor:
        del severity, generator
        return ppg.amax(dim=1, keepdim=True) + ppg.amin(
            dim=1,
            keepdim=True,
        ) - ppg

    def _time_reversal(
        self,
        ppg: torch.Tensor,
        severity: torch.Tensor,
        generator: torch.Generator | None,
    ) -> torch.Tensor:
        del severity, generator
        return torch.flip(ppg, dims=(1,))

    def _segment_shuffle(
        self,
        ppg: torch.Tensor,
        severity: torch.Tensor,
        generator: torch.Generator | None,
    ) -> torch.Tensor:
        del severity
        output = ppg.clone()
        chunk_count = 5
        for row in range(len(ppg)):
            chunks = torch.tensor_split(ppg[row], chunk_count)
            order = torch.randperm(
                chunk_count,
                device=ppg.device,
                generator=generator,
            )
            output[row] = torch.cat([chunks[int(index)] for index in order])
        return output

    def _time_scaling(
        self,
        ppg: torch.Tensor,
        severity: torch.Tensor,
        generator: torch.Generator | None,
    ) -> torch.Tensor:
        output = torch.empty_like(ppg)
        samples = ppg.shape[1]
        direction = torch.where(
            torch.rand(len(ppg), device=ppg.device, generator=generator) < 0.5,
            -1.0,
            1.0,
        )
        factors = 1.0 + direction * (0.08 + 0.22 * severity)
        for row in range(len(ppg)):
            scaled_length = max(8, int(samples * float(factors[row])))
            scaled = F.interpolate(
                ppg[row : row + 1, None, :],
                size=scaled_length,
                mode="linear",
                align_corners=False,
            )[0, 0]
            if scaled_length >= samples:
                start = (scaled_length - samples) // 2
                output[row] = scaled[start : start + samples]
            else:
                padding = samples - scaled_length
                left_padding = padding // 2
                right_padding = padding - left_padding
                output[row] = torch.cat(
                    (
                        scaled[0].repeat(left_padding),
                        scaled,
                        scaled[-1].repeat(right_padding),
                    )
                )
        return output


def build_augmenter(config: dict, *, sampling_rate: float) -> PPGBatchAugmenter:
    transforms = config.get("transforms", LABEL_PRESERVING_TRANSFORMS)
    return PPGBatchAugmenter(
        transforms,
        sampling_rate=sampling_rate,
        probability=float(config.get("probability", 0.8)),
        maximum_transforms=int(config.get("maximum_transforms", 1)),
        minimum_severity=float(config.get("minimum_severity", 0.25)),
        maximum_severity=float(config.get("maximum_severity", 1.0)),
    )
