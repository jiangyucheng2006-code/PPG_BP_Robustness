"""The eleven signal transformations illustrated in the STP paper.

Each transformed signal is paired with the original signal as the
reconstruction target.  This differs from the earlier C6 classification
control, which used transformation identities as pseudo-labels.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch.nn import functional as F


STP_RECONSTRUCTION_TRANSFORMS = (
    "gaussian_noise",
    "powerline_noise",
    "motion_artifacts",
    "baseline_drift",
    "rsa",
    "random_mask",
    "hard_clipping",
    "amplitude_negation",
    "temporal_inversion",
    "temporal_permutation",
    "temporal_warping",
)


class STPTransformBank:
    """Apply one paper-defined transformation to each PPG in a batch."""

    def __init__(
        self,
        transforms: Sequence[str] = STP_RECONSTRUCTION_TRANSFORMS,
        *,
        sampling_rate: float = 125.0,
        minimum_severity: float = 0.25,
        maximum_severity: float = 1.0,
        include_identity: bool = False,
        paper_disclosed: bool = False,
    ) -> None:
        self.transforms = tuple(transforms)
        unknown = set(self.transforms).difference(STP_RECONSTRUCTION_TRANSFORMS)
        if unknown:
            raise ValueError(f"Unknown STP transformations: {sorted(unknown)}")
        if not self.transforms:
            raise ValueError("At least one transformation is required")
        if not 0 <= minimum_severity <= maximum_severity <= 1:
            raise ValueError("severity values must lie in [0, 1]")
        self.sampling_rate = float(sampling_rate)
        self.minimum_severity = float(minimum_severity)
        self.maximum_severity = float(maximum_severity)
        self.include_identity = bool(include_identity)
        self.paper_disclosed = bool(paper_disclosed)

    @staticmethod
    def _generator(device: torch.device, seed: int | None) -> torch.Generator | None:
        if seed is None:
            return None
        generator = torch.Generator(device=device)
        generator.manual_seed(seed)
        return generator

    @staticmethod
    def _range(signal: torch.Tensor) -> torch.Tensor:
        return (signal.amax(dim=1) - signal.amin(dim=1)).clamp_min(1e-6)

    def __call__(
        self,
        clean: torch.Tensor,
        *,
        forced_transform: str | None = None,
        forced_severity: float | None = None,
        seed: int | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if clean.ndim != 3 or clean.shape[1] != 1:
            raise ValueError("STP reconstruction expects [batch, 1, samples]")
        device = clean.device
        generator = self._generator(device, seed)
        ppg = clean[:, 0].clone()
        batch = len(ppg)
        severity = (
            torch.full(
                (batch,),
                float(forced_severity),
                dtype=ppg.dtype,
                device=device,
            )
            if forced_severity is not None
            else self.minimum_severity
            + (self.maximum_severity - self.minimum_severity)
            * torch.rand(batch, device=device, generator=generator)
        )

        if forced_transform is not None:
            if forced_transform not in self.transforms:
                raise ValueError(f"Transformation not configured: {forced_transform}")
            indices = torch.full(
                (batch,),
                self.transforms.index(forced_transform),
                dtype=torch.long,
                device=device,
            )
        else:
            identity_offset = 1 if self.include_identity else 0
            indices = torch.randint(
                -identity_offset,
                len(self.transforms),
                (batch,),
                device=device,
                generator=generator,
            )

        transformed = ppg.clone()
        for index, name in enumerate(self.transforms):
            selected = indices == index
            if bool(selected.any()):
                transformed[selected] = getattr(self, f"_{name}")(
                    ppg[selected],
                    severity[selected],
                    generator,
                )
        return transformed[:, None], clean, indices

    def _gaussian_noise(self, x, severity, generator):
        noise = torch.randn(
            x.shape,
            dtype=x.dtype,
            device=x.device,
            generator=generator,
        )
        return x + noise * (0.005 + 0.045 * severity)[:, None] * self._range(x)[:, None]

    def _powerline_noise(self, x, severity, generator):
        batch, samples = x.shape
        if self.paper_disclosed:
            frequency = torch.full((batch,), 50.0, device=x.device)
        else:
            frequency = torch.where(
                torch.rand(batch, device=x.device, generator=generator) < 0.5,
                50.0,
                60.0,
            )
        phase = 2 * torch.pi * torch.rand(batch, device=x.device, generator=generator)
        time = torch.arange(samples, device=x.device) / self.sampling_rate
        wave = torch.sin(2 * torch.pi * frequency[:, None] * time + phase[:, None])
        return x + wave * (0.003 + 0.027 * severity)[:, None] * self._range(x)[:, None]

    def _motion_artifacts(self, x, severity, generator):
        noise = torch.randn(
            x.shape,
            dtype=x.dtype,
            device=x.device,
            generator=generator,
        )
        low_frequency = F.avg_pool1d(
            noise[:, None],
            kernel_size=25,
            stride=1,
            padding=12,
        )[:, 0]
        low_frequency /= low_frequency.std(dim=1, keepdim=True).clamp_min(1e-6)
        impulse_mask = (
            torch.rand(x.shape, device=x.device, generator=generator)
            < (0.001 + 0.004 * severity[:, None])
        )
        impulses = (
            torch.randn(x.shape, device=x.device, generator=generator)
            * impulse_mask
            * self._range(x)[:, None]
        )
        corrupted = (
            x
            + low_frequency
            * (0.02 + 0.18 * severity)[:, None]
            * self._range(x)[:, None]
            + impulses * severity[:, None]
        )
        if not self.paper_disclosed:
            return corrupted
        output = x.clone()
        samples = x.shape[1]
        for row in range(len(x)):
            fraction = 0.01 + 0.09 * float(severity[row])
            length = max(1, int(round(samples * fraction)))
            start = int(
                torch.randint(
                    0,
                    max(1, samples - length + 1),
                    (),
                    device=x.device,
                    generator=generator,
                )
            )
            output[row, start : start + length] = corrupted[
                row,
                start : start + length,
            ]
        return output

    def _baseline_drift(self, x, severity, generator):
        batch, samples = x.shape
        time = torch.arange(samples, device=x.device) / self.sampling_rate
        frequency = (
            torch.full((batch,), 0.05, device=x.device)
            if self.paper_disclosed
            else 0.05
            + 0.45 * torch.rand(
                batch,
                device=x.device,
                generator=generator,
            )
        )
        phase = 2 * torch.pi * torch.rand(batch, device=x.device, generator=generator)
        drift = torch.sin(2 * torch.pi * frequency[:, None] * time + phase[:, None])
        slope = (
            2 * torch.rand(batch, device=x.device, generator=generator) - 1
        )[:, None] * torch.linspace(-0.5, 0.5, samples, device=x.device)
        drift = 0.7 * drift + 0.3 * slope
        return x + drift * (0.03 + 0.22 * severity)[:, None] * self._range(x)[:, None]

    def _rsa(self, x, severity, generator):
        """Approximate respiratory sinus arrhythmia by smooth time modulation."""

        batch, samples = x.shape
        if self.paper_disclosed:
            time = torch.arange(samples, device=x.device) / self.sampling_rate
            respiratory_frequency = 0.15 + 0.25 * torch.rand(
                batch,
                device=x.device,
                generator=generator,
            )
            phase = 2 * torch.pi * torch.rand(
                batch,
                device=x.device,
                generator=generator,
            )
            speed = 1 + 0.05 * severity[:, None] * torch.sin(
                2 * torch.pi * respiratory_frequency[:, None] * time
                + phase[:, None]
            )
            locations = torch.cumsum(speed, dim=1)
            locations = (locations - locations[:, :1]) / (
                locations[:, -1:] - locations[:, :1]
            ).clamp_min(1e-6)
            locations = locations * 2 - 1
            grid = torch.stack(
                (locations, torch.zeros_like(locations)),
                dim=-1,
            )[:, None]
            return F.grid_sample(
                x[:, None, None],
                grid,
                mode="bilinear",
                padding_mode="border",
                align_corners=True,
            )[:, 0, 0]
        control_points = 9
        random_curve = torch.randn(
            batch,
            1,
            control_points,
            device=x.device,
            generator=generator,
        )
        curve = F.interpolate(
            random_curve,
            size=samples,
            mode="linear",
            align_corners=True,
        )[:, 0]
        curve = torch.cumsum(curve, dim=1)
        curve = curve - curve[:, :1]
        curve /= curve[:, -1:].abs().clamp_min(1e-6)
        base = torch.linspace(-1, 1, samples, device=x.device)[None]
        locations = base + curve * (0.01 + 0.08 * severity)[:, None]
        locations = locations.clamp(-1, 1)
        grid = torch.stack((locations, torch.zeros_like(locations)), dim=-1)[:, None]
        warped = F.grid_sample(
            x[:, None, None],
            grid,
            mode="bilinear",
            padding_mode="border",
            align_corners=True,
        )[:, 0, 0]
        return warped

    def _random_mask(self, x, severity, generator):
        output = x.clone()
        samples = x.shape[1]
        for row in range(len(x)):
            if self.paper_disclosed:
                masked_probability = float(0.05 + 0.35 * severity[row])
                masked_mean = max(1.0, samples * 0.04)
                unmasked_mean = masked_mean * (1 - masked_probability) / max(
                    masked_probability,
                    1e-6,
                )
                position = 0
                masked = False
                while position < samples:
                    mean_length = masked_mean if masked else unmasked_mean
                    probability = min(1.0, 1.0 / max(mean_length, 1.0))
                    uniform = torch.rand(
                        (),
                        device=x.device,
                        generator=generator,
                    ).clamp_min(torch.finfo(x.dtype).eps)
                    run = int(
                        torch.floor(
                            torch.log(uniform)
                            / torch.log(
                                torch.tensor(
                                    1 - probability,
                                    device=x.device,
                                    dtype=x.dtype,
                                ).clamp_min(torch.finfo(x.dtype).eps)
                            )
                        )
                    ) + 1
                    stop = min(samples, position + run)
                    if masked:
                        output[row, position:stop] = 0
                    position = stop
                    masked = not masked
                continue
            length = max(2, int(samples * (0.03 + 0.15 * float(severity[row]))))
            start = int(
                torch.randint(
                    0,
                    max(1, samples - length + 1),
                    (),
                    device=x.device,
                    generator=generator,
                )
            )
            output[row, start : start + length] = 0
        return output

    def _hard_clipping(self, x, severity, generator):
        del generator
        output = x.clone()
        for row in range(len(x)):
            fraction = 0.01 + 0.19 * float(severity[row])
            upper = torch.quantile(x[row], 1 - fraction)
            output[row] = x[row].clamp_max(upper)
        return output

    def _amplitude_negation(self, x, severity, generator):
        del severity, generator
        return x.amax(dim=1, keepdim=True) + x.amin(dim=1, keepdim=True) - x

    def _temporal_inversion(self, x, severity, generator):
        del severity, generator
        return torch.flip(x, dims=(1,))

    def _temporal_permutation(self, x, severity, generator):
        output = x.clone()
        for row in range(len(x)):
            chunk_count = 3 + int(round(3 * float(severity[row])))
            chunks = torch.tensor_split(x[row], chunk_count)
            order = torch.randperm(
                chunk_count,
                device=x.device,
                generator=generator,
            )
            output[row] = torch.cat([chunks[int(index)] for index in order])
        return output

    def _temporal_warping(self, x, severity, generator):
        batch, samples = x.shape
        control_points = 6
        increments = 1 + (
            torch.rand(
                batch,
                control_points,
                device=x.device,
                generator=generator,
            )
            - 0.5
        ) * (0.2 + 0.8 * severity[:, None])
        cumulative = torch.cumsum(increments, dim=1)
        cumulative = (cumulative - cumulative[:, :1]) / (
            cumulative[:, -1:] - cumulative[:, :1]
        ).clamp_min(1e-6)
        curve = F.interpolate(
            cumulative[:, None],
            size=samples,
            mode="linear",
            align_corners=True,
        )[:, 0]
        locations = curve * 2 - 1
        grid = torch.stack((locations, torch.zeros_like(locations)), dim=-1)[:, None]
        return F.grid_sample(
            x[:, None, None],
            grid,
            mode="bilinear",
            padding_mode="border",
            align_corners=True,
        )[:, 0, 0]


def build_stp_transform_bank(
    config: dict,
    *,
    sampling_rate: float,
) -> STPTransformBank:
    return STPTransformBank(
        config.get("transforms", STP_RECONSTRUCTION_TRANSFORMS),
        sampling_rate=sampling_rate,
        minimum_severity=float(config.get("minimum_severity", 0.25)),
        maximum_severity=float(config.get("maximum_severity", 1.0)),
        include_identity=bool(config.get("include_identity", False)),
        paper_disclosed=bool(config.get("paper_disclosed", False)),
    )
