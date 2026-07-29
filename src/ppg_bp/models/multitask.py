"""PPG-only physiology-guided multi-task model."""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import nn


class AuxiliaryHead(nn.Sequential):
    def __init__(
        self,
        input_features: int,
        outputs: int,
        *,
        hidden_features: int = 128,
        dropout: float = 0.2,
    ) -> None:
        super().__init__(
            nn.BatchNorm1d(input_features),
            nn.Dropout(dropout),
            nn.Linear(input_features, hidden_features),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_features, outputs),
        )
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight)
                nn.init.zeros_(module.bias)


class PhysiologyGuidedMultiTaskModel(nn.Module):
    """Attach physiological auxiliary heads to a pooled-feature BP backbone.

    Auxiliary labels guide the shared PPG representation during training.
    Inference still requires only the PPG waveform.
    """

    def __init__(
        self,
        backbone: nn.Module,
        tasks: Sequence[str],
        *,
        feature_dim: int = 512,
        age_classes: int = 4,
        bp_classes: int = 3,
        hidden_features: int = 128,
        auxiliary_dropout: float = 0.2,
    ) -> None:
        super().__init__()
        supported = {"heart_rate", "age_group", "bp_class"}
        unknown = set(tasks).difference(supported)
        if unknown:
            raise ValueError(f"Unsupported auxiliary tasks: {sorted(unknown)}")
        if not hasattr(backbone, "forward_features"):
            raise TypeError("The multi-task backbone must expose forward_features")
        self.backbone = backbone
        self.tasks = tuple(tasks)
        self.auxiliary_heads = nn.ModuleDict()
        output_sizes = {
            "heart_rate": 1,
            "age_group": age_classes,
            "bp_class": bp_classes,
        }
        for task in self.tasks:
            self.auxiliary_heads[task] = AuxiliaryHead(
                feature_dim,
                output_sizes[task],
                hidden_features=hidden_features,
                dropout=auxiliary_dropout,
            )

    def _blood_pressure_from_features(self, features: torch.Tensor) -> torch.Tensor:
        target_heads = getattr(self.backbone, "target_heads", None)
        if target_heads is not None:
            return torch.cat([head(features) for head in target_heads], dim=1)
        head = getattr(self.backbone, "head", None)
        if head is None or isinstance(head, nn.Identity):
            raise TypeError("Backbone does not expose a pooled-feature BP head")
        return head(features)

    def forward(self, inputs: torch.Tensor) -> dict[str, torch.Tensor]:
        features = self.backbone.forward_features(inputs)
        outputs = {"bp": self._blood_pressure_from_features(features)}
        for task, head in self.auxiliary_heads.items():
            outputs[task] = head(features)
        return outputs
