"""Artifact-aware wrapper for PPG blood-pressure backbones."""

from __future__ import annotations

import torch
from torch import nn


class ArtifactAwareBPModel(nn.Module):
    """BP regression with optional artifact-type and severity heads."""

    def __init__(
        self,
        backbone: nn.Module,
        *,
        feature_dim: int = 512,
        artifact_classes: int = 0,
        hidden_features: int = 128,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        if not hasattr(backbone, "forward_features"):
            raise TypeError("Artifact-aware backbone must expose forward_features")
        self.backbone = backbone
        self.artifact_classes = int(artifact_classes)
        if self.artifact_classes > 0:
            self.artifact_head = nn.Sequential(
                nn.BatchNorm1d(feature_dim),
                nn.Dropout(dropout),
                nn.Linear(feature_dim, hidden_features),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),
                nn.Linear(hidden_features, self.artifact_classes),
            )
            self.severity_head = nn.Sequential(
                nn.BatchNorm1d(feature_dim),
                nn.Dropout(dropout),
                nn.Linear(feature_dim, hidden_features),
                nn.ReLU(inplace=True),
                nn.Linear(hidden_features, 1),
                nn.Sigmoid(),
            )
            for module in (self.artifact_head, self.severity_head):
                for layer in module.modules():
                    if isinstance(layer, nn.Linear):
                        nn.init.kaiming_normal_(layer.weight)
                        nn.init.zeros_(layer.bias)
        else:
            self.artifact_head = None
            self.severity_head = None

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
        if self.artifact_head is not None and self.severity_head is not None:
            outputs["artifact_type"] = self.artifact_head(features)
            outputs["artifact_severity"] = self.severity_head(features).flatten()
        return outputs
