"""Transformer modules for the STP reproduction track.

The module follows the three stages shown in Ma et al. (2024):

1. transformed-PPG reconstruction;
2. blood-pressure pattern adaptation;
3. systolic/diastolic value estimation.

All stages share the same convolutional projection, positional embedding, and
Transformer encoder.  Encoder weights are transferred sequentially between
the stages.
"""

from __future__ import annotations

import math
from collections.abc import Mapping

import torch
from torch import nn
from torch.nn import functional as F


class _GradientReversalFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, inputs: torch.Tensor, strength: float) -> torch.Tensor:
        ctx.strength = float(strength)
        return inputs.view_as(inputs)

    @staticmethod
    def backward(ctx, gradient: torch.Tensor):
        return -ctx.strength * gradient, None


class GradientReversal(nn.Module):
    """Identity in the forward pass and sign reversal in backpropagation."""

    def __init__(self, strength: float = 1.0) -> None:
        super().__init__()
        self.strength = float(strength)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return _GradientReversalFunction.apply(inputs, self.strength)


class STPEncoder(nn.Module):
    """One-dimensional convolutional projection and Transformer encoder."""

    def __init__(
        self,
        *,
        input_channels: int = 1,
        signal_length: int = 512,
        patch_size: int = 8,
        embedding_dim: int = 128,
        encoder_layers: int = 4,
        attention_heads: int = 8,
        feedforward_dim: int = 256,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if signal_length % patch_size:
            raise ValueError("signal_length must be divisible by patch_size")
        if embedding_dim % attention_heads:
            raise ValueError("embedding_dim must be divisible by attention_heads")
        self.input_channels = int(input_channels)
        self.signal_length = int(signal_length)
        self.patch_size = int(patch_size)
        self.embedding_dim = int(embedding_dim)
        self.token_count = self.signal_length // self.patch_size

        self.projection = nn.Conv1d(
            self.input_channels,
            self.embedding_dim,
            kernel_size=self.patch_size,
            stride=self.patch_size,
        )
        self.position_embedding = nn.Parameter(
            torch.zeros(1, self.token_count, self.embedding_dim)
        )
        layer = nn.TransformerEncoderLayer(
            d_model=self.embedding_dim,
            nhead=attention_heads,
            dim_feedforward=feedforward_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            layer,
            num_layers=encoder_layers,
            norm=nn.LayerNorm(self.embedding_dim),
        )
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.trunc_normal_(self.position_embedding, std=0.02)

    def forward(self, signal: torch.Tensor) -> torch.Tensor:
        if signal.ndim != 3:
            raise ValueError("STP expects [batch, channels, samples]")
        if signal.shape[1] != self.input_channels:
            raise ValueError(
                f"Expected {self.input_channels} channels, got {signal.shape[1]}"
            )
        if signal.shape[-1] != self.signal_length:
            signal = F.interpolate(
                signal,
                size=self.signal_length,
                mode="linear",
                align_corners=False,
            )
        tokens = self.projection(signal).transpose(1, 2)
        tokens = tokens + self.position_embedding
        return self.transformer(tokens)

    @staticmethod
    def pool(tokens: torch.Tensor) -> torch.Tensor:
        return tokens.mean(dim=1)


class STPTokenPool(nn.Module):
    """Aggregate Transformer tokens without discarding beat variability."""

    def __init__(self, features: int, mode: str = "mean") -> None:
        super().__init__()
        self.mode = str(mode).lower()
        supported = {
            "mean",
            "statistics",
            "attention",
            "attentive_statistics",
        }
        if self.mode not in supported:
            raise ValueError(
                f"Unsupported STP pooling mode {mode!r}; expected {sorted(supported)}"
            )
        if self.mode in {"attention", "attentive_statistics"}:
            self.attention = nn.Sequential(
                nn.LayerNorm(features),
                nn.Linear(features, 1),
            )
        else:
            self.attention = None
        self.output_features = (
            features
            if self.mode in {"mean", "attention"}
            else 2 * features
        )

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        if self.attention is None:
            mean = tokens.mean(dim=1)
            if self.mode == "mean":
                return mean
            variance = (tokens - mean[:, None]).square().mean(dim=1)
        else:
            weights = self.attention(tokens).squeeze(-1).softmax(dim=1)
            mean = (tokens * weights[..., None]).sum(dim=1)
            if self.mode == "attention":
                return mean
            variance = (
                (tokens - mean[:, None]).square() * weights[..., None]
            ).sum(dim=1)
        standard_deviation = variance.clamp_min(1e-6).sqrt()
        return torch.cat((mean, standard_deviation), dim=1)


class STPSelfSupervisedModel(nn.Module):
    """Transformer encoder-decoder that reconstructs clean PPG."""

    def __init__(
        self,
        encoder: STPEncoder,
        *,
        decoder_layers: int = 2,
        attention_heads: int = 8,
        feedforward_dim: int = 256,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.encoder = encoder
        self.decoder_queries = nn.Parameter(
            torch.zeros(1, encoder.token_count, encoder.embedding_dim)
        )
        layer = nn.TransformerDecoderLayer(
            d_model=encoder.embedding_dim,
            nhead=attention_heads,
            dim_feedforward=feedforward_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.decoder = nn.TransformerDecoder(
            layer,
            num_layers=decoder_layers,
            norm=nn.LayerNorm(encoder.embedding_dim),
        )
        self.reconstruction_head = nn.Linear(
            encoder.embedding_dim,
            encoder.input_channels * encoder.patch_size,
        )
        nn.init.trunc_normal_(self.decoder_queries, std=0.02)

    def forward(self, transformed_ppg: torch.Tensor) -> torch.Tensor:
        memory = self.encoder(transformed_ppg)
        queries = self.decoder_queries.expand(len(transformed_ppg), -1, -1)
        queries = queries + self.encoder.position_embedding
        causal_mask = nn.Transformer.generate_square_subsequent_mask(
            self.encoder.token_count,
            device=queries.device,
            dtype=queries.dtype,
        )
        decoded = self.decoder(queries, memory, tgt_mask=causal_mask)
        patches = self.reconstruction_head(decoded)
        reconstructed = patches.reshape(
            len(transformed_ppg),
            self.encoder.token_count,
            self.encoder.input_channels,
            self.encoder.patch_size,
        )
        return reconstructed.permute(0, 2, 1, 3).reshape(
            len(transformed_ppg),
            self.encoder.input_channels,
            self.encoder.signal_length,
        )


class STPPatchDiscriminator(nn.Module):
    """One-dimensional PatchGAN used for three BP-pattern classes.

    The disclosed method first reduces the encoder feature map to a ``1 x N``
    sequence, predicts local pattern responses, then averages those responses
    for the final class prediction.
    """

    def __init__(
        self,
        *,
        classes: int = 3,
        hidden_features: int = 128,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        middle = max(16, hidden_features // 2)
        self.network = nn.Sequential(
            nn.Conv1d(1, hidden_features, kernel_size=7, padding=3),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(dropout),
            nn.Conv1d(hidden_features, hidden_features, kernel_size=5, padding=2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(dropout),
            nn.Conv1d(hidden_features, middle, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv1d(middle, classes, kernel_size=3, padding=1),
        )

    def forward(self, features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if features.ndim != 2:
            raise ValueError(
                "STP PatchGAN expects the 1 x N vector produced by global "
                "average pooling"
            )
        sequence = features.unsqueeze(1)
        patch_logits = self.network(sequence)
        return patch_logits.mean(dim=-1), patch_logits


class STPPatternAdapter(nn.Module):
    """Transferred encoder and three-class BP-pattern discriminator."""

    def __init__(
        self,
        encoder: STPEncoder,
        *,
        classes: int = 3,
        hidden_features: int = 128,
        dropout: float = 0.2,
        pooling: str = "mean",
        discriminator: str = "mlp",
        gradient_reversal_strength: float = 1.0,
    ) -> None:
        super().__init__()
        self.encoder = encoder
        self.pool = STPTokenPool(encoder.embedding_dim, pooling)
        self.discriminator_kind = str(discriminator).lower()
        self.gradient_reversal = GradientReversal(gradient_reversal_strength)
        if self.discriminator_kind == "patchgan":
            self.pattern_discriminator = STPPatchDiscriminator(
                classes=classes,
                hidden_features=hidden_features,
                dropout=dropout,
            )
        elif self.discriminator_kind == "mlp":
            self.pattern_discriminator = nn.Sequential(
                nn.LayerNorm(self.pool.output_features),
                nn.Linear(self.pool.output_features, hidden_features),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_features, classes),
            )
        else:
            raise ValueError("discriminator must be 'mlp' or 'patchgan'")

    def forward(
        self,
        signal: torch.Tensor,
        *,
        return_features: bool = False,
        return_patch_logits: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        tokens = self.encoder(signal)
        features = self.pool(tokens)
        adversarial_features = self.gradient_reversal(features)
        if self.discriminator_kind == "patchgan":
            logits, patch_logits = self.pattern_discriminator(
                adversarial_features
            )
        else:
            logits = self.pattern_discriminator(adversarial_features)
            patch_logits = logits[..., None]
        if return_patch_logits:
            return logits, patch_logits
        if return_features:
            return logits, features
        return logits


class STPBPRegressor(nn.Module):
    """Transferred encoder and SBP/DBP value regressor."""

    def __init__(
        self,
        encoder: STPEncoder,
        *,
        outputs: int = 2,
        hidden_features: int = 128,
        dropout: float = 0.2,
        pooling: str = "mean",
    ) -> None:
        super().__init__()
        self.encoder = encoder
        self.pool = STPTokenPool(encoder.embedding_dim, pooling)
        self.bp_value_regressor = nn.Sequential(
            nn.LayerNorm(self.pool.output_features),
            nn.Linear(self.pool.output_features, hidden_features),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_features, outputs),
        )

    def forward(
        self,
        signal: torch.Tensor,
        *,
        return_features: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        tokens = self.encoder(signal)
        features = self.pool(tokens)
        values = self.bp_value_regressor(features)
        if return_features:
            return values, features
        return values


def build_stp_encoder(config: Mapping[str, object]) -> STPEncoder:
    return STPEncoder(
        input_channels=int(config.get("input_channels", 1)),
        signal_length=int(config.get("signal_length", 512)),
        patch_size=int(config.get("patch_size", 8)),
        embedding_dim=int(config.get("embedding_dim", 128)),
        encoder_layers=int(config.get("encoder_layers", 4)),
        attention_heads=int(config.get("attention_heads", 8)),
        feedforward_dim=int(config.get("feedforward_dim", 256)),
        dropout=float(config.get("dropout", 0.1)),
    )


def build_stp_model(config: Mapping[str, object]) -> nn.Module:
    """Build one STP stage from a configuration mapping."""

    stage = str(config["stage"]).lower()
    encoder = build_stp_encoder(config)
    if stage == "pretrain":
        return STPSelfSupervisedModel(
            encoder,
            decoder_layers=int(config.get("decoder_layers", 2)),
            attention_heads=int(config.get("attention_heads", 8)),
            feedforward_dim=int(config.get("feedforward_dim", 256)),
            dropout=float(config.get("dropout", 0.1)),
        )
    if stage == "pattern":
        return STPPatternAdapter(
            encoder,
            classes=int(config.get("pattern_classes", 3)),
            hidden_features=int(config.get("head_features", 128)),
            dropout=float(config.get("head_dropout", 0.2)),
            pooling=str(config.get("pooling", "mean")),
            discriminator=str(config.get("discriminator", "mlp")),
            gradient_reversal_strength=float(
                config.get("gradient_reversal_strength", 1.0)
            ),
        )
    if stage == "bp":
        return STPBPRegressor(
            encoder,
            outputs=int(config.get("outputs", 2)),
            hidden_features=int(config.get("head_features", 128)),
            dropout=float(config.get("head_dropout", 0.2)),
            pooling=str(config.get("pooling", "mean")),
        )
    raise ValueError(f"Unsupported STP stage: {stage}")


def encoder_state_dict(model_or_checkpoint: object) -> dict[str, torch.Tensor]:
    """Extract an encoder state dict from any STP-stage checkpoint."""

    if isinstance(model_or_checkpoint, nn.Module):
        return model_or_checkpoint.encoder.state_dict()
    if not isinstance(model_or_checkpoint, Mapping):
        raise TypeError("Expected an STP model or checkpoint mapping")
    if "encoder" in model_or_checkpoint:
        state = model_or_checkpoint["encoder"]
        if not isinstance(state, Mapping):
            raise TypeError("checkpoint['encoder'] is not a state dict")
        return dict(state)
    state = model_or_checkpoint.get("model", model_or_checkpoint)
    if not isinstance(state, Mapping):
        raise TypeError("Checkpoint does not contain model weights")
    prefix = "encoder."
    extracted = {
        str(key)[len(prefix) :]: value
        for key, value in state.items()
        if str(key).startswith(prefix)
    }
    if not extracted:
        raise KeyError("No STP encoder weights found in checkpoint")
    return extracted


def transfer_encoder(
    model: nn.Module,
    checkpoint: Mapping[str, object],
    *,
    strict: bool = True,
) -> None:
    """Transfer the shared encoder between consecutive STP stages."""

    if not hasattr(model, "encoder"):
        raise TypeError("Target model does not expose an STP encoder")
    model.encoder.load_state_dict(encoder_state_dict(checkpoint), strict=strict)


def sinusoidal_position_encoding(length: int, dimension: int) -> torch.Tensor:
    """Reference sinusoidal encoding, useful for checkpoint conversions."""

    positions = torch.arange(length, dtype=torch.float32)[:, None]
    scales = torch.exp(
        torch.arange(0, dimension, 2, dtype=torch.float32)
        * (-math.log(10_000.0) / dimension)
    )
    encoding = torch.zeros(length, dimension)
    encoding[:, 0::2] = torch.sin(positions * scales)
    encoding[:, 1::2] = torch.cos(positions * scales[: encoding[:, 1::2].shape[1]])
    return encoding
