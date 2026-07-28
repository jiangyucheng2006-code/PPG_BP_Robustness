from .qumphy_xresnet1d import (
    AdaptiveScaleAttentionStem,
    ConcatenatedPPGVPGAttentionStem,
    GatedPPGVPGStem,
    MultiScaleInputStem,
    QumphyGatedDerivativeXResNet1D,
    QumphyConcatAttentionDerivativeXResNet1D,
    QumphyIndependentHeadsXResNet1D,
    QumphyTaskAttentionXResNet1D,
    QumphyXResNet1D,
    TaskSpecificAttentionHead1D,
    qumphy_attention_multiscale_xresnet1d50,
    qumphy_concat_attention_derivative_xresnet1d50,
    qumphy_gated_derivative_task_heads_xresnet1d50,
    qumphy_gated_derivative_xresnet1d50,
    qumphy_independent_heads_multiscale_xresnet1d50,
    qumphy_multiscale_xresnet1d50,
    qumphy_task_attention_multiscale_xresnet1d50,
    qumphy_xresnet1d50,
    qumphy_xresnet1d101,
)
from .xresnet1d import XResNet1D, xresnet1d50, xresnet1d101


def build_model(model_config: dict):
    """Build a project model from the YAML/checkpoint model section."""

    factories = {
        "xresnet1d": {50: xresnet1d50, 101: xresnet1d101},
        "qumphy_xresnet1d": {50: qumphy_xresnet1d50, 101: qumphy_xresnet1d101},
        "qumphy_multiscale_xresnet1d": {50: qumphy_multiscale_xresnet1d50},
        "qumphy_attention_multiscale_xresnet1d": {
            50: qumphy_attention_multiscale_xresnet1d50
        },
        "qumphy_task_attention_multiscale_xresnet1d": {
            50: qumphy_task_attention_multiscale_xresnet1d50
        },
        "qumphy_gated_derivative_xresnet1d": {
            50: qumphy_gated_derivative_xresnet1d50
        },
        "qumphy_gated_derivative_task_heads_xresnet1d": {
            50: qumphy_gated_derivative_task_heads_xresnet1d50
        },
        "qumphy_independent_heads_multiscale_xresnet1d": {
            50: qumphy_independent_heads_multiscale_xresnet1d50
        },
        "qumphy_concat_attention_derivative_xresnet1d": {
            50: qumphy_concat_attention_derivative_xresnet1d50
        },
    }
    model_name = str(model_config["name"])
    depth = int(model_config["depth"])
    if model_name not in factories:
        raise ValueError(f"Unsupported model name: {model_name}")
    factory = factories[model_name].get(depth)
    if factory is None:
        raise ValueError(f"Unsupported depth {depth} for {model_name}")
    return factory(
        input_channels=int(model_config.get("input_channels", 1)),
        outputs=int(model_config.get("outputs", 2)),
        dropout=float(
            model_config.get(
                "dropout",
                0.5 if model_name.startswith("qumphy_") else 0.2,
            )
        ),
    )


__all__ = [
    "AdaptiveScaleAttentionStem",
    "ConcatenatedPPGVPGAttentionStem",
    "GatedPPGVPGStem",
    "QumphyGatedDerivativeXResNet1D",
    "QumphyConcatAttentionDerivativeXResNet1D",
    "QumphyIndependentHeadsXResNet1D",
    "QumphyTaskAttentionXResNet1D",
    "QumphyXResNet1D",
    "MultiScaleInputStem",
    "TaskSpecificAttentionHead1D",
    "qumphy_attention_multiscale_xresnet1d50",
    "qumphy_concat_attention_derivative_xresnet1d50",
    "qumphy_gated_derivative_task_heads_xresnet1d50",
    "qumphy_gated_derivative_xresnet1d50",
    "qumphy_independent_heads_multiscale_xresnet1d50",
    "qumphy_multiscale_xresnet1d50",
    "qumphy_task_attention_multiscale_xresnet1d50",
    "qumphy_xresnet1d50",
    "qumphy_xresnet1d101",
    "XResNet1D",
    "xresnet1d50",
    "xresnet1d101",
    "build_model",
]
