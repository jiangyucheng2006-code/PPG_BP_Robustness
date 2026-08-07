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
from .multitask import PhysiologyGuidedMultiTaskModel
from .robustness import ArtifactAwareBPModel
from .stp import (
    GradientReversal,
    STPBPRegressor,
    STPEncoder,
    STPPatternAdapter,
    STPPatchDiscriminator,
    STPSelfSupervisedModel,
    STPTokenPool,
    build_stp_encoder,
    build_stp_model,
    transfer_encoder,
)


def build_model(model_config: dict):
    """Build a project model from the YAML/checkpoint model section."""

    model_name = str(model_config["name"])
    if model_name == "physiology_guided_multitask":
        backbone = build_model(model_config["backbone"])
        return PhysiologyGuidedMultiTaskModel(
            backbone,
            tuple(model_config.get("tasks", ())),
            feature_dim=int(model_config.get("feature_dim", 512)),
            age_classes=int(model_config.get("age_classes", 4)),
            bp_classes=int(model_config.get("bp_classes", 3)),
            hidden_features=int(model_config.get("hidden_features", 128)),
            auxiliary_dropout=float(model_config.get("auxiliary_dropout", 0.2)),
        )
    if model_name == "artifact_aware_bp":
        backbone = build_model(model_config["backbone"])
        return ArtifactAwareBPModel(
            backbone,
            feature_dim=int(model_config.get("feature_dim", 512)),
            artifact_classes=int(model_config.get("artifact_classes", 0)),
            hidden_features=int(model_config.get("hidden_features", 128)),
            dropout=float(model_config.get("artifact_dropout", 0.2)),
        )
    if model_name == "stp":
        return build_stp_model(model_config)

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
    "PhysiologyGuidedMultiTaskModel",
    "ArtifactAwareBPModel",
    "STPBPRegressor",
    "GradientReversal",
    "STPEncoder",
    "STPPatternAdapter",
    "STPPatchDiscriminator",
    "STPSelfSupervisedModel",
    "STPTokenPool",
    "build_stp_encoder",
    "build_stp_model",
    "transfer_encoder",
    "xresnet1d50",
    "xresnet1d101",
    "build_model",
]
