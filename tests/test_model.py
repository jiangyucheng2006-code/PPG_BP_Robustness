import torch

from ppg_bp.models import (
    AdaptiveScaleAttentionStem,
    QumphyTaskAttentionXResNet1D,
    qumphy_attention_multiscale_xresnet1d50,
    qumphy_multiscale_xresnet1d50,
    qumphy_task_attention_multiscale_xresnet1d50,
    qumphy_xresnet1d50,
    xresnet1d50,
)


def test_xresnet_output_shape() -> None:
    model = xresnet1d50()
    model.eval()
    with torch.no_grad():
        output = model(torch.randn(2, 1, 1250))
    assert output.shape == (2, 2)


def test_qumphy_xresnet_matches_upstream_parameter_count() -> None:
    model = qumphy_xresnet1d50()
    assert sum(parameter.numel() for parameter in model.parameters()) == 886_690
    model.eval()
    with torch.no_grad():
        output = model(torch.randn(2, 1, 1250))
    assert output.shape == (2, 2)


def test_multiscale_xresnet_is_lightweight() -> None:
    baseline = qumphy_xresnet1d50()
    multiscale = qumphy_multiscale_xresnet1d50()
    added_parameters = sum(p.numel() for p in multiscale.parameters()) - sum(
        p.numel() for p in baseline.parameters()
    )
    assert 0 < added_parameters < 5_000
    multiscale.eval()
    with torch.no_grad():
        output = multiscale(torch.randn(2, 1, 1250))
    assert output.shape == (2, 2)


def test_adaptive_scale_attention_weights_and_output() -> None:
    torch.manual_seed(42)
    model = qumphy_attention_multiscale_xresnet1d50()
    model.eval()
    inputs = torch.randn(2, 1, 1250)
    with torch.no_grad():
        outputs = model(inputs)
        stem = model.stem[0]
        assert isinstance(stem, AdaptiveScaleAttentionStem)
        _, weights = stem.extract_scale_features(inputs)

    multiscale_parameters = sum(
        parameter.numel() for parameter in qumphy_multiscale_xresnet1d50().parameters()
    )
    attention_parameters = sum(parameter.numel() for parameter in model.parameters())
    assert 0 < attention_parameters - multiscale_parameters < 5_000
    assert outputs.shape == (2, 2)
    assert weights.shape == (2, 3)
    assert torch.allclose(weights.sum(dim=1), torch.ones(2), atol=1e-6)
    assert torch.allclose(weights, torch.full_like(weights, 1 / 3), atol=1e-6)


def test_task_specific_attention_heads_start_as_identity() -> None:
    torch.manual_seed(42)
    model = qumphy_task_attention_multiscale_xresnet1d50()
    model.eval()
    inputs = torch.randn(2, 1, 1250)
    with torch.no_grad():
        feature_map = model.forward_feature_map(inputs)
        outputs = model(inputs)
        attention_maps = [
            task_head.attention_maps(feature_map) for task_head in model.task_heads
        ]

    assert isinstance(model, QumphyTaskAttentionXResNet1D)
    assert outputs.shape == (2, 2)
    for channel_scale, temporal_scale in attention_maps:
        assert channel_scale.shape == (2, 256, 1)
        assert temporal_scale.shape == (2, 1, feature_map.shape[-1])
        assert torch.allclose(channel_scale, torch.ones_like(channel_scale), atol=1e-6)
        assert torch.allclose(temporal_scale, torch.ones_like(temporal_scale), atol=1e-6)

    multiscale_parameters = sum(
        parameter.numel() for parameter in qumphy_multiscale_xresnet1d50().parameters()
    )
    task_attention_parameters = sum(parameter.numel() for parameter in model.parameters())
    assert 0 < task_attention_parameters - multiscale_parameters < 100_000
