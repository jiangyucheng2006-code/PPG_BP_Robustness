import torch

from ppg_bp.models import qumphy_multiscale_xresnet1d50, qumphy_xresnet1d50, xresnet1d50


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
