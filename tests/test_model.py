import torch

from ppg_bp.models import qumphy_xresnet1d50, xresnet1d50


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
