import torch

from ppg_bp.models import xresnet1d50


def test_xresnet_output_shape() -> None:
    model = xresnet1d50()
    model.eval()
    with torch.no_grad():
        output = model(torch.randn(2, 1, 1250))
    assert output.shape == (2, 2)

