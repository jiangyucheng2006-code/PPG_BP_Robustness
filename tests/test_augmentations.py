import torch

from ppg_bp.data import ALL_STP_TRANSFORMS, PPGBatchAugmenter


def test_all_stp_transforms_preserve_batch_shape() -> None:
    time = torch.linspace(0, 10, 1250)
    ppg = torch.stack(
        (
            0.5 + 0.3 * torch.sin(2 * torch.pi * 1.2 * time),
            0.5 + 0.2 * torch.sin(2 * torch.pi * 1.5 * time),
        )
    )
    vpg = torch.gradient(ppg, dim=1)[0]
    inputs = torch.stack((ppg, vpg), dim=1)
    augmenter = PPGBatchAugmenter(
        ALL_STP_TRANSFORMS,
        probability=1.0,
    )

    for transform in ALL_STP_TRANSFORMS:
        output, labels, severity = augmenter(
            inputs,
            forced_transform=transform,
            forced_severity=0.7,
            seed=42,
        )
        assert output.shape == inputs.shape
        assert torch.isfinite(output).all()
        assert torch.all(labels == ALL_STP_TRANSFORMS.index(transform) + 1)
        assert torch.allclose(severity, torch.full_like(severity, 0.7))


def test_random_augmentation_is_seed_reproducible() -> None:
    inputs = torch.randn(4, 2, 1250)
    augmenter = PPGBatchAugmenter(
        ("gaussian_noise", "motion_artifact", "contact_compression"),
        probability=1.0,
        maximum_transforms=2,
    )
    first = augmenter(inputs, seed=7)
    second = augmenter(inputs, seed=7)
    for first_tensor, second_tensor in zip(first, second):
        assert torch.allclose(first_tensor, second_tensor)
