import json

import numpy as np
import torch

from ppg_bp.data import (
    STP_RECONSTRUCTION_TRANSFORMS,
    STPTransformBank,
    STPWindowDataset,
    bp_pattern_labels,
    cycle_windows,
    filter_abp_fir,
    fixed_length_windows,
    template_quality_mask,
    wavelet_filter_ppg,
)
from ppg_bp.models import (
    STPBPRegressor,
    STPEncoder,
    STPPatternAdapter,
    STPSelfSupervisedModel,
    STPTokenPool,
    transfer_encoder,
)
from ppg_bp.models.stp import GradientReversal


def small_encoder() -> STPEncoder:
    return STPEncoder(
        signal_length=128,
        patch_size=8,
        embedding_dim=32,
        encoder_layers=1,
        attention_heads=4,
        feedforward_dim=64,
        dropout=0,
    )


def test_stp_three_stage_shapes_and_transfer() -> None:
    pretrain = STPSelfSupervisedModel(
        small_encoder(),
        decoder_layers=1,
        attention_heads=4,
        feedforward_dim=64,
        dropout=0,
    )
    pattern = STPPatternAdapter(small_encoder(), hidden_features=16)
    bp = STPBPRegressor(small_encoder(), hidden_features=16)
    inputs = torch.randn(2, 1, 128)

    with torch.no_grad():
        reconstruction = pretrain(inputs)
        pattern_logits = pattern(inputs)
        bp_values = bp(inputs)
    assert reconstruction.shape == inputs.shape
    assert pattern_logits.shape == (2, 3)
    assert bp_values.shape == (2, 2)

    checkpoint = {"encoder": pretrain.encoder.state_dict()}
    transfer_encoder(pattern, checkpoint)
    for source, destination in zip(
        pretrain.encoder.parameters(),
        pattern.encoder.parameters(),
    ):
        assert torch.equal(source, destination)


def test_stp_token_pool_modes() -> None:
    tokens = torch.randn(3, 16, 8)
    expected_features = {
        "mean": 8,
        "attention": 8,
        "statistics": 16,
        "attentive_statistics": 16,
    }
    for mode, features in expected_features.items():
        pooled = STPTokenPool(8, mode)(tokens)
        assert pooled.shape == (3, features)
        assert torch.isfinite(pooled).all()


def test_all_paper_stp_transformations_reconstruct_original_target() -> None:
    inputs = torch.linspace(0, 1, 128)[None, None].repeat(2, 1, 1)
    transforms = STPTransformBank(sampling_rate=64)
    for name in STP_RECONSTRUCTION_TRANSFORMS:
        transformed, target, indices = transforms(
            inputs,
            forced_transform=name,
            forced_severity=0.7,
            seed=42,
        )
        assert transformed.shape == inputs.shape
        assert torch.isfinite(transformed).all()
        assert torch.equal(target, inputs)
        assert torch.all(indices == STP_RECONSTRUCTION_TRANSFORMS.index(name))


def test_disclosed_transform_definitions() -> None:
    inputs = torch.linspace(0, 1, 128)[None, None]
    transforms = STPTransformBank(
        sampling_rate=125,
        paper_disclosed=True,
    )
    negated, _, _ = transforms(
        inputs,
        forced_transform="amplitude_negation",
        forced_severity=0.7,
        seed=42,
    )
    assert torch.equal(negated, -inputs)

    clipped, _, _ = transforms(
        inputs,
        forced_transform="hard_clipping",
        forced_severity=0.7,
        seed=42,
    )
    assert clipped.min() > inputs.min()
    assert clipped.max() < inputs.max()

    first, _, _ = transforms(
        inputs,
        forced_transform="temporal_warping",
        forced_severity=0.7,
        seed=42,
    )
    second, _, _ = transforms(
        inputs,
        forced_transform="temporal_warping",
        forced_severity=0.7,
        seed=42,
    )
    assert first.shape == inputs.shape
    assert torch.equal(first, second)


def test_five_cycle_window_and_bp_patterns() -> None:
    sampling_rate = 100
    time = np.arange(20 * sampling_rate) / sampling_rate
    signal = np.sin(2 * np.pi * 1.2 * time)
    windows, bounds = cycle_windows(
        signal,
        sampling_rate,
        cycles=5,
        overlap_cycles=2,
        output_samples=128,
    )
    assert len(windows) > 2
    assert windows.shape[1] == 128
    assert bounds.shape == (len(windows), 2)
    assert bounds[1, 0] - bounds[0, 0] > 2 * sampling_rate
    labels = np.asarray([[85, 55], [120, 75], [145, 95]], dtype=np.float32)
    assert bp_pattern_labels(labels).tolist() == [0, 1, 2]
    threshold_labels = np.asarray(
        [[90, 60], [129, 79], [130, 79], [120, 80]],
        dtype=np.float32,
    )
    assert bp_pattern_labels(threshold_labels).tolist() == [1, 1, 2, 2]


def test_stp_fixed_length_windows() -> None:
    signal = np.linspace(-2, 3, 25, dtype=np.float32)
    windows, bounds = fixed_length_windows(signal, window_samples=10)
    assert windows.shape == (2, 10)
    assert bounds.tolist() == [[0, 10], [10, 20]]
    assert np.allclose(windows.min(axis=1), 0)
    assert np.allclose(windows.max(axis=1), 1)


def test_stp_disclosed_filters_and_template_screening() -> None:
    sampling_rate = 125
    time = np.arange(90 * sampling_rate) / sampling_rate
    clean = np.sin(2 * np.pi * 1.2 * time)
    noisy_ppg = clean + 0.4 * np.sin(2 * np.pi * 45 * time) + 0.2
    filtered_ppg = wavelet_filter_ppg(noisy_ppg)
    filtered_abp = filter_abp_fir(
        100 + 20 * clean + 2 * np.sin(2 * np.pi * 50 * time),
        sampling_rate,
    )
    assert filtered_ppg.shape == noisy_ppg.shape
    assert filtered_abp.shape == noisy_ppg.shape
    assert np.isfinite(filtered_ppg).all()
    windows = np.stack([clean[:128], clean[:128] + 0.01, -clean[:128]])
    valid, statistics = template_quality_mask(windows, standard_deviations=1.0)
    assert valid.tolist() == [True, True, False]
    assert statistics["correlation_lower"] < 1


def test_stp_patchgan_pattern_adapter() -> None:
    model = STPPatternAdapter(
        small_encoder(),
        hidden_features=16,
        discriminator="patchgan",
        gradient_reversal_strength=1.0,
    )
    inputs = torch.randn(2, 1, 128)
    discriminator_inputs = []
    hook = model.pattern_discriminator.register_forward_pre_hook(
        lambda _module, arguments: discriminator_inputs.append(arguments[0].shape)
    )
    logits, patch_logits = model(inputs, return_patch_logits=True)
    hook.remove()
    assert logits.shape == (2, 3)
    assert patch_logits.shape[:2] == (2, 3)
    assert patch_logits.shape[-1] == small_encoder().embedding_dim
    assert discriminator_inputs == [torch.Size((2, small_encoder().embedding_dim))]


def test_stp_gradient_reversal_matches_minmax_direction() -> None:
    inputs = torch.tensor([1.0, -2.0], requires_grad=True)
    GradientReversal(1.0)(inputs).sum().backward()
    assert torch.equal(inputs.grad, torch.tensor([-1.0, -1.0]))


def test_manifest_dataset_modes(tmp_path) -> None:
    subject_dir = tmp_path / "subjects" / "mimiciii"
    subject_dir.mkdir(parents=True)
    np.save(subject_dir / "p1_signals.npy", np.random.rand(3, 128).astype("float32"))
    np.save(
        subject_dir / "p1_labels.npy",
        np.asarray([[120, 80], [130, 85], [140, 90]], dtype="float32"),
    )
    np.save(subject_dir / "p1_patterns.npy", np.asarray([1, 1, 2]))
    manifest = {
        "window_samples": 128,
        "subjects": [
            {
                "subject_id": "p1",
                "source": "MIMICIII",
                "split": "train",
                "windows": 3,
                "signals": "subjects/mimiciii/p1_signals.npy",
                "labels": "subjects/mimiciii/p1_labels.npy",
                "patterns": "subjects/mimiciii/p1_patterns.npy",
            }
        ],
    }
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))

    pretrain = STPWindowDataset(tmp_path, "train", "pretrain")
    pattern = STPWindowDataset(tmp_path, "train", "pattern")
    bp = STPWindowDataset(tmp_path, "train", "bp")
    assert pretrain[0].shape == (1, 128)
    assert pattern[0][1].dtype == torch.long
    assert bp[0][1].shape == (2,)
    assert pattern.target_values("patterns").tolist() == [1, 1, 2]
    assert bp.target_values("labels").shape == (3, 2)
