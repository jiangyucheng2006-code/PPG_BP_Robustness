import json

import numpy as np
import torch

from ppg_bp.data import (
    STP_RECONSTRUCTION_TRANSFORMS,
    STPTransformBank,
    STPWindowDataset,
    bp_pattern_labels,
    cycle_windows,
)
from ppg_bp.models import (
    STPBPRegressor,
    STPEncoder,
    STPPatternAdapter,
    STPSelfSupervisedModel,
    transfer_encoder,
)


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


def test_five_cycle_window_and_bp_patterns() -> None:
    sampling_rate = 100
    time = np.arange(20 * sampling_rate) / sampling_rate
    signal = np.sin(2 * np.pi * 1.2 * time)
    windows, bounds = cycle_windows(
        signal,
        sampling_rate,
        cycles=5,
        stride_cycles=2,
        output_samples=128,
    )
    assert len(windows) > 2
    assert windows.shape[1] == 128
    assert bounds.shape == (len(windows), 2)
    labels = np.asarray([[85, 55], [120, 75], [145, 95]], dtype=np.float32)
    assert bp_pattern_labels(labels).tolist() == [0, 1, 2]


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
