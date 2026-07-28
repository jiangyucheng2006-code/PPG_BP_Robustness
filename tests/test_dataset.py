import json

import h5py
import numpy as np

from ppg_bp.data import PulseDBMemmapDataset, read_pulsedb_subject


def test_memmap_dataset(tmp_path) -> None:
    time = np.linspace(0, 10, 1250, dtype=np.float32)
    signals = np.stack(
        [np.sin((index + 1) * 2 * np.pi * time) for index in range(4)]
    ).astype(np.float32)
    np.save(tmp_path / "ppg.npy", signals)
    np.save(tmp_path / "labels.npy", np.array([[120, 80], [121, 81], [122, 82], [123, 83]], dtype=np.float32))
    np.save(tmp_path / "split.npy", np.array([0, 0, 1, 2], dtype=np.uint8))
    (tmp_path / "dataset_meta.json").write_text(
        json.dumps({"n_samples": 4, "window_samples": 1250}), encoding="utf-8"
    )
    dataset = PulseDBMemmapDataset(tmp_path, "train")
    signal, label = dataset[0]
    assert len(dataset) == 2
    assert signal.shape == (1, 1250)
    assert label.tolist() == [120.0, 80.0]
    assert abs(float(signal.mean())) < 1e-5

    filtered = PulseDBMemmapDataset(
        tmp_path,
        "train",
        label_filter={"sbp_min": 121, "sbp_max": 200, "dbp_min": 40, "dbp_max": 130},
    )
    assert len(filtered) == 1
    assert filtered[0][1].tolist() == [121.0, 81.0]

    np.save(tmp_path / "alternate_split.npy", np.array([1, 0, 0, 2], dtype=np.uint8))
    alternate = PulseDBMemmapDataset(
        tmp_path,
        "train",
        normalization="none",
        split_filename="alternate_split.npy",
    )
    assert len(alternate) == 2
    assert alternate[0][1].tolist() == [121.0, 81.0]

    derivative = PulseDBMemmapDataset(
        tmp_path,
        "train",
        input_representation="ppg_vpg_apg",
    )
    derivative_signal, _ = derivative[0]
    assert derivative_signal.shape == (3, 1250)
    assert abs(float(derivative_signal[1].mean())) < 1e-5
    assert abs(float(derivative_signal[2].mean())) < 1e-5
    assert abs(float(derivative_signal[1].std()) - 1.0) < 1e-3
    assert abs(float(derivative_signal[2].std()) - 1.0) < 1e-3


def test_direct_single_segment_matlab_layout(tmp_path) -> None:
    path = tmp_path / "p000001.mat"
    with h5py.File(path, "w") as handle:
        group = handle.create_group("Subj_Wins")
        group.create_dataset("PPG_F", data=np.linspace(0, 1, 1250)[None, :])
        group.create_dataset("IncludeFlag", data=np.array([[1]], dtype=np.uint8))
        group.create_dataset("SegSBP", data=np.array([[120.0]]))
        group.create_dataset("SegDBP", data=np.array([[80.0]]))
        group.create_dataset("PPG_ABP_Corr", data=np.array([[0.9]]))
        group.create_dataset("SubjectID", data=np.array([[ord(c)] for c in "p000001"], dtype=np.uint16))

    subject = read_pulsedb_subject(path)
    assert subject.subject_id == "p000001"
    assert subject.ppg.shape == (1, 1250)
    assert subject.labels.tolist() == [[120.0, 80.0]]
