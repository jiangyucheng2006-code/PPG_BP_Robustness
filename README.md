# PPG-BP Robustness

PPG-only blood pressure estimation under contact-pressure variation, motion
artifacts, and acquisition-domain shift. The current implementation establishes
a calibration-free PulseDB baseline before introducing robustness components.

## Baseline

| Component | Setting |
|---|---|
| Input | 10 s PPG window at 125 Hz (1,250 samples) |
| Targets | Systolic and diastolic blood pressure |
| Backbone | 1-D XResNet-50 or XResNet-101 |
| Objective | MSE on standardized SBP/DBP targets |
| Signal normalization | Per-window z-score |
| Split protocol | Subject-disjoint, calibration-free |
| Metrics | MAE, RMSE, bias, and error standard deviation |

The main configuration uses the VitalDB-derived PulseDB calibration-free
subsets. The official test set is kept fixed; validation subjects are held out
from the official training subset.

| Split | Windows |
|---|---:|
| Train | 412,920 |
| Validation | 52,560 |
| Test | 57,600 |

## Baseline result

The first full XResNet-101 run completed on the VitalDB-derived,
calibration-free split. The checkpoint from epoch 7 was selected using
validation mean MAE and then evaluated once on the fixed test set.

| Model | SBP MAE | DBP MAE | Mean MAE |
|---|---:|---:|---:|
| Train-set mean predictor | 14.94 | 9.43 | 12.19 |
| XResNet-101 | **13.46** | **8.55** | **11.00** |

Values are in mmHg. Training curves, run settings, and the complete metric
summary are available in
[`results/xresnet101_vital_calibfree`](results/xresnet101_vital_calibfree).

## Installation

Python 3.10 or later is required.

```bash
python -m pip install -e ".[dev]"
```

## Data preparation

Download the PulseDB subset files from the source listed in
[`data/README.md`](data/README.md), then convert the VitalDB training and
calibration-free test subsets:

```bash
python scripts/prepare_pulsedb_subsets.py \
  --train <DATA_ROOT>/PulseDB/VitalDB_Subsets/VitalDB_Train_Subset.mat \
  --test <DATA_ROOT>/PulseDB/VitalDB_Subsets/VitalDB_CalFree_Test_Subset.mat \
  --output <DATA_ROOT>/processed/pulsedb_full
```

The converter writes memory-mapped `ppg.npy`, `labels.npy`, and `split.npy`
arrays. Raw waveforms and generated arrays are not tracked by Git.

## Training

Set the data root and run the XResNet-101 configuration:

```powershell
$env:PPG_BP_DATA_ROOT = "D:/Datasets/PPG_BP_Robustness"
python scripts/train_baseline.py `
  --config configs/pulsedb_xresnet101.yaml `
  --output outputs/pulsedb_xresnet101_full `
  --resume
```

Training writes the following local artifacts:

| File | Description |
|---|---|
| `best.pt` | Checkpoint with the lowest validation MAE |
| `last.pt` | Latest resumable training state |
| `history.json` | Per-epoch training and validation metrics |
| `metrics.json` | Final evaluation on the fixed test set |

The smoke configuration is limited to pipeline verification:

```powershell
python scripts/train_baseline.py `
  --config configs/pulsedb_xresnet50_smoke.yaml `
  --output outputs/smoke_xresnet50
```

## Repository layout

```text
configs/        Experiment configurations
data/           Dataset source and layout
docs/           References
scripts/        Data preparation, audit, and training entry points
src/ppg_bp/     Dataset, model, and metric implementations
tests/          Unit tests
```

## Tests

```bash
python -m pytest
```

## Planned extensions

- condition-invariant representations for contact-pressure variation;
- motion-corruption augmentation and artifact-aware feature learning;
- domain-robust training and external-dataset evaluation;
- controlled ablations against the PPG-only XResNet baseline.

Core references and upstream implementations are listed in
[`docs/REFERENCES.md`](docs/REFERENCES.md).
