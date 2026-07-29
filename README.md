# PPG-BP Robustness

PPG-only blood pressure estimation under contact-pressure variation, motion
artifacts, and acquisition-domain shift. The current implementation establishes
a calibration-free PulseDB baseline before introducing robustness components.

## Baseline

| Component | Setting |
|---|---|
| Input | 10 s PPG window at 125 Hz (1,250 samples) |
| Targets | Systolic and diastolic blood pressure |
| Backbone | QUMPHY-compatible 1-D XResNet-50 (886,690 parameters) |
| Objective | MSE on SBP/DBP in mmHg |
| Signal normalization | PulseDB filtered PPG (`PPG_F`), no additional normalization |
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

The official-compatible XResNet-50 is the project baseline. Its topology and
parameter count match the upstream QUMPHY implementation. The checkpoint from
epoch 4 was selected using validation SBP MAE and evaluated once on the fixed
PulseDB VitalDB calibration-free test set.

| Model | SBP MAE | DBP MAE | Mean MAE |
|---|---:|---:|---:|
| Train-set mean predictor | 14.94 | 9.43 | 12.19 |
| Independent wide XResNet-101 | 13.46 | 8.55 | 11.00 |
| **Official-compatible XResNet-50** | **12.49** | **8.06** | **10.28** |

Values are in mmHg. Training curves, run settings, and the complete metric
summary are available in
[`results/qumphy_xresnet50_vital_calibfree`](results/qumphy_xresnet50_vital_calibfree).

## Experiment status

The baseline has since been extended with multi-scale morphology, PPG-derived
VPG/APG channels, train-only label-density weighting, independent SBP/DBP
regression heads, and controlled attention variants. Each change, result, and
retention decision is recorded in
[`docs/EXPERIMENTS.md`](docs/EXPERIMENTS.md).

The current reproducible CalFree candidate is the multi-scale PPG+VPG model
with moderate label-density weighting and separate SBP/DBP heads. The
non-competitive concatenation-attention variant achieved a 10.018 mmHg
single-seed mean MAE and is awaiting multi-seed confirmation.

A physiology-guided multi-task ablation has also been completed. Heart rate,
age group, and BP-pattern supervision were learned from PPG during training,
while inference remained PPG-only. The auxiliary tasks were learnable but did
not improve BP accuracy under direct hard sharing. Configurations and results
are summarized in
[`results/multitask_m0_m4`](results/multitask_m0_m4).

The C0-C6 robustness ablation evaluates the same PPG-only model under 24
controlled corruption conditions. Full mixed-artifact augmentation (C2)
reduced average corrupted MAE from 21.931 to 10.618 mmHg and reduced the
clean-to-corrupted prediction shift from 17.384 to 2.374 mmHg. The synthetic
corruptions, aggregate results, and limitations are documented in
[`results/robustness_c0_c6`](results/robustness_c0_c6).

A separate S1-S3 track now reproduces the published STP architecture as
closely as the public information allows: transformed-signal reconstruction
with a Transformer encoder-decoder, three-class BP-pattern adaptation, and
SBP/DBP regression with sequential encoder transfer. It uses WESAD and
PPG-DaLiA for unpaired pretraining and MIMIC-III PPG/ABP for the two downstream
stages. The private Mindray cohort is omitted. Confirmed details, provisional
hyperparameters, data preparation, and run commands are documented in
[`docs/STP_REPRODUCTION.md`](docs/STP_REPRODUCTION.md).

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

For the M-series experiments, create the compact training-only age and
heart-rate labels after the waveform conversion:

```powershell
python scripts/prepare_pulsedb_auxiliary_labels.py `
  --dataset-root <DATA_ROOT>/processed/pulsedb_full
```

Age comes from the official metadata. Heart rate is estimated from synchronized
ECG and checked against PPG periodicity. Neither age nor ECG is a model input at
inference.

## Training

Set the data root and run the official-compatible XResNet-50 configuration:

```powershell
$env:PPG_BP_DATA_ROOT = "D:/Datasets/PPG_BP_Robustness"
python scripts/train_baseline.py `
  --config configs/pulsedb_qumphy_xresnet50.yaml `
  --output outputs/pulsedb_qumphy_xresnet50_full `
  --resume
```

Training writes the following local artifacts:

| File | Description |
|---|---|
| `best.pt` | Checkpoint with the lowest validation MAE |
| `last.pt` | Latest resumable training state |
| `history.json` | Per-epoch training and validation metrics |
| `metrics.json` | Final evaluation on the fixed test set |

Run one M-series configuration with:

```powershell
python scripts/train_multitask.py `
  --config configs/pulsedb_m4_hr_age_bpclass.yaml `
  --output outputs/pulsedb_m4_hr_age_bpclass `
  --resume
```

Run the complete C-series robustness suite with:

```powershell
python scripts/run_c_suite.py `
  --status outputs/c_robustness_suite/status.json
```

Run the faithful public-data STP reproduction sequentially with:

```powershell
python scripts/train_stp.py `
  --config configs/stp_s1_public_pretrain.yaml `
  --output outputs/stp_s1_public_pretrain `
  --resume
```

Continue with S2 and S3 using the commands in
[`docs/STP_REPRODUCTION.md`](docs/STP_REPRODUCTION.md).

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
