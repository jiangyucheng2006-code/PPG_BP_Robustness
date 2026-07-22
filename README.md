# Robust PPG-Based Blood Pressure Estimation

This repository contains a reproducible PPG-only blood pressure estimation
baseline and the planned robustness research built on top of it. The project
aims to reduce prediction errors caused by:

- contact-pressure variation;
- motion artifacts;
- acquisition and dataset differences.

Auxiliary signals or condition labels may be used during training when available, but the final model is intended to use PPG only.

## Current baseline

- Main dataset: PulseDB v2.0.
- Input: one 10-second PPG segment sampled at 125 Hz.
- Output: jointly predicted systolic and diastolic BP.
- Backbone: independently implemented 1-D XResNet-50/101.
- Evaluation: subject-wise splits with MAE, RMSE, bias, and error SD.

The public OOD-generalization benchmark by Moulaeifard et al. (2025) is the
primary methodological reference. Its released code and the official PulseDB
repository are linked in [`docs/REFERENCES.md`](docs/REFERENCES.md).

## Workflow

1. Download and audit the official PulseDB data.
2. Harmonize PPG preprocessing and BP labels.
3. Reproduce the PPG-only XResNet baseline.
4. Add pressure-, motion-, and domain-robustness components.
5. Perform subject-wise evaluation, external validation, and ablation studies.

## Repository structure

- `configs/`: experiment configurations.
- `data/`: instructions and local dataset locations; raw data are not uploaded.
- `notebooks/`: exploratory analysis.
- `src/`: preprocessing, models, training, and evaluation code.
- `tests/`: automated tests.

## Data policy

Large public datasets, generated outputs, model checkpoints, and any sensitive recordings must remain outside GitHub. Only download instructions and processing code should be committed.

## Start here

The step-by-step Chinese guide is in
[`docs/START_HERE_zh.md`](docs/START_HERE_zh.md).

For a beginner-friendly record of downloaded data, completed work, local
results, and one-click usage, see
[`docs/WHAT_I_DID_zh.md`](docs/WHAT_I_DID_zh.md) or double-click
`PPG_BP项目入口.bat` on Windows.
