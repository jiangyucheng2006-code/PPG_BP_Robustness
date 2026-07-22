# Robust PPG-Based Blood Pressure Estimation

This repository contains the code for a PPG-only blood pressure estimation project that aims to reduce prediction errors caused by:

- contact-pressure variation;
- motion artifacts;
- acquisition and dataset differences.

Auxiliary signals or condition labels may be used during training when available, but the final model is intended to use PPG only.

## Planned workflow

1. Select suitable public datasets.
2. Harmonize PPG preprocessing and BP labels.
3. Reproduce a reliable PPG-BP baseline.
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
