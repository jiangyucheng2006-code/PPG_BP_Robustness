# M0-M4 physiology-guided multi-task ablation

These experiments test whether physiological auxiliary supervision improves
PPG-only blood-pressure estimation. The shared encoder receives only PPG and
PPG-derived VPG. ECG-derived heart rate and age metadata are training labels,
not inference inputs.

## Results

All values are from seed 42 on the fixed PulseDB VitalDB calibration-free test
set. MAE values are in mmHg except heart rate, which is in bpm.

| ID | Auxiliary tasks | Best validation MAE | SBP MAE | DBP MAE | Mean MAE |
|---|---|---:|---:|---:|---:|
| **M0** | None | 10.579 | **12.150** | **7.886** | **10.018** |
| M1 | Heart rate | 10.605 | 12.689 | 8.266 | 10.477 |
| M2 | Age group | 10.603 | 13.026 | 8.016 | 10.521 |
| M3 | Heart rate + age group | 10.580 | 12.797 | 8.213 | 10.505 |
| M4 | Heart rate + age group + BP pattern | 10.597 | 12.299 | 8.434 | 10.366 |

| ID | Heart-rate MAE | Age-group accuracy | BP-pattern accuracy |
|---|---:|---:|---:|
| M1 | 1.944 | - | - |
| M2 | - | 55.3% | - |
| M3 | 2.146 | 58.3% | - |
| M4 | **1.765** | 58.2% | 59.2% |

## Interpretation

The network learned the auxiliary targets, but direct hard sharing did not
improve BP prediction. M4 recovered part of the loss seen in M1-M3, yet its
mean BP MAE remained 0.348 mmHg above M0. M0/B6-4 therefore remains the
selected seed-42 model. The next multi-task controls should address gradient
conflict and task scheduling instead of adding more auxiliary heads.

Exact metrics are recorded in [`summary.json`](summary.json). Model
checkpoints, raw data, and full local logs are intentionally not tracked.
