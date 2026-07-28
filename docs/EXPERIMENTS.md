# Baseline experiment log

This file records each controlled change made after selecting the
QUMPHY-compatible XResNet-50 backbone. Unless stated otherwise, all experiments
use the same PulseDB VitalDB training data, subject-disjoint validation split,
and fixed calibration-free test set.

The input remains single-sensor PPG. VPG and APG are calculated from the same
PPG waveform and do not require additional sensors at inference.

## Evaluation protocol

- **CalFree test:** 57,600 windows from subjects excluded from training.
- **AAMI subset:** 666 measurements from 116 unseen subjects, selected by
  PulseDB to satisfy the AAMI cohort and blood-pressure distribution rules.
- **AAMI numerical check:** absolute mean error no greater than 5 mmHg and
  error standard deviation no greater than 8 mmHg for both SBP and DBP.
- **BHS Grade A:** at least 60%, 85%, and 95% of errors within 5, 10, and
  15 mmHg.

AAMI results through B5 were used for exploratory diagnosis. From B6 onward,
architecture selection is based on the subject-disjoint validation set; the
AAMI subset is kept frozen. A final publication claim will require a locked
model and an untouched external evaluation.

## Experiment sequence

The table reports mean SBP/DBP MAE in mmHg for seed 42 unless a multi-seed
result is explicitly shown.

| ID | Controlled change | CalFree mean MAE | AAMI mean MAE | Decision |
|---|---|---:|---:|---|
| B0 | QUMPHY-compatible XResNet-50, raw PPG | 10.276 | — | Starting backbone |
| B1 | Huber loss and cosine learning-rate schedule | 10.451 | — | Rejected; worse than B0 |
| B2 | Multi-scale input stem with kernels 3/7/15 | 10.178 | 16.665 | Retained as the main backbone |
| B3 | Adaptive attention over the three input scales | 10.209 | — | Rejected; no reliable validation gain |
| B3-v2 | Task-specific channel and temporal attention | 10.122 | 16.935 | Rejected; AAMI generalization worsened |
| B4-1 | PPG and first derivative (VPG) | 10.198 | 15.307 | Retained for stricter-distribution experiments |
| B4-2 | PPG, VPG, and second derivative (APG) | 10.146 | 16.112 | Not retained; APG improved CalFree slightly but hurt AAMI |
| B5-1 | Moderate train-only label-density weighting | 10.755 | **13.979** | Retained; best exploratory AAMI result |
| B5-2 | Strong label-density weighting | 11.349 | 14.062 | Rejected; excessive weighting hurt overall accuracy |
| B6-1 | Separate PPG/VPG stems with competitive softmax gating | 10.706 | Frozen | Gate alone gave little improvement |
| B6-2 | B6-1 plus separate SBP and DBP regression heads | 10.475 | Frozen | Stable, but not the lowest average error |
| B6-3 | Separate SBP/DBP heads without competitive gating | 10.460 | Frozen | Current reproducible CalFree candidate |
| B6-4 | Concatenated PPG/VPG features, non-competitive channel attention, separate heads | **10.018** | Frozen | Best seed-42 CalFree result; multi-seed confirmation required |

## What changed in each stage

### B1: optimization recipe

Only the loss and learning-rate schedule changed. Huber loss and cosine decay
did not improve the baseline, so subsequent controlled experiments returned to
MSE and a constant learning rate.

### B2: multi-scale morphology

Three convolution branches observe short-, medium-, and longer-range waveform
patterns before the XResNet encoder. This produced a small, repeatable
improvement and became the common backbone for later experiments.

### B3: attention variants

B3 learned one weight per receptive-field scale. B3-v2 added separate
channel-temporal attention for SBP and DBP. The seed-42 CalFree result improved
slightly, but validation and AAMI results did not support a reliable
generalization benefit.

### B4: derivative channels

B4-1 added VPG, the first numerical derivative of PPG. B4-2 also added APG,
the second derivative. APG marginally improved the in-distribution CalFree
score but amplified noise and performed worse on the AAMI subset. VPG was
therefore retained. A smoothed, branch-specific APG design remains a possible
future control.

### B5: label-distribution weighting

The official training subset contains few very-high-BP samples: approximately
1.72% have SBP at least 160 mmHg and 0.38% have DBP at least 100 mmHg.
B5 computes inverse-density weights from training labels only. Moderate
weighting reduced systematic underestimation on the AAMI subset; strong
weighting degraded the CalFree result.

### B6: output heads and derivative attention

B6 separates the SBP and DBP regressors because the two targets do not depend
on identical waveform features.

- B6-1 uses competitive softmax gating between separate PPG and VPG stems.
- B6-2 adds independent SBP/DBP heads.
- B6-3 removes competitive gating while keeping independent heads.
- B6-4 preserves both branch feature sets by concatenation, then applies
  non-competitive channel attention before fusion.

B6-4 stopped after nine epochs and selected epoch 3. Its seed-42 CalFree
metrics were 12.150 mmHg SBP MAE, 7.886 mmHg DBP MAE, and 10.018 mmHg mean
MAE. The corresponding validation mean MAE was 10.579 mmHg, which did not
improve over B6-3. The test result is promising, but B6-4 is not promoted until
it is repeated with the remaining seeds.

## Multi-seed comparison

Seeds 42, 7, and 2026 were run for the B5-1, B6-2, and B6-3 comparison.
Values are mean ± sample standard deviation in mmHg.

| Model | Validation mean MAE | CalFree mean MAE | SBP MAE | DBP MAE |
|---|---:|---:|---:|---:|
| B5-1 | 10.575 ± 0.119 | 10.528 ± 0.199 | 12.866 ± 0.308 | 8.190 ± 0.102 |
| B6-2, competitive attention + heads | 10.600 ± 0.103 | 10.456 ± 0.054 | 12.811 ± 0.092 | 8.101 ± 0.051 |
| **B6-3, heads only** | **10.491 ± 0.067** | **10.340 ± 0.138** | **12.610 ± 0.175** | **8.069 ± 0.153** |

B6-3 outperformed B6-2 on the CalFree test for all three seeds. The competitive
attention model had lower variance, but a higher average error. B6-4 tests
whether attention can be retained without forcing PPG and VPG to suppress each
other.

## Current interpretation

- Multi-scale feature extraction is a useful backbone change.
- VPG improves robustness to the stricter BP distribution more than raw APG.
- Moderate label-density weighting improves high-BP behavior, but introduces a
  trade-off with the ordinary CalFree score.
- Separate SBP and DBP heads are the most consistent B6 improvement.
- Competitive early PPG/VPG attention is not supported by the current
  multi-seed evidence.
- Non-competitive concatenation attention produced the best single-seed
  CalFree result, but still requires multi-seed confirmation.
- No current model meets the AAMI numerical limits or BHS Grade A. The main
  remaining problem is error spread and large tail errors, not only mean bias.

## Configuration map

| Experiment | Configuration |
|---|---|
| B4-1 | `configs/pulsedb_b4_vpg_control.yaml` |
| B4-2 | `configs/pulsedb_b4_vpg_apg_control.yaml` |
| B5-1 | `configs/pulsedb_b5_density_moderate.yaml` |
| B5-2 | `configs/pulsedb_b5_density_strong.yaml` |
| B6-1 | `configs/pulsedb_b6_gated_fusion.yaml` |
| B6-2 | `configs/pulsedb_b6_gated_task_heads.yaml` |
| B6-3 | `configs/pulsedb_b6_independent_heads_control.yaml` |
| B6-4 | `configs/pulsedb_b6_concat_attention_task_heads.yaml` |

## M series: physiology-guided multi-task learning

The M series tests whether physiological supervision can improve the shared
PPG representation. The model input remains PPG and its numerical first
derivative (VPG). Synchronized ECG and age metadata are used only to construct
training labels; they are not inference inputs.

- M0: BP-only control using the B6-4 backbone.
- M1: M0 plus heart-rate regression.
- M2: M0 plus age-group classification.
- M3: M0 plus heart-rate and age-group tasks.
- M4: M3 plus low/normal/high BP-pattern classification.

All experiments use seed 42 and the same subject-disjoint validation and fixed
CalFree test splits.

| ID | Auxiliary tasks | Best validation MAE | CalFree SBP MAE | CalFree DBP MAE | CalFree mean MAE |
|---|---|---:|---:|---:|---:|
| **M0** | None | 10.579 | **12.150** | **7.886** | **10.018** |
| M1 | Heart rate | 10.605 | 12.689 | 8.266 | 10.477 |
| M2 | Age group | 10.603 | 13.026 | 8.016 | 10.521 |
| M3 | Heart rate + age group | 10.580 | 12.797 | 8.213 | 10.505 |
| M4 | Heart rate + age group + BP pattern | 10.597 | 12.299 | 8.434 | 10.366 |

The auxiliary predictions were meaningful: M4 achieved 1.765 bpm heart-rate
MAE, 58.2% age-group accuracy, and 59.2% BP-pattern accuracy on the test set.
However, every hard-sharing variant reduced BP accuracy relative to M0. The
current evidence therefore supports retaining M0/B6-4 and treating gradient
conflict management, task-weight scheduling, or partially shared encoders as
future controlled experiments.

| Experiment | Configuration |
|---|---|
| M0 | `configs/pulsedb_m0_bp_only.yaml` |
| M1 | `configs/pulsedb_m1_heart_rate.yaml` |
| M2 | `configs/pulsedb_m2_age_group.yaml` |
| M3 | `configs/pulsedb_m3_hr_age.yaml` |
| M4 | `configs/pulsedb_m4_hr_age_bpclass.yaml` |
