# C0-C6 robustness ablation

This series evaluates whether PPG-only BP estimation remains stable when the
input waveform is corrupted. All models use seed 42, the same subject-disjoint
PulseDB VitalDB training/validation split, and the fixed calibration-free test
set.

The robustness suite applies eight controlled corruptions at light, medium,
and strong severity:

- motion artifact;
- transient spikes;
- simulated contact compression;
- baseline drift;
- Gaussian noise;
- amplitude clipping;
- sensor gain and offset;
- low-pass device response.

These corruptions are synthetic. In particular, contact compression is a
waveform simulation and is not a substitute for evaluation on real
pressure-labelled recordings.

## Results

MAE and prediction-shift values are in mmHg. Robust mean MAE is the unweighted
average across all 24 corruption conditions. Prediction shift is the absolute
change between clean and corrupted predictions for the same input segment.

| ID | Training strategy | Clean mean MAE | Robust mean MAE | Mean prediction shift | Worst-condition MAE |
|---|---|---:|---:|---:|---:|
| C0 | Unmodified M0/B6-4 checkpoint | **10.018** | 21.931 | 17.384 | 67.376 |
| C1 | Weak single-artifact augmentation | 10.316 | 10.729 | 3.170 | 12.234 |
| **C2** | **Full mixed-artifact augmentation** | 10.342 | **10.618** | **2.374** | 12.025 |
| C3 | Contact-compression and motion emphasis | 10.349 | 10.776 | 2.622 | 12.991 |
| C4 | Clean/corrupted prediction consistency | 10.455 | 10.636 | 2.888 | **11.748** |
| C5 | Artifact-type and severity auxiliary heads | 10.403 | 10.784 | 3.206 | 11.952 |
| C6 | Transformation pretraining and artifact-aware fine-tuning | 10.513 | 10.949 | 3.236 | 12.346 |

C2 reduced the average corrupted MAE by 51.6% and the prediction shift by
86.3% relative to C0, while increasing clean mean MAE by 0.324 mmHg. Under
strong simulated motion, Gaussian noise, and contact compression, mean MAE
changed as follows:

| Corruption | C0 | C2 |
|---|---:|---:|
| Motion artifact | 64.493 | **12.025** |
| Gaussian noise | 67.376 | **11.101** |
| Contact compression | 13.702 | **10.661** |

C4 produced the lowest worst-condition error, but C2 had the best average
corrupted accuracy and the smallest clean-to-corrupted prediction shift.
Therefore C2 is the selected robustness baseline and C4 is retained as the
consistency-learning control.

C6 is STP-inspired rather than a reproduction of the published STP model. Its
transformation classifier reached 99.38% validation accuracy, but that
representation did not improve downstream BP estimation. A faithful STP
reproduction is tracked separately from this robustness ablation.

No C-series model meets the AAMI numerical limits or BHS Grade A on the clean
calibration-free test set. C2 clean SBP/DBP MAEs are 12.468/8.216 mmHg.

Exact aggregate values are recorded in [`summary.json`](summary.json).
Checkpoints, raw waveforms, and full local logs are intentionally not tracked.
