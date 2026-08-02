# STP downstream optimization suite

## Purpose

The first complete public-data run produced 36.63% S2 pattern accuracy and
17.89 mmHg mean S3 MAE. The optimization suite tests specific explanations
for those errors instead of only increasing the epoch count.

The S1 reconstruction checkpoint is fixed for every experiment. This isolates
the effect of downstream training and keeps every comparison on the same
pretrained representation and subject-wise splits.

## Controlled variants

| ID | Change from the preceding variant | Question |
|---|---|---|
| O0 | Mean pooling, MSE, window-proportional sampling | Reproduce the downstream control with longer patience |
| O1 | Huber BP loss and class-aware S2 loss | Are outliers and class imbalance limiting training? |
| O2 | Subject-balanced sampling | Are long records dominating shorter subjects? |
| O3 | Three-epoch head warm-up and 0.25× encoder LR | Does aggressive fine-tuning erase the S1 representation? |
| O4 | Attention pooling | Can the model select the most informative pulse tokens? |
| O5 | Attentive mean and standard deviation | Does beat-to-beat variability add information beyond the mean token? |
| O6 | Pulse-pressure consistency objective | Does a physiological relationship improve joint SBP/DBP learning? |

## Transfer-path comparison

Every variant is trained in two forms:

- `direct_s1`: transfer the S1 encoder directly to BP regression;
- `via_s2`: run BP-pattern adaptation before BP regression.

This directly tests whether the weak S2 classifier helps or damages the final
regressor. It prevents the published three-stage sequence from being accepted
without an ablation.

## Repeated evaluation

The complete matrix contains:

- 7 controlled variants;
- 2 transfer paths;
- 3 random seeds (`42`, `17`, and `73`);
- 21 S2 runs and 42 S3 runs;
- 63 sequential training stages in total.

Each stage permits up to 120 epochs with eight-epoch early stopping. Based on
the measured throughput of the first run, the expected wall-clock duration is
approximately 7-10 hours. Multiple seeds are required because a single
favorable initialization is not sufficient evidence of improvement.

The queue is resumable. Every stage stores its resolved configuration,
history, logs, checkpoints, and metrics under
`outputs/stp_optimization_suite`. `status.json` and `summary.json` track the
current stage and best completed result.

## Commands

Start or resume the complete queue:

```powershell
python scripts/run_stp_optimization_suite.py
```

Open the live monitor:

```powershell
powershell -ExecutionPolicy Bypass -File scripts/watch_stp_optimization.ps1
```

The smoke test uses all seven variants, one seed, one epoch, and a small data
subset. All 21 smoke-test stages completed before the full queue was started.
