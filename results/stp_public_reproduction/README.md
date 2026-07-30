# STP public-data reproduction

This run verifies the complete public-data S1-S3 training path:

1. transformed-PPG reconstruction using WESAD and PPG-DaLiA;
2. three-class BP-pattern adaptation using MIMIC-III;
3. SBP/DBP regression using MIMIC-III.

## Data

| Source | Subjects / record groups | Windows | Role |
|---|---:|---:|---|
| WESAD | 15 | 44,249 | S1 |
| PPG-DaLiA | 15 | 76,464 | S1 |
| MIMIC-III | 403 | 454,367 | S2 and S3 |

The MIMIC preparation attempted 500 public record groups. A total of 403
produced valid paired PPG/ABP windows; the rest were unavailable, interrupted
by the remote server, or failed signal-quality checks.

All splits are subject-wise. No window from a validation or test subject is
used for training.

## Results

| Stage | Best epoch | Test result |
|---|---:|---|
| S1 reconstruction | 42 | MSE 0.0135; MAE 0.0724 |
| S2 BP-pattern classification | 5 | accuracy 36.63%; macro recall 35.19% |
| S3 BP regression | 8 | SBP MAE 23.13 mmHg; DBP MAE 12.65 mmHg |

The final mean MAE is 17.89 mmHg. SBP and DBP both receive BHS grade D and do
not meet the numerical AAMI error criteria.

## Interpretation

The run confirms that data preparation, sequential encoder transfer, training,
early stopping, and held-out evaluation work end to end. It does **not**
reproduce the paper's reported numerical result. The main observed limitation
is weak S2 pattern separation, followed by large subject-independent S3 error.

This is a public-only reproduction. The original private Mindray cohort,
exact record list, and unpublished training details are unavailable. The
result is therefore used as a transparent baseline for subsequent
optimization, not as a claim of exact paper reproduction.

Model checkpoints remain local because generated weights and datasets are
excluded from Git. The machine-readable metrics are in
[`summary.json`](summary.json).
