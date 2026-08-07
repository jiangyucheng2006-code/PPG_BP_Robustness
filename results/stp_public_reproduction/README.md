# STP faithful public-data reproduction (v4)

This run follows the public parts of the disclosed STP pipeline and keeps all
unpublished choices explicitly provisional. The full pipeline completed in
approximately 8 hours 20 minutes on an RTX 4060 Laptop GPU.

## Audited data

| Source and role | Subjects | Windows |
|---|---:|---:|
| MIMIC-III, unpaired pretraining | 300 | 122,904 |
| WESAD, unpaired pretraining | 15 | 20,800 |
| PPG-DaLiA, unpaired pretraining | 15 | 31,175 |
| MIMIC-III, paired BP training | 200 | 129,389 |
| **Total** | **530** | **304,268** |

The strict audit passed. All arrays were finite and within their expected
ranges, all required files were present, BP-pattern labels were consistent,
and no subject crossed a data split or appeared in both paired and unpaired
MIMIC roles. The paired cohort used a fixed subject-wise 140/30/30
train/validation/test split.

## Held-out test results

| Stage | Best / completed epoch | Test result |
|---|---:|---|
| F1 reconstruction | 20 / 28 | MSE 0.01487; MAE 0.07037 |
| F2 BP-pattern adaptation | 1 / 9 | accuracy 49.91%; macro recall 33.33% |
| F3 BP regression | 8 / 16 | SBP MAE 22.63; DBP MAE 11.66; mean MAE 17.15 mmHg |

F2 collapsed to the majority class: every one of the 20,622 test windows was
predicted as class 0. Consequently, its apparent accuracy does not indicate
useful three-class discrimination.

For F3, SBP RMSE was 27.64 mmHg and DBP RMSE was 15.09 mmHg. Both outputs
received BHS grade D and failed the numerical AAMI error checks. In addition,
the 30-subject public test split and its BP distribution are insufficient for
a formal AAMI evaluation.

## Interpretation

The result verifies that the audited public-only F1-to-F2-to-F3 implementation
runs end to end, but it does not reproduce the article's reported performance.
Compared with the earlier v2 attempt, mean MAE improved from 18.03 to 17.15
mmHg, while the F2 collapse remained. The unavailable 683-subject private
cohort, exact MIMIC record list, and undisclosed architecture and training
details prevent a claim of exact numerical reproduction.

Generated datasets and model checkpoints remain local. Machine-readable
results are provided in [`summary.json`](summary.json).
