# STP optimization suite result

All 63 planned stages completed: 21 S2 runs and 42 S3 runs covering seven
variants, three seeds, and direct-F1 versus via-F2 transfer paths.

The lowest single-run mean MAE was 15.8477 mmHg (`O1_robust_objectives`, direct
F1 transfer, seed 42: SBP 20.0560, DBP 11.6394 mmHg). The lowest via-F2 result
was 16.5560 mmHg (`O1_robust_objectives`, seed 73). No run met AAMI numerical
error limits.

The identical direct/via values for O3-O6 are not evidence of successful F2
adaptation. Their three-epoch encoder freeze lasted through the best F2 epoch,
so the selected F2 encoder remained byte-identical to F1. These results belong
to the earlier 403-subject approximation and must not be mixed with the later
530-subject public-method run.

`aggregate.csv` reports the three-seed average for each variant and transfer
path.
