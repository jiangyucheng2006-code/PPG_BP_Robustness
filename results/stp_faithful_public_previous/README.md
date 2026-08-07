# Previous public STP run

These are the completed results produced before the method re-audit. They are
preserved so the earlier experiment remains inspectable; none of these weights
is used by the new `stp_public_method_v2` run.

## Three-stage result (seed 42)

- F1 used all 330 public unpaired subjects and completed 50 epochs. Test
  reconstruction MSE was 0.003550 and MAE was 0.028498.
- F2 used the 200 paired public MIMIC-III subjects. It stopped after nine
  epochs and predicted every test window as the low-BP class: accuracy 0.3122,
  macro recall 0.3333.
- F3 stopped after 13 epochs. Test SBP MAE was 25.5875 mmHg, DBP MAE was
  13.2423 mmHg, and mean MAE was 19.4149 mmHg. It did not meet the AAMI
  numerical error limits.

## Transfer-path check

`summary.csv` contains all seven completed F3 comparisons. The best observed
mean MAE was 17.9182 mmHg for the complete F1-to-F2-to-F3 path at seed 73.
However, random initialization at seed 42 reached 18.0910 mmHg, and the F2
classifier collapsed at seed 42. The previous run therefore did not establish
a reliable benefit from the STP transfer path.

This result is not directly comparable to the article's headline result. The
authors used 683 additional private paired ICU subjects and evaluated more than
85 test subjects; the public-only F3 test set contains 30 subjects.
