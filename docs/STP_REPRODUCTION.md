# STP public-data reproduction

This is a separate reproduction of:

> C. Ma, P. Zhang, H. Zhang, Z. Liu, F. Song, Y. He, and G. Zhang,
> “STP: Self-supervised transfer learning based on transformer for
> noninvasive blood pressure estimation using photoplethysmography,”
> *Expert Systems with Applications*, vol. 249, 2024, 123809.
> https://doi.org/10.1016/j.eswa.2024.123809

It does not replace the XResNet baseline or the C-series robustness
experiments.

## Reproduction boundary

The implementation separates details visible in the publisher's paper preview
and official figures from settings that cannot be verified without the full
methods text or the private data.

### Directly confirmed from the paper

- Three sequential modules:
  1. self-supervised representation learning;
  2. BP pattern adaptation;
  3. BP value estimation.
- A shared 1-D convolutional projection, positional embedding, and Transformer
  encoder is transferred between the three modules.
- The first module uses a Transformer decoder and reconstruction loss.
- The pattern discriminator predicts hypotensive, normotensive, or
  hypertensive BP.
- The final regressor estimates BP values.
- Every signal window contains five pulse cycles and advances by two cycles.
- The eleven illustrated transformations are:
  Gaussian noise, powerline noise, motion artifacts, baseline drift,
  respiratory sinus arrhythmia (RSA), random masking, hard clipping,
  amplitude negation, temporal inversion, temporal permutation, and temporal
  warping.
- Evaluation is subject-wise and the reported full cohort contains 1,213
  subjects.
- The reported full-cohort errors are 0.85 ± 4.21 mmHg for SBP and
  0.49 ± 2.76 mmHg for DBP.

### Public-data reconstruction

The public part is prepared from:

- [WESAD](https://archive.ics.uci.edu/dataset/465/wesad): unpaired wrist BVP
  for reconstruction pretraining;
- [PPG-DaLiA](https://archive.ics.uci.edu/dataset/495/ppg+dalia): unpaired
  wrist BVP for reconstruction pretraining;
- [MIMIC-III Waveform Database](https://physionet.org/content/mimic3wdb/1.0/):
  paired PPG and invasive ABP for pattern adaptation and BP regression.

The dataset combination is strongly supported by the STP references and by a
same-author study using the same 1,213-subject cohort. That cohort contains
500 MIMIC-III subjects, 15 WESAD subjects, 15 PPG-DaLiA subjects, and 683
private Mindray ICU subjects. The private 683-subject cohort is omitted here,
as requested. The public-only run therefore cannot be treated as a numerical
replication of the paper's full-cohort result.

### Settings that remain provisional

The accessible paper preview does not expose the exact:

- Transformer depth, embedding width, head count, and decoder depth;
- fixed resampled waveform length;
- filter cutoffs and outlier thresholds;
- magnitude distributions for the eleven transformations;
- BP-pattern threshold rule and the pattern adversarial-loss equation;
- exact MIMIC record IDs, recording duration, subject split ratio, optimizer,
  batch size, and learning-rate schedule.

These values are explicit in the YAML files and can be corrected without
changing the pipeline when the full paper, supplementary material, or original
code becomes available. They must not be described as author-confirmed
hyperparameters.

## Implementation

| Stage | Configuration | Input | Objective | Transferred output |
|---|---|---|---|---|
| S1 | `stp_s1_public_pretrain.yaml` | transformed WESAD/PPG-DaLiA PPG | reconstruct clean PPG | Transformer encoder |
| S2 | `stp_s2_public_pattern.yaml` | paired MIMIC-III PPG | 3-class BP pattern loss | adapted Transformer encoder |
| S3 | `stp_s3_public_bp.yaml` | paired MIMIC-III PPG | normalized SBP/DBP MSE | final BP model |

The older `scripts/pretrain_stp.py` belongs only to C6. It predicts
transformation type using XResNet and is intentionally retained as an
STP-inspired ablation. The faithful reproduction entry point is
`scripts/train_stp.py`.

## Data preparation

Install the updated project dependencies first:

```powershell
python -m pip install -e ".[dev]"
```

Download and prepare WESAD and PPG-DaLiA:

```powershell
$env:PPG_BP_DATA_ROOT = "D:/Datasets/PPG_BP_Robustness"
python scripts/prepare_stp_public_data.py `
  --raw-root "$env:PPG_BP_DATA_ROOT/STP_Public_Raw" `
  --output "$env:PPG_BP_DATA_ROOT/processed/stp_public" `
  --download-unpaired `
  --skip-mimic
```

Prepare the paired public records separately. The default uses a published
index of simultaneous PPG/ABP MIMIC-III waveform records, selects 500 unique
record groups with a fixed seed, and streams no more than 30 minutes from each
record rather than downloading the multi-terabyte database:

```powershell
python scripts/prepare_stp_public_data.py `
  --raw-root "$env:PPG_BP_DATA_ROOT/STP_Public_Raw" `
  --output "$env:PPG_BP_DATA_ROOT/processed/stp_public" `
  --skip-wesad `
  --skip-ppg-dalia `
  --mimic-max-subjects 500
```

The paired-record index comes from
[`v3551G/BP-prediction-survey`](https://github.com/v3551G/BP-prediction-survey).
It makes the public pipeline practical, but it is not claimed to be the
authors' unpublished list of 500 MIMIC records. Use
`--discover-mimic-records` to inspect the PhysioNet matched database directly;
this is much slower.

The preparation script merges new subjects into the existing manifest, so the
unpaired and paired sources may be prepared in separate sessions. They can
also be prepared in one command:

```powershell
python scripts/prepare_stp_public_data.py `
  --raw-root "$env:PPG_BP_DATA_ROOT/STP_Public_Raw" `
  --output "$env:PPG_BP_DATA_ROOT/processed/stp_public" `
  --download-unpaired `
  --mimic-max-subjects 500
```

## Sequential training

Run the stages in order because each stage reads the preceding best encoder:

```powershell
python scripts/train_stp.py `
  --config configs/stp_s1_public_pretrain.yaml `
  --output outputs/stp_s1_public_pretrain `
  --resume

python scripts/train_stp.py `
  --config configs/stp_s2_public_pattern.yaml `
  --output outputs/stp_s2_public_pattern `
  --resume

python scripts/train_stp.py `
  --config configs/stp_s3_public_bp.yaml `
  --output outputs/stp_s3_public_bp `
  --resume
```

Each output directory contains `best.pt`, `last.pt`, `history.json`, and
`metrics.json`. Checkpoints also store the encoder separately so the transfer
between S1, S2, and S3 is auditable.
