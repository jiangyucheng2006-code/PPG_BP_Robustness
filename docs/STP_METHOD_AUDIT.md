# STP method audit

This audit separates facts that are explicitly disclosed by Ma et al. from
implementation choices that cannot be recovered from the public article,
figures, or the corresponding patent. It must be read before interpreting any
STP result in this repository.

## Publicly confirmed protocol

| Area | Confirmed method | Repository implementation |
|---|---|---|
| Cohort | 300 MIMIC-III subjects without invasive BP, 15 WESAD subjects, and 15 PPG-DaLiA subjects for unpaired pretraining | F1 selects only `pretrain_unpaired` entries from those three sources |
| Paired public cohort | 200 MIMIC-III subjects with PPG and invasive BP | F2 and F3 select only `supervised_paired` MIMIC-III entries; paired and unpaired MIMIC records are selected by canonical patient ID with cross-role overlap rejected |
| Private cohort | 683 paired ICU subjects from seven hospitals | unavailable and not substituted |
| Sampling | all signals resampled to 125 Hz | implemented before segmentation |
| PPG filtering | nine-level db8 DWT; approximation/baseline and the stated high-frequency detail groups removed | implemented as cA9 plus cD3-cD1 removal; coefficient-index interpretation is recorded in code |
| ABP filtering | FIR band-pass at 0.5-35 Hz | used for phase alignment; unfiltered calibrated ABP is retained for BP labels |
| Quality screening | first-difference horizontal/missing check, DBP at least 25 mmHg, within-record mean-template difference and correlation with 3-sigma limits | implemented per subject/record before PPG normalization; windows containing originally missing PPG or ABP samples are rejected rather than silently repaired |
| Alignment | maximum PPG-ABP correlation with up to 500 ms delay | implemented without circular wraparound |
| Segmentation | paired PPG-ABP uses five-cycle windows with two-cycle overlap; self-supervised PPG uses fixed-length windows | implemented as a three-cycle step for paired windows and fixed, non-overlapping provisional 512-sample windows for F1 |
| Split | subject-wise 70/15/15, with no subject shared between splits | implemented separately for unpaired and paired roles |
| F1 | one-dimensional convolutional projection, positional embedding, Transformer encoder-decoder, causal decoder mask, clean-signal reconstruction with MSE after one of 11 transformations | implemented |
| F2 | transfer the F1 encoder, remove the decoder, apply global average pooling to obtain the disclosed 1 x N feature vector, then add GRL and a fully convolutional PatchGAN to learn low/normal/high BP patterns | implemented with GAP over the token/time axis so the N embedding features are retained before GRL/PatchGAN |
| F3 | transfer the F2 encoder, add a BP value regressor, estimate SBP and DBP with MSE | implemented |
| BP patterns | high: SBP >=130 or DBP >=80; low: SBP <90 or DBP <60; otherwise normal | implemented with high-BP precedence in a conflicting pair |
| Optimizer | Adam, learning rate 1e-3, momentum 0.8, value 0.999 described as weight decay in the patent | interpreted as Adam beta1=0.8 and beta2=0.999; actual L2 weight decay is zero |

The 11 confirmed transformations are Gaussian noise, 50 Hz power-line noise,
motion artifacts affecting 1-10% of the signal, 0.05 Hz baseline drift,
respiratory sinus arrhythmia modulation, random geometric masking, hard
clipping at both peaks and valleys, amplitude multiplication by -1, temporal
inversion, temporal permutation, and global temporal resampling by a scale
factor. Validation and test corruptions are seeded deterministically so early
stopping never compares different random validation tasks.

## Not publicly recoverable

The following are not disclosed sufficiently to reproduce the authors' exact
experiment:

- the 683-subject private cohort and its subject identifiers;
- the exact 200 paired and 300 unpaired MIMIC-III record list;
- Transformer layer count, embedding width, head count, feed-forward width,
  dropout, normalization placement, initialization, and the decoder input/query
  construction;
- convolution and PatchGAN channel widths and kernel sizes;
- the detailed BP value regressor and its temporal pooling rule;
- batch size, epoch count, stopping rule, and learning-rate schedule;
- numerical probabilities and strengths for most of the 11 transformations;
- the fixed-length F1 window length and stride, the exact PPG/ABP peak
  detector, BP aggregation within a five-cycle window, and the amount of each
  MIMIC record used;
- the exact definition of the template difference/correlation statistics;
- the exact FIR tap count and the coefficient-index convention behind the
  patent's wavelet-level notation;
- the exact multi-class extension and update schedule for the published
  adversarial equations.

The configured 512 samples, 128-dimensional embedding, four encoder layers,
two decoder layers, eight heads, 256 feed-forward features, batch size 8, and
50/30/50 maximum epochs are therefore provisional. The public record selector
uses at most 30 minutes from one usable segment per MIMIC subject, also a
provisional choice. Early stopping after eight validation epochs without
improvement is a user-requested safety rule, not a paper-reported setting.

## Consequence for comparison

The paper evaluates 1,213 subjects, so a 15% subject-wise test set exceeds 85
subjects. The public-only F2/F3 experiment has 200 paired subjects and therefore
only 30 test subjects. It cannot satisfy the AAMI subject-count requirement and
cannot be treated as a reproduction of the reported 0.85 +/- 4.21 mmHg SBP and
0.49 +/- 2.76 mmHg DBP errors, even when the numerical BP-error criteria are
reported for reference.

Primary sources:

- Ma et al., *Expert Systems with Applications* 249 (2024), article 123809,
  DOI: 10.1016/j.eswa.2024.123809.
- CN115836846A, corresponding patent and disclosed implementation flow.
