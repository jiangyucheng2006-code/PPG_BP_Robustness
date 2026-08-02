# STP public-method reproduction

This track reproduces the implementation details that can be verified from the
STP article, its figures, and the authors' corresponding patent. It is kept
separate from the earlier S1--S3 approximation. See `STP_METHOD_AUDIT.md` for
the confirmed/provisional boundary.

## Reproduction boundary

It is not a literal reproduction of the paper's headline experiment. The
authors used 683 private, high-quality paired ICU subjects from seven hospitals,
did not publish their subject identifiers or code, and did not disclose all
network widths, layer counts, training epochs, batch size, or transformation
amplitudes. The public-only experiment therefore cannot contain the same 1,213
subjects or guarantee the reported result.

## Disclosed data roles

| Source | Subjects | BP labels | Role |
|---|---:|---|---|
| MIMIC-III, invasive BP missing | 300 | No | Self-supervised reconstruction only |
| WESAD | 15 | No | Self-supervised reconstruction only |
| PPG-DaLiA | 15 | No | Self-supervised reconstruction only |
| MIMIC-III, paired PPG--ABP | 200 | Yes | Pattern adaptation and BP regression |
| Private multi-hospital ICU cohort | 683 | Yes | Unavailable; omitted |

The executable public reproduction uses 530 subjects when all requested public
records pass quality screening. The 200 paired subjects are split by subject at
70/15/15, producing 140/30/30 subjects. The unpaired sources are also split by
subject for leakage-free reconstruction evaluation.

## Disclosed preprocessing reproduced

- resample every complete record to 125 Hz;
- PPG: nine-level db8 DWT, zero cA9 and the three highest-frequency detail
  groups cD3--cD1, then reconstruct;
- ABP: 0.5--35 Hz FIR band-pass;
- reject horizontal/missing waveforms from first differences;
- reject BP windows with DBP below 25 mmHg or SBP not above DBP;
- create an average PPG template within each record and reject windows whose
  mean difference is too high or correlation too low using 3-sigma thresholds;
- min-max normalize each PPG window to [0, 1];
- align paired PPG and ABP by maximum cross-correlation within 500 ms;
- segment paired records into five-cycle windows with a two-cycle stride;
- assign BP pattern labels as hypotension (<90 SBP or <60 DBP), hypertension
  (>=130 SBP or >=80 DBP), and normal otherwise.

Every processed manifest entry records its role, source, split, filters, phase
lag, and quality thresholds so exclusions are auditable.

## Disclosed model/training reproduced

1. F1: one-dimensional projection plus positional embedding, Transformer
   encoder, causally masked Transformer decoder, and reconstruction MSE after
   one of eleven PPG transformations.
2. F2: transfer the F1 encoder, remove the decoder, add a gradient reversal
   layer and one-dimensional PatchGAN, and learn the three BP patterns.
3. F3: transfer the F2 encoder, apply global average pooling, and estimate SBP
   and DBP using MSE.
4. Use Adam at 1e-3 with beta1=0.8 and beta2=0.999. The patent's translated
   phrase "weight decay 0.999, momentum 0.8" is interpreted as Adam's standard
   beta2/beta1 parameters; actual weight decay is zero.

## Provisional details

The following values are explicit configuration choices rather than claims
about the authors' unreleased implementation: 512 resampled samples per window,
128-dimensional embeddings, four encoder layers, two decoder layers, eight
heads, 256 feed-forward features, the PatchGAN channel widths, 50/30/50 epochs,
batch size 8, and the numeric severity ranges of transformations whose symbols
were defined but values were not published.

The epoch values are maximum budgets. Every stage uses validation-based early
stopping with a patience of eight epochs; the paper did not disclose its epoch
count or stopping rule.

These values must remain labelled provisional in any report. A true exact
reproduction requires the authors' code, exact public subject list, private
683-subject cohort, and missing hyperparameters.

## Commands

Run the complete public pipeline in a new output directory:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\run_stp_faithful_public.ps1 -SkipDataPreparation -RunName stp_public_method_v2
```

Outputs are written to `outputs/<RunName>/F1`, `F2`, and `F3`. A strict cohort
audit is written to `outputs/<RunName>/data_audit.json` before training starts.
The runner never points to the earlier STP checkpoints.

## Transfer-path ablation

The overnight ablation compares the complete `F1 -> F2 -> F3` transfer path
against direct `F1 -> F3` transfer and BP regression from random
initialization. Seeds 42 and 73 form a paired two-seed comparison, with a
third seed (137) for the complete path. Data, subject splits, architecture,
loss, and epoch budget remain fixed. This isolates the value of the
self-supervised and BP-pattern stages without changing the test set.

The runner waits for the faithful baseline, executes each comparison
sequentially, and writes `outputs/stp_faithful_public_ablations/summary.csv`.

Primary sources:

- Ma et al., *Expert Systems with Applications* 2024,
  DOI: 10.1016/j.eswa.2024.123809.
- CN115836846A, the authors' corresponding STP patent.
