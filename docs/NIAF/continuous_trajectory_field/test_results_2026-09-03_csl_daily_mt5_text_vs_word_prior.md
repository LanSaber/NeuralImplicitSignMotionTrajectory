# CSL-Daily mT5 Text-Only vs Word-Prior Test Results

Evaluation date: 2026-09-03

Status: complete

## 1. Checkpoints and comparison

Both checkpoints were selected by validation performance before the stopped
training jobs were evaluated. Test metrics were not used for checkpoint
selection. Immutable copies were made before evaluation:

| Training condition | Evaluation checkpoint | Epoch / step | SHA-256 |
|---|---|---:|---|
| Pure mT5 text-only | `csl_daily_signtrajfield_v2_mt5_text_only_full/evaluation/checkpoints/best_epoch0010_step00000720.pt` | 10 / 720 | `06ca0a2613005b6e3949bab0e5d7ded999b212723debd3e7685a58c077e44c54` |
| mT5 with 50% word-prior dropout | `csl_daily_signtrajfield_v2_mt5_word_prior_full/evaluation/checkpoints/best_epoch0010_step00000720.pt` | 10 / 720 | `7620dc20da2de868546bf281517dbee2b6dad72a5b842c275ce642b761ffe6e4` |

The test matrix contains three conditions:

1. the pure text-only checkpoint with the prior disabled;
2. the mixed-trained checkpoint with the prior disabled; and
3. the same mixed-trained checkpoint with the prior enabled.

This separates the effect of mixed prior training from the effect of enabling
the prior at inference time.

## 2. Protocol and integrity checks

| Item | Value |
|---|---|
| Dataset / split | CSL-Daily / `test` |
| Test examples | 1,176 |
| Main text encoder | mT5-base, frozen, 768-dimensional |
| Predicted-length protocol | Text-predicted duration and frame count |
| Context / sample rate | 20 / 20 FPS |
| Default metric | Translated DTW-MPJPE |
| PA metric | Corrected partwise PA-DTW-MPJPE, fitting and scoring the same keypoint subset |
| Aligned diagnostics | Ground-truth frame count, reported separately from DTW |
| Retrieval bank | Training split only: 133,689 entries, 2,000 lexicon keys |
| Prior generation | Online frozen SoftArranger with `x=None`; no test motion or test scaffold cache is used |

The missing 1,176 test FK entries were generated without overwriting the
existing train/validation cache. All three exports use the same ordered test
manifest, whose SHA-256 is
`c5753aba5a07c7242c8491a1d5eb6690e4443a0253461ac110b482b383ff8669`.

Every condition produced 1,176 sample NPZ files and 1,176 ground-truth NPZ
files. Every default and PA report contains 1,176 pairs, 9,408 metric rows,
and zero skipped reference sequences. No evaluation job recorded a traceback,
runtime error, missing-cache error, or OOM.

## 3. Predicted duration

| Condition | Duration MAE (seconds) |
|---|---:|
| Pure text-only | 1.099569 |
| Mixed, prior off | 1.099998 |
| Mixed, prior on | 1.099998 |

Duration prediction is text-only by design, so the two modes of the mixed
checkpoint have identical duration error.

## 4. Default translated DTW

All values are flow predictions. Raw values are accumulated DTW costs;
normalized values divide by the optimal path length. Lower is better.
`Mean hands` is the arithmetic mean of the left- and right-hand values.

| Part | Pure raw | Pure normalized | Mixed-off raw | Mixed-off normalized | Mixed-on raw | Mixed-on normalized |
|---|---:|---:|---:|---:|---:|---:|
| Body | 5.202740 | 0.047014 | 5.178990 | 0.046942 | 5.179634 | 0.046949 |
| Left hand | 7.069874 | 0.055217 | 7.076670 | 0.055179 | 7.076415 | 0.055237 |
| Right hand | 6.917920 | 0.059594 | 6.898520 | 0.059387 | 6.898326 | 0.059431 |
| Mean hands | 6.993897 | 0.057405 | 6.987595 | 0.057283 | 6.987370 | 0.057334 |
| Whole body | 15.044536 | 0.131985 | 15.043736 | 0.132335 | 15.039980 | 0.132289 |

The three whole-body default-DTW results are effectively tied. Relative to
mixed-off, enabling the prior improves raw whole-body DTW by only `0.025%`
and normalized DTW by `0.035%`. Pure text-only has `0.266%` lower normalized
whole-body DTW than mixed-off despite essentially identical raw DTW.

## 5. Corrected partwise PA-DTW

All PA reports record metric preset `t2m_partwise_pa_same_subset`. `Mean hands`
is the arithmetic mean of the left- and right-hand values.

| Part | Pure raw | Pure normalized | Mixed-off raw | Mixed-off normalized | Mixed-on raw | Mixed-on normalized |
|---|---:|---:|---:|---:|---:|---:|
| Body | 4.718158 | 0.042002 | 4.743586 | 0.042337 | 4.741508 | 0.042311 |
| Left hand | 1.392284 | 0.011958 | 1.385327 | 0.011963 | 1.384191 | 0.011950 |
| Right hand | 1.868503 | 0.015885 | 1.857946 | 0.016066 | 1.857944 | 0.016072 |
| Mean hands | 1.630394 | 0.013921 | 1.621637 | 0.014015 | 1.621067 | 0.014011 |
| Whole body | 8.865489 | 0.075143 | 9.109214 | 0.076811 | 9.093177 | 0.076531 |

Pure text-only is clearly better on whole-body PA-DTW. Mixed-off is `2.75%`
worse in raw PA-DTW and `2.22%` worse after path normalization. Enabling the
prior recovers a small amount within the mixed model (`0.18%` raw and `0.36%`
normalized), but mixed-on remains `2.57%` worse raw and `1.85%` worse
normalized than pure text-only.

## 6. Ground-truth-frame-count aligned diagnostics

These diagnostics isolate pose and trajectory quality from predicted-duration
error. They must not be mixed with the predicted-duration DTW scores above.

| Metric | Pure text-only | Mixed, prior off | Mixed, prior on |
|---|---:|---:|---:|
| Composite selection score | 12.866417 | 12.862328 | 12.857185 |
| Endpoint loss | 11.554755 | 11.547154 | 11.543192 |
| Joint loss | 1.661020 | 1.659397 | 1.659049 |
| Hand-relative loss | 0.111239 | 0.111789 | 0.111760 |
| Path loss | 0.351425 | 0.347945 | 0.346233 |
| FK jerk loss | 0.235637 | 0.233073 | 0.232772 |
| Duration loss | 0.158624 | 0.158652 | 0.158652 |
| Total loss | 12.455627 | 12.444127 | 12.439889 |
| Dense analytic FK jerk ratio | 2.437347 | 1.878495 | 1.842395 |

All selectors are feasible. Inside the mixed checkpoint, enabling the prior
improves total loss by only `0.034%` and the composite by `0.040%`. The dense
analytic FK jerk ratio improves more visibly, but it is a descriptive ratio,
not a simple error metric.

The prior controls behaved correctly: prior availability and all prior gates
were exactly zero in the off pass. Availability was one in the on pass, with
mean body/face/left-hand/right-hand gates of approximately
`0.410/0.425/0.436/0.447`.

## 7. Paired whole-body uncertainty analysis

The same 1,176 examples were resampled 10,000 times with seed `20260903`.
Differences are candidate minus reference, so negative values favor the
candidate.

### Default DTW

| Contrast | Raw mean difference [95% CI] | Normalized mean difference [95% CI] |
|---|---:|---:|
| Mixed-off minus pure | -0.000800 [-0.045223, 0.044046] | +0.000350 [-0.000184, 0.000908] |
| Mixed-on minus pure | -0.004557 [-0.048126, 0.039252] | +0.000305 [-0.000223, 0.000859] |
| Mixed-on minus mixed-off | -0.003756 [-0.006780, -0.000797] | -0.0000458 [-0.0001083, 0.00000825] |

Pure and mixed training are indistinguishable under whole-body default DTW.
The tiny raw-DTW gain from enabling the prior is reliable in this bootstrap,
but the normalized interval still crosses zero.

### PA-DTW

| Contrast | Raw mean difference [95% CI] | Normalized mean difference [95% CI] |
|---|---:|---:|
| Mixed-off minus pure | +0.243725 [0.206509, 0.280733] | +0.001667 [0.001263, 0.002078] |
| Mixed-on minus pure | +0.227688 [0.191381, 0.264014] | +0.001388 [0.000995, 0.001789] |
| Mixed-on minus mixed-off | -0.016037 [-0.018853, -0.013388] | -0.000279 [-0.000392, -0.000185] |

The PA result is decisive: pure text-only is better than the mixed-trained
checkpoint in both inference modes. The prior yields a small reliable recovery
inside the mixed model but does not close the pure-text gap.

## 8. Conclusion

The pure mT5 text-only model is the best overall choice from this experiment.
Default translated DTW is a statistical tie, while the articulation-focused
whole-body PA-DTW clearly favors pure training. Word-prior inference makes the
mixed-trained model slightly better than its own prior-off mode, but its gains
are too small to justify the retrieval path here and do not recover the
performance lost relative to pure text-only training.

This supports keeping the mT5 sentence-conditioned model as the main baseline
before trying sentence-level retrieval. It also reinforces the earlier concern
that the current word prior is not delivering a useful enough motion prior,
even when it is available and its learned gates are active.

## 9. Leakage qualification

Inference retrieval is restricted to `manifest_train.jsonl`, and online
generation receives no test motion. However, the frozen SoftArranger checkpoint
records `word_split=all` from its original adapter training. Therefore the
retrieval bank is train-only, but the adapter weights are not a strictly
train-only artifact. A future publication-grade prior comparison should retrain
that adapter using training data only. This qualification does not affect the
pure text-only condition, which never initializes the adapter/provider.

## 10. Result locations

```text
experiments/NIAF/continuous_trajectory_field/
  csl_daily_signtrajfield_v2_mt5_text_only_full/evaluation/
    checkpoints/best_epoch0010_step00000720.pt
    best_epoch0010_step00000720_test_aligned_text_only.json
    best_epoch0010_step00000720_test_predicted_text_only/
      export_summary.json
      dtw_mpjpe_t2m_default_h2s_betas.json
      dtw_mpjpe_t2m_pa_h2s_betas.json

  csl_daily_signtrajfield_v2_mt5_word_prior_full/evaluation/
    checkpoints/best_epoch0010_step00000720.pt
    best_epoch0010_step00000720_test_aligned_both.json
    best_epoch0010_step00000720_test_predicted_text_only/
      export_summary.json
      dtw_mpjpe_t2m_default_h2s_betas.json
      dtw_mpjpe_t2m_pa_h2s_betas.json
    best_epoch0010_step00000720_test_predicted_word_prior/
      export_summary.json
      dtw_mpjpe_t2m_default_h2s_betas.json
      dtw_mpjpe_t2m_pa_h2s_betas.json
```

Slurm jobs: FK cache `142272`; predicted export/default/PA `142273`--`142275`;
aligned diagnostics `142276`--`142277`.
