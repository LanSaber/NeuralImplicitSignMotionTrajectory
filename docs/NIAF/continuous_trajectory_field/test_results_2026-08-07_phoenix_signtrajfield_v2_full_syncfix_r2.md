# SignTrajField-v2 PHOENIX Dual-Mode Test Results

Evaluation date: 2026-08-07

Status: complete

## 1. Checkpoint and protocol

The requested rolling checkpoint was:

```text
experiments/NIAF/continuous_trajectory_field/
  phoenix_signtrajfield_v2_full_syncfix_r2/checkpoints/best.pt
```

Because training was still active, evaluation used an immutable copy of that
file made before either mode was launched:

```text
evaluation/checkpoints/best_epoch0010_step0000280.pt
SHA-256: f532b28a61d60dc9548a4f0d74c59a647b9538c811d04c773b2637bd34b0c8f3
```

Checkpoint identity and evaluation protocol:

| Item | Value |
|---|---|
| Model type | `dual_mode_continuous_trajectory_field` |
| Trajectory contract | 2 |
| Epoch / global step | 10 / 280 |
| Dataset / split | PHOENIX / `test` |
| Test examples | 642 |
| Length mode | Text-predicted duration and frame count |
| Context / sample rate | 20 / 20 FPS |
| Text-only prior | Disabled; DTW reference branch is `continuous_coarse_smplx` |
| Word-prior prior | Online frozen SoftArranger; DTW reference branch is `adapter_context_smplx` |
| Default metric | Translated DTW-MPJPE |
| PA metric | Partwise PA-DTW-MPJPE, fitting and scoring the same keypoint subset |

Both modes exported 642 samples, produced 642 pairs in both DTW reports, and
recorded zero skipped prior sequences. The duration prediction is text-only by
design, so both modes have the same duration MAE: `0.614823` seconds.

All DTW values below are `flow` means. Raw values are accumulated costs;
normalized values divide by the optimal warping-path length. Lower is better.

## 2. Default DTW by part

| Part | Text-only raw | Word-prior raw | Text-only normalized | Word-prior normalized | Prior change in normalized score | Better mode |
|---|---:|---:|---:|---:|---:|---|
| Body | 4.007737 | 4.026999 | 0.046647 | 0.046891 | +0.524% | Text-only |
| Left hand | 5.198709 | 5.253898 | 0.052178 | 0.052582 | +0.775% | Text-only |
| Right hand | 5.142658 | 5.155137 | 0.055310 | 0.055524 | +0.386% | Text-only |
| Whole body | 10.081853 | 10.226834 | 0.117494 | 0.118881 | +1.181% | Text-only |

## 3. Corrected partwise PA-DTW by part

| Part | Text-only raw | Word-prior raw | Text-only normalized | Word-prior normalized | Prior change in normalized score | Better mode |
|---|---:|---:|---:|---:|---:|---|
| Body | 3.417415 | 3.434615 | 0.039627 | 0.039747 | +0.302% | Text-only |
| Left hand | 1.103774 | 1.099669 | 0.012800 | 0.012729 | -0.555% | Word-prior |
| Right hand | 1.263887 | 1.272612 | 0.014715 | 0.014811 | +0.655% | Text-only |
| Whole body | 6.539330 | 6.605332 | 0.075265 | 0.075562 | +0.394% | Text-only |

The PA reports record metric preset `t2m_partwise_pa_same_subset`. Thus the body
values use the corrected definition: the Procrustes transform is fitted and
scored on the same 12 upper-body keypoints.

## 4. Ground-truth-frame-count aligned diagnostics

These diagnostics query both modes at the ground-truth frame count. They
isolate pose and trajectory quality from predicted-duration error and should
not be mixed with the predicted-duration DTW scores above.

| Metric | Text-only | Word-prior | Prior relative change | Result |
|---|---:|---:|---:|---|
| Endpoint loss | 10.230866 | 10.353960 | +1.203% | Text-only better |
| Joint loss | 1.339079 | 1.362611 | +1.757% | Text-only better |
| Hand-relative loss | 0.103340 | 0.103930 | +0.570% | Text-only better |
| Path loss | 0.630908 | 0.627059 | -0.610% | Word-prior better |
| FK jerk loss | 0.303516 | 0.303592 | +0.025% | Essentially tied |
| Dense analytic FK jerk ratio | 0.212856 | 0.287711 | +35.167% | Word-prior has more target-relative jerk energy |
| Duration loss | 0.117287 | 0.117287 | 0.000% | Tied by design |
| Total loss | 11.126500 | 11.250222 | +1.112% | Text-only better |

The dense analytic FK jerk ratio is descriptive rather than a simple
lower-is-better error. The configured FK jerk loss is the appropriate aligned
loss comparison and is nearly identical between the two modes.

## 5. Selection result and conclusion

| Selection field | Value |
|---|---:|
| Text-only composite score | 11.610074 |
| Word-prior composite score | 11.737144 |
| Word-prior relative degradation | 1.094% |
| Maximum allowed degradation | 2.000% |
| Feasible | `true` |

**Conclusion:** text-only is the better overall inference mode for this
checkpoint. It has lower default DTW for every part, lower whole-body PA-DTW,
and better aligned endpoint, joint, hand-relative, and total losses. Word-prior
provides small localized improvements in normalized PA left-hand articulation
(`0.555%`) and aligned hand-path loss (`0.610%`), but these do not offset its
broader regressions. The word-prior mode still passes the configured 2%
non-regression gate.

## 6. Result locations

```text
experiments/NIAF/continuous_trajectory_field/
  phoenix_signtrajfield_v2_full_syncfix_r2/evaluation/
    best_epoch0010_step0000280_test_predicted_text_only/
      export_summary.json
      dtw_mpjpe_t2m_default_h2s_betas.json
      dtw_mpjpe_t2m_pa_h2s_betas.json
    best_epoch0010_step0000280_test_predicted_word_prior/
      export_summary.json
      dtw_mpjpe_t2m_default_h2s_betas.json
      dtw_mpjpe_t2m_pa_h2s_betas.json
    best_epoch0010_step0000280_test_aligned_both.json
```
