# SignTrajField-v2 How2Sign Dual-Mode Test Results

Evaluation date: 2026-08-09

Status: complete

## 1. Checkpoint and protocol

The requested rolling training checkpoint was:

```text
experiments/NIAF/continuous_trajectory_field/
  how2sign_signtrajfield_v2_full/checkpoints/best.pt
```

Because training was still active, evaluation used an immutable copy made
before either inference mode was launched:

```text
evaluation/checkpoints/best_epoch0010_step0001200.pt
SHA-256: 02800c9b2bfacabe9128c456c738d513bfb66cdeb7a30d36668cdce342f9a944
```

Checkpoint identity and evaluation protocol:

| Item | Value |
|---|---|
| Model type | `dual_mode_continuous_trajectory_field` |
| Trajectory contract | 2 |
| Epoch / global step | 10 / 1,200 |
| Dataset / split | How2Sign / `test` |
| Test examples | 2,308 |
| Length mode | Text-predicted duration and frame count |
| Context / sample rate | 20 / 20 FPS |
| Text-only prior | Disabled; DTW reference branch is `continuous_coarse_smplx` |
| Word-prior prior | Online frozen SoftArranger; DTW reference branch is `adapter_context_smplx` |
| Retrieval bank | 113,524 entries and 41,260 lexicon keys |
| Default metric | Translated DTW-MPJPE |
| PA metric | Partwise PA-DTW-MPJPE, fitting and scoring the same keypoint subset |

Both modes exported 2,308 samples, produced 2,308 pairs in both DTW reports,
and recorded zero skipped prior sequences. The duration prediction is
text-only by design, so both modes have the same duration MAE: `1.506753`
seconds.

All DTW values below are `flow` means. Raw values are accumulated costs;
normalized values divide by the optimal warping-path length. Lower is better.

## 2. Default DTW by part

| Part | Text-only raw | Word-prior raw | Text-only normalized | Word-prior normalized | Prior change in normalized score | Better mode |
|---|---:|---:|---:|---:|---:|---|
| Body | 5.292571 | 5.330931 | 0.039751 | 0.039942 | +0.480% | Text-only |
| Left hand | 8.218462 | 8.234683 | 0.059309 | 0.059355 | +0.078% | Text-only |
| Right hand | 7.749796 | 7.763409 | 0.055738 | 0.055808 | +0.126% | Text-only |
| Whole body | 14.116677 | 14.176827 | 0.105664 | 0.105953 | +0.273% | Text-only |

## 3. Corrected partwise PA-DTW by part

| Part | Text-only raw | Word-prior raw | Text-only normalized | Word-prior normalized | Prior change in normalized score | Better mode |
|---|---:|---:|---:|---:|---:|---|
| Body | 4.721679 | 4.742303 | 0.035553 | 0.035659 | +0.300% | Text-only |
| Left hand | 1.989267 | 1.982881 | 0.014760 | 0.014733 | -0.182% | Word-prior |
| Right hand | 2.130778 | 2.122955 | 0.015883 | 0.015841 | -0.263% | Word-prior |
| Whole body | 9.737756 | 9.761592 | 0.072438 | 0.072558 | +0.166% | Text-only |

The PA reports record metric preset `t2m_partwise_pa_same_subset`. Thus the
body values use the corrected definition: the Procrustes transform is fitted
and scored on the same 12 upper-body keypoints. Word-prior has small localized
PA improvements for both hands, but text-only remains better for body and
whole-body PA-DTW.

## 4. Ground-truth-frame-count aligned diagnostics

These diagnostics query both modes at the ground-truth frame count. They
isolate pose and trajectory quality from predicted-duration error and should
not be mixed with the predicted-duration DTW scores above.

| Metric | Text-only | Word-prior | Prior relative change | Result |
|---|---:|---:|---:|---|
| Endpoint loss | 9.433403 | 9.434553 | +0.012% | Text-only better |
| Joint loss | 1.208475 | 1.210725 | +0.186% | Text-only better |
| Hand-relative loss | 0.097880 | 0.098119 | +0.244% | Text-only better |
| Path loss | 0.569160 | 0.556742 | -2.182% | Word-prior better |
| FK jerk loss | 0.145494 | 0.144859 | -0.437% | Word-prior better |
| Dense analytic FK jerk ratio | 5.714401 | 5.512140 | -3.539% | Descriptive only |
| Duration loss | 0.215569 | 0.215569 | 0.000% | Tied by design |
| Total loss | 10.383310 | 10.383282 | -0.00027% | Effectively tied |

The dense analytic FK jerk ratio is descriptive rather than a simple
lower-is-better error. The configured FK jerk loss is the appropriate aligned
loss comparison.

## 5. Selection result and conclusion

| Selection field | Value |
|---|---:|
| Text-only composite score | 10.711331 |
| Word-prior composite score | 10.708595 |
| Word-prior relative degradation | -0.0255% |
| Maximum allowed degradation | 2.000% |
| Feasible | `true` |

**Conclusion:** text-only is the stronger overall inference mode for this
checkpoint. It has lower default DTW for every part, lower body and whole-body
PA-DTW, and better aligned endpoint, joint, and hand-relative losses.
Word-prior gives small corrected PA-DTW improvements for left hand (`0.182%`)
and right hand (`0.263%`), plus better aligned path and FK-jerk losses. Its
aligned composite is also `0.0255%` better, but that difference and the total
loss difference are effectively ties and do not offset the broader DTW
regressions. Use word-prior only when the small hand-PA and path-smoothness
gains are more important than the body/whole-body and default-DTW results.

## 6. Result locations

```text
experiments/NIAF/continuous_trajectory_field/
  how2sign_signtrajfield_v2_full/evaluation/
    checkpoints/
      best_epoch0010_step0001200.pt
    best_epoch0010_step0001200_test_predicted_text_only/
      export_summary.json
      dtw_mpjpe_t2m_default_h2s_betas.json
      dtw_mpjpe_t2m_pa_h2s_betas.json
    best_epoch0010_step0001200_test_predicted_word_prior/
      export_summary.json
      dtw_mpjpe_t2m_default_h2s_betas.json
      dtw_mpjpe_t2m_pa_h2s_betas.json
    best_epoch0010_step0001200_test_aligned_both.json
```
