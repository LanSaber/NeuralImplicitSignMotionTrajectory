# SignTrajField Multi-Dataset Test Results

Evaluation date: 2026-07-24

Recorded: 2026-08-04

Updated: 2026-08-06

Status: predicted-duration and ground-truth-frame-count-sampling export,
default DTW, and PA-DTW complete for CSL-Daily, How2Sign, and PHOENIX;
ground-truth-frame-count aligned diagnostics complete for PHOENIX only

## 1. Scope and protocol

This document records the test results for immutable snapshots of the rolling
`last.pt` checkpoints from the three runs listed in
[`active_training_runs.md`](active_training_runs.md). The evaluation definitions
follow [`evaluation_guide.md`](evaluation_guide.md).

The DTW results use the deployable predicted-duration protocol:

- split: `test`;
- length mode: `predicted`;
- context and sample rates: 20 FPS;
- scaffold: frozen adapter constructed online;
- default score: translated DTW-MPJPE;
- PA score: per-frame, partwise Procrustes-aligned DTW-PA-MPJPE, fitting and
  scoring the same keypoint subset for each reported part; and
- parts: body, left hand, right hand, and whole body.

The model predicts its context duration and output length from text. Ground
truth retains its original length and is used only for saving and scoring; it
does not determine the generated sequence length. DTW aligns the resulting
unequal-length sequences. Lower is better for every DTW metric.

`dtw_mean` is the raw accumulated cost and depends on sequence length.
`ndtw_mean` divides the cost by the optimal warping-path length and is the more
appropriate value for comparisons across datasets with different sequence
length distributions.

The PA-DTW values were rerun on 2026-08-05 with metric preset
`t2m_partwise_pa_same_subset`. For each part, Procrustes fitting and scoring now
use the same keypoint subset. This corrects the body metric, which previously
fit all 144 SMPL-X joints before scoring the 12 upper-body keypoints. Hand and
whole-body values are unchanged because their earlier fitting and scoring sets
already matched. The superseded reports are retained on disk for
reproducibility but are not used in the tables below.

## 2. Checkpoint and dataset identity

| Dataset | Configuration | Epoch | Global step | Test examples |
|---|---|---:|---:|---:|
| CSL-Daily | `csl_daily_continuous_trajectory_stage2_part_experts_joint_full.yaml` | 12 | 864 | 1,176 |
| How2Sign | `how2sign_continuous_trajectory_stage2_part_experts_joint_full.yaml` | 28 | 3,360 | 2,308 |
| PHOENIX | `phoenix_continuous_trajectory_stage2_part_experts_full.yaml` | 25 | 2,775 | 642 |

The evaluated checkpoints are immutable copies made from each run's rolling
`last.pt` before submission:

| Dataset | Evaluation checkpoint | SHA-256 |
|---|---|---|
| CSL-Daily | `evaluation/checkpoints/last_epoch0012_step0000864.pt` | `78846351e5a242498ebf80dc5146233b4d99409a8c733e3d094af41bd0baeb3b` |
| How2Sign | `evaluation/checkpoints/last_epoch0028_step0003360.pt` | `830ab3087791421a5332515e449dd4f8e60a3c7eda303d526b4be92a8721f468` |
| PHOENIX | `evaluation/checkpoints/last_epoch0025_step0002775.pt` | `af178569a99d5f1c38264ae21ffecf03b74b189fc0af54dce26312eb8953c5c7` |

All three models condition on sentence text. CSL-Daily uses its training-only
word bank, How2Sign uses the external SignASL `all` dictionary recorded by its
configuration, and PHOENIX uses the training-only balanced word bank.

## 3. Headline predicted-duration results

The DTW columns below are `flow/wholebody` means.

| Dataset | Duration MAE (s) | Raw DTW | Path-normalized DTW | Raw PA-DTW | Path-normalized PA-DTW |
|---|---:|---:|---:|---:|---:|
| CSL-Daily | 1.395126 | 15.852010 | 0.138124 | 9.283869 | 0.078047 |
| How2Sign | 1.447260 | 14.160820 | 0.110410 | 9.760543 | 0.074856 |
| PHOENIX | 0.630654 | 10.859118 | 0.121546 | 7.022712 | 0.076638 |

## 4. Raw DTW and PA-DTW by part

| Dataset | Part | DTW `dtw_mean` | PA-DTW `dtw_mean` |
|---|---|---:|---:|
| CSL-Daily | Body | 5.415030 | 4.989084 |
| CSL-Daily | Left hand | 7.378177 | 1.357588 |
| CSL-Daily | Right hand | 6.947026 | 1.845596 |
| CSL-Daily | Whole body | 15.852010 | 9.283869 |
| How2Sign | Body | 5.337687 | 4.697010 |
| How2Sign | Left hand | 7.919316 | 1.890148 |
| How2Sign | Right hand | 7.643352 | 2.047009 |
| How2Sign | Whole body | 14.160820 | 9.760543 |
| PHOENIX | Body | 4.050117 | 3.572913 |
| PHOENIX | Left hand | 5.590703 | 1.085922 |
| PHOENIX | Right hand | 5.426758 | 1.203613 |
| PHOENIX | Whole body | 10.859118 | 7.022712 |

## 5. Path-normalized DTW and PA-DTW by part

| Dataset | Part | DTW `ndtw_mean` | PA-DTW `ndtw_mean` |
|---|---|---:|---:|
| CSL-Daily | Body | 0.048746 | 0.044718 |
| CSL-Daily | Left hand | 0.057806 | 0.011826 |
| CSL-Daily | Right hand | 0.058960 | 0.015249 |
| CSL-Daily | Whole body | 0.138124 | 0.078047 |
| How2Sign | Body | 0.041562 | 0.036510 |
| How2Sign | Left hand | 0.055621 | 0.014359 |
| How2Sign | Right hand | 0.055328 | 0.015656 |
| How2Sign | Whole body | 0.110410 | 0.074856 |
| PHOENIX | Body | 0.046063 | 0.040249 |
| PHOENIX | Left hand | 0.054654 | 0.011780 |
| PHOENIX | Right hand | 0.055256 | 0.013096 |
| PHOENIX | Whole body | 0.121546 | 0.076638 |

## 6. Ground-truth-frame-count sampling results

This diagnostic uses `length_mode: ground_truth_sampling`. It preserves the
text-predicted context and trajectory instance but queries that instance using
the ground-truth frame count. It therefore isolates output sampling length
without rebuilding the adapter context at an oracle length. The duration MAE
remains the prediction diagnostic reported in Section 3; it does not determine
the sampled output length in this protocol.

### 6.1 Headline results

The DTW columns below are `flow/wholebody` means.

| Dataset | Raw DTW | Path-normalized DTW | Raw PA-DTW | Path-normalized PA-DTW |
|---|---:|---:|---:|---:|
| CSL-Daily | 15.604137 | 0.146495 | 9.237933 | 0.082162 |
| How2Sign | 13.851626 | 0.115524 | 9.587200 | 0.078639 |
| PHOENIX | 10.486691 | 0.127629 | 6.800263 | 0.080363 |

### 6.2 Raw DTW and PA-DTW by part

| Dataset | Part | DTW `dtw_mean` | PA-DTW `dtw_mean` |
|---|---|---:|---:|
| CSL-Daily | Body | 5.272985 | 4.899727 |
| CSL-Daily | Left hand | 7.443641 | 1.352231 |
| CSL-Daily | Right hand | 7.011592 | 1.849433 |
| CSL-Daily | Whole body | 15.604137 | 9.237933 |
| How2Sign | Body | 5.256500 | 4.652888 |
| How2Sign | Left hand | 8.004199 | 1.856895 |
| How2Sign | Right hand | 7.727691 | 2.015047 |
| How2Sign | Whole body | 13.851626 | 9.587200 |
| PHOENIX | Body | 3.878456 | 3.441040 |
| PHOENIX | Left hand | 5.434549 | 1.057229 |
| PHOENIX | Right hand | 5.279243 | 1.177508 |
| PHOENIX | Whole body | 10.486691 | 6.800263 |

### 6.3 Path-normalized DTW and PA-DTW by part

| Dataset | Part | DTW `ndtw_mean` | PA-DTW `ndtw_mean` |
|---|---|---:|---:|
| CSL-Daily | Body | 0.051977 | 0.047907 |
| CSL-Daily | Left hand | 0.059458 | 0.012626 |
| CSL-Daily | Right hand | 0.061670 | 0.016107 |
| CSL-Daily | Whole body | 0.146495 | 0.082162 |
| How2Sign | Body | 0.043478 | 0.038328 |
| How2Sign | Left hand | 0.057141 | 0.015088 |
| How2Sign | Right hand | 0.056951 | 0.016503 |
| How2Sign | Whole body | 0.115524 | 0.078639 |
| PHOENIX | Body | 0.048406 | 0.042233 |
| PHOENIX | Left hand | 0.055790 | 0.012330 |
| PHOENIX | Right hand | 0.056541 | 0.013643 |
| PHOENIX | Whole body | 0.127629 | 0.080363 |

### 6.4 Whole-body comparison with predicted-length sampling

| Dataset | Mode | Predicted length | Ground-truth frame count |
|---|---|---:|---:|
| CSL-Daily | Raw DTW | 15.852010 | 15.604137 |
| CSL-Daily | Path-normalized DTW | 0.138124 | 0.146495 |
| CSL-Daily | Raw PA-DTW | 9.283869 | 9.237933 |
| CSL-Daily | Path-normalized PA-DTW | 0.078047 | 0.082162 |
| How2Sign | Raw DTW | 14.160820 | 13.851626 |
| How2Sign | Path-normalized DTW | 0.110410 | 0.115524 |
| How2Sign | Raw PA-DTW | 9.760543 | 9.587200 |
| How2Sign | Path-normalized PA-DTW | 0.074856 | 0.078639 |
| PHOENIX | Raw DTW | 10.859118 | 10.486691 |
| PHOENIX | Path-normalized DTW | 0.121546 | 0.127629 |
| PHOENIX | Raw PA-DTW | 7.022712 | 6.800263 |
| PHOENIX | Path-normalized PA-DTW | 0.076638 | 0.080363 |

Raw accumulated costs decrease under ground-truth-frame-count sampling for all
three datasets, while the path-normalized means increase. These columns should
therefore be reported with their normalization labels rather than interpreted
as a single interchangeable DTW score.

## 7. Ground-truth-frame-count aligned diagnostics

PHOENIX aligned diagnostics completed over all 642 test examples:

| Metric | SignTrajField prediction | Frozen scaffold |
|---|---:|---:|
| Endpoint loss | 10.726286 | 11.465789 |
| Joint loss | 1.455639 | 1.573091 |
| Hand-relative loss | 0.110850 | 0.113915 |
| Path loss | 0.400523 | 0.361444 |

Additional PHOENIX diagnostics:

- dense analytic FK jerk ratio: `0.689100`;
- duration diagnostic loss: `0.119612`;
- selection feasible: `true`;
- selection score: `-0.720099`; and
- selection constraint violation: `0.0`.

CSL-Daily and How2Sign aligned evaluations did not produce result JSONs. Both
failed before the first example because their test-split FK caches were absent:

```text
CSL-Daily:
  .../csl_daily_upper_smplx/meta/niaf_fk_rot6d_h2s_fixed_v1/test/

How2Sign:
  .../how2sign_soke_upper_smplx/meta/niaf_fk_rot6d_h2s_fixed_v1/test/
```

This limitation does not invalidate their predicted-duration DTW reports,
which export generated and ground-truth SMPL-X motions and perform FK during
DTW scoring.

## 8. Completion checks

| Length mode | Dataset | Exported samples | Default DTW pairs | PA-DTW pairs | Skipped adapter priors |
|---|---|---:|---:|---:|---:|
| Predicted | CSL-Daily | 1,176 | 1,176 | 1,176 | 0 |
| Predicted | How2Sign | 2,308 | 2,308 | 2,308 | 0 |
| Predicted | PHOENIX | 642 | 642 | 642 | 0 |
| Ground-truth sampling | CSL-Daily | 1,176 | 1,176 | 1,176 | 0 |
| Ground-truth sampling | How2Sign | 2,308 | 2,308 | 2,308 | 0 |
| Ground-truth sampling | PHOENIX | 642 | 642 | 642 | 0 |

For every dataset and both protocols, `export_summary.json` records the intended
length mode, `context_fps: 20.0`, and `sample_fps: [20.0]`.

## 9. Result locations

```text
experiments/NIAF/continuous_trajectory_field/
  csl_daily_continuous_trajectory_stage2_part_experts_joint_full/evaluation/
    last_epoch0012_step0000864_test_predicted/
      export_summary.json
      dtw_mpjpe_t2m_default_h2s_betas.json
      dtw_mpjpe_t2m_pa_partwise_same_subset_h2s_betas.json
    last_epoch0012_step0000864_test_ground_truth_sampling/
      export_summary.json
      dtw_mpjpe_t2m_default_h2s_betas.json
      dtw_mpjpe_t2m_pa_partwise_same_subset_h2s_betas.json

  how2sign_continuous_trajectory_stage2_part_experts_joint_full/evaluation/
    last_epoch0028_step0003360_test_predicted/
      export_summary.json
      dtw_mpjpe_t2m_default_h2s_betas.json
      dtw_mpjpe_t2m_pa_partwise_same_subset_h2s_betas.json
    last_epoch0028_step0003360_test_ground_truth_sampling/
      export_summary.json
      dtw_mpjpe_t2m_default_h2s_betas.json
      dtw_mpjpe_t2m_pa_partwise_same_subset_h2s_betas.json

  phoenix_continuous_trajectory_stage2_part_experts_full/evaluation/
    last_epoch0025_step0002775_test_aligned.json
    last_epoch0025_step0002775_test_predicted/
      export_summary.json
      dtw_mpjpe_t2m_default_h2s_betas.json
      dtw_mpjpe_t2m_pa_partwise_same_subset_h2s_betas.json
    last_epoch0025_step0002775_test_ground_truth_sampling/
      export_summary.json
      dtw_mpjpe_t2m_default_h2s_betas.json
      dtw_mpjpe_t2m_pa_partwise_same_subset_h2s_betas.json
```
