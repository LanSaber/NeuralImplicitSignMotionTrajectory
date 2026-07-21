# Continuous Trajectory Field Improvement Roadmap

Date: 2026-07-20

Status: Stage 0 selection repair partially completed; Stage 1 and Stage 2
implementation and five-sequence PHOENIX gates completed; pilot-500 and full
training not launched because neither architecture passed every continuation
gate

Completion marks in this document refer to work verified on 2026-07-20:

- `[x]` implemented and exercised;
- `[ ]` still open or deliberately blocked by an unmet gate.

## 1. Executive Recommendation

The next paper-oriented direction should be a **retrieval-guided, semantically
warped hierarchical neural implicit sign field**. Keep the current non-oracle
continuous trajectory contract, but improve the internal representation in three
ordered steps:

1. Fix checkpoint selection and establish reliable validation metrics.
2. Activate the local residual fields that are already implemented but disabled
   in the current full run.
3. Add a learned monotonic semantic phase warp and hand-specific local experts.

The key hypothesis is that the remaining error is primarily **local motion
content and internal timing**, especially hand articulation, rather than the
single predicted sequence length. Increasing model size or jerk regularization
alone is unlikely to address that problem.

## 2. Evidence From the Current Experiment

The measurements below use epoch 72 on all 642 PHOENIX test sequences with
predicted-length sampling.

### 2.1 Field versus frozen adapter scaffold

| Metric | Part | Field | Scaffold | Relative field gain |
|---|---|---:|---:|---:|
| Default nDTW-JPE | Body | 0.04596 | 0.04689 | +1.98% |
| Default nDTW-JPE | Left hand | 0.05478 | 0.05737 | +4.52% |
| Default nDTW-JPE | Right hand | 0.05491 | 0.05708 | +3.80% |
| Default nDTW-JPE | Whole body | 0.12182 | 0.12598 | +3.30% |
| PA-nDTW-JPE | Body | 0.05127 | 0.05236 | +2.07% |
| PA-nDTW-JPE | Left hand | 0.01144 | 0.01103 | **-3.80%** |
| PA-nDTW-JPE | Right hand | 0.01290 | 0.01222 | **-5.53%** |
| PA-nDTW-JPE | Whole body | 0.07393 | 0.07578 | +2.45% |

The field improves global motion, but after rigid alignment it degrades both
hands relative to the adapter. This points to lost local articulation rather
than a global placement problem.

### 2.2 Length prediction is useful but not the dominant bottleneck

The epoch-72 length predictor has:

- MAE: 12.60 frames, or 0.64 seconds at 20 FPS;
- signed bias: +2.63 frames;
- Pearson correlation: 0.849; and
- 43.8% of predictions within eight frames.

However, changing only the final query count to the exact GT frame count makes
path-normalized whole-body error worse:

| Protocol | Predicted count | GT count | Relative GT-count change |
|---|---:|---:|---:|
| Default whole-body nDTW-JPE | 0.12182 | 0.12667 | -3.98% |
| PA whole-body nDTW-JPE | 0.07393 | 0.07657 | -3.57% |

Raw DTW sums decrease because the path becomes shorter, but raw sums depend on
path length. The normalized result shows that exact sequence length alone does
not repair the trajectory. Duration should remain a learned component, but it
should not be the main architecture investment.

### 2.3 The implemented hierarchy is currently disabled

The full-run config sets:

```yaml
model:
  max_local_fields: 0
```

Training logs consequently report `active_local_fields: 0`. The current model is
therefore a global prior SIREN plus a global residual SIREN. The code for
retrieval-confidence-weighted local centers, widths, masks, and modulations
already exists and can be tested before adding a new architecture.

### 2.4 Training and checkpoint selection need correction

- Epoch 72 does not consistently beat epoch 22; body and hand nDTW generally
  stagnate or regress while training loss continues to decrease.
- Every validation checkpoint from epoch 5 through epoch 85 reports
  `selection_feasible: 0`.
- The selection rule requires `pred_loss_endpoint <= 1.8`, while observed
  validation endpoint losses are approximately 10.45 to 11.19. The threshold is
  on the wrong scale for this run.
- `best.pt` is epoch 30, but it is the lowest penalized score among infeasible
  checkpoints, not a checkpoint satisfying the intended constraints.

This must be fixed before interpreting a new full training run.

## 3. Proposed Paper Direction

### 3.1 Representation

Retain one finite trajectory instance per sentence:

$$
\Theta_i = H_\psi(y_i, a_i, r_i),
$$

where (y_i) is sentence text, (a_i) is the frozen adapter-predicted context,
and (r_i) is train-bank retrieval evidence. No GT motion, GT gloss, or GT
duration is used at inference.

Add a monotonic semantic phase function (u_i(t)):

$$
u_i(t) =
\frac{
  \int_0^{t/T_i} \left[\operatorname{softplus}(v_i(\xi)) + \epsilon\right] d\xi
}{
  \int_0^1 \left[\operatorname{softplus}(v_i(\xi)) + \epsilon\right] d\xi
},
\qquad u_i(t) \in [0,1].
$$

The final continuous motion is then:

$$
\hat{x}_i(t) =
\Phi_g(u_i(t);\Theta_i^g)
+ \sum_{m=1}^{M_i}
  \bar{w}_{i,m}(u_i(t))
  \Phi_{i,m}(u_i(t);\Theta_{i,m}^{\ell}).
$$

Rotational corrections must continue to be composed on (SO(3)). The phase
function, global field parameters, local centers and widths, part gates, and
duration all become serializable members of `TrajectoryInstance`. Querying a
trajectory at a new frame rate must not rerun text, retrieval, or the adapter.

### 3.2 Why semantic phase instead of only duration

The current time mapping is globally affine: one duration stretches the entire
sentence uniformly. Sign motion has nonuniform timing: lexical holds,
transitions, fingerspelling, and coarticulation need different local speeds. A
monotonic phase warp separates:

- **what motion occurs**, represented by the implicit trajectory fields; and
- **when it progresses**, represented by the phase-speed field.

The GT-count ablation changes only global sampling density. It cannot correct
internal timing, so it does not test this richer hypothesis.

### 3.3 Retrieval-guided local capacity

Use retrieval evidence to allocate local fields where the adapter is uncertain
or where its predicted hand motion changes rapidly. The global field should
model torso, coarse arm paths, and sentence-level continuity. Local fields
should recover lexical and hand detail.

For the paper, compare:

1. uniformly spaced local fields;
2. learned local fields without retrieval confidence; and
3. retrieval-confidence-adaptive local fields.

This ablation is necessary to show that adaptive capacity contributes beyond
simply adding parameters.

## 4. Staged Implementation Plan

### Stage 0: Repair validation and model selection

Do this before another expensive full run.

Implementation status:

- [x] Scaffold-relative constraints and normalized constraint scales are
  supported.
- [x] Every rejected validation checkpoint records its rejection reasons.
- [x] Feasible and infeasible checkpoints are separated as `best.pt` and
  `best_infeasible.pt`; pilot/full configs require at least one feasible
  checkpoint.
- [x] Validation checkpoints can be retained for later offline selection.
- [ ] Predicted-length nDTW/PA-nDTW is not yet executed inside the automatic
  training-time selector. It was run offline for the five-sequence gate below.
- [ ] Default whole-body nDTW and PA hand nDTW still need to replace the native-
  length endpoint proxies in automatic checkpoint selection.

1. Replace absolute selection thresholds with scaffold-relative constraints or
   calibrate thresholds from the validation distribution.
2. Select on predicted-length validation only. GT-count sampling remains an
   oracle diagnostic.
3. Include default whole-body nDTW, PA left/right-hand nDTW, hand path ratio, and
   FK jerk ratio in offline checkpoint selection.
4. Save a reason for every rejected checkpoint and assert that at least one
   checkpoint is feasible.
5. Keep epoch 22 and epoch 30 as frozen global-field baselines.

A suitable relative rule is:

$$
S = \Delta_{\mathrm{whole,nDTW}}
+ \lambda_h \frac{\Delta_{\mathrm{PA,left}} + \Delta_{\mathrm{PA,right}}}{2}
+ \lambda_j P_{\mathrm{jerk}}
+ \lambda_p P_{\mathrm{path}},
$$

where each (Delta) is measured against the same frozen adapter scaffold.

### Stage 1: Enable the existing local fields

Start with a pilot using the current implementation, without phase warping or
new part-specific decoders. A reasonable initial configuration is:

```yaml
model:
  max_local_fields: 16
  frames_per_local_field: 24
  minimum_local_width: 0.05
  maximum_local_width: 0.30
  quantile_temperature: 0.02

objective:
  lambda_local_modulation: 1.0e-5
  lambda_local_width: 0.0

analytic_dynamics:
  query_count: 64
```

Warm-start the shared global field from the chosen baseline, reinitialize the
previously frozen local heads, and use a new optimizer. Train the local branch
alone for a short warmup, then jointly fine-tune at a lower learning rate.

Required new diagnostics:

- [x] active local-field count;
- [x] local residual RMS by body/left hand/right hand/face;
- [x] local-window coverage and overlap;
- [x] center density versus retrieval confidence;
- [x] local contribution with the global residual disabled; and
- [x] PA-hand nDTW versus the global-only baseline.

Execution status:

- [x] Enable up to 16 retrieval-guided local fields with the documented widths,
  regularization, and 64-query dynamics grid.
- [x] Warm-start the epoch-30 global checkpoint, reset local heads, run a local-
  only warmup, and then jointly fine-tune.
- [x] Complete a 200-step five-sequence PHOENIX overfit and predicted-length DTW
  evaluation.
- [x] Complete a five-sequence path-weighted retry to test the observed path/
  articulation tradeoff.
- [ ] Run pilot-500. Blocked because no five-sequence checkpoint was feasible
  and PA hand metrics regressed against the scaffold.
- [ ] Run the full balanced PHOENIX split. Blocked by the same gate.

### Stage 2: Add part-specific local experts

The current local fields share one residual SIREN across every articulator and
use one sequence-level gate per part. Replace this with structured experts:

- a low-frequency body/arm field;
- a higher-frequency left-hand field;
- a higher-frequency right-hand field; and
- an optional face/expression field.

Predict local time-dependent gates rather than only one global gate per
sequence. Keep wrist-relative hand FK loss, fingertip velocity/path losses, and
rotation-geodesic losses. This should directly target the PA-hand regression.

Execution status:

- [x] Replace the shared local residual decoder with independent body, left-hand,
  right-hand, and face experts.
- [x] Use lower body frequency and higher hand frequencies while retaining the
  existing global branch and SO(3) residual composition.
- [x] Predict and serialize time-dependent per-local-field part gates.
- [x] Preserve backward-compatible Stage 1 checkpoint loading and reinitialize
  only the new local branch when warm-starting.
- [x] Complete a 200-step five-sequence PHOENIX run and predicted-length default
  and PA-nDTW evaluation.
- [ ] Run pilot-500. Blocked because the path proxy still failed and PA hand
  nDTW remained worse than the paired scaffold despite improving over Stage 1.
- [ ] Run the full balanced PHOENIX split. Blocked by the same gate.

### Stage 3: Add the monotonic phase field

Predict nonnegative phase speed from text and adapter context, integrate it, and
normalize its endpoints. Recommended training sequence:

1. Initialize close to the identity warp.
2. Derive train-only pseudo phase targets from adapter-to-GT alignment, or use a
   differentiable Soft-DTW objective on a reduced joint set.
3. Regularize phase speed and curvature so the warp remains smooth and cannot
   hide pose errors through extreme compression.
4. Fine-tune trajectory and phase fields jointly.

The inference path must use only the predicted phase parameters in
`TrajectoryInstance`.

### Stage 4: Improve continuous-time supervision

The current analytic dynamics loss uses 32 queries regardless of sequence
duration. Use a duration-aware query budget, for example:

$$
K_i = \operatorname{clip}(\lceil 8T_i \rceil, 32, 128).
$$

Also add:

- multi-grid consistency at independently sampled time sets;
- uncertainty- and hand-speed-biased training queries;
- velocity and acceleration matching in FK space;
- a band-limited high-frequency hand residual loss; and
- jerk as an upper-bound regularizer, not the sole detail objective.

Do not simply increase jerk weight. A trajectory can achieve low jerk by losing
lexical peaks and hand path length. Smoothness must be constrained jointly with
detail and path preservation.

## 5. Experiment Matrix

| ID | Global field | Local fields | Part experts | Phase warp | Purpose | Status |
|---|---|---|---|---|---|---|
| B0 | Adapter only | No | No | No | Frozen scaffold baseline | Available |
| B1 | Yes | No | No | No | Current continuous global baseline | Completed |
| A1 | Yes | Uniform | No | No | Added-capacity control | Not run |
| A2 | Yes | Learned | No | No | Learned localization control | Not run |
| A3 | Yes | Retrieval-guided | No | No | Confidence allocation test | Five-sequence gate completed; no-go |
| A4 | Yes | Retrieval-guided | Yes | No | Hand-detail test | Five-sequence gate completed; improves A3, no-go |
| A5 | Yes | Retrieval-guided | Yes | Yes | Proposed full method | Not implemented |

Run each stage first on the five-sequence overfit set, then pilot-500, then the
full balanced PHOENIX training split. Do not launch the full matrix until A1
demonstrates a nonzero useful local contribution.

## 6. Acceptance Criteria

Use predicted-length test evaluation as the primary protocol.

- [ ] Default whole-body nDTW must improve over the current 0.12182 full-test
  baseline. Not evaluated on the full test split because the earlier gate failed.
- [ ] PA left-hand nDTW must be at most the paired scaffold value. Stage 2
  improved 7.59% over Stage 1 but remained 42.86% worse than its paired
  scaffold.
- [ ] PA right-hand nDTW must be at most the paired scaffold value. Stage 2
  improved 5.23% over Stage 1 but remained 48.30% worse than its paired
  scaffold.
- [ ] Both hand sequence-win fractions should exceed 0.50. Not evaluated after
  the aggregate PA-hand gate failed.
- [ ] FK jerk ratio should remain below GT, preferably 0.70 to 0.90, while hand
  path ratio remains approximately 0.90 to 1.10. The selected Stage 2 checkpoint
  had an in-bound 0.942 jerk ratio but failed the path constraint; later
  checkpoints improved hand pose while exceeding GT jerk.
- [ ] Predictions sampled at 20, 40, and 80 FPS from the same trajectory instance
  must agree after resampling within a documented tolerance. The query contract
  is unit-tested, but the Stage 1 and Stage 2 checkpoints were exported only at
  20 FPS.
- [x] No GT pose, GT duration, or GT gloss enters predicted-length inference.
- [ ] At least one checkpoint must satisfy the corrected validation constraints.
  No checkpoint from the Stage 1 or Stage 2 five-sequence runs was feasible.

For a paper, report default and PA DTW-JPE/nDTW by body and hands, length error,
FK velocity/acceleration/jerk, hand path ratio, resolution consistency, semantic
back-translation or recognition metrics when available, confidence intervals,
and qualitative failure cases.

## 7. What Not to Prioritize Next

- Do not make GT frame count the main solution; current normalized results do not
  support it.
- Do not increase only SIREN width or depth before testing the disabled local
  hierarchy.
- Do not strengthen jerk regularization without a hand-detail and path constraint.
- Do not replace the continuous field with interpolated frame-indexed local
  codes; that would weaken the central trajectory representation claim.
- Do not introduce GT anchors or GT-derived support during deployable inference.
- Do not compare methods using raw DTW sums alone when sequence lengths differ.

## 8. Staged Experiment Record

The first concrete continuation test was the **Stage 1 local-field gate**, not
another global full run:

1. [x] Correct the scale of the selection constraints and record rejection
   reasons. Predicted-length offline metrics still need integration into the
   automatic selector.
2. [x] Enable up to 16 local fields with a 64-query dynamics grid.
3. [x] Warm-start shared weights from the selected global checkpoint and reset
   local heads.
4. [x] Run the five-sequence overfit.
5. [ ] Run pilot-500. It was intentionally not launched after the five-sequence
   gate failed.
6. [x] Apply the continuation gate. Local residuals were nontrivial and useful
   relative to the global-only ablation, but PA hands regressed against the
   scaffold and path/jerk constraints could not be satisfied together.

This experiment is comparatively cheap, exercises code already present, and
directly tests whether hierarchical continuous capacity can recover the hand
detail missing from the current global model.

### 8.1 Stage 1 execution record

The canonical run used five PHOENIX training sequences, 16 retrieval-guided
local fields, a 64-query dynamics grid, the epoch-30 global checkpoint, five
local-only warmup steps, and 200 total optimizer steps. Slurm job `140339`
completed successfully.

| Native-length proxy | Selected epoch 50 | Scaffold | Global-only | Gate |
|---|---:|---:|---:|---|
| Endpoint loss | 4.75907 | 5.66861 | 4.83857 | Pass versus scaffold |
| Hand-relative loss | 0.06102 | 0.05508 | 0.06138 | Fail versus scaffold |
| Hand path loss | 0.48848 | 0.30831 | 0.49287 | Fail |
| FK jerk ratio | 0.90034 | - | - | Pass configured 0.60-0.95 bound |
| Local residual RMS | 0.00673 | - | 0 | Nonzero |

The final epoch reduced endpoint loss to 3.32564 and hand-relative loss to
0.03838, but hand path loss remained 0.47412 and jerk rose to 1.16981. The
selector therefore retained epoch 50 as the lowest-penalty checkpoint, but it
was still infeasible. The canonical run predates the feasible-only filename
repair, so its checkpoint is named `best.pt`; new runs store an equivalent
failure as `best_infeasible.pt`.

Predicted-length offline evaluation on the same five sequences produced:

| Metric | Part | Full local field | Global-only | Scaffold | Full vs scaffold |
|---|---|---:|---:|---:|---:|
| Default nDTW | Body | 0.02674 | 0.02697 | 0.02807 | +4.72% |
| Default nDTW | Left hand | 0.03114 | 0.03146 | 0.03668 | +15.11% |
| Default nDTW | Right hand | 0.04214 | 0.04233 | 0.04044 | -4.21% |
| Default nDTW | Whole body | 0.05266 | 0.05344 | 0.07444 | +29.25% |
| PA-nDTW | Body | 0.02855 | 0.02884 | 0.03142 | +9.14% |
| PA-nDTW | Left hand | 0.00999 | 0.01011 | 0.00646 | **-54.61%** |
| PA-nDTW | Right hand | 0.01181 | 0.01192 | 0.00755 | **-56.48%** |
| PA-nDTW | Whole body | 0.03759 | 0.03796 | 0.04774 | +21.25% |

The local branch is measurably useful relative to the global-only ablation:
it improves every reported default and PA part by approximately 0.45% to 1.45%.
However, the complete field still loses substantial aligned hand articulation
relative to the adapter scaffold. This fails the main Stage 1 hypothesis.

A path-weighted retry (Slurm job `140340`) verified that loss reweighting alone
does not resolve the conflict. Its best proxy checkpoint reduced path loss to
0.24041, below the scaffold's 0.30831, but hand-relative loss worsened to 0.08751
and jerk rose to 1.28505. At epoch 200, path loss was 0.01957 while hand-relative
loss remained 0.08859 and jerk reached 1.59906. No checkpoint was feasible.

Decision: do not launch pilot-500 or the full split from this Stage 1 design.
The next architecture work therefore moved to the hand-specific experts in
Stage 2 before spending compute on larger-scale training.

### 8.2 Stage 2 execution record

Stage 2 replaced the shared local decoder with separate body, left-hand,
right-hand, and face experts and added time-dependent part gates. It otherwise
kept the Stage 1 objective, schedule, retrieval-guided local centers, and global
warm start so that the five-sequence comparison isolated the architecture
change. Slurm job `140364` ran 200 optimizer steps; jobs `140365` and `140366`
exported and evaluated the selected predicted-length trajectories.

The lowest-penalty checkpoint was epoch 70:

| Native-length proxy | Selected epoch 70 | Scaffold | Gate |
|---|---:|---:|---|
| Endpoint loss | 4.26868 | 5.66861 | Pass versus scaffold |
| Hand-relative loss | 0.05426 | 0.05508 | Pass versus scaffold |
| Hand path loss | 0.49281 | 0.30831 | Fail; relative excess 0.18450 exceeds 0.10 |
| FK jerk ratio | 0.94231 | - | Pass configured 0.60-0.95 bound |
| Local residual RMS | 0.00463 | - | Nonzero |

At this checkpoint, local residual RMS was 0.00330 for body, 0.00409 for the
left hand, 0.00433 for the right hand, and 0.00835 for face. Mean part gates
were respectively 0.51461, 0.50946, 0.50947, and 0.53362, confirming that every
expert was active. The local experts also improved their paired global-only
ablation by 0.26% to 0.75%, depending on part and alignment protocol.

Predicted-length evaluation on the same five sequences shows a consistent
Stage 2 improvement:

| Metric | Part | Stage 1 | Stage 2 | Stage 2 vs Stage 1 | Scaffold | Stage 2 vs scaffold |
|---|---|---:|---:|---:|---:|---:|
| Default nDTW | Body | 0.02674 | 0.02461 | +7.98% | 0.02807 | +12.32% |
| Default nDTW | Left hand | 0.03114 | 0.02942 | +5.53% | 0.03668 | +19.81% |
| Default nDTW | Right hand | 0.04214 | 0.03987 | +5.38% | 0.04044 | +1.40% |
| Default nDTW | Whole body | 0.05266 | 0.04854 | +7.83% | 0.07444 | +34.79% |
| PA-nDTW | Body | 0.02855 | 0.02620 | +8.22% | 0.03142 | +16.61% |
| PA-nDTW | Left hand | 0.00999 | 0.00923 | +7.59% | 0.00646 | **-42.86%** |
| PA-nDTW | Right hand | 0.01181 | 0.01119 | +5.23% | 0.00755 | **-48.30%** |
| PA-nDTW | Whole body | 0.03759 | 0.03453 | +8.16% | 0.04774 | +27.67% |

Decision: Stage 2 does improve Stage 1 on every default and PA component and
repairs the Stage 1 default right-hand regression relative to the scaffold.
However, it does not yet recover scaffold-level aligned hand articulation, and
the selected native-length proxy still violates the hand-path constraint. Do
not launch pilot-500 or the full split from this checkpoint. The next experiment
should directly address path preservation before adding Stage 3 phase warping
or scaling the training set.

## 9. Result Sources

The numeric observations in this document come from:

```text
experiments/NIAF/continuous_trajectory_field/
  phoenix_continuous_trajectory_full_smooth_q32_jreg1e2_w10_b32x8/
    metrics.jsonl
    evaluation/predicted_length_dtw_epoch0072_test642/
      dtw_comparison_summary.json
      dtw_pa_comparison_summary.json
      epoch0022_vs_epoch0072_comparison.json
      predicted_length_audit.json
    evaluation/ground_truth_sampling_dtw_epoch0072_test642/
      gt_count_vs_predicted_length_dtw_comparison.json
      gt_count_vs_predicted_length_pa_comparison.json

  phoenix_continuous_trajectory_stage1_local16_overfit5_q64/
    metrics.jsonl
    checkpoints/best.pt
    checkpoints/last.pt
    evaluation/predicted_length_first5/
      export_summary.json
      dtw_full_vs_scaffold.json
      dtw_pa_full_vs_scaffold.json
      dtw_full_vs_global.json
      dtw_pa_full_vs_global.json

  phoenix_continuous_trajectory_stage1_local16_overfit5_path_tune/
    metrics.jsonl
    selection_summary.json
    checkpoints/best_infeasible.pt
    checkpoints/last.pt

  phoenix_continuous_trajectory_stage2_part_experts_overfit5/
    metrics.jsonl
    selection_summary.json
    checkpoints/epoch0070.pt
    checkpoints/best_infeasible.pt
    checkpoints/last.pt
    evaluation/predicted_length_first5_epoch0070/
      export_summary.json
      dtw_full_vs_scaffold.json
      dtw_pa_full_vs_scaffold.json
      dtw_full_vs_global.json
      dtw_pa_full_vs_global.json
```
