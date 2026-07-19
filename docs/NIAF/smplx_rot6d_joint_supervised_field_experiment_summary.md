# SMPL-X rot6D Joint-Supervised Field Experiment Summary

Date: 2026-07-05

This document summarizes the current oracle SMPL-X continuous-field experiments under `NIAF/oracle_smplx_field`. The goal was to test whether a continuous field in renderable SMPL-X pose space can fit and interpolate signing trajectories better than the earlier VAE-latent field experiment.

## 1. Experiment Setup

### Data

- Dataset: How2Sign upper-body SMPL-X data
- Data path: `/media/cvpr/haomian/data/SOKE_FLOW/how2sign_soke_upper_smplx`
- Split: `val`
- Initial pilot size: 3 sequences
- Sequence lengths in the main pilot: 48, 156, and 160 frames

### Representation

The field predicts compact SMPL-X pose in rot6D format:

```text
x(tau) = [41 rotations * 6D, 10 expression coefficients]
     = 256 dimensions
```

Predicted rot6D rotations are projected to valid rotation matrices, converted to SMPL-X axis-angle where needed, and passed through differentiable SMPL-X FK. Metrics are computed on rot6D features, rotation geodesic distance, and upper-body/hand joints.

### Models

The current implemented comparison includes:

- `linear_rot6d`: linear interpolation in compact rot6D feature space.
- `slerp`: per-joint rotation SLERP plus linear expression interpolation.
- `direct_siren`: direct SIREN field, `tau -> pose`.
- `residual_siren`: residual SIREN on top of an interpolation scaffold.
- `residual_siren_linear`: residual SIREN with linear rot6D scaffold.
- `residual_siren_slerp`: residual SIREN with SLERP scaffold.

### Metrics

Primary reported held-out metrics:

- `heldout_wholebody_jpe`: weighted upper-body + hands joint position error. Lower is better.
- `heldout_rot6d_l1`: rot6D feature L1 error. Lower is better.
- `heldout_geo_deg`: mean rotation geodesic error in degrees. Lower is better.
- `heldout_joint_r2`: joint-space R2. Higher is better.
- `heldout_plr`: path-length ratio. Closer to 1 is better; low values indicate shortcutting/over-smoothing.

## 2. Implemented Runs

### Stage-0: Basic Oracle Fitting

Output directory:

```text
experiments/NIAF/oracle_smplx_field/how2sign_stage0
```

Configuration:

```text
num_sequences: 3
time_modes: [uniform]
fit_modes: [fit_all, even_odd]
models: [linear_rot6d, slerp, direct_siren, residual_siren]
loss_schedules: [S1]
steps: 1000
```

Summary:

- `residual_siren` fit-all nearly perfectly reconstructs the observed trajectory.
- `direct_siren` is much worse than all interpolation baselines.
- In `even_odd`, `residual_siren` is effectively identical to its linear scaffold.
- `linear_rot6d` and `slerp` are very close for one-frame gaps.

Held-out means:

| Method | Fit mode | JPE | rot6D L1 | Geo deg | Joint R2 | PLR |
|---|---|---:|---:|---:|---:|---:|
| `linear_rot6d` | `even_odd` | 0.013755 | 0.034209 | 4.648 | 0.9639 | 0.8199 |
| `slerp` | `even_odd` | 0.013518 | 0.033528 | 4.628 | 0.9646 | 0.8146 |
| `direct_siren` | `even_odd` | 0.109208 | 0.328820 | 69.679 | -1.3502 | 3.5094 |
| `residual_siren` | `even_odd` | 0.013755 | 0.034210 | 4.648 | 0.9639 | 0.8199 |

Fit-all means:

| Method | Fit mode | JPE | rot6D L1 | Geo deg | Joint R2 | PLR |
|---|---|---:|---:|---:|---:|---:|
| `linear_rot6d` | `fit_all` | 0.000000 | 0.000000 | 0.082 | 1.0000 | 1.0000 |
| `slerp` | `fit_all` | 0.000000 | 0.000000 | 0.082 | 1.0000 | 1.0000 |
| `direct_siren` | `fit_all` | 0.033640 | 0.314421 | 67.516 | 0.6344 | 1.2226 |
| `residual_siren` | `fit_all` | 0.000000 | 0.000001 | 0.082 | 1.0000 | 1.0000 |

Interpretation:

The residual field has enough capacity to reproduce observed SMPL-X trajectories, but under anchor-only supervision it does not discover a better in-between path than the scaffold. Direct SIREN is not a good choice for this setting.

## 3. Stage-1 Sparse-Mask Diagnostic

Output directory:

```text
experiments/NIAF/oracle_smplx_field/how2sign_stage1_sparse_masks
```

Configuration:

```text
num_sequences: 3
time_modes: [uniform]
fit_modes: [stride_4, block_middle_25, keyframe_sparse_8]
models: [linear_rot6d, slerp, residual_siren_linear, residual_siren_slerp]
loss_schedules: [S1]
steps: 1000
```

Purpose:

The original `even_odd` setting has only one-frame gaps, so it is too easy for interpolation baselines. Stage-1 introduced harder sparse-anchor and contiguous-gap settings.

Held-out means:

| Method | Fit mode | JPE | rot6D L1 | Geo deg | Joint R2 | PLR |
|---|---|---:|---:|---:|---:|---:|
| `linear_rot6d` | `stride_4` | 0.025182 | 0.050645 | 6.943 | 0.8879 | 0.6850 |
| `slerp` | `stride_4` | 0.025069 | 0.050240 | 6.926 | 0.8887 | 0.6856 |
| `residual_siren_linear` | `stride_4` | 0.025181 | 0.050647 | 6.943 | 0.8879 | 0.6850 |
| `residual_siren_slerp` | `stride_4` | 0.025069 | 0.050242 | 6.926 | 0.8887 | 0.6856 |
| `linear_rot6d` | `block_middle_25` | 0.106853 | 0.125133 | 17.567 | -0.9318 | 0.3143 |
| `slerp` | `block_middle_25` | 0.107750 | 0.125635 | 17.482 | -0.9601 | 0.3110 |
| `residual_siren_linear` | `block_middle_25` | 0.106853 | 0.125136 | 17.567 | -0.9319 | 0.3143 |
| `residual_siren_slerp` | `block_middle_25` | 0.107750 | 0.125639 | 17.482 | -0.9601 | 0.3110 |
| `linear_rot6d` | `keyframe_sparse_8` | 0.066222 | 0.099135 | 13.787 | 0.4396 | 0.4059 |
| `slerp` | `keyframe_sparse_8` | 0.066484 | 0.099731 | 13.782 | 0.4432 | 0.4053 |
| `residual_siren_linear` | `keyframe_sparse_8` | 0.066222 | 0.099137 | 13.787 | 0.4396 | 0.4059 |
| `residual_siren_slerp` | `keyframe_sparse_8` | 0.066484 | 0.099735 | 13.782 | 0.4432 | 0.4053 |

Interpretation:

The harder masks expose real interpolation difficulty:

- `stride_4` remains mostly reasonable, but PLR around `0.68` indicates shortcutting.
- `block_middle_25` is very hard and often produces negative joint R2.
- `keyframe_sparse_8` is also hard and has low PLR around `0.40`.

However, residual SIREN still stays essentially equal to its scaffold. `residual_siren_linear` copies linear interpolation, and `residual_siren_slerp` copies SLERP.

## 4. Stage-2 Dense Physical Regularization

Output directory:

```text
experiments/NIAF/oracle_smplx_field/how2sign_stage2_dense_physics
```

Configuration:

```text
num_sequences: 3
time_modes: [uniform]
fit_modes: [stride_4, block_middle_25, keyframe_sparse_8]
models: [linear_rot6d, slerp, residual_siren_linear, residual_siren_slerp]
loss_schedules: [S4]
steps: 1000
dense_points: 96
```

Current code's `S4` schedule adds dense finite-difference physical regularization on predicted FK joints:

```text
lambda_dense_joint_acc: 0.1
lambda_dense_joint_jerk: 0.05
lambda_res: 1.0e-6
```

This regularization is unsupervised with respect to held-out frames: it uses only the predicted dense trajectory and does not look at withheld GT frames.

Held-out means:

| Method | Fit mode | JPE | rot6D L1 | Geo deg | Joint R2 | PLR |
|---|---|---:|---:|---:|---:|---:|
| `linear_rot6d` | `stride_4` | 0.025182 | 0.050645 | 6.943 | 0.8879 | 0.6850 |
| `residual_siren_linear` + dense | `stride_4` | 0.025175 | 0.050646 | 6.943 | 0.8879 | 0.6850 |
| `slerp` | `stride_4` | 0.025069 | 0.050240 | 6.926 | 0.8887 | 0.6856 |
| `residual_siren_slerp` + dense | `stride_4` | 0.025064 | 0.050242 | 6.926 | 0.8887 | 0.6856 |
| `linear_rot6d` | `block_middle_25` | 0.106853 | 0.125133 | 17.567 | -0.9318 | 0.3143 |
| `residual_siren_linear` + dense | `block_middle_25` | 0.106848 | 0.125135 | 17.567 | -0.9317 | 0.3143 |
| `slerp` | `block_middle_25` | 0.107750 | 0.125635 | 17.482 | -0.9601 | 0.3110 |
| `residual_siren_slerp` + dense | `block_middle_25` | 0.107745 | 0.125640 | 17.482 | -0.9599 | 0.3110 |
| `linear_rot6d` | `keyframe_sparse_8` | 0.066222 | 0.099135 | 13.787 | 0.4396 | 0.4059 |
| `residual_siren_linear` + dense | `keyframe_sparse_8` | 0.066219 | 0.099137 | 13.786 | 0.4396 | 0.4058 |
| `slerp` | `keyframe_sparse_8` | 0.066484 | 0.099731 | 13.782 | 0.4432 | 0.4053 |
| `residual_siren_slerp` + dense | `keyframe_sparse_8` | 0.066477 | 0.099735 | 13.782 | 0.4432 | 0.4053 |

Mean deltas against each scaffold:

| Residual model | Baseline | Fit mode | Delta JPE | Delta R2 | Delta PLR |
|---|---|---|---:|---:|---:|
| `residual_siren_linear` | `linear_rot6d` | `stride_4` | -0.000006 | +0.000033 | +0.000021 |
| `residual_siren_linear` | `linear_rot6d` | `block_middle_25` | -0.000006 | +0.000137 | -0.000008 |
| `residual_siren_linear` | `linear_rot6d` | `keyframe_sparse_8` | -0.000004 | +0.000024 | -0.000076 |
| `residual_siren_slerp` | `slerp` | `stride_4` | -0.000005 | +0.000028 | +0.000007 |
| `residual_siren_slerp` | `slerp` | `block_middle_25` | -0.000005 | +0.000168 | +0.000004 |
| `residual_siren_slerp` | `slerp` | `keyframe_sparse_8` | -0.000006 | +0.000009 | -0.000047 |

Interpretation:

Dense acceleration/jerk regularization is stable and active, but the improvement over the scaffold is extremely small. The residual field still mostly preserves the interpolation baseline rather than learning a meaningfully different trajectory.

## 5. Current Conclusions

1. **SMPL-X rot6D + FK pipeline works.** The experiment can load compact rot6D motion, convert it through differentiable SMPL-X FK, train fields, save NPZ outputs, and report pose/joint metrics.

2. **Residual SIREN has strong fit-all capacity.** With all frames observed, residual SIREN reconstructs the trajectory nearly exactly.

3. **Direct SIREN is not competitive.** It performs poorly even in fit-all and fails badly in held-out interpolation.

4. **Simple interpolation is a strong baseline.** Linear rot6D and SLERP are very similar for short gaps. SLERP is slightly better in some cases but not by a large margin.

5. **Sparse and span masks reveal shortcutting.** `stride_4`, `block_middle_25`, and `keyframe_sparse_8` show much worse PLR and R2 than `even_odd`, especially for high-motion or long middle gaps.

6. **Residual fields currently do not improve interpolation.** Under anchor-only S1 and dense acceleration/jerk S4, residual SIREN mostly reproduces its scaffold.

7. **Dense smoothness alone is not enough.** Acceleration/jerk regularization gives tiny metric changes but does not restore missing path length or semantic hand motion in hard gaps.

## 6. Recommended Next Experiments

Based on the original plan and current results, the most useful next experiments are:

### 6.1 Official 10-Sequence Stage-1 Pilot

Run the planned broader pilot:

```text
num_sequences: 10
time_modes: [uniform, joint_arclength, hand_arclength]
fit_modes: [fit_all, even_odd, stride_4]
models: [linear_rot6d, slerp, cubic, direct_siren, residual_siren]
loss_schedules: [S1, S2, S3]
steps: 2000
```

This will answer whether hand-aware time parameterization or supervised velocity loss provides real gains.

### 6.2 Add Supervised Velocity and Path-Length Losses

The dense smoothness regularizer did not recover missing path length. The next loss should compare predicted dynamics to GT dynamics on observed frames:

```text
velocity loss on train frames
hand-weighted path-length preservation
possibly local speed-profile matching
```

### 6.3 Add More Baselines

Still missing from the design plan:

```text
cubic interpolation
B-spline interpolation/control points
joint-only SIREN diagnostic
piecewise residual SIREN
```

### 6.4 Dense Query and Visualization

Still missing:

```text
2x and 4x dense query exports
decimation consistency metrics
hand trajectory plots
hand velocity curves
top-k high-error frame renderings
side-by-side decoded pose videos
```

## 7. Current Artifact Index

Main experiment outputs:

```text
experiments/NIAF/oracle_smplx_field/how2sign_stage0
experiments/NIAF/oracle_smplx_field/how2sign_stage1_sparse_masks
experiments/NIAF/oracle_smplx_field/how2sign_stage2_dense_physics
```

Important files in each output directory:

```text
grid.json
metrics_rows.csv
metrics_rows.jsonl
summary.json
npz/
```

Relevant configs:

```text
NIAF/oracle_smplx_field/configs/how2sign_rot6d_fk_pilot.yaml
NIAF/oracle_smplx_field/configs/how2sign_rot6d_fk_stage1.yaml
NIAF/oracle_smplx_field/configs/how2sign_rot6d_fk_stage2_dense_physics.yaml
```

Core implementation:

```text
NIAF/oracle_smplx_field/experiment.py
NIAF/oracle_smplx_field/losses.py
NIAF/oracle_smplx_field/models/siren.py
NIAF/oracle_smplx_field/models/baselines.py
NIAF/oracle_smplx_field/geometry/smplx_fk.py
```
