# Oracle Latent-Field Fitting Experiment Summary

Date: 2026-07-03

Refactor note: after the NIAF namespace refactor, the implementation lives under
`NIAF/oracle_latent_field`, commands should use `python -m
NIAF.oracle_latent_field...`, and new outputs should go under
`experiments/NIAF/oracle_latent_field`. The original small artifacts were copied
into the new experiment namespace so the paths in this note are usable.

This note summarizes the current oracle latent-field fitting experiment for
SoftArrangerFlow on How2Sign. It covers the implemented setting, the staged
grid result, the latent trajectory visualization, and the current recommendation.

The main takeaway is:

> A residual SIREN can almost perfectly fit the observed VAE latent tokens when
> all tokens are provided, but the same family does not yet interpolate held-out
> latent tokens reliably. The current result supports using residual SIREN +
> position-only latent loss as an oracle fitting baseline, but it does not yet
> support replacing the discrete latent trajectory with a pure global implicit
> field generator.

---

## 1. Experiment Goal

The experiment tests whether a ground-truth VAE latent trajectory
`z_1, ..., z_L` can be represented by a compact continuous field:

```text
f_theta(s) -> z_hat(s)
```

where `s` is a normalized temporal coordinate and `z_hat(s)` is a normalized
VAE latent token. This is an oracle experiment: it fits one field per ground
truth sequence using the ground-truth latent trajectory and ground-truth length.
It does not use text conditioning, retrieval, flow matching, or length prediction.

The intended diagnosis is:

- Can the field fit the latent tokens?
- Does decoded motion from the fitted latents stay close to the frozen VAE
  reconstruction?
- Does the field interpolate between latent tokens in a useful way?
- Do dense queries behave sensibly for short sequences?

---

## 2. Data And Model Setting

Dataset:

```text
/media/cvpr/haomian/data/SOKE_FLOW/how2sign_soke_upper_smplx
```

Split:

```text
val
```

Frozen VAE checkpoint:

```text
experiments/flow/VAE/how2sign_rot6d_vae_jerk_b128x4/checkpoints/best.pt
```

Latent statistics were copied from:

```text
experiments/flow/how2sign_latent_adapter_residual_signasl_all_b128x4_online_retry2/config.json
```

The experiment uses normalized VAE latents, and decoded metrics compare the
field-decoded SMPL-X feature sequence against the ground truth and against the
frozen VAE reconstruction. Pose arc length, when used, is computed from GT
SMPL-X joints. The staged run below used `uniform` and `latent_arclength`; the
full implementation also supports `pose_arclength`.

The VAE has a maximum of 400 motion frames, corresponding to 100 latent tokens.
Therefore dense query decoding is restricted to short sequences:

- `2x` dense query is decoded only when `2 * L <= 100`.
- `4x` dense query is decoded only when `4 * L <= 100`.

---

## 3. Implementation Artifacts

The experiment implementation lives under:

```text
NIAF/oracle_latent_field/
```

Key entrypoint:

```text
NIAF/oracle_latent_field/scripts/fit_oracle_field.py
```

How2Sign config:

```text
NIAF/oracle_latent_field/configs/how2sign_pilot.yaml
```

The full config grid includes:

| ID | Method |
|---|---|
| `A1` | SIREN, hidden 128, depth 3 |
| `A2` | SIREN, hidden 256, depth 3 |
| `A3` | SIREN, hidden 512, depth 3 |
| `A4` | SIREN, hidden 256, depth 4 |
| `A5` | Residual SIREN, hidden 256, depth 3 |
| `A6` | Fourier-feature MLP, hidden 256, depth 3, 16 frequencies |
| `A7` | ReLU MLP, hidden 256, depth 3 |
| `linear` | Linear interpolation baseline |
| `cubic` | Cubic spline baseline |
| `bspline` | B-spline baseline |
| `dct` | DCT baseline, 32 components |

`A5` is the anchored residual field:

```text
f(s) = LinearInterp(z_tokens; s) + r_theta(s)
```

This means it starts from an interpolation scaffold and learns a SIREN residual.
In `fit_all`, the anchor scaffold already passes through all tokens, so A5 is a
strong oracle-fit method. In `even_odd`, the scaffold sees only the training
tokens, so the held-out positions are a real interpolation test.

Loss presets:

| ID | Weights |
|---|---|
| `L1` | latent position only |
| `L2` | latent position + velocity |
| `L3` | latent position + velocity + acceleration |
| `L4` | latent position + velocity + jerk |
| `L5` | `L4` + decoded feature loss |

---

## 4. Staged Grid Run

The staged grid was intentionally smaller than the full pilot grid so that we
could get a first decision signal quickly.

Command:

```bash
conda run -n SOKE python -m NIAF.oracle_latent_field.scripts.fit_oracle_field \
  --config NIAF/oracle_latent_field/configs/how2sign_pilot.yaml \
  --out_dir experiments/NIAF/oracle_latent_field/how2sign_pilot_stage1 \
  --max_sequences 10 \
  --models A2,A5,linear,cubic,bspline,dct \
  --time_modes uniform,latent_arclength \
  --losses L1,L2,L4 \
  --fit_modes fit_all,even_odd \
  --steps 1000 \
  --warmup_steps 100 \
  --batch_points 64 \
  --no_save_npz
```

Artifacts:

```text
experiments/NIAF/oracle_latent_field/how2sign_pilot_stage1/
```

Run size:

| Quantity | Value |
|---|---:|
| Sequences | 10 |
| Grid configs per sequence | 40 |
| Total rows | 400 |
| Errors | 0 |
| Runtime | 2506.77 sec |

The 40 configs per sequence come from:

- neural: `A2,A5` x 2 time modes x 3 losses x 2 fit modes = 24;
- baselines: `linear,cubic,bspline,dct` x 2 time modes x 1 baseline loss x
  2 fit modes = 16.

Dense query coverage:

- `2x` decoded for 6 of 10 sequences;
- `4x` decoded for 4 of 10 sequences;
- the remaining dense cases were skipped because they exceeded the VAE latent
  limit.

A DCT-only rerun with ridge regularization was also executed after the first
stage exposed unstable DCT behavior:

```text
experiments/NIAF/oracle_latent_field/how2sign_pilot_stage1_dct_ridge/
```

---

## 5. Quantitative Results

### 5.1 Fit-all: observed tokens are easy to fit

In `fit_all`, every latent token is available to the field during fitting. This
tests representational capacity and whether the decoded output stays close to
the frozen VAE reconstruction.

Best nontrivial neural methods:

| Config | Mean latent rel L2 | Mean decoded feature MAE | VAE feature MAE | Field-VAE MAE gap |
|---|---:|---:|---:|---:|
| `A5/latent_arclength/L1/fit_all` | 0.00205 | 0.130794 | 0.130789 | 0.00000489 |
| `A5/uniform/L1/fit_all` | 0.00194 | 0.130796 | 0.130789 | 0.00000724 |
| `A2/uniform/L1/fit_all` | 0.00660 | 0.130921 | 0.130789 | 0.00013180 |
| `A2/latent_arclength/L1/fit_all` | 0.01222 | 0.131580 | 0.130789 | 0.00079067 |

Interpretation:

- `A5 + L1` nearly matches the frozen VAE reconstruction.
- Uniform and latent arc length are both viable for fit-all.
- The plain SIREN `A2` also fits, but worse than the residual SIREN.
- Velocity and jerk regularization did not help in this short 1000-step staged
  run. For example, `A5/uniform/L2/fit_all` had mean latent rel L2 `0.07283`,
  and `A5/uniform/L4/fit_all` had `0.11100`.

The exact interpolation baselines (`linear` and `cubic`) have zero token error
in `fit_all` by construction. They are useful sanity checks, but they do not
answer whether a compact neural field can learn the trajectory.

### 5.2 Even-odd: interpolation is still poor

In `even_odd`, the model trains on odd-indexed latent tokens and evaluates on
held-out even-indexed tokens. This is the important test for whether the latent
trajectory is smooth enough for a continuous field to recover missing tokens.

Best held-out results from the staged run:

| Config | Mean held-out rel L2 | Mean held-out R2 | Mean decoded Field-VAE MAE gap |
|---|---:|---:|---:|
| `linear/latent_arclength/baseline/even_odd` | 0.94711 | -0.04734 | 0.17252 |
| `A5/latent_arclength/L4/even_odd` | 0.97829 | -0.10051 | 0.18778 |
| `linear/uniform/baseline/even_odd` | 0.98150 | -0.08589 | 0.18232 |
| `A5/latent_arclength/L1/even_odd` | 0.98311 | -0.12401 | 0.18085 |
| `A5/uniform/L1/even_odd` | 1.02283 | -0.16448 | 0.19090 |

Interpretation:

- Held-out rel L2 is roughly `0.95` to `1.03` for the best methods.
- Held-out R2 is negative, so the interpolated latents are not explaining the
  missing-token variance well.
- Decoded feature MAE increases from the VAE baseline `0.13079` to roughly
  `0.303` to `0.319` in the best even-odd cases.
- Latent arc length helps slightly for interpolation, especially for linear and
  A5, but the improvement is not enough to claim compatibility.

This is the strongest evidence from the staged run: fitting observed tokens is
not the problem; interpolation through the current latent space is the problem.

### 5.3 DCT ridge rerun

After adding ridge regularization to the DCT baseline, DCT was numerically safer
but not competitive with A5 for fit-all or interpolation.

| Config | Mean latent rel L2 | Mean held-out rel L2 | Field-VAE MAE gap |
|---|---:|---:|---:|
| `dct/uniform/baseline/fit_all` | 0.29536 | 0.29536 | 0.09974 |
| `dct/latent_arclength/baseline/fit_all` | 0.24695 | 0.24695 | 0.07719 |
| `dct/uniform/baseline/even_odd` | 0.58818 all-token | 1.03821 held-out | 0.21858 |
| `dct/latent_arclength/baseline/even_odd` | 0.97697 all-token | 1.84771 held-out | 0.23916 |

Interpretation:

- Ridge prevents extreme blow-ups in the staged DCT rerun.
- DCT is still not a good choice for this latent trajectory family.
- Latent arc length improves DCT fit-all but remains fragile for even-odd.

---

## 6. Latent Trajectory Visualization

We exported `z_sent` and `z_fit` latent arrays for one representative short
sequence, then rendered them with:

```text
NIAF/oracle_latent_field/visualization/visualize_latent_trajectory.py
```

Sample:

| Field | Value |
|---|---|
| Sample ID | `48Hj_AyP3Fk_13-2-rgb_front` |
| Text | `She should be always looking for a cue from you.` |
| Source index | 448 |
| Frames | 48 |
| Latent tokens | 12 |

Visualization command:

```bash
conda run -n SOKE python -m NIAF.oracle_latent_field.visualization.visualize_latent_trajectory \
  --npz experiments/NIAF/oracle_latent_field/how2sign_trajectory_viz_export/latent_npz/0000_A5_uniform_L1_fit_all.npz \
        experiments/NIAF/oracle_latent_field/how2sign_trajectory_viz_export/latent_npz/0000_A5_uniform_L1_even_odd.npz \
  --latent_key z_sent z_fit \
  --out_dir experiments/NIAF/oracle_latent_field/how2sign_trajectory_viz \
  --annotate_every 1 \
  --fig_width 13.5 \
  --fig_height 5.8
```

### 6.1 A5 uniform L1 fit-all

In this case, `z_fit` almost exactly overlays `z_sent`.

![A5 uniform L1 fit-all PCA trajectory](../../experiments/NIAF/oracle_latent_field/how2sign_trajectory_viz/0000_A5_uniform_L1_fit_all_z_sent-z_fit_pca2d_trajectory.png)

PCA explained variance:

| PC | Ratio |
|---|---:|
| PC1 | 0.23428 |
| PC2 | 0.15268 |

Adjacent latent-step distances:

| Curve | Mean | Max | P95 |
|---|---:|---:|---:|
| `z_sent` | 18.27941 | 22.98899 | 22.23470 |
| `z_fit` | 18.27941 | 22.98899 | 22.23470 |

Interpretation:

- The fitted trajectory is visually and metrically identical to the GT latent
  trajectory for this sample.
- This confirms that A5 can represent the observed sequence when all tokens are
  available.

### 6.2 A5 uniform L1 even-odd

In this case, `z_fit` diverges from `z_sent` and tends to shortcut the trajectory.

![A5 uniform L1 even-odd PCA trajectory](../../experiments/NIAF/oracle_latent_field/how2sign_trajectory_viz/0000_A5_uniform_L1_even_odd_z_sent-z_fit_pca2d_trajectory.png)

PCA explained variance:

| PC | Ratio |
|---|---:|
| PC1 | 0.25585 |
| PC2 | 0.17829 |

Adjacent latent-step distances:

| Curve | Mean | Max | P95 |
|---|---:|---:|---:|
| `z_sent` | 18.27941 | 22.98899 | 22.23470 |
| `z_fit` | 10.92407 | 13.13378 | 13.07544 |

Interpretation:

- The even-odd field produces a smoother, lower-step trajectory than the ground
  truth.
- This looks like shortcutting or over-smoothing rather than faithful
  interpolation.
- The visualization agrees with the aggregate even-odd metrics.

---

## 7. Current Recommendation

For reporting the oracle fitting upper bound, use:

```text
A5 / L1 / fit_all
```

with either:

```text
uniform
```

or:

```text
latent_arclength
```

The uniform version has slightly lower latent rel L2 in the staged run, while
the latent-arc-length version has the smallest decoded field-to-VAE feature gap.
Both are close enough that the safer recommendation is to report both, with A5
as the main method.

For future model design, do not yet move to a pure global implicit latent field
that predicts the whole sentence trajectory from text. The current latent space
appears easy to memorize but hard to interpolate. Better next directions are:

- anchored residual fields with sparse anchors;
- a content path plus a separate time-warp or speed field;
- improved VAE latent regularization for interpolation smoothness;
- evaluating whether decoded or GT pose arc length gives better interpolation
  than uniform time or latent arc length;
- running the larger pilot after the staged-grid signal is incorporated.

In short:

```text
Use A5 + L1 as the oracle fit baseline.
Treat interpolation failure as the main obstacle.
Do not claim that the current VAE latent space is already continuous-field ready.
```

---

## 8. Remaining Work Before Main Claims

The staged result is useful but not a final paper-level result. Before making a
strong claim, run:

- a larger pilot: 50-100 sequences;
- the full grid, or a reduced grid guided by the staged result;
- `pose_arclength` time parameterization;
- `sparse_stride_4` in addition to `even_odd`;
- more visual samples, including high-motion and low-motion examples;
- optional DTW decoded metrics if runtime allows;
- a comparison against any future VAE trained with explicit latent smoothness or
  interpolation constraints.
