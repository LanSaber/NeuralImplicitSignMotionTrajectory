# Meta-Implicit MAML-Style Continuous Sign Field Design

Date: 2026-07-10

This document summarizes the MAML-style meta-implicit trajectory path added under `NIAF/continuous_sign_field`. It is a parallel path beside the earlier residual-flow pipeline, designed to address the observed failure mode where the stride-32 GT-anchor model degenerates to smooth scaffold interpolation and loses motion details.

---

## 1. Motivation

The previous continuous sign field experiment used a global residual-flow Transformer:

```text
text + scaffold + time
  -> residual flow model
  -> predicted residual trajectory
  -> scaffold + residual
```

For the GT-anchor stride-32 experiment, the scaffold already passes through sparse ground-truth anchors. The model then has to recover high-frequency intra-anchor details from weak sequence-level conditioning. In practice, the learned residual becomes very small, so the generated sequence visually collapses toward the smooth Slerp scaffold.

The meta-implicit path changes the formulation. Instead of one global model directly predicting an averaged residual, each sequence gets a small adapted latent code. The shared decoder represents the trajectory residual as an implicit function of normalized time, scaffold state, text context, and the sequence code.

---

## 2. Core Formulation

For sequence \(i\), the generated pose trajectory is:

$$
\hat{x}_i(\tau) = s_i(\tau) + D_\theta(\tau, s_i(\tau), z_i, c_i),
$$

where:

- \(s_i(\tau)\) is the scaffold trajectory, such as stride-32 GT-anchor Slerp;
- \(D_\theta\) is a shared meta-implicit residual decoder;
- \(z_i\) is a sequence-specific latent code;
- \(c_i\) is text/scaffold context;
- \(\tau \in [0,1]\) is normalized sequence time.

The initial code is predicted by a context network:

$$
z_{i,0} = G_\phi(c_i, s_i).
$$

Then a small inner loop adapts only \(z_i\):

$$
z_{i,k+1} =
z_{i,k}
-
\alpha \nabla_{z_i}
\mathcal{L}_{support}
\left(
s_i + D_\theta(\tau, s_i, z_{i,k}, c_i),
x_i
\right).
$$

The default implementation is first-order and latent-code-only. Model weights \(\theta,\phi\) are updated only by the outer optimizer.

---

## 3. Pipeline

### 3.1 Training Flow

```text
PHOENIX SMPL-X rot6D sequence
  -> existing ContinuousSignDataset
  -> existing ScaffoldProvider
  -> scaffold S(t)
  -> ContextToCode predicts z0
  -> support/query split
  -> inner loop adapts z on support frames
  -> decoder predicts residual on query frames
  -> outer loss updates shared model parameters
```

The default diagnostic experiment is oracle sparse support:

- scaffold source: `gt_anchors`
- scaffold interpolation: `slerp`
- anchor stride: `32`
- support mode: anchors plus every 8th valid frame
- query frames: all valid non-support frames

This is intentional. Anchors alone are not enough, because the scaffold already matches anchor frames and the residual target at anchors is near zero. Extra sparse support frames provide supervision for missing detail while still testing whether the model can reconstruct unseen intra-support frames.

### 3.2 Inference / Export Flow

```text
sample
  -> scaffold S(t)
  -> z0 from ContextToCode
  -> meta_prior = S(t) + D(t, S(t), z0)
  -> optional support adaptation
  -> meta_adapted = S(t) + D(t, S(t), z_adapted)
  -> save renderable NPZ files
```

The combined exported `sample_XXXX.npz` contains:

- `smplx`: adapted output, for compatibility with existing visualizers;
- `coarse_smplx`: scaffold;
- `meta_prior_smplx`: output before inner-loop adaptation;
- `meta_adapted_smplx`: output after inner-loop adaptation;
- `support_mask`;
- `query_mask`.

---

## 4. Implementation Components

Main model and adaptation code:

- `NIAF/continuous_sign_field/models/meta_implicit.py`
  - `ContextToCode`
  - `MetaImplicitResidualField`
  - SIREN/Fourier implicit residual decoder
- `NIAF/continuous_sign_field/meta_learning.py`
  - support/query mask construction
  - first-order latent-code adaptation
  - residual, velocity, and acceleration detail losses

Training and export:

- `NIAF/continuous_sign_field/scripts/train_meta_implicit_field.py`
- `NIAF/continuous_sign_field/scripts/export_meta_implicit_samples.py`

Visualization:

- `NIAF/continuous_sign_field/visualization/visualize_meta_implicit_compare.py`

Configs:

- `NIAF/continuous_sign_field/configs/phoenix_meta_implicit_gt_anchor_stride32_overfit5.yaml`
- `NIAF/continuous_sign_field/configs/phoenix_meta_implicit_gt_anchor_stride32.yaml`

Slurm launcher:

- `scripts/NIAF/train_meta_implicit_field_sbatch.sh`

---

## 5. Loss Design

The inner-loop support loss adapts only \(z_i\):

$$
\mathcal{L}_{support}
=
\lambda_{res}^{in}
\left\|
\hat{r}(\tau_s) - r^*(\tau_s)
\right\|
+
\lambda_{pose}^{in}
\left\|
\hat{x}(\tau_s) - x(\tau_s)
\right\|.
$$

The outer loss is computed mainly on query frames:

$$
\mathcal{L}_{outer}
=
\lambda_{endpoint}
\mathcal{L}_{endpoint}
+
\lambda_{res}
\mathcal{L}_{residual}
+
\lambda_{vel}
\mathcal{L}_{\Delta residual}
+
\lambda_{acc}
\mathcal{L}_{\Delta^2 residual}
+
\lambda_{support}
\mathcal{L}_{support-consistency}.
$$

Endpoint losses reuse the existing continuous sign field components:

- rot6D pose loss;
- rotation geodesic loss;
- SMPL-X FK joint loss;
- hand-weighted FK loss;
- expression loss.

The extra residual velocity and acceleration terms are meant to fight the interpolation-collapse failure mode by directly supervising missing detail in the residual trajectory.

---

## 6. Recommended Experiment Order

### Stage A: 5-Sequence Overfit

Use:

```bash
python -m NIAF.continuous_sign_field.scripts.train_meta_implicit_field \
  --config NIAF/continuous_sign_field/configs/phoenix_meta_implicit_gt_anchor_stride32_overfit5.yaml
```

Success criteria:

- adapted output beats scaffold on query-frame losses;
- adapted residual RMS is non-trivial;
- hand path ratio moves closer to GT;
- four-panel visualization shows details beyond smooth interpolation.

### Stage B: Pilot Training

Use:

```bash
python -m NIAF.continuous_sign_field.scripts.train_meta_implicit_field \
  --config NIAF/continuous_sign_field/configs/phoenix_meta_implicit_gt_anchor_stride32.yaml
```

Compare:

- scaffold;
- meta prior;
- meta adapted.

Important ablations:

- `support_mode: anchors_only`
- `support_stride: 16`
- `support_stride: 8`

### Stage C: Non-Oracle Extension

After the oracle diagnostic works, switch scaffold source to adapter/text-predicted scaffold:

```yaml
scaffold:
  source: adapter_pred
```

For deployable generation, use `meta_prior` as the non-oracle output. Inner-loop adaptation should be treated as an oracle or pseudo-support ablation unless a non-GT support source is defined.

---

## 7. Visualization

Export samples:

```bash
python -m NIAF.continuous_sign_field.scripts.export_meta_implicit_samples \
  --config NIAF/continuous_sign_field/configs/phoenix_meta_implicit_gt_anchor_stride32.yaml \
  --checkpoint experiments/NIAF/continuous_sign_field/phoenix_meta_implicit_gt_anchor_stride32/checkpoints/last.pt \
  --split train \
  --num_samples 5 \
  --seed 123 \
  --out_dir visualize/NIAF/continuous_sign_field/phoenix_meta_implicit_train5/npz
```

Render one four-panel comparison:

```bash
python -m NIAF.continuous_sign_field.visualization.visualize_meta_implicit_compare \
  --gt visualize/NIAF/continuous_sign_field/phoenix_meta_implicit_train5/npz/gt_0000.npz \
  --sample visualize/NIAF/continuous_sign_field/phoenix_meta_implicit_train5/npz/sample_0000.npz \
  --out_dir visualize/NIAF/continuous_sign_field/phoenix_meta_implicit_train5/render \
  --fps 20 \
  --width 512 \
  --height 512 \
  --software_face_stride 1 \
  --device cpu \
  --view_transform none \
  --upper_body_only
```

The video layout is:

```text
Ground truth | Scaffold | Meta prior | Meta adapted
```

---

## 8. Current Verification

The implementation was smoke-tested with:

- manual unit checks for model shape, support/query masks, z-only adaptation, and export keys;
- `compileall` on `NIAF/continuous_sign_field`;
- one-batch CPU training;
- one-batch CPU validation and checkpoint saving;
- one-sample export;
- two-frame four-panel render.

The SOKE environment did not include `pytest`, so the tests in `tests/test_niaf_meta_implicit.py` were run manually.

---

## 9. Notes and Caveats

1. The current implementation is first-order latent-code adaptation, closer to CAVIA than full MAML.
2. The model is expected to need overfit tuning before full training. A residual RMS near zero means it is still collapsing to the scaffold.
3. `anchors_only` is expected to be weak because stride-32 Slerp already matches anchor poses.
4. The Slurm script may need to be invoked with `bash scripts/NIAF/train_meta_implicit_field_sbatch.sh` if executable permissions are unavailable on the mounted filesystem.
