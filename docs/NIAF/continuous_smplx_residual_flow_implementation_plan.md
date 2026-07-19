# Implementation Plan: Lexically Anchored Continuous SMPL-X Residual Flow for Text-to-3D Sign Production

Date: 2026-07-05

This document gives a concrete implementation plan for a new text-to-3D sign language production pipeline. The goal is to move beyond the WACV SoftArrangerFlow formulation, which performs residual flow matching in a frozen VAE latent sequence, and implement a physically supervised continuous SMPL-X trajectory generator in rot6D pose space.

The recommended first full system is:

```text
Text + lexical retrieval
  -> lexical memory / SoftArranger conditioning
  -> duration or tempo prediction
  -> coarse SMPL-X scaffold
  -> residual conditional flow matching in SMPL-X rot6D space
  -> rot6D projection + SMPL-X FK supervision
  -> renderable SMPL-X signing motion
```

The core research question should be:

> Can we generate sign language as a continuous, physically meaningful SMPL-X pose field, instead of a fixed-length VAE latent-token sequence?

---

## 1. Design Goals

### 1.1 What this pipeline should achieve

1. Generate renderable upper-body SMPL-X signing motion directly in rot6D pose space.
2. Use lexical retrieval as semantic grounding, but avoid making the SoftArranger itself the only new contribution.
3. Learn a residual trajectory distribution around a coarse scaffold using conditional flow matching.
4. Apply physical supervision through SMPL-X forward kinematics, hand-weighted 3D joint losses, rotation geodesic losses, velocity losses, and path-length preservation.
5. Support arbitrary temporal querying in the future by treating trajectory time as a continuous coordinate.
6. Keep a duration or tempo module instead of claiming that length disappears.

### 1.2 What this pipeline should not be

Avoid framing the new work as only:

```text
SoftArrangerFlow, but in SMPL-X pose space.
```

That would be too close to the WACV submission. The new paper should instead be framed as:

```text
Lexically anchored continuous SMPL-X sign fields with function-space residual flow matching and physical trajectory supervision.
```

---

## 2. Key Differences from WACV SoftArrangerFlow

### 2.1 WACV formulation

The WACV system generates a discrete latent sequence:

$$
z_1 \in \mathbb{R}^{L \times d_z},
$$

where:

$$
L = \left\lceil \frac{T}{4} \right\rceil.
$$

It then trains a latent residual flow:

$$
z_0 = z_a + \sigma \xi,
$$

$$
z_t = (1 - t)z_0 + t z_1,
$$

$$
v^* = z_1 - z_0.
$$

The model predicts:

$$
\hat{z}_1 = z_t + (1 - t)v_\theta(z_t,t,y),
$$

and a frozen VAE decoder maps the final latent to rot6D SMPL-X motion.

### 2.2 New formulation

The new system should generate a continuous or densely sampled SMPL-X pose trajectory:

$$
X_1(\tau) =
[
R^{6D}_{body}(\tau),
R^{6D}_{hands}(\tau),
R^{6D}_{jaw}(\tau),
\epsilon_{face}(\tau)
],
$$

where:

$$
\tau \in [-1,1].
$$

A coarse scaffold trajectory is:

$$
S(\tau).
$$

The residual target is:

$$
R_1(\tau) = X_1(\tau) - S(\tau).
$$

The flow model learns to transport a smooth source residual to the ground-truth residual:

$$
R_0(\tau) = \sigma \xi_{smooth}(\tau),
$$

$$
R_t(\tau) = (1-t)R_0(\tau) + tR_1(\tau),
$$

$$
u^*(\tau) = R_1(\tau) - R_0(\tau).
$$

The velocity model predicts:

$$
u_\theta = u_\theta(R_t(\tau), t, \tau, S(\tau), h_y, \mathcal{M}),
$$

and the endpoint is:

$$
\hat{R}_1(\tau) = R_t(\tau) + (1-t)u_\theta(R_t(\tau), t, \tau, S(\tau), h_y, \mathcal{M}),
$$

$$
\hat{X}_1(\tau) = S(\tau) + \hat{R}_1(\tau).
$$

The final prediction is supervised directly in SMPL-X pose and 3D joint space.

---

## 3. Current Experimental Lessons to Encode in the Implementation

The implementation should be shaped by the current oracle experiments.

### 3.1 Latent-field lesson

The oracle latent-field experiment showed:

- residual SIREN can nearly perfectly fit observed VAE latent tokens in `fit_all`;
- held-out even-odd latent interpolation is poor;
- the current VAE latent space is not interpolation-friendly enough for a pure global implicit latent field.

Implementation consequence:

```text
Do not build the main model as a pure global latent field.
```

### 3.2 SMPL-X field lesson

The oracle SMPL-X rot6D/FK experiment showed:

- the rot6D + differentiable SMPL-X FK pipeline works;
- direct SIREN is not competitive;
- linear rot6D and SLERP are strong interpolation baselines;
- sparse and span masks reveal shortcutting;
- residual SIREN currently mostly copies the interpolation scaffold;
- dense acceleration and jerk regularization alone gives tiny improvements.

Implementation consequence:

```text
Use a scaffolded residual model, but make the residual model semantically conditioned and train it with supervised velocity/path losses, not only smoothness losses.
```

### 3.3 NIAF lesson

NIAF motivates the shift from discrete waypoints to continuous functions:

$$
A(\tau) = \Phi(\tau;\theta).
$$

The important transferable ideas are:

1. query trajectories at arbitrary temporal resolution;
2. use analytical or well-defined derivatives;
3. supervise dynamics, not only positions;
4. avoid relying on finite-difference artifacts when possible.

Implementation consequence:

```text
Adopt continuous time coordinates and dynamics losses, but do not copy NIAF as a pure text-to-SIREN parameter generator yet.
```

---

## 4. Data Representation

### 4.1 Compact SMPL-X feature

Use the same compact upper-body SMPL-X representation as the current pipeline:

```text
41 rotations * 6D + 10 expression coefficients = 256 dimensions
```

For each frame:

$$
x_i \in \mathbb{R}^{256}.
$$

The full sequence is:

$$
X = [x_0, x_1, \ldots, x_{T-1}] \in \mathbb{R}^{T \times 256}.
$$

### 4.2 Rotation conversion

Input compact SMPL-X is often stored in axis-angle form:

$$
a_{i,j} \in \mathbb{R}^3.
$$

Convert axis-angle to rotation matrix:

$$
R_{i,j} = \exp([a_{i,j}]_\times) \in SO(3).
$$

Convert rotation matrix to rot6D by taking the first two columns:

$$
r^{6D}_{i,j} = [R_{i,j}^{[:,1]}, R_{i,j}^{[:,2]}] \in \mathbb{R}^{6}.
$$

During prediction, convert rot6D back to a valid rotation matrix by Gram-Schmidt projection:

$$
b_1 = \frac{a_1}{\|a_1\|_2 + \epsilon},
$$

$$
b_2 = \frac{a_2 - (b_1^\top a_2)b_1}{\|a_2 - (b_1^\top a_2)b_1\|_2 + \epsilon},
$$

$$
b_3 = b_1 \times b_2,
$$

$$
R = [b_1,b_2,b_3].
$$

### 4.3 Dataset record format

Each training item should expose:

```python
sample = {
    "id": str,
    "text": str,
    "x_6d": FloatTensor[T, 256],
    "axis_angle": FloatTensor[T, D_axis],
    "expr": FloatTensor[T, 10],
    "mask": BoolTensor[T],
    "joints": FloatTensor[T, J, 3],
    "duration": int,
    "retrieval": Dict,
}
```

If computing SMPL-X FK online is too expensive, cache the GT joints:

```text
cache/joints/{split}/{sample_id}.npz
```

Recommended cached arrays:

```text
x_6d:        [T, 256]
axis_angle:  [T, D_axis]
expr:        [T, 10]
joints:      [T, J, 3]
hand_joints: [T, J_hand, 3]
mask:        [T]
```

---

## 5. Proposed Code Structure

Recommended new module root:

```text
continuous_sign_field/
```

Suggested structure:

```text
continuous_sign_field/
  configs/
    how2sign_stage0_scaffold.yaml
    how2sign_stage1_gt_anchor_flow.yaml
    how2sign_stage2_pred_anchor_flow.yaml
    how2sign_stage3_lexical_conditioned_flow.yaml
    how2sign_stage4_timewarp.yaml

  data/
    smplx_dataset.py
    collate.py
    retrieval_dataset.py
    preprocess_cache.py

  geometry/
    rotations.py
    smplx_fk.py
    geodesic.py
    slerp.py

  scaffold/
    anchor_provider.py
    gt_anchor_provider.py
    saf_anchor_provider.py
    interpolation.py
    time_parameterization.py

  models/
    text_encoder.py
    lexical_memory.py
    duration_head.py
    conditioner.py
    residual_flow_transformer.py
    residual_flow_mlp.py
    timewarp_head.py

  flow/
    smooth_noise.py
    interpolants.py
    solvers.py

  losses/
    flow_losses.py
    pose_losses.py
    dynamics_losses.py
    path_losses.py
    total_loss.py

  train/
    train_residual_flow.py
    train_duration.py
    train_timewarp.py

  eval/
    metrics.py
    soke_metrics.py
    back_translation.py
    visualize.py
    export_smplx.py

  scripts/
    run_stage0_scaffold_eval.py
    run_stage1_gt_anchor_flow.py
    run_stage2_pred_anchor_flow.py
    run_stage3_lexical_flow.py
    sample_text2sign.py
    export_video.py
```

Reuse existing code when possible:

```text
NIAF/oracle_smplx_field/geometry/smplx_fk.py
NIAF/oracle_smplx_field/models/baselines.py
NIAF/oracle_smplx_field/losses.py
flow/visualize/visualize_latent_trajectory.py, as a visualization reference
```

---

## 6. Pipeline Overview

### 6.1 Training-time inputs

For a training sequence:

```text
text y
GT pose trajectory X_1[0:T]
retrieved lexical candidates C(y)
optional predicted scaffold S_pred[0:T]
```

### 6.2 Inference-time inputs

At inference:

```text
text y only
preconstructed lexical dictionary
```

The model predicts or constructs:

```text
retrieved lexical memory M
predicted duration T_hat
coarse scaffold S(tau)
residual flow endpoint R_hat_1(tau)
final pose X_hat_1(tau)
```

### 6.3 End-to-end inference

```text
1. Encode input text y.
2. Retrieve lexical candidates C(y).
3. Build lexical memory M.
4. Predict duration T_hat.
5. Generate or retrieve coarse scaffold anchors A.
6. Interpolate anchors into S(tau).
7. Sample smooth source residual R_0(tau).
8. Integrate residual flow from t=0 to t=1.
9. Output X_hat_1(tau) = S(tau) + R_hat_1(tau).
10. Project rot6D to SO(3).
11. Convert to SMPL-X axis-angle if needed.
12. Render or evaluate.
```

---

## 7. Scaffold Design

The scaffold is essential. Current oracle results suggest that pure direct SIREN is weak, while interpolation scaffolds are strong but shortcut in hard gaps.

### 7.1 Scaffold sources

Implement three scaffold sources.

#### Source A: ground-truth anchors

Used only for oracle training/debugging.

Given GT trajectory:

$$
X_1 = [x_0,\ldots,x_{T-1}],
$$

select anchors with stride q:

$$
A_m = X_1[mq].
$$

Recommended starting values:

```text
q = 4 frames
q ablation = [2, 4, 8, 16]
```

#### Source B: current SoftArrangerFlow anchors

Use the WACV model as a coarse anchor provider.

```text
text y
  -> current SoftArrangerFlow inference
  -> decoded SMPL-X rot6D sequence X_saf
  -> downsample to anchors A_pred
```

This is the safest first full text-to-sign implementation.

#### Source C: new direct anchor generator

Later version.

```text
text + lexical memory
  -> transformer anchor generator
  -> sparse anchors A_pred
```

This removes dependence on the frozen VAE but is more difficult.

### 7.2 Interpolation scaffold

Given anchors A and query times tau, construct:

$$
S(\tau) = \text{Interp}(A,\tau).
$$

Implement both:

```text
linear rot6D interpolation
per-joint SLERP + linear expression interpolation
```

Use SLERP as the main scaffold if it is stable, because it respects rotation geometry better.

### 7.3 Scaffold-only evaluation

Before training flow, always report scaffold-only metrics:

```text
linear_rot6d scaffold
slerp scaffold
current SoftArrangerFlow decoded scaffold
```

This is necessary because the residual model must beat the scaffold, not merely copy it.

---

## 8. Residual Flow Matching Objective

### 8.1 Residual target

Let the GT trajectory be:

$$
X_1(\tau) \in \mathbb{R}^{256}.
$$

Let the scaffold be:

$$
S(\tau) \in \mathbb{R}^{256}.
$$

Define the target residual:

$$
R_1(\tau) = X_1(\tau) - S(\tau).
$$

### 8.2 Smooth source residual

Sample smooth noise:

$$
R_0(\tau) = \sigma \xi_{smooth}(\tau).
$$

Do not use independent per-frame noise as the default source, because it encourages jitter.

Implementation options:

1. sample Gaussian noise and apply a 1D Gaussian filter over time;
2. sample low-frequency DCT coefficients and reconstruct a trajectory;
3. sample cubic control points and interpolate;
4. set R_0 = 0 for deterministic ablation.

Recommended first version:

```text
Gaussian noise + temporal Gaussian smoothing
```

### 8.3 Flow bridge

For:

$$
t \sim U(0,1),
$$

define:

$$
R_t(\tau) = (1-t)R_0(\tau) + tR_1(\tau).
$$

The target velocity is:

$$
u^*(\tau) = R_1(\tau) - R_0(\tau).
$$

The model predicts:

$$
\hat{u}(\tau) = u_\theta(R_t(\tau), t, \tau, S(\tau), h_y, \mathcal{M}).
$$

The flow-matching loss is:

$$
\mathcal{L}_{FM}
=
\mathbb{E}_{t,\tau}
\left[
\left\|
 u_\theta(R_t(\tau), t, \tau, S(\tau), h_y, \mathcal{M})
 -
 u^*(\tau)
\right\|_2^2
\right].
$$

### 8.4 Endpoint prediction during training

At any sampled t, the clean endpoint estimate is:

$$
\hat{R}_1(\tau)
=
R_t(\tau) + (1-t)u_\theta(R_t(\tau), t, \tau, S(\tau), h_y, \mathcal{M}).
$$

Then:

$$
\hat{X}_1(\tau) = S(\tau) + \hat{R}_1(\tau).
$$

Apply physical endpoint losses to:

$$
\hat{X}_1(\tau).
$$

---

## 9. Velocity Network Architecture

### 9.1 Recommended first architecture: residual flow Transformer

Do not start with a pointwise SIREN as the main generator. Use a temporal Transformer because sign motion depends on neighboring frames and phrase-level context.

Input at query point i:

$$
f_i = [R_t(\tau_i), S(\tau_i), e_\tau(\tau_i), e_t(t), v_S(\tau_i), a_S(\tau_i)].
$$

Where:

- R_t is the current residual state;
- S is the scaffold pose;
- e_tau is Fourier time encoding of trajectory coordinate;
- e_t is flow-time embedding;
- v_S is scaffold velocity;
- a_S is scaffold acceleration.

Project to hidden dimension:

$$
h_i^{(0)} = W_f f_i.
$$

Run a temporal Transformer:

$$
H = \text{Transformer}(h^{(0)}_{1:K}).
$$

Condition on text and lexical memory using cross-attention:

$$
H' = \text{CrossAttn}(H, [H_y; \mathcal{M}]).
$$

Output velocity:

$$
\hat{u}_{1:K} = W_o H'.
$$

Recommended initial hyperparameters:

```text
hidden_dim: 512
num_layers: 6
num_heads: 8
mlp_ratio: 4
output_dim: 256
dropout: 0.1
activation: GELU
```

### 9.2 Alternative architecture: FiLM Fourier MLP

Use this for fast ablation or dense continuous querying:

$$
h^{(0)} = [R_t(\tau), S(\tau), e_\tau(\tau), e_t(t)].
$$

Each layer is modulated by text/retrieval context c:

$$
h^{(\ell+1)} = \phi((W^{(\ell)}h^{(\ell)}) \odot (1+\gamma^{(\ell)}(c)) + \beta^{(\ell)}(c)).
$$

This is closer to NIAF, but it should be an ablation first.

### 9.3 Do not diffuse or flow over SIREN weights first

Avoid using flow matching over SIREN parameter vectors:

$$
\theta \sim p(\theta|y).
$$

Reason: SIREN parameter space is non-identifiable. Many parameter settings can represent similar trajectories, making generative modeling unstable.

Prefer trajectory-space flow:

$$
R_t(\tau) \rightarrow R_1(\tau).
$$

---

## 10. Conditioning Design

### 10.1 Text features

Use the existing frozen or fine-tuned text encoder:

$$
H_y = F_{text}(y).
$$

Use both:

```text
pooled sentence feature h_y
prefix/token-level text features H_y
```

### 10.2 Lexical memory

Retrieve candidate word-level signs:

$$
C(y) = \{(\ell_k, w_k, d_k)\}_{k=1}^{K_y}.
$$

Build memory tokens:

$$
m_{k,j} = P_x x_{k,j} + P_e e_k + p^{word}_j.
$$

Candidate gate:

$$
g_k = \sigma(MLP([h_y,e_k,\bar{x}_k,h_y \odot e_k])).
$$

Memory tokens passed to the flow model:

$$
\mathcal{M} = \{m_{k,j}\}_{k,j} \cup \{m_\emptyset\}.
$$

Important implementation point:

```text
Candidate memory should condition the continuous field, but the paper should not present SoftArranger as the main new contribution again.
```

### 10.3 Local scaffold context

For each query time tau, add local scaffold features:

```text
S(tau)
scaffold velocity dS/dtau
scaffold acceleration d2S/dtau2
nearest anchor index
local phase within anchor interval
```

This helps the residual model know where it is relative to the coarse trajectory.

---

## 11. Loss Functions

The total loss should combine flow matching and physical endpoint supervision.

$$
\mathcal{L}
=
\lambda_{FM}\mathcal{L}_{FM}
+
\lambda_{6D}\mathcal{L}_{6D}
+
\lambda_{geo}\mathcal{L}_{geo}
+
\lambda_{joint}\mathcal{L}_{joint}
+
\lambda_{hand}\mathcal{L}_{hand}
+
\lambda_{expr}\mathcal{L}_{expr}
+
\lambda_{vel}\mathcal{L}_{vel}
+
\lambda_{acc}\mathcal{L}_{acc}
+
\lambda_{path}\mathcal{L}_{path}
+
\lambda_{res}\mathcal{L}_{res}.
$$

### 11.1 Flow matching loss

$$
\mathcal{L}_{FM}
=
\frac{1}{BKD}
\sum_{b,i}
\left\|
\hat{u}_{b,i} - u^*_{b,i}
\right\|_2^2.
$$

### 11.2 rot6D endpoint loss

$$
\mathcal{L}_{6D}
=
\frac{1}{BTD}
\sum_{b,i}
\left\|
\hat{X}^{6D}_{b,i} - X^{6D}_{b,i}
\right\|_1.
$$

### 11.3 Rotation geodesic loss

Convert rot6D predictions to rotation matrices:

$$
\hat{R}_{i,j} = \Pi_{SO(3)}(\hat{r}^{6D}_{i,j}).
$$

Then:

$$
\mathcal{L}_{geo}
=
\frac{1}{BT|\mathcal{J}_{rot}|}
\sum_{b,i,j}
 d_{SO(3)}(\hat{R}_{b,i,j}, R_{b,i,j}),
$$

where:

$$
d_{SO(3)}(\hat{R},R)
=
\arccos
\left(
\frac{\operatorname{Tr}(\hat{R}^{\top}R)-1}{2}
\right).
$$

Clamp the arccos input:

$$
c = \operatorname{clamp}(c, -1+\epsilon, 1-\epsilon).
$$

### 11.4 Joint loss through SMPL-X FK

Run differentiable FK:

$$
\hat{J}_{b,i} = FK_{SMPLX}(\hat{X}_{b,i}).
$$

Then:

$$
\mathcal{L}_{joint}
=
\frac{1}{BT|\mathcal{J}|}
\sum_{b,i}
\left\|
\hat{J}_{b,i} - J_{b,i}
\right\|_1.
$$

### 11.5 Hand-weighted joint loss

Hands are central to sign language, so add:

$$
\mathcal{L}_{hand}
=
\frac{1}{BT|\mathcal{J}_{hand}|}
\sum_{b,i}
\left\|
\hat{J}^{hand}_{b,i} - J^{hand}_{b,i}
\right\|_1.
$$

Use:

$$
\lambda_{hand} > \lambda_{joint}.
$$

### 11.6 Expression loss

For facial expression coefficients:

$$
\mathcal{L}_{expr}
=
\frac{1}{BTD_e}
\sum_{b,i}
\left\|
\hat{\epsilon}_{b,i} - \epsilon_{b,i}
\right\|_1.
$$

### 11.7 Supervised velocity loss

Use finite differences on joints:

$$
\Delta \hat{J}_{i} = \hat{J}_{i+1} - \hat{J}_i,
$$

$$
\Delta J_i = J_{i+1} - J_i.
$$

Then:

$$
\mathcal{L}_{vel}
=
\frac{1}{B(T-1)|\mathcal{J}|}
\sum_{b,i}
\left\|
\Delta \hat{J}_{b,i} - \Delta J_{b,i}
\right\|_1.
$$

Hand-weighted version:

$$
\mathcal{L}_{vel}^{hand}
=
\frac{1}{B(T-1)|\mathcal{J}_{hand}|}
\sum_{b,i}
\left\|
\Delta \hat{J}^{hand}_{b,i} - \Delta J^{hand}_{b,i}
\right\|_1.
$$

### 11.8 Acceleration loss

$$
\Delta^2 \hat{J}_i = \hat{J}_{i+1} - 2\hat{J}_{i} + \hat{J}_{i-1}.
$$

$$
\mathcal{L}_{acc}
=
\frac{1}{B(T-2)|\mathcal{J}|}
\sum_{b,i}
\left\|
\Delta^2 \hat{J}_{b,i} - \Delta^2 J_{b,i}
\right\|_1.
$$

Use acceleration with a smaller weight than velocity.

### 11.9 Hand path-length preservation

Current oracle SMPL-X experiments show shortcutting in sparse settings. Add a path-length preservation term:

$$
PL(\hat{J}^{hand})
=
\sum_{i=0}^{T-2}
\left\|
\hat{J}^{hand}_{i+1}-\hat{J}^{hand}_{i}
\right\|_2.
$$

$$
PLR
=
\frac{PL(\hat{J}^{hand})}{PL(J^{hand}) + \epsilon}.
$$

Loss:

$$
\mathcal{L}_{path}
=
|PLR - 1|.
$$

Optionally compute per-hand and per-wrist path losses separately.

### 11.10 Residual magnitude loss

Prevent the residual from destroying a good scaffold:

$$
\mathcal{L}_{res}
=
\frac{1}{BTD}
\sum_{b,i}
\left\|
\hat{X}_{b,i} - S_{b,i}
\right\|_2^2.
$$

Keep this weight small. If it is too large, the model will copy the scaffold, which is already a known failure mode.

### 11.11 Initial loss weights

Use these as starting values, then tune by validation:

```yaml
lambda_fm: 1.0
lambda_6d: 1.0
lambda_geo: 0.05
lambda_joint: 10.0
lambda_hand: 20.0
lambda_expr: 1.0
lambda_vel: 5.0
lambda_vel_hand: 10.0
lambda_acc: 1.0
lambda_path: 0.5
lambda_res: 1.0e-4
```

If units differ, normalize losses by their running standard deviation or tune weights after inspecting magnitudes.

---

## 12. Training Stages

### Stage 0: scaffold baselines

Goal:

```text
Establish how strong the scaffold is before adding flow.
```

Run:

```text
linear rot6D interpolation
SLERP interpolation
current SoftArrangerFlow decoded output
current SoftArrangerFlow downsampled anchors + interpolation
```

Report:

```text
JPE
hand JPE
rot6D L1
geodesic error
joint R2
PLR
velocity error
```

Success criterion:

```text
The evaluation code reproduces the oracle SMPL-X findings and exposes shortcutting under sparse anchors.
```

### Stage 1: residual flow with GT anchors

Goal:

```text
Test whether residual flow can recover dense GT motion when the scaffold comes from ground-truth anchors.
```

Use:

```text
anchor stride q = 4
GT anchors
SLERP scaffold
text conditioning optional
lexical conditioning optional
```

Train residual flow with:

```text
L_FM + endpoint physical losses
```

This is an oracle upper bound. It should outperform scaffold-only.

If it does not outperform scaffold-only, fix losses/architecture before moving on.

### Stage 2: residual flow with predicted anchors

Goal:

```text
Train the flow to correct realistic scaffold errors.
```

Use current SoftArrangerFlow as anchor provider:

```text
y -> SoftArrangerFlow -> X_saf -> anchors -> scaffold S_pred
```

Train with scheduled anchor mixing:

```text
epoch 0-20%: 80% GT anchors, 20% predicted anchors
epoch 20-60%: 50% GT anchors, 50% predicted anchors
epoch 60-100%: 20% GT anchors, 80% predicted anchors
final fine-tune: 100% predicted anchors
```

### Stage 3: add lexical memory conditioning

Goal:

```text
Show that lexical memory helps residual trajectory generation, not just the scaffold.
```

Compare:

```text
text only
text + pooled retrieval statistics
text + lexical memory cross-attention
text + lexical memory + candidate gates
```

This stage is important for semantic faithfulness and for differentiating the new model from generic pose-space flow.

### Stage 4: add duration and tempo

First version:

```text
Use the existing duration predictor.
```

Later version:

```text
Add time-warp / speed field.
```

The content-progress coordinate is:

$$
s \in [0,1].
$$

The physical time speed is:

$$
\frac{dt}{ds} = \operatorname{softplus}(r_\theta(s,y,\mathcal{M})).
$$

Duration is:

$$
T = \int_0^1 \frac{dt}{ds} ds.
$$

For the first paper version, this can be an extension or ablation. Do not block the main implementation on time-warp.

### Stage 5: joint fine-tuning

Only after Stage 2 and Stage 3 are stable:

```text
unfreeze selected scaffold/anchor generator modules
keep text encoder mostly frozen
fine-tune residual flow + duration head + lightweight adapter layers
```

Do not fully end-to-end train everything at the start.

---

## 13. Sampling and Solvers

### 13.1 Training endpoint estimate

During training, endpoint estimate from one random t is:

$$
\hat{R}_1 = R_t + (1-t)u_\theta(R_t,t,c).
$$

Use this for endpoint losses.

### 13.2 Inference ODE

At inference, initialize:

$$
R_{t=0}(\tau) = R_0(\tau).
$$

Solve:

$$
\frac{dR_t(\tau)}{dt} = u_\theta(R_t(\tau),t,\tau,S(\tau),h_y,\mathcal{M}).
$$

Euler update:

$$
R_{t+\Delta t} = R_t + \Delta t\,u_\theta(R_t,t,c).
$$

Heun update:

$$
k_1 = u_\theta(R_t,t,c),
$$

$$
\tilde{R} = R_t + \Delta t k_1,
$$

$$
k_2 = u_\theta(\tilde{R},t+\Delta t,c),
$$

$$
R_{t+\Delta t} = R_t + \frac{\Delta t}{2}(k_1+k_2).
$$

Recommended inference steps:

```text
1-step Euler
2-step Heun
4-step Heun
8-step Heun
```

Report quality-latency curves.

### 13.3 Deterministic and stochastic modes

Deterministic generation:

```text
R_0 = 0
```

Stochastic generation:

```text
R_0 = sigma * smooth_noise
```

Use deterministic mode for main benchmark comparability. Use stochastic mode for diversity analysis.

---

## 14. Time Parameterization

Implement at least three time modes.

### 14.1 Uniform time

$$
\tau_i = -1 + \frac{2i}{T-1}.
$$

This is the default.

### 14.2 Joint arc length

Compute cumulative joint-space path length:

$$
s_i =
\frac{
\sum_{r=1}^{i}\|J_r - J_{r-1}\|_2
}{
\sum_{r=1}^{T-1}\|J_r - J_{r-1}\|_2 + \epsilon
}.
$$

Then:

$$
\tau_i = 2s_i - 1.
$$

### 14.3 Hand arc length

Use only hand joints:

$$
s_i^{hand} =
\frac{
\sum_{r=1}^{i}\|J_r^{hand} - J_{r-1}^{hand}\|_2
}{
\sum_{r=1}^{T-1}\|J_r^{hand} - J_{r-1}^{hand}\|_2 + \epsilon
}.
$$

Then:

$$
\tau_i = 2s_i^{hand} - 1.
$$

Use hand arc length as an ablation because sign language is highly hand-dominant.

---

## 15. Training Step Pseudocode

```python
def train_step(batch):
    # 1. Load GT pose trajectory.
    x_gt = batch["x_6d"]          # [B, T, 256]
    mask = batch["mask"]          # [B, T]
    text = batch["text"]

    # 2. Build text and lexical conditioning.
    H_y, h_y = text_encoder(text)
    memory = lexical_memory_builder(batch["retrieval"], H_y)

    # 3. Build scaffold.
    anchors = anchor_provider(batch)       # GT anchors or predicted anchors
    S = scaffold_interpolator(anchors, batch["tau"])  # [B, T, 256]

    # 4. Define residual target.
    R1 = x_gt - S

    # 5. Sample smooth source residual.
    R0 = smooth_noise_like(R1, sigma=config.sigma_noise)

    # 6. Sample flow time.
    t = torch.rand(B, 1, 1, device=x_gt.device)

    # 7. Interpolate residual state.
    Rt = (1.0 - t) * R0 + t * R1
    u_star = R1 - R0

    # 8. Predict velocity.
    u_pred = flow_model(
        Rt=Rt,
        t=t,
        tau=batch["tau"],
        scaffold=S,
        text_tokens=H_y,
        lexical_memory=memory,
        mask=mask,
    )

    # 9. Flow loss.
    loss_fm = masked_mse(u_pred, u_star, mask)

    # 10. Endpoint estimate.
    R1_hat = Rt + (1.0 - t) * u_pred
    x_hat = S + R1_hat

    # 11. Project rotations and run FK.
    R_hat = rot6d_to_rotmat(x_hat[..., :246])
    J_hat = smplx_fk(R_hat, expr=x_hat[..., 246:])

    R_gt = rot6d_to_rotmat(x_gt[..., :246])
    J_gt = batch["joints"]

    # 12. Physical losses.
    losses = {}
    losses["fm"] = loss_fm
    losses["rot6d"] = rot6d_l1(x_hat, x_gt, mask)
    losses["geo"] = geodesic_loss(R_hat, R_gt, mask)
    losses["joint"] = joint_l1(J_hat, J_gt, mask)
    losses["hand"] = hand_joint_l1(J_hat, J_gt, mask)
    losses["expr"] = expr_l1(x_hat, x_gt, mask)
    losses["vel"] = joint_velocity_loss(J_hat, J_gt, mask)
    losses["acc"] = joint_acceleration_loss(J_hat, J_gt, mask)
    losses["path"] = hand_path_length_loss(J_hat, J_gt, mask)
    losses["res"] = residual_magnitude_loss(x_hat, S, mask)

    # 13. Weighted sum.
    loss = sum(config.loss_weights[k] * v for k, v in losses.items())

    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
    optimizer.step()

    return loss, losses
```

---

## 16. Inference Pseudocode

```python
@torch.no_grad()
def generate(text, seed=0, num_steps=4, deterministic=True):
    # 1. Text and retrieval.
    H_y, h_y = text_encoder(text)
    retrieval = retrieve_candidates(text)
    memory = lexical_memory_builder(retrieval, H_y)

    # 2. Duration.
    T_hat = duration_head(h_y, retrieval.stats)
    tau = make_uniform_tau(T_hat)

    # 3. Scaffold.
    anchors = predicted_anchor_provider(text, retrieval)
    S = scaffold_interpolator(anchors, tau)

    # 4. Initial residual.
    if deterministic:
        R = torch.zeros_like(S)
    else:
        torch.manual_seed(seed)
        R = smooth_noise_like(S, sigma=config.sigma_noise)

    # 5. ODE solve.
    if solver == "euler":
        R = euler_solve(flow_model, R, S, tau, H_y, memory, num_steps)
    elif solver == "heun":
        R = heun_solve(flow_model, R, S, tau, H_y, memory, num_steps)

    # 6. Final pose.
    X_hat = S + R

    # 7. Convert to valid SMPL-X.
    R_hat = rot6d_to_rotmat(X_hat[..., :246])
    axis_angle = rotmat_to_axis_angle(R_hat)
    expr = X_hat[..., 246:]
    smplx_params = pack_upper_body_smplx(axis_angle, expr)

    return smplx_params
```

---

## 17. Configuration Template

```yaml
experiment_name: how2sign_continuous_smplx_residual_flow_v1
seed: 42

paths:
  data_root: /media/cvpr/haomian/data/SOKE_FLOW/how2sign_soke_upper_smplx
  retrieval_dict: /path/to/word_motion_dictionary
  saf_checkpoint: /path/to/current_softarrangerflow_checkpoint.pt
  smplx_model_dir: /path/to/smplx/models
  output_dir: experiments/continuous_sign_field/how2sign_v1

data:
  split: train
  max_frames: 400
  representation_dim: 256
  num_rotations: 41
  expr_dim: 10
  fps: 25
  normalize_x6d: true
  cache_fk_joints: true

scaffold:
  source: saf_predicted        # choices: gt, saf_predicted, anchor_transformer
  interpolation: slerp         # choices: linear_rot6d, slerp
  anchor_stride: 4
  scheduled_anchor_mixing: true
  gt_anchor_prob_start: 0.8
  gt_anchor_prob_end: 0.0

conditioning:
  use_text_tokens: true
  use_pooled_text: true
  use_lexical_memory: true
  use_candidate_gates: true
  use_scaffold_velocity: true
  use_scaffold_acceleration: true

model:
  type: residual_flow_transformer
  hidden_dim: 512
  num_layers: 6
  num_heads: 8
  mlp_ratio: 4
  dropout: 0.1
  tau_fourier_bands: 16
  flow_time_fourier_bands: 16
  output_dim: 256

flow:
  source_noise: smooth_gaussian
  sigma_noise: 0.2
  smoothing_kernel_size: 9
  smoothing_sigma: 2.0
  train_t_sampling: uniform
  inference_solver: heun
  inference_steps: 4
  deterministic_default: true

loss_weights:
  fm: 1.0
  rot6d: 1.0
  geo: 0.05
  joint: 10.0
  hand: 20.0
  expr: 1.0
  vel: 5.0
  vel_hand: 10.0
  acc: 1.0
  path: 0.5
  res: 0.0001

optimization:
  optimizer: adamw
  lr: 0.0001
  weight_decay: 0.01
  betas: [0.9, 0.95]
  batch_size: 16
  grad_clip: 1.0
  max_epochs: 100
  warmup_steps: 2000
  cosine_schedule: true
  mixed_precision: true

evaluation:
  eval_every_steps: 2000
  metrics:
    - dtw_jpe
    - dtw_pa_jpe
    - hand_jpe
    - rot6d_l1
    - geodesic_deg
    - joint_r2
    - plr
    - velocity_error
    - acceleration_error
    - duration_mae
    - back_translation_bleu4
  render_videos: true
  num_visualizations: 20
```

---

## 18. Evaluation Plan

### 18.1 Main baselines

Compare against:

```text
1. Current SoftArrangerFlow
2. SoftArrangerFlow prior-only
3. Scaffold-only from SoftArrangerFlow anchors
4. Scaffold + deterministic residual regression
5. Scaffold + residual diffusion
6. Scaffold + residual flow matching
7. Scaffold + residual flow matching + physical losses
8. Direct text-conditioned pose field
9. No lexical retrieval
10. No hand/path/velocity losses
```

### 18.2 Automatic metrics

Use existing SOKE-compatible metrics:

```text
DTW-JPE body
DTW-JPE hand
DTW-PA-JPE body
DTW-PA-JPE hand
back-translation BLEU-4
```

Add physically meaningful trajectory metrics:

```text
unwarped JPE
hand JPE
rot6D L1
rotation geodesic error in degrees
joint-space R2
hand velocity error
hand acceleration error
hand path-length ratio
jerk statistics
expression coefficient error
duration MAE
latency per sentence
```

### 18.3 Important diagnostic metrics

#### Path-length ratio

Low PLR indicates shortcutting:

$$
PLR = \frac{PL(\hat{J}^{hand})}{PL(J^{hand}) + \epsilon}.
$$

Target:

```text
closer to 1 than scaffold-only
```

#### Flow endpoint gain over scaffold

For a metric E where lower is better:

$$
Gain = \frac{E_{scaffold} - E_{flow}}{E_{scaffold}}.
$$

Report gains for:

```text
hand JPE
DTW hand JPE
velocity error
PLR distance to 1
```

#### Semantic preservation

Back-translation BLEU-4 should not drop relative to SoftArrangerFlow. If geometric metrics improve but BLEU drops, the model may be smoothing away meaning-critical hand/facial motion.

### 18.4 Visualizations

Export for each validation epoch:

```text
side-by-side videos: GT, scaffold, flow output
hand trajectory 3D plots
hand velocity curves
path-length curves
highest-error frame renderings
flow residual magnitude over time
candidate attention over time
```

---

## 19. Ablation Matrix

### 19.1 Scaffold and anchor ablations

| Variant | Purpose |
|---|---|
| linear rot6D scaffold | lower-bound interpolation scaffold |
| SLERP scaffold | rotation-aware scaffold |
| GT anchors | oracle upper bound |
| SoftArrangerFlow predicted anchors | main practical setting |
| anchor stride 2/4/8/16 | quality vs sparsity |

### 19.2 Flow objective ablations

| Variant | Purpose |
|---|---|
| endpoint regression only | test if flow is needed |
| flow only | test pure FM objective |
| flow + rot6D | basic endpoint supervision |
| flow + FK joint | physical supervision |
| flow + FK joint + velocity | dynamics supervision |
| flow + FK joint + velocity + path | anti-shortcut version |

### 19.3 Conditioning ablations

| Variant | Purpose |
|---|---|
| scaffold only | no semantic condition |
| text pooled only | sentence-level semantics |
| text tokens only | token-level semantics |
| lexical memory only | retrieval contribution |
| text + lexical memory | main setting |
| no candidate gates | tests retrieval reliability |
| no NULL memory | tests missing lexical evidence |

### 19.4 Solver ablations

| Variant | Purpose |
|---|---|
| 1-step Euler | fastest |
| 2-step Heun | quality-speed balance |
| 4-step Heun | recommended main |
| 8-step Heun | high quality |
| deterministic R0=0 | benchmark stability |
| stochastic R0 | diversity |

---

## 20. Implementation Milestones

### Milestone 1: dataset and geometry validation

Deliverables:

```text
smplx_dataset.py
rot6d_to_rotmat unit tests
axis_angle_to_rot6d unit tests
SMPL-X FK batch test
cached joints
```

Checks:

```text
rot6D -> rotmat matrices are orthonormal
geodesic self-error is near zero
FK output shape is correct
cached GT joints match online FK
```

### Milestone 2: scaffold-only benchmark

Deliverables:

```text
GT anchor scaffold evaluation
SAF anchor scaffold evaluation
linear vs SLERP comparison
summary.json and metrics_rows.csv
```

Checks:

```text
SLERP and linear are close for short gaps
sparse anchors expose PLR shortcutting
metrics match oracle SMPL-X trends
```

### Milestone 3: residual flow with GT anchors

Deliverables:

```text
residual_flow_transformer.py
smooth_noise.py
flow_losses.py
pose_losses.py
train_stage1_gt_anchor_flow.py
```

Pass criterion:

```text
GT-anchor flow improves scaffold-only on hand JPE and PLR.
```

### Milestone 4: residual flow with predicted anchors

Deliverables:

```text
saf_anchor_provider.py
scheduled anchor mixing
train_stage2_pred_anchor_flow.py
```

Pass criterion:

```text
Predicted-anchor flow improves current SoftArrangerFlow scaffold or at least improves hand dynamics without BLEU drop.
```

### Milestone 5: lexical conditioning

Deliverables:

```text
lexical_memory.py
candidate cross-attention
conditioning ablations
```

Pass criterion:

```text
Text + lexical memory outperforms text-only and no-retrieval variants.
```

### Milestone 6: full benchmark

Deliverables:

```text
How2Sign validation/test results
CSL-Daily validation/test results
PHOENIX validation/test results
SOKE-compatible metric tables
quality-latency curves
qualitative videos
```

Pass criterion:

```text
Improves at least hand DTW-JPE, hand DTW-PA-JPE, dynamics metrics, or latency-quality tradeoff over WACV SoftArrangerFlow.
```

---

## 21. Risk Register and Fixes

### Risk 1: model copies the scaffold

Symptoms:

```text
flow output nearly identical to scaffold
residual magnitude near zero
JPE gain tiny
PLR unchanged
```

Fixes:

```text
reduce lambda_res
increase velocity/path losses
train on harder predicted-anchor errors
add lexical memory cross-attention
use residual target normalization
increase flow model capacity
```

### Risk 2: flow improves geometry but hurts semantics

Symptoms:

```text
hand JPE improves
back-translation BLEU drops
human comprehension worsens
```

Fixes:

```text
increase lexical conditioning strength
add back-translation auxiliary loss if available
preserve candidate attention information
reduce smoothing/path loss weights
inspect high-motion handshape transitions
```

### Risk 3: output rotations are invalid or jittery

Symptoms:

```text
geodesic error spikes
rendered hands twist
rot6D projection unstable
```

Fixes:

```text
normalize rot6D projection carefully
add geodesic loss
try tangent-space residual update
reduce source noise sigma
apply temporal consistency in joint space
```

### Risk 4: FK supervision is too slow

Symptoms:

```text
training throughput too low
GPU memory high
```

Fixes:

```text
cache GT joints
compute FK only for predicted frames sampled for physical loss
use gradient checkpointing
reduce K for physical losses
train with mixed precision
split hand/body losses
```

### Risk 5: too close to WACV submission

Symptoms:

```text
paper reads as SoftArrangerFlow moved to pose space
main ablation only shows VAE latent vs SMPL-X output
```

Fixes:

```text
center the method on continuous SMPL-X sign fields
include function-space flow matching over tau
include physical trajectory supervision
include arbitrary-resolution query experiments
include anti-shortcut analysis
make SoftArranger a conditioning module, not the main contribution
```

---

## 22. Suggested Paper-Level Claims After Implementation

Strong claims to aim for:

1. Continuous SMPL-X sign fields produce physically meaningful hand/body trajectories without relying on a frozen VAE latent bottleneck.
2. Function-space residual flow matching improves over scaffold-only generation while preserving lexical semantic grounding.
3. Hand-weighted velocity and path-length supervision reduces shortcutting, a failure mode exposed by oracle SMPL-X field diagnostics.
4. The model can be queried at multiple temporal resolutions with consistent motion quality.
5. Lexical memory improves semantic faithfulness compared with text-only continuous pose flow.

Avoid claiming:

```text
No length or duration modeling is needed.
```

Instead say:

```text
The trajectory representation is decoupled from fixed latent-token length; duration and tempo are modeled separately.
```

---

## 23. Minimal First Experiment to Run

The first meaningful experiment should be:

```text
Dataset: How2Sign validation pilot
Scaffold: GT anchors, stride 4
Interpolation: SLERP
Model: residual_flow_transformer
Conditioning: text pooled only, then text + lexical memory
Loss: FM + rot6D + geo + joint + hand + velocity + path
Solver: 4-step Heun for inference
```

Compare:

```text
SLERP scaffold only
residual regression
residual flow matching
residual flow matching + physical losses
```

If this does not beat SLERP scaffold-only on held-out frames, do not move to predicted anchors yet.

---

## 24. Minimal Full Text-to-Sign Experiment

After the GT-anchor experiment works, run:

```text
Dataset: How2Sign
Scaffold: current SoftArrangerFlow predicted output
Anchor stride: 4
Interpolation: SLERP
Conditioning: text tokens + lexical memory
Loss: full physical objective
Inference: deterministic R0=0, 4-step Heun
```

Compare against:

```text
current SoftArrangerFlow
SoftArrangerFlow prior-only
SoftArrangerFlow scaffold-only
new scaffold + residual flow
```

Success condition:

```text
The new model improves hand geometry and dynamics while maintaining or improving back-translation BLEU-4.
```

---

## 25. Summary

The implementation should proceed in this order:

```text
1. Validate rot6D + FK data pipeline.
2. Reproduce scaffold-only oracle baselines.
3. Train residual flow with GT anchors.
4. Add physical endpoint losses and path-length preservation.
5. Switch to predicted SoftArrangerFlow anchors with scheduled mixing.
6. Add text and lexical memory conditioning.
7. Evaluate against WACV SoftArrangerFlow and scaffold-only baselines.
8. Add tempo/time-warp only after the core pipeline works.
```

The main system should be:

```text
Lexically anchored continuous SMPL-X residual flow matching.
```

This is different from the WACV system because the new target is not a discrete frozen-VAE latent endpoint. The new target is a physically supervised SMPL-X sign trajectory field with continuous-time conditioning, function-space residual flow, and explicit hand/body dynamics supervision.
