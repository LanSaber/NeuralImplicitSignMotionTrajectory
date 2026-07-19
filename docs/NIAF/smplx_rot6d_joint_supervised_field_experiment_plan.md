# Experiment Plan: Predicting SMPL-X Rotations in rot6D, Supervised with 3D Joints

**Project context:** SoftArrangerFlow / continuous sign motion generation
**Experiment type:** Oracle feasibility experiment first; text-conditioned generation later
**Main question:** Is a continuous field in renderable SMPL-X pose space more interpolation-friendly than the current VAE latent space?

---

## 0. Executive Summary

The current oracle latent-field experiment showed that a residual SIREN can almost perfectly fit observed VAE latent tokens, but it does not interpolate held-out latent tokens reliably. The next experiment should test whether the problem is the current VAE latent geometry rather than the continuous-field idea itself.

This plan proposes an **oracle continuous SMPL-X pose-field fitting experiment**:

```text
continuous time/progress coordinate -> rot6D SMPL-X pose + expression
```

The model predicts SMPL-X rotations in continuous 6D rotation representation, converts them to valid rotation matrices, runs SMPL-X forward kinematics, and applies major supervision in 3D joint space.

The key design is:

$$
\hat{x}(\tau) = f_{\theta}(\tau),
\quad
\hat{x}(\tau) = [\hat{u}_{1}(\tau), \ldots, \hat{u}_{N_R}(\tau), \hat{e}(\tau)],
$$

where each rotation output is a 6D rotation vector:

$$
\hat{u}_{j}(\tau) \in \mathbb{R}^{6},
$$

and expression coefficients are:

$$
\hat{e}(\tau) \in \mathbb{R}^{D_e}.
$$

The 6D rotations are projected to valid rotation matrices:

$$
\hat{R}_{j}(\tau) = \operatorname{Rot6DToMat}(\hat{u}_{j}(\tau)),
$$

then SMPL-X forward kinematics gives 3D joints:

$$
\hat{J}(\tau) = FK_{\text{SMPL-X}}(\hat{R}_{1:N_R}(\tau), \hat{e}(\tau), \beta, \Pi),
$$

where `beta` denotes shape parameters and `Pi` denotes fixed neutral or dataset-specific root, translation, lower-body, and camera/body normalization choices.

The experiment should start as an oracle fitting/interpolation test, not a full text-to-sign generator. If it succeeds, the final model can become an **anchored continuous SMPL-X sign field** conditioned on text and retrieved lexical memory.

---

## 1. Motivation

### 1.1 Why move from latent trajectory to SMPL-X pose trajectory?

The latent-field experiment gave a useful diagnosis:

```text
Fit-all: observed latent tokens are easy to fit.
Even-odd: held-out latent tokens are poorly interpolated.
```

This suggests that the current frozen VAE latent space is not necessarily a smooth physical trajectory space. A smooth curve in VAE latent space may cut through regions that decode to poor or over-smoothed signing.

In contrast, SMPL-X pose and joint space have stronger physical meaning:

- 3D joint velocity corresponds to actual hand/body motion.
- 3D acceleration and jerk correspond to physical smoothness.
- Rotation geodesic distance has a geometric interpretation.
- Forward kinematics enforces a valid articulated body.
- Dense querying can be evaluated visually and with joint-space dynamics.

### 1.2 Why not predict raw axis-angle?

Raw axis-angle is not a good Euclidean regression target:

1. It has discontinuities near rotation angle `pi`.
2. Equivalent rotations can have different axis-angle values.
3. L1/L2 losses on axis-angle are not true rotation distances.
4. Smoothness in axis-angle space does not always imply smooth motion in `SO(3)`.

The recommended representation is:

```text
predict rot6D -> convert to rotation matrix -> supervise with geodesic + 3D joints
```

This matches the existing SoftArrangerFlow preprocessing direction, where compact SMPL-X axis-angle rotations are converted to continuous 6D rotation representation for learning.

---

## 2. Experiment Goal

### 2.1 Main goal

Test whether a continuous field can represent and interpolate a ground-truth SMPL-X signing trajectory better in real pose space than in VAE latent space.

For each ground-truth sequence, fit a per-sequence continuous function:

$$
f_{\theta}: \tau \mapsto \hat{x}(\tau),
\quad
\tau \in [-1, 1].
$$

The target is the frame-level SMPL-X rot6D sequence:

$$
x_0, x_1, \ldots, x_{T-1}.
$$

This is an **oracle experiment** because it uses the ground-truth pose sequence and ground-truth duration. It does not use text conditioning, retrieval, duration prediction, or flow matching.

### 2.2 Hypotheses

**H1.** A continuous field in rot6D + FK joint space will interpolate held-out frames better than the previous VAE latent-field experiment.

**H2.** Joint-space velocity, acceleration, and jerk losses will improve temporal coherence more reliably than raw latent-space derivative losses.

**H3.** A residual field anchored by interpolation scaffolds will outperform a pure global SIREN for sign trajectories, especially on short or sharp motions.

**H4.** Some sharp hand/finger/non-manual transitions should not be over-smoothed; therefore, success should be judged by both smoothness and semantic/articulatory preservation.

---

## 3. Data Representation

### 3.1 Input compact SMPL-X format

The current compact SMPL-X motion can be represented as:

$$
a_i = [r^{aa}_{i,1}, \ldots, r^{aa}_{i,N_R}, e_i],
$$

where:

- `i` is the frame index.
- `N_R` is the number of predicted SMPL-X rotations.
- `r^{aa}_{i,j} in R^3` is the axis-angle vector for joint `j` at frame `i`.
- `e_i in R^{D_e}` is the facial expression vector.

In the current setup, verify the exact dimensions, but the likely layout is:

```text
axis-angle compact dimension = 133
N_R = 41 rotations
D_e = 10 expression coefficients
133 = 41 * 3 + 10
rot6D compact dimension = 256
256 = 41 * 6 + 10
```

### 3.2 Convert axis-angle to rotation matrix

For each joint rotation:

$$
R_{i,j} = \operatorname{AxisAngleToMat}(r^{aa}_{i,j}),
\quad
R_{i,j} \in SO(3).
$$

### 3.3 Convert rotation matrix to rot6D

Use the first two columns of the rotation matrix:

$$
u_{i,j} = \operatorname{MatToRot6D}(R_{i,j})
= [R_{i,j}^{(:,1)}, R_{i,j}^{(:,2)}],
\quad
u_{i,j} \in \mathbb{R}^{6}.
$$

The frame-level target feature is:

$$
x_i = [u_{i,1}, \ldots, u_{i,N_R}, e_i],
\quad
x_i \in \mathbb{R}^{D_x},
$$

where:

$$
D_x = 6N_R + D_e.
$$

### 3.4 Important normalization rule

Do **not** directly convert normalized 6D features to rotation matrices.

Two safe options:

**Option A: no normalization for rot6D.**

Use raw rot6D values in `[-1, 1]` as the network target. Normalize only expression coefficients if needed.

**Option B: normalized network output, then de-normalize before geometry.**

If the network predicts normalized features:

$$
\tilde{x}(\tau) = f_{\theta}(\tau),
$$

then de-normalize before rot6D projection and FK:

$$
\hat{x}(\tau) = \sigma_x \odot \tilde{x}(\tau) + \mu_x.
$$

Only after de-normalization should we compute:

$$
\hat{R}_{j}(\tau) = \operatorname{Rot6DToMat}(\hat{u}_{j}(\tau)).
$$

For the first experiment, **Option A is recommended** for rotations.

---

## 4. rot6D to Rotation Matrix Projection

Given a predicted 6D vector:

$$
\hat{u}_j = [a_1, a_2],
\quad
a_1, a_2 \in \mathbb{R}^{3},
$$

construct an orthonormal rotation matrix using Gram-Schmidt:

$$
b_1 = \frac{a_1}{\|a_1\|_2 + \epsilon},
$$

$$
\tilde{b}_2 = a_2 - (b_1^\top a_2)b_1,
$$

$$
b_2 = \frac{\tilde{b}_2}{\|\tilde{b}_2\|_2 + \epsilon},
$$

$$
b_3 = b_1 \times b_2.
$$

Then:

$$
\hat{R}_j = [b_1, b_2, b_3].
$$

This ensures:

$$
\hat{R}_j \in SO(3)
$$

up to numerical precision.

---

## 5. Time / Progress Parameterization

### 5.1 Uniform time

For a sequence with `T` frames:

$$
\tau_i = -1 + \frac{2i}{T-1},
\quad
i = 0, \ldots, T-1.
$$

This is the simplest setting and should be the default baseline.

### 5.2 Joint arc-length progress

Compute frame-to-frame 3D joint displacement:

$$
d_i^{joint} = \|J_i - J_{i-1}\|_2,
\quad
i = 1, \ldots, T-1.
$$

Cumulative progress:

$$
s_i^{joint} =
\frac{\sum_{r=1}^{i} d_r^{joint}}
{\sum_{r=1}^{T-1} d_r^{joint} + \epsilon}.
$$

Map to `[-1, 1]`:

$$
\tau_i^{joint} = 2s_i^{joint} - 1.
$$

This parameterization asks the field to model the shape of the motion path rather than the uniform frame rate.

### 5.3 Hand-weighted arc-length progress

For sign language, hands matter more than torso motion. Define weighted joints:

$$
d_i^{hand} =
\sum_{q \in \mathcal{H}} w_q \|J_{i,q} - J_{i-1,q}\|_2,
$$

where `H` contains wrists, hands, and finger joints.

Then:

$$
s_i^{hand} =
\frac{\sum_{r=1}^{i} d_r^{hand}}
{\sum_{r=1}^{T-1} d_r^{hand} + \epsilon},
\quad
\tau_i^{hand} = 2s_i^{hand} - 1.
$$

This should be tested because latent arc length only slightly helped the previous latent experiment, while hand-aware physical arc length may be more appropriate for signing.

### 5.4 Physical-time derivative scaling

If the physical sequence duration is:

$$
T_{sec} = \frac{T-1}{fps},
$$

and the field uses:

$$
\tau \in [-1, 1],
$$

then:

$$
\frac{d}{dt} = \frac{2}{T_{sec}} \frac{d}{d\tau}.
$$

Therefore:

$$
\dot{J}(t) = \frac{2}{T_{sec}} \nabla_{\tau} J(\tau),
$$

$$
\ddot{J}(t) = \left(\frac{2}{T_{sec}}\right)^2 \nabla_{\tau}^{2} J(\tau),
$$

$$
\dddot{J}(t) = \left(\frac{2}{T_{sec}}\right)^3 \nabla_{\tau}^{3} J(\tau).
$$

This scaling is important if comparing velocity, acceleration, or jerk across different sequence lengths.

---

## 6. Model Families

The experiment should compare simple interpolation baselines, direct neural fields, and anchored residual fields.

### 6.1 Baseline B1: linear interpolation in rot6D feature space

For a query time `tau`, interpolate between neighboring observed frames:

$$
\hat{x}(\tau) = \operatorname{LinearInterp}(\{(\tau_i, x_i)\}_{i \in \Omega_{train}}, \tau).
$$

This is simple and strong for short gaps, but it may be geometrically invalid before rot6D projection.

### 6.2 Baseline B2: rotation interpolation with SLERP

For each joint, interpolate on `SO(3)`:

$$
\hat{R}_j(\tau) = \operatorname{SLERP}(R_{p,j}, R_{q,j}, \alpha),
$$

where `p` and `q` are the neighboring anchor frames and `alpha` is the local interpolation coefficient.

Expression coefficients can be linearly interpolated:

$$
\hat{e}(\tau) = (1-\alpha)e_p + \alpha e_q.
$$

This is a useful geometry-aware baseline.

### 6.3 Baseline B3: cubic interpolation

Use cubic interpolation for rot6D features or for expression coefficients. For rotations, prefer a rotation-aware version if available, such as Squad or cubic interpolation in the Lie algebra.

### 6.4 Model F1: direct SIREN pose field

A direct SIREN predicts all SMPL-X pose features from time:

$$
\hat{x}(\tau) = f_{\theta}(\tau).
$$

Layer form:

$$
h^{(0)} = \tau,
$$

$$
h^{(l)} = \sin(\omega_0(W^{(l)}h^{(l-1)} + b^{(l)})),
\quad
l = 1, \ldots, L,
$$

$$
\hat{x}(\tau) = W_{out}h^{(L)} + b_{out}.
$$

Recommended first setting:

```yaml
model: direct_siren
hidden_dim: 256
depth: 3
omega0_first: 30
omega0_hidden: 1
output_dim: D_x
```

### 6.5 Model F2: anchored residual SIREN pose field

This is the recommended main model for oracle fitting:

$$
\hat{x}(\tau) = A(\tau) + \alpha r_{\theta}(\tau),
$$

where:

- `A(tau)` is an interpolation scaffold built from the observed anchor frames.
- `r_theta(tau)` is a SIREN residual.
- `alpha` is a residual scale, initialized small or learned.

For example:

$$
A(\tau) = \operatorname{LinearInterp}(\{(\tau_i, x_i)\}_{i \in \Omega_{train}}, \tau).
$$

Then:

$$
\hat{x}(\tau) = A(\tau) + r_{\theta}(\tau).
$$

Recommended first setting:

```yaml
model: residual_siren_rot6d
hidden_dim: 256
depth: 3
omega0_first: 20
omega0_hidden: 1
residual_init: zero
residual_scale_init: 0.1
anchor_scaffold: linear_rot6d
```

This is analogous to the previous latent-space A5 residual SIREN, but now the residual is trained with physical 3D joint supervision.

### 6.6 Model F3: piecewise residual SIREN

For longer sign sentences, a single global field may shortcut the path. Use local fields over temporal windows:

$$
\hat{x}(\tau) = \sum_{m=1}^{M} w_m(\tau) \left[A_m(\tau) + r_{\theta_m}(\tau)\right],
$$

where `w_m(tau)` are smooth window weights.

Use this only after F2 is evaluated.

### 6.7 Diagnostic F4: joint-only field

Predict 3D joints directly:

$$
\hat{J}(\tau) = g_{\theta}(\tau).
$$

This is not the final model because it may violate SMPL-X kinematics, but it is useful as an upper-bound diagnostic. If joint-only interpolation is easy but rot6D+FK interpolation is hard, the bottleneck is rotation representation or FK constraints.

---

## 7. Forward Kinematics Pipeline

For each query `tau`:

1. Predict rot6D pose and expression:

$$
\hat{x}(\tau) = [\hat{u}_{1:N_R}(\tau), \hat{e}(\tau)].
$$

2. Convert each 6D rotation to a rotation matrix:

$$
\hat{R}_{j}(\tau) = \operatorname{Rot6DToMat}(\hat{u}_{j}(\tau)).
$$

3. Construct full SMPL-X pose parameters.

If the compact representation contains only upper body, hands, jaw, and expression, fill missing parts as neutral:

```text
root/global orientation: dataset choice or neutral
translation: zero or dataset choice
lower body: neutral
shape beta: fixed mean or subject-specific if available
upper body: predicted
hands: predicted
jaw: predicted
expression: predicted
```

4. Run SMPL-X forward kinematics:

$$
\hat{J}(\tau) = FK_{\text{SMPL-X}}(\hat{R}_{1:N_R}(\tau), \hat{e}(\tau), \beta, \Pi).
$$

5. Normalize joints for loss.

Recommended root-relative normalization:

$$
\bar{J}_{i,q} = J_{i,q} - J_{i,root}.
$$

For upper-body signing, `root` can be pelvis, spine, neck, or shoulder center. A shoulder-center root is often stable:

$$
J_{i,root} = \frac{1}{2}(J_{i,left\_shoulder} + J_{i,right\_shoulder}).
$$

Use the same root normalization for predictions and targets.

---

## 8. Training Modes

Let:

```text
Omega_all = all frame indices {0, ..., T-1}
Omega_train = frames used for fitting
Omega_eval = frames used for evaluation
```

### 8.1 fit_all

```text
Omega_train = all frames
Omega_eval = all frames
```

Purpose:

- Tests capacity.
- Checks that the model can reproduce observed SMPL-X motion.
- Not sufficient to prove interpolation.

### 8.2 even_odd

Two versions should be run:

```text
Version A:
Omega_train = even frames
Omega_eval = odd frames

Version B:
Omega_train = odd frames
Omega_eval = even frames
```

Purpose:

- Tests frame-level interpolation.
- Directly comparable to the previous latent-field even-odd result.

### 8.3 stride_4

```text
Omega_train = {0, 4, 8, ...}
Omega_eval = all other frames
```

Purpose:

- Tests sparse-anchor interpolation.
- Important because future generation may produce coarse anchors and query dense motion.

### 8.4 random_drop_30

```text
Randomly hold out 30% of frames.
```

Purpose:

- More realistic missing-frame interpolation test.
- Repeat with 3 to 5 random seeds.

### 8.5 span_drop

Hold out one contiguous temporal segment:

```text
Omega_eval = frames in [a, b]
Omega_train = all other frames
```

Example settings:

```text
span length = 10% of sequence
span length = 20% of sequence
```

Purpose:

- Tests whether the field can bridge missing transitions.
- Useful for detecting shortcutting.

### 8.6 dense_query

Train on all original frames, then query at higher temporal resolution:

$$
\tau_k^{2x} = -1 + \frac{2k}{2T-1},
\quad
k = 0, \ldots, 2T-1.
$$

and:

$$
\tau_k^{4x} = -1 + \frac{2k}{4T-1},
\quad
k = 0, \ldots, 4T-1.
$$

Purpose:

- Tests arbitrary-resolution synthesis.
- No direct ground truth may be available; evaluate by smoothness, decimation consistency, and visual rendering.

---

## 9. Loss Functions

The recommended training objective is:

$$
\mathcal{L}_{total}
=
\lambda_{6D}\mathcal{L}_{6D}
+
\lambda_{geo}\mathcal{L}_{geo}
+
\lambda_{joint}\mathcal{L}_{joint}
+
\lambda_{expr}\mathcal{L}_{expr}
+
\lambda_{vel}\mathcal{L}_{vel}
+
\lambda_{acc}\mathcal{L}_{acc}
+
\lambda_{jerk}\mathcal{L}_{jerk}
+
\lambda_{res}\mathcal{L}_{res}.
$$

Start simple and add terms gradually.

---

### 9.1 rot6D feature loss

For supervised frames `i in Omega_train`:

$$
\mathcal{L}_{6D}
=
\frac{1}{|\Omega_{train}|N_R}
\sum_{i \in \Omega_{train}}
\sum_{j=1}^{N_R}
\|\hat{u}_{i,j} - u_{i,j}\|_1.
$$

This anchors the network to the original SMPL-X representation.

---

### 9.2 Rotation geodesic loss

Convert predicted and target rotations to matrices:

$$
\hat{R}_{i,j}, R_{i,j} \in SO(3).
$$

The geodesic distance is:

$$
d_{SO(3)}(\hat{R}, R)
=
\arccos
\left(
\operatorname{clip}
\left(
\frac{\operatorname{tr}(\hat{R}^{\top}R)-1}{2},
-1+\epsilon,
1-\epsilon
\right)
\right).
$$

Then:

$$
\mathcal{L}_{geo}
=
\frac{1}{|\Omega_{train}|N_R}
\sum_{i \in \Omega_{train}}
\sum_{j=1}^{N_R}
 d_{SO(3)}(\hat{R}_{i,j}, R_{i,j}).
$$

This is more geometrically meaningful than raw rot6D L1.

---

### 9.3 3D joint loss through SMPL-X FK

Let:

$$
\hat{J}_{i,q}, J_{i,q} \in \mathbb{R}^{3}
$$

be the predicted and target 3D positions for joint `q`.

Use a weighted joint loss:

$$
\mathcal{L}_{joint}
=
\frac{1}{|\Omega_{train}|}
\sum_{i \in \Omega_{train}}
\sum_{q \in \mathcal{Q}}
 w_q \|\hat{J}_{i,q} - J_{i,q}\|_1.
$$

Recommended weights:

```text
hands/fingers: 3.0 to 5.0
wrists:        3.0
elbows:        2.0
head/neck:     2.0
torso:         1.0
face/jaw landmarks, if available: 2.0 to 4.0
```

For sign language, hand/finger joints should receive the highest weight.

---

### 9.4 Expression loss

If expression coefficients are part of the compact representation:

$$
\mathcal{L}_{expr}
=
\frac{1}{|\Omega_{train}|}
\sum_{i \in \Omega_{train}}
\|\hat{e}_{i} - e_i\|_1.
$$

This term matters because non-manual cues are important for signing.

---

### 9.5 Joint velocity loss

Compute target velocity by finite differences on target joints:

$$
v_{i,q}^{gt}
=
\frac{J_{i+1,q}-J_{i-1,q}}{2\Delta t}.
$$

Compute predicted velocity analytically with autograd:

$$
\hat{v}_{q}(\tau_i)
=
\frac{2}{T_{sec}}
\frac{d\hat{J}_{q}(\tau)}{d\tau}\Bigg|_{\tau=\tau_i}.
$$

Then:

$$
\mathcal{L}_{vel}
=
\frac{1}{|\Omega_{train}|}
\sum_{i \in \Omega_{train}}
\sum_{q \in \mathcal{Q}}
 w_q
\|\hat{v}_{i,q} - v_{i,q}^{gt}\|_1.
$$

If analytic FK derivatives are too slow, use dense finite differences on predicted joints during the first implementation.

---

### 9.6 Joint acceleration loss

Target acceleration:

$$
a_{i,q}^{gt}
=
\frac{J_{i+1,q} - 2J_{i,q} + J_{i-1,q}}{\Delta t^2}.
$$

Predicted acceleration:

$$
\hat{a}_{q}(\tau_i)
=
\left(\frac{2}{T_{sec}}\right)^2
\frac{d^2\hat{J}_{q}(\tau)}{d\tau^2}\Bigg|_{\tau=\tau_i}.
$$

Loss:

$$
\mathcal{L}_{acc}
=
\frac{1}{|\Omega_{train}|}
\sum_{i \in \Omega_{train}}
\sum_{q \in \mathcal{Q}}
 w_q
\|\hat{a}_{i,q} - a_{i,q}^{gt}\|_1.
$$

Use acceleration loss only after position and velocity losses are stable.

---

### 9.7 Jerk regularization

Predicted jerk:

$$
\hat{j}_{q}(\tau_i)
=
\left(\frac{2}{T_{sec}}\right)^3
\frac{d^3\hat{J}_{q}(\tau)}{d\tau^3}\Bigg|_{\tau=\tau_i}.
$$

Jerk regularization:

$$
\mathcal{L}_{jerk}
=
\frac{1}{|\Omega_{train}|}
\sum_{i \in \Omega_{train}}
\sum_{q \in \mathcal{Q}}
 w_q
\|\hat{j}_{i,q}\|_1.
$$

Use a small weight. Too much jerk regularization can erase meaningful fast handshape transitions.

---

### 9.8 Residual magnitude regularization

For residual SIREN:

$$
\hat{x}(\tau) = A(\tau) + r_{\theta}(\tau),
$$

regularize the residual:

$$
\mathcal{L}_{res}
=
\frac{1}{|\Omega_{train}|}
\sum_{i \in \Omega_{train}}
\|r_{\theta}(\tau_i)\|_2^2.
$$

This prevents the residual from destroying the anchor scaffold early in training.

---

### 9.9 Optional angular velocity loss

For each rotation matrix, angular velocity can be estimated from:

$$
\Omega_j(\tau) = R_j(\tau)^\top \frac{dR_j(\tau)}{dt}.
$$

The vector form is:

$$
\omega_j(\tau) = \operatorname{vee}(\Omega_j(\tau)).
$$

Then supervise:

$$
\mathcal{L}_{angvel}
=
\frac{1}{|\Omega_{train}|N_R}
\sum_{i \in \Omega_{train}}
\sum_{j=1}^{N_R}
\|\hat{\omega}_{i,j} - \omega^{gt}_{i,j}\|_1.
$$

This is optional and should be added only after the main joint-space pipeline works.

---

## 10. Recommended Loss Schedules

### 10.1 Schedule S1: position-only warmup

Use for the first sanity check:

$$
\mathcal{L}_{S1}
=
\mathcal{L}_{6D}
+
\lambda_{joint}\mathcal{L}_{joint}
+
\lambda_{expr}\mathcal{L}_{expr}.
$$

Initial weights:

```yaml
lambda_6D: 1.0
lambda_joint: 10.0
lambda_expr: 1.0
```

Adjust `lambda_joint` depending on joint units. If joints are in meters, `10.0` to `100.0` may be reasonable. If joints are in millimeters, use a smaller value.

### 10.2 Schedule S2: add geodesic loss

$$
\mathcal{L}_{S2}
=
\mathcal{L}_{S1}
+
\lambda_{geo}\mathcal{L}_{geo}.
$$

Initial weight:

```yaml
lambda_geo: 0.1
```

### 10.3 Schedule S3: add velocity

$$
\mathcal{L}_{S3}
=
\mathcal{L}_{S2}
+
\lambda_{vel}\mathcal{L}_{vel}.
$$

Initial weight:

```yaml
lambda_vel: 0.1
```

### 10.4 Schedule S4: add jerk regularization

$$
\mathcal{L}_{S4}
=
\mathcal{L}_{S3}
+
\lambda_{jerk}\mathcal{L}_{jerk}.
$$

Initial weight:

```yaml
lambda_jerk: 1.0e-4
```

Ramp `lambda_jerk` from zero after the first 20% of optimization steps.

### 10.5 Schedule S5: full dynamics

$$
\mathcal{L}_{S5}
=
\mathcal{L}_{S4}
+
\lambda_{acc}\mathcal{L}_{acc}.
$$

Initial weight:

```yaml
lambda_acc: 0.01
```

Use this only after S3/S4 are stable.

---

## 11. Evaluation Metrics

### 11.1 Feature-space metrics

rot6D L1:

$$
E_{6D}
=
\frac{1}{|\Omega_{eval}|N_R}
\sum_{i \in \Omega_{eval}}
\sum_{j=1}^{N_R}
\|\hat{u}_{i,j} - u_{i,j}\|_1.
$$

Expression L1:

$$
E_{expr}
=
\frac{1}{|\Omega_{eval}|}
\sum_{i \in \Omega_{eval}}
\|\hat{e}_i - e_i\|_1.
$$

### 11.2 Rotation metrics

Mean geodesic error:

$$
E_{geo}
=
\frac{1}{|\Omega_{eval}|N_R}
\sum_{i \in \Omega_{eval}}
\sum_{j=1}^{N_R}
 d_{SO(3)}(\hat{R}_{i,j}, R_{i,j}).
$$

Report in radians and degrees:

$$
E_{geo}^{deg} = \frac{180}{\pi} E_{geo}.
$$

### 11.3 3D joint metrics

Mean joint position error:

$$
E_{JPE}
=
\frac{1}{|\Omega_{eval}|}
\sum_{i \in \Omega_{eval}}
\sum_{q \in \mathcal{Q}}
 w_q \|\hat{J}_{i,q} - J_{i,q}\|_2.
$$

Report separately:

```text
body JPE
left-hand JPE
right-hand JPE
both-hands JPE
head/face landmark error, if available
all upper-body JPE
```

### 11.4 Temporal dynamics metrics

Velocity error:

$$
E_{vel}
=
\frac{1}{|\Omega_{eval}|}
\sum_{i \in \Omega_{eval}}
\sum_{q \in \mathcal{Q}}
 w_q \|\hat{v}_{i,q} - v^{gt}_{i,q}\|_2.
$$

Acceleration error:

$$
E_{acc}
=
\frac{1}{|\Omega_{eval}|}
\sum_{i \in \Omega_{eval}}
\sum_{q \in \mathcal{Q}}
 w_q \|\hat{a}_{i,q} - a^{gt}_{i,q}\|_2.
$$

Average jerk magnitude:

$$
M_{jerk}
=
\frac{1}{|\Omega_{eval}|}
\sum_{i \in \Omega_{eval}}
\sum_{q \in \mathcal{Q}}
 w_q \|\hat{j}_{i,q}\|_2.
$$

### 11.5 Path-length ratio

Over-smoothing often appears as path shortcutting. Track the path-length ratio:

$$
PLR
=
\frac{
\sum_{i=1}^{T-1} \sum_{q \in \mathcal{Q}} w_q \|\hat{J}_{i,q} - \hat{J}_{i-1,q}\|_2
}{
\sum_{i=1}^{T-1} \sum_{q \in \mathcal{Q}} w_q \|J_{i,q} - J_{i-1,q}\|_2 + \epsilon
}.
$$

Interpretation:

```text
PLR close to 1: good path length preservation
PLR much smaller than 1: over-smoothed shortcutting
PLR much larger than 1: jitter or overactive motion
```

### 11.6 R2 score in joint space

For held-out frames:

$$
R^2_J
=
1 -
\frac{
\sum_{i \in \Omega_{eval}} \|\hat{J}_i - J_i\|_2^2
}{
\sum_{i \in \Omega_{eval}} \|J_i - \bar{J}\|_2^2 + \epsilon
}.
$$

The previous latent-field even-odd experiment had negative R2 in latent space. A useful real-space field should achieve positive R2 on held-out 3D joints.

### 11.7 Dense-query consistency

For dense queries, evaluate decimation consistency. If the field is queried at 2x resolution and then decimated back to original frame times, the result should match the original query:

$$
E_{decimate}
=
\frac{1}{T}
\sum_{i=0}^{T-1}
\|\hat{J}^{2x}(\tau_i) - \hat{J}^{1x}(\tau_i)\|_1.
$$

Also render 2x and 4x sequences and inspect hand/finger stability.

---

## 12. Baseline Comparison Table

Run the following methods:

| ID | Method | Output | Purpose |
|---|---|---|---|
| B1 | Linear rot6D interpolation | rot6D + expression | Basic interpolation baseline |
| B2 | SLERP rotations + linear expression | SO(3) + expression | Geometry-aware interpolation baseline |
| B3 | Cubic / Squad interpolation | SO(3) or rot6D | Smooth interpolation baseline |
| B4 | B-spline control points | rot6D or SO(3) | Tests low-pass smoothing behavior |
| F1 | Direct SIREN | rot6D + expression | Pure continuous field |
| F2 | Residual SIREN with rot6D scaffold | rot6D + expression | Main proposed oracle field |
| F3 | Piecewise residual SIREN | rot6D + expression | Long-sequence variant |
| F4 | Joint-only SIREN diagnostic | 3D joints | Upper-bound diagnostic, not final output |
| L0 | Previous latent-field A5 | VAE latent | Reference from current experiment |

---

## 13. Experiment Grid

### 13.1 Stage 0: single-sequence debugging

Use 1 short sequence, 1 medium sequence, and 1 high-motion sequence.

```yaml
num_sequences: 3
fit_modes: [fit_all, even_odd]
time_modes: [uniform]
models: [B1, B2, F1, F2]
loss_schedules: [S1]
steps: 1000
```

Pass condition:

```text
fit_all should visually and numerically match the GT pose.
even_odd should not collapse to an over-smoothed path.
SMPL-X FK should be stable and differentiable.
```

### 13.2 Stage 1: 10-sequence pilot

```yaml
num_sequences: 10
fit_modes: [fit_all, even_odd, stride_4]
time_modes: [uniform, joint_arclength, hand_arclength]
models: [B1, B2, B3, F1, F2]
loss_schedules: [S1, S2, S3]
steps: 2000
batch_points: 128
```

Goal:

```text
Determine whether rot6D+FK fields interpolate better than latent fields.
Identify best time parameterization and loss schedule.
```

### 13.3 Stage 2: 50-100 sequence pilot

```yaml
num_sequences: 50_to_100
fit_modes: [fit_all, even_odd, stride_4, random_drop_30, span_drop]
time_modes: [uniform, hand_arclength]
models: [B1, B2, F2]
loss_schedules: [S1, S3, S4]
steps: 3000
batch_points: 256
random_seeds: [0, 1, 2]
```

Goal:

```text
Produce report-level evidence.
Check robustness across sequence length and motion intensity.
```

### 13.4 Stage 3: dense-query and visual analysis

For the best F2 configuration:

```yaml
dense_query_rates: [2, 4]
render_samples_per_bucket: 5
buckets:
  - short_low_motion
  - short_high_motion
  - medium
  - long
  - high_hand_motion
  - high_expression_motion
```

Render:

```text
GT original
linear interpolation baseline
SLERP baseline
F2 predicted original-rate
F2 dense 2x
F2 dense 4x
```

---

## 14. Suggested Hyperparameters

### 14.1 SIREN architecture

```yaml
siren:
  hidden_dim: 256
  depth: 3
  omega0_first: 20
  omega0_hidden: 1
  activation: sine
  final_activation: none
  output_dim: 256  # verify actual D_x
```

### 14.2 Residual SIREN

```yaml
residual_siren:
  scaffold: linear_rot6d
  residual_init: zero
  residual_scale_init: 0.1
  residual_scale_learnable: true
  hidden_dim: 256
  depth: 3
```

### 14.3 Optimization

```yaml
optimizer: AdamW
learning_rate: 1.0e-4
weight_decay: 1.0e-4
steps_stage0: 1000
steps_stage1: 2000
steps_stage2: 3000
batch_points: 128_to_256
grad_clip_norm: 1.0
mixed_precision: false
```

For per-sequence oracle fitting, mixed precision is not necessary and may make geometry operations less stable.

### 14.4 Loss weights, first pass

```yaml
loss_weights:
  lambda_6D: 1.0
  lambda_geo: 0.1
  lambda_joint: 10.0
  lambda_expr: 1.0
  lambda_vel: 0.1
  lambda_acc: 0.01
  lambda_jerk: 1.0e-4
  lambda_res: 1.0e-4
```

These are starting points. After the first 10-sequence pilot, rebalance using gradient magnitudes and unit scales.

---

## 15. Implementation Blueprint

This next experiment should be implemented inside the same top-level `NIAF`
namespace as the refactored oracle latent-field experiment.

### 15.1 Directory structure

```text
NIAF/oracle_smplx_field/
  configs/
    how2sign_rot6d_fk_pilot.yaml
  scripts/
    fit_oracle_smplx_field.py
    export_smplx_field_viz.py
    summarize_oracle_smplx_field.py
  models/
    siren.py
    residual_siren.py
    interpolation.py
  geometry/
    rotation_6d.py
    smplx_fk.py
    joint_metrics.py
  losses/
    pose_losses.py
    dynamics_losses.py
  visualization/
    plot_joint_trajectory.py
    plot_velocity_curves.py
    render_smplx_sequence.py
```

### 15.2 Core pseudocode

```python
# Load GT compact SMPL-X sequence
axis_angle, expr, mask = load_sequence(sample_id)

# Convert target rotations
R_gt = axis_angle_to_matrix(axis_angle)      # [T, N_R, 3, 3]
u_gt = matrix_to_rot6d(R_gt)                # [T, N_R, 6]
x_gt = concat(u_gt.flatten(1), expr)        # [T, D_x]

# Compute target joints once
J_gt = smplx_fk(R_gt, expr, neutral_params) # [T, N_J, 3]
J_gt = root_relative(J_gt)

# Build time coordinates
tau = build_tau(T, mode="uniform")          # [T, 1]

# Split train/eval frames
train_idx, eval_idx = make_split(T, fit_mode="even_odd")

# Build model
model = ResidualSiren(scaffold=LinearInterp(tau[train_idx], x_gt[train_idx]))

for step in range(num_steps):
    idx = sample_points(train_idx, batch_points)
    tau_b = tau[idx].requires_grad_(True)

    # Predict rot6D + expression
    x_hat = model(tau_b)
    u_hat, expr_hat = split_pose_expr(x_hat)

    # Project rot6D to rotation matrices
    R_hat = rot6d_to_matrix(u_hat)

    # Forward kinematics
    J_hat = smplx_fk(R_hat, expr_hat, neutral_params)
    J_hat = root_relative(J_hat)

    # Compute losses
    loss = loss_6d(u_hat, u_gt[idx])
    loss += lambda_geo * loss_geodesic(R_hat, R_gt[idx])
    loss += lambda_joint * loss_joint(J_hat, J_gt[idx])
    loss += lambda_expr * loss_expr(expr_hat, expr[idx])

    if use_velocity:
        v_hat = analytic_or_finite_velocity(model, tau_b)
        loss += lambda_vel * loss_velocity(v_hat, v_gt[idx])

    if use_jerk:
        jerk_hat = analytic_or_finite_jerk(model, tau_b)
        loss += lambda_jerk * jerk_regularizer(jerk_hat)

    loss.backward()
    clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
    optimizer.zero_grad()

# Evaluate on train, held-out, and dense queries
metrics = evaluate(model, tau, x_gt, R_gt, J_gt, train_idx, eval_idx)
save_npz_and_render(metrics, model_outputs)
```

### 15.3 Practical FK note

If the SMPL-X layer accepts only axis-angle, convert predicted rotation matrices to axis-angle after rot6D projection:

$$
\hat{r}^{aa}_{j} = \operatorname{MatToAxisAngle}(\hat{R}_{j}).
$$

However, for stable gradients, it is preferable to use an FK implementation that can accept rotation matrices directly.

---

## 16. Visualizations

For each reported sample, export the following:

### 16.1 3D hand trajectory plot

Plot left and right wrist/fingertip trajectories in 3D or PCA-2D:

```text
GT trajectory
linear / SLERP baseline
residual SIREN field
held-out frame markers
start/end markers
```

### 16.2 Adjacent joint-step distance

For hands:

$$
d_i^{hand}
=
\sum_{q \in \mathcal{H}} w_q \|J_{i,q} - J_{i-1,q}\|_2.
$$

Plot GT versus prediction. This directly checks whether the model is shortcutting.

### 16.3 Velocity curves

Plot hand/wrist velocity magnitudes:

$$
\|v_{i,left\_wrist}\|_2,
\quad
\|v_{i,right\_wrist}\|_2.
$$

Compare GT, baseline, and field.

### 16.4 High-error frame rendering

For the top-k held-out errors, render:

```text
previous observed frame
held-out GT frame
field prediction
next observed frame
SLERP prediction
linear rot6D prediction
```

This identifies whether errors are from:

- hand location;
- finger articulation;
- wrist orientation;
- head/jaw/facial expression;
- global body posture;
- over-smoothed transition;
- physically implausible rotation.

### 16.5 Dense-query rendering

Render 1x, 2x, and 4x query sequences. Check:

```text
hand jitter
finger jitter
jaw/facial flicker
unwanted body drift
smoothness of holds
preservation of fast transitions
```

---

## 17. Success Criteria

The experiment supports moving to a real-space continuous sign field if most of the following are true:

### 17.1 Fit-all capacity

```text
F2 fit_all should reconstruct observed frames nearly exactly.
```

Expected:

- low rot6D L1;
- low geodesic error;
- low hand/body JPE;
- no visual artifacts.

### 17.2 Held-out interpolation

For even-odd and stride-4:

```text
F2 should outperform the previous latent-field interpolation result after decoding.
F2 should outperform or match linear/SLERP baselines on hand JPE and temporal metrics.
```

Minimum desired signals:

```text
held-out joint-space R2 > 0
path-length ratio close to 1
hand JPE lower than linear rot6D interpolation
no obvious shortcutting in high-motion samples
```

### 17.3 Dynamics

Velocity and jerk curves should show:

```text
less jitter than frame-wise baselines
less shortcutting than over-smoothed interpolation
preserved velocity peaks for meaningful fast signs
```

### 17.4 Dense querying

At 2x and 4x query rates:

```text
dense output should be smooth and visually plausible
original-rate decimation should remain close to 1x output
no facial or finger flickering
```

---

## 18. Failure Cases and Interpretation

### Case A: fit_all succeeds, even_odd fails

Interpretation:

```text
The field can memorize the pose sequence, but interpolation is still hard.
```

Next action:

- try hand-arc-length time;
- try SLERP scaffold instead of linear rot6D scaffold;
- add joint velocity loss;
- use piecewise fields;
- inspect high-error frames.

### Case B: linear/SLERP beats SIREN on interpolation

Interpretation:

```text
The neural field may be overfitting or under-regularized.
```

Next action:

- reduce SIREN frequency `omega0`;
- increase residual regularization;
- train fewer steps;
- use scaffold-only + small residual;
- add validation early stopping on held-out frames.

### Case C: pose field is smooth but handshape is wrong

Interpretation:

```text
Joint position losses alone are insufficient for sign articulation.
```

Next action:

- increase finger rotation geodesic weight;
- add fingertip and finger joint weights;
- add handshape-specific rotation loss;
- inspect MANO hand joints separately.

### Case D: rot6D loss is low but joint loss is high

Interpretation:

```text
SMPL-X mapping or joint normalization may be inconsistent.
```

Next action:

- verify joint order;
- verify neutral lower-body/root settings;
- verify expression and jaw handling;
- compare FK of GT rot6D against stored GT joints;
- check units and root normalization.

### Case E: joint loss is low but rotations look unnatural

Interpretation:

```text
Joint positions do not fully constrain rotation and hand orientation.
```

Next action:

- increase geodesic loss;
- add orientation loss for wrists and fingers;
- add palm normal loss;
- render mesh, not only joints.

---

## 19. Connection to Future Text-Conditioned Model

If the oracle experiment is successful, build the actual generation model as an anchored field:

$$
\hat{x}(\tau; y, \mathcal{M})
=
A(\tau; y, \mathcal{M}) + r_{\theta}(\tau; y, \mathcal{M}),
$$

where:

- `y` is the input sentence;
- `M` is the unordered retrieved word-motion memory;
- `A(tau; y, M)` is a coarse path predicted by SoftArrangerFlow or a new soft arranger in pose space;
- `r_theta` is a continuous residual pose field.

A future text-conditioned architecture can use one of these designs:

### Option 1: Query-conditioned Transformer field

For each query `tau`, create:

$$
q(\tau) = P_y h_y + \gamma(\tau),
$$

then attend to lexical memory and output pose features:

$$
\hat{x}(\tau) = Decoder(q(\tau), \mathcal{M}).
$$

### Option 2: Hypernetwork-modulated SIREN

Predict SIREN modulation parameters from text and memory:

$$
\theta(y, \mathcal{M}) = H_{\psi}(y, \mathcal{M}),
$$

then:

$$
\hat{x}(\tau) = \Phi(\tau; \theta(y, \mathcal{M})).
$$

### Option 3: Adapter-residual pose field

Use current SoftArrangerFlow to produce a coarse sequence, interpolate it into a continuous scaffold, and predict a residual:

$$
A(\tau; y, \mathcal{M}) = \operatorname{Interp}(\text{SoftArrangerFlowPrior}(y, \mathcal{M})),
$$

$$
\hat{x}(\tau; y, \mathcal{M}) = A(\tau; y, \mathcal{M}) + r_{\theta}(\tau; y, \mathcal{M}).
$$

This is the safest transition from the current system.

---

## 20. Duration and Length Handling

A continuous field removes the need to generate a fixed sequence of discrete waypoints internally, but it does **not** remove duration modeling.

At inference, the model still needs one of:

```text
1. duration head: predict total duration, then sample the field;
2. EOS head: query until stop probability crosses a threshold;
3. time-warp field: predict speed over normalized sign progress.
```

For the oracle experiment, use ground-truth duration. For the future generator, the safest design is:

$$
D_{sec} = h_{dur}(y, \mathcal{M}),
$$

then sample:

$$
K = \lceil D_{sec} \cdot fps \rceil,
$$

$$
\tau_k = -1 + \frac{2k}{K-1},
\quad
k = 0, \ldots, K-1.
$$

A more advanced version predicts a time-warp:

$$
\frac{dt}{ds} = \operatorname{softplus}(r_{\psi}(s; y, \mathcal{M})),
$$

with total duration:

$$
D_{sec} = \int_0^1 \frac{dt}{ds} ds.
$$

---

## 21. Recommended Report Template

For each stage, report:

```text
1. dataset split and number of sequences
2. sequence length statistics
3. model configurations
4. time parameterizations
5. fit modes
6. loss schedules
7. aggregate metrics
8. per-length bucket metrics
9. per-motion-intensity bucket metrics
10. visual examples
11. failure analysis
```

### 21.1 Main result table

| Method | Time mode | Fit mode | rot6D L1 | Geo deg | Body JPE | Hand JPE | Vel err | PLR | Joint R2 |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| Linear rot6D | uniform | even_odd | | | | | | | |
| SLERP | uniform | even_odd | | | | | | | |
| Direct SIREN | uniform | even_odd | | | | | | | |
| Residual SIREN | uniform | even_odd | | | | | | | |
| Residual SIREN | hand arc | even_odd | | | | | | | |

### 21.2 Fit-all table

| Method | Time mode | rot6D L1 | Geo deg | Body JPE | Hand JPE | Expr L1 | Visual pass |
|---|---|---:|---:|---:|---:|---:|---|
| Direct SIREN | uniform | | | | | | |
| Residual SIREN | uniform | | | | | | |

### 21.3 Dense-query table

| Method | Query rate | Decimation err | Mean jerk | Hand jitter score | Visual pass |
|---|---:|---:|---:|---:|---|
| Residual SIREN | 1x | | | | |
| Residual SIREN | 2x | | | | |
| Residual SIREN | 4x | | | | |

---

## 22. Minimal First Run

Use this as the first command target once implemented:

```bash
python -m NIAF.oracle_smplx_field.scripts.fit_oracle_smplx_field \
  --config NIAF/oracle_smplx_field/configs/how2sign_rot6d_fk_pilot.yaml \
  --out_dir experiments/NIAF/oracle_smplx_field/how2sign_stage0 \
  --max_sequences 3 \
  --models linear_rot6d,slerp,residual_siren \
  --time_modes uniform \
  --fit_modes fit_all,even_odd \
  --loss_schedules S1 \
  --steps 1000 \
  --batch_points 128 \
  --save_npz \
  --render
```

Then run the 10-sequence pilot:

```bash
python -m NIAF.oracle_smplx_field.scripts.fit_oracle_smplx_field \
  --config NIAF/oracle_smplx_field/configs/how2sign_rot6d_fk_pilot.yaml \
  --out_dir experiments/NIAF/oracle_smplx_field/how2sign_stage1 \
  --max_sequences 10 \
  --models linear_rot6d,slerp,cubic,direct_siren,residual_siren \
  --time_modes uniform,joint_arclength,hand_arclength \
  --fit_modes fit_all,even_odd,stride_4 \
  --loss_schedules S1,S2,S3 \
  --steps 2000 \
  --batch_points 128 \
  --save_npz
```

---

## 23. Final Recommendation

The next step should not be a full text-conditioned real-space generator immediately. Run the oracle SMPL-X field test first.

Recommended order:

```text
1. Implement rot6D -> rotation matrix -> SMPL-X FK -> weighted joint loss.
2. Validate FK by reconstructing GT joints from GT rot6D.
3. Run fit_all and even_odd on 3 sequences.
4. Compare residual SIREN against linear rot6D and SLERP baselines.
5. Add hand-weighted arc-length time.
6. Add velocity loss in 3D joint space.
7. Run 50-100 sequence pilot.
8. If interpolation improves clearly over latent fields, build anchored text-conditioned SMPL-X pose field.
```

The most likely strong AAAI direction is:

```text
Continuous SMPL-X Sign Fields: retrieval-conditioned, manifold-aware pose-field generation with rot6D rotations and 3D joint dynamics supervision.
```

This would directly address the observed limitation of the current latent-field approach while preserving the strengths of SoftArrangerFlow: unordered lexical memory, soft candidate selection, and sentence-level naturalization.
