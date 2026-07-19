# Oracle Latent-Field Fitting Design for SoftArrangerFlow

## 0. Purpose

This document specifies an **oracle latent-field fitting experiment** for evaluating whether the current SoftArrangerFlow VAE latent space is suitable for continuous implicit trajectory modeling.

The experiment answers one central question:

> Given a ground-truth sentence latent trajectory $z_{1:L}$, can a compact continuous function $f_\theta(s)$ represent the trajectory accurately, interpolate between latent tokens safely, and decode into natural SMPL-X signing motion?

This is an **oracle** experiment because it uses the ground-truth latent sequence and ground-truth duration during fitting. It is not yet a deployable text-to-sign model. Its purpose is to test whether replacing discrete latent-slot prediction with an implicit latent sign field is technically viable.

---

## 1. Why this experiment is needed

Current SoftArrangerFlow operates in a frozen temporal VAE latent trajectory space:

$$
z_1 = \mathrm{norm}_z(E_\phi(x, M_T)), \qquad z_1 \in \mathbb{R}^{L \times d_z}, \qquad L = \lceil T/4 \rceil.
$$

At inference, the system predicts the output frame length $\hat T$, converts it to latent length $\hat L=\lceil \hat T/4 \rceil$, constructs a sentence-length word prior, and refines it with latent flow. This creates a strong dependence on the predicted discrete latent length.

The implicit-field idea is to replace a discrete latent sequence

$$
\{z_1, z_2, \ldots, z_L\}
$$

with a continuous latent trajectory

$$
z(s) = f_\theta(s), \qquad s \in [-1,1] \text{ or } s\in[0,1].
$$

This is inspired by Neural Implicit Action Fields, where a motion/action chunk is modeled as a continuous function $A(\tau)=\Phi(\tau;\theta)$ that can be queried at arbitrary temporal resolutions and analytically differentiated.

However, the visualization you produced shows that adjacent latent tokens may have large 256-D L2 jumps. Therefore, before building the full text-conditioned model, we should test whether the current VAE latent trajectories are actually representable as smooth continuous fields.

---

## 2. Core hypothesis

### Hypothesis H1: latent-field compatibility

A ground-truth latent trajectory $z_{1:L}$ produced by the frozen VAE can be approximated by a compact continuous field $f_\theta(s)$ such that:

1. latent reconstruction error is small;
2. decoded SMPL-X pose error is close to the frozen VAE reconstruction upper bound;
3. interpolation between latent tokens decodes to plausible signing motion;
4. dense temporal querying improves smoothness without erasing fast sign-relevant articulations.

### Hypothesis H0: latent-field incompatibility

The current VAE latent space is not smooth enough for continuous implicit modeling. In this case, even an oracle-fitted field either:

1. cannot fit the latent tokens well;
2. fits latent tokens but decodes to unnatural motion;
3. interpolates poorly between adjacent tokens;
4. over-smooths high-frequency handshape, wrist, jaw, or facial details.

If H0 is supported, the next step is to improve the VAE latent space before replacing the length predictor.

---

## 3. What “oracle” means here

For each ground-truth signing sequence, we fit a separate continuous latent field:

$$
f_\theta: s \mapsto \hat z(s), \qquad \hat z(s) \in \mathbb{R}^{d_z}.
$$

The field is optimized directly against the target latent tokens from the frozen VAE:

$$
z_i = z(s_i), \qquad i=1,\ldots,L.
$$

The experiment uses:

- ground-truth latent length $L$;
- ground-truth latent trajectory $z_{1:L}$;
- ground-truth frame sequence $x_{1:T}$ for decoded-pose evaluation;
- no text conditioning;
- no retrieval candidates;
- no length prediction;
- no flow matching.

Therefore, the oracle field gives an **upper bound** on what a future text-conditioned implicit latent field could achieve if it were able to predict the correct function parameters.

---

## 4. Dataset preparation

### 4.1 Input data

For each dataset, prepare a set of validation or training sequences:

$$
\mathcal{D}_{oracle}=\{(y^{(n)}, x^{(n)}, z^{(n)}, M_T^{(n)}, M_L^{(n)})\}_{n=1}^{N}.
$$

Where:

- $y$: sentence text, used only for logging;
- $x\in\mathbb{R}^{T\times D}$: compact rot6D SMPL-X motion;
- $z\in\mathbb{R}^{L\times d_z}$: normalized VAE latent trajectory;
- $M_T$: frame mask;
- $M_L$: latent-token mask;
- $T$: valid SMPL-X frame count;
- $L=\lceil T/4\rceil$: valid latent length.

### 4.2 Suggested sample sizes

Use three scales:

| Split | Number of sequences | Purpose |
|---|---:|---|
| debug | 5-10 | verify implementation and visualizations |
| pilot | 50-100 per dataset | choose architecture and losses |
| main | 300-1000 per dataset | report aggregate feasibility results |

For early debugging, choose sequences with diverse lengths:

- short: $L<25$;
- medium: $25\le L<60$;
- long: $L\ge60$.

### 4.3 Filtering

Recommended filters:

- remove sequences with fewer than 8 latent tokens;
- remove samples with severe SMPL-X extraction failure;
- keep a separate list of high-motion samples where adjacent latent-token distance is above the 95th percentile;
- keep both low-motion and high-motion examples for qualitative comparison.

---

## 5. Time parameterization

The field input can be parameterized in several ways. This is a key part of the experiment.

### 5.1 Uniform progress parameterization

The simplest option is:

$$
s_i = -1 + \frac{2(i-1)}{L-1}, \qquad i=1,\ldots,L.
$$

or equivalently $s_i\in[0,1]$:

$$
s_i = \frac{i-1}{L-1}.
$$

This treats each latent token as equally spaced in signing progress.

**Pros**

- simple;
- directly compatible with the current latent sequence;
- easiest to use for a future text-conditioned generator.

**Cons**

- does not account for uneven latent speed;
- fast transitions and holds receive equal parameter distance;
- may force the SIREN to represent sharp latent jumps in a small region.

### 5.2 Latent arc-length parameterization

Compute adjacent latent distances:

$$
d_i = \|z_i-z_{i-1}\|_2, \qquad i=2,\ldots,L.
$$

Then define cumulative normalized arc length:

$$
s_1 = 0,
$$

$$
s_i = \frac{\sum_{r=2}^{i} d_r}{\sum_{r=2}^{L} d_r + \epsilon}, \qquad i=2,\ldots,L.
$$

Optionally map to $[-1,1]$:

$$
\tilde s_i = 2s_i - 1.
$$

**Pros**

- gives more coordinate-space distance to large latent transitions;
- separates path geometry from tempo;
- useful for detecting whether difficulty comes from timing or trajectory shape.

**Cons**

- not directly available at inference unless a time-warp/speed field is predicted;
- can hide duration/tempo errors;
- may overfit to latent-space artifacts.

### 5.3 Pose arc-length parameterization

Compute arc length using decoded or ground-truth joint positions:

$$
d_i^{pose}=\|J(x_{t_i}) - J(x_{t_{i-1}})\|_2,
$$

where $J(\cdot)$ extracts body/hand joints and $t_i$ is the frame corresponding to latent token $i$. Then normalize cumulative distance as above.

**Pros**

- closer to physical motion;
- may better reflect meaningful hand/body movement;
- helps diagnose whether latent jumps correspond to real pose changes.

**Cons**

- depends on SMPL-X quality;
- can underrepresent non-manual signals if joint-only;
- may conflict with latent-space geometry.

### 5.4 Required comparison

Run all three parameterizations:

1. uniform progress;
2. latent arc length;
3. pose arc length.

If arc-length parameterization strongly improves fitting and interpolation, the final model should likely use a **content trajectory plus time-warp** design rather than a simple uniform field.

---

## 6. Model choices

### 6.1 Main model: SIREN latent field

The default oracle field is a SIREN:

$$
f_\theta(s)=W_{out}h^{(K)}(s)+b_{out},
$$

$$
h^{(0)}=s,
$$

$$
h^{(k)}=\sin\left(\omega_k(W_k h^{(k-1)}+b_k)\right), \qquad k=1,\ldots,K.
$$

Output:

$$
f_\theta(s)\in\mathbb{R}^{d_z}.
$$

Recommended default:

| Hyperparameter | Default |
|---|---:|
| input dimension | 1 |
| output dimension | $d_z$, e.g. 256 |
| hidden layers | 3 |
| hidden width | 256 |
| first-layer $\omega_0$ | 30 |
| hidden-layer $\omega$ | 1 or 10 |
| activation | sine |
| output activation | linear |
| optimizer | AdamW |
| learning rate | 1e-4 to 1e-3 |
| steps | 2k-10k per sequence |

### 6.2 Residual SIREN variant

A useful variant is residual fitting around a simple interpolant:

$$
f_\theta(s)=\mathrm{LinearInterp}(z_{1:L};s)+r_\theta(s).
$$

This answers a different question:

> Are the difficult parts mostly local residual corrections around a standard interpolation curve?

This variant often stabilizes optimization and makes jerk regularization safer.

### 6.3 Low-rank SIREN variant

For long sequences or high-dimensional latents, use a low-rank output basis:

$$
f_\theta(s)=\mu + Bc_\theta(s),
$$

where:

- $B\in\mathbb{R}^{d_z\times r}$ is a PCA basis fitted on latent trajectories;
- $c_\theta(s)\in\mathbb{R}^{r}$, with $r\in\{32,64,128\}$.

This tests whether most temporal structure lies in a lower-dimensional latent subspace.

### 6.4 Baselines

Compare SIREN against:

| Baseline | Description | Purpose |
|---|---|---|
| linear interpolation | piecewise linear interpolation between latent tokens | simplest continuous baseline |
| cubic spline | smooth interpolation through tokens | tests classical smooth curve fitting |
| B-spline | control-point representation | compares against low-pass parametric curves |
| DCT basis | truncated cosine basis | tests frequency-limited representation |
| Fourier-feature MLP | MLP with Fourier encoding | tests whether sine activations are necessary |
| ReLU MLP | same size as SIREN but ReLU | tests role of smooth activation |

The most important comparison is not only reconstruction error. It is whether the baseline preserves fast hand/finger transitions without producing jitter.

---

## 7. Fitting modes

### 7.1 Fit-all mode

Fit all latent tokens:

$$
\min_\theta \sum_{i=1}^{L}\ell(f_\theta(s_i),z_i).
$$

This measures maximum representational capacity.

**Question answered:**

> Can the field represent the exact latent trajectory when every token is observed?

### 7.2 Odd-even interpolation mode

Fit only even-indexed tokens and evaluate on odd-indexed tokens:

$$
\mathcal{I}_{train}=\{i: i \text{ even}\}, \qquad \mathcal{I}_{test}=\{i: i \text{ odd}\}.
$$

Train:

$$
\min_\theta \sum_{i\in\mathcal{I}_{train}}\ell(f_\theta(s_i),z_i).
$$

Evaluate:

$$
\frac{1}{|\mathcal{I}_{test}|}\sum_{i\in\mathcal{I}_{test}}\|f_\theta(s_i)-z_i\|.
$$

**Question answered:**

> Does the field learn a meaningful continuous path or merely memorize discrete tokens?

### 7.3 Sparse-knot mode

Fit only every $m$-th token:

$$
\mathcal{I}_{train}=\{1,1+m,1+2m,\ldots,L\}, \qquad m\in\{2,4,8\}.
$$

Evaluate on all held-out tokens.

**Question answered:**

> How many latent anchor points are needed to reconstruct the sequence?

This is useful for deciding whether a future model can predict a small set of modulation tokens instead of every latent token.

### 7.4 Dense-query mode

After fitting, query the field at higher temporal resolution:

$$
s'_k = -1 + \frac{2(k-1)}{K'-1}, \qquad K'=rL, \qquad r\in\{2,4\}.
$$

Decode:

$$
\hat x' = D_\phi(\mathrm{denorm}_z(f_\theta(s'_{1:K'}))).
$$

Compare against temporally resampled ground-truth SMPL-X motion.

**Question answered:**

> Does dense querying produce smoother and plausible motion, or does it create invalid in-between poses?

### 7.5 Peak-transition mode

Find the top $K_{peak}$ adjacent latent jumps:

$$
d_i=\|z_i-z_{i-1}\|_2.
$$

For each peak transition $(i-1)\rightarrow i$, query midpoint:

$$
s_{mid}=\frac{s_{i-1}+s_i}{2}, \qquad z_{mid}=f_\theta(s_{mid}).
$$

Decode local sequences containing:

1. token $i-1$;
2. midpoint;
3. token $i$.

Render frames and inspect whether the midpoint is physically and linguistically plausible.

**Question answered:**

> Are high latent jumps real sign transitions or latent-space discontinuities?

---

## 8. Loss design

Use a weighted objective:

$$
\mathcal{L}
=\lambda_z\mathcal{L}_{z}
+\lambda_v\mathcal{L}_{vel}
+\lambda_a\mathcal{L}_{acc}
+\lambda_j\mathcal{L}_{jerk}
+\lambda_{dec}\mathcal{L}_{dec}
+\lambda_{mid}\mathcal{L}_{mid}.
$$

### 8.1 Latent position loss

$$
\mathcal{L}_z
=\frac{1}{L}\sum_{i=1}^{L}\mathrm{SmoothL1}(f_\theta(s_i),z_i).
$$

SmoothL1 is recommended because some adjacent latent jumps may be valid but large.

### 8.2 Analytical latent velocity supervision

Because $f_\theta$ is differentiable, compute:

$$
\dot z_\theta(s_i)=\frac{\partial f_\theta(s)}{\partial s}\bigg|_{s=s_i}.
$$

Target velocity from finite differences:

$$
\dot z_i^{gt}
\approx
\frac{z_{i+1}-z_{i-1}}{s_{i+1}-s_{i-1}}.
$$

Then:

$$
\mathcal{L}_{vel}
=\frac{1}{L-2}\sum_{i=2}^{L-1}\mathrm{SmoothL1}(\dot z_\theta(s_i),\dot z_i^{gt}).
$$

This is analogous to NIAF's idea of supervising analytical derivatives, but here the derivative is in **latent progress space**, not robot joint space.

### 8.3 Analytical latent acceleration supervision

Compute:

$$
\ddot z_\theta(s_i)=\frac{\partial^2 f_\theta(s)}{\partial s^2}\bigg|_{s=s_i}.
$$

Target acceleration:

$$
\ddot z_i^{gt}
\approx
\frac{\dot z_{i+1}^{gt}-\dot z_{i-1}^{gt}}{s_{i+1}-s_{i-1}}.
$$

Loss:

$$
\mathcal{L}_{acc}
=\frac{1}{L-4}\sum_{i=3}^{L-2}\mathrm{SmoothL1}(\ddot z_\theta(s_i),\ddot z_i^{gt}).
$$

### 8.4 Jerk regularization

Do not usually supervise jerk with finite-difference targets, because the target jerk is noisy. Instead regularize the analytical third derivative:

$$
\mathcal{L}_{jerk}
=\frac{1}{L}\sum_{i=1}^{L}\left\|\frac{\partial^3 f_\theta(s)}{\partial s^3}\bigg|_{s=s_i}\right\|_2^2.
$$

Use a small coefficient. Too much jerk regularization may smooth away fast sign-relevant transitions.

Recommended starting value:

$$
\lambda_j \in [10^{-7},10^{-4}].
$$

Tune by checking hand/finger preservation, not only average smoothness.

### 8.5 Decoded-pose loss

Decode the fitted latent sequence:

$$
\hat x = D_\phi(\mathrm{denorm}_z(\hat z_{1:L})), \qquad \hat z_i=f_\theta(s_i).
$$

Compare with ground-truth compact pose $x$ or VAE reconstruction $x^{vae}=D_\phi(\mathrm{denorm}_z(z_{1:L}))$.

Recommended decoded loss:

$$
\mathcal{L}_{dec}
=\lambda_{rot}\|\hat x-x\|_1
+\lambda_{joint}\|J(\hat x)-J(x)\|_1
+\lambda_{hand}\|J_{hand}(\hat x)-J_{hand}(x)\|_1
+\lambda_{velx}\|\nabla\hat x-\nabla x\|_1
+\lambda_{accx}\|\nabla^2\hat x-\nabla^2x\|_1.
$$

If decoding during every optimization step is too expensive, apply $\mathcal{L}_{dec}$ every $N$ steps or use it only during the final fine-tuning phase.

### 8.6 Midpoint interpolation loss

For adjacent latent tokens, query midpoint:

$$
z_{i+1/2}=f_\theta\left(\frac{s_i+s_{i+1}}{2}\right).
$$

If dense ground-truth pose is available after temporal resampling, decode a dense sequence and compare to resampled pose. Otherwise, use a weaker manifold consistency loss:

$$
\mathcal{L}_{mid}
=\sum_i \left\|D_\phi(z_{i+1/2}) - \mathrm{InterpPose}(x_i,x_{i+1})\right\|_1.
$$

This loss should be optional because linear interpolation in pose space is not always linguistically correct.

---

## 9. Recommended loss schedules

### 9.1 Stage A: position warm-up

Use only latent position loss:

$$
\mathcal{L}=\mathcal{L}_z.
$$

Steps: 500-1000.

Purpose: ensure the field finds the rough trajectory before derivative constraints are applied.

### 9.2 Stage B: velocity and acceleration fitting

Use:

$$
\mathcal{L}=\lambda_z\mathcal{L}_z+\lambda_v\mathcal{L}_{vel}+\lambda_a\mathcal{L}_{acc}.
$$

Steps: 1000-3000.

Recommended initial weights:

| Loss | Weight |
|---|---:|
| $\lambda_z$ | 1.0 |
| $\lambda_v$ | 0.05 |
| $\lambda_a$ | 0.01 |

### 9.3 Stage C: light jerk regularization

Use:

$$
\mathcal{L}=\lambda_z\mathcal{L}_z+\lambda_v\mathcal{L}_{vel}+\lambda_a\mathcal{L}_{acc}+\lambda_j\mathcal{L}_{jerk}.
$$

Steps: 1000-3000.

Recommended initial weights:

| Loss | Weight |
|---|---:|
| $\lambda_z$ | 1.0 |
| $\lambda_v$ | 0.05 |
| $\lambda_a$ | 0.01 |
| $\lambda_j$ | 1e-6 |

### 9.4 Stage D: decoded-pose fine-tuning

Optional final stage:

$$
\mathcal{L}=\lambda_z\mathcal{L}_z+\lambda_{dec}\mathcal{L}_{dec}+\lambda_j\mathcal{L}_{jerk}.
$$

Recommended weights:

| Loss | Weight |
|---|---:|
| $\lambda_z$ | 1.0 |
| $\lambda_{dec}$ | 0.1 |
| $\lambda_j$ | 1e-6 |

Use decoded-pose fine-tuning only after the latent fit is stable.

---

## 10. Optimization details

### 10.1 Per-sequence fitting

For each sequence:

1. initialize a fresh SIREN;
2. optimize only that SIREN's parameters;
3. save fitted parameters, predictions, metrics, and plots;
4. aggregate metrics over sequences.

This is slow but clean and gives a true oracle upper bound.

### 10.2 Batch fitting alternative

A faster alternative is to fit a batch of independent fields using a shared architecture and per-sequence embeddings, but this makes the oracle less pure. Use per-sequence fitting first.

### 10.3 Learning rate

Recommended grid:

| LR | Use case |
|---:|---|
| 1e-3 | small SIREN, short sequences |
| 5e-4 | default |
| 1e-4 | long sequences or unstable derivative losses |

Use cosine decay or ReduceLROnPlateau.

### 10.4 Gradient clipping

Use:

$$
\|g\|_2 \le 1.0.
$$

Derivative losses can produce large gradients.

### 10.5 Early stopping

Stop if:

- validation/held-out latent error does not improve for 500 steps;
- fit-all latent error is below target threshold;
- decoded-pose error begins to worsen while latent error improves.

### 10.6 Mixed precision

Avoid mixed precision for derivative-heavy losses at first. Use full precision for autograd derivatives.

---

## 11. Metrics

### 11.1 Latent reconstruction metrics

For fitted tokens:

$$
\mathrm{MAE}_z=\frac{1}{Ld_z}\sum_{i,c}|\hat z_{i,c}-z_{i,c}|.
$$

$$
\mathrm{L2}_z=\frac{1}{L}\sum_i\|\hat z_i-z_i\|_2.
$$

Relative L2:

$$
\mathrm{RelL2}_z=\frac{\sum_i\|\hat z_i-z_i\|_2}{\sum_i\|z_i-\bar z\|_2+\epsilon}.
$$

R-squared:

$$
R_z^2=1-\frac{\sum_i\|\hat z_i-z_i\|_2^2}{\sum_i\|z_i-\bar z\|_2^2+\epsilon}.
$$

### 11.2 Derivative metrics

Velocity error:

$$
E_{vel}=\frac{1}{L-2}\sum_{i=2}^{L-1}\|\dot z_\theta(s_i)-\dot z_i^{gt}\|_2.
$$

Acceleration error:

$$
E_{acc}=\frac{1}{L-4}\sum_{i=3}^{L-2}\|\ddot z_\theta(s_i)-\ddot z_i^{gt}\|_2.
$$

Jerk magnitude:

$$
E_{jerk}=\frac{1}{L}\sum_i\left\|\frac{\partial^3 f_\theta(s_i)}{\partial s^3}\right\|_2.
$$

### 11.3 Decoded-pose metrics

Compute three decoded sequences:

1. original ground truth: $x$;
2. VAE reconstruction: $x^{vae}=D_\phi(z)$;
3. fitted-field reconstruction: $\hat x=D_\phi(\hat z)$.

Report:

| Metric | Definition |
|---|---|
| VAE upper-bound error | error between $x^{vae}$ and $x$ |
| Field error | error between $\hat x$ and $x$ |
| Field-to-VAE gap | error($\hat x$, $x$) - error($x^{vae}$, $x$) |
| Field-to-VAE latent gap | error($\hat z$, $z$) |

Use the same geometric metrics as the main paper when possible:

- body DTW-JPE;
- hand DTW-JPE;
- body DTW-PA-JPE;
- hand DTW-PA-JPE;
- unwarped joint error;
- hand/finger joint error;
- jaw/facial expression error if available;
- pose velocity, acceleration, and jerk metrics.

### 11.4 Smoothness metrics

For decoded motion:

$$
E_{vel}^{pose}=\frac{1}{T-1}\sum_t\|J(\hat x_t)-J(\hat x_{t-1})\|_2.
$$

$$
E_{acc}^{pose}=\frac{1}{T-2}\sum_t\|J(\hat x_{t+1})-2J(\hat x_t)+J(\hat x_{t-1})\|_2.
$$

$$
E_{jerk}^{pose}=\frac{1}{T-3}\sum_t\|J(\hat x_{t+2})-3J(\hat x_{t+1})+3J(\hat x_t)-J(\hat x_{t-1})\|_2.
$$

Report these separately for:

- body;
- dominant hand;
- non-dominant hand;
- fingers;
- jaw/face if available.

### 11.5 Adjacent-vs-random latent distance ratio

Compute adjacent distance:

$$
d_{adj}=\mathbb{E}_i\|z_i-z_{i-1}\|_2.
$$

Compute random non-neighbor distance:

$$
d_{rand}=\mathbb{E}_{|i-j|>m}\|z_i-z_j\|_2.
$$

Then:

$$
R_{smooth}=\frac{d_{adj}}{d_{rand}}.
$$

Interpretation:

| $R_{smooth}$ | Interpretation |
|---:|---|
| $<0.4$ | strongly temporally coherent latent path |
| 0.4-0.7 | moderately smooth latent path |
| 0.7-0.9 | weakly smooth latent path |
| $>0.9$ | adjacent tokens nearly as far as random tokens |

This metric should be reported before and after any VAE improvements.

---

## 12. Pass/fail criteria

The oracle field is considered successful if most of the following hold.

### 12.1 Fit-all success

| Criterion | Suggested threshold |
|---|---:|
| relative latent L2 | $<0.10$ |
| latent $R^2$ | $>0.90$ |
| field-to-VAE decoded gap | $<10\%$ of VAE reconstruction error or current model gap |
| visible hand/finger degradation | minimal |

### 12.2 Interpolation success

| Criterion | Suggested threshold |
|---|---:|
| odd-token latent $R^2$ | $>0.80$ |
| odd-token decoded hand error | close to linear/cubic interpolation or better |
| midpoint decoded frames | visually plausible |
| high-jump midpoint failure rate | $<20\%$ |

### 12.3 Dense-query success

| Criterion | Desired behavior |
|---|---|
| dense decoded motion | smoother than discrete VAE reconstruction |
| pose jerk | decreases |
| hand articulation | preserved |
| fast transitions | not over-smoothed |
| non-manual timing | not visibly delayed or flattened |

### 12.4 Decision table

| Result pattern | Interpretation | Next action |
|---|---|---|
| fit-all good, interpolation good, dense decode good | current VAE latent space supports implicit fields | build text-conditioned implicit latent field |
| fit-all good, interpolation bad | latent tokens are fit-able but path between tokens is not reliable | add VAE interpolation/manifold losses |
| latent fit good, decoded pose poor | VAE decoder has manifold holes or off-manifold sensitivity | retrain/fine-tune VAE with interpolation and decoded smoothness losses |
| fit-all poor | SIREN capacity/time parameterization insufficient or latent path too irregular | try arc length, higher width, residual field, or improve VAE |
| jerk regularization improves smoothness but hurts hands | over-smoothing sign-critical detail | use hand-aware loss and adaptive jerk penalty |

---

## 13. Main experimental grid

### 13.1 Architecture grid

| ID | Model | Width | Depth | Notes |
|---|---|---:|---:|---|
| A1 | SIREN | 128 | 3 | small, fast |
| A2 | SIREN | 256 | 3 | default |
| A3 | SIREN | 512 | 3 | capacity test |
| A4 | SIREN | 256 | 4 | deeper test |
| A5 | residual SIREN | 256 | 3 | linear interpolation plus residual |
| A6 | Fourier MLP | 256 | 3 | alternative INR |
| A7 | ReLU MLP | 256 | 3 | activation ablation |

### 13.2 Time grid

| ID | Parameterization |
|---|---|
| T1 | uniform progress |
| T2 | latent arc length |
| T3 | pose arc length |

### 13.3 Loss grid

| ID | Losses |
|---|---|
| L1 | latent position only |
| L2 | latent position + velocity |
| L3 | latent position + velocity + acceleration |
| L4 | latent position + velocity + jerk |
| L5 | latent position + velocity + jerk + decoded pose |

### 13.4 Fitting-mode grid

| ID | Mode |
|---|---|
| F1 | fit all tokens |
| F2 | even-to-odd interpolation |
| F3 | sparse knots, stride 4 |
| F4 | dense query 2x |
| F5 | dense query 4x |
| F6 | peak-transition midpoint rendering |

### 13.5 Minimal recommended pilot

To keep the pilot manageable, run:

- A2, A5, cubic spline, B-spline;
- T1 and T2;
- L1, L2, L4;
- F1, F2, F5, F6;
- 50 sequences per dataset.

---

## 14. Qualitative visualizations

For each selected sequence, generate the following plots.

### 14.1 PCA trajectory overlay

Project ground-truth latent tokens and fitted field samples to the same PCA space.

Show:

- ground-truth discrete tokens;
- fitted field sampled at token locations;
- dense fitted field samples;
- start and end markers;
- color by progress.

Important: Fit and metrics must always use full $d_z$-dimensional latent vectors. PCA is only for visualization.

### 14.2 Adjacent latent-token distance plot

Plot:

- $\|z_i-z_{i-1}\|_2$ for ground truth;
- $\|\hat z_i-\hat z_{i-1}\|_2$ for fitted field;
- mean;
- 95th percentile;
- top peaks.

### 14.3 Velocity/acceleration/jerk plot

Plot latent derivative magnitude:

$$
\|\dot z(s)\|_2, \quad \|\ddot z(s)\|_2, \quad \|z^{(3)}(s)\|_2.
$$

Compare against finite differences from ground-truth latent tokens.

### 14.4 Peak-transition frames

For top high-distance transitions, render:

1. ground-truth frame before transition;
2. ground-truth frame after transition;
3. decoded linear latent midpoint;
4. decoded SIREN midpoint;
5. decoded dense-field sequence around the transition.

This is the most important qualitative check.

### 14.5 Dense-sampling video

For each sequence, export videos:

- original SMPL-X;
- VAE reconstruction;
- oracle field at original latent rate;
- oracle field at 2x latent rate;
- oracle field at 4x latent rate.

Use the same camera and rendering setup.

---

## 15. Implementation blueprint

Current namespace after refactor:

```text
NIAF/oracle_latent_field/
```

Run modules as:

```bash
python -m NIAF.oracle_latent_field.scripts.fit_oracle_field
```

New experiment outputs should be written under:

```text
experiments/NIAF/oracle_latent_field/
```

### 15.1 Repository layout

```text
NIAF/oracle_latent_field/
  configs/
    oracle_siren_default.yaml
    oracle_siren_pilot.yaml
    oracle_ablation_grid.yaml
  data/
    export_latents.py
    build_oracle_index.py
  models/
    siren.py
    fourier_mlp.py
    spline_baselines.py
  losses/
    latent_derivatives.py
    decoded_pose_losses.py
  scripts/
    fit_oracle_field.py
    eval_oracle_field.py
    render_peak_transitions.py
    make_oracle_report.py
  outputs/
    metrics/
    plots/
    videos/
    fitted_params/
```

### 15.2 Export script

`export_latents.py` should save:

```python
{
    "sample_id": str,
    "dataset": str,
    "text": str,
    "x": FloatTensor[T, D],
    "z": FloatTensor[L, dz],
    "frame_mask": BoolTensor[T],
    "latent_mask": BoolTensor[L],
    "T": int,
    "L": int,
    "fps": float,
}
```

Also save VAE reconstruction:

```python
x_vae = vae.decode(denorm_z(z))
```

This allows every field method to be compared to the VAE upper bound.

### 15.3 SIREN model skeleton

```python
import math
import torch
import torch.nn as nn

class SineLayer(nn.Module):
    def __init__(self, in_dim, out_dim, omega=30.0, is_first=False):
        super().__init__()
        self.in_dim = in_dim
        self.omega = omega
        self.is_first = is_first
        self.linear = nn.Linear(in_dim, out_dim)
        self.reset_parameters()

    def reset_parameters(self):
        with torch.no_grad():
            if self.is_first:
                bound = 1.0 / self.in_dim
            else:
                bound = math.sqrt(6.0 / self.in_dim) / self.omega
            self.linear.weight.uniform_(-bound, bound)
            self.linear.bias.uniform_(-bound, bound)

    def forward(self, x):
        return torch.sin(self.omega * self.linear(x))

class SirenLatentField(nn.Module):
    def __init__(self, dz=256, hidden=256, depth=3, omega0=30.0, omega=1.0):
        super().__init__()
        layers = [SineLayer(1, hidden, omega=omega0, is_first=True)]
        for _ in range(depth - 1):
            layers.append(SineLayer(hidden, hidden, omega=omega, is_first=False))
        self.net = nn.Sequential(*layers)
        self.out = nn.Linear(hidden, dz)

    def forward(self, s):
        # s: [N] or [N, 1]
        if s.ndim == 1:
            s = s[:, None]
        return self.out(self.net(s))
```

### 15.4 Derivative utilities

```python
def grad_wrt_s(y, s):
    # y: [N, C], s: [N, 1]
    grads = []
    for c in range(y.shape[-1]):
        g = torch.autograd.grad(
            y[:, c].sum(), s,
            create_graph=True,
            retain_graph=True,
            allow_unused=False,
        )[0]
        grads.append(g)
    return torch.cat(grads, dim=-1)  # [N, C]

def derivatives(field, s):
    s = s.detach().clone().requires_grad_(True)
    z = field(s)
    dz = grad_wrt_s(z, s)
    ddz = grad_wrt_s(dz, s)
    dddz = grad_wrt_s(ddz, s)
    return z, dz, ddz, dddz
```

For efficiency, compute full derivatives only at sampled collocation points, not necessarily all tokens every step.

### 15.5 Fitting loop skeleton

```python
def fit_one_sequence(sample, field, vae, cfg):
    z_gt = sample["z"].to(cfg.device)          # [L, dz]
    x_gt = sample["x"].to(cfg.device)          # [T, D]
    s = make_time_grid(z_gt, mode=cfg.time_mode).to(cfg.device)  # [L, 1]

    optimizer = torch.optim.AdamW(
        field.parameters(),
        lr=cfg.lr,
        weight_decay=cfg.weight_decay,
    )

    for step in range(cfg.steps):
        optimizer.zero_grad(set_to_none=True)

        idx = sample_collocation_indices(len(s), cfg.batch_points)
        s_b = s[idx].detach().clone().requires_grad_(True)
        z_b_gt = z_gt[idx]

        z_b, dz_b, ddz_b, dddz_b = derivatives(field, s_b)

        loss_z = smooth_l1(z_b, z_b_gt)
        loss = cfg.lambda_z * loss_z

        if cfg.lambda_vel > 0:
            dz_gt = finite_difference_velocity(z_gt, s)
            loss_vel = smooth_l1(dz_b, dz_gt[idx])
            loss = loss + cfg.lambda_vel * loss_vel

        if cfg.lambda_acc > 0:
            ddz_gt = finite_difference_acceleration(z_gt, s)
            loss_acc = smooth_l1(ddz_b, ddz_gt[idx])
            loss = loss + cfg.lambda_acc * loss_acc

        if cfg.lambda_jerk > 0:
            loss_jerk = (dddz_b.pow(2).sum(dim=-1)).mean()
            loss = loss + cfg.lambda_jerk * loss_jerk

        # Optional decoded-pose loss every N steps.
        if cfg.lambda_dec > 0 and step >= cfg.dec_start and step % cfg.dec_every == 0:
            z_hat_full = field(s).unsqueeze(0)   # [1, L, dz]
            x_hat = vae.decode(denorm_z(z_hat_full))
            loss_dec = decoded_pose_loss(x_hat, x_gt[None])
            loss = loss + cfg.lambda_dec * loss_dec

        loss.backward()
        torch.nn.utils.clip_grad_norm_(field.parameters(), cfg.grad_clip)
        optimizer.step()

    return evaluate_one_sequence(sample, field, vae, cfg)
```

---

## 16. Configuration template

```yaml
experiment_name: oracle_siren_pilot
seed: 1234

data:
  datasets: [How2Sign, CSL-Daily, PHOENIX-2014T]
  split: val
  max_sequences_per_dataset: 100
  min_latent_len: 8
  max_latent_len: 120

model:
  type: siren
  dz: 256
  hidden: 256
  depth: 3
  omega0: 30.0
  omega: 1.0
  residual_interp: false

fit:
  mode: fit_all   # fit_all | even_odd | sparse_knot
  time_mode: uniform   # uniform | latent_arclength | pose_arclength
  steps: 5000
  batch_points: 64
  lr: 5.0e-4
  weight_decay: 0.0
  grad_clip: 1.0

loss:
  lambda_z: 1.0
  lambda_vel: 0.05
  lambda_acc: 0.01
  lambda_jerk: 1.0e-6
  lambda_dec: 0.0
  dec_start: 3000
  dec_every: 20

output:
  save_fitted_params: true
  save_plots: true
  save_videos: true
  dense_query_factors: [2, 4]
  peak_top_k: 5
```

---

## 17. Report format

For each dataset, create one summary table:

| Method | Time | Loss | Fit mode | RelL2 z ↓ | R2 z ↑ | VAE body error ↓ | Field body error ↓ | Gap ↓ | Hand error ↓ | Pose jerk ↓ | Peak midpoint fail ↓ |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Linear interp | uniform | - | dense | | | | | | | | |
| Cubic spline | uniform | - | dense | | | | | | | | |
| B-spline | uniform | pos | fit-all | | | | | | | | |
| SIREN | uniform | pos | fit-all | | | | | | | | |
| SIREN | uniform | pos+vel | fit-all | | | | | | | | |
| SIREN | latent arc | pos+vel+jerk | fit-all | | | | | | | | |
| Residual SIREN | latent arc | pos+vel+jerk | fit-all | | | | | | | | |

Also create stratified results by:

- sentence length;
- latent length;
- mean adjacent latent distance;
- 95th percentile adjacent latent distance;
- dataset;
- high-motion vs low-motion samples.

---

## 18. Expected outcomes and interpretation

### Outcome A: SIREN fits full latents and interpolates well

This supports moving to a text-conditioned implicit latent field.

Next method:

$$
z(s)=f_{\theta(y,\mathcal{M})}(s),
$$

where $\theta$ or modulation vectors are predicted from sentence text $y$ and unordered word-candidate memory $\mathcal{M}$.

### Outcome B: SIREN fits all tokens but interpolation fails

This means the latent path can be memorized but the space between tokens is unreliable.

Next action:

- retrain/fine-tune VAE with latent interpolation losses;
- add decoded midpoint losses;
- add latent velocity/acceleration regularization during VAE training;
- use residual field instead of pure field.

### Outcome C: SIREN underfits high-distance transitions

This may mean that fast sign transitions need higher frequency capacity.

Next action:

- increase width/depth;
- use higher $\omega_0$;
- use residual SIREN;
- use piecewise fields;
- use arc-length parameterization;
- reduce jerk penalty near high-motion areas.

### Outcome D: SIREN smooths away hand articulation

This is dangerous for sign language.

Next action:

- add hand-weighted decoded loss;
- reduce jerk coefficient;
- use adaptive jerk penalty that is lower around high hand velocity;
- model body and hands with separate fields;
- use a global field plus local high-frequency residual field.

---

## 19. Recommended VAE improvements if oracle fitting fails

If the oracle experiment shows latent discontinuities, improve the VAE before changing the generator.

### 19.1 Latent velocity regularization

$$
\mathcal{L}_{zvel}=\frac{1}{L-1}\sum_{i=2}^{L}\|z_i-z_{i-1}\|_1.
$$

Use lightly. Too much will collapse meaningful motion.

### 19.2 Latent acceleration regularization

$$
\mathcal{L}_{zacc}=\frac{1}{L-2}\sum_{i=2}^{L-1}\|z_{i+1}-2z_i+z_{i-1}\|_1.
$$

### 19.3 Decoded midpoint interpolation loss

For midpoint latent:

$$
z_{i+1/2}=\frac{z_i+z_{i+1}}{2}.
$$

Decode a local sequence or dense sequence and enforce plausible in-between pose:

$$
\mathcal{L}_{interp}=\|D_\phi(z_{i+1/2})-x_{i+1/2}^{target}\|_1.
$$

The target can be temporally resampled ground truth or a carefully designed kinematic interpolation target.

### 19.4 Local manifold consistency

Sample:

$$
\tilde z=(1-\alpha)z_i+\alpha z_{i+1}, \qquad \alpha\sim U(0,1).
$$

Decode $\tilde z$, and regularize decoded pose velocity/acceleration to avoid implausible in-between poses.

### 19.5 Hand-aware latent smoothness

Use a decoded hand loss to prevent the VAE from hiding hand discontinuities in latent space:

$$
\mathcal{L}_{hand}=\|J_{hand}(D_\phi(z))-J_{hand}(x)\|_1.
$$

---

## 20. How this connects to the future implicit SoftArrangerFlow model

If oracle latent-field fitting succeeds, the next model should not directly predict discrete latent length $\hat L$. Instead, build a continuous field conditioned on text and lexical memory.

### 20.1 Continuous query version of Soft Word Arranger

Current discrete query:

$$
q_i=P_qh_y+p_i^{sent}.
$$

Continuous query:

$$
q(s)=P_qh_y+\gamma(s), \qquad s\in[0,1].
$$

where $\gamma(s)$ is a Fourier or SIREN progress encoding.

The arranged prior becomes:

$$
z_p(s)=\mathrm{SoftArranger}(q(s),\mathcal{M}).
$$

### 20.2 Continuous adapter

$$
z_a(s)=z_p(s)+\Delta_\eta(z_p(s),s,h_y).
$$

### 20.3 Function-space flow matching

Instead of discrete flow on $z_{1:L}$, train flow at randomly sampled collocation points:

$$
z_0(s)=z_a(s)+\sigma\xi(s),
$$

$$
z_t(s)=(1-t)z_0(s)+tz_1(s),
$$

$$
v^*(s)=z_1(s)-z_0(s).
$$

Train:

$$
\mathcal{L}_{fm}=\mathbb{E}_{s,t}\|v_\theta(z_t(s),t,s,y,\mathcal{M})-v^*(s)\|^2.
$$

### 20.4 Duration is still needed

A continuous latent field removes the need to predict a fixed discrete latent sequence length before generation, but the final system still needs a duration or stopping mechanism.

Three options:

1. **Duration head:** predict physical duration $D$, then query field at required frame rate.
2. **EOS/stopping head:** query until $p_{end}(s)$ or $p_{end}(t)$ crosses a threshold.
3. **Time-warp field:** predict $dt/ds=\mathrm{softplus}(r(s))$, then obtain duration from integration.

The oracle experiment helps decide whether the future model should use uniform progress or arc-length/time-warp progress.

---

## 21. Practical recommendation

Run the experiment in this order:

1. **Export VAE latents** for 50-100 validation sequences per dataset.
2. **Compute latent smoothness diagnostics**: adjacent distance, random distance, acceleration, high-distance peaks.
3. **Fit SIREN in fit-all mode** using uniform progress and latent position loss.
4. **Add velocity and light jerk regularization** and check whether decoded motion improves.
5. **Run odd-even interpolation** to test whether the field learns a continuous path.
6. **Run dense-query decoding** at 2x and 4x resolution.
7. **Render high-distance transition midpoints** and inspect hands, wrists, jaw, and face.
8. **Compare uniform vs latent arc-length parameterization.**
9. **Decide whether to build implicit SoftArrangerFlow or first improve the VAE.**

The most important result is not the lowest latent MSE. The most important result is whether dense and midpoint decoded motion remains linguistically and physically plausible.

---

## 22. One-page summary for a future paper

**Oracle Latent-Field Fitting.** To test whether the frozen VAE latent space supports continuous implicit sign trajectories, we fit a per-sequence SIREN $f_\theta(s)$ directly to ground-truth sentence latents $z_{1:L}$. This oracle setting removes text, retrieval, and length-prediction errors, isolating the representational question. We compare uniform progress, latent arc-length progress, and pose arc-length progress, and optimize latent reconstruction with analytical velocity supervision and light jerk regularization. We evaluate both token-level reconstruction and decoded SMPL-X motion, including dense temporal querying and midpoint decoding around high-distance latent transitions. If the fitted field approaches the VAE reconstruction upper bound and interpolates safely, the current latent space is suitable for a text-conditioned implicit latent sign field. Otherwise, failures identify whether to improve the VAE manifold, add interpolation losses, or use piecewise/adaptive continuous fields.

---

## 23. Checklist before implementation

- [ ] Confirm VAE decoder can accept variable latent length.
- [ ] Confirm `norm_z` and `denorm_z` are exactly consistent with SoftArrangerFlow training.
- [ ] Fit and evaluate in full latent dimension, not PCA space.
- [ ] Keep VAE reconstruction as the upper-bound baseline.
- [ ] Report body and hand metrics separately.
- [ ] Render top high-distance transitions.
- [ ] Compare uniform and arc-length parameterizations.
- [ ] Avoid overly large jerk penalty.
- [ ] Track whether smoothness improvements erase sign-critical hand/facial details.
- [ ] Use the results to decide whether to improve the VAE before replacing the length predictor.
