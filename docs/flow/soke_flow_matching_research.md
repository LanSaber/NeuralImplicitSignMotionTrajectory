# Flow Matching and Diffusion Direction for SOKE

Research snapshot date: 2026-05-29.

This note summarizes recent flow-matching and diffusion work that is relevant to replacing SOKE's autoregressive motion-token generator with a continuous motion generator. The practical recommendation is to start with a text/retrieval-conditioned flow-matching model over normalized SMPL-X feature sequences, then add stronger sign-language structure through gloss, isolated-sign retrieval, and phonological attributes.

## Why This Direction Fits SOKE

The current SOKE generator predicts integer VQ token IDs for body, left hand, and right hand streams. These IDs are useful compression targets, but they are weak semantic units: token `37` does not intrinsically mean a handshape, location, movement, or gloss. A language model can learn correlations between text and token strings, but the output space is still an arbitrary geometric codebook.

Flow matching and diffusion are better matched to this problem because they can generate continuous motion features directly:

```text
text / gloss / retrieval condition
        -> motion generator
        -> normalized SMPL-X feature sequence [T, 133]
        -> SMPL-X vertices and joints
```

For SOKE, the most useful variant is probably not pure text-to-motion from scratch. A stronger starting point is retrieval-conditioned residual generation:

```text
text
  -> retrieve isolated signs or similar sentence motion
  -> initialize a rough continuous motion sequence
  -> flow/diffusion model refines timing, co-articulation, and body/hand details
```

That design preserves the semantic advantage of retrieval while using a continuous generator to fix the artifacts that token decoding and hard concatenation produce.

## Relevant Motion Flow-Matching Papers

| Work | Main idea | What to borrow for SOKE |
| --- | --- | --- |
| Motion Flow Matching for Human Motion Synthesis and Editing, arXiv 2312.08895, https://arxiv.org/abs/2312.08895 | Applies flow matching to human motion synthesis and editing. Reports much fewer sampling steps than classic diffusion while keeping competitive text/action-to-motion quality. | Use an ODE-style velocity model over motion sequences. Sampling trajectory editing is relevant for inpainting missing frames or refining retrieved clips. |
| FlowMotion, arXiv 2504.01338, https://arxiv.org/abs/2504.01338 | Conditional flow matching for text-driven human motion, with a target-predictive objective aimed at reducing jitter. | Add explicit target/clean-motion prediction or auxiliary clean reconstruction loss to reduce hand jitter. |
| MotionFlux, arXiv 2508.19527, https://arxiv.org/abs/2508.19527 | Rectified flow matching plus preference alignment for efficient text-guided motion. | Use rectified flow for fast deterministic sampling, and later consider preference/ranking losses from visual or sign-quality comparisons. |
| MotionHiFlow, arXiv 2604.23264, https://arxiv.org/abs/2604.23264 | Hierarchical flow matching: low temporal scales capture coarse semantics; higher scales refine fine temporal details. Uses topology-aware motion representation. | Sign language needs both phrase-level semantics and detailed hand articulation. A coarse-to-fine model is likely useful after a first baseline. |
| FlowCoMotion, arXiv 2604.11083, https://arxiv.org/abs/2604.11083 | Combines discrete token information with continuous latent details, then predicts a velocity field conditioned on text. | This directly addresses our concern: keep tokens only as high-level cues, not as the final LM output vocabulary. |
| Riemannian Motion Generation, arXiv 2603.15016, https://arxiv.org/abs/2603.15016 | Models motion on product manifolds and trains with Riemannian flow matching. Emphasizes rotation geometry instead of treating everything as Euclidean. | If continuous 6D/axis-angle SMPL-X pose has artifacts, move toward geometry-aware rotation handling. This is more advanced than the first baseline. |
| Unified Motion Flow, CVPR 2026 project, https://githubhgh.github.io/umf/ | Uses a motion VAE latent space and flow matching for single/multi-person text-to-motion, reducing autoregressive error accumulation. | Train in a continuous latent space rather than directly in raw high-dimensional pose if raw pose training is unstable. |

## Relevant Sign-Language Diffusion Papers

| Work | Main idea | What to borrow for SOKE |
| --- | --- | --- |
| Neural Sign Actors, CVPR 2024, https://arxiv.org/abs/2312.02702 | Text-conditioned diffusion for 3D sign-language production using SMPL-X-style body skeleton structure. | Strong evidence that diffusion over 3D signing avatars is a reasonable path. Their anatomically informed graph idea is relevant for body/hand structure. |
| Toward Phonology-Guided Sign Language Motion Generation, arXiv 2603.17388, https://arxiv.org/abs/2603.17388 | MDM-style SMPL-X diffusion baseline and conditioning study with ASL-LEX phonological attributes such as handshape, location, and movement. | Do not rely only on raw text. Add gloss and phonological attributes through separate condition pathways. |
| Text-Driven 3D Hand Motion Generation from Sign Language Data, arXiv 2508.15902, https://arxiv.org/abs/2508.15902 | Trains a text-conditioned hand motion diffusion model using sign-language data and sign attribute descriptions. | Hands are the hardest part of SOKE. A hand-specialized branch or extra hand loss is justified. |
| Towards Continuous Sign Language Conversation from Isolated Signs, arXiv 2605.14705, https://arxiv.org/abs/2605.14705 | Builds continuous signing from isolated clips and uses a diffusion Transformer for duration alignment and co-articulatory boundary inpainting. | Very relevant to retrieval-conditioned SOKE: retrieve isolated signs, then generate transitions and timing. |
| G2P-DDM, AAAI 2024, https://ojs.aaai.org/index.php/AAAI/article/view/28441 | Gloss-to-pose with VQ-VAE and discrete diffusion over variable-length pose tokens. | Useful comparison to SOKE's token direction, but still less attractive than continuous flow for final motion quality. |

## Recommended SOKE Implementation

### Stage 0: Establish the Ceiling of the Existing Tokenizer

Before training a new generator, measure whether the existing VQ-VAE tokenizer can reconstruct sign motion well:

```text
GT normalized pose -> VQ encode -> VQ decode -> SMPL-X render
```

If this reconstruction is poor, the LM cannot recover high-quality motion. If it is good, then the current failure mostly comes from token prediction and semantic conditioning.

### Stage 1: Continuous Flow-Matching Baseline

Use the existing normalized SOKE feature representation:

```text
x_1: ground-truth normalized pose, shape [B, T, 133]
mask: valid frames, shape [B, T]
c: text condition embedding
```

Train a Transformer/DiT-style velocity model:

```text
x_0 ~ N(0, I)
t ~ Uniform(0, 1)
x_t = (1 - t) * x_0 + t * x_1
v_target = x_1 - x_0

loss = || mask * (v_theta(x_t, t, c) - v_target) ||_2^2
```

At inference:

```text
x <- N(0, I)
for t from 0 to 1:
    x <- x + dt * v_theta(x, t, c)
```

Use Euler or Heun integration with 10 to 30 steps. This should already be faster than classic diffusion and avoids autoregressive token drift.

### Stage 2: Add Classifier-Free Guidance

During training, randomly drop the condition:

```text
c = empty_condition with probability p_uncond
```

During sampling:

```text
v = v_uncond + guidance_scale * (v_cond - v_uncond)
```

This usually improves condition following. Start with `guidance_scale` around `1.5` to `3.0`.

### Stage 3: Retrieval-Conditioned Residual Flow

Use SOKE's existing retrieval idea, but retrieve continuous motion features instead of only token snippets.

Possible training setup:

```text
r: retrieved continuous motion, padded/resampled to target length
epsilon ~ N(0, I)
x_0 = r + sigma * epsilon
x_1 = ground-truth motion
x_t = (1 - t) * x_0 + t * x_1
v_target = x_1 - x_0
condition = text embedding + retrieval embedding
```

This turns the generator into a motion refiner. It only has to correct timing, transitions, signer style, and local hand/body detail, which is easier than generating the whole sequence from text alone.

### Stage 4: Add Sign-Language Structure

The recent sign-language papers point in the same direction: raw text alone is weak. Add one or more structured conditions:

- gloss sequence
- isolated word/sign retrieval
- handshape labels
- hand location labels
- movement labels
- palm orientation labels
- non-manual markers, if available

Use separate encoders for text and sign attributes instead of concatenating everything into one flat sentence. This makes the model's conditioning more controllable and easier to ablate.

## Model Skeleton

A minimal SOKE flow model can be organized as:

```text
mGPT/models/soke_flow.py
    Lightning module
    - loads H2S data
    - normalizes/pads motion
    - samples t and noise
    - computes flow loss
    - samples videos for validation

mGPT/archs/soke_dit.py
    Transformer denoiser / velocity field
    input: x_t [B, T, 133], t [B], mask [B, T], condition
    output: velocity [B, T, 133]

configs/soke_flow.yaml
    TRAIN.STAGE: flow
    model target: mGPT.models.soke_flow.SOKEFlow
    text encoder, hidden size, layers, sampling steps
```

The first version should not use the VQ token files at all. It should train from the same continuous pose tensors used before tokenization.

## Losses to Use

Start simple:

```text
L_flow = MSE(v_pred, v_target) over valid frames
```

Then add motion-specific terms:

```text
L_pose = smooth_l1(x1_pred, x1)             # optional clean-motion prediction
L_vel  = smooth_l1(delta(x1_pred), delta(x1))
L_acc  = smooth_l1(delta2(x1_pred), delta2(x1))
L_hand = higher_weight * hand_pose_loss
```

For sign language, weight the hand slices more than body:

```text
body/residual: x[..., :30] and x[..., 120:]
left hand:     x[..., 30:75]
right hand:    x[..., 75:120]
```

The hand loss should probably be 2x to 5x stronger than the body loss, because hand articulation carries most sign information.

## Length Handling

Flow matching generates a fixed tensor length per sample, so length must be handled explicitly.

Reasonable first version:

1. During training, use the ground-truth length and a frame mask.
2. During validation, use the dataset/reference length so quality can be compared fairly.
3. Later, train a small length predictor conditioned on text/gloss/retrieval.
4. For retrieval-conditioned generation, initialize length from the retrieved sequence or predicted gloss durations.

This avoids mixing two hard problems at once.

## Evaluation Plan

Use the current visualization pipeline, but compare four outputs:

1. Ground truth.
2. VQ reconstruction.
3. Current SOKE LM token generation.
4. New flow-matching generation.

Quantitative checks:

- body MPJPE
- left/right hand MPJPE
- velocity error
- acceleration/jitter
- sequence length error
- text-motion retrieval score, if an evaluator is available
- sign/gloss classifier accuracy, if a classifier is available

Visual checks:

- front-facing side-by-side videos
- hand shape stability
- contact/transition smoothness
- whether motion collapses to generic gestures
- whether retrieved signs survive after refinement

## Minimal Milestone Plan

1. Add a `soke_flow.yaml` config and a Lightning model that trains on `[B, T, 133]` continuous poses.
2. Implement a small Transformer velocity network with text conditioning and frame masks.
3. Train with flow loss only, using ground-truth lengths.
4. Add validation sampling and render MP4s using `test/visualize_soke_test.py` style rendering.
5. Add classifier-free guidance.
6. Add retrieval-conditioned residual flow.
7. Add gloss/phonology conditioning if annotations can be prepared.

The first target is not perfect sign production. The first target is to prove that continuous flow generation gives smoother, more plausible SMPL-X sequences than the current token LM baseline.

## Bottom Line

Flow matching is feasible and likely better aligned with SOKE than direct LM prediction of arbitrary motion-token IDs. The best path is:

```text
current baseline:
text -> LM -> arbitrary VQ IDs -> VQ decode

recommended baseline:
text/retrieval -> flow model -> continuous SMPL-X features

recommended full system:
text -> gloss/phonology/retrieval plan -> residual flow model -> continuous SMPL-X features
```

The implementation should keep the VQ tokenizer only as an analysis baseline at first. It should not be the main output space of the generator.
