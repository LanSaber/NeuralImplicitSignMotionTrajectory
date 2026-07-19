# Latent Flow Matching Hyperparameters

This note summarizes the hyperparameter settings used by the current SOKE
latent flow-matching pipeline. It is meant as a quick reference for training,
evaluation, ablations, and paper-method descriptions.

Primary implementation files:

- `flow/train_text_conditional.py`
- `flow/model.py`
- `flow/adapter_prior.py`
- `scripts/flow/train_text_conditional_sbatch.sh`
- `docs/flow/soke_smplx_flow_pipeline.md`

## Main Pipeline Identity

The main current setting is:

```text
motion_space      = latent
rotation_rep      = rot6d
text_conditioning = token_prefix
model_size        = base
source_mode       = noise | residual | adapter_residual
```

In words: the model trains a rectified-flow Transformer in the frozen VAE
latent space. Text is encoded by a frozen local T5 encoder, and token-level T5
states are prepended as prefix tokens to the motion Transformer. In the full
SoftArrangerFlow setting, the flow source is initialized from a frozen
soft-arranger/content-style adapter prior.

## Data And Representation

The prepared dataset still stores compact 133D SMPL-X axis-angle motion. At
load time, the training code can convert it into the representation consumed by
the neural network.

Current main setting:

```text
stored motion              = compact 133D axis-angle
network rotation_rep       = rot6d
rot6d network dimension    = 256
min_frames                 = 40
max_frames                 = 400
length_multiple            = 4
random_crop                = disabled in main runs
```

For latent flow, the frozen VAE further compresses the normalized rot6d motion:

```text
latent_dim        = 256
downsample_factor = 4
raw max_frames    = 400
latent max_frames = 100
```

The VAE uses deterministic latents:

```text
z = encoder_mu(x)
```

The VAE is frozen during flow training. Its weights are not saved inside flow
checkpoints.

## Frozen VAE Settings

The current dataset-level VAE checkpoints use the same basic architecture:

```text
rotation_rep         = rot6d
latent_dim           = 256
hidden_dim           = 512
num_layers           = 6
num_heads            = 8
downsample_factor    = 4
max_frames           = 400
lr                   = 3e-4
weight_decay         = 0.0
velocity_loss_weight = 1.0
accel_loss_weight    = 0.5
jerk_loss_weight     = 0.25
kl_warmup_epochs     = 300
validation_metric    = recon_axis_angle
save_top_k           = 3
```

Dataset-specific VAE checkpoints used by the main latent runs:

| Dataset | VAE checkpoint |
| --- | --- |
| How2Sign | `experiments/flow/VAE/how2sign_rot6d_vae_jerk_b128x4/checkpoints/best.pt` |
| PHOENIX-2014T | `experiments/flow/VAE/phoenix_rot6d_vae_jerk_b16x4_online_retry1/checkpoints/best.pt` |
| CSL-Daily | `experiments/flow/VAE/csl_daily_rot6d_vae_jerk_b16x4_online/checkpoints/best.pt` |

## Text Conditioning

Text encoder:

```text
text_model_path  = deps/flan-t5-base
max_text_tokens  = 64
text_dim         = 768
local_files_only = true
frozen           = true
```

The main setting is:

```text
text_conditioning = token_prefix
```

This uses T5 token embeddings rather than only one pooled sentence embedding.
The Transformer input is:

```text
[T5 text tokens, flow time token, motion latent tokens]
```

The output motion-token positions are projected back to a latent velocity
prediction.

The older/original option remains available:

```text
text_conditioning = pooled
```

In pooled mode, the T5 sequence is mean-pooled and added to every motion frame.

## Condition Field

The flow code can choose which manifest field is used as conditioning text:

```text
condition_field = text | gloss | text_gloss | label_word
```

The intended inference-compatible setting is usually:

```text
condition_field = text
```

CSL-Daily can use:

```text
condition_field = label_word
```

This is useful when the raw sentence is not whitespace-segmented in a way that
the lexical matcher can use. Older checkpoints may not store `condition_field`;
the current loader falls back to `text`.

## Flow Model Size

Available presets:

| `model_size` | `hidden_dim` | `num_layers` | `num_heads` |
| --- | ---: | ---: | ---: |
| `custom` | CLI-provided | CLI-provided | CLI-provided |
| `small` | 256 | 4 | 4 |
| `base` | 512 | 8 | 8 |
| `large` | 768 | 12 | 12 |
| `xl` | 1024 | 16 | 16 |

Main setting:

```text
model_size  = base
hidden_dim  = 512
num_layers  = 8
num_heads   = 8
ffn_dim     = 2048
dropout     = 0.0
```

In latent base mode, the trainable flow model has about 28M parameters.

## Flow Training Objective

The model learns the rectified-flow velocity field:

```text
z0       = source latent
z1       = normalized GT VAE latent
t        ~ Uniform(0, 1)
zt       = (1 - t) * z0 + t * z1
target v = z1 - z0
v_pred   = f_theta(zt, t, text)
```

For latent mode, the direct latent losses are:

```text
L_flow          = MSE(v_pred, z1 - z0)
L_latent_recon  = SmoothL1(zt + (1 - t) * v_pred, z1)
```

The predicted clean latent is decoded through the frozen VAE, then decoded
motion is compared to the normalized GT motion:

```text
L_pose  = SmoothL1(decoded_motion_pred, motion_gt)
L_vel   = SmoothL1(diff(decoded_motion_pred), diff(motion_gt))
L_accel = SmoothL1(diff2(decoded_motion_pred), diff2(motion_gt))
```

Main flow loss weights:

```text
latent_loss_weight   = 1.0
pose_loss_weight     = 2.0
velocity_loss_weight = 2.0
accel_loss_weight    = 1.0
```

Total latent loss:

```text
L = 1.0 * (L_flow + L_latent_recon)
  + 2.0 * L_pose
  + 2.0 * L_vel
  + 1.0 * L_accel
```

Feature weighting:

```text
hand_weight      = 3.0
hand_valid_floor = 0.2
```

Padded frames are masked. Invalid hand frames are down-weighted, but not fully
removed.

Important: flow training currently does not include a jerk loss. Jerk is used
in VAE and adapter training.

## Source Modes

### `source_mode=noise`

Pure text-conditioned latent flow:

```text
z0 = smooth Gaussian latent noise
z1 = normalized VAE latent of GT motion
```

This is the main ablation baseline.

### `source_mode=residual`

Raw word-concat residual flow:

```text
text -> lexical word lookup -> concatenate word poses -> resample to sentence length
prior_raw -> frozen VAE encoder -> normalized prior latent z_prior
z0 = z_prior + residual_noise_scale * smooth_noise
```

Main setting:

```text
residual_noise_scale = 0.25
```

### `source_mode=adapter_residual`

Full SoftArrangerFlow source:

```text
text
  -> lexical word candidates
  -> frozen T5 word/sentence features
  -> frozen SoftWordArranger
  -> frozen ContentStyleAdapter
  -> z_adapt

z0 = z_adapt + residual_noise_scale * smooth_noise
```

This mode is latent-only. The only trainable module is the flow Transformer.
Frozen modules:

```text
T5 inside adapter prior
VAE encoder/decoder
SoftWordArranger
ContentStyleAdapter
```

## Noise And Sampling

Main latent run settings:

```text
noise_samples       = 4
noise_smoothing     = 3
sampler             = heun
sample_steps        = 100
residual_noise_scale= 0.25
```

`noise_samples` repeats each batch item with multiple independent `(z0, t)`
draws. Thus the effective flow-matching batch is:

```text
effective_batch = batch_size * noise_samples
```

`noise_smoothing` applies a temporal Gaussian filter to the source noise and
renormalizes valid frames. This prevents frame-independent noise from creating
fast jittery motion.

Heun sampling uses predictor-corrector integration:

```text
v0 = f(x, t0)
x_pred = x + dt * v0
v1 = f(x_pred, t1)
x = x + 0.5 * dt * (v0 + v1)
```

## Optimizer And Checkpointing

Main settings:

```text
optimizer       = AdamW
lr              = 3e-4
weight_decay    = 0.0
grad_clip       = 1.0
epochs          = 3000
seed            = 42
val_every       = 1
save_every      = 500
save_last_every = 50
save_top_k      = 3
sample_every    = 500
wandb_project   = soke-flow
```

The top-k validation checkpoints are selected by validation flow metric.

## Main Dataset Run Settings

These values come from the current experiment configs.

### Text-Only Latent Flow

| Dataset | Data dir | Batch / GPUs | Source | VAE |
| --- | --- | --- | --- | --- |
| How2Sign | `/media/cvpr/haomian/data/SOKE_FLOW/how2sign_soke_upper_smplx` | `256 x 2` | `noise` | How2Sign rot6d jerk VAE |
| PHOENIX-2014T | `/media/cvpr/haomian/data/SOKE_FLOW/phoenix_upper_smplx` | `256 x 2` | `noise` | PHOENIX rot6d jerk VAE |
| CSL-Daily | `/media/cvpr/haomian/data/SOKE_FLOW/csl_daily_upper_smplx` | `128 x 2` retry | `noise` | CSL-Daily rot6d jerk VAE |

Shared settings:

```text
motion_space      = latent
rotation_rep      = rot6d
text_conditioning = token_prefix
model_size        = base
noise_samples     = 4
noise_smoothing   = 3
sampler           = heun
sample_steps      = 100
val_every         = 1
distributed       = ddp
ddp_backend       = nccl
```

### Frozen Adapter-Residual Latent Flow

| Dataset | Batch / GPUs | Word split | Adapter checkpoint |
| --- | --- | --- | --- |
| How2Sign | `128 x 4` | `all` | `experiments/flow/adapter/how2sign_soft_arranger_signasl_all_b256x4_online_r3_gloo/checkpoints/best.pt` |
| PHOENIX-2014T | `64 x 4` | `all.balanced` | `experiments/flow/adapter/phoenix_soft_arranger_adapter_balanced_b64x4_gloss_online/checkpoints/epoch_0300.pt` |
| CSL-Daily | `64 x 2` | `all` | `experiments/flow/adapter/csl_daily_soft_arranger_lw_2xb256/checkpoints/last.pt` |

Shared settings:

```text
source_mode          = adapter_residual
motion_space         = latent
rotation_rep         = rot6d
text_conditioning    = token_prefix
model_size           = base
residual_noise_scale = 0.25
noise_samples        = 4
noise_smoothing      = 3
```

CSL-Daily uses `condition_field=label_word` in the current adapter-residual
flow config.

## Frozen Soft-Arranger Adapter Prior Settings

The full adapter-residual branch depends on a separately trained adapter
checkpoint. Current common settings:

```text
prior_mode              = soft_arranger
num_word_candidates     = 32
num_negative_candidates = 16
shuffle_word_candidates = true during adapter training
adapter hidden_dim      = 512
content_dim             = 256
style_dim               = 128
adapter num_layers      = 4
adapter num_heads       = 8
arranger_hidden_dim     = 512
arranger_num_heads      = 8
arranger_dropout        = 0.0
adapter lr              = 3e-4
adapter epochs          = 1000
adapter save_top_k      = 3
```

Common adapter loss weights:

```text
latent_loss_weight              = 1.0
pose_loss_weight                = 0.5
velocity_loss_weight            = 0.5
accel_loss_weight               = 0.25
jerk_loss_weight                = 0.1
style_loss_weight               = 0.1
content_pair_loss_weight        = 0.1
arranger_prior_loss_weight      = 1.0
gate_bce_loss_weight            = 0.1
gate_sparsity_loss_weight       = 0.01
attention_smoothness_weight     = 0.0 or 0.01, depending on run
null_usage_loss_weight          = 0.01
```

These adapter settings are not optimized during flow training. They define the
frozen prior generator used by `source_mode=adapter_residual`.

## Code Defaults Versus Main Runs

Be careful when reporting hyperparameters: the Python defaults are not always
the same as the main latent experiments.

| Setting | Code default | Main latent runs |
| --- | --- | --- |
| `motion_space` | `smplx` | `latent` |
| `rotation_rep` | `axis_angle` | `rot6d` |
| `text_conditioning` | `pooled` | `token_prefix` |
| `model_size` | `custom` | `base` |
| `hidden_dim` | `256` | `512` |
| `num_layers` | `4` | `8` |
| `num_heads` | `4` | `8` |
| `limit_train` | `32` | `0`, meaning full train split |
| `noise_samples` | `8` | `4` |
| `noise_smoothing` | `9` | `3` |
| `val_every` | `0` | `1` |

For paper or experiment reporting, use the experiment `config.json` files as
the source of truth.
