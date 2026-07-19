# SOKE Content-Style Adapter

This document describes the current adapter pipeline under `flow/`. The adapter is a latent-space bridge between word-level signing motion and sentence-level signing motion. It is designed to improve a word-motion prior before the prior is used by the downstream residual/latent flow pipeline.

There are now two prior modes:

```text
prior_mode=concat
  sentence text -> lexical word lookup -> concatenate word poses -> VAE encode -> z_word -> adapter -> z_adapt

prior_mode=soft_arranger
  sentence text + unordered word candidates -> SoftWordArranger -> z_prior_aligned -> adapter -> z_adapt
```

The original concat adapter is kept unchanged. The soft arranger is the current improvement path because it does not require the word clips to be presented in sentence order.

## Implemented Files

```text
flow/content_style_adapter.py
flow/temporal_word_attention.py
flow/train_adapter.py
flow/evaluate_adapter.py
scripts/flow/train_adapter_sbatch.sh
```

Related visualization scripts:

```text
flow/visualize/visualize_compare_npz.py
flow/visualize/visualize_compare_three_npz.py
flow/visualize/visualize_npz.py
```

## Frozen Codec

The adapter works in the latent space of the frozen temporal VAE:

```text
experiments/flow/VAE/chatsign175_sentence_word_joint_rot6d_vae_b32/checkpoints/best.pt
```

Current VAE configuration:

```text
rotation_rep      = rot6d
input_dim         = 256
latent_dim        = 256
downsample_factor = 4
max_frames        = 400
max_latent_frames = 100
```

The VAE is not trained by the adapter. It is used only as:

```text
compact SMPL-X / rot6d x -> VAE encoder -> latent z
latent z -> VAE decoder -> compact SMPL-X / rot6d x
```

The adapter trainer uses the VAE checkpoint's original normalization stats by default:

```text
/media/cvpr/haomian/data/SOKE_FLOW/chatsign_175_sentence_word_joint/meta/mean_rot6d.npy
/media/cvpr/haomian/data/SOKE_FLOW/chatsign_175_sentence_word_joint/meta/std_rot6d.npy
```

This is important. If the VAE was trained with different stats from the adapter input, the latent and decoded losses are not comparable.

## Datasets

Sentence dataset:

```text
/media/cvpr/haomian/data/SOKE_FLOW/chatsign_175
```

Word dataset:

```text
/media/cvpr/haomian/data/SOKE_FLOW/chatsign_175_word
```

The current ChatSign setup intentionally uses the same 169 sentence samples for train, val, and test. This is useful for testing whether the pipeline can fit the paired word-prior to sentence-motion mapping, but it is not a held-out generalization benchmark.

## ContentStyleAdapter

Implemented in:

```text
flow/content_style_adapter.py
```

Input:

```text
z:    [B, L, Dz]
mask: [B, L]
```

For the current VAE:

```text
Dz = 256
L  = ceil(T / 4)
```

Architecture:

```text
z
  -> Linear(Dz, hidden_dim)
  -> learned temporal position embedding
  -> TransformerEncoder
  -> LayerNorm
  -> residual delta head
  -> z_adapt = z + delta
```

The adapter also has content and style branches:

```text
content_tokens = content_head(h)
content_pooled = masked_mean(content_tokens)

style_tokens = style_head(h)
style_pooled = masked_mean(style_tokens)
style_logits = style_classifier(style_pooled)
```

Forward output:

```text
z_adapt
delta
content_tokens
content_pooled
style_tokens
style_pooled
style_logits
```

Important detail: the final residual delta layer is zero-initialized, so the adapter starts close to identity:

```text
z_adapt ~= z_source
```

For `prior_mode=concat`, `z_source` is the encoded word-concat prior. For `prior_mode=soft_arranger`, `z_source` is the soft-arranged latent prior.

## Concat Prior Mode

The concat mode uses the existing lexical word prior:

```text
sentence text
  -> WordMotionPrior.match_text
  -> matched word clips
  -> concatenate word poses
  -> resample to target sentence length T
  -> x_word
  -> frozen VAE encoder
  -> z_word
  -> ContentStyleAdapter
  -> z_adapt
```

This is simple and strong for in-dataset fitting, but it assumes the lexical matches and their concatenation order are useful. It does not solve sign ordering, word omission, coarticulation, or synonym/gloss planning.

## Soft Arranger Prior Mode

Implemented in:

```text
flow/temporal_word_attention.py
```

For a deeper design and debugging guide, see:

```text
docs/flow/adapter/soke_soft_word_arranger.md
```

The soft arranger replaces fixed concatenation with unordered word-memory attention:

```text
sentence text
  + matched word text features
  + matched/random word motion latents
  -> SoftWordArranger
  -> z_prior_aligned
  -> ContentStyleAdapter
  -> z_adapt
```

### Candidate Builder

`WordCandidateBuilder` reuses the lexical matching rules from `WordMotionPrior`.

For each sentence:

```text
positive candidates = lexical word matches from the sentence
negative candidates = random word clips from the word dictionary
candidate order     = shuffled during training
```

Current defaults:

```text
num_word_candidates     = 32
num_negative_candidates = 16
candidate_selection     = flat
max_positive_variants_per_key = 0
shuffle_word_candidates = true
```

For word dictionaries with many motion variants per lexicon key, prefer:

```bash
--candidate-selection round_robin \
--max-positive-variants-per-key 2
```

This prevents one high-variant word from consuming the whole positive candidate
budget before other matched words are included.

If fewer candidates exist, the batch is padded with masked slots. Candidate order has no positional embedding, so the model is encouraged to treat candidates as a set/memory bank.

### Text Features

The arranger uses the frozen local T5 encoder:

```text
flow/text_encoder.py::FrozenT5TextEncoder
deps/flan-t5-base
```

Features:

```text
sentence_text_feature = T5(sentence text)  -> [B, 768]
word_text_feature     = T5(word label)     -> [B, K, 768]
```

T5 is frozen and its weights are not saved in adapter checkpoints.

### Word Motion Features

Each candidate word clip is encoded through the frozen VAE:

```text
word compact SMPL-X
  -> rot6d conversion and VAE normalization
  -> VAE encoder
  -> deterministic mu latent
  -> adapter latent normalization
  -> z_word_clip [Lw, Dz]
```

For the current VAE:

```text
Dz = 256
Lw <= max_word_latent_frames
```

### SoftWordArranger

Inputs:

```text
sentence_text:      [B, Dt]
word_text:          [B, K, Dt]
word_latents:       [B, K, Lw, Dz]
word_latent_mask:   [B, K, Lw]
candidate_mask:     [B, K]
target_latent_mask: [B, Ls]
```

Outputs:

```text
z_prior_aligned: [B, Ls, Dz]
attention:       [B, Ls, K, Lw]
null_attention:  [B, Ls]
word_gate_logits:[B, K]
word_gate_probs: [B, K]
word_usage:      [B, K]
null_usage:      [B]
```

The target query is:

```text
query[t] = target position embedding[t] + projected sentence text
```

The word memory token is:

```text
memory[k,u] =
  projected word latent[k,u]
  + projected word text[k]
  + local word phase embedding[u]
```

The candidate gate is an MLP over:

```text
sentence text
word text
pooled word motion
sentence_text * word_text
```

The custom multi-head attention score is:

```text
score[t,k,u] =
  query[t] dot key[k,u] / sqrt(head_dim)
  + log(sigmoid(gate[k]) + eps)
  + mask
```

A learned NULL memory token is appended, so the arranger can leave a target time step unsupported by candidate words.

## Training Flow

Implemented in:

```text
flow/train_adapter.py
```

Shared setup:

```text
1. Load sentence motion x_sent from UpperSMPLXFlowDataset.
2. Convert/normalize to the VAE rotation representation.
3. Encode x_sent with the frozen VAE:
     z_sent_raw = VAE.encode(x_sent).mu
4. Normalize latents using train-split latent mean/std:
     z_sent = (z_sent_raw - latent_mean) / latent_std
```

Concat mode source:

```text
1. Build x_word with WordMotionPrior.
2. Encode x_word with the frozen VAE.
3. Normalize to z_word.
4. Use z_source = z_word.
```

Soft arranger source:

```text
1. Build unordered word candidates.
2. Encode each candidate word clip with the frozen VAE.
3. Encode sentence text and word labels with frozen T5.
4. Run SoftWordArranger to produce z_prior_aligned.
5. Use z_source = z_prior_aligned.
```

Adapter step:

```text
z_source -> ContentStyleAdapter -> z_adapt
denormalize z_adapt -> frozen VAE decoder -> x_adapt
```

The latent mean/std are computed once from the train split and saved in:

```text
checkpoint["latent_config"]["stats"]
```

## Loss Design

Base adapter losses:

```text
loss =
  latent_loss_weight       * SmoothL1(z_adapt, z_sent)
  + pose_loss_weight       * SmoothL1(x_adapt, x_sent)
  + velocity_loss_weight   * velocity_loss(x_adapt, x_sent)
  + accel_loss_weight      * acceleration_loss(x_adapt, x_sent)
  + jerk_loss_weight       * jerk_loss(x_adapt, x_sent)
  + style_loss_weight      * style_domain_ce
  + content_pair_weight    * SmoothL1(content_pooled_source, content_pooled_sent)
  + delta_loss_weight      * MSE(delta, 0)
  + orth_loss_weight       * content_style_covariance_penalty
  + content_domain_weight  * gradient_reversal_content_domain_ce
```

Default base weights:

```text
latent_loss_weight       = 1.0
pose_loss_weight         = 0.5
velocity_loss_weight     = 0.5
accel_loss_weight        = 0.25
jerk_loss_weight         = 0.1
style_loss_weight        = 0.1
content_pair_loss_weight = 0.1
delta_loss_weight        = 0.001
orth_loss_weight         = 0.0
content_domain_confusion_loss_weight = 0.0
gradient_reversal_lambda = 1.0
```

Temporal losses are computed on decoded normalized motion:

```text
velocity     = x[t+1] - x[t]
acceleration = x[t+2] - 2*x[t+1] + x[t]
jerk         = x[t+3] - 3*x[t+2] + 3*x[t+1] - x[t]
```

Pose loss is feature-weighted:

```text
hand_weight       = 5.0
jaw_weight        = 2.0
expression_weight = 2.0
```

Additional soft-arranger losses:

```text
arranger_prior_loss = SmoothL1(z_prior_aligned, z_sent)
gate_bce_loss       = BCE(word_gate_logits, weak lexical positive/negative labels)
gate_sparsity_loss  = mean(word_gate_probs over valid candidates)
null_usage_loss     = mean(null_usage)
group_coverage_loss = unordered lower-bound usage over matched word occurrences
group_entropy_peak = per-frame entropy hinge over matched occurrence groups
attention_variation = hinge loss against frozen frame-to-frame attention
prior_velocity_loss = SmoothL1(diff(z_prior_aligned), diff(z_sent))
prior_accel_loss    = SmoothL1(diff2(z_prior_aligned), diff2(z_sent))
variance_floor_loss = prevents z_prior_aligned temporal std collapse
negative_usage_loss = mean attention usage on random negative candidates
```

Default arranger weights:

```text
arranger_prior_loss_weight  = 1.0
gate_bce_loss_weight        = 0.1
gate_sparsity_loss_weight   = 0.01
attention_smoothness_weight = 0.0
null_usage_loss_weight      = 0.01
group_coverage_loss_weight  = 0.02
group_coverage_mass         = 0.5
group_entropy_peak_loss_weight = 0.0
group_entropy_peak_target      = 0.6931471805599453  # log(2)
attention_variation_loss_weight = 0.01
attention_variation_target      = 0.05
prior_velocity_loss_weight      = 0.25
prior_accel_loss_weight         = 0.05
prior_variance_floor_loss_weight = 0.05
prior_variance_floor_ratio       = 0.5
negative_usage_loss_weight       = 0.02
```

The group entropy peakiness loss encourages a clear matched occurrence-group choice at each target frame by penalizing group entropy above the target. It does not impose English lexical order; it only discourages per-frame averaging over many groups.

The coverage and variation losses intentionally do not impose English lexical order, because sign-language order may differ from the written sentence order.

## Checkpointing

The trainer writes:

```text
config.json
checkpoints/last.pt
checkpoints/epoch_XXXX.pt
checkpoints/best.pt
checkpoints/best_01.pt
checkpoints/best_02.pt
checkpoints/best_03.pt
checkpoints/best_top_k.json
```

The best checkpoints are selected by validation latent loss. Lower is better. `--save-top-k 3` keeps the best three validation checkpoints.

Soft-arranger checkpoints additionally store:

```text
adapter_model
arranger_model
arranger_config
text_config
candidate_config
latent_config
loss_config
```

Frozen VAE and frozen T5 weights are not saved inside adapter checkpoints.

## Slurm Training

Main launcher:

```text
scripts/flow/train_adapter_sbatch.sh
```

Default paths:

```text
data_dir       = /media/cvpr/haomian/data/SOKE_FLOW/chatsign_175
word_data_dir  = /media/cvpr/haomian/data/SOKE_FLOW/chatsign_175_word
vae_checkpoint = experiments/flow/VAE/chatsign175_sentence_word_joint_rot6d_vae_b32/checkpoints/best.pt
batch_size     = 16
epochs         = 1000
```

### Concat Adapter

```bash
sbatch scripts/flow/train_adapter_sbatch.sh \
  --run-name chatsign175_adapter_jointvae_b16_online \
  --data-dir /media/cvpr/haomian/data/SOKE_FLOW/chatsign_175 \
  --word-data-dir /media/cvpr/haomian/data/SOKE_FLOW/chatsign_175_word \
  --vae-checkpoint experiments/flow/VAE/chatsign175_sentence_word_joint_rot6d_vae_b32/checkpoints/best.pt \
  --prior-mode concat \
  --batch-size 16 \
  --epochs 1000 \
  --val-every 10 \
  --save-every 100 \
  --save-last-every 10 \
  --save-top-k 3 \
  --wandb-online
```

### Soft Word Arranger Adapter

This is the current soft-arranger training command:

```bash
sbatch scripts/flow/train_adapter_sbatch.sh \
  --run-name chatsign175_soft_arranger_adapter_b16_online \
  --data-dir /media/cvpr/haomian/data/SOKE_FLOW/chatsign_175 \
  --word-data-dir /media/cvpr/haomian/data/SOKE_FLOW/chatsign_175_word \
  --vae-checkpoint experiments/flow/VAE/chatsign175_sentence_word_joint_rot6d_vae_b32/checkpoints/best.pt \
  --prior-mode soft_arranger \
  --text-model-path deps/flan-t5-base \
  --num-word-candidates 32 \
  --num-negative-candidates 16 \
  --candidate-selection round_robin \
  --max-positive-variants-per-key 2 \
  --batch-size 16 \
  --epochs 1000 \
  --val-every 10 \
  --save-every 100 \
  --save-last-every 10 \
  --save-top-k 3 \
  --wandb-online
```

### Ablation Modes

Use these four cells to separate the contribution of the unordered soft arranger from the content-style adapter:

```text
full model
  --prior-mode soft_arranger

disable soft arranger only
  --prior-mode concat
  # equivalent explicit alias: --disable-softarranger
  # WordMotionPrior.match_text uses the first available variant for each matched lexicon key.

disable adapter only
  --prior-mode soft_arranger
  --disable-adapter
  # z_adapt is bypassed as z_prior_aligned; only the arranger is trainable.

disable both
  --prior-mode concat
  --disable-adapter
  # pure deterministic word-concat prior; no trainable parameters.
```

The `--disable-adapter` path decodes the source latent prior directly:

```text
adapter enabled:  z_source -> ContentStyleAdapter -> z_adapt
adapter disabled: z_source -----------------------> z_adapt
```

When both the soft arranger and adapter are disabled, training is only a metric/logging baseline because there is no learnable module left.

### Resume Training

The `--epochs` value is the total target epoch, not the number of additional epochs. To train 1000 more epochs after a run has finished epoch 1000, resume with `--epochs 2000`.

```bash
sbatch scripts/flow/train_adapter_sbatch.sh \
  --run-name chatsign175_soft_arranger_adapter_b16_online \
  --data-dir /media/cvpr/haomian/data/SOKE_FLOW/chatsign_175 \
  --word-data-dir /media/cvpr/haomian/data/SOKE_FLOW/chatsign_175_word \
  --vae-checkpoint experiments/flow/VAE/chatsign175_sentence_word_joint_rot6d_vae_b32/checkpoints/best.pt \
  --prior-mode soft_arranger \
  --text-model-path deps/flan-t5-base \
  --num-word-candidates 32 \
  --num-negative-candidates 16 \
  --batch-size 16 \
  --epochs 2000 \
  --val-every 10 \
  --save-every 100 \
  --save-last-every 10 \
  --save-top-k 3 \
  --resume-from-checkpoint experiments/flow/adapter/chatsign175_soft_arranger_adapter_b16_online/checkpoints/last.pt \
  --wandb-online \
  --wandb-id 7dc9fwu2 \
  --wandb-resume must
```

The current resumed job printed:

```text
Resumed from .../checkpoints/last.pt (epoch=1000, global_step=10000, optimizer=yes)
epoch=1001 ...
```

Current W&B project:

```text
soke-flow-adapter
```

Current soft-arranger W&B run:

```text
https://wandb.ai/hh3443-new-york-university/soke-flow-adapter/runs/7dc9fwu2
```

Do not put API keys into documentation. Use `--wandb-online` with the existing launcher key pattern, or pass a key through a private file with `--wandb-api-key-file`.

## Evaluation

Implemented in:

```text
flow/evaluate_adapter.py
```

Evaluation loads:

```text
adapter checkpoint
frozen VAE checkpoint
sentence dataset split
word dataset
latent stats saved inside the adapter checkpoint
```

Concat checkpoint evaluation:

```bash
/media/cvpr/haomian/python_envs/soke/bin/python -m flow.evaluate_adapter \
  --checkpoint experiments/flow/adapter/chatsign175_adapter_jointvae_b16_online/checkpoints/best.pt \
  --split test \
  --index 0 \
  --num_samples 169 \
  --out_dir visualize/adapter_chatsign175_best_test \
  --device cuda
```

Soft-arranger checkpoint evaluation:

```bash
/media/cvpr/haomian/python_envs/soke/bin/python -m flow.evaluate_adapter \
  --checkpoint experiments/flow/adapter/chatsign175_soft_arranger_adapter_b16_online/checkpoints/best.pt \
  --split test \
  --index 0 \
  --num_samples 169 \
  --out_dir visualize/adapter_soft_arranger_best_test \
  --device cuda
```

For permutation robustness, evaluate with shuffled candidates:

```bash
/media/cvpr/haomian/python_envs/soke/bin/python -m flow.evaluate_adapter \
  --checkpoint experiments/flow/adapter/chatsign175_soft_arranger_adapter_b16_online/checkpoints/best.pt \
  --split test \
  --index 0 \
  --num_samples 169 \
  --out_dir visualize/adapter_soft_arranger_best_test_shuffle \
  --device cuda \
  --shuffle_word_candidates \
  --candidate_seed 123
```

### Evaluation Outputs

Concat mode writes:

```text
gt_XX.npz
word_prior_XX.npz
adapted_XX.npz
metrics.json
```

Soft-arranger mode writes:

```text
gt_XX.npz
raw_concat_prior_XX.npz
soft_arranger_prior_XX.npz
adapted_XX.npz
attention_XX.npz
metrics.json
```

For soft-arranger mode, `adapted_XX.npz` stores `coarse_motion` and `coarse_smplx` as the soft-arranger prior. That means the existing three-way renderer shows:

```text
GT | adapted output | soft-arranger prior
```

The raw concat prior is also saved separately as `raw_concat_prior_XX.npz`.

`attention_XX.npz` contains:

```text
attention
word_gate_probs
word_usage
null_usage
candidate_mask
candidate_labels
candidate_group_ids
group_mask
group_usage
group_entropy
group_peak_prob
group_texts
attention_temporal_l1
attention_frame_cosine
prior_latent_std
gt_latent_std
prior_delta_rms
gt_delta_rms
candidate_names
candidate_stats
```

Use this file to inspect which word candidates were selected or ignored, whether attention is frozen over time, and whether the arranger prior has collapsed to low temporal variance.

## Visualization

Two-way comparison:

```bash
/media/cvpr/haomian/python_envs/soke/bin/python -m flow.visualize.visualize_compare_npz \
  --gt_npz visualize/adapter_soft_arranger_best_test/gt_00.npz \
  --pred_npz visualize/adapter_soft_arranger_best_test/adapted_00.npz \
  --out_dir visualize/adapter_soft_arranger_best_test/render \
  --view_transform none
```

Three-way comparison:

```bash
/media/cvpr/haomian/python_envs/soke/bin/python -m flow.visualize.visualize_compare_three_npz \
  --gt_npz visualize/adapter_soft_arranger_best_test/gt_00.npz \
  --pred_npz visualize/adapter_soft_arranger_best_test/adapted_00.npz \
  --out_dir visualize/adapter_soft_arranger_best_test/render_three \
  --view_transform none
```

For `adapted_00.npz`, the third panel is loaded from `coarse_smplx`, which is the soft-arranger prior in soft-arranger mode.

To visualize the raw concat prior directly, compare `raw_concat_prior_00.npz` as the prediction file.

## Current Observations

The concat adapter can strongly overfit the 169 paired ChatSign samples. In the previous concat run, the adapter reduced latent, pose, velocity, acceleration, and jerk distances against the raw word prior on nearly every sample.

The soft arranger is a harder but more useful setup:

```text
raw concat latent prior  -> order-sensitive baseline
soft arranger prior      -> unordered word-memory prior
adapted output           -> final adapter correction
```

Early soft-arranger training showed that `z_prior_aligned` was already much closer to `z_sent` than raw word-concat latents, and the run was still improving near epoch 1000. Therefore the current experiment was resumed to epoch 2000.

## Caveats And Next Checks

The present ChatSign split is not held out:

```text
train = val = test = 169 samples
```

So the adapter result should be read as in-dataset fitting and robustness testing, not generalization.

Recommended checks after training finishes:

```text
1. Evaluate best.pt on all 169 samples.
2. Compare raw_concat_prior vs soft_arranger_prior vs adapted output.
3. Run candidate-shuffle evaluation and check that metrics stay close.
4. Inspect attention_XX.npz for word usage, gate probabilities, and null usage.
5. Render several three-way videos with view_transform=none.
```

Acceptance target for the soft arranger:

```text
prior_latent_l1 < word_latent_l1
adapted_latent_l1 < prior_latent_l1
candidate-order shuffle changes attention layout but not metrics much
```
