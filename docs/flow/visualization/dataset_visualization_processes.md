# Dataset Visualization Processes

This note summarizes the current five-way visualization process for PHOENIX,
CSL-Daily, and How2Sign. It complements
`docs/flow/visualization/flow_visualize_usage.md`, which documents the
individual visualizer options.

Run from the SOKE project root:

```bash
cd /media/cvpr/haomian/NeuralImplicitSignMotionTrajectory

export PYTHONNOUSERSITE=1
export PYTHONPATH=/media/cvpr/haomian/NeuralImplicitSignMotionTrajectory
export SOKE_PY=/media/cvpr/haomian/python_envs/soke/bin/python
```

## Shared Protocol

The standard inspection layout is:

```text
Original frames | Ground truth | Word Prior | SOKE | Flow matching
```

The intended meaning of each panel:

| Panel | Source |
| --- | --- |
| Original frames | Dataset RGB frames or source video. |
| Ground truth | GT sentence SMPL-X from the prepared flow dataset. |
| Word Prior | Raw word-prior concatenation, before adapter and flow. |
| SOKE | Original SOKE pipeline generation exported as vertices. |
| Flow matching | Flow checkpoint output. |

Use:

```bash
--background_color white \
--panel_order source gt prior soke pred \
--resample_to_source_frames \
--upper_body_only \
--view_transform none
```

`--resample_to_source_frames` makes the RGB source frame count the output
timeline. GT, Word Prior, SOKE, and flow sequences are linearly resampled to
that length.

The flow inference condition should be the sentence text. Gloss is supervised
data, not an inference input. For CSL-Daily, `label_word` is a text
segmentation field and may be useful for raw Word Prior matching, but keep the
raw `text` in metadata and do not use `gloss` as the inference condition.

The high-level procedure is the same for all datasets:

1. Select rows from the dataset manifest and save a small JSONL manifest.
2. Run `flow.sample_text_conditional` to generate `sample_XX.npz` and
   `gt_XX.npz`.
3. Build or export the raw Word Prior as a separate `word_prior_XX.npz`.
   Prefer lazy word loading for large word-prior datasets so only matched word
   clips are loaded.
4. Export SOKE vertices with `test/visualize_soke_test.py --save_npz --no_video`.
5. Render with `flow.visualize.visualize_compare_three_npz_with_source`.
6. Check `render_meta.json`, optionally write a compact `sequence_meta.json`,
   and verify MP4s with `ffprobe`.

When the separate Word Prior file contains normal `smplx` / `motion` keys,
pass:

```bash
--prior_smplx_key smplx \
--prior_motion_key motion
```

## Shared Commands

### Sample Flow Outputs

```bash
$SOKE_PY -m flow.sample_text_conditional \
  --checkpoint "$FLOW_CKPT" \
  --data_dir "$DATA_DIR" \
  --word_data_dir "$WORD_DATA_DIR" \
  --word_split "$WORD_SPLIT" \
  --adapter_checkpoint "$ADAPTER_CKPT" \
  --manifest "$MANIFEST" \
  --num_prompts "$N" \
  --match_manifest_lengths \
  --condition_field text \
  --no_negative_candidates \
  --out_dir "$FLOW_OUT" \
  --seed 20260627 \
  --device auto \
  --steps 100 \
  --sampler heun
```

If a checkpoint contains stale `/dev/shm/...` paths, override `--data_dir`,
`--word_data_dir`, and `--adapter_checkpoint` explicitly.

### Export SOKE Vertices

```bash
$SOKE_PY test/visualize_soke_test.py \
  --cfg "$SOKE_CFG" \
  --checkpoint "$SOKE_CKPT" \
  --split "$SPLIT" \
  --sample_name="$SAMPLE_NAMES" \
  --out_dir "$SOKE_OUT" \
  --num_samples "$N" \
  --batch_size 1 \
  --num_workers 0 \
  --use_gpus 0 \
  --save_npz \
  --no_video
```

Use `--sample_name="$SAMPLE_NAMES"` with an equals sign. This matters for
How2Sign names that begin with `--`, because otherwise argparse can interpret
the first sample name as an option.

### Render Five-Way Video

```bash
$SOKE_PY -m flow.visualize.visualize_compare_three_npz_with_source \
  --gt "$FLOW_OUT/gt_${i}.npz" \
  --pred "$FLOW_OUT/sample_${i}.npz" \
  --prior "$WORD_PRIOR_OUT/word_prior_${i}.npz" \
  --prior_smplx_key smplx \
  --prior_motion_key motion \
  --soke_vertices "$SOKE_OUT" \
  --source_root "$SOURCE_ROOT" \
  --source_split "$SOURCE_SPLIT" \
  --source_fit "$SOURCE_FIT" \
  --background_color white \
  --panel_order source gt prior soke pred \
  --resample_to_source_frames \
  --upper_body_only \
  --out_dir "$RENDER_OUT" \
  --fps "$FPS" \
  --width "$WIDTH" \
  --height "$HEIGHT" \
  --software_face_stride 1 \
  --device cpu \
  --view_transform none \
  --source_label "Original frames" \
  --gt_label "Ground truth" \
  --prior_label "Word Prior" \
  --soke_label "SOKE" \
  --pred_label "Flow matching"
```

## PHOENIX

PHOENIX source data:

| Item | Path or value |
| --- | --- |
| RGB frames | `/media/cvpr/haomian/data/phoenix/fullFrame-210x260px` |
| Sentence SMPL-X | `/media/cvpr/haomian/data/SOKE_FLOW/phoenix_upper_smplx` |
| Word prior data | `/media/cvpr/haomian/data/SOKE_FLOW/phoenix_upper_smplx_word_ctc` |
| Flow checkpoint | `experiments/flow/phoenix_latent_adapter_residual_b64x4_all_balanced_online_retry3_ncclif/checkpoints/best.pt` or `last.pt` |
| VAE checkpoint | `experiments/flow/VAE/phoenix_rot6d_vae_jerk_b16x4_online_retry1/checkpoints/best.pt` |
| Adapter checkpoint | `experiments/flow/adapter/phoenix_soft_arranger_adapter_balanced_b64x4_gloss_online/checkpoints/epoch_0300.pt` |
| SOKE checkpoint | `experiments/mgpt/SOKE_soke_lm_2n_b128_online_139176_1/checkpoints/min-phoenix_DTW_MPJPE_PA_lhandepoch=84.ckpt` |
| SOKE config | `experiments/mgpt/SOKE_soke_lm_2n_b128_online_139176_1/config_soke_lm_2n_b128_online_139176_train.yaml` |
| Word split | `all.balanced` for flow sampling; adapter checkpoint was trained with a balanced train split |
| Flow condition | `text` |
| Source split | `train`, `dev`, or `test` |
| Source fit | `cover` |
| FPS | `25` for PHOENIX RGB frame timeline |

PHOENIX frame lookup:

```text
/media/cvpr/haomian/data/phoenix/fullFrame-210x260px/<split>/<sample_name>/*.png
```

Example training sample:

```text
/media/cvpr/haomian/data/phoenix/fullFrame-210x260px/train/11August_2010_Wednesday_tagesschau-1/*.png
```

Notes:

- Some manifest rows store `source_name` as `train/<sample_name>`. The source
  visualizer also tries the basename, so `--source_root ...fullFrame-210x260px`
  plus `--source_split train` works.
- Existing generated batches used white background, `512x512` panels, and
  upper-body-only mesh.
- If reproducing older adapter-debug visualizations, check whether the adapter
  prior was built from gloss. For final text-to-sign visualization, pass
  `--condition_field text` for the flow generation.

## CSL-Daily

CSL-Daily source data:

| Item | Path or value |
| --- | --- |
| RGB frames | `/media/cvpr/haomian/data/CSL-Daily/sentences/frames_512x512` |
| Sentence SMPL-X | `/media/cvpr/haomian/data/SOKE_FLOW/csl_daily_upper_smplx` |
| Word prior data | `/media/cvpr/haomian/data/SOKE_FLOW/csl_daily_upper_smplx_word_ctc` |
| Flow checkpoint | `experiments/flow/csl_daily_latent_adapter_residual_2xb64/checkpoints/best.pt` or `last.pt` |
| VAE checkpoint | `experiments/flow/VAE/csl_daily_rot6d_vae_jerk_b16x4_online/checkpoints/best.pt` |
| Adapter checkpoint | `experiments/flow/adapter/csl_daily_soft_arranger_lw_2xb256/checkpoints/last.pt` |
| SOKE checkpoint | `experiments/mgpt/SOKE_soke_lm_2n_b128_online_139176_1/checkpoints/min-csl_DTW_MPJPE_PA_lhandepoch=84.ckpt` |
| SOKE config | `experiments/mgpt/SOKE_soke_lm_2n_b128_online_139176_1/config_soke_lm_2n_b128_online_139176_train.yaml` |
| Word split | `all` |
| Flow condition | `text` for final inference; `label_word` only for tokenized-text/debug reproduction |
| Source split | empty |
| Source fit | `cover` |
| FPS | `25` |

CSL-Daily frame lookup:

```text
/media/cvpr/haomian/data/CSL-Daily/sentences/frames_512x512/<sample_name>/*.jpg
```

Use:

```bash
SOURCE_ROOT=/media/cvpr/haomian/data/CSL-Daily/sentences/frames_512x512
SOURCE_SPLIT=
FPS=25
WIDTH=512
HEIGHT=512
SOURCE_FIT=cover
```

Notes:

- The source frames are already square `512x512`, so `cover` does not create
  the same crop concern as How2Sign.
- The checkpoint may contain stale `/dev/shm/...` paths. Always override
  `--data_dir` and `--word_data_dir` with the persistent paths above.
- Some source clips can be missing even when the manifest row exists. In the
  previous train batch, `S000005_P0004_T00` had no matching source frames or
  MP4, so it was skipped and replaced with the next available row.
- `label_word` is a segmented text field and can improve raw Word Prior
  matching. Keep the raw `text` and `gloss` metadata in `render_meta.json` /
  `sequence_meta.json` so each video is auditable.

## How2Sign

How2Sign source data:

| Item | Path or value |
| --- | --- |
| RGB videos | `/media/cvpr/haomian/data/how2sign/sentence_level` |
| Sentence SMPL-X | `/media/cvpr/haomian/data/SOKE_FLOW/how2sign_soke_upper_smplx` |
| Word prior data | `/media/cvpr/haomian/data/SOKE_FLOW/how2sign_soke_upper_smplx_word_signasl` |
| Flow checkpoint | `experiments/flow/how2sign_latent_adapter_residual_signasl_all_b128x4_online_retry2/checkpoints/best.pt` |
| VAE checkpoint | `experiments/flow/VAE/how2sign_rot6d_vae_jerk_b128x4/checkpoints/best.pt` |
| Adapter checkpoint | `experiments/flow/adapter/how2sign_soft_arranger_signasl_all_b256x4_online_r3_gloo/checkpoints/best.pt` |
| SOKE checkpoint | `experiments/mgpt/SOKE_soke_lm_2n_b128_online_139176_1/checkpoints/min-how2sign_DTW_MPJPE_PA_lhandepoch=89.ckpt` |
| SOKE config | `experiments/mgpt/SOKE_soke_lm_2n_b128_online_139176_1/config_soke_lm_2n_b128_online_139176_train.yaml` |
| Word split | `all` |
| Flow condition | `text` |
| Source split | `train/rgb_front/raw_videos`, `val/rgb_front/raw_videos`, or `test/rgb_front/raw_videos` |
| Source fit | current baseline: `cover` |
| FPS | use `24` for most clips, or source-specific FPS from manifest / `ffprobe` |

How2Sign video lookup:

```text
/media/cvpr/haomian/data/how2sign/sentence_level/<split>/rgb_front/raw_videos/<sample_name>.mp4
```

Use:

```bash
SOURCE_ROOT=/media/cvpr/haomian/data/how2sign/sentence_level
SOURCE_SPLIT=train/rgb_front/raw_videos
FPS=24
WIDTH=512
HEIGHT=512
SOURCE_FIT=cover
```

Notes:

- How2Sign source videos are usually `1280x720`, while the mesh panels are
  square. Do not directly resize the full frame to `512x512`, because that
  distorts the signer. The current baseline uses `--source_fit cover`, which
  center-crops the 16:9 frame to a square and then resizes it with no aspect
  distortion.
- The baseline crop is not yet signer-aware. If hands are cut off, add a
  future source crop mode that uses a fixed per-sequence person/keypoint box
  with margin, then resizes to the mesh panel.
- Source FPS is mixed across How2Sign: common values include 23, 24, 30, 50,
  and 60 fps. The prepared motion is 20 fps. With
  `--resample_to_source_frames`, the renderer uses the MP4 frame count as the
  timeline, so the visual comparison remains aligned even when the output FPS
  is set to a constant value for inspection.
- Every manifest entry in
  `/media/cvpr/haomian/data/SOKE_FLOW/how2sign_soke_upper_smplx/meta`
  had a matching source MP4 during the coverage check.
- Use `--sample_name="$SAMPLE_NAMES"` when exporting SOKE vertices because
  many How2Sign names begin with `--`.

The first How2Sign baseline crop batch was written to:

```text
visualize/how2sign_fiveway_train_rows009_018_cover_baseline/rendered_cover
```

It used rows 9-18 from the train manifest, raw Word Prior with lazy word
loading, SOKE epoch 89, center-crop `cover`, white background, and `512x512`
panels.

## Verification Checklist

After rendering:

```bash
find "$RENDER_OUT" -maxdepth 1 -name '*.mp4' | wc -l

ffprobe -v error -select_streams v:0 \
  -show_entries stream=width,height,nb_frames,r_frame_rate,duration \
  -of default=noprint_wrappers=1 \
  "$RENDER_OUT/sample_00_source_gt_word_soke_flow.mp4"
```

Inspect:

```text
<render_out>/render_meta.json
<render_out>/sequence_meta.json
```

The metadata should record the sequence name, raw text, gloss if present,
condition field, source frame count, original mesh sequence lengths, and final
output frame count.
