# Flow Visualization Usage

This document explains how to run the visualization programs in:

```text
flow/visualize/
```

Run commands from the SOKE project root:

```bash
cd /media/cvpr/haomian/NeuralImplicitSignMotionTrajectory

export PYTHONNOUSERSITE=1
export PYTHONPATH=/media/cvpr/haomian/NeuralImplicitSignMotionTrajectory
export SOKE_PY=/media/cvpr/haomian/python_envs/soke/bin/python
```

The examples below use the software renderer on CPU. This is slower than GPU rendering, but it is reliable on the cluster.

## Output Layout

Current visualization outputs are organized by pipeline:

```text
visualize/flow/    # Flow matching, adapter, VAE, Word Prior, and five-way comparison outputs.
visualize/soke/    # Original SOKE pipeline exports and SOKE-only render checks.
```

Dataset-only, SMPL-X-only, source-panel smoke tests, and other ambiguous folders may remain directly under `visualize/`.

## Entry Points

| Module | Use case |
| --- | --- |
| `flow.visualize.visualize_npz` | Render one or more `.npz` motion files as individual MP4 videos. |
| `flow.visualize.visualize_compare_npz` | Render ground truth and prediction side by side. |
| `flow.visualize.visualize_compare_three_npz` | Render ground truth, flow prediction, and Word Prior side by side. |
| `flow.visualize.visualize_compare_three_npz_with_source` | Render original frames, ground truth, Word Prior, optional SOKE output, and flow prediction in one aligned video. |
| `flow.visualize.visualize_posfix_contact_sentences` | Build sentence poses from `/media/cvpr/haomian/data/posfix_contact` word tracking files and render upper-body sentence videos. |
| `test/visualize_soke_test.py` | Export SOKE generated vertices as `.npz` files for the source-aware comparison visualizer. |

## Common Options

| Option | Recommended value | Notes |
| --- | --- | --- |
| `--device` | `cpu` | Reliable for rendering. Use `cuda` or `auto` only after checking the environment. |
| `--model_dir` | `deps/smpl_models` | Must contain the SMPL-X model files. Sentence upper-body rendering also needs `smplx_vert_segmentation.json`. |
| `--fps` | `20` | Matches the current sentence visualization outputs. |
| `--width` / `--height` | `512` / `512` | Panel size before the header is added. |
| `--software_face_stride` | `1` | Use `1` for final videos. Higher values are faster but create mesh holes. |
| `--max_frames` | `0` | `0` renders the full sequence. Use a small number for smoke tests. |
| `--view_transform` | `none` | Use `none` for canonical generated samples and current posfix-contact sentence videos. |

For final inspection videos, prefer:

```bash
--width 512 \
--height 512 \
--software_face_stride 1 \
--device cpu
```

For the current source-aware PHOENIX inspection protocol, prefer:

```bash
--source_fit cover \
--background_color white \
--panel_order source gt prior soke pred \
--resample_to_source_frames \
--upper_body_only
```

This gives the layout:

```text
Original frames | Ground truth | Word Prior | SOKE | Flow matching
```

With `--resample_to_source_frames`, the original RGB frame count is the video timeline. GT, word-prior, SOKE, and flow mesh sequences are linearly resampled to exactly that length.

For quick smoke tests, use smaller panels and a frame cap:

```bash
--width 256 \
--height 256 \
--max_frames 20
```

## Input NPZ Format

The generic `.npz` visualizers can read either:

| Key | Shape | Meaning |
| --- | --- | --- |
| `smplx` | `[T, 182]` | Full SMPL-X parameter sequence. |
| `motion` | `[T, 133]` | Compact SOKE motion representation, converted to `[T, 182]` internally. |

`visualize_compare_three_npz` uses these defaults:

| Panel | Default keys |
| --- | --- |
| Ground truth | `motion`, or `smplx` if present. |
| Flow prediction | `motion`, or `smplx` if present. |
| Word Prior | `coarse_motion`, or `coarse_smplx` if present. |

Override the keys with options such as `--pred_smplx_key`, `--prior_smplx_key`, or `--prior_motion_key` when needed.

The source-aware visualizer also uses the sample name to find matching original frames and SOKE exports. It checks these keys in the prediction `.npz`:

```text
name, source_name, sequence_name, video_id, id
```

You can override the lookup with `--source_name`.

## Render Single NPZ Files

Use `visualize_npz` when you only want to inspect generated samples one at a time.

```bash
$SOKE_PY -m flow.visualize.visualize_npz \
  --input visualize/flow_chatsign175_residual_epoch1300_test_samples/sample_00.npz \
  --out_dir visualize/flow_npz \
  --fps 20 \
  --width 512 \
  --height 512 \
  --software_face_stride 1 \
  --device cpu \
  --view_transform none
```

You can pass a directory instead of a single file. The script renders `sample_*.npz` files first.

Output name:

```text
<out_dir>/<input_stem>.mp4
```

## Render GT vs Prediction

Use `visualize_compare_npz` for a two-panel comparison.

```bash
$SOKE_PY -m flow.visualize.visualize_compare_npz \
  --gt visualize/flow_chatsign175_residual_epoch1300_test_samples/gt_00.npz \
  --pred visualize/flow_chatsign175_residual_epoch1300_test_samples/sample_00.npz \
  --out_dir visualize/flow_compare_pair \
  --fps 20 \
  --width 512 \
  --height 512 \
  --software_face_stride 1 \
  --device cpu \
  --view_transform none \
  --end_mode blank
```

Panels:

| Side | Content |
| --- | --- |
| Left | Ground truth |
| Right | Flow prediction |

If the sequences have different lengths, the video length is the longer sequence. With `--end_mode blank`, the shorter side becomes black after it ends. With `--end_mode hold`, the shorter side freezes on its last frame.

Output name:

```text
<out_dir>/<pred_stem>_vs_gt.mp4
```

## Render GT vs Flow vs Word Prior

Use `visualize_compare_three_npz` for residual-flow outputs that contain a Word Prior.

```bash
$SOKE_PY -m flow.visualize.visualize_compare_three_npz \
  --gt visualize/flow_chatsign175_residual_epoch1300_test_samples/gt_00.npz \
  --pred visualize/flow_chatsign175_residual_epoch1300_test_samples/sample_00.npz \
  --out_dir visualize/flow_compare_three \
  --fps 20 \
  --width 512 \
  --height 512 \
  --software_face_stride 1 \
  --device cpu \
  --view_transform none \
  --end_mode blank
```

Panels:

| Panel | Content |
| --- | --- |
| Left | Ground truth |
| Middle | Flow prediction |
| Right | Word Prior |

By default, the prior is read from the same `.npz` as `--pred`, using `coarse_smplx` or `coarse_motion`. To use a separate prior file:

```bash
$SOKE_PY -m flow.visualize.visualize_compare_three_npz \
  --gt path/to/gt.npz \
  --pred path/to/sample.npz \
  --prior path/to/prior.npz \
  --out_dir visualize/flow_compare_three \
  --software_face_stride 1 \
  --device cpu
```

Output name:

```text
<out_dir>/<pred_stem>_gt_flow_word.mp4
```

## Render Source-Aware Five-Panel Videos

Use `visualize_compare_three_npz_with_source` when you want to compare generated motion against the original PHOENIX RGB frames and the original SOKE pipeline output.

Current recommended layout:

```text
Original frames | Ground truth | Word Prior | SOKE | Flow matching
```

The PHOENIX original frames are stored under:

```text
/media/cvpr/haomian/data/phoenix/fullFrame-210x260px
```

For a training sample named `11August_2010_Wednesday_tagesschau-1`, the visualizer expects frames at:

```text
/media/cvpr/haomian/data/phoenix/fullFrame-210x260px/train/11August_2010_Wednesday_tagesschau-1/*.png
```

### Step 1: Export SOKE Vertices

The five-panel renderer reads SOKE output from vertex `.npz` files. Export those first with `test/visualize_soke_test.py`.

Example checkpoint:

```text
experiments/mgpt/SOKE_soke_lm_2n_b128_online_139176_1/checkpoints/min-phoenix_DTW_MPJPE_PA_lhandepoch=84.ckpt
```

Matching config:

```text
experiments/mgpt/SOKE_soke_lm_2n_b128_online_139176_1/config_soke_lm_2n_b128_online_139176_train.yaml
```

Example export for five selected training sequences:

```bash
SOKE_EXP=experiments/mgpt/SOKE_soke_lm_2n_b128_online_139176_1
SOKE_OUT=visualize/soke_min_phoenix_epoch84_train_5
SAMPLE_NAMES=11August_2010_Wednesday_tagesschau-1,11August_2010_Wednesday_tagesschau-4,11August_2010_Wednesday_tagesschau-5,11August_2010_Wednesday_tagesschau-6,11August_2010_Wednesday_tagesschau-7

$SOKE_PY test/visualize_soke_test.py \
  --cfg "$SOKE_EXP/config_soke_lm_2n_b128_online_139176_train.yaml" \
  --checkpoint "$SOKE_EXP/checkpoints/min-phoenix_DTW_MPJPE_PA_lhandepoch=84.ckpt" \
  --split train \
  --sample_name "$SAMPLE_NAMES" \
  --out_dir "$SOKE_OUT" \
  --num_samples 5 \
  --batch_size 1 \
  --num_workers 0 \
  --use_gpus 0 \
  --save_npz \
  --no_video
```

Important options:

| Option | Meaning |
| --- | --- |
| `--sample_name` | Comma-separated exact sequence names. This keeps the export focused on only the samples you will render. |
| `--save_npz` | Saves vertices for later combined visualization. |
| `--no_video` | Skips the local two-panel SOKE MP4 and only writes `.npz` files. |

Each exported file contains:

| Key | Meaning |
| --- | --- |
| `vertices_ref` | Ground-truth SMPL-X vertices from the SOKE dataloader. |
| `vertices_soke` | SOKE generated vertices. |
| `faces` | SMPL-X mesh faces. |
| `length` | Valid GT length. |
| `length_soke` | Valid SOKE generated length. |
| `name` | Sequence name used for matching with flow samples. |
| `text` | Text condition. |
| `src` | Source text metadata when available. |

### Step 2: Render Five Panels

This command renders the five-panel comparison and resamples all mesh sequences to the original video frame count.

```bash
SRC_DIR=visualize/phoenix_latent_adapter_residual_retry3_ncclif_last_train_indices000_008_text
SOKE_DIR=visualize/soke_min_phoenix_epoch84_train_5
OUT_DIR=visualize/phoenix_retry3_last_train_5_source_soke_white_cover_upper_resampled_to_video
SOURCE_ROOT=/media/cvpr/haomian/data/phoenix/fullFrame-210x260px

mkdir -p "$OUT_DIR"

for i in 00 01 02 03 04; do
  $SOKE_PY -m flow.visualize.visualize_compare_three_npz_with_source \
    --gt "$SRC_DIR/gt_${i}.npz" \
    --pred "$SRC_DIR/sample_${i}.npz" \
    --out_dir "$OUT_DIR" \
    --source_root "$SOURCE_ROOT" \
    --source_split train \
    --source_fit cover \
    --resample_to_source_frames \
    --background_color white \
    --panel_order source gt prior soke pred \
    --prior_label "Word Prior" \
    --soke_vertices "$SOKE_DIR" \
    --soke_label SOKE \
    --upper_body_only \
    --device cpu \
    --smplx_batch_size 128 \
    --width 512 \
    --height 512 \
    --fps 20 \
    --end_mode hold
done
```

Important options:

| Option | Meaning |
| --- | --- |
| `--source_root` | Root containing RGB frames or videos. For PHOENIX, use `fullFrame-210x260px`. |
| `--source_split` | Split folder under the source root, such as `train`, `dev`, or `test`. |
| `--source_fit cover` | Resizes and center-crops original frames so the source panel is fully filled. |
| `--background_color white` | Uses a white render and panel background. |
| `--panel_order source gt prior soke pred` | Produces `Original frames | Ground truth | Word Prior | SOKE | Flow matching`. |
| `--prior_label "Word Prior"` | Labels the third panel as `Word Prior`. This is now the default, but the explicit option makes the layout clear in scripts. |
| `--soke_vertices` | A single SOKE vertex `.npz`, or a directory of `*_soke_vertices.npz` files matched by sample name. |
| `--resample_to_source_frames` | Uses the original source frame count as the output length and linearly resamples every mesh sequence to that length. |
| `--upper_body_only` | Renders only the upper-body mesh. This is useful because the lower body is mostly static in these samples. |
| `--meta_filename render_meta.json` | Writes or updates a render metadata file under `--out_dir`. |
| `--no_meta` | Disables metadata writing. |

Output name:

```text
<out_dir>/<pred_stem>_source_gt_word_soke_flow.mp4
```

For the example above, the verified output frame counts are:

| Sample | Original frame count | Output frame count |
| --- | ---: | ---: |
| `sample_00` | 86 | 86 |
| `sample_01` | 126 | 126 |
| `sample_02` | 168 | 168 |
| `sample_03` | 185 | 185 |
| `sample_04` | 71 | 71 |

### Render Metadata

By default, `visualize_compare_three_npz_with_source` writes:

```text
<out_dir>/render_meta.json
```

The file is updated after each MP4 is written. This works even if you render one sample per process in a shell loop: each run reads the existing metadata file, replaces the record for the same output video if present, and appends the new record.

Each render record includes:

| Field | Meaning |
| --- | --- |
| `output_video` | Path to the generated MP4. |
| `sequence.name` | Sequence/source name used for matching source frames and SOKE output. |
| `sequence.text` | Condition text stored in the prediction or GT `.npz`. |
| `sequence.raw_text` | Original dataset text, when the sample `.npz` was produced from a manifest that stores it. |
| `sequence.gloss` | Gloss annotation, when present in the sample `.npz`. |
| `sequence.label_word` | Label-word annotation, when present. |
| `inputs` | GT, prediction, Word Prior, SOKE, and source-root paths. |
| `npz_metadata` | Scalar metadata copied from the input `.npz` files. |
| `layout` | Panel order, labels, background color, source fit, and upper-body setting. |
| `frames` | Original sequence lengths, source frame count, output frame count, and resampling mode. |
| `render` | FPS, panel size, face stride, view transform, and render device. |

`gloss` and other dataset annotations are only available if the upstream sample `.npz` contains them. New outputs from `flow.sample_text_conditional` preserve manifest fields such as `raw_text`, `gloss`, `label_word`, `source_name`, `source_split`, signer, dataset, and source frame counts.

### Timeline Modes

| Mode | Behavior | Recommended use |
| --- | --- | --- |
| No `--resample_to_source_frames` | The video length is the longest mesh sequence. Source frames are sampled to that length. Shorter mesh sequences use `--end_mode blank` or `--end_mode hold`. | Comparing raw generated sequence lengths. |
| With `--resample_to_source_frames` | The source RGB frame count is the video length. GT, word prior, SOKE, and flow meshes are linearly resampled to exactly that length. | Comparing motion against original video frames. |

When `--resample_to_source_frames` is enabled, `--end_mode` is usually only a fallback because all mesh panels are already the same length after resampling.

## Posfix-Contact Sentence Visualization

Use `visualize_posfix_contact_sentences` to build sentence-level pose sequences by concatenating word-level tracking files from:

```text
/media/cvpr/haomian/data/posfix_contact
```

The script reads:

```text
/media/cvpr/haomian/data/posfix_contact/SENTENCES_coverage.txt
```

For each selected sentence, it parses the `WITH video` line and loads each word from:

```text
<data_dir>/<word>/optim_tracking_ehm.pkl
```

The rendered view is upper-body-only by default. Use `--full_body` if you want the full mesh.

### Direct Sentence Concatenation

This renders one upper-body sentence video from directly concatenated word poses.

```bash
$SOKE_PY -m flow.visualize.visualize_posfix_contact_sentences \
  --sentence_ids 1 \
  --out_dir visualize/posfix_contact_sentence_concat \
  --fps 20 \
  --width 512 \
  --height 512 \
  --software_face_stride 1 \
  --device cpu
```

Outputs:

```text
visualize/posfix_contact_sentence_concat/sentence_01_concat.npz
visualize/posfix_contact_sentence_concat/sentence_01_concat_upper.mp4
```

The saved sentence `.npz` contains:

| Key | Meaning |
| --- | --- |
| `motion` | Compact `[T, 133]` pose sequence. |
| `smplx` | Full `[T, 182]` SMPL-X sequence. |
| `left_valid`, `right_valid` | Hand-validity tracks. |
| `text` | Original sentence text from the coverage file. |
| `words` | Parsed signed word list. |
| `boundaries` | Word and transition frame ranges. |
| `source_paths` | Source tracking pickle for each word. |
| `label` | Display label used in the video header. |
| `fps` | Video frame rate. |

### Side-by-Side Boundary Comparison

The current recommended comparison is:

| Side | Construction |
| --- | --- |
| Left | Direct word concatenation with 25 frames trimmed at each word boundary. |
| Right | 30 frames trimmed at each word boundary, plus a 10-frame linear transition between words. |

Run:

```bash
$SOKE_PY -m flow.visualize.visualize_posfix_contact_sentences \
  --sentence_ids 1 \
  --compare_smoothing \
  --left_trim_frames 25 \
  --trim_frames 30 \
  --transition_frames 10 \
  --out_dir visualize/posfix_contact_sentence_compare \
  --fps 20 \
  --width 512 \
  --height 512 \
  --software_face_stride 1 \
  --device cpu
```

Outputs:

```text
visualize/posfix_contact_sentence_compare/sentence_01_directtrim25.npz
visualize/posfix_contact_sentence_compare/sentence_01_trim30_interp10.npz
visualize/posfix_contact_sentence_compare/sentence_01_directtrim25_vs_trim30_interp10_upper.mp4
```

For Sentence 1, this produces a 613-frame video at 20 fps, or 30.65 seconds.

### Keeping Both Sides Frame-Aligned

For a sentence with `N` signed words, there are `N - 1` word boundaries.

The left direct-trim side removes:

```text
2 * left_trim_frames * (N - 1)
```

The right smoothed side removes two trims and then adds transition frames:

```text
(2 * trim_frames - transition_frames) * (N - 1)
```

To keep both panels the same length, choose values that satisfy:

```text
2 * left_trim_frames = 2 * trim_frames - transition_frames
```

Current setting:

```text
2 * 25 = 2 * 30 - 10
50 = 50
```

So the two sides have the same number of frames.

### Dry Run Without Rendering

Use `--no_render` to build the `.npz` files and check frame counts before rendering an MP4.

```bash
$SOKE_PY -m flow.visualize.visualize_posfix_contact_sentences \
  --sentence_ids 1 \
  --compare_smoothing \
  --left_trim_frames 25 \
  --trim_frames 30 \
  --transition_frames 10 \
  --out_dir visualize/posfix_contact_sentence_compare \
  --no_render \
  --device cpu
```

Expected output includes lines like:

```text
Saved visualize/posfix_contact_sentence_compare/sentence_01_directtrim25.npz: 13 words, 613 frames, 30.65s
Saved visualize/posfix_contact_sentence_compare/sentence_01_trim30_interp10.npz: 13 words, 613 frames, 30.65s
```

## Verify Output Videos

Check video metadata:

```bash
ffprobe -v error \
  -select_streams v:0 \
  -show_entries stream=width,height,nb_frames,duration,r_frame_rate \
  -of default=noprint_wrappers=1:nokey=0 \
  visualize/posfix_contact_sentence_compare/sentence_01_directtrim25_vs_trim30_interp10_upper.mp4
```

Extract one frame for visual inspection:

```bash
mkdir -p /tmp/soke_visualize_check

ffmpeg -y -v error \
  -i visualize/posfix_contact_sentence_compare/sentence_01_directtrim25_vs_trim30_interp10_upper.mp4 \
  -vf 'select=eq(n\,300)' \
  -frames:v 1 \
  /tmp/soke_visualize_check/frame300.jpg
```

List generated outputs:

```bash
find visualize/posfix_contact_sentence_compare \
  -maxdepth 1 \
  -type f \
  -printf '%f %s bytes\n' \
  | sort
```

For source-aware five-panel videos, count frames directly from the MP4 and compare them with the original frame directory:

```bash
$SOKE_PY - <<'PY'
import imageio.v2 as imageio
from pathlib import Path

base = Path("visualize/phoenix_retry3_last_train_5_source_soke_white_cover_upper_resampled_to_video")
for path in sorted(base.glob("*.mp4")):
    reader = imageio.get_reader(str(path))
    count = 0
    first_shape = None
    try:
        for frame in reader:
            if first_shape is None:
                first_shape = frame.shape
            count += 1
    finally:
        reader.close()
    print(f"{path.name}: frames={count}, shape={first_shape}")
PY
```

## Troubleshooting

| Symptom | Fix |
| --- | --- |
| Mesh has holes | Use `--software_face_stride 1`. |
| Render is slow | First test with `--max_frames 20`, smaller `--width/--height`, or CPU smoke runs before final quality. |
| Wrong orientation | Try `--view_transform how2sign_front`, `rot_y_180`, or another listed transform. Current generated and posfix-contact outputs use `none`. |
| Missing SMPL-X files | Check `--model_dir deps/smpl_models`. |
| Missing original frames | Check `--source_root`, `--source_split`, and the sample `name` stored in the prediction `.npz`. Use `--source_name` to override the lookup. |
| Original frame panel has padding | Use `--source_fit cover` to fill the whole source panel. |
| SOKE panel is unavailable | First export SOKE vertices with `test/visualize_soke_test.py --save_npz --no_video`, then pass the file or directory through `--soke_vertices`. |
| SOKE file does not match the flow sample | Make sure `--sample_name` in the SOKE export uses the same sequence names as the flow `.npz` files. |
| Five-panel video has the wrong length | Use `--resample_to_source_frames` when original frames should define the timeline. |
| Missing posfix word | Check that `<data_dir>/<word>/optim_tracking_ehm.pkl` exists and that the word appears in `SENTENCES_coverage.txt`. |
| Side-by-side comparison has unequal lengths | Adjust trims so `2 * left_trim_frames = 2 * trim_frames - transition_frames`, or use `--end_mode hold`/`blank` intentionally. |
