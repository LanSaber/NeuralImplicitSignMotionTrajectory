# SOKE Visualization Pipeline

This pipeline runs SOKE test generation from a checkpoint and renders a side-by-side MP4:

- Left: ground-truth SMPL-X mesh sequence.
- Right: SOKE generated SMPL-X mesh sequence.
- If the two sequences have different lengths, they are not time-aligned or stretched. The video runs until the longer sequence finishes. The shorter side is blank after it ends by default.

The visualization entry point is:

```bash
test/visualize_soke_test.py
```

The Slurm launcher is:

```bash
scripts/visualize_soke_test_sbatch.sh
```

## Flow Visualization Entry Points

Flow-matching visualization scripts live under:

```text
flow/visualize/
```

Use these module paths from the project root:

```bash
python -m flow.visualize.visualize_npz
python -m flow.visualize.visualize_compare_npz
python -m flow.visualize.visualize_compare_three_npz
```

The three scripts cover:

| Module | Output |
| --- | --- |
| `flow.visualize.visualize_npz` | Single generated SMPL-X sequence MP4. |
| `flow.visualize.visualize_compare_npz` | Ground truth and flow output side by side. |
| `flow.visualize.visualize_compare_three_npz` | Ground truth, flow output, and residual word-concatenation prior side by side. |

For detailed usage, including the posfix-contact sentence visualizer and boundary-trim comparison workflow, see [`flow_visualize_usage.md`](../flow/visualization/flow_visualize_usage.md).

For residual flow checkpoints, `sample_text_conditional` saves `coarse_motion` and `coarse_smplx` in each `sample_*.npz`. The three-way renderer uses `coarse_smplx` by default for the word-concatenation panel.

Example two-way flow comparison:

```bash
cd /media/cvpr/haomian/NeuralImplicitSignMotionTrajectory

PYTHONNOUSERSITE=1 PYTHONPATH=/media/cvpr/haomian/NeuralImplicitSignMotionTrajectory \
/media/cvpr/haomian/python_envs/soke/bin/python -m flow.visualize.visualize_compare_npz \
  --gt visualize/flow_chatsign175_residual_epoch1300_test_samples/gt_00.npz \
  --pred visualize/flow_chatsign175_residual_epoch1300_test_samples/sample_00.npz \
  --out_dir visualize/flow_compare_pair \
  --view_transform none \
  --fps 20 \
  --width 512 \
  --height 512 \
  --software_face_stride 1 \
  --device cpu \
  --end_mode blank
```

Example three-way residual flow comparison:

```bash
cd /media/cvpr/haomian/NeuralImplicitSignMotionTrajectory

PYTHONNOUSERSITE=1 PYTHONPATH=/media/cvpr/haomian/NeuralImplicitSignMotionTrajectory \
/media/cvpr/haomian/python_envs/soke/bin/python -m flow.visualize.visualize_compare_three_npz \
  --gt visualize/flow_chatsign175_residual_epoch1300_test_samples/gt_00.npz \
  --pred visualize/flow_chatsign175_residual_epoch1300_test_samples/sample_00.npz \
  --out_dir visualize/flow_compare_three \
  --view_transform none \
  --fps 20 \
  --width 512 \
  --height 512 \
  --software_face_stride 1 \
  --device cpu \
  --end_mode blank
```

For `WIDTH=512` and `HEIGHT=512`, two-way flow comparison videos are `1024x564`, and three-way flow comparison videos are `1536x568`.

## Recommended Slurm Run

Run from the project root:

```bash
cd /media/cvpr/haomian/NeuralImplicitSignMotionTrajectory

OUT_DIR=visualize/soke_test_last_v1_front_full \
NUM_SAMPLES=4 \
START_INDEX=0 \
CHECKPOINT=experiments/mgpt/SOKE_2026-03-03-18-21-42/checkpoints/last-v1.ckpt \
RENDERER=software \
SOFTWARE_FACE_STRIDE=1 \
MAX_FRAMES=0 \
WIDTH=512 \
HEIGHT=512 \
FPS=20 \
sbatch --export=ALL scripts/visualize_soke_test_sbatch.sh
```

This produces full videos under:

```text
visualize/soke_test_last_v1_front_full/
```

For `WIDTH=512` and `HEIGHT=512`, each output video is `1024x576`: two 512-wide panels plus a 64-pixel text header.

## Quick Debug Run

Use `MAX_FRAMES` to render only the first few frames per video. This is useful for checking camera direction, mesh holes, and text layout.

```bash
cd /media/cvpr/haomian/NeuralImplicitSignMotionTrajectory

OUT_DIR=visualize/soke_test_debug \
NUM_SAMPLES=1 \
CHECKPOINT=experiments/mgpt/SOKE_2026-03-03-18-21-42/checkpoints/last-v1.ckpt \
RENDERER=software \
SOFTWARE_FACE_STRIDE=1 \
MAX_FRAMES=20 \
WIDTH=256 \
HEIGHT=256 \
FPS=20 \
sbatch --export=ALL scripts/visualize_soke_test_sbatch.sh
```

The video is still generated from the real checkpoint, but rendering stops after 20 frames.

## Direct Python Run

This is useful for local debugging inside an allocated GPU session:

```bash
cd /media/cvpr/haomian/NeuralImplicitSignMotionTrajectory

/media/cvpr/haomian/python_envs/soke/bin/python test/visualize_soke_test.py \
  --cfg configs/soke.yaml \
  --checkpoint experiments/mgpt/SOKE_2026-03-03-18-21-42/checkpoints/last-v1.ckpt \
  --split test \
  --out_dir visualize/soke_test_direct \
  --num_samples 4 \
  --start_index 0 \
  --batch_size 1 \
  --num_workers 0 \
  --use_gpus 0 \
  --fps 20 \
  --width 512 \
  --height 512 \
  --max_frames 0 \
  --end_mode blank \
  --renderer software \
  --software_face_stride 1
```

Note: the current visualization path requires CUDA because SMPL-X coordinate reconstruction calls CUDA internally.

## Important Options

| Option | Slurm variable | Default | Meaning |
| --- | --- | --- | --- |
| `--checkpoint` | `CHECKPOINT` | `experiments/mgpt/SOKE_2026-03-03-18-21-42/checkpoints/last-v1.ckpt` | Checkpoint file. If a directory is passed, the script looks for `last.ckpt`. |
| `--out_dir` | `OUT_DIR` | `visualize/soke_test` | Directory for MP4 outputs. |
| `--num_samples` | `NUM_SAMPLES` | `4` | Number of test samples to render. |
| `--start_index` | `START_INDEX` | `0` | Skip this many test samples before rendering. |
| `--width` | `WIDTH` | `512` | Width of each GT/prediction panel. |
| `--height` | `HEIGHT` | `512` | Height of each GT/prediction panel. |
| `--fps` | `FPS` | `20` | Output video frame rate. |
| `--max_frames` | `MAX_FRAMES` | `0` | Debug cap per output video. `0` means render the full longer sequence. |
| `--end_mode` | `END_MODE` | `blank` | `blank` leaves the shorter side black after it ends. `hold` freezes its last frame. |
| `--renderer` | `RENDERER` | `auto` | `software` is the most reliable on this cluster. `auto` tries `pyrender` first, then falls back. |
| `--software_face_stride` | `SOFTWARE_FACE_STRIDE` | `1` | Draw every Nth face in software mode. Use `1` for a complete mesh. Higher values are faster but create holes. |

## Renderer Notes

Use:

```bash
RENDERER=software
SOFTWARE_FACE_STRIDE=1
```

This is currently the most reliable mode on the cluster. The `pyrender`/EGL renderer may fail with an error like `Invalid device ID (0)` on some nodes. The software renderer avoids that EGL issue.

Do not use `SOFTWARE_FACE_STRIDE=2` or `3` for final inspection videos. It skips mesh triangles and creates black holes in the rendered body.

## Logs

Slurm logs are written to:

```text
logs/sbatch/soke_vis_test_<job_id>.out
logs/sbatch/soke_vis_test_<job_id>.err
```

Check job status:

```bash
squeue -j <job_id> -o '%.18i %.9T %.20M %.20R %.30j'
```

Watch output:

```bash
tail -f logs/sbatch/soke_vis_test_<job_id>.out
```

## Verify Output Videos

List generated videos:

```bash
find visualize/soke_test_last_v1_front_full -maxdepth 1 -name '*.mp4' -printf '%f %s bytes\n' | sort
```

Check video metadata:

```bash
for f in visualize/soke_test_last_v1_front_full/*.mp4; do
  echo "$(basename "$f")"
  ffprobe -v error -select_streams v:0 \
    -show_entries stream=width,height,nb_frames,duration,r_frame_rate \
    -of default=noprint_wrappers=1:nokey=0 "$f"
done
```

Extract a frame for visual checking:

```bash
mkdir -p /tmp/soke_vis_check

ffmpeg -y -v error \
  -i visualize/soke_test_last_v1_front_full/0000_-fZc293MpJk_2-1-rgb_front.mp4 \
  -vf 'select=eq(n\,9)' \
  -frames:v 1 \
  /tmp/soke_vis_check/frame10.jpg
```

## Known Good Output

The following command was used successfully to generate four full front-facing videos:

```bash
OUT_DIR=visualize/soke_test_last_v1_front_full \
NUM_SAMPLES=4 \
START_INDEX=0 \
CHECKPOINT=experiments/mgpt/SOKE_2026-03-03-18-21-42/checkpoints/last-v1.ckpt \
RENDERER=software \
SOFTWARE_FACE_STRIDE=1 \
MAX_FRAMES=0 \
WIDTH=512 \
HEIGHT=512 \
FPS=20 \
sbatch --export=ALL scripts/visualize_soke_test_sbatch.sh
```

It produced:

```text
0000_-fZc293MpJk_2-1-rgb_front.mp4  244 frames, 12.2s
0001_-fZc293MpJk_3-1-rgb_front.mp4  400 frames, 20.0s
0002_-fZc293MpJk_4-1-rgb_front.mp4  372 frames, 18.6s
0003_-fZc293MpJk_5-1-rgb_front.mp4  400 frames, 20.0s
```
