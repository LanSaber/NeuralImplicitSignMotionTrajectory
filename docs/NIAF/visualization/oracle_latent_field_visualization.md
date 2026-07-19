# Oracle Latent-Field Visualization

This note documents the visualization workflow for the oracle latent-field
experiment. It covers two complementary views:

1. latent trajectory visualization: compare `z_sent` and `z_fit` in PCA space;
2. decoded-pose visualization: compare ground truth poses, VAE-decoded ground
   truth latents, and fitted-field decoded poses in a three-panel MP4.

Run commands from the repository root:

```bash
cd /media/cvpr/haomian/NeuralImplicitSignMotionTrajectory
```

Use the `SOKE` conda environment:

```bash
conda run -n SOKE python ...
```

Namespace note: the latent-field visualization code now lives under
`NIAF/oracle_latent_field`, and new visualization artifacts should be written
under `experiments/NIAF/oracle_latent_field`.

---

## 1. Relevant Files

Latent trajectory renderer:

```text
NIAF/oracle_latent_field/visualization/visualize_latent_trajectory.py
```

Decoded-pose renderer:

```text
flow/visualize/visualize_compare_three_npz.py
```

Oracle fitting entrypoint:

```text
NIAF/oracle_latent_field/scripts/fit_oracle_field.py
```

How2Sign oracle config:

```text
NIAF/oracle_latent_field/configs/how2sign_pilot.yaml
```

---

## 2. Export Visualization NPZs

The oracle fitting script can export both latent arrays and decoded motion NPZs.
Use:

- `--save_latent_npz` for latent trajectory plotting;
- `--save_npz` for decoded-pose rendering.

Example command for one representative sample:

```bash
conda run -n SOKE python -m NIAF.oracle_latent_field.scripts.fit_oracle_field \
  --config NIAF/oracle_latent_field/configs/how2sign_pilot.yaml \
  --out_dir experiments/NIAF/oracle_latent_field/how2sign_pose_viz_export \
  --max_sequences 1 \
  --models A5 \
  --time_modes uniform \
  --losses L1 \
  --fit_modes fit_all,even_odd \
  --steps 1000 \
  --warmup_steps 100 \
  --batch_points 64 \
  --save_npz \
  --save_latent_npz
```

This creates:

```text
experiments/NIAF/oracle_latent_field/how2sign_pose_viz_export/
  latent_npz/
    0000_A5_uniform_L1_fit_all.npz
    0000_A5_uniform_L1_even_odd.npz
  npz/
    0000_A5_uniform_L1_fit_all/
      gt_000.npz
      vae_000.npz
      sample_000.npz
    0000_A5_uniform_L1_even_odd/
      gt_000.npz
      vae_000.npz
      sample_000.npz
```

The decoded-pose NPZ meanings are:

| File | Meaning |
|---|---|
| `gt_000.npz` | original ground-truth normalized pose converted back to SMPL-X |
| `vae_000.npz` | frozen VAE reconstruction from the ground-truth latent tokens |
| `sample_000.npz` | oracle field decoded from fitted latent tokens |

---

## 3. Latent Trajectory Visualization

Use the oracle latent trajectory renderer:

```bash
conda run -n SOKE python -m NIAF.oracle_latent_field.visualization.visualize_latent_trajectory \
  --npz experiments/NIAF/oracle_latent_field/how2sign_pose_viz_export/latent_npz/0000_A5_uniform_L1_fit_all.npz \
        experiments/NIAF/oracle_latent_field/how2sign_pose_viz_export/latent_npz/0000_A5_uniform_L1_even_odd.npz \
  --latent_key z_sent z_fit \
  --out_dir experiments/NIAF/oracle_latent_field/how2sign_trajectory_viz \
  --annotate_every 1 \
  --fig_width 13.5 \
  --fig_height 5.8
```

Input keys:

| Key | Meaning |
|---|---|
| `z_sent` | ground-truth normalized VAE latent trajectory |
| `z_fit` | fitted field latent trajectory queried at token locations |
| `latent_mask` | valid latent-token mask |

Outputs:

```text
experiments/NIAF/oracle_latent_field/how2sign_trajectory_viz/
  *_pca2d_trajectory.png
  *_pca2d_trajectory_metrics.json
  *_pca2d_trajectory_projection.npz
  latent_trajectory_summary.json
```

The PNG has two panels:

- left: PCA projection of latent tokens, connected in temporal order;
- right: adjacent-token L2 distance over time.

Interpretation:

- In `fit_all`, `z_fit` should nearly overlay `z_sent` if fitting succeeded.
- In `even_odd`, divergence or shortcutting indicates interpolation failure.
- The PCA plot is only qualitative; metrics are computed in full latent
  dimension before projection.

Example current outputs:

```text
experiments/NIAF/oracle_latent_field/how2sign_trajectory_viz/
  0000_A5_uniform_L1_fit_all_z_sent-z_fit_pca2d_trajectory.png
  0000_A5_uniform_L1_even_odd_z_sent-z_fit_pca2d_trajectory.png
```

---

## 4. Decoded-Pose Visualization

Use the existing three-way SMPL-X renderer:

```text
flow/visualize/visualize_compare_three_npz.py
```

For this experiment, map the renderer panels as:

| Renderer role | Oracle visualization meaning |
|---|---|
| `--gt` | ground-truth pose |
| `--pred` | ground-truth latent decoded by the frozen VAE |
| `--prior` | fitted-field decoded pose |

### 4.1 Fit-all Comparison

```bash
conda run -n SOKE python -m flow.visualize.visualize_compare_three_npz \
  --gt experiments/NIAF/oracle_latent_field/how2sign_pose_viz_export/npz/0000_A5_uniform_L1_fit_all/gt_000.npz \
  --pred experiments/NIAF/oracle_latent_field/how2sign_pose_viz_export/npz/0000_A5_uniform_L1_fit_all/vae_000.npz \
  --prior experiments/NIAF/oracle_latent_field/how2sign_pose_viz_export/npz/0000_A5_uniform_L1_fit_all/sample_000.npz \
  --out_dir experiments/NIAF/oracle_latent_field/how2sign_pose_viz/fit_all \
  --view_transform none \
  --fps 20 \
  --width 384 \
  --height 384 \
  --software_face_stride 1 \
  --device cpu \
  --end_mode hold \
  --gt_label "GT pose" \
  --pred_label "GT latent decoded by VAE" \
  --prior_label "A5 fit-all decoded pose" \
  --pred_smplx_key smplx \
  --prior_smplx_key smplx
```

Expected output:

```text
experiments/NIAF/oracle_latent_field/how2sign_pose_viz/fit_all/vae_000_gt_flow_word.mp4
```

For easier inspection, copy or rename the output:

```bash
cp experiments/NIAF/oracle_latent_field/how2sign_pose_viz/fit_all/vae_000_gt_flow_word.mp4 \
   experiments/NIAF/oracle_latent_field/how2sign_pose_viz/a5_fit_all_gt_vae_field.mp4
```

### 4.2 Even-odd Comparison

```bash
conda run -n SOKE python -m flow.visualize.visualize_compare_three_npz \
  --gt experiments/NIAF/oracle_latent_field/how2sign_pose_viz_export/npz/0000_A5_uniform_L1_even_odd/gt_000.npz \
  --pred experiments/NIAF/oracle_latent_field/how2sign_pose_viz_export/npz/0000_A5_uniform_L1_even_odd/vae_000.npz \
  --prior experiments/NIAF/oracle_latent_field/how2sign_pose_viz_export/npz/0000_A5_uniform_L1_even_odd/sample_000.npz \
  --out_dir experiments/NIAF/oracle_latent_field/how2sign_pose_viz/even_odd \
  --view_transform none \
  --fps 20 \
  --width 384 \
  --height 384 \
  --software_face_stride 1 \
  --device cpu \
  --end_mode hold \
  --gt_label "GT pose" \
  --pred_label "GT latent decoded by VAE" \
  --prior_label "A5 even-odd decoded pose" \
  --pred_smplx_key smplx \
  --prior_smplx_key smplx
```

Expected output:

```text
experiments/NIAF/oracle_latent_field/how2sign_pose_viz/even_odd/vae_000_gt_flow_word.mp4
```

For easier inspection:

```bash
cp experiments/NIAF/oracle_latent_field/how2sign_pose_viz/even_odd/vae_000_gt_flow_word.mp4 \
   experiments/NIAF/oracle_latent_field/how2sign_pose_viz/a5_even_odd_gt_vae_field.mp4
```

### 4.3 What To Inspect

The three panels are:

1. `GT pose`: original ground-truth SMPL-X sequence.
2. `GT latent decoded by VAE`: the frozen VAE upper-bound reconstruction.
3. `A5 ... decoded pose`: decoded pose from the fitted latent field.

For `fit_all`, the third panel should be visually almost identical to the
middle panel if latent fitting succeeded.

For `even_odd`, inspect whether the third panel:

- shortcuts the motion;
- over-smooths hand or wrist transitions;
- lags behind the VAE reconstruction;
- creates implausible in-between poses.

This visual comparison should be read together with the latent metrics:

- `heldout_rel_l2_z`;
- `heldout_r2_z`;
- `field_feature_mae`;
- `field_to_vae_feature_mae_gap`.

---

## 5. Verify Outputs

Check MP4 metadata:

```bash
for f in experiments/NIAF/oracle_latent_field/how2sign_pose_viz/*.mp4; do
  echo "$f"
  ffprobe -v error -select_streams v:0 \
    -show_entries stream=width,height,nb_frames,duration,r_frame_rate \
    -of default=noprint_wrappers=1:nokey=0 "$f"
done
```

Extract a preview frame:

```bash
mkdir -p experiments/NIAF/oracle_latent_field/how2sign_pose_viz/previews

ffmpeg -y -v error \
  -i experiments/NIAF/oracle_latent_field/how2sign_pose_viz/a5_fit_all_gt_vae_field.mp4 \
  -vf 'select=eq(n\,18)' \
  -frames:v 1 \
  experiments/NIAF/oracle_latent_field/how2sign_pose_viz/previews/fit_all_frame018.jpg

ffmpeg -y -v error \
  -i experiments/NIAF/oracle_latent_field/how2sign_pose_viz/a5_even_odd_gt_vae_field.mp4 \
  -vf 'select=eq(n\,18)' \
  -frames:v 1 \
  experiments/NIAF/oracle_latent_field/how2sign_pose_viz/previews/even_odd_frame018.jpg
```

The current representative videos are:

```text
experiments/NIAF/oracle_latent_field/how2sign_pose_viz/a5_fit_all_gt_vae_field.mp4
experiments/NIAF/oracle_latent_field/how2sign_pose_viz/a5_even_odd_gt_vae_field.mp4
```

They are 48 frames at 20 FPS with three 384-pixel panels.

---

## 6. Notes

- Use `--view_transform none` for the oracle NPZ exports, because they are
  already converted through the local SMPL-X flow representation.
- Use `--software_face_stride 1` for final inspection to avoid mesh holes.
- The three-way renderer names outputs from the `--pred` stem, so rendering
  multiple comparisons with the same `vae_000.npz` filename into one directory
  can overwrite files. Use separate output directories or rename/copy the final
  MP4s.
- The latent PCA renderer was moved from `flow/visualize` into
  `NIAF/oracle_latent_field/visualization` so the oracle experiment owns its own
  latent-field plotting tools.
