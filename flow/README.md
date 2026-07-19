# Unconditional Upper-Body SMPL-X Flow

This folder contains a simplified flow-matching pipeline for SOKE. It ignores text and learns:

```text
Gaussian noise -> plausible upper-body SMPL-X signing sequence
```

The target representation is the 133D compact SMPL-X feature:

```text
0:30      upper body
30:75     left hand
75:120    right hand
120:123   jaw
123:133   expression
```

It drops global root orientation, lower body, shape/betas, and translation.

## 1. Prepare Data

```bash
cd /media/cvpr/haomian/NeuralImplicitSignMotionTrajectory

/media/cvpr/haomian/python_envs/soke/bin/python -m flow.dataset.prepare_dataset \
  --pkl_dir /media/cvpr/haomian/data/how2sign_pkls_cropTrue_shapeFalse \
  --soke_root /media/cvpr/haomian/data/SOKE/How2Sign \
  --out_dir /media/cvpr/haomian/data/SOKE_FLOW/how2sign_upper_smplx \
  --target_fps 20 \
  --overwrite
```

For a quick smoke dataset:

```bash
/media/cvpr/haomian/python_envs/soke/bin/python -m flow.dataset.prepare_dataset \
  --out_dir /media/cvpr/haomian/data/SOKE_FLOW/how2sign_upper_smplx_smoke \
  --limit 32 \
  --overwrite
```

## 2. Overfit One Clip

This is the first sanity check.

```bash
/media/cvpr/haomian/python_envs/soke/bin/python -m flow.train_unconditional \
  --data_dir /media/cvpr/haomian/data/SOKE_FLOW/how2sign_upper_smplx_smoke \
  --out_dir experiments/flow/overfit_1clip \
  --limit_train 1 \
  --limit_val 1 \
  --batch_size 1 \
  --epochs 200 \
  --hidden_dim 256 \
  --num_layers 4 \
  --num_heads 4 \
  --num_workers 0 \
  --sample_every 20
```

## 3. Train Small Unconditional Model

```bash
/media/cvpr/haomian/python_envs/soke/bin/python -m flow.train_unconditional \
  --data_dir /media/cvpr/haomian/data/SOKE_FLOW/how2sign_upper_smplx \
  --out_dir experiments/flow/upper_smplx_uncond \
  --batch_size 32 \
  --epochs 100 \
  --hidden_dim 512 \
  --num_layers 8 \
  --num_heads 8 \
  --num_workers 4 \
  --sample_every 10
```

Add `--wandb` if you want online logging.

## 4. Sample

Save generated compact and full SMPL-X parameters:

```bash
/media/cvpr/haomian/python_envs/soke/bin/python -m flow.sample_unconditional \
  --checkpoint experiments/flow/upper_smplx_uncond/checkpoints/last.pt \
  --out_dir visualize/flow_uncond_samples \
  --num_samples 4 \
  --length 196 \
  --steps 20
```

Render MP4s too:

```bash
/media/cvpr/haomian/python_envs/soke/bin/python -m flow.sample_unconditional \
  --checkpoint experiments/flow/upper_smplx_uncond/checkpoints/last.pt \
  --out_dir visualize/flow_uncond_samples \
  --num_samples 4 \
  --length 196 \
  --steps 20 \
  --render \
  --render_device cpu \
  --software_face_stride 1
```

## Expected Result

The unconditional model will not follow a sentence. The first useful milestone is generic but meaningful signing-like motion:

- upright upper body
- stable hands
- smooth hand motion
- no severe jitter or pose explosions

If the one-clip overfit fails, fix representation, normalization, or loss weighting before adding text.
