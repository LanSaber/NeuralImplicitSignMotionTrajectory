# Flow Back-Translation Evaluation

This document records how to evaluate Phoenix generated pose sequences with the
SpaMo sign-language translator. There are two producer families:

| Producer | Example checkpoint | Back-translation input state |
| --- | --- | --- |
| Improved latent flow matching | `experiments/flow/phoenix_latent_adapter_residual_b64x4_all_balanced_online_retry3_ncclif/checkpoints/best.pt` | `flow.sample_text_conditional` already saves `sample_*.npz` with compact `motion` |
| Baseline SOKE LM | `experiments/mgpt/SOKE_soke_lm_2n_b128_online_139176_1/checkpoints/min-phoenix_DTW_MPJPE_PA_lhandepoch=84.ckpt` | visualization exports are vertices only, so export compact `m_rst` motion first |

Both paths end in the same SpaMo wrapper: a JSONL manifest whose `motion_path`
points to NPZ files containing `motion` with shape `[T, 133]`.

## SpaMo Checkpoint

The validated translator checkpoint is:

```text
/media/cvpr/haomian/SpaMo/logs/slt/phoenix_soke_upper_smplx_rot6d_jubail/phoenix_soke_upper_smplx_rot6d_oracle_20260622-190546/checkpoints/epoch=119-12.6935.ckpt
```

The checkpoint consumes SOKE compact upper-body SMPL-X directly:

| SLT setting | Value |
| --- | --- |
| `visual_input_type` | `upper_smplx` |
| `upper_smplx_rotation_rep` | `rot6d` |
| `upper_smplx_use_vqvae` | `False` |
| `fusion_strategy` | `concat` |
| `alignment_mode` | `ot` |

The SpaMo `How2SignUpperSMPLXDataset` dataloader reads `motion`, converts the
compact axis-angle pose to rot6d, and normalizes it with the Phoenix training
statistics:

```text
/media/cvpr/haomian/data/SOKE_FLOW/phoenix_upper_smplx/meta/mean_rot6d.npy
/media/cvpr/haomian/data/SOKE_FLOW/phoenix_upper_smplx/meta/std_rot6d.npy
```

## Fair And Oracle Modes

Use one of these modes and name the output JSON accordingly:

| Mode | Flag | Meaning |
| --- | --- | --- |
| Fair generated-pose back-translation | pass `--disable_gt_text_eval_reordering` | SLT uses pose features plus the fixed prompt only |
| Oracle diagnostic | omit `--disable_gt_text_eval_reordering` | OT evaluation may use the reference sentence to reorder visual features |

The SpaMo checkpoint is an OT/oracle-style run. The oracle mode is useful for
diagnosis, but do not report it as a fair generated-pose-to-text benchmark.

## Environment

Run SOKE/flow data generation with the SOKE environment, and run SLT inference
with the SpaMo conda environment:

```bash
export SOKE_ROOT=/media/cvpr/haomian/NeuralImplicitSignMotionTrajectory
export SPAMO_ROOT=/media/cvpr/haomian/SpaMo
export SOKE_PY=/media/cvpr/haomian/python_envs/soke/bin/python

cd "$SOKE_ROOT"
export PYTHONNOUSERSITE=1
export PYTHONPATH="$SOKE_ROOT"
export TOKENIZERS_PARALLELISM=false
export WANDB_MODE=offline

export SPAMO_HF_CACHE=/media/cvpr/haomian/SpaMo/.cache/huggingface
mkdir -p "$SPAMO_HF_CACHE"
```

On the GB10 machine, PyTorch sees CUDA but warns that the GPU capability is newer
than the build officially supports. CPU SpaMo inference is slower but reliable.

## Flow-Matching Model

### Generate Flow Samples

For the current improved Phoenix flow checkpoint:

```bash
CHECKPOINT="$SOKE_ROOT/experiments/flow/phoenix_latent_adapter_residual_b64x4_all_balanced_online_retry3_ncclif/checkpoints/best.pt"
DATA_DIR=/media/cvpr/haomian/data/SOKE_FLOW/phoenix_upper_smplx
WORD_DATA_DIR=/media/cvpr/haomian/data/SOKE_FLOW/phoenix_upper_smplx_word_ctc
WORD_SPLIT=all.balanced
SAMPLES_DIR="$SOKE_ROOT/visualize/phoenix_latent_adapter_residual_all_balanced_retry3_best_test100_seed123_noneg_t2m_eval"
SEED=123

mkdir -p "$SAMPLES_DIR"
```

### Word-Series Prior Used By Flow

The Phoenix production checkpoint is an adapter-residual latent flow model. It
does not sample from text alone. During `flow.sample_text_conditional`, the
sentence text selected from the manifest is also used to build a word-level
source prior:

```text
manifest row
  -> condition_field text
  -> sentence word tokens
  -> lexical lookup in phoenix_upper_smplx_word_ctc/meta/manifest_all.balanced.jsonl
  -> word-pose candidates
  -> frozen soft arranger
  -> frozen content-style adapter
  -> latent source prior z_adapt
  -> flow refinement
  -> saved sample_*.npz motion/smplx
```

The key inputs for this word-series branch are:

| Setting | Value |
| --- | --- |
| `source_mode` | `adapter_residual` |
| `motion_space` | `latent` |
| `condition_field` | `text` for the production Phoenix run |
| `word_data_dir` | `/media/cvpr/haomian/data/SOKE_FLOW/phoenix_upper_smplx_word_ctc` |
| `word_split` | `all.balanced` |
| negative candidates | disabled with `--no_negative_candidates` for clean evaluation |

This word-series process happens before back-translation. SpaMo sees only the
final generated pose NPZs. If you change the word list, `condition_field`, word
split, candidate settings, or adapter checkpoint, regenerate the flow
`sample_*.npz` files before running SLT.

Create the random-100 manifest used by the quantitative flow protocol:

```bash
TEST_MANIFEST="$DATA_DIR/meta/manifest_test.jsonl"

TEST_MANIFEST="$TEST_MANIFEST" SAMPLES_DIR="$SAMPLES_DIR" SEED="$SEED" "$SOKE_PY" - <<'PY'
import json
import os
import random
from pathlib import Path

src = Path(os.environ["TEST_MANIFEST"])
out_dir = Path(os.environ["SAMPLES_DIR"])
seed = int(os.environ.get("SEED", "123"))

rows = [json.loads(line) for line in src.read_text(encoding="utf-8").splitlines() if line.strip()]
rng = random.Random(seed)
indices = rng.sample(range(len(rows)), 100)
selected = [rows[i] for i in indices]

manifest = out_dir / f"manifest_test_random100_seed{seed}.jsonl"
with manifest.open("w", encoding="utf-8") as handle:
    for row in selected:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")

summary = {
    "source_manifest": str(src),
    "output_manifest": str(manifest),
    "seed": seed,
    "source_count": len(rows),
    "sample_count": len(selected),
    "indices": indices,
}
(out_dir / "sample_manifest_summary.json").write_text(
    json.dumps(summary, indent=2, ensure_ascii=False),
    encoding="utf-8",
)
print(json.dumps({k: summary[k] for k in ["source_count", "sample_count"]}, indent=2))
PY
```

Generate predictions. Do not pass `--skip_save_outputs`; SpaMo needs the saved
`sample_*.npz` files:

```bash
MANIFEST="$SAMPLES_DIR/manifest_test_random100_seed123.jsonl"

"$SOKE_PY" -m flow.sample_text_conditional \
  --checkpoint "$CHECKPOINT" \
  --data_dir "$DATA_DIR" \
  --word_data_dir "$WORD_DATA_DIR" \
  --word_split "$WORD_SPLIT" \
  --manifest "$MANIFEST" \
  --num_prompts 100 \
  --match_manifest_lengths \
  --seed 123 \
  --device auto \
  --no_negative_candidates \
  --out_dir "$SAMPLES_DIR"
```

Flow sample files are directly compatible:

| Flow NPZ key | Shape | Meaning | Use for SLT |
| --- | ---: | --- | --- |
| `motion` | `[T, 133]` | compact upper-body SMPL-X axis-angle | yes |
| `representation` | `[T, 256]` | flow model's rot6d representation | no |
| `smplx` | `[T, 182]` | expanded SMPL-X parameter layout | no |

Use `motion`, not `representation` or `smplx`.

### Build The Flow SpaMo Wrapper

```bash
SAMPLES_DIR="$SOKE_ROOT/visualize/phoenix_latent_adapter_residual_all_balanced_retry3_best_test100_seed123_noneg_t2m_eval"
BT_DIR="$SOKE_ROOT/visualize/phoenix_flow_slt_backtranslation"
DATASET_TAG=phoenix14t_flow_generated
NUM_SAMPLES=100
CONFIG_NAME=phoenix_flow_slt_upper_smplx_rot6d.yaml

mkdir -p "$BT_DIR"
```

Create the wrapper manifest:

```bash
SAMPLES_DIR="$SAMPLES_DIR" BT_DIR="$BT_DIR" DATASET_TAG="$DATASET_TAG" NUM_SAMPLES="$NUM_SAMPLES" "$SOKE_PY" - <<'PY'
import json
import os
from pathlib import Path

import numpy as np

sample_dir = Path(os.environ["SAMPLES_DIR"])
out_dir = Path(os.environ["BT_DIR"])
dataset_tag = os.environ.get("DATASET_TAG", "generated_pose")
limit = int(os.environ.get("NUM_SAMPLES", "0") or 0)

sample_paths = sorted(sample_dir.glob("sample_*.npz"))
if limit > 0:
    sample_paths = sample_paths[:limit]
if not sample_paths:
    raise RuntimeError(f"No sample_*.npz files found in {sample_dir}")

manifest = out_dir / "manifest_test.jsonl"
with manifest.open("w", encoding="utf-8") as handle:
    for sample_path in sample_paths:
        with np.load(sample_path, allow_pickle=True) as data:
            if "motion" not in data.files:
                raise ValueError(f"{sample_path} does not contain a motion key")
            text = str(data["text"].item()) if "text" in data.files else ""
            name = str(data["name"].item()) if "name" in data.files else sample_path.stem
            length = int(data["length"].item()) if "length" in data.files else int(data["motion"].shape[0])
            rotation_rep = str(data["rotation_rep"].item()) if "rotation_rep" in data.files else ""
            motion_shape = list(data["motion"].shape)
            representation_shape = list(data["representation"].shape) if "representation" in data.files else None

        row = {
            "name": name,
            "motion_path": str(sample_path),
            "text": text,
            "gloss": "",
            "num_frames": length,
            "dataset": dataset_tag,
            "source_sample": str(sample_path),
            "source_rotation_rep": rotation_rep,
            "source_motion_shape": motion_shape,
            "source_representation_shape": representation_shape,
        }
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")

print(f"Wrote {len(sample_paths)} rows to {manifest}")
PY
```

Create the SpaMo dataset config:

```bash
cat > "$BT_DIR/$CONFIG_NAME" <<EOF
target: dataset.how2sign_smplx_data.How2SignUpperSMPLXData
params:
  data_dir: /media/cvpr/haomian/data/SOKE_FLOW/phoenix_upper_smplx
  batch_size: 2
  num_workers: 0
  train_split: test
  val_split: test
  test_split: test
  manifest_path: $BT_DIR/manifest_test.jsonl
  mean_path: /media/cvpr/haomian/data/SOKE_FLOW/phoenix_upper_smplx/meta/mean_rot6d.npy
  std_path: /media/cvpr/haomian/data/SOKE_FLOW/phoenix_upper_smplx/meta/std_rot6d.npy
  min_frames: 40
  max_frames: 400
  length_multiple: 8
  random_crop: false
  normalize: true
  mask_invalid_hands: false
  rotation_rep: rot6d
EOF
```

## Baseline SOKE LM

The baseline SOKE checkpoint is:

```text
/media/cvpr/haomian/NeuralImplicitSignMotionTrajectory/experiments/mgpt/SOKE_soke_lm_2n_b128_online_139176_1/checkpoints/min-phoenix_DTW_MPJPE_PA_lhandepoch=84.ckpt
```

The matching training config is:

```text
/media/cvpr/haomian/NeuralImplicitSignMotionTrajectory/experiments/mgpt/SOKE_soke_lm_2n_b128_online_139176_1/config_soke_lm_2n_b128_online_139176_train.yaml
```

### Why SOKE Needs A Motion Export

The existing visualization export:

```text
/media/cvpr/haomian/NeuralImplicitSignMotionTrajectory/visualize/soke_min_phoenix_epoch84_train_next10
```

contains only mesh vertices:

| Key | Meaning |
| --- | --- |
| `vertices_ref` | GT SMPL-X vertices |
| `vertices_soke` | SOKE generated vertices |
| `faces` | mesh topology |
| `length`, `length_soke`, `name`, `text`, `src` | metadata |

SpaMo cannot consume vertices. It needs compact upper-body SMPL-X motion
`[T, 133]`. Regenerate the same samples from the SOKE checkpoint and save
`result["m_rst"]` as `sample_*.npz`.

### Optional: Generate The Vertex Export

Use this only if the vertex directory does not already exist or if you want to
choose a different sample set:

```bash
SOKE_EXP="$SOKE_ROOT/experiments/mgpt/SOKE_soke_lm_2n_b128_online_139176_1"
SOKE_VERTEX_DIR="$SOKE_ROOT/visualize/soke_min_phoenix_epoch84_train_next10"
SAMPLE_NAMES=train/11August_2010_Wednesday_tagesschau-9,train/11August_2010_Wednesday_tagesschau-10,train/11August_2010_Wednesday_tagesschau-11,train/11August_2010_Wednesday_tagesschau-12,train/11August_2010_Wednesday_tagesschau-13,train/25October_2010_Monday_tagesschau-14,train/25October_2010_Monday_tagesschau-15,train/25October_2010_Monday_tagesschau-16,train/25October_2010_Monday_tagesschau-18,train/25October_2010_Monday_tagesschau-19

"$SOKE_PY" test/visualize_soke_test.py \
  --cfg "$SOKE_EXP/config_soke_lm_2n_b128_online_139176_train.yaml" \
  --checkpoint "$SOKE_EXP/checkpoints/min-phoenix_DTW_MPJPE_PA_lhandepoch=84.ckpt" \
  --split train \
  --sample_name "$SAMPLE_NAMES" \
  --out_dir "$SOKE_VERTEX_DIR" \
  --num_samples 10 \
  --batch_size 1 \
  --num_workers 0 \
  --use_gpus 0 \
  --save_npz \
  --no_video
```

### Export Compact SOKE Motion

This exporter reads the sample names from the vertex NPZs, reruns the SOKE LM,
and writes `sample_*.npz` files with `motion=[T,133]`:

```bash
SOKE_EXP="$SOKE_ROOT/experiments/mgpt/SOKE_soke_lm_2n_b128_online_139176_1"
SOKE_VERTEX_DIR="$SOKE_ROOT/visualize/soke_min_phoenix_epoch84_train_next10"
SOKE_MOTION_DIR="$SOKE_ROOT/visualize/soke_min_phoenix_epoch84_train_next10_bt_motion"

mkdir -p "$SOKE_MOTION_DIR"

PYTHONNOUSERSITE=1 \
PYTHONPATH="$SOKE_ROOT" \
CUDA_VISIBLE_DEVICES=0 \
PYOPENGL_PLATFORM=egl \
TOKENIZERS_PARALLELISM=false \
WANDB_MODE=offline \
SOKE_VERTEX_DIR="$SOKE_VERTEX_DIR" \
SOKE_MOTION_DIR="$SOKE_MOTION_DIR" \
SOKE_EXP="$SOKE_EXP" \
"$SOKE_PY" - <<'PY'
import argparse
import importlib.util
import json
from pathlib import Path

import numpy as np
import torch

from mGPT.data.build_data import build_data
from mGPT.models.build_model import build_model
from mGPT.utils.load_checkpoint import load_pretrained, load_pretrained_vae

import os

root = Path("/media/cvpr/haomian/NeuralImplicitSignMotionTrajectory")
src_vert_dir = Path(os.environ["SOKE_VERTEX_DIR"])
out_dir = Path(os.environ["SOKE_MOTION_DIR"])
soke_exp = Path(os.environ["SOKE_EXP"])
cfg_path = soke_exp / "config_soke_lm_2n_b128_online_139176_train.yaml"
ckpt_path = soke_exp / "checkpoints/min-phoenix_DTW_MPJPE_PA_lhandepoch=84.ckpt"

spec = importlib.util.spec_from_file_location(
    "visualize_soke_test",
    root / "test/visualize_soke_test.py",
)
vis = importlib.util.module_from_spec(spec)
spec.loader.exec_module(vis)

names = []
vertex_meta = {}
for path in sorted(src_vert_dir.glob("*_soke_vertices.npz")):
    with np.load(path, allow_pickle=True) as data:
        name = str(data["name"].item())
        names.append(name)
        vertex_meta[name] = {
            "source_vertex_npz": str(path),
            "vertex_length_ref": int(data["length"].item()),
            "vertex_length_soke": int(data["length_soke"].item()),
        }
if not names:
    raise RuntimeError(f"No *_soke_vertices.npz files found in {src_vert_dir}")

args = argparse.Namespace(
    cfg=str(cfg_path),
    cfg_assets=str(root / "configs/assets.yaml"),
    checkpoint=str(ckpt_path),
    vae_checkpoint=None,
    split="train",
    out_dir=str(out_dir),
    num_samples=len(names),
    start_index=0,
    sample_name=",".join(names),
    batch_size=1,
    num_workers=0,
    use_gpus="0",
    fps=20,
    width=512,
    height=512,
    max_frames=0,
    end_mode="blank",
    renderer="software",
    software_face_stride=1,
    pyopengl_platform="egl",
    seed=1234,
    save_npz=True,
    no_video=True,
    dry_run=False,
)

if not torch.cuda.is_available():
    raise RuntimeError("SOKE generation requires CUDA in this repository path.")

torch.manual_seed(args.seed)
np.random.seed(args.seed)
torch.set_grad_enabled(False)
device = torch.device("cuda")

cfg = vis.load_cfg(args)
datamodule = build_data(cfg, phase="test")
loader = vis.make_loader(
    datamodule,
    cfg,
    args.batch_size,
    args.num_workers,
    sample_name=args.sample_name,
)
model = build_model(cfg, datamodule)
if cfg.TRAIN.PRETRAINED_VAE:
    load_pretrained_vae(cfg, model)
load_pretrained(cfg, model, phase="test")
model.to(device)
model.eval()

out_dir.mkdir(parents=True, exist_ok=True)
manifest_rows = []
saved = 0

for batch in loader:
    if batch is None:
        continue
    batch = vis.move_batch_to_device(batch, device)
    result = model.val_t2m_forward(batch, vis=True)
    batch_size = batch["motion"].shape[0]

    for idx in range(batch_size):
        name = batch["name"][idx]
        text = batch.get("text", [""] * batch_size)[idx]
        src = batch.get("src", [""] * batch_size)[idx]
        pred_len = min(vis.to_int(result["lengths_rst"][idx]), result["m_rst"].shape[1])
        ref_len = min(vis.to_int(result["length"][idx]), result["m_ref"].shape[1])
        if pred_len <= 0:
            continue

        motion = result["m_rst"][idx, :pred_len].detach().cpu().numpy().astype(np.float32)
        gt_motion = result["m_ref"][idx, :ref_len].detach().cpu().numpy().astype(np.float32)
        if motion.ndim != 2 or motion.shape[-1] != 133:
            raise ValueError(f"{name}: expected generated motion [T,133], got {motion.shape}")

        sample_path = out_dir / f"sample_{saved:02d}.npz"
        gt_path = out_dir / f"gt_{saved:02d}.npz"
        meta = vertex_meta.get(name, {})
        np.savez_compressed(
            sample_path,
            motion=motion,
            text=np.asarray(text),
            name=np.asarray(name),
            length=np.asarray(pred_len, dtype=np.int64),
            src=np.asarray(src),
            source_checkpoint=np.asarray(str(ckpt_path)),
            source_config=np.asarray(str(cfg_path)),
            source_vertex_npz=np.asarray(meta.get("source_vertex_npz", "")),
        )
        np.savez_compressed(
            gt_path,
            motion=gt_motion,
            text=np.asarray(text),
            name=np.asarray(name),
            length=np.asarray(ref_len, dtype=np.int64),
            src=np.asarray(src),
            source_checkpoint=np.asarray(str(ckpt_path)),
            source_config=np.asarray(str(cfg_path)),
            source_vertex_npz=np.asarray(meta.get("source_vertex_npz", "")),
        )
        manifest_rows.append({
            "name": name,
            "motion_path": str(sample_path),
            "text": str(text),
            "gloss": "",
            "num_frames": int(pred_len),
            "dataset": "phoenix14t_soke_generated",
            "source_sample": str(sample_path),
            "source_checkpoint": str(ckpt_path),
            "source_vertex_npz": meta.get("source_vertex_npz", ""),
            "source_motion_shape": list(motion.shape),
        })
        print(f"Saved {sample_path}: motion={motion.shape}")
        saved += 1

manifest = out_dir / "manifest_test.jsonl"
with manifest.open("w", encoding="utf-8") as handle:
    for row in manifest_rows:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")

summary = {
    "source_vertex_dir": str(src_vert_dir),
    "checkpoint": str(ckpt_path),
    "config": str(cfg_path),
    "split": "train",
    "num_samples": saved,
    "manifest": str(manifest),
    "sample_names": names,
}
(out_dir / "summary.json").write_text(
    json.dumps(summary, ensure_ascii=False, indent=2),
    encoding="utf-8",
)
print(json.dumps(summary, ensure_ascii=False, indent=2))
PY
```

### Build The SOKE SpaMo Wrapper

The exporter already writes `manifest_test.jsonl`. Create the SpaMo dataset
config beside it:

```bash
BT_DIR="$SOKE_ROOT/visualize/soke_min_phoenix_epoch84_train_next10_bt_motion"
CONFIG_NAME=phoenix_soke_epoch84_train_next10_slt_upper_smplx_rot6d.yaml

cat > "$BT_DIR/$CONFIG_NAME" <<EOF
target: dataset.how2sign_smplx_data.How2SignUpperSMPLXData
params:
  data_dir: /media/cvpr/haomian/data/SOKE_FLOW/phoenix_upper_smplx
  batch_size: 2
  num_workers: 0
  train_split: test
  val_split: test
  test_split: test
  manifest_path: $BT_DIR/manifest_test.jsonl
  mean_path: /media/cvpr/haomian/data/SOKE_FLOW/phoenix_upper_smplx/meta/mean_rot6d.npy
  std_path: /media/cvpr/haomian/data/SOKE_FLOW/phoenix_upper_smplx/meta/std_rot6d.npy
  min_frames: 40
  max_frames: 400
  length_multiple: 8
  random_crop: false
  normalize: true
  mask_invalid_hands: false
  rotation_rep: rot6d
EOF
```

## Run SpaMo SLT

The standard `slt.mt5_slt_test` CLI is almost usable, but in the current SpaMo
environment the Lightning datamodule expects `setup(None)` while the helper calls
`setup()`. Until that is patched, use this direct Python runner for both model
families.

Set these variables for the model you are evaluating:

```bash
# Flow fair mode example:
BT_DIR="$SOKE_ROOT/visualize/phoenix_flow_slt_backtranslation"
DATA_CONFIG="$BT_DIR/phoenix_flow_slt_upper_smplx_rot6d.yaml"
OUT_JSON="$BT_DIR/slt_backtranslation_epoch119_no_gt_reorder.json"
GT_TEXT_REORDER=0

# Flow oracle diagnostic example:
BT_DIR="$SOKE_ROOT/visualize/phoenix_flow_slt_backtranslation"
DATA_CONFIG="$BT_DIR/phoenix_flow_slt_upper_smplx_rot6d.yaml"
OUT_JSON="$BT_DIR/slt_backtranslation_epoch119_gt_reorder_oracle.json"
GT_TEXT_REORDER=1

# Baseline SOKE oracle diagnostic example:
BT_DIR="$SOKE_ROOT/visualize/soke_min_phoenix_epoch84_train_next10_bt_motion"
DATA_CONFIG="$BT_DIR/phoenix_soke_epoch84_train_next10_slt_upper_smplx_rot6d.yaml"
OUT_JSON="$BT_DIR/slt_backtranslation_epoch119_gt_reorder_oracle.json"
GT_TEXT_REORDER=1
```

Then run:

```bash
SLT_CKPT="$SPAMO_ROOT/logs/slt/phoenix_soke_upper_smplx_rot6d_jubail/phoenix_soke_upper_smplx_rot6d_oracle_20260622-190546/checkpoints/epoch=119-12.6935.ckpt"
SPAMO_DEVICE=cpu

(
  cd "$SPAMO_ROOT"
  PYTHONNOUSERSITE=1 \
  SLT_CKPT="$SLT_CKPT" \
  DATA_CONFIG="$DATA_CONFIG" \
  OUT_JSON="$OUT_JSON" \
  GT_TEXT_REORDER="$GT_TEXT_REORDER" \
  SPAMO_DEVICE="$SPAMO_DEVICE" \
  SPAMO_HF_CACHE="$SPAMO_HF_CACHE" \
  conda run -n spamo python - <<'PY'
import json
import os
import sys
import time
from pathlib import Path

import torch
from omegaconf import OmegaConf
from utils.helpers import instantiate_from_config
from slt.mt5_slt_test import (
    parse_args_with_auto_config,
    build_model,
    load_checkpoint,
    run_inference,
    compute_metrics,
    compute_latency_stats,
)

ckpt = os.environ["SLT_CKPT"]
data_config = os.environ["DATA_CONFIG"]
out_json = Path(os.environ["OUT_JSON"])
cache_dir = os.environ.get("SPAMO_HF_CACHE", "/media/cvpr/haomian/SpaMo/.cache/huggingface")
device_name = os.environ.get("SPAMO_DEVICE", "cpu")
gt_text_reorder = os.environ.get("GT_TEXT_REORDER", "0") == "1"

sys.argv = [
    "flow_or_soke_backtranslation",
    "--checkpoint", ckpt,
    "--data_config", data_config,
    "--split", "test",
    "--batch_size", "2",
    "--num_workers", "0",
    "--device", device_name,
    "--cache_dir", cache_dir,
    "--num_print_samples", "0",
]
if not gt_text_reorder:
    sys.argv.append("--disable_gt_text_eval_reordering")

_parser, args = parse_args_with_auto_config()
device = torch.device(args.device)

model = build_model(args)
ot_eps = args.local_align_eps_end if args.local_align_enabled else None
load_checkpoint(model, args.checkpoint, ot_eps=ot_eps)
model.to(device)
model.eval()

cfg = OmegaConf.load(data_config)
data_module = instantiate_from_config(cfg)
data_module.setup(None)
loader = data_module.test_dataloader()

started = time.time()
predictions, references, latency_records = run_inference(
    model,
    loader,
    device,
    noise_std=0.0,
    max_batches=None,
)
elapsed = time.time() - started
metrics = compute_metrics(predictions, references, tokenizer=args.sacrebleu_tokenizer)
latency_stats = compute_latency_stats(latency_records)

payload = {
    "checkpoint": ckpt,
    "data_config": data_config,
    "disable_gt_text_eval_reordering": bool(args.disable_gt_text_eval_reordering),
    "oracle_gt_text_eval_reordering": not bool(args.disable_gt_text_eval_reordering),
    "device": str(device),
    "num_samples": len(predictions),
    "elapsed_sec": round(elapsed, 2),
    "metrics": metrics,
    "latency_stats": latency_stats,
    "latency_records": latency_records,
    "predictions": predictions,
    "references": references,
}
out_json.parent.mkdir(parents=True, exist_ok=True)
out_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

print(json.dumps({
    "num_samples": payload["num_samples"],
    "elapsed_sec": payload["elapsed_sec"],
    "oracle_gt_text_eval_reordering": payload["oracle_gt_text_eval_reordering"],
    "metrics": payload["metrics"],
}, ensure_ascii=False, indent=2))
print(f"Saved: {out_json}")
PY
)
```

The output JSON contains:

| Key | Meaning |
| --- | --- |
| `predictions` | Back-translated German sentences from generated poses |
| `references` | Reference text saved in the source NPZ or wrapper manifest |
| `metrics` | BLEU-1/2/3/4, ROUGE-L F1, WER |
| `latency_records`, `latency_stats` | SLT inference latency |
| `oracle_gt_text_eval_reordering` | `true` for oracle diagnostics, `false` for fair mode |

## Verified Runs

### Flow Matching, Fair Mode

Input:

```text
/media/cvpr/haomian/NeuralImplicitSignMotionTrajectory/visualize/phoenix_latent_adapter_residual_all_balanced_retry3_best_test100_seed123_noneg_t2m_eval
```

Output:

```text
/media/cvpr/haomian/NeuralImplicitSignMotionTrajectory/visualize/phoenix_flow_slt_backtranslation/slt_backtranslation_epoch119_no_gt_reorder.json
```

| Metric | Value |
| --- | ---: |
| BLEU-1 | 14.90 |
| BLEU-2 | 5.65 |
| BLEU-3 | 2.24 |
| BLEU-4 | 1.08 |
| ROUGE-L F1 | 11.93 |
| WER | 210.14 |

### Flow Matching, Oracle Diagnostic

Output:

```text
/media/cvpr/haomian/NeuralImplicitSignMotionTrajectory/visualize/phoenix_flow_slt_backtranslation/slt_backtranslation_epoch119_gt_reorder_oracle.json
```

| Metric | Value |
| --- | ---: |
| BLEU-1 | 27.20 |
| BLEU-2 | 13.81 |
| BLEU-3 | 8.05 |
| BLEU-4 | 5.38 |
| ROUGE-L F1 | 17.07 |
| WER | 98.65 |

### Baseline SOKE, Oracle Diagnostic

Input:

```text
/media/cvpr/haomian/NeuralImplicitSignMotionTrajectory/visualize/soke_min_phoenix_epoch84_train_next10_bt_motion
```

Output:

```text
/media/cvpr/haomian/NeuralImplicitSignMotionTrajectory/visualize/soke_min_phoenix_epoch84_train_next10_bt_motion/slt_backtranslation_epoch119_gt_reorder_oracle.json
```

| Metric | Value |
| --- | ---: |
| BLEU-1 | 18.86 |
| BLEU-2 | 3.38 |
| BLEU-3 | 1.54 |
| BLEU-4 | 0.89 |
| ROUGE-L F1 | 10.99 |
| WER | 106.37 |

## Caveats

- This protocol is validated for Phoenix generated samples and the Phoenix SOKE
  upper-SMPLX rot6d SLT checkpoint.
- Flow matching outputs can be used directly if they were produced by
  `flow.sample_text_conditional` without `--skip_save_outputs`.
- Baseline SOKE vertex NPZs cannot be passed to SpaMo. Always export compact
  `motion` first.
- The generated sample text is used as the BLEU/ROUGE/WER reference. For
  arbitrary prompts, make sure `text` in each NPZ row is the intended reference.
- The SpaMo run config references a missing dataset config path
  `configs/dataset/phoenix_soke_upper_smplx_rot6d_jubail_data.yaml`; the wrapper
  YAMLs above replace it for generated samples.
- If `slt.mt5_slt_test.build_dataloader()` is patched to call
  `data_module.setup(None)`, the same evaluation can be run through the standard
  CLI instead of the direct Python runner.
