#!/bin/bash
#SBATCH --job-name=csl_v3_motion_contrast
#SBATCH --output=/media/cvpr/haomian/NeuralImplicitSignMotionTrajectory/logs/sbatch/%x_%j.out
#SBATCH --error=/media/cvpr/haomian/NeuralImplicitSignMotionTrajectory/logs/sbatch/%x_%j.err
#SBATCH --partition=spark
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --mem=100G
#SBATCH --time=72:00:00

set -eo pipefail
trap 'echo "ERROR: Phase-A motion-contrast launch failed at line $LINENO with exit code $?" >&2' ERR

PROJECT_DIR="${PROJECT_DIR:-/media/cvpr/haomian/NeuralImplicitSignMotionTrajectory}"
PYTHON_ENV="${PYTHON_ENV:-/media/cvpr/haomian/python_envs/SOKE}"
PYTHON_BIN="${PYTHON_BIN:-$PYTHON_ENV/bin/python}"
CFG="$PROJECT_DIR/NIAF/continuous_trajectory_field/configs/csl_daily_signtrajfield_v3_sentence_memory_phase_a_motion_contrast_v1.yaml"
SENTENCE_MEMORY_DIR="${SENTENCE_MEMORY_DIR:-/media/cvpr/haomian/data/SOKE_FLOW/csl_daily_upper_smplx/meta/niaf_sentence_memory/mt5_vae_mu_train_v1}"
OUT_DIR="${OUT_DIR:-$PROJECT_DIR/experiments/NIAF/continuous_trajectory_field/csl_daily_signtrajfield_v3_sentence_memory_phase_a_motion_contrast_v1}"
PARTITION_DIR="${PARTITION_DIR:-${OUT_DIR}.prerequisites/validation_text_partition}"
V2_BASE_CHECKPOINT="$PROJECT_DIR/experiments/NIAF/continuous_trajectory_field/csl_daily_signtrajfield_v2_mt5_text_only_full/checkpoints/best.pt"
EXPECTED_V2_BASE_SHA256="06ca0a2613005b6e3949bab0e5d7ded999b212723debd3e7685a58c077e44c54"

for forbidden in RESUME WARM_START BASE_CHECKPOINT; do
  if [[ -n "${!forbidden:-}" ]]; then
    echo "ERROR: fresh Phase A forbids inherited $forbidden" >&2
    exit 1
  fi
done

if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "ERROR: Python executable not found: $PYTHON_BIN" >&2
  exit 1
fi
if [[ ! -f "$CFG" ]]; then
  echo "ERROR: Phase-A motion-contrast config not found: $CFG" >&2
  exit 1
fi
if [[ ! -f "$V2_BASE_CHECKPOINT" ]]; then
  echo "ERROR: configured v2 base checkpoint is missing: $V2_BASE_CHECKPOINT" >&2
  exit 1
fi
CONFIGURED_V2_BASE="$("$PYTHON_BIN" - "$CFG" "$PROJECT_DIR" <<'PY'
import sys
from pathlib import Path
from NIAF.continuous_sign_field.config import load_config

cfg = load_config(Path(sys.argv[1]))
value = cfg.get("train", {}).get("base_checkpoint")
if not value:
    raise SystemExit("ERROR: config has no train.base_checkpoint")
path = Path(value)
print((path if path.is_absolute() else Path(sys.argv[2]) / path).resolve())
PY
)"
if [[ "$CONFIGURED_V2_BASE" != "$(realpath -- "$V2_BASE_CHECKPOINT")" ]]; then
  echo "ERROR: config train.base_checkpoint differs from the pinned v2 base" >&2
  exit 1
fi
ACTUAL_V2_BASE_SHA256="$(sha256sum -- "$V2_BASE_CHECKPOINT" | awk '{print $1}')"
if [[ "$ACTUAL_V2_BASE_SHA256" != "$EXPECTED_V2_BASE_SHA256" ]]; then
  echo "ERROR: v2 base checkpoint SHA256 mismatch" >&2
  echo "Expected: $EXPECTED_V2_BASE_SHA256" >&2
  echo "Actual:   $ACTUAL_V2_BASE_SHA256" >&2
  exit 1
fi
if [[ ! -f "$SENTENCE_MEMORY_DIR/READY" ]]; then
  echo "ERROR: sentence-memory bank is not READY: $SENTENCE_MEMORY_DIR" >&2
  exit 1
fi

export PATH="$PYTHON_ENV/bin:$PATH"
export PYTHONNOUSERSITE=1
export PYTHONUNBUFFERED=1
export PYTHONPATH="$PROJECT_DIR:${PYTHONPATH:-}"
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-4}"

# Materialize and verify the immutable development/confirmation assignment
# before training. Training consumes only development indices from the same
# canonical validation manifest, preserving audited neighbors_val identity.
cd "$PROJECT_DIR"
echo "Fresh v2 base: $V2_BASE_CHECKPOINT"
echo "Fresh v2 base SHA256: $ACTUAL_V2_BASE_SHA256"
"$PYTHON_BIN" -m \
  NIAF.continuous_trajectory_field.scripts.build_validation_text_partition \
  --config "$CFG" \
  --bank_dir "$SENTENCE_MEMORY_DIR" \
  --out_dir "$PARTITION_DIR" \
  --development_text_count 256 \
  --expected_novel_text_count 796

# This experiment has an explicit no-test and no-W&B contract. Copy/audit only
# the train and validation tables so provider/checkpoint identity cannot open or
# hash neighbors_test.npz.
export CFG SENTENCE_MEMORY_DIR OUT_DIR
export RUN_TAG="csl_daily_signtrajfield_v3_sentence_memory_phase_a_motion_contrast_v1"
export EPOCHS=4
export WANDB=0
export WANDB_MODE=disabled
export STAGE_SENTENCE_MEMORY=1
export STAGE_ONLY_REQUESTED_NEIGHBORS=1
export SPLITS="train val"
export DEVICE=cuda
export TEXT_DEVICE=cpu
export DISTRIBUTED=ddp
export DDP_BACKEND=nccl

exec bash "$PROJECT_DIR/scripts/NIAF/train_continuous_trajectory_field_sbatch.sh"
