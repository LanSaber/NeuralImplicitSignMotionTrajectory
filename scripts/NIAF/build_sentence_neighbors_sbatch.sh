#!/bin/bash
#SBATCH --job-name=csl_rag_neighbors
#SBATCH --output=/media/cvpr/haomian/NeuralImplicitSignMotionTrajectory/logs/sbatch/%x_%j.out
#SBATCH --error=/media/cvpr/haomian/NeuralImplicitSignMotionTrajectory/logs/sbatch/%x_%j.err
#SBATCH --partition=spark
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=04:00:00

set -euo pipefail
trap 'echo "ERROR: build_sentence_neighbors_sbatch.sh failed at line $LINENO with exit code $?" >&2' ERR

PROJECT_DIR="${PROJECT_DIR:-/media/cvpr/haomian/NeuralImplicitSignMotionTrajectory}"
PYTHON_ENV="${PYTHON_ENV:-/media/cvpr/haomian/python_envs/SOKE}"
PYTHON_BIN="${PYTHON_BIN:-$PYTHON_ENV/bin/python}"
CFG="${CFG:-NIAF/continuous_trajectory_field/configs/csl_daily_signtrajfield_v3_sentence_memory_phase_a.yaml}"
BANK_DIR="${BANK_DIR:-/media/cvpr/haomian/data/SOKE_FLOW/csl_daily_upper_smplx/meta/niaf_sentence_memory/mt5_vae_mu_train_v1}"
TOP_M="${TOP_M:-64}"
BATCH_SIZE="${BATCH_SIZE:-128}"
DEVICE="${DEVICE:-cuda}"
TEXT_DEVICE="${TEXT_DEVICE:-cuda}"
OVERWRITE="${OVERWRITE:-0}"

if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "ERROR: Python executable not found: $PYTHON_BIN" >&2
  exit 1
fi
if [[ ! -f "$PROJECT_DIR/$CFG" && ! -f "$CFG" ]]; then
  echo "ERROR: configuration does not exist: $CFG" >&2
  exit 1
fi
if [[ ! -d "$BANK_DIR" || ! -f "$BANK_DIR/READY" ]]; then
  echo "ERROR: BANK_DIR is not a ready sentence-memory bank: $BANK_DIR" >&2
  exit 1
fi
case "$OVERWRITE" in
  0|1) ;;
  *) echo "ERROR: OVERWRITE must be 0 or 1" >&2; exit 1 ;;
esac
if ! [[ "$TOP_M" =~ ^[1-9][0-9]*$ ]]; then
  echo "ERROR: TOP_M must be a positive integer; got $TOP_M" >&2
  exit 1
fi
if ! [[ "$BATCH_SIZE" =~ ^[1-9][0-9]*$ ]]; then
  echo "ERROR: BATCH_SIZE must be a positive integer; got $BATCH_SIZE" >&2
  exit 1
fi

if [[ "$OVERWRITE" != "1" ]]; then
  for split in train val test; do
    neighbor_path="$BANK_DIR/neighbors_${split}.npz"
    if [[ -e "$neighbor_path" ]]; then
      echo "ERROR: refusing existing neighbor table without OVERWRITE=1: $neighbor_path" >&2
      exit 1
    fi
  done
fi

export PATH="$PYTHON_ENV/bin:$PATH"
export PYTHONNOUSERSITE=1
export PYTHONUNBUFFERED=1
export PYTHONPATH="$PROJECT_DIR:${PYTHONPATH:-}"
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export HF_HOME="${HF_HOME:-/media/cvpr/haomian/.cache/huggingface}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-$HF_HOME/hub}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HF_HOME/transformers}"

mkdir -p "$PROJECT_DIR/logs/sbatch"
cd "$PROJECT_DIR"

BUILD_CMD=(
  "$PYTHON_BIN" -m NIAF.continuous_trajectory_field.scripts.build_sentence_neighbors
  --config "$CFG"
  --bank_dir "$BANK_DIR"
  --splits train val test
  --top_m "$TOP_M"
  --batch_size "$BATCH_SIZE"
  --device "$DEVICE"
  --text_device "$TEXT_DEVICE"
)
if [[ "$OVERWRITE" == "1" ]]; then BUILD_CMD+=(--overwrite); fi

AUDIT_CMD=(
  "$PYTHON_BIN" -m NIAF.continuous_trajectory_field.scripts.audit_sentence_memory
  --config "$CFG"
  --bank_dir "$BANK_DIR"
  --verify_hashes
  --splits train val test
)

echo "Job ID: ${SLURM_JOB_ID:-local}"
echo "Config: $CFG"
echo "Bank: $BANK_DIR"
echo "Top-M/batch: $TOP_M/$BATCH_SIZE"
echo "Device/text device: $DEVICE/$TEXT_DEVICE"
echo "W&B: disabled (offline preprocessing job)"
printf 'Build command:'
printf ' %q' "${BUILD_CMD[@]}"
printf '\n'
"${BUILD_CMD[@]}"

printf 'Audit command:'
printf ' %q' "${AUDIT_CMD[@]}"
printf '\n'
"${AUDIT_CMD[@]}"
