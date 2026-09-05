#!/bin/bash
#SBATCH --job-name=csl_rag_bank
#SBATCH --output=/media/cvpr/haomian/NeuralImplicitSignMotionTrajectory/logs/sbatch/%x_%j.out
#SBATCH --error=/media/cvpr/haomian/NeuralImplicitSignMotionTrajectory/logs/sbatch/%x_%j.err
#SBATCH --partition=spark
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=06:00:00

set -euo pipefail
trap 'echo "ERROR: build_sentence_motion_bank_sbatch.sh failed at line $LINENO with exit code $?" >&2' ERR

PROJECT_DIR="${PROJECT_DIR:-/media/cvpr/haomian/NeuralImplicitSignMotionTrajectory}"
PYTHON_ENV="${PYTHON_ENV:-/media/cvpr/haomian/python_envs/SOKE}"
PYTHON_BIN="${PYTHON_BIN:-$PYTHON_ENV/bin/python}"
CFG="${CFG:-NIAF/continuous_trajectory_field/configs/csl_daily_signtrajfield_v3_sentence_memory_phase_a.yaml}"
BANK_DIR="${BANK_DIR:-/media/cvpr/haomian/data/SOKE_FLOW/csl_daily_upper_smplx/meta/niaf_sentence_memory/mt5_vae_mu_train_v1}"
BATCH_SIZE="${BATCH_SIZE:-64}"
NUM_WORKERS="${NUM_WORKERS:-8}"
DEVICE="${DEVICE:-cuda}"
TEXT_DEVICE="${TEXT_DEVICE:-cuda}"
OVERWRITE="${OVERWRITE:-0}"

if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "ERROR: Python executable not found: $PYTHON_BIN" >&2
  exit 1
fi
case "$OVERWRITE" in
  0|1) ;;
  *) echo "ERROR: OVERWRITE must be 0 or 1" >&2; exit 1 ;;
esac
if [[ -e "$BANK_DIR" && "$OVERWRITE" != "1" ]]; then
  echo "ERROR: refusing existing BANK_DIR without OVERWRITE=1: $BANK_DIR" >&2
  exit 1
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
  "$PYTHON_BIN" -m NIAF.continuous_trajectory_field.scripts.build_sentence_motion_bank
  --config "$CFG"
  --split train
  --out_dir "$BANK_DIR"
  --batch_size "$BATCH_SIZE"
  --num_workers "$NUM_WORKERS"
  --device "$DEVICE"
  --text_device "$TEXT_DEVICE"
)
if [[ "$OVERWRITE" == "1" ]]; then BUILD_CMD+=(--overwrite); fi

echo "Job ID: ${SLURM_JOB_ID:-local}"
echo "Config: $CFG"
echo "Bank: $BANK_DIR"
echo "Batch/workers: $BATCH_SIZE/$NUM_WORKERS"
echo "Device/text device: $DEVICE/$TEXT_DEVICE"
echo "W&B: disabled (offline preprocessing job)"
printf 'Command:'
printf ' %q' "${BUILD_CMD[@]}"
printf '\n'

exec "${BUILD_CMD[@]}"
