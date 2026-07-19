#!/bin/bash
#SBATCH --job-name=niaf_ct_eval
#SBATCH --output=/media/cvpr/haomian/NeuralImplicitSignMotionTrajectory/logs/sbatch/%x_%j.out
#SBATCH --error=/media/cvpr/haomian/NeuralImplicitSignMotionTrajectory/logs/sbatch/%x_%j.err
#SBATCH --partition=spark
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --mem=100G
#SBATCH --time=04:00:00

set -eo pipefail
trap 'echo "ERROR: evaluate_continuous_trajectory_field_sbatch.sh failed at line $LINENO with exit code $?" >&2' ERR

PROJECT_DIR="${PROJECT_DIR:-/media/cvpr/haomian/NeuralImplicitSignMotionTrajectory}"
PYTHON_ENV="${PYTHON_ENV:-/media/cvpr/haomian/python_envs/SOKE}"
PYTHON_BIN="${PYTHON_BIN:-$PYTHON_ENV/bin/python}"
CFG="${CFG:?Set CFG to the continuous trajectory configuration}"
CHECKPOINT="${CHECKPOINT:?Set CHECKPOINT to the checkpoint snapshot}"
OUT_JSON="${OUT_JSON:?Set OUT_JSON to the result path}"
SPLIT="${SPLIT:-test}"
BATCH_SIZE="${BATCH_SIZE:-32}"
LIMIT="${LIMIT:-0}"
MAX_BATCHES="${MAX_BATCHES:-0}"
SCAFFOLD_MODE="${SCAFFOLD_MODE:-config}"
DEVICE="${DEVICE:-auto}"
TEXT_DEVICE="${TEXT_DEVICE:-cpu}"
DDP_BACKEND="${DDP_BACKEND:-nccl}"
DDP_TIMEOUT_MIN="${DDP_TIMEOUT_MIN:-120}"

export PATH="$PYTHON_ENV/bin:$PATH"
export PYTHONNOUSERSITE=1
export PYTHONUNBUFFERED=1
export PYTHONPATH="$PROJECT_DIR:${PYTHONPATH:-}"
export TOKENIZERS_PARALLELISM=false
export HOME="${HOME_VALUE:-/media/cvpr/haomian}"
export HF_HOME="${HF_HOME:-/media/cvpr/haomian/.cache/huggingface}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-$HF_HOME/hub}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HF_HOME/transformers}"
unset NCCL_ASYNC_ERROR_HANDLING
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"
export NCCL_SOCKET_IFNAME="${NCCL_SOCKET_IFNAME:-enP7s7}"
export GLOO_SOCKET_IFNAME="${GLOO_SOCKET_IFNAME:-enP7s7}"
export NCCL_IB_DISABLE="${NCCL_IB_DISABLE:-1}"

if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "ERROR: Python executable not found: $PYTHON_BIN" >&2
  exit 1
fi

if [[ -n "${SLURM_JOB_ID:-}" ]]; then
  MASTER_ADDR="${MASTER_ADDR:-$(scontrol show hostnames "$SLURM_NODELIST" | head -n 1)}"
  MASTER_PORT="${MASTER_PORT:-$((20000 + SLURM_JOB_ID % 20000))}"
  export MASTER_ADDR MASTER_PORT
  if [[ -n "${SLURM_NTASKS:-}" ]]; then export WORLD_SIZE="$SLURM_NTASKS"; fi
fi

mkdir -p "$PROJECT_DIR/logs/sbatch" "$(dirname "$OUT_JSON")"
cd "$PROJECT_DIR"

CMD=(
  srun --kill-on-bad-exit=1
  "$PYTHON_BIN" -m NIAF.continuous_trajectory_field.scripts.evaluate_continuous_trajectory_field
  --config "$CFG"
  --checkpoint "$CHECKPOINT"
  --split "$SPLIT"
  --out_json "$OUT_JSON"
  --batch_size "$BATCH_SIZE"
  --limit "$LIMIT"
  --max_batches "$MAX_BATCHES"
  --scaffold_mode "$SCAFFOLD_MODE"
  --device "$DEVICE"
  --text_device "$TEXT_DEVICE"
  --distributed ddp
  --ddp_backend "$DDP_BACKEND"
  --ddp_timeout_min "$DDP_TIMEOUT_MIN"
)

echo "Job ID: ${SLURM_JOB_ID:-local}"
echo "Nodes: ${SLURM_NODELIST:-local} world_size=${WORLD_SIZE:-1}"
echo "Checkpoint: $CHECKPOINT"
echo "Split: $SPLIT batch_per_rank=$BATCH_SIZE limit=$LIMIT max_batches=$MAX_BATCHES"
echo "Scaffold mode: $SCAFFOLD_MODE"
echo "Output: $OUT_JSON"
printf 'Command:'
printf ' %q' "${CMD[@]}"
printf '\n'
"${CMD[@]}"
