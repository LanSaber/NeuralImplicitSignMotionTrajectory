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
WORD_PRIOR="${WORD_PRIOR:-auto}"
SENTENCE_MEMORY="${SENTENCE_MEMORY:-auto}"
DEVICE="${DEVICE:-auto}"
TEXT_DEVICE="${TEXT_DEVICE:-cpu}"
DDP_BACKEND="${DDP_BACKEND:-nccl}"
DDP_TIMEOUT_MIN="${DDP_TIMEOUT_MIN:-120}"
SENTENCE_MEMORY_DIR="${SENTENCE_MEMORY_DIR:-}"
STAGE_SENTENCE_MEMORY="${STAGE_SENTENCE_MEMORY:-0}"
LOCAL_SENTENCE_MEMORY_DIR=""

export PATH="$PYTHON_ENV/bin:$PATH"
export PYTHONNOUSERSITE=1
export PYTHONUNBUFFERED=1
export PYTHONPATH="$PROJECT_DIR:${PYTHONPATH:-}"
export TOKENIZERS_PARALLELISM=false
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

if [[ -d "$CHECKPOINT" ]]; then
  CHECKPOINT_DIR="$CHECKPOINT"
  CHECKPOINT=""
  # An infeasible checkpoint is diagnostic evidence, never a promoted model.
  for CANDIDATE in best.pt last.pt; do
    if [[ -f "$CHECKPOINT_DIR/$CANDIDATE" ]]; then
      CHECKPOINT="$CHECKPOINT_DIR/$CANDIDATE"
      break
    fi
  done
  if [[ -z "$CHECKPOINT" ]]; then
    echo "ERROR: no selected or final checkpoint found under $CHECKPOINT_DIR" >&2
    exit 1
  fi
elif [[ ! -f "$CHECKPOINT" ]]; then
  echo "ERROR: checkpoint does not exist: $CHECKPOINT" >&2
  exit 1
fi

if [[ -n "${SLURM_JOB_ID:-}" ]]; then
  MASTER_ADDR="${MASTER_ADDR:-$(scontrol show hostnames "$SLURM_NODELIST" | head -n 1)}"
  MASTER_PORT="${MASTER_PORT:-$((20000 + SLURM_JOB_ID % 20000))}"
  export MASTER_ADDR MASTER_PORT
  if [[ -n "${SLURM_NTASKS:-}" ]]; then export WORLD_SIZE="$SLURM_NTASKS"; fi
fi

cleanup_sentence_memory() {
  local exit_code=$?
  trap - EXIT
  if [[ -n "$LOCAL_SENTENCE_MEMORY_DIR" ]]; then
    srun --nodes="$SLURM_NNODES" --ntasks="$SLURM_NNODES" --ntasks-per-node=1 \
      bash "$PROJECT_DIR/scripts/NIAF/stage_sentence_memory_node.sh" \
      cleanup "$LOCAL_SENTENCE_MEMORY_DIR" || true
  fi
  exit "$exit_code"
}
trap cleanup_sentence_memory EXIT

case "$STAGE_SENTENCE_MEMORY" in
  0|1) ;;
  *)
    echo "ERROR: STAGE_SENTENCE_MEMORY must be 0 or 1" >&2
    exit 1
    ;;
esac
if [[ "$STAGE_SENTENCE_MEMORY" == "1" ]]; then
  if [[ -z "${SLURM_JOB_ID:-}" || -z "$SENTENCE_MEMORY_DIR" ]]; then
    echo "ERROR: staging requires Slurm and SENTENCE_MEMORY_DIR" >&2
    exit 1
  fi
  LOCAL_SENTENCE_MEMORY_DIR="/tmp/signtraj_sentence_memory_${SLURM_JOB_ID}"
  srun --nodes="$SLURM_NNODES" --ntasks="$SLURM_NNODES" --ntasks-per-node=1 \
    bash "$PROJECT_DIR/scripts/NIAF/stage_sentence_memory_node.sh" \
    stage "$LOCAL_SENTENCE_MEMORY_DIR" "$SENTENCE_MEMORY_DIR" \
    "$PROJECT_DIR" "$PYTHON_BIN" "$CFG" "train $SPLIT"
  export SIGNTRAJ_SENTENCE_MEMORY_DIR="$LOCAL_SENTENCE_MEMORY_DIR"
elif [[ -n "$SENTENCE_MEMORY_DIR" ]]; then
  if [[ ! -d "$SENTENCE_MEMORY_DIR" || ! -f "$SENTENCE_MEMORY_DIR/READY" ]]; then
    echo "ERROR: SENTENCE_MEMORY_DIR is not a ready bank: $SENTENCE_MEMORY_DIR" >&2
    exit 1
  fi
  export SIGNTRAJ_SENTENCE_MEMORY_DIR="$SENTENCE_MEMORY_DIR"
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
  --word_prior "$WORD_PRIOR"
  --sentence_memory "$SENTENCE_MEMORY"
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
echo "Word prior: $WORD_PRIOR"
echo "Sentence memory: $SENTENCE_MEMORY"
echo "Sentence-memory bank: ${SIGNTRAJ_SENTENCE_MEMORY_DIR:-config value} staged=$STAGE_SENTENCE_MEMORY"
echo "Output: $OUT_JSON"
printf 'Command:'
printf ' %q' "${CMD[@]}"
printf '\n'
"${CMD[@]}"
